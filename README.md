# Inferno

<p align="center">
  <img src="assets/logo.svg" alt="Inferno logo" width="160">
</p>

> [!WARNING]
> **Inferno is an experiment, not a product.** It began as a personal
> engineering challenge: can a very large sparse, long-context GLM-5.2-style
> model be made to run locally on a 64 GB Apple Silicon machine at all? It is a
> work in progress, developed in the open, and **results are not guaranteed** —
> correctness, stability, and performance are all still being worked out. Treat
> everything here as exploratory. It is not intended for production use, and
> there is no promise that any given build produces correct output or runs at a
> usable speed.

Inferno aims to be a super lightweight, highly efficient Rust inference engine
for running GLM-5.2 Q2 on Apple Silicon with Metal, targeting machines such as a
MacBook Pro with 64 GB of unified memory.

The name is intentional. Inferno means "hell" in English, and it fits the
project: running a very large sparse, long-context GLM-5.2-style model locally
on a 64 GB Apple Silicon machine is a hard systems problem. The constraints are
tight, the memory budget is unforgiving, and the runtime has to stay narrow to
be viable. The goal is not comfort or generality. The goal is to build a small,
direct, auditable runtime that does only what this target needs and does it well.

## Status

Inferno started as an engineering challenge and remains an experiment. It is a
work in progress, and **results are not guaranteed yet** — the project exists to
explore whether this hard target is reachable, not to ship a finished tool.
Expect rough edges, breaking changes, incomplete paths, and the occasional dead
end. Parts of the design described below are targets and directions, not
completed guarantees.

The runtime is being shaped around one hard target first: the Antirez GLM-5.2 Q2
GGUF artifact, paged KV cache by default, sparse MoE routing, and Metal-focused
execution for the bottlenecks that matter.

The artifact declares eight routed experts per token, and Inferno executes all
eight. Startup fails if `config.json` or the GGUF metadata declares a different
routing count, preventing an accidental quality-changing approximation.

Multi-token prediction is part of the default generation path when the model
metadata exposes the GLM-5.2 MTP head and enables shared index selection. It is
not an optional approximation and does not reduce the top-8 MoE contract.

This is not a general model zoo. Inferno keeps the inference path small,
explicit, and optimized for Q2 GLM-5.2-style execution so it can become fast
enough for real local use while staying simple enough to audit end to end. That
is the ambition — whether it gets all the way there is exactly what the
experiment is testing.

## Multi-Token Prediction

Inferno uses the GLM-5.2 MTP head to propose up to two tokens before asking
the main model to verify them. MTP is enabled automatically when the loaded
artifact and configuration expose the required tensors and metadata; no CLI
flag is required.

One MTP generation step follows this sequence:

```txt
committed hidden state
  -> MTP draft step 0
  -> up to one additional shared-head draft step
  -> one causal target-model verification
  -> accept matching prefix
  -> commit accepted KV rows only
```

For `D <= 2` draft tokens, the main shapes are:

```txt
draft token ids:       [D]
verifier input ids:    [1, 1 + D]
verifier hidden state: [1, 1 + D, 6144]
verified token ids:    [1 + D]
```

The first MTP draft computes the DSA top-k token selection. **IndexShare**
reuses that selection for the remaining draft steps instead of running the
indexer repeatedly. The first draft also establishes the fixed compressed MLA
state used by the MTP sequence. **KVShare** lets later drafts query that state
without appending speculative K/V rows.

The target model verifies the current token and all drafts together as a short
causal sequence. Drafts are accepted from left to right until the first
mismatch. Rejected drafts are discarded, and only the verified prefix is
appended to the request KV cache. This preserves greedy target-model semantics
while amortizing one full backbone pass across multiple accepted tokens.

The two-draft limit is specific to SSD-streamed MoE execution. Wider verifier
batches activate many more unique experts, and measured SSD traffic grows
faster than the number of accepted tokens. Two drafts retained the accepted
prefix while reducing verifier rows and expert reads on the target machine.

## Memory Strategy

Inferno treats unified memory and SSD as two levels of one runtime memory
hierarchy. Always-used model tensors remain memory-mapped, routed experts use a
bounded Metal cache, and request KV uses an SSD-backed cold tier with a Metal
working set.

### KV cache

GLM-5.2 uses MLA, so Inferno caches the compressed 512-value latent and the
64-value RoPE component instead of expanded K/V for all 64 attention heads. At
batch size 1, the logical cache shapes for one layer are:

```txt
latent KV:  [1, 1, T, 512]
RoPE state: [1, 1, T, 64]
```

The authoritative history is a Q8 row-compressed, append-only SSD block store.
Its in-memory index addresses records by:

```txt
layer_index + tensor_kind(key/value) + token_start + token_count
```

The Metal hot tier has two modes:

```txt
short context: keep every layer's cache resident when it fits the hot budget
long context:  reuse one Metal window and stream layers from SSD
```

The default hot budget is 512 MiB. With the current 78-layer model, batch size
1, 128-token pages, and F32 hot storage, this keeps approximately 2,944 tokens
resident across all layers. When the next page no longer fits, Inferno releases
the all-layer tier and reuses one layer-sized Metal window. A background reader
prefetches the next layer while the current layer executes.

For long-context sparse attention, DSA reads only the selected Q8 rows from SSD
and decodes them into Metal buffers. The complete history is retained; changing
tiers never discards previous tokens. The temporary KV and DSA index files are
removed when generation ends.

Each accepted latent/RoPE row is read from shared Metal storage, encoded as Q8
on the CPU, and appended to the SSD store. Apple Silicon uses unified memory,
so this read is a small host-memory copy after GPU synchronization rather than
a transfer across a discrete-GPU bus. The store writes record headers and Q8
payloads with vectored I/O, avoiding a second combined-record allocation.

Two alternatives were measured on the target model and rejected: direct GPU
Q8 encoding took approximately 1.347 ms per complete 78-layer token append,
and a zero-copy CPU view took approximately 0.517 ms. The retained path took
approximately 0.417 ms. Kernel dispatch and synchronization cost more than the
small F32 copy in this MLA-compressed workload. Full cold-tier misses still
reconstruct F32 rows before upload and remain a separate optimization target.

### Routed expert cache

The source weights remain in the GGUF artifact. For inference, Inferno uses a
lossless derived file named
`GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack` that stores one fixed
record per layer and expert:

```txt
4 KiB header -> [layer][expert][gate][up][down]
```

The pack preserves the exact Q2 bytes and top-8 routing; it changes only their
physical order on SSD. Keeping each expert's three matrices adjacent reduces
seeks compared with the component-major GGUF layout. The runtime validates the
pack version, source layout fingerprint, dimensions, component strides, and
exact file size before using it.

In a controlled comparison with identical prompt, routing, output, and MTP
acceptance, the ExpertPack increased end-to-end decode throughput from 0.247 to
0.612 tokens/s, approximately 2.5x. Time to first token fell from 84.5 to 51.1
seconds, and physical reads fell from 201.9 to 182.5 GB. These results isolate
the storage layout before later cache and scheduling improvements; the format
does not alter model values or output quality.

Inferno caches selected Q2 gate, up, and down matrices in shared Metal slabs.
The default is 12 expert slots per routed layer, approximately 11.3 GB including
the main sparse layers and MTP head.

Each layer uses a segmented LRU:

```txt
probation: newly loaded or weakly reused experts
protected: experts promoted after reuse
```

Cache hits execute directly from the resident Metal slot. On a miss, up to 32
parallel `pread` workers load gate and up matrices before loading down matrices.
Metal starts fused SwiGLU work as small ready waves arrive. The expert-pack file
uses `F_NOCACHE`: Inferno's bounded cache owns useful reuse, while one-time
misses bypass the macOS page cache and avoid displacing always-hot model data.
The cache never evicts an expert selected by the current token; if every slot
is temporarily protected, overflow experts use a bounded staging slab that is
reused after the current layer completes.

### Budget control

The CLI currently uses fixed, independent budgets:

```txt
expert cache: 12 slots per routed layer
hot KV cache: 512 MiB
```

They can be overridden with `--expert-cache-gb` and `--hot-kv-cache-gb`.
Because Apple Silicon uses unified memory, both allocations consume the same
physical RAM. Their sum must leave enough space for dense weights,
intermediates, macOS, and filesystem cache; excessive values can trigger swap
and reduce throughput.

Inferno also contains a dynamic controller that can rebalance RAM between KV
and experts using context length, hit rates, and memory pressure. It is not yet
enabled by the `generate` CLI path, so normal runs do not currently resize the
two caches automatically.

## Performance

| Best decode tokens/s |
| ---: |
| 1.5 |

## Model

Inferno currently targets one model:

```txt
GLM-5.2-UD-Q2_K_RoutedQ2K.gguf
```

Source repository:

```txt
https://huggingface.co/antirez/glm-5.2-gguf
```

Artifact URL:

```txt
https://huggingface.co/antirez/glm-5.2-gguf/blob/main/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf
```

The runtime is built around this Q2 GGUF layout. Full-precision GLM-5.2 shards
are not part of the inference path.

Inferno's ExpertPack is derived losslessly from this artifact: every routed
expert payload remains byte-for-byte identical to the Q2 data published by
Antirez. Inferno changes only the on-disk ordering required for efficient expert
streaming.

Credit and thanks to Antirez for producing and publishing the GLM-5.2 Q2 GGUF
artifact that makes Inferno's local Apple Silicon work possible.

## How to Run

Build the release binary:

```bash
cargo build --release
```

Place the target model files under:

```txt
models/glm-5.2/
```

Expected files:

```txt
models/glm-5.2/config.json
models/glm-5.2/tokenizer.json
models/glm-5.2/GLM-5.2-UD-Q2_K_RoutedQ2K.gguf
models/glm-5.2/GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack
```

Create the lossless expert pack once after downloading the GGUF artifact:

```bash
cargo run --release -p inferno --example pack_experts -- \
  --model models/glm-5.2
```

The generated file is approximately 241 GB. Creation is resumable at complete
expert-record boundaries and publishes the final path atomically. Subsequent
generation validates and selects it automatically.

Run generation:

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy."
```

Start an interactive local chat session:

```bash
target/release/inferno
```

With no arguments, Inferno starts chat mode using `models/glm-5.2`. Use the
explicit `chat` subcommand only when overriding defaults, for example
`target/release/inferno chat --model /path/to/model`.

The model, Metal backend, and routed-expert cache remain alive across turns.
The input prompt is `inferno>`. During generation, terminal input is disabled
and any attempted keystrokes are discarded. GLM reasoning streams in gray;
when the model emits `</think>`, the final answer begins on a new white line
without a label. Final answers are retained in conversation history; previous
reasoning is displayed to the user but excluded from later prompts, as required
by the GLM chat template. Use `/clear` to reset the conversation and `/exit` to
close the session.

Each turn currently re-prefills the accumulated conversation into a fresh
request KV cache. This preserves multi-turn correctness while persistent
cross-turn KV reuse remains a future latency optimization.

MTP, IndexShare, and KVShare are selected automatically from the model
metadata. Generation falls back to ordinary single-token verification only
when MTP is unavailable or too few output tokens remain to benefit from it.

Run with telemetry and throughput measurement:

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy." \
  --telemetry-file /tmp/inferno-memory.log \
  --measure-tokens-per-second \
  --throughput-file /tmp/inferno-throughput.tsv
```

Inspect telemetry while generation is running:

```bash
tail -f /tmp/inferno-memory.log
```

Compare throughput after each optimization:

```bash
tail -n 5 /tmp/inferno-throughput.tsv
```

The throughput report also includes the number of MTP verification passes,
drafted tokens, accepted drafts, and the resulting acceptance rate.

Profile one prefill result and each decode token by subsystem:

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy." \
  --max-new-tokens 3 \
  --profile-token-costs
```

This diagnostic mode synchronizes Metal around measured stages. Its reported
token latency is therefore intentionally higher than normal generation. Use it
to rank attention, routing, expert SSD loading, expert Q2 math, KV I/O, output
projection, and argmax costs; use `--measure-tokens-per-second` without
`--profile-token-costs` for the production throughput baseline.

Limit generation explicitly when needed:

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy." \
  --max-new-tokens 16
```

If `--max-new-tokens` is omitted, Inferno generates until EOS or context limit.

Validate the workspace:

```bash
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
```

Measure SSD KV bandwidth and Metal hot-load latency:

```bash
cargo bench -p runtime --bench ssd_kv
```

## License

Inferno is licensed under the MIT License.
