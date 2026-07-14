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

This is not a general model zoo. Inferno keeps the inference path small,
explicit, and optimized for Q2 GLM-5.2-style execution so it can become fast
enough for real local use while staying simple enough to audit end to end. That
is the ambition — whether it gets all the way there is exactly what the
experiment is testing.

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

The 256 routed experts per sparse layer remain in the GGUF artifact on SSD.
Inferno executes the artifact's exact top-8 routing and caches selected Q2 gate,
up, and down matrices in shared Metal slabs. The default is 16 expert slots per
routed layer, approximately 14.9 GB for the 75 main sparse layers.

Each layer uses a segmented LRU:

```txt
probation: newly loaded or weakly reused experts
protected: experts promoted after reuse
```

Cache hits execute directly from the resident Metal slot. On a miss, parallel
`pread` workers load the gate and up matrices first. Metal starts their fused
SwiGLU projection while the same workers continue loading the down matrices.
All ready gate/up waves are queued before the down waves, which overlaps SSD
reads with useful GPU work without changing the result. The cache never evicts
an expert selected by the current token; if every slot is temporarily
protected, that expert uses a one-shot transient buffer.

### Budget control

The CLI currently uses fixed, independent budgets:

```txt
expert cache: 16 slots per routed layer
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

## Measured Results

Performance is not yet production-ready. Results are reported with their exact
revision and machine state because repeated expert streaming is sensitive to
filesystem-cache, thermal, and memory conditions.

Test conditions:

```txt
hardware:         Apple Silicon MacBook Pro, 64 GB unified memory
build:            cargo build --release
model:            GLM-5.2-UD-Q2_K_RoutedQ2K.gguf
prompt:           "Hi" (13 tokens after chat-template rendering)
generated tokens: 8
MoE execution:    top-8, matching the artifact
cache settings:   defaults; speculative MTP disabled
```

### Reference top-8 benchmark

The retained reference was measured on 2026-07-13 at revision `cbf1a78` plus
the staged expert-I/O path that became `1fafb61`:

| Runtime | Run | Time to first token | Decode throughput | End-to-end throughput |
| --- | --- | ---: | ---: | ---: |
| Previous ready-first path | Warm baseline | 28.351 s | 1.447 tokens/s | 0.241 tokens/s |
| Staged gate/up/down path | Measurement 1 | 30.106 s | **1.505 tokens/s** | 0.230 tokens/s |
| Staged gate/up/down path | Measurement 2 | 31.850 s | **1.493 tokens/s** | 0.219 tokens/s |

The two staged measurements average **1.499 decode tokens/s**. All runs used
exact top-8 routing. They reported a 46.71% expert-cache hit rate, 79.210 GB of
logical expert reads, 14.864 GB of expert-cache capacity, and a 100% KV hit
rate for this short prompt.

### Degraded-state diagnostic

On 2026-07-14, revision `d0183f0` produced 0.573, 0.430, and 0.357 decode
tokens/s, with a median of **0.430 tokens/s**. These runs reported a 41.56%
expert-cache hit rate and 86.865 GB of logical expert reads.

This lower result is not attributed solely to the current revision. The exact
historical `cbf1a78` binary was rebuilt and rerun under the same degraded
machine state: it achieved 0.502 tokens/s while reproducing its historical
46.71% hit rate and 79.210 GB read count. The same revision previously measured
1.447 tokens/s. This controlled comparison shows a substantial environmental
effect that must be isolated before establishing a new reference baseline.

`Decode throughput` excludes prefill and the first generated token. It is the
best measure of steady token generation. `End-to-end throughput` divides all
generated tokens by total command time and therefore includes model startup,
prefill, and time to first token.

Reproduce the measurement with:

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Hi" \
  --max-new-tokens 8 \
  --measure-tokens-per-second \
  --throughput-file /tmp/inferno-throughput.tsv
```

Run the command at least three times to expose variation caused by macOS
filesystem-cache state and system load. Compare medians rather than selecting
the fastest run. The TSV output preserves the full timing, expert-cache,
KV-cache, SSD-read, and routed-expert-count metrics for later comparisons.

### Quality smoke checks

Inferno loads the model's complete EOS list from `generation_config.json` and
uses GLM's adjacent-pair RoPE layout. The exact top-8 path still requires a
repeatable reference-logit comparison and a larger evaluation suite before
quality can be considered validated.

### Current throughput limit

The GGUF directory contains approximately 20.49 GB of always-active Q8 weights,
0.55 GB of F32 tensors, and 240.99 GB of routed Q2 expert weights. The reference
top-8 benchmark read 79.210 GB of logical expert data; the degraded diagnostic
read 86.865 GB.

A three-token-per-second target allows only 0.333 seconds per token. At the
1.499 tokens/s reference, one token takes approximately 0.667 seconds. The
short-prompt KV hit rate is already 100%, so the immediate limit remains the
routed-expert and sparse-attention paths rather than KV capacity. Further work
must reduce or amortize expert misses and projection time while preserving the
artifact's exact top-8 routing behavior.

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

Credit to Antirez for publishing the GLM-5.2 GGUF conversion used as the target
artifact for this implementation.

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
```

Run generation:

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy."
```

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
