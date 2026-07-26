# Inferno

<p align="center">
  <img src="assets/logo.svg" alt="Inferno logo" width="150">
</p>

Inferno is a lightweight Rust inference engine for running selected
Mixture-of-Experts models on Apple Silicon through native Metal kernels. It
currently supports GLM-5.2 Q2 and two exact Laguna S 2.1 artifacts: the
official INT4 Safetensors checkpoint and Antirez's mixed Q2_K/Q3_K GGUF. Its
primary target is a MacBook Pro with 64 GB of unified memory.

The name reflects the engineering challenge: these models are larger than the
available memory, so useful local inference requires careful coordination of
Metal, unified memory, caching, and SSD streaming. Inferno stays deliberately
narrow. It supports a small set of exact model layouts and optimizes each path
independently instead of becoming a general inference framework.

> [!WARNING]
> Inferno is experimental and under active development. It is not production
> ready, and correctness, stability, long-context behavior, and performance are
> still being validated.

## Current Scope

- Apple Silicon and Metal only.
- GLM-5.2 Q2 through `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf`.
- Laguna S 2.1 through either its official INT4 Safetensors checkpoint or
  `laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf`.
- Exact top-8 routing for GLM and exact top-10 routing for Laguna.
- Native Metal execution with no CPU compute fallback.
- Interactive chat, one-shot generation, and a streaming Responses API.
- Persistent expert caches across chat turns and server requests.
- Model-specific expert and KV cache policies.
- Greedy decoding.
- Experimental GLM MTP speculative decoding, disabled by default because it is
  currently slower than ordinary decode on the 64 GB target.

Inferno is not a general model loader and does not support arbitrary GGUF
or Safetensors architectures and quantization formats.

## Performance

| Model | Decode throughput |
| --- | ---: |
| GLM-5.2 Q2 | **1.711 tokens/s** |
| Laguna S 2.1 INT4 Safetensors | **4.680 tokens/s** |
| Laguna S 2.1 mixed Q2_K/Q3_K GGUF | **54.53 tokens/s** |

Best observed decode results on a 64 GB Apple Silicon MacBook Pro using release
builds and exact routing. Prompt prefill is excluded. Results depend on prompt
length, expert-cache state, SSD activity, thermal state, and memory pressure.

## Requirements

- A Mac with Apple Silicon and Metal support.
- 64 GB of unified memory is the current development target.
- Rust and Cargo.
- The Hugging Face CLI for the download commands below.
- GLM requires more than 503 GB of free SSD space for its source GGUF and
  ExpertPack, excluding temporary KV data and build outputs.
- Laguna requires approximately 72 GB for its INT4 Safetensors checkpoint or
  45 GB for the Antirez GGUF; additional free space is required for build
  outputs and normal system operation.

Install the Hugging Face CLI with Homebrew if needed:

```bash
brew install hf
```

## Models

### GLM-5.2 Q2

Inferno uses the Q2 GGUF published by Antirez:

```text
GLM-5.2-UD-Q2_K_RoutedQ2K.gguf
```

Source:
[antirez/glm-5.2-gguf](https://huggingface.co/antirez/glm-5.2-gguf)

Inferno also supports a lossless derived ExpertPack. It preserves the original
Q2 expert bytes but stores each routed expert's gate, up, and down matrices
next to each other. This layout reduces scattered SSD reads. The ExpertPack is
optional, but the documented performance assumes it is present.

ExpertPack:
[allemanfredi/inferno-glm-5.2-q2-expertpack](https://huggingface.co/allemanfredi/inferno-glm-5.2-q2-expertpack)

#### Download

Create the model directory:

```bash
mkdir -p models/glm-5.2
```

Download the source model files:

```bash
hf download antirez/glm-5.2-gguf \
  config.json \
  generation_config.json \
  tokenizer.json \
  GLM-5.2-UD-Q2_K_RoutedQ2K.gguf \
  --local-dir models/glm-5.2
```

Download the prebuilt ExpertPack:

```bash
hf download allemanfredi/inferno-glm-5.2-q2-expertpack \
  GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack \
  --local-dir models/glm-5.2
```

The final directory must contain:

```text
models/glm-5.2/
  config.json
  generation_config.json
  tokenizer.json
  GLM-5.2-UD-Q2_K_RoutedQ2K.gguf
  GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack
```

The two weight artifacts occupy approximately 503 GB in total. Inferno
validates the ExpertPack against the source GGUF before using it.

To recreate the ExpertPack locally instead of downloading it:

```bash
cargo run --release -p inferno --example pack_experts -- \
  --model models/glm-5.2
```

Creation is resumable at complete expert-record boundaries.

### Laguna S 2.1

#### Antirez GGUF

Inferno supports this exact mixed-quantization artifact:

```text
antirez/Laguna-S-2.1-GGUF
laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf
```

The first 20 routed MoE layers use Q2_K experts and the final 27 use Q3_K
experts. Always-used matrices use Q8_0. Inferno validates the complete
814-tensor directory and executes these formats directly with native Metal
kernels; it does not provide a generic GGUF compatibility path.

Download the GGUF:

```bash
mkdir -p models/laguna-s-2.1-gguf
hf download antirez/Laguna-S-2.1-GGUF \
  laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf \
  --local-dir models/laguna-s-2.1-gguf
```

Inferno uses `tokenizers`, so the model directory also needs the original
Laguna configuration and tokenizer sidecars. After accepting Poolside's model
license and running `hf auth login`, download only those two files:

```bash
hf download poolside/Laguna-S-2.1-INT4 \
  config.json \
  tokenizer.json \
  --local-dir models/laguna-s-2.1-gguf
```

The final directory is:

```text
models/laguna-s-2.1-gguf/
  config.json
  tokenizer.json
  laguna-s-2.1-RoutedQ2_K-Last27Q3_K.gguf
```

Published GGUF size: `48,260,803,968` bytes. SHA-256:
`61fc66596597985cb9408a8530de6322d9e0d5b1d2ad4ed6503938018e0ce903`.

#### Official INT4 Safetensors

Inferno supports the official
[poolside/Laguna-S-2.1-INT4](https://huggingface.co/poolside/Laguna-S-2.1-INT4)
checkpoint directly. It reads the published configuration, tokenizer,
Safetensors index, and 15 weight shards. No conversion step is required.

Accept the model license on Hugging Face and authenticate the CLI, then
download the checkpoint:

```bash
hf auth login
mkdir -p models/laguna-s-2.1-int4
hf download poolside/Laguna-S-2.1-INT4 \
  --local-dir models/laguna-s-2.1-int4
```

The directory must contain `config.json`, `generation_config.json`,
`tokenizer.json`, `model.safetensors.index.json`, and all 15
`model-*.safetensors` shards.

## Build

```bash
cargo build --release
```

The production binary is:

```text
target/release/inferno
```

## Usage

### Chat

Start an interactive chat using the default `models/glm-5.2` directory:

```bash
target/release/inferno
```

Use `/clear` to reset the conversation and `/exit` to close the session. The
model and expert cache stay alive across turns, but each turn currently
re-prefills the accumulated conversation into a new request KV cache.

Start Laguna chat explicitly:

```bash
target/release/inferno chat \
  --model models/laguna-s-2.1-gguf
```

Inferno detects the architecture from `config.json` and the weight container
from the exact files in the model directory. The GGUF path reuses its mapped
weights and resets only sequence-specific F16 KV state before re-prefilling the
conversation. The Safetensors path also keeps its global expert cache alive
across turns.

### One-Shot Generation

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy."
```

Run the same request with Laguna:

```bash
target/release/inferno generate \
  --model models/laguna-s-2.1-gguf \
  --prompt "Tell me the capital of Italy."
```

Without `--max-new-tokens`, generation continues until an EOS token or the
context limit.

### Responses API and Codex

Inferno serves either model through its local streaming Responses API. Start
the selected persistent model process in one terminal.

GLM:

```bash
target/release/inferno serve --model models/glm-5.2
```

Laguna:

```bash
target/release/inferno serve --model models/laguna-s-2.1-gguf
```

The server exposes only the loaded model from `/v1/models`. Requests must use
`glm-5.2-q2` for GLM, `laguna-s-2.1-int4` for Laguna Safetensors, or
`laguna-s-2.1-gguf` for the Antirez GGUF.

Install the included Codex catalog and profiles:

```bash
mkdir -p ~/.codex
cp examples/inferno.config.toml ~/.codex/inferno.config.toml
cp examples/inferno-laguna.config.toml ~/.codex/inferno-laguna.config.toml
cp examples/inferno-laguna-gguf.config.toml ~/.codex/inferno-laguna-gguf.config.toml
cp examples/inferno.models.json ~/.codex/inferno.models.json
```

Restart Codex after replacing `inferno.models.json`; the `/models` picker reads
the catalog when the session starts. The Laguna GGUF profile uses a 4,096-token
service context and compacts at 3,072 tokens. Inferno rejects Codex fallback
metadata instead of starting an unexpectedly large prefill.

Start Codex with GLM:

```bash
codex --profile inferno
```

Start Codex with Laguna Safetensors:

```bash
codex --profile inferno-laguna
```

Start Codex with the Antirez GGUF:

```bash
codex --profile inferno-laguna-gguf
```

Codex sends each turn through the streaming Responses API. Inferno translates
the conversation and direct function tools into the loaded model's native
prompt, generates either text or a tool call, and returns that action to Codex.
The model, Metal backend, and model-specific expert cache remain alive between
requests.

Requests are processed one at a time because they share one Metal runtime.
While Laguna GGUF is generating, additional inference requests receive HTTP
429 instead of accumulating in memory. Its server path defaults to at most
2,048 output tokens unless `serve --max-new-tokens` sets an explicit cap.
The profiles expose only `exec_command` and `write_stdin` to the model. Codex
still enforces its sandbox and approval policy, but omitting unrelated tool
schemas keeps model prefill smaller. Plugin namespaces and hosted web search
are not exposed to the local model.

### Measurement

Measure decode throughput:

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy." \
  --max-new-tokens 8 \
  --measure-tokens-per-second
```

Replace the model path with `models/laguna-s-2.1-gguf` to measure the Antirez
GGUF.

Record memory telemetry without mixing it into generated text:

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy." \
  --telemetry-file /tmp/inferno-memory.log
```

Record adaptive cache decisions:

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy." \
  --enable-unified-memory-controller \
  --memory-controller-log /tmp/inferno-memory-controller.tsv
```

### Common Options

| Option | Purpose |
| --- | --- |
| `--max-new-tokens <N>` | Limits the number of generated tokens. |
| `--measure-tokens-per-second` | Reports time to first token and decode throughput. |
| `--expert-cache-gb <GB>` | Pins the routed-expert working-set budget for cache-backed artifacts. |
| `--hot-kv-cache-gb <GB>` | Pins the GLM hot Metal KV budget. |
| `--enable-unified-memory-controller` | Enables the model-specific adaptive memory controller; disabled by default. |
| `--enable-telemetry` | Prints GLM runtime memory telemetry. |
| `--telemetry-file <PATH>` | Writes GLM memory telemetry to a file. |
| `--memory-controller-log <PATH>` | Writes adaptive memory decisions to TSV. |
| `--speculative-mtp` | Enables the experimental GLM MTP path. |

Use the executable help as the authoritative CLI reference:

```bash
target/release/inferno --help
target/release/inferno generate --help
target/release/inferno chat --help
target/release/inferno serve --help
```

## Memory Strategy

Inferno uses the SSD as a large, slower storage tier and unified memory as a
smaller, faster tier. Cache policy is model-specific because GLM and Laguna
show different expert-locality and attention behavior.

**Expert cache.** GLM selects eight experts and uses a per-layer SLRU cache.
Keeping layer budgets separate matches GLM's measured locality and prevents one
layer from evicting useful experts from another.

The Laguna Safetensors path selects ten experts and uses one global O(1) LRU
cache across its routed layers. Its reuse is uneven across layers, so a global
budget lets layers with useful locality keep more experts while avoiding empty
or underused per-layer partitions. Cached experts are direct Metal views over
the mapped INT4 tensors, not second copies. On the 64 GB target, the automatic
working-set cap is 24 GB; smaller available-memory budgets reduce it
automatically. Missing experts are prefetched with parallel I/O workers.

The Antirez GGUF path has no separate logical expert cache. Each layer's
packed Q2_K or Q3_K expert tensor remains memory-mapped, and Metal receives a
zero-copy view of those bytes. Only the ten expert ranges selected by the
router are read by the kernel. macOS therefore manages physical residency
through its unified page cache instead of Inferno allocating a second copy.

```text
selected expert -> cache hit  -> execute from Metal
                -> cache miss -> read from SSD -> cache -> execute
```

**GLM KV cache.** Inferno stores the complete compressed history in an
append-only Q8 block store on SSD. A bounded Metal window contains only the
rows currently needed by attention. DSA reduces this work by selecting the
most relevant history rows for long-context attention.

```text
required KV row -> hot hit  -> read from Metal
                -> cold miss -> read from SSD -> Metal window -> use
```

GLM-5.2 MLA makes this practical by storing 512 latent values and 64 RoPE
values per token and layer instead of expanded K/V for every attention head.
The source GGUF remains memory-mapped so macOS can manage always-used weights
through its page cache.

**Laguna KV cache.** Twelve layers retain the full context, while 36
sliding-attention layers retain only their 512-token window. The Safetensors
path stores FP8 KV; the GGUF path stores F16 KV because the artifact has no
published FP8 KV scales. Both stay on Metal. The bounded sliding state lowers
KV pressure and leaves more unified memory available for model pages.

For GLM, the expert cache and hot KV window compete for the same physical
unified memory. Its fixed default starts from 30 expert slots per routed layer
and a 512 MiB hot KV budget. `--enable-unified-memory-controller` enables a
native GLM controller that samples Mach and Metal counters without starting
subprocesses, tracks separate prefill and decode Metal high-water marks, and
releases prefill-only buffers before decode.

During decode, the controller changes one cache at a time and measures the next
eight-token window. It keeps a change only when decode throughput improves;
otherwise it restores the previous budget. Expert slots and hot KV are tuned
independently. The controller targets 4 GiB of effective headroom, treats 3 GiB
as the hard floor, and shrinks immediately when swap or compression grows.
Explicit cache-size options pin that cache and disable automatic resizing for
it.

Laguna Safetensors has a separate controller because its experts use zero-copy
mapped weights and its FP8 KV cache is already bounded by sliding attention. It
starts from an automatically measured global expert working set capped at 24 GB
and observes 32-token decode windows. A proposed cache size is warmed and
measured, then the previous size is restored and measured again. The change is
retained only when the candidate beats both baseline samples by at least 10%.
A long generation can complete multiple controller windows without waiting for
another request. Short responses contribute their decode samples to the next
turn, while prompt prefill is excluded from timing and expert-cache counters.
Trial phases therefore continue across persistent chat turns and server
requests without counting unrelated prefill work. High expert SSD traffic can
trigger a larger-cache trial; smaller capacities are selected at the next
observation under measured memory pressure. Persistent runtimes pay the initial
stabilization window once and retain cache contents and controller decisions
across requests. After accepting or rejecting a candidate, the controller
waits 2,048 decode tokens before probing again. Normal compression of
reclaimable mapped pages is not treated as pressure while effective RAM
headroom remains safe. Swap growth, critically low headroom, or Metal
working-set pressure causes a shrink.

For Laguna Safetensors, `--expert-cache-gb` selects a fixed working set and
cannot be combined with the adaptive controller. The Antirez GGUF rejects both
expert-cache options because its mmap-backed expert residency is managed by
macOS rather than by that cache.

## Development

```bash
cargo fmt --all --check
cargo check --workspace
cargo test --workspace
```

## Credits

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) was created by Z.ai.

The target Q2 GGUF was produced and published by Antirez in
[antirez/glm-5.2-gguf](https://huggingface.co/antirez/glm-5.2-gguf).

[Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1-INT4) was created
and published by Poolside.

The mixed Q2_K/Q3_K Laguna GGUF is produced and published by Antirez in
[antirez/Laguna-S-2.1-GGUF](https://huggingface.co/antirez/Laguna-S-2.1-GGUF).

## License

Inferno is licensed under the [MIT License](LICENSE).
