# Inferno

<p align="center">
  <img src="assets/logo.svg" alt="Inferno logo" width="150">
</p>

Inferno is a lightweight Rust inference engine built specifically for running
GLM-5.2 Q2 on Apple Silicon through native Metal kernels. Its primary target is
a MacBook Pro with 64 GB of unified memory.

The name reflects the engineering challenge: the model is much larger than the
available memory, so useful local inference requires careful coordination of
Metal, unified memory, and SSD streaming. Inferno stays deliberately narrow. It
supports one model layout and optimizes that path instead of becoming a general
inference framework.

> [!WARNING]
> Inferno is experimental and under active development. It is not production
> ready, and correctness, stability, long-context behavior, and performance are
> still being validated.

## Current Scope

- Apple Silicon and Metal only.
- `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` only.
- Exact top-8 routed-expert execution.
- Native Metal execution with no CPU compute fallback.
- Interactive chat and one-shot generation.
- SSD-streamed routed experts and compressed KV history.
- Greedy decoding.
- Experimental MTP speculative decoding, disabled by default because it is
  currently slower than ordinary decode on the 64 GB target.

Inferno is not a general model loader and does not support arbitrary GGUF
architectures or quantization formats.

## Performance

| Metric | Result |
| --- | ---: |
| Decode throughput | **1.711 tokens/s** |

Measured with exact top-8 routing on a 64 GB Apple Silicon MacBook Pro using a
release build. Prompt prefill is excluded. Results depend on prompt length,
expert-cache state, SSD activity, and memory pressure.

## Requirements

- A Mac with Apple Silicon and Metal support.
- 64 GB of unified memory is the current development target.
- Rust and Cargo.
- The Hugging Face CLI for the download commands below.
- More than 503 GB of free SSD space for the source GGUF and ExpertPack,
  excluding temporary KV data and build outputs.

Install the Hugging Face CLI with Homebrew if needed:

```bash
brew install hf
```

## Model

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

### Download

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

Use an explicit model directory when needed:

```bash
target/release/inferno chat --model /path/to/model
```

### One-Shot Generation

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy."
```

Without `--max-new-tokens`, generation continues until an EOS token or the
context limit.

### Measurement

Measure decode throughput:

```bash
target/release/inferno generate \
  --model models/glm-5.2 \
  --prompt "Tell me the capital of Italy." \
  --max-new-tokens 8 \
  --measure-tokens-per-second
```

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
| `--expert-cache-gb <GB>` | Pins the routed-expert RAM budget. |
| `--hot-kv-cache-gb <GB>` | Pins the hot Metal KV budget. |
| `--enable-unified-memory-controller` | Enables adaptive expert and hot-KV memory rebalancing. |
| `--enable-telemetry` | Prints runtime memory telemetry. |
| `--telemetry-file <PATH>` | Writes memory telemetry to a file. |
| `--memory-controller-log <PATH>` | Writes adaptive memory decisions to TSV. |
| `--speculative-mtp` | Enables the experimental MTP path. |

Use the executable help as the authoritative CLI reference:

```bash
target/release/inferno --help
target/release/inferno generate --help
target/release/inferno chat --help
```

## Memory Strategy

Inferno uses the SSD as a large, slower storage tier and unified memory as a
smaller, faster tier. It maintains two separate caches.

**Expert cache.** For every token, the router selects eight experts. If an
expert is already resident in a Metal cache slot, Inferno uses it immediately.
Otherwise, Inferno reads that expert's Q2 weights from the ExpertPack on SSD,
places them in a reusable slot, and executes it. When the cache is full, a less
useful expert is replaced. Reusing a resident expert avoids another SSD read.

```text
selected expert -> cache hit  -> execute from Metal
                -> cache miss -> read from SSD -> cache -> execute
```

**KV cache.** The KV cache is the model's memory of previous tokens. Inferno
stores the complete compressed history in an append-only Q8 block store on SSD.
A bounded Metal window contains only the rows currently needed by attention.
If a row is not in that window, Inferno reads it from SSD and loads it into the
window. DSA reduces this work by selecting only the most relevant history rows
for long-context attention.

```text
required KV row -> hot hit  -> read from Metal
                -> cold miss -> read from SSD -> Metal window -> use
```

GLM-5.2 MLA makes this practical by storing 512 latent values and 64 RoPE
values per token and layer instead of expanded K/V for every attention head.
The source GGUF remains memory-mapped so macOS can manage always-used weights
through its page cache.

The expert cache and hot KV window compete for the same physical unified
memory. Inferno's fixed default starts from 30 expert slots per routed layer
and a 512 MiB hot KV budget. `--enable-unified-memory-controller` enables a
native controller that samples Mach and Metal counters without starting
subprocesses, tracks separate prefill and decode Metal high-water marks, and
releases prefill-only buffers before decode.

During decode, the controller changes one cache at a time and measures the next
eight-token window. It keeps a change only when decode throughput improves;
otherwise it restores the previous budget. Expert slots and hot KV are tuned
independently. The controller targets 4 GiB of effective headroom, treats 3 GiB
as the hard floor, and shrinks immediately when swap or compression grows.
Explicit cache-size options pin that cache and disable automatic resizing for
it.

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

## License

Inferno is licensed under the [MIT License](LICENSE).
