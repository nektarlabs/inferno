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

This is not a general model zoo. Inferno keeps the inference path small,
explicit, and optimized for Q2 GLM-5.2-style execution so it can become fast
enough for real local use while staying simple enough to audit end to end. That
is the ambition — whether it gets all the way there is exactly what the
experiment is testing.

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
  --measure-tokens-per-second
```

Inspect telemetry while generation is running:

```bash
tail -f /tmp/inferno-memory.log
```

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

## License

Inferno is licensed under the MIT License.
