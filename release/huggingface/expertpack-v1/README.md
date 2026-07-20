---
license: mit
base_model: antirez/glm-5.2-gguf
tags:
  - glm-5.2
  - gguf
  - q2
  - moe
  - apple-silicon
  - metal
  - inferno
---

# Inferno ExpertPack v1 for GLM-5.2 Q2

This repository contains the Inferno ExpertPack for
`GLM-5.2-UD-Q2_K_RoutedQ2K.gguf`.

The ExpertPack is a lossless physical reordering of the routed Q2 expert
payloads published by Antirez. It keeps the quantized expert bytes unchanged
and places each expert's gate, up, and down matrices next to each other. This
layout is designed for demand-driven SSD streaming by Inferno on Apple Silicon.

## Important

This artifact is not a standalone model and does not replace the source GGUF.
Inferno also requires:

- `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf`
- `config.json`
- `tokenizer.json`

These source files are available from
[antirez/glm-5.2-gguf](https://huggingface.co/antirez/glm-5.2-gguf).

## Artifact

| Field | Value |
| --- | --- |
| File | `GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack` |
| Size | `240987934720` bytes |
| Format | Inferno ExpertPack |
| Format version | `1` |
| Source | `GLM-5.2-UD-Q2_K_RoutedQ2K.gguf` |
| Runtime | [Inferno](https://github.com/nektarlabs/inferno) |

The file starts with a 4 KiB header followed by fixed-size expert records in
layer-major, expert-major order:

```text
header -> [layer][expert][gate][up][down]
```

This lets Inferno issue large contiguous SSD reads for selected experts instead
of relying on scattered page faults into the component-major GGUF layout.

## Intended Use

The ExpertPack is intended only for running the target GLM-5.2 Q2 artifact with
Inferno on Apple Silicon. It contains routed-expert weights only. Embeddings,
attention weights, dense layers, normalization weights, the output head,
configuration, and tokenizer remain in their original files.

## Setup

Place the source files and ExpertPack in one directory:

```text
models/glm-5.2/
  config.json
  tokenizer.json
  GLM-5.2-UD-Q2_K_RoutedQ2K.gguf
  GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack
```

Build and start Inferno:

```bash
git clone https://github.com/nektarlabs/inferno.git
cd inferno
cargo build --release
target/release/inferno
```

Inferno discovers the ExpertPack in the model directory and validates its
version, source layout fingerprint, dimensions, component strides, and exact
file size before use. A pack generated from a different GGUF is rejected.

## Download

After this repository is published, the artifact can be downloaded with:

```bash
hf download allemanfredi/inferno-glm-5.2-q2-expertpack \
  GLM-5.2-UD-Q2_K_RoutedQ2K-Inferno-ExpertPack-v1.q2pack \
  --local-dir models/glm-5.2
```

## Reproducibility

The ExpertPack can be recreated losslessly from the source GGUF:

```bash
cargo run --release -p inferno --example pack_experts -- \
  --model models/glm-5.2
```

Creation is resumable at complete expert-record boundaries. The final file is
published atomically after its structure and size have been validated.

## Limitations

- This format is specific to Inferno and is not a GGUF replacement.
- Compatibility is limited to the exact source layout recorded in the header.
- Runtime behavior and output quality remain those of the source quantized
  model because expert values are not modified.
- This work is under active development and has not been validated on every
  Apple Silicon configuration.

## Credits

[GLM-5.2](https://huggingface.co/zai-org/GLM-5.2) was created and released by
Z.ai.

The Q2 GGUF used as the direct source for this artifact was produced and
published by Antirez in
[antirez/glm-5.2-gguf](https://huggingface.co/antirez/glm-5.2-gguf).

Inferno changes only the physical ordering of routed-expert payloads required
for its SSD streaming path. It does not modify their quantized values.

## License

This derived artifact is distributed under the MIT License, following the
license of the source GLM-5.2 model. See the source model repositories and the
[Inferno license](https://github.com/nektarlabs/inferno/blob/main/LICENSE) for
details.
