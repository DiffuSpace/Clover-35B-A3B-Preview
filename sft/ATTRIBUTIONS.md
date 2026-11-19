# Clover upstream attribution and modifications

Clover is derived from [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
Upstream model assets: Copyright 2026 Alibaba Cloud. See the accompanying LICENSE
for the upstream Apache License, Version 2.0, including its copyright statement.

Modified upstream assets distributed with Clover:

- `config.json`: text-only Clover MoE architecture, ordinary RoPE, removal of
  vision/MTP and training settings, a 128K context limit, and custom Hugging Face
  inference metadata.
- `generation_config.json`: Clover generation defaults, repetition and length controls,
  and version metadata.
- `tokenizer.json` and `tokenizer_config.json`: token 151669 remapped to `<|mask|>`,
  affected BPE merges removed, special-token metadata updated, and multimodal
  processor configuration removed; context limit set to 128K. Other vocabulary
  IDs are preserved.
- `chat_template.jinja`: text-only content and simplified inference rendering;
  the file carries its own source and modification notice.
- Exported checkpoint shards and index: vision and MTP tensors removed and
  text tensor names normalized by Clover's conversion scripts.

`modeling_clover_moe.py` derives from the Hugging Face Transformers Qwen3.5-MoE
implementation used for the Qwen3.6 architecture. Its upstream copyright is
Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
Clover adds block-denoising generation, fixed GDN preview and clean-state caching.
The source URL and modification notice are retained in that file.
Clover also aligns MoE accumulation and causal convolution activation with its
SGLang serving implementation, while keeping
the standalone PyTorch causal recurrence.

The right-convolution and block-size-corrected softmax GDN operations in
`modeling_clover_moe.py` are the Clover authors' own PyTorch implementation.

`sampling_clover_moe.py` implements response-only penalties and confidence
handling to match the SGLang Clover LowConfidence decoding semantics.

These attributions describe provenance and do not imply upstream endorsement.

## Local SFT package correction

The local `modeling_clover_moe.py` removes the extra preview-only decay on
clean-state history reads to match Clover training semantics. Clean recurrent
state decay and all parameter names/shapes are preserved. This correction is
also included in future native inference exports. Original and local source
hashes are recorded in `UPSTREAM.json`.
