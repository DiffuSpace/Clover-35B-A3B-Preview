<p align="center">
  <a href="https://diffuspace.ai/">
    <img src="https://assets.diffuspace.ai/brand/logofull.svg" alt="DiffuSpace" width="360" />
  </a>
</p>

<h1 align="center">Clover-35B-A3B-Preview</h1>

<p align="center">
  <b>The first diffusion language model from DiffuSpace.</b>
</p>

<p align="center">
  <a href="https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview"><img src="https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Model-yellow" alt="Hugging Face" /></a>
  <a href="https://modelscope.cn/organization/Diffuspace"><img src="https://img.shields.io/badge/ModelScope-DiffuSpace-blue" alt="ModelScope" /></a>
  <a href="https://diffuspace.ai/blog/clover-35b/"><img src="https://img.shields.io/badge/Blog-Clover-green" alt="Blog" /></a>
  <a href="https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/blob/main/LICENSE"><img src="https://img.shields.io/badge/Model_License-Apache_2.0-blue" alt="Model License: Apache 2.0" /></a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#inference-backends">Inference Backends</a> ·
  <a href="#supervised-fine-tuning">Fine-Tuning</a>
</p>

## Introduction

We introduce **Clover-35B-A3B-Preview**, the first model released by **DiffuSpace**. Clover delivers state-of-the-art performance among open-source diffusion language models at release on the benchmarks reported in our [launch blog](https://diffuspace.ai/blog/clover-35b/).

Built on [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), Clover combines a text-only mixture-of-experts backbone with block diffusion generation:

- **35B total parameters, approximately 3B active per token.** A hybrid Gated DeltaNet (GDN) / full-attention MoE architecture.
- **Bidirectional attention within each block.** Additional learned GDN parameters support block diffusion while preserving a causal prefix.
- **128K context.** Up to 131,072 tokens across the prompt and generated response.

## Performance

[![Clover agentic benchmark results](https://diffuspace.ai/assets/blog/clover/agentic-benchmarks.jpg)](https://diffuspace.ai/blog/clover-35b/#agentic-benchmarks)

See the [launch blog](https://diffuspace.ai/blog/clover-35b/#agentic-benchmarks) for benchmark results, evaluation settings, and comparisons.

## Quick Start

### Hugging Face Transformers

Install PyTorch with CUDA support, then the Hugging Face dependencies. This example uses Transformers 5.19.0 and PyTorch 2.10 or later.

```bash
pip install "transformers==5.19.0" accelerate safetensors
```

The example runs on one GPU with enough memory for the BF16 weights (approximately 69 GB) plus inference state.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Diffuspace/Clover-35B-A3B-Preview"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()

messages = [{"role": "user", "content": "What is diffusion?"}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
inputs = tokenizer(
    prompt, add_special_tokens=False, return_tensors="pt"
).to("cuda:0")

output = model.generate(**inputs, max_new_tokens=24576)
completion = output[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(completion, skip_special_tokens=True))
```

Generation defaults are included in the checkpoint. For decoding options and configuration details, see the [Hugging Face model card](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview#generation-controls).

## Inference Backends

| Backend | Use case | Decoding |
| --- | --- | --- |
| [Hugging Face Transformers](#hugging-face-transformers) | Standalone Python inference and experimentation | Block diffusion denoising |
| [SGLang](#sglang-and-llamacpp) | Serving and evaluation | Block diffusion denoising and Self-speculative decoding |
| [llama.cpp](#sglang-and-llamacpp) | Local inference | Self-speculative decoding |

### SGLang and llama.cpp

Our SGLang implementation supports both **Block diffusion denoising** and **Self-speculative decoding** for Clover. The llama.cpp implementation supports **Self-speculative decoding**. Backend code and setup guides will be added to this repository.

> [!NOTE]
> We recommend our official **SGLang implementation for serving and evaluation**. Small numerical differences between the Hugging Face implementation and SGLang's kernels can lead to different generated text.

## Supervised Fine-Tuning

**Coming soon.** SFT code and training instructions will be released in [`sft/`](./sft/).

## License and Acknowledgments

The model is distributed under **Apache-2.0**. See the [model license](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/blob/main/LICENSE) and [attributions](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview/blob/main/ATTRIBUTIONS.md) for details.

Clover builds on Qwen3.6-35B-A3B. We thank the [Qwen](https://github.com/QwenLM), [Hugging Face Transformers](https://github.com/huggingface/transformers), [SGLang](https://github.com/sgl-project/sglang), and [llama.cpp](https://github.com/ggml-org/llama.cpp) communities for their open-source work.
