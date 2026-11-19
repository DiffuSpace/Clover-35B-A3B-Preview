# Clover SFT

Fine-tune [Clover](https://huggingface.co/Diffuspace/Clover-35B-A3B-Preview)
with block diffusion, using ms-swift for training orchestration and FSDP2 for
distributed full-parameter SFT. LoRA is supported as well. GSM8K is used below
as a quick-start example; you can supply your own conversation dataset.

## Architecture

The package registers a model and template through ms-swift’s external-plugin
interface. ms-swift manages data loading, optimization, scheduling and
checkpointing; Accelerate integrates FSDP2 sharding and decoder activation
checkpointing. Clover’s training behavior lives in `clover_sft/`:

```text
Conversation JSONL
  → CloverTemplate: tokenize, align blocks, pad
  → collator: sample noise and denoising targets
  → CloverForSFT: forward over [noisy tokens | clean tokens]
  → LM head: cross entropy on selected noisy positions
  → ms-swift Trainer: accumulated-token normalization and optimizer step
```

| Module | Responsibility |
| --- | --- |
| `swift_plugin.py` | Model/template registration and the ms-swift data/loss interface |
| `data.py` | Chat encoding, padding roles and noise sampling |
| `model.py`, `loss.py` | Training forward, chunked output projection/CE and native checkpoint export |
| `attention.py` | Block visibility, semantic RoPE positions and PyTorch FlexAttention |
| `gdn.py` | Clean-prefix convolution, independent right convolution and BD-GDN using public FLA operators |
| `moe.py`, `grouped_mm.py` | Block coreset routing, recomputation replay and grouped expert GEMMs |

Noisy blocks read their own noisy contents and preceding clean blocks. Clean
attention is causal. GDN preview reads the clean prefix state without modifying
or decaying it; clean recurrence updates retain decay. Quack grouped GEMMs keep
expert down-projection outputs in FP32 until router weighting. The model keeps
native Clover parameter names and exports standalone Hugging Face checkpoints.

## Quick start

The tested environment uses Python 3.12, PyTorch 2.11 with CUDA 12.9, and H200
GPUs. The grouped expert kernels require a Quack-supported Hopper/Blackwell
GPU. Use a CUDA-enabled PyTorch installation; dependency versions are specified
in `pyproject.toml`.

```bash
git clone https://github.com/DiffuSpace/Clover-35B-A3B-Preview.git
cd Clover-35B-A3B-Preview/sft

uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[test]'

# Download and convert GSM8K to messages JSONL.
python -m clover_sft.prepare_gsm8k

# Full-parameter training on eight GPUs.
NPROC_PER_NODE=8 bash scripts/train.sh
```

For a short run without checkpoint saving:

```bash
NPROC_PER_NODE=8 bash scripts/train.sh --max_steps 5 --save_strategy no
```

Use local model weights and your own dataset:

```bash
MODEL=/path/to/Clover-35B-A3B-Preview \
DATASET=/path/to/train.jsonl \
OUTPUT_DIR=outputs/my-sft \
NPROC_PER_NODE=8 bash scripts/train.sh \
  --gradient_accumulation_steps 2 --learning_rate 1e-5
```

The launcher accepts additional ms-swift training arguments. Full SFT uses
FSDP2 with BF16 computation and FP32 master parameters. Gradients are sharded
after every microbatch; optimizer updates follow the accumulation interval.

For LoRA on a GPU with enough memory for the complete base model:

```bash
TUNER_TYPE=lora NPROC_PER_NODE=1 bash scripts/train.sh \
  --lora_rank 8 --lora_alpha 16
```

LoRA targets attention/GDN projections; packed expert parameters remain frozen.

## Data and objective

Provide JSONL with one example per line, using the standard messages format:

```json
{"messages": [{"role": "user", "content": "Your question"}, {"role": "assistant", "content": "Your answer"}]}
```

The current template supports one text-only user/assistant exchange per example,
with an optional system message. It uses Clover’s non-thinking chat template and
appends EOS to the response. Each example occupies one batch row:

```text
prompt | pre-pad | response + EOS | post-pad | padding to 4096 tokens
```

- Block size is fixed to the checkpoint’s configuration: 32 for Clover Preview.
- Pre-pad aligns the response start; post-pad completes its final block.
  Padding does not shift the semantic positions of real tokens.
- Response/EOS and post-pad are eligible for masking and supervision. Prompt,
  pre-pad and batch padding are excluded. Post-pad participates bidirectionally
  within its noisy block; clean history skips padding.
- One masking probability is sampled per example. Loss is mean cross entropy
  over selected positions, with token-weighted normalization across accumulated
  microbatches and data-parallel ranks. Noise streams are independent per rank.

The forward processes two copies of each sequence, giving 8192 physical slots.
Examples exceeding the 4096-token limit after alignment are skipped. The
objective is diffusion CE plus router load-balancing auxiliary loss, with
`router_aux_loss_coef=0.001` by default. Set `--router_aux_loss_coef 0` to disable
the auxiliary term. Packing, context parallelism and multi-turn training are
not enabled. Keep `max_length=4096`, `use_logits_to_keep=false` and label
smoothing disabled.

The router loss uses Switch-style statistics for each layer's active physical
tokens: real clean tokens and noisy response/post-pad. Inactive slots are
excluded. Statistics use coreset-filtered probabilities, actual nonzero routes
and the complete expert count. Layer losses are summed; the logged `aux_loss`
is this unweighted sum. Auxiliary losses are averaged over microbatches and
data-parallel ranks, independently of CE's supervised-token normalization.
Recomputation reuses routing decisions and recomputes differentiable scores.

## Checkpoints

Full SFT saves native Clover weights and inference code. By default the launcher
saves model weights only; restarting from them does not restore optimizer or
scheduler state. LoRA saves adapters and requires the base model for inference.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "outputs/my-sft/<run>/checkpoint-<step>"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, dtype="bfloat16", device_map="auto"
)
```

The bundled inference code includes the corrected preview history read.
For LoRA inference, use that corrected implementation with the base model;
adapter weights alone do not include inference source files.

## Development

```bash
python scripts/check_scope.py
ruff check .
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m pytest -q tests
```

Tests cover padding/visibility, inference-compatible block execution, router
balance and accumulation scaling, gradients, recomputation, grouped kernels
and two-rank FSDP2. CUDA tests are skipped when
no suitable devices are visible.

See [ATTRIBUTIONS.md](ATTRIBUTIONS.md), [UPSTREAM.json](UPSTREAM.json) and
[LICENSE](LICENSE) for source provenance and licensing.

[← Back to Clover](../README.md)
