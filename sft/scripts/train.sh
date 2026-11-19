#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-Diffuspace/Clover-35B-A3B-Preview}"
MODEL_REVISION="${MODEL_REVISION:-4b0402b65e523d0183a14c431955438714cb16ff}"
DATASET="${DATASET:-data/gsm8k/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gsm8k}"
TUNER_TYPE="${TUNER_TYPE:-full}"
ROUTER_AUX_LOSS_COEF="${ROUTER_AUX_LOSS_COEF:-0.001}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

parallel=()
gradient_checkpointing=true
model_dtype=bfloat16
if [[ "$TUNER_TYPE" == full ]]; then
    model_dtype=float32
fi
if [[ "$TUNER_TYPE" == full && "$NPROC_PER_NODE" -gt 1 ]]; then
    # Load native BF16 weights; Accelerate prepares FP32 master parameters
    # for FSDP mixed-precision optimization.
    model_dtype=bfloat16
    # Reduce-scatter every microbatch so accumulation retains sharded gradients.
    # Optimizer updates still follow gradient_accumulation_steps.
    parallel=(--fsdp configs/fsdp.json --accelerator_config
        '{"dispatch_batches":false,"gradient_accumulation_kwargs":{"sync_each_batch":true}}')
    gradient_checkpointing=false # FSDP wraps decoder layers with activation checkpointing.
fi
if [[ ! -f "$DATASET" ]]; then
    echo "Dataset not found: $DATASET; run python -m clover_sft.prepare_gsm8k first." >&2
    exit 1
fi

exec swift sft \
    --external_plugins clover_sft/swift_plugin.py \
    --model "$MODEL" --model_revision "$MODEL_REVISION" \
    --model_type clover_sft --template clover_sft --use_hf true \
    --dataset "$DATASET" --split_dataset_ratio 0 \
    --packing false --padding_free false --max_length 4096 --truncation_strategy delete \
    --sequence_parallel_size 1 --use_logits_to_keep false --router_aux_loss_coef "$ROUTER_AUX_LOSS_COEF" \
    --tuner_type "$TUNER_TYPE" \
    --target_modules q_proj k_proj v_proj o_proj in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj \
    --torch_dtype "$model_dtype" --bf16 true --fp16 false \
    --gradient_checkpointing "$gradient_checkpointing" --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
    --learning_rate 1e-5 --num_train_epochs 1 --warmup_ratio 0.03 \
    --dataloader_drop_last true --dataloader_num_workers 0 --dataset_num_proc 1 \
    --logging_steps 1 --save_steps 200 --save_total_limit 1 --save_only_model true \
    --report_to none --output_dir "$OUTPUT_DIR" \
    "${parallel[@]}" "$@"
