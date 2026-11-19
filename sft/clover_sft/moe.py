"""Per-block preview routing and grouped expert matmuls with native tensor names."""

import torch
import torch.nn.functional as F

from .grouped_mm import _ExpertDownProjection
from .modeling_clover_moe import coreset_torch


def expert_forward(experts, hidden, ids, weights):
    flat = ids.flatten()
    order = flat.argsort(stable=True)
    tokens = torch.arange(ids.shape[0], device=hidden.device).repeat_interleave(ids.shape[1])[order]
    counts = torch.zeros(experts.gate_up_proj.shape[0], device=hidden.device, dtype=torch.int32)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    offsets = counts.cumsum(0, dtype=torch.int32)
    if hidden.is_cuda and hidden.dtype == torch.bfloat16:
        gate, up = F.grouped_mm(hidden[tokens], experts.gate_up_proj.transpose(1, 2), offs=offsets).chunk(2, dim=-1)
        activated = (F.silu(gate.float()) * up.float()).to(hidden.dtype)
        output = _ExpertDownProjection.apply(activated, experts.down_proj, offsets)
    else:
        # CPU/FP32 correctness oracle; the optimized CUDA recipe uses BF16.
        pieces = []
        start = 0
        for expert, count in enumerate(counts.tolist()):
            values = hidden[tokens[start:start + count]]
            gate, up = F.linear(values, experts.gate_up_proj[expert]).chunk(2, dim=-1)
            activated = (F.silu(gate.float()) * up.float()).to(hidden.dtype)
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                pieces.append(F.linear(activated.float(), experts.down_proj[expert].float()))
            start += count
        output = torch.cat(pieces, dim=0)
    output = (output.float() * weights.flatten()[order, None]).to(hidden.dtype)
    result = torch.zeros_like(hidden, dtype=torch.float32)
    return result.index_add(0, tokens, output.float()).to(hidden.dtype)


def load_balancing_loss(probabilities, ids, valid):
    """Switch loss over active routes, keeping the complete expert domain.

    Coresets may leave fewer than top-k experts with nonzero probability.
    Normalize assignment counts by the actual number of active routes, not
    by the padded top-k slots. Discrete routing statistics have no gradient.
    """
    probabilities = probabilities[valid]
    ids = ids[valid]
    active = probabilities.detach().gather(-1, ids) > 0
    counts = probabilities.new_zeros(probabilities.shape[-1])
    counts.scatter_add_(0, ids.flatten(), active.flatten().to(counts.dtype))
    probability_sum = probabilities.sum(dim=0)
    denominator = (counts.sum() * probabilities.shape[0]).clamp_min(1)
    return probabilities.shape[-1] * (probability_sum * counts).sum() / denominator


def moe_forward(module, hidden, layout, replay, *, compute_aux_loss=False):
    config = module.config
    batch, physical, dim = hidden.shape
    logits = module.gate(hidden).float()
    length, block = layout.length, layout.block_size
    valid = layout.physical_valid
    if "allowed" not in replay:
        preview = logits[:, :length].reshape(-1, block, config.num_experts)
        active = valid[:, :length].reshape(-1, block)
        if config.moe_block_coreset_enabled and config.moe_block_coreset_top_p < 1.0:
            filtered = coreset_torch(preview.detach(), active, config.num_experts_per_tok, config.moe_block_coreset_top_p)
            allowed = torch.isfinite(filtered).reshape(batch, length, config.num_experts)
        else:
            allowed = torch.ones_like(logits[:, :length], dtype=torch.bool)
        replay["allowed"] = allowed
    logits = torch.cat((logits[:, :length].masked_fill(~replay["allowed"], float("-inf")), logits[:, length:]), dim=1)
    if "ids" not in replay:
        replay["ids"] = logits.detach().argsort(dim=-1, descending=True, stable=True)[..., :config.num_experts_per_tok]
    ids = replay["ids"]
    probabilities = logits.softmax(dim=-1)
    scores = probabilities.gather(-1, ids)
    scores = scores / scores.sum(-1, keepdim=True).clamp_min(1e-12)
    flat_hidden = hidden[valid]
    output = expert_forward(module.experts, flat_hidden, ids[valid], scores[valid])
    if config.num_shared_experts:
        output = output + module.shared_expert(flat_hidden) * module.shared_expert_gate(flat_hidden).sigmoid()
    result = hidden.new_zeros((batch, physical, dim))
    result = result.index_put((valid,), output)
    if compute_aux_loss:
        return result, load_balancing_loss(probabilities, ids, valid)
    return result
