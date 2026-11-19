"""Differentiable single-sample BD-GDN using public PyTorch and FLA operations.

Noisy queries read the clean state without writing it (beta=0, decay=1).
Interleaving each noisy block's reads before its clean block's updates allows
one ordinary gated-delta scan, with gradients through the entire clean prefix.
There is no packing planner or cross-rank state transport.
"""

import torch
import torch.nn.functional as F

from .modeling_clover_moe import recurrent_commit_torch


def _gather(raw, indices):
    return raw.gather(1, indices.clamp_min(0)[..., None].expand(-1, -1, raw.shape[-1])) * (indices >= 0)[..., None]


def convolution(module, raw, layout):
    clean_valid, noisy_valid = layout.clean_valid, layout.noisy_valid
    length, block = layout.length, layout.block_size
    noisy, clean = raw[:, :length], raw[:, length:]
    index = torch.arange(length, device=raw.device)[None].expand(raw.shape[0], -1)
    latest = torch.where(clean_valid, index, -1).cummax(1).values

    def previous(cursor):
        value = latest.gather(1, (cursor - 1).clamp_min(0))
        return torch.where(cursor > 0, value, -1)

    block_start = index // block * block
    history = torch.where(block_start > 0, latest.gather(1, (block_start - 1).clamp_min(0)), -1)
    width = module.conv1d.weight.shape[-1]
    histories = []
    for _ in range(width - 1):
        histories.append(history)
        history = previous(history)
    history_table = torch.stack(histories, dim=-1) if histories else None
    y_clean = torch.zeros_like(clean, dtype=torch.float32)
    y_noisy = torch.zeros_like(noisy, dtype=torch.float32)
    clean_cursor = index
    for distance in range(width):
        weight = module.conv1d.weight[:, 0, width - distance - 1]
        # Match the released clean convolution's input-dtype products/FP32 sum.
        y_clean = y_clean + (_gather(clean, clean_cursor) * weight).float()
        if distance == 0:
            source = noisy
        else:
            offset = distance - index.remainder(block) - 1
            old = history_table.gather(2, offset.clamp(0, width - 2)[..., None]).squeeze(-1)
            source = torch.where(
                (index.remainder(block) >= distance)[..., None],
                _gather(noisy, index - distance), _gather(clean, old),
            )
        y_noisy = y_noisy + source.float() * weight.float()
        clean_cursor = previous(clean_cursor)
    for distance in range(1, min(width, block)):
        right = index + distance
        valid = (index.remainder(block) + distance < block) & (right < length)
        source = _gather(noisy, right.clamp_max(length - 1))
        y_noisy = y_noisy + source.float() * module.conv_right_weight[:, 0, distance - 1].float() * valid[..., None]
    y_noisy = F.silu(y_noisy).to(raw.dtype) * noisy_valid[..., None]
    y_clean = F.silu(y_clean).to(raw.dtype) * clean_valid[..., None]
    return torch.cat((y_noisy, y_clean), dim=1)


def interleave(noisy, clean, block):
    batch, length = noisy.shape[:2]
    shape = (batch, length // block, block, *noisy.shape[2:])
    return torch.stack((noisy.reshape(shape), clean.reshape(shape)), dim=2).flatten(1, 3).contiguous()


def uninterleave(value, block):
    batch, doubled = value.shape[:2]
    shaped = value.reshape(batch, doubled // (2 * block), 2, block, *value.shape[2:])
    return shaped[:, :, 0].flatten(1, 2), shaped[:, :, 1].flatten(1, 2)


def scan(q, k, v, g, beta, layout):
    length, block = layout.length, layout.block_size
    q_noisy, q_clean = q[:, :length], q[:, length:]
    k_clean, v_clean = k[:, length:], v[:, length:]
    valid = layout.clean_valid
    q_read = q_noisy * layout.noisy_valid[..., None, None]
    q_clean = q_clean * valid[..., None, None]
    k_clean = k_clean * valid[..., None, None]
    v_clean = v_clean * valid[..., None, None]
    q_scan = interleave(q_read, q_clean, block)
    k_scan = interleave(torch.zeros_like(k_clean), k_clean, block)
    v_scan = interleave(torch.zeros_like(v_clean), v_clean, block)
    g_clean = g[:, length:] * valid[..., None]
    beta_clean = beta[:, length:] * valid[..., None]
    g_scan = interleave(torch.zeros_like(g_clean), g_clean, block).float()
    beta_scan = interleave(torch.zeros_like(beta_clean), beta_clean, block).float()
    heads = v_scan.shape[2]
    q_scan = q_scan.repeat_interleave(heads // q_scan.shape[2], dim=2)
    k_scan = k_scan.repeat_interleave(heads // k_scan.shape[2], dim=2)
    if layout.gdn_backend == "torch":
        initial = torch.zeros(
            (q.shape[0], heads, v.shape[-1], q.shape[-1]), device=q.device, dtype=torch.float32,
        )
        output, _ = recurrent_commit_torch(
            q_scan.float(), k_scan.float(), v_scan.float(), g_scan, beta_scan, initial,
            torch.ones(q_scan.shape[:2], device=q.device, dtype=torch.bool),
        )
    elif layout.gdn_backend == "fla":
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        output, _ = chunk_gated_delta_rule(
            q_scan.float(), k_scan.float(), v_scan.float(), g_scan, beta_scan,
            scale=q.shape[-1] ** -0.5, use_qk_l2norm_in_kernel=True,
        )
    else:
        raise ValueError(f"Unknown GDN backend: {layout.gdn_backend}")
    return uninterleave(output, block)


def gdn_forward(module, hidden, layout):
    batch, physical, _ = hidden.shape
    length, block = layout.length, layout.block_size
    active = torch.cat((layout.noisy_valid, layout.clean_valid), dim=1)
    x = hidden * active[..., None]
    mixed = convolution(module, module.in_proj_qkv(x), layout)
    q, k, v = mixed.split((module.kh * module.kd, module.kh * module.kd, module.vh * module.vd), dim=-1)
    q = q.reshape(batch, physical, module.kh, module.kd)
    k = k.reshape(batch, physical, module.kh, module.kd)
    v = v.reshape(batch, physical, module.vh, module.vd)
    g = -module.A_log.float().exp() * F.softplus(module.in_proj_a(x).float() + module.dt_bias.float())
    beta = module.in_proj_b(x).sigmoid()
    history_readout, clean = scan(q, k, v, g, beta, layout)
    # Preview reads the clean prefix without decaying it. Only clean-stream
    # state updates in scan() consume the decay gates.
    history_readout = history_readout.float()
    qf = q[:, :length].float().repeat_interleave(module.vh // module.kh, dim=2)
    kf = k[:, :length].float().repeat_interleave(module.vh // module.kh, dim=2)
    qf = qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)
    kf = kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
    qf = qf.reshape(batch, -1, block, module.vh, module.kd)
    kf = kf.reshape_as(qf)
    vf = v[:, :length].float().reshape(batch, -1, block, module.vh, module.vd)
    valid = layout.noisy_valid.reshape(batch, -1, block)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        logits = torch.einsum("bnthk,bnshk->bnhts", qf, kf)
    logits = logits * module.bd_softmax_logit_scale.float()[None, None, :, None, None]
    logits = logits.masked_fill(~valid[:, :, None, None, :], float("-inf"))
    history_logits = module.bd_softmax_history_bias.float()[None, None, :, None, None]
    history_logits = history_logits + (
        module.bd_softmax_history_c.float()[None, None, :, None, None]
        * (valid.sum(-1).clamp_min(1).float() / 32.0).log()[:, :, None, None, None]
    )
    history_logits = history_logits.expand(batch, length // block, module.vh, block, 1)
    weights = torch.softmax(torch.cat((logits, history_logits), dim=-1), dim=-1)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        local = torch.einsum("bnhts,bnshv->bnthv", weights[..., :-1], vf)
    local = local.reshape(batch, length, module.vh, module.vd)
    historical_weight = weights[..., -1].permute(0, 1, 3, 2).reshape(batch, length, module.vh)
    noisy = (local + historical_weight[..., None] * history_readout).to(hidden.dtype)
    noisy = noisy * layout.noisy_valid[..., None, None]
    output = torch.cat((noisy, clean.to(hidden.dtype)), dim=1)
    z = module.in_proj_z(x).reshape(batch, physical, module.vh, module.vd)
    output = module.norm(output, z)
    return module.out_proj(output.reshape(batch, physical, -1))
