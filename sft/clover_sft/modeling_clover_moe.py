# Copyright 2025 The Qwen Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# Derived from Hugging Face Transformers Qwen3.5-MoE (used by Qwen3.6):
# https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
# Modified for Clover: standalone Torch block-denoising inference, fixed GDN
# preview, clean-state caches, and text-only MoE architecture.
# Local SFT correction: remove preview-only history decay; preserve clean decay.
"""Standalone HF inference for Clover's causal-prefix/block-bidirectional MoE.

The architecture fixes softmax-blocksize GDN with independent right convolution,
a causal prefix, bidirectional block preview, and same-position logits.
Checkpoint names remain unchanged. Preview never mutates a clean cache; completed
blocks are replayed causally before their state is persisted. The implementation
uses standalone PyTorch operations defined in this file.
"""

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, GenerationConfig
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.generation.utils import GenerateDecoderOnlyOutput

from .configuration_clover_moe import CloverMoeConfig
from .sampling_clover_moe import (
    PENALTY_DEFAULTS,
    CloverPenaltyState,
    predict_with_confidence,
    validate_penalties,
)


# Clover authors' own right-convolution and GDN preview operations.
def bd_gdn_conv1d(
    mixed_qkv: torch.Tensor,
    *,
    clean_conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
    conv_weights: torch.Tensor,
    block_size: int,
    active_valid_mask: torch.Tensor,
    conv_right_weight: torch.Tensor,
) -> torch.Tensor:
    """Apply Clover independent right convolution and SiLU without updating cache.

    ``clean_conv_states`` contains the last ``kernel_size - 1`` committed QKV
    projections. Preview tokens are appended only in a temporary tensor.
    """
    num_tokens, channels = mixed_qkv.shape
    batch_size = cache_indices.numel()
    if num_tokens != batch_size * block_size:
        raise ValueError(
            "BD-GDN preview requires one fixed-size block per request: "
            f"{num_tokens=} != {batch_size=} * {block_size=}"
        )
    if active_valid_mask.numel() != num_tokens:
        raise ValueError(f"{active_valid_mask.numel()=} does not match {num_tokens=}")

    safe_cache_indices = cache_indices.to(torch.long).clamp_min(0)
    history = clean_conv_states.index_select(0, safe_cache_indices)
    if history.shape[1] != channels:
        raise ValueError(
            f"BD-GDN conv cache has {history.shape[1]} channels, expected {channels}"
        )

    valid = (
        active_valid_mask.reshape(batch_size, block_size)
        & (cache_indices >= 0)[:, None]
    )
    active_requests = valid.any(dim=1)
    history = history * active_requests[:, None, None].to(history.dtype)
    current = mixed_qkv.reshape(batch_size, block_size, channels)
    current = current * valid[:, :, None].to(current.dtype)
    convolution_input = torch.cat(
        (history, current.transpose(1, 2)),
        dim=-1,
    )

    weight = conv_weights.reshape(channels, 1, -1)
    output = F.conv1d(
        convolution_input.float(),
        weight.float(),
        groups=channels,
    )
    width = weight.shape[-1]
    expected_shape = (channels, 1, width - 1)
    if tuple(conv_right_weight.shape) != expected_shape:
        raise ValueError(f"Clover right weight shape must be {expected_shape}")
    for distance in range(1, min(width, block_size)):
        output[:, :, :-distance] += (
            current[:, distance:, :].transpose(1, 2).float()
            * conv_right_weight[:, 0, distance - 1].float()[None, :, None]
        )
    output = F.silu(output)

    output = output.transpose(1, 2)
    output = output * valid[:, :, None].to(output.dtype)
    return output.reshape(num_tokens, channels).to(mixed_qkv.dtype)


def bd_gdn_local_memory(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    *,
    clean_ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    active_valid_mask: torch.Tensor,
    scale: float,
    softmax_logit_scale: torch.Tensor,
    softmax_history_bias: torch.Tensor,
    softmax_history_c: torch.Tensor,
) -> torch.Tensor:
    """Clover bidirectional softmax preview with a read-only clean-state bucket.

    Normalized q/k determine block attention logits without beta. The history
    bucket logit is bias + history_c * log(valid_block_size / 32), and its
    value is the undecayed recurrent readout from clean [B, H, V, K] state.
    ``g`` is retained for call compatibility; only clean recurrence uses decay.
    """
    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(f"Expected query shape [1, T, H, K], got {query.shape}")
    num_tokens = query.shape[1]
    batch_size = cache_indices.numel()
    if num_tokens % batch_size != 0:
        raise ValueError(
            f"BD-GDN token count {num_tokens} is not divisible by batch {batch_size}"
        )
    block_size = num_tokens // batch_size
    if active_valid_mask.numel() != num_tokens:
        raise ValueError(f"{active_valid_mask.numel()=} does not match {num_tokens=}")

    num_value_heads = value.shape[2]
    if num_value_heads % query.shape[2] or num_value_heads % key.shape[2]:
        raise ValueError("BD-GDN value heads must be divisible by query and key heads")
    query = query.repeat_interleave(num_value_heads // query.shape[2], dim=2)
    key = key.repeat_interleave(num_value_heads // key.shape[2], dim=2)

    valid = active_valid_mask.reshape(batch_size, block_size)
    valid_float = valid[:, :, None].float()
    query_float = query.reshape(
        batch_size, block_size, num_value_heads, query.shape[-1]
    ).float()
    key_float = key.reshape(
        batch_size, block_size, num_value_heads, key.shape[-1]
    ).float()
    value_float = value.reshape(
        batch_size, block_size, num_value_heads, value.shape[-1]
    ).float()
    query_float = query_float * torch.rsqrt(
        query_float.square().sum(dim=-1, keepdim=True) + 1e-6
    )
    key_float = key_float * torch.rsqrt(
        key_float.square().sum(dim=-1, keepdim=True) + 1e-6
    )

    safe_cache_indices = cache_indices.to(torch.long).clamp_min(0)
    clean_state = clean_ssm_states.index_select(0, safe_cache_indices).float()
    active_requests = valid.any(dim=1)
    clean_state = clean_state * active_requests[:, None, None, None].float()

    scaled_query = query_float * scale
    clean_output = torch.einsum(
        "blhk,bhvk->blhv",
        scaled_query,
        clean_state,
    )
    history = clean_output

    for name, scalar in (
        ("logit_scale", softmax_logit_scale),
        ("history_bias", softmax_history_bias),
        ("history_c", softmax_history_c),
    ):
        if scalar.numel() != num_value_heads:
            raise ValueError(f"Clover {name} must have one entry per value head")
    logit_scale = softmax_logit_scale.reshape(-1).float().to(query_float.device)
    history_bias = softmax_history_bias.reshape(-1).float().to(query_float.device)
    history_c = softmax_history_c.reshape(-1).float().to(query_float.device)
    # [B, H, L(query), S(key)] logits over unscaled l2-normalized q/k.
    logits = torch.einsum("blhk,bshk->bhls", query_float, key_float)
    logits = logits * logit_scale.view(1, -1, 1, 1)
    logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
    history_logit = history_bias.view(1, -1, 1, 1).expand(batch_size, -1, block_size, 1)
    valid_block_size = valid.sum(dim=-1).clamp_min(1).float()
    block_size_log_ratio = torch.log(valid_block_size / 32.0)
    history_logit = history_logit + (
        history_c.view(1, -1, 1, 1) * block_size_log_ratio.view(batch_size, 1, 1, 1)
    )
    weights = torch.softmax(torch.cat((logits, history_logit), dim=-1), dim=-1)
    local_output = torch.einsum(
        "bhls,bshv->blhv",
        weights[..., :-1],
        value_float * valid_float[..., None],
    )
    history_weight = weights[..., -1].permute(0, 2, 1)
    output = history_weight[..., None] * history + local_output
    output = output * valid_float[..., None]
    return output.to(query.dtype).reshape(
        1, num_tokens, num_value_heads, value.shape[-1]
    )


@dataclass
class CloverCache:
    layers: list
    attention_mask: torch.Tensor

    def get_seq_length(self, layer_idx=0):
        return self.attention_mask.shape[-1]


class CloverRMSNorm(nn.Module):
    def __init__(self, size, eps, zero_centered=True):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(size) if zero_centered else torch.ones(size)
        )
        self.eps, self.zero_centered = eps, zero_centered

    def forward(self, x, gate=None):
        y = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        weight = 1 + self.weight.float() if self.zero_centered else self.weight.float()
        y = y * weight
        if gate is not None:
            y = y * F.silu(gate.float())
        return y.to(x.dtype)


def recurrent_commit_torch(q, k, v, g, beta, state, valid):
    h = v.shape[2]
    q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)
    k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    q = q.repeat_interleave(h // q.shape[2], 2) / q.shape[-1] ** 0.5
    k = k.repeat_interleave(h // k.shape[2], 2)
    state = state.float()
    out = []
    for t in range(v.shape[1]):
        decayed = state * g[:, t].float().exp()[..., None, None]
        remembered = torch.einsum("bhvk,bhk->bhv", decayed, k[:, t])
        delta = (v[:, t].float() - remembered) * beta[:, t].float()[..., None]
        updated = decayed + delta[..., None] * k[:, t, :, None, :]
        state = torch.where(valid[:, t, None, None, None], updated, state)
        y = torch.einsum("bhvk,bhk->bhv", state, q[:, t])
        out.append(y * valid[:, t, None, None])
    return torch.stack(out, dim=1).to(v.dtype), state


class CloverGatedDeltaNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.kh, self.vh = c.linear_num_key_heads, c.linear_num_value_heads
        self.kd, self.vd = c.linear_key_head_dim, c.linear_value_head_dim
        self.channels = 2 * self.kh * self.kd + self.vh * self.vd
        self.conv1d = nn.Conv1d(
            self.channels,
            self.channels,
            c.linear_conv_kernel_dim,
            groups=self.channels,
            bias=False,
        )
        self.conv_right_weight = nn.Parameter(
            torch.zeros(self.channels, 1, c.linear_conv_kernel_dim - 1)
        )
        self.in_proj_qkv = nn.Linear(c.hidden_size, self.channels, bias=False)
        self.in_proj_z = nn.Linear(c.hidden_size, self.vh * self.vd, bias=False)
        self.in_proj_a = nn.Linear(c.hidden_size, self.vh, bias=False)
        self.in_proj_b = nn.Linear(c.hidden_size, self.vh, bias=False)
        self.A_log = nn.Parameter(torch.zeros(self.vh))
        self.dt_bias = nn.Parameter(torch.ones(self.vh))
        self.bd_softmax_logit_scale = nn.Parameter(torch.full((self.vh,), 10.0))
        self.bd_softmax_history_bias = nn.Parameter(torch.full((self.vh,), 5.0))
        self.bd_softmax_history_c = nn.Parameter(torch.full((self.vh,), 1.0))
        self.norm = CloverRMSNorm(self.vd, c.rms_norm_eps, zero_centered=False)
        self.out_proj = nn.Linear(self.vh * self.vd, c.hidden_size, bias=False)

    def forward(self, x, cache, valid, preview):
        b, t, _ = x.shape
        x = x * valid[..., None]
        raw = self.in_proj_qkv(x)
        width = self.config.linear_conv_kernel_dim
        if cache is None:
            conv_state = raw.new_zeros((b, self.channels, width - 1))
            ssm_state = torch.zeros(
                (b, self.vh, self.vd, self.kd), device=x.device, dtype=torch.float32
            )
        else:
            conv_state, ssm_state = cache["conv"], cache["ssm"]
        if preview:
            flat = raw.reshape(-1, self.channels).contiguous()
            kw = dict(
                clean_conv_states=conv_state,
                cache_indices=torch.arange(b, device=x.device),
                conv_weights=self.conv1d.weight,
                block_size=t,
                active_valid_mask=valid.reshape(-1),
                conv_right_weight=self.conv_right_weight,
            )
            mixed = bd_gdn_conv1d(flat, **kw).view(b, t, self.channels)
            next_conv = conv_state
        else:
            if bool(valid.all()):
                joined = torch.cat((conv_state, raw.transpose(1, 2)), -1)
                mixed = causal_conv_silu(joined, self.conv1d.weight).transpose(1, 2)
                next_conv = (
                    joined[..., -(width - 1) :] if width > 1 else joined[..., :0]
                ).clone()
            else:
                history, pieces = conv_state, []
                for i in range(t):
                    window = torch.cat((history, raw[:, i, :, None]), -1)
                    y = F.silu((window * self.conv1d.weight[:, 0]).float().sum(-1)).to(
                        raw.dtype
                    )
                    pieces.append(y * valid[:, i, None])
                    history = torch.where(
                        valid[:, i, None, None], window[..., 1:], history
                    )
                mixed, next_conv = torch.stack(pieces, 1), history.clone()
        q, k, v = mixed.split(
            (self.kh * self.kd, self.kh * self.kd, self.vh * self.vd), -1
        )
        q, k, v = (
            q.reshape(b, t, self.kh, self.kd),
            k.reshape(b, t, self.kh, self.kd),
            v.reshape(b, t, self.vh, self.vd),
        )
        g = -self.A_log.float().exp() * F.softplus(
            self.in_proj_a(x).float() + self.dt_bias.float()
        )
        if preview:
            qf = q.reshape(1, b * t, self.kh, self.kd)
            kf = k.reshape(1, b * t, self.kh, self.kd)
            vf = v.reshape(1, b * t, self.vh, self.vd)
            kw = dict(
                clean_ssm_states=ssm_state,
                cache_indices=torch.arange(b, device=x.device),
                active_valid_mask=valid.reshape(-1),
                scale=self.kd**-0.5,
                softmax_logit_scale=self.bd_softmax_logit_scale,
                softmax_history_bias=self.bd_softmax_history_bias,
                softmax_history_c=self.bd_softmax_history_c,
            )
            y = bd_gdn_local_memory(
                qf,
                kf,
                vf,
                g.reshape(1, b * t, self.vh),
                **kw,
            ).view(b, t, self.vh, self.vd)
            next_ssm = ssm_state
        else:
            beta = self.in_proj_b(x).sigmoid()
            y, next_ssm = recurrent_commit_torch(q, k, v, g, beta, ssm_state, valid)
        # Match the serving reference: normalize and gate in fp32, then cast once.
        z = self.in_proj_z(x).reshape(b, t, self.vh, self.vd)
        y = self.norm(y, z)
        y = self.out_proj(y.reshape(b, t, -1))
        return y, {"conv": next_conv, "ssm": next_ssm}


class CloverAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        h, d, kv = c.num_attention_heads, c.head_dim, c.num_key_value_heads
        self.q_proj = nn.Linear(c.hidden_size, h * d * 2, bias=c.attention_bias)
        self.k_proj = nn.Linear(c.hidden_size, kv * d, bias=c.attention_bias)
        self.v_proj = nn.Linear(c.hidden_size, kv * d, bias=c.attention_bias)
        self.o_proj = nn.Linear(h * d, c.hidden_size, bias=c.attention_bias)
        self.q_norm = CloverRMSNorm(d, c.rms_norm_eps)
        self.k_norm = CloverRMSNorm(d, c.rms_norm_eps)

    def rotary(self, x, positions):
        rd = int(self.config.head_dim * self.config.partial_rotary_factor)
        inv = 1.0 / self.config.rope_parameters["rope_theta"] ** (
            torch.arange(0, rd, 2, device=x.device, dtype=torch.float32) / rd
        )
        phase = positions.float()[..., None] * inv
        cos, sin = phase.cos().to(x.dtype)[:, None], phase.sin().to(x.dtype)[:, None]
        first, second = x[..., :rd].chunk(2, -1)
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), -1
        )
        return torch.cat((rotated, x[..., rd:]), -1)

    def forward(self, x, cache, valid, prefix_valid, positions, preview):
        b, t, _ = x.shape
        c = self.config
        q, gate = (
            self.q_proj(x)
            .view(b, t, c.num_attention_heads, 2 * c.head_dim)
            .chunk(2, -1)
        )
        q = self.rotary(self.q_norm(q).transpose(1, 2), positions)
        k = self.rotary(
            self.k_norm(
                self.k_proj(x).view(b, t, c.num_key_value_heads, c.head_dim)
            ).transpose(1, 2),
            positions,
        )
        v = self.v_proj(x).view(b, t, c.num_key_value_heads, c.head_dim).transpose(1, 2)
        if cache is not None:
            k = torch.cat((cache["key"], k), 2)
            v = torch.cat((cache["value"], v), 2)
        next_state = {"key": k, "value": v}
        prefix = prefix_valid.shape[-1]
        key_valid = torch.cat((prefix_valid, valid), 1)
        allow = key_valid[:, None, None, :].expand(b, 1, t, prefix + t)
        if not preview:
            causal = (
                torch.arange(prefix + t, device=x.device)[None, :]
                <= prefix + torch.arange(t, device=x.device)[:, None]
            )
            allow = allow & causal[None, None]
        allow = allow & valid[:, None, :, None]
        k = k.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1)
        v = v.repeat_interleave(c.num_attention_heads // c.num_key_value_heads, 1)
        if c.attention_backend == "sdpa":
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allow, dropout_p=0.0)
        else:
            scores = (q.float() @ k.float().transpose(-1, -2)) * c.head_dim**-0.5
            scores = scores.masked_fill(~allow, -float("inf"))
            probs = torch.nan_to_num(scores.softmax(-1)).to(v.dtype)
            y = probs @ v
        y = y.transpose(1, 2).reshape(b, t, -1) * gate.reshape(b, t, -1).sigmoid()
        return self.o_proj(y), next_state


def coreset_torch(logits, valid, top_k, top_p):
    b, t, e = logits.shape
    ids = logits.float().argsort(dim=-1, descending=True, stable=True)[..., :top_k]
    scores = logits.float().softmax(-1).gather(-1, ids) * valid[..., None]
    totals = logits.new_zeros((b, e), dtype=torch.float32)
    totals.scatter_add_(1, ids.reshape(b, -1), scores.reshape(b, -1))
    probs = totals / totals.sum(-1, keepdim=True).clamp_min(1e-9)
    order = probs.argsort(dim=-1, descending=True, stable=True)
    sorted_probs = probs.gather(-1, order)
    keep = (sorted_probs.cumsum(-1) - sorted_probs) < top_p
    keep[:, 0] = True
    allowed = torch.zeros_like(keep).scatter(1, order, keep)
    return logits.masked_fill(valid[..., None] & ~allowed[:, None, :], -float("inf"))


def linear_fp32_output(x, weight):
    """Keep the down-projection accumulator in FP32 until route weighting."""
    if (
        x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and not torch.is_grad_enabled()
    ):
        try:
            return torch.mm(x, weight.T, out_dtype=torch.float32)
        except TypeError:
            # Older PyTorch versions do not expose CUDA's FP32 output option.
            pass
    with torch.autocast(device_type=x.device.type, enabled=False):
        return F.linear(x.float(), weight.float())


def causal_conv_silu(joined, weight):
    """Match serving's input-dtype products, FP32 sum and fused activation."""
    width = weight.shape[-1]
    length = joined.shape[-1] - width + 1
    total = torch.zeros(
        (*joined.shape[:-1], length), device=joined.device, dtype=torch.float32
    )
    for tap in range(width):
        total.add_(
            (joined[..., tap : tap + length] * weight[:, 0, tap][None, :, None]).float()
        )
    return F.silu(total).to(joined.dtype)


class CloverMLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_proj = nn.Linear(
            c.hidden_size, c.shared_expert_intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            c.hidden_size, c.shared_expert_intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            c.shared_expert_intermediate_size, c.hidden_size, bias=False
        )

    def forward(self, x):
        gate, up = self.gate_proj(x), self.up_proj(x)
        return self.down_proj((F.silu(gate.float()) * up.float()).to(x.dtype))


class CloverExperts(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(c.num_experts, 2 * c.moe_intermediate_size, c.hidden_size)
        )
        self.down_proj = nn.Parameter(
            torch.empty(c.num_experts, c.hidden_size, c.moe_intermediate_size)
        )

    def forward(self, x, ids, weights):
        out = torch.zeros_like(x, dtype=torch.float32)
        # Each routed contribution is rounded once, then summed in FP32.
        for expert in ids.unique().tolist():
            token, slot = torch.where(ids == expert)
            gate, up = F.linear(x[token], self.gate_up_proj[expert]).chunk(2, -1)
            hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
            y = linear_fp32_output(hidden, self.down_proj[expert])
            routed = (y * weights[token, slot, None].float()).to(x.dtype)
            out.index_add_(0, token, routed.float())
        return out.to(x.dtype)


class CloverMoeBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.gate = nn.Linear(c.hidden_size, c.num_experts, bias=False)
        self.experts = CloverExperts(c)
        if c.num_shared_experts:
            self.shared_expert = CloverMLP(c)
            self.shared_expert_gate = nn.Linear(c.hidden_size, 1, bias=False)

    def forward(self, x, valid, preview):
        c = self.config
        b, t, d = x.shape
        logits = self.gate(x)
        if preview and c.moe_block_coreset_enabled:
            logits = coreset_torch(
                logits, valid, c.num_experts_per_tok, c.moe_block_coreset_top_p
            )
        probs = logits.float().softmax(-1)
        ids = logits.float().argsort(dim=-1, descending=True, stable=True)[
            ..., : c.num_experts_per_tok
        ]
        weights = probs.gather(-1, ids)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        y = self.experts(
            x.reshape(-1, d),
            ids.reshape(-1, c.num_experts_per_tok),
            weights.reshape(-1, c.num_experts_per_tok),
        ).view(b, t, d)
        if c.num_shared_experts:
            y = y + self.shared_expert(x) * self.shared_expert_gate(x).sigmoid()
        return y


class CloverDecoderLayer(nn.Module):
    def __init__(self, c, i):
        super().__init__()
        self.preview_type = c.layer_types[i]
        if self.preview_type == "linear_attention":
            self.linear_attn = CloverGatedDeltaNet(c)
        else:
            self.self_attn = CloverAttention(c)
        self.input_layernorm = CloverRMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = CloverRMSNorm(c.hidden_size, c.rms_norm_eps)
        self.mlp = CloverMoeBlock(c)

    def forward(self, x, cache, valid, prefix_valid, positions, preview):
        hidden = self.input_layernorm(x)
        if self.preview_type == "linear_attention":
            hidden, next_cache = self.linear_attn(hidden, cache, valid, preview)
        else:
            hidden, next_cache = self.self_attn(
                hidden, cache, valid, prefix_valid, positions, preview
            )
        x = x + hidden
        return (
            x + self.mlp(self.post_attention_layernorm(x), valid, preview),
            next_cache,
        )


class CloverMoePreTrainedModel(PreTrainedModel):
    config_class = CloverMoeConfig
    base_model_prefix = "model"
    _no_split_modules = ["CloverDecoderLayer"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, CloverExperts):
            nn.init.normal_(module.gate_up_proj, std=self.config.initializer_range)
            nn.init.normal_(module.down_proj, std=self.config.initializer_range)
        elif isinstance(module, CloverRMSNorm):
            nn.init.constant_(module.weight, 0.0 if module.zero_centered else 1.0)
        elif isinstance(module, CloverGatedDeltaNet):
            nn.init.zeros_(module.A_log)
            nn.init.ones_(module.dt_bias)
            if hasattr(module, "conv_right_weight"):
                nn.init.zeros_(module.conv_right_weight)
            nn.init.constant_(module.bd_softmax_logit_scale, 10.0)
            nn.init.constant_(module.bd_softmax_history_bias, 5.0)
            nn.init.constant_(module.bd_softmax_history_c, 1.0)


class CloverMoeModel(CloverMoePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [CloverDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = CloverRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        return_dict=True,
        preview=False,
        **kwargs,
    ):
        if kwargs:
            raise TypeError(f"Unsupported Clover forward arguments: {sorted(kwargs)}")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        b, t, _ = x.shape
        if t == 0:
            raise ValueError("Empty input is not supported")
        if past_key_values is not None and not isinstance(past_key_values, CloverCache):
            raise TypeError("past_key_values must be CloverCache")
        prefix_valid = (
            past_key_values.attention_mask
            if past_key_values is not None
            else torch.zeros((b, 0), device=x.device, dtype=torch.bool)
        )
        if prefix_valid.shape[0] != b:
            raise ValueError("Cache batch size mismatch")
        valid = torch.ones((b, t), device=x.device, dtype=torch.bool)
        if attention_mask is not None:
            if attention_mask.shape not in ((b, t), (b, prefix_valid.shape[-1] + t)):
                raise ValueError(
                    "attention_mask must describe current tokens or the complete cached sequence"
                )
            if attention_mask.shape[-1] != t and not torch.equal(
                attention_mask[:, :-t].bool(), prefix_valid
            ):
                raise ValueError("attention_mask changes the committed prefix")
            valid = attention_mask[:, -t:].to(device=x.device, dtype=torch.bool)
        if preview and (t > self.config.block_size or self.config.block_size % t):
            raise ValueError("Preview block size must divide configured block_size")
        if position_ids is None:
            positions = prefix_valid.sum(-1, keepdim=True) + valid.long().cumsum(-1) - 1
            positions = positions.masked_fill(~valid, 0).clamp_min(0)
        else:
            if position_ids.shape != (b, t):
                raise ValueError("position_ids must have shape [batch, tokens]")
            positions = position_ids
        if torch.any((positions >= self.config.max_position_embeddings) & valid):
            raise ValueError("Sequence exceeds max_position_embeddings")
        next_layers = []
        for i, layer in enumerate(self.layers):
            state = past_key_values.layers[i] if past_key_values is not None else None
            x, state = layer(x, state, valid, prefix_valid, positions, preview)
            next_layers.append(state)
        use_cache = (
            getattr(self.config, "use_cache", True) if use_cache is None else use_cache
        )
        cache = None
        if use_cache:
            cache = (
                past_key_values
                if preview
                else CloverCache(next_layers, torch.cat((prefix_valid, valid), 1))
            )
        output = BaseModelOutputWithPast(
            last_hidden_state=self.norm(x), past_key_values=cache
        )
        return output if return_dict else output.to_tuple()


class CloverMoeModelLM(CloverMoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = CloverMoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        return_dict=True,
        preview=False,
        labels=None,
        **kwargs,
    ):
        if labels is not None:
            raise NotImplementedError(
                "This release implements inference, not a training loss"
            )
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            return_dict=True,
            preview=preview,
            **kwargs,
        )
        output = CausalLMOutputWithPast(
            logits=self.lm_head(out.last_hidden_state),
            past_key_values=out.past_key_values,
        )
        return output if return_dict else output.to_tuple()

    @torch.inference_mode()
    def generate(
        self, inputs=None, generation_config=None, attention_mask=None, **kwargs
    ):
        """Block LowConfidence decoding. Returns prompt+completion, like HF generate.

        Supported: max_new_tokens/max_length, greedy or temperature/top-k/top-p
        sampling, EOS/padding, return_dict_in_generate, output_scores, streamer,
        and stopping_criteria. Response penalties and length/thinking controls follow
        SGLang LowConfidence semantics. Custom overrides: block_size, threshold.
        Beam search and speculative decoding are not exposed.
        Batched requests run sequentially to keep independent clean-state caches.
        """
        input_ids = kwargs.pop("input_ids", inputs)
        if input_ids is None or input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, tokens]")
        gen = deepcopy(
            generation_config or self.generation_config or GenerationConfig()
        )
        custom = deepcopy(self.config.denoising_config)
        for key in ("block_size", "threshold"):
            if key in kwargs:
                custom[key] = kwargs.pop(key)
        block_size = custom.get("block_size", self.config.block_size)
        threshold = custom["threshold"]
        if block_size <= 0 or self.config.block_size % block_size:
            raise ValueError(
                "block_size must be positive and divide the configured maximum block size"
            )
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        streamer = kwargs.pop("streamer", None)
        stopping = kwargs.pop("stopping_criteria", None)
        allowed = {
            "max_new_tokens",
            "max_length",
            "do_sample",
            "temperature",
            "top_k",
            "top_p",
            "eos_token_id",
            "pad_token_id",
            "return_dict_in_generate",
            "output_scores",
            "num_beams",
            "num_return_sequences",
            "use_cache",
        }
        allowed.update(PENALTY_DEFAULTS)
        for key in list(kwargs):
            if key not in allowed:
                raise TypeError(f"Unsupported generation argument: {key}")
            setattr(gen, key, kwargs.pop(key))
        defaults = GenerationConfig().to_dict()
        for key, value in gen.to_dict().items():
            if (
                key not in allowed
                and not key.startswith("_")
                and key not in ("transformers_version", "bos_token_id")
            ):
                if value is None or (
                    value is False and defaults.get(key) in (None, False)
                ):
                    continue
                if value != defaults.get(key):
                    raise NotImplementedError(
                        f"Unsupported nondefault generation setting: {key}"
                    )
        validate_penalties(gen, self.config.vocab_size)
        if (gen.num_beams or 1) != 1 or (gen.num_return_sequences or 1) != 1:
            raise NotImplementedError(
                "Clover block generation supports one sequence per prompt, without beams"
            )
        if gen.temperature is None:
            gen.temperature = 1.0
        if gen.use_cache is False:
            raise NotImplementedError("Clover generation requires clean-state caching")
        if gen.do_sample and gen.temperature <= 0:
            raise ValueError("Sampling requires temperature > 0")
        if gen.top_p is not None and not 0 < gen.top_p <= 1:
            raise ValueError("top_p must be in (0,1]")
        if input_ids.shape[0] > 1 and (streamer is not None or gen.output_scores):
            raise ValueError(
                "Streaming and output_scores currently require batch size 1"
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask shape mismatch")
        eos = (
            gen.eos_token_id
            if gen.eos_token_id is not None
            else getattr(self.config, "eos_token_id", None)
        )
        eos = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
        eos = sorted(set(eos) | set(gen.stop_token_ids))
        pad = gen.pad_token_id
        if pad is None:
            pad = getattr(self.config, "pad_token_id", None)
        if pad is None:
            pad = eos[0] if eos else 0
        if gen.output_scores and gen.do_sample:
            raise ValueError("output_scores is available for greedy generation only")
        results = []
        all_scores = []
        for row in range(input_ids.shape[0]):
            prompt = input_ids[row][attention_mask[row].bool()].unsqueeze(0)
            if prompt.shape[1] == 0:
                raise ValueError("Each prompt must have at least one token")
            # Prompt IDs are committed as literal context, even when an ID
            # also serves as the mask placeholder in newly generated blocks.
            maximum = gen.max_new_tokens
            if maximum is None:
                maximum = max(0, (gen.max_length or 20) - prompt.shape[1])
            if maximum < 0:
                raise ValueError("max_new_tokens must be nonnegative")
            # Preview always resolves complete blocks, including the final one
            # when the requested output length ends partway through a block.
            preview_length = ((maximum + block_size - 1) // block_size) * block_size
            if prompt.shape[1] + preview_length > self.config.max_position_embeddings:
                raise ValueError(
                    "Prompt plus full preview blocks exceeds context length: "
                    f"{prompt.shape[1]} + {preview_length} > "
                    f"{self.config.max_position_embeddings} tokens"
                )
            completed = prompt
            if streamer is not None:
                streamer.put(prompt.cpu())
            cache = None
            if maximum:
                # Prefill needs states only; avoid allocating prompt-wide logits.
                cache = self.model(prompt, use_cache=True).past_key_values
            response = []
            penalties = CloverPenaltyState(
                gen, self.config.vocab_size, prompt.device, eos + [pad]
            )
            stopped = False
            while len(response) < maximum and not stopped:
                remaining = min(block_size, maximum - len(response))
                block = torch.full(
                    (1, block_size),
                    self.config.mask_token_id,
                    device=prompt.device,
                    dtype=prompt.dtype,
                )
                valid = torch.ones_like(block, dtype=torch.bool)
                pending = valid.clone()
                token_scores = [None] * block_size
                for _ in range(block_size):
                    out = self(
                        block,
                        attention_mask=valid,
                        past_key_values=cache,
                        use_cache=True,
                        preview=True,
                    )
                    logits = out.logits[0].float()
                    logits[:, self.config.mask_token_id] = -float("inf")
                    penalties.apply(logits, block, self.config.mask_token_id)
                    prediction, confidence = predict_with_confidence(logits, gen)
                    confidence = confidence.masked_fill(~pending[0], -float("inf"))
                    transfer = (confidence > threshold) & pending[0]
                    if not bool(transfer.any()):
                        transfer[confidence.argmax()] = True
                    if gen.output_scores:
                        for pos in transfer.nonzero().flatten().tolist():
                            token_scores[pos] = logits[pos : pos + 1].detach().clone()
                    block[0, transfer] = prediction[transfer]
                    pending[0, transfer] = False
                    if not bool(pending.any()):
                        break
                if bool(pending.any()):
                    raise RuntimeError(
                        "Denoising step limit reached with unresolved masks"
                    )
                # Only the returned continuation is limited by max_new_tokens;
                # every position above has participated in block denoising.
                new = block[0, :remaining]
                # Stop only after full block resolution: unfilled earlier positions
                # may precede an EOS selected out of positional order.
                for i, token in enumerate(new.tolist()):
                    response.append(token)
                    completed = torch.cat((completed, new[i : i + 1].view(1, 1)), 1)
                    if gen.output_scores:
                        all_scores.append(token_scores[i])
                    if streamer is not None:
                        streamer.put(new[i : i + 1].cpu())
                    if token in eos or (
                        stopping is not None
                        and bool(
                            torch.as_tensor(
                                stopping(
                                    completed,
                                    tuple(all_scores) if gen.output_scores else None,
                                )
                            ).all()
                        )
                    ):
                        stopped = True
                        break
                penalties.commit(new[: i + 1])
                if len(response) < maximum and not stopped:
                    # Causal replay commits only the resolved block. Preview state
                    # and any masked-token activations are never persisted.
                    cache = self.model(
                        new.view(1, -1), past_key_values=cache, use_cache=True
                    ).past_key_values
            if streamer is not None:
                streamer.end()
            results.append(torch.cat((input_ids[row], input_ids.new_tensor(response))))
        length = max(r.numel() for r in results)
        sequences = input_ids.new_full((len(results), length), pad)
        for i, result in enumerate(results):
            sequences[i, : result.numel()] = result
        if gen.return_dict_in_generate:
            return GenerateDecoderOnlyOutput(
                sequences=sequences,
                scores=tuple(all_scores) if gen.output_scores else None,
            )
        return sequences

    def diffusion_generate(self, *args, **kwargs):
        return self.generate(*args, **kwargs)
