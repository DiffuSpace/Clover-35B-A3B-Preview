"""Standard PyTorch mask construction for one independently padded sample per row."""

from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from .data import Role


@dataclass
class TrainingLayout:
    roles: torch.Tensor
    positions: torch.Tensor
    block_size: int
    block_mask: object = None
    gdn_backend: str = "fla"

    @property
    def length(self):
        return self.roles.shape[1]

    @property
    def clean_valid(self):
        return (self.roles == Role.PROMPT) | (self.roles == Role.RESPONSE)

    @property
    def noisy_valid(self):
        return (self.roles == Role.RESPONSE) | (self.roles == Role.POST_PAD)

    @property
    def physical_valid(self):
        return torch.cat((self.noisy_valid, self.clean_valid), dim=1)


def make_mask_mod(layout):
    length, block = layout.length, layout.block_size
    roles = layout.roles

    def mask_mod(b, h, q_idx, kv_idx):
        # Rounded Flex tiles can probe outside the sequence.
        q, k = q_idx % length, kv_idx % length
        clean_q, clean_k = q_idx >= length, kv_idx >= length
        role_q, role_k = roles[b, q], roles[b, k]
        clean_role_q = (role_q == Role.PROMPT) | (role_q == Role.RESPONSE)
        clean_role_k = (role_k == Role.PROMPT) | (role_k == Role.RESPONSE)
        noisy_role_q = (role_q == Role.RESPONSE) | (role_q == Role.POST_PAD)
        noisy_role_k = (role_k == Role.RESPONSE) | (role_k == Role.POST_PAD)
        valid_q = torch.where(clean_q, clean_role_q, noisy_role_q)
        valid_k = torch.where(clean_k, clean_role_k, noisy_role_k)
        allowed = (
            (~clean_q & ~clean_k & (q // block == k // block))
            | (~clean_q & clean_k & (k // block < q // block))
            | (clean_q & clean_k & (k <= q))
        )
        # Every active token, including post-pad, participates bidirectionally
        # inside its noisy block. Clean keys exclude all alignment padding.
        allowed = allowed & valid_q & valid_k
        # Isolated rows keep a self edge, preventing all-masked softmax rows.
        return ((allowed | (q_idx == kv_idx)) & (q_idx < 2 * length) & (kv_idx < 2 * length))

    return mask_mod


def build_block_mask(layout):
    return create_block_mask(
        make_mask_mod(layout), B=layout.roles.shape[0], H=None,
        Q_LEN=2 * layout.length, KV_LEN=2 * layout.length, device=layout.roles.device,
    )


compiled_flex_attention = torch.compile(flex_attention, fullgraph=True)


def attention_forward(module, hidden, layout):
    batch, length, _ = hidden.shape
    config = module.config
    q, gate = module.q_proj(hidden).view(batch, length, config.num_attention_heads, 2 * config.head_dim).chunk(2, -1)
    positions = torch.cat((layout.positions, layout.positions), dim=1)
    q = module.rotary(module.q_norm(q).transpose(1, 2), positions)
    k = module.k_proj(hidden).view(batch, length, config.num_key_value_heads, config.head_dim)
    k = module.rotary(module.k_norm(k).transpose(1, 2), positions)
    v = module.v_proj(hidden).view(batch, length, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    if hidden.is_cuda:
        if layout.block_mask is None:
            layout.block_mask = build_block_mask(layout)
        output = compiled_flex_attention(q, k, v, block_mask=layout.block_mask, enable_gqa=True)
    else:
        # Small CPU oracle only; production CUDA always uses FlexAttention.
        idx = torch.arange(length, device=hidden.device)
        masks = [make_mask_mod(layout)(b, 0, idx[:, None], idx[None, :]) for b in range(batch)]
        output = torch.nn.functional.scaled_dot_product_attention(
            q, k.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, 1),
            v.repeat_interleave(config.num_attention_heads // config.num_key_value_heads, 1),
            attn_mask=torch.stack(masks)[:, None], dropout_p=0.0,
        )
    output = output.transpose(1, 2).reshape(batch, length, -1)
    return module.o_proj(output * gate.reshape(batch, length, -1).sigmoid())
