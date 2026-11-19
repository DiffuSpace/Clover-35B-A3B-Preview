import pytest
import torch

from clover_sft.configuration_clover_moe import CloverMoeConfig
from clover_sft.data import align_sample, collate_samples


@pytest.fixture
def tiny_config():
    return CloverMoeConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, block_size=4,
        mask_token_id=63, eos_token_id=2, bos_token_id=2, pad_token_id=0,
        router_aux_loss_coef=0,  # CE/inference parity cases disable regularization explicitly.
    )


@pytest.fixture
def tiny_batch():
    sample = align_sample([10, 11, 12], [20, 21, 22, 23, 2], block_size=4, pad_id=0, length=16)
    return collate_samples([sample], mask_id=63, generator=torch.Generator().manual_seed(0))
