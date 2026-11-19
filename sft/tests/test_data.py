import pytest
import torch

from clover_sft.attention import TrainingLayout, make_mask_mod
from clover_sft.data import IGNORE_INDEX, Role, align_sample, collate_samples


def test_roles_distinguish_real_eos_from_same_id_padding():
    sample = align_sample([10, 11, 12], [20, 21, 0], block_size=4, pad_id=0, length=12)
    assert sample["input_ids"] == [10, 11, 12, 0, 20, 21, 0, 0, 0, 0, 0, 0]
    assert sample["token_roles"] == [1, 1, 1, 2, 3, 3, 3, 4, 0, 0, 0, 0]
    assert sample["labels"] == [-100, -100, -100, -100, 20, 21, 0, 0, -100, -100, -100, -100]
    assert sample["position_ids"] == [0, 1, 2, 2, 3, 4, 5, 6, 0, 0, 0, 0]


def test_default_length_is_exactly_4096():
    sample = align_sample([10, 11, 12], [20, 2], block_size=32, pad_id=0)
    assert {len(v) for v in sample.values()} == {4096}
    assert sample["token_roles"].count(Role.PRE_PAD) == 29
    assert sample["token_roles"].count(Role.POST_PAD) == 30


def test_limit_includes_alignment_and_never_silently_truncates_eos():
    with pytest.raises(ValueError, match="exceeds fixed length"):
        align_sample([1] * 5, [2] * 5, block_size=4, pad_id=0, length=12)


def test_noise_supervises_post_pad_but_never_pre_or_batch_pad():
    sample = align_sample([10, 11, 12], [20, 21, 2], block_size=4, pad_id=0, length=16)
    batch = collate_samples([sample], mask_id=63, generator=torch.Generator().manual_seed(0))
    selected = batch["labels"] != IGNORE_INDEX
    assert selected[0, 7]  # sampled POST_PAD is a real PAD prediction target
    assert not selected[0, :4].any() and not selected[0, 8:].any()
    assert torch.equal(batch["noisy_input_ids"][~selected], batch["input_ids"][~selected])
    assert (batch["noisy_input_ids"][selected] == 63).all()


def test_attention_role_rules_have_an_independent_dense_oracle():
    sample = align_sample([10, 11, 12], [20, 21, 22, 23, 2], block_size=4, pad_id=0, length=16)
    roles = torch.tensor([sample["token_roles"]])
    layout = TrainingLayout(roles, torch.tensor([sample["position_ids"]]), 4)
    mask = make_mask_mod(layout)
    for q in range(32):
        for k in range(32):
            q_clean, k_clean = q >= 16, k >= 16
            qi, ki = q % 16, k % 16
            rq, rk = sample["token_roles"][qi], sample["token_roles"][ki]
            q_allowed = rq in ([Role.PROMPT, Role.RESPONSE] if q_clean
                               else [Role.RESPONSE, Role.POST_PAD])
            k_allowed = rk in ([Role.PROMPT, Role.RESPONSE] if k_clean
                               else [Role.RESPONSE, Role.POST_PAD])
            relation = ((not q_clean and not k_clean and qi // 4 == ki // 4)
                        or (not q_clean and k_clean and ki // 4 < qi // 4)
                        or (q_clean and k_clean and ki <= qi))
            expected = q == k or (q_allowed and k_allowed and relation)
            assert bool(mask(0, 0, torch.tensor(q), torch.tensor(k))) == expected, (q, k)


def test_post_pad_visibility_controls_actual_attention_gradients(tiny_config):
    from clover_sft.attention import attention_forward
    from clover_sft.modeling_clover_moe import CloverAttention

    torch.manual_seed(67)
    sample = align_sample([10, 11, 12], [20, 2], block_size=4, pad_id=0, length=12)
    layout = TrainingLayout(torch.tensor([sample['token_roles']]), torch.tensor([sample['position_ids']]), 4)
    module = CloverAttention(tiny_config)
    hidden = torch.randn(1, 24, tiny_config.hidden_size, requires_grad=True)
    output = attention_forward(module, hidden, layout)
    gradient, = torch.autograd.grad(output[:, 7].sum(), hidden, retain_graph=True)
    assert gradient[:, 6].abs().sum() > 0  # post-pad tokens attend each other
    gradient, = torch.autograd.grad(output[:, 6].sum(), hidden)
    assert gradient[:, 7].abs().sum() > 0  # later PAD key remains visible
    assert not layout.physical_valid[0, :4].any()  # noisy prompt / PRE_PAD are inactive


def test_post_pad_visibility_is_block_local():
    # Synthetic two-turn roles exercise the mask contract; the public data
    # template itself continues to accept one user/assistant exchange only.
    roles = torch.tensor([[1, 1, 1, 2, 3, 3, 4, 4, 1, 1, 2, 2, 3, 3, 4, 4]])
    layout = TrainingLayout(roles, torch.arange(16)[None], 4)
    mask = make_mask_mod(layout)
    index = torch.arange(32)
    allowed = mask(0, 0, index[:, None], index[None, :])
    assert allowed[4:8, 4:8].all()  # response and post-pad all attend each other
    assert allowed[12:16, 12:16].all()
    assert not allowed[12:16, 6:8].any()  # no prior noisy-block post-pad
    assert not allowed[12:16, 22:24].any()  # no prior clean-history post-pad
    assert allowed[12:16, 20:22].all()  # earlier real clean response remains visible
    assert allowed[3].sum() == 1 and allowed[:, 3].sum() == 1  # PRE_PAD isolation
