from copy import deepcopy

import pytest
import torch

from clover_sft.attention import TrainingLayout
from clover_sft.gdn import gdn_forward
from clover_sft.model import CloverForSFT
from clover_sft.modeling_clover_moe import CloverGatedDeltaNet

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_fla_read_only_queries_match_recurrent_reference_and_gradients(tiny_config, tiny_batch):
    torch.manual_seed(19)
    # Compare backend gradients in FP32 to isolate scan math from BF16
    # rounding boundaries in the surrounding convolution/norm/projections.
    # The fixed-4K optimizer test below exercises the complete BF16 path.
    reference = CloverGatedDeltaNet(tiny_config).cuda()
    fast = deepcopy(reference)
    x = torch.randn(1, 32, tiny_config.hidden_size, device="cuda", requires_grad=True)
    y = x.detach().clone().requires_grad_()
    role = tiny_batch["token_roles"].cuda()
    pos = tiny_batch["position_ids"].cuda()
    expected = gdn_forward(reference, x, TrainingLayout(role, pos, 4, gdn_backend="torch"))
    actual = gdn_forward(fast, y, TrainingLayout(role, pos, 4, gdn_backend="fla"))
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=3e-2)
    weight = torch.randn_like(actual)
    (actual * weight).sum().backward()
    (expected * weight).sum().backward()
    torch.testing.assert_close(y.grad, x.grad, atol=2e-2, rtol=0.15)
    for (name, parameter), (_, other) in zip(fast.named_parameters(), reference.named_parameters()):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-2, rtol=0.15, msg=name)


def test_fixed_4k_cuda_model_update(tiny_config):
    from clover_sft.data import align_sample, collate_samples

    torch.manual_seed(29)
    model = CloverForSFT(tiny_config).cuda().to(torch.bfloat16).train()
    model.gradient_checkpointing_enable()
    row = align_sample([10, 11, 12], [20, 21, 22, 23, 2], block_size=4, pad_id=0)
    data = collate_samples([row], mask_id=63, generator=torch.Generator().manual_seed(0))
    data = {key: value.cuda() for key, value in data.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.model.layers[0].linear_attn.conv_right_weight.detach().clone()
    loss = model(**data).loss
    assert loss.isfinite()
    loss.backward()
    assert model.model.layers[0].linear_attn.conv_right_weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.model.layers[0].linear_attn.conv_right_weight)
