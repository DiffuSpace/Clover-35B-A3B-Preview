"""CUDA grouped down projection against an independent FP64 matmul oracle."""

from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from clover_sft.grouped_mm import _ExpertDownProjection
from clover_sft.modeling_clover_moe import CloverExperts
from clover_sft.moe import expert_forward

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


def reference_down(inputs, weight, counts):
    output = []
    start = 0
    for expert, count in enumerate(counts):
        output.append(inputs[start:start + count].double() @ weight[expert].double().T)
        start += count
    return torch.cat(output).float()


@pytest.mark.parametrize('counts', [(2, 0, 7, 0), (0, 1, 8, 0), (0, 0, 0, 0)])
@pytest.mark.parametrize('train_inputs,train_weights', [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize('features', [(16, 32), (512, 2048)])
def test_grouped_down_outputs_and_gradients(counts, train_inputs, train_weights, features):
    torch.manual_seed(47)
    inner, output = features
    x = torch.randn(sum(counts), inner, device='cuda', dtype=torch.bfloat16).requires_grad_(train_inputs)
    weight = torch.randn(len(counts), output, inner, device='cuda', dtype=torch.bfloat16).requires_grad_(train_weights)
    other_x = x.detach().clone().requires_grad_(train_inputs)
    other_w = weight.detach().clone().requires_grad_(train_weights)
    offsets = torch.tensor(counts, device='cuda', dtype=torch.int32).cumsum(0, dtype=torch.int32)
    actual = _ExpertDownProjection.apply(x, weight, offsets)
    expected = reference_down(other_x, other_w, counts)
    assert actual.dtype == torch.float32
    # Different FP32 reduction orders differ most at near-cancelling outputs.
    # Check both individual errors and aggregate accuracy against FP64; an
    # early BF16 output cast has orders of magnitude more relative error.
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)
    if actual.numel():
        relative_error = (actual - expected).norm() / expected.norm().clamp_min(1e-20)
        assert relative_error < 1e-6
    dy = torch.randn_like(actual) * 0.137
    actual.backward(dy)
    expected.backward(dy)
    for parameter, other in ((x, other_x), (weight, other_w)):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-5, rtol=8e-3)
        else:
            assert parameter.grad is None


def test_grouped_backward_preserves_cancelling_fp32_upstream_gradients():
    x = torch.ones(2, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weight = torch.ones(2, 32, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    offsets = torch.tensor([2, 2], device='cuda', dtype=torch.int32)
    output = _ExpertDownProjection.apply(x, weight, offsets)
    dy = torch.tensor([1.001, -1.0001], device='cuda').view(2, 1).expand(2, 32)
    output.backward(dy)
    # Both rows round to +/-1 in BF16, falsely yielding zero dW if the
    # upstream gradient is cast once before the grouped weight-gradient GEMM.
    expected = dy.T @ x.detach().float()
    assert expected.abs().min() > 0
    torch.testing.assert_close(weight.grad[0], expected.to(torch.bfloat16), atol=1e-7, rtol=8e-3)
    assert weight.grad[1].count_nonzero() == 0


def test_grouped_moe_preserves_fp32_routing_weight_gradients(tiny_config):
    torch.manual_seed(53)
    module = CloverExperts(tiny_config).cuda().to(torch.bfloat16)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(std=0.1)
    reference = deepcopy(module)
    x = torch.randn(3, 32, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    other = x.detach().clone().requires_grad_()
    ids = torch.tensor([[0, 2], [2, 0], [0, 2]], device='cuda')
    scores = torch.tensor([[0.31, 0.69], [0.57, 0.43], [0.83, 0.17]], device='cuda', requires_grad=True)
    other_scores = scores.detach().clone().requires_grad_()
    actual = expert_forward(module, x, ids, scores)
    contributions = []
    # CPU-known dispatch, without sorting, offsets or grouped GEMM. Batch each
    # expert's tokens so dW accumulates in FP32 before its single BF16 cast;
    # separate token-wise calls would instead accumulate rounded BF16 dWs.
    for expert, slots in ((0, [0, 1, 0]), (2, [1, 0, 1])):
        gate, up = F.linear(other, reference.gate_up_proj[expert]).chunk(2, dim=-1)
        activated = (F.silu(gate.float()) * up.float()).to(other.dtype)
        projected = F.linear(activated.float(), reference.down_proj[expert].float())
        selected_scores = other_scores[torch.arange(3, device='cuda'), torch.tensor(slots, device='cuda')]
        contributions.append((projected * selected_scores[:, None]).to(other.dtype).float())
    expected = sum(contributions).to(other.dtype)
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=8e-3)
    dy = torch.randn_like(actual)
    actual.backward(dy)
    expected.backward(dy)
    torch.testing.assert_close(scores.grad, other_scores.grad, atol=2e-5, rtol=1e-4)
    torch.testing.assert_close(x.grad, other.grad, atol=2e-4, rtol=8e-3)
    for parameter, other_parameter in zip(module.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, other_parameter.grad, atol=2e-4, rtol=8e-3)


def test_grouped_down_has_no_per_expert_matmuls_or_host_scalar_reads():
    from torch.profiler import ProfilerActivity, profile

    x = torch.randn(9, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(4, 32, 16, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    offsets = torch.tensor([2, 2, 9, 9], device='cuda', dtype=torch.int32)
    dy = torch.randn(9, 32, device='cuda')

    def run():
        output = _ExpertDownProjection.apply(x, weight, offsets)
        torch.autograd.grad(output, (x, weight), dy)

    run()  # exclude first-use compilation/cache preparation
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU]) as captured:
        run()
    torch.cuda.synchronize()
    calls = {event.key: event.count for event in captured.key_averages()}
    assert calls.get('quack::gemm_out') == 7, calls  # forward + 3 dX + 3 dW
    for name in ('aten::mm', 'aten::item', 'aten::_local_scalar_dense', 'cudaMemcpyAsync', 'cudaStreamSynchronize'):
        assert calls.get(name, 0) == 0, (name, calls)
