import torch

from clover_sft.moe import _ExpertDownProjection, load_balancing_loss


def test_aux_loss_matches_independent_topk_formula_and_gradients():
    logits = torch.tensor([[2., 1., -1., 0.], [2., 1., 3., 0.], [9., 0., 0., 0.]], requires_grad=True)
    reference = logits.detach().clone().requires_grad_()
    ids = torch.tensor([[0, 1], [2, 0], [0, 1]])
    valid = torch.tensor([True, True, False])
    actual = load_balancing_loss(logits.softmax(-1), ids, valid)
    # Independent expert-wise oracle: four actual assignments and two tokens.
    expected = 0
    for expert in range(4):
        frequency = sum(int(expert in row) for row in [[0, 1], [2, 0]]) / 4
        expected = expected + 4 * frequency * reference[:2].softmax(-1)[:, expert].mean()
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference.grad)
    assert torch.count_nonzero(logits.grad[2]) == 0


def test_aux_loss_retains_all_experts_and_counts_only_nonzero_routes():
    # A one-expert coreset pads top-k=2 with zero-probability assignments.
    # Collapsing all active tokens to one of E=4 experts must cost E, not 1 or 2.
    probabilities = torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]])
    ids = torch.tensor([[0, 1], [0, 2]])
    actual = load_balancing_loss(probabilities, ids, torch.tensor([True, True]))
    torch.testing.assert_close(actual, torch.tensor(4.))


def test_aux_loss_ignores_invalid_rows_and_handles_empty_populations():
    logits = torch.randn(3, 4, requires_grad=True)
    ids = torch.tensor([[0, 1], [1, 2], [2, 3]])
    probabilities = logits.softmax(-1)
    valid = torch.tensor([True, False, False])
    expected = load_balancing_loss(probabilities[:1], ids[:1], valid[:1])
    torch.testing.assert_close(load_balancing_loss(probabilities, ids, valid), expected)
    empty = load_balancing_loss(probabilities, ids, torch.zeros_like(valid))
    assert empty.item() == 0
    empty.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_packed_down_projection_matches_autograd_with_an_empty_expert():
    torch.manual_seed(31)
    x = torch.randn(5, 8, requires_grad=True)
    weight = torch.randn(3, 4, 8, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    actual = _ExpertDownProjection.apply(x, weight, torch.tensor([2, 2, 5], dtype=torch.int32))
    expected = torch.cat((
        reference_x[:2] @ reference_weight[0].T,
        reference_x[2:] @ reference_weight[2].T,
    ))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad, atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, reference_weight.grad, atol=0, rtol=0)
