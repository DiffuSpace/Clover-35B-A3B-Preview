from copy import deepcopy

import torch
import torch.nn.functional as F

from clover_sft.loss import CloverDenoisingHead


def test_chunked_head_matches_full_logits_loss_and_gradients():
    torch.manual_seed(11)
    head = CloverDenoisingHead(32, 64, bias=False)
    reference = deepcopy(head)
    hidden = torch.randn(7, 32, requires_grad=True)
    other = hidden.detach().clone().requires_grad_()
    targets = torch.tensor([1, 3, 5, 7, 9, 11, 13])
    actual = head(hidden, targets, chunk_size=2)
    expected = F.cross_entropy(reference(other), targets)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(hidden.grad, other.grad)
    torch.testing.assert_close(head.weight.grad, reference.weight.grad)


def test_empty_targets_still_have_connected_zero_gradients():
    head = CloverDenoisingHead(32, 64, bias=False)
    hidden = torch.empty(0, 32, requires_grad=True)
    loss = head(hidden, torch.empty(0, dtype=torch.long), chunk_size=2)
    assert loss.item() == 0
    loss.backward()
    assert hidden.grad is not None
    assert head.weight.grad is not None
    assert torch.count_nonzero(head.weight.grad) == 0
