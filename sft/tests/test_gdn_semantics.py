"""Independent contracts for read-only preview and decaying clean recurrence."""

import math

import torch

from clover_sft.attention import TrainingLayout
from clover_sft.gdn import gdn_forward
from clover_sft.modeling_clover_moe import (
    CloverGatedDeltaNet,
    bd_gdn_local_memory,
    recurrent_commit_torch,
)


def test_preview_matches_undecayed_history_bucket_formula():
    query = torch.tensor([[[[1., 0.]], [[0., 1.]]]], requires_grad=True)
    key = query.detach().clone().requires_grad_()
    value = torch.tensor([[[[2., 3.]], [[4., 7.]]]], requires_grad=True)
    state = torch.tensor([[[[1., 2.], [3., 5.]]]], requires_grad=True)
    gate = torch.tensor([[[-0.7], [-4.0]]], requires_grad=True)
    output = bd_gdn_local_memory(
        query, key, value, gate, clean_ssm_states=state,
        cache_indices=torch.tensor([0]), active_valid_mask=torch.ones(2, dtype=torch.bool),
        scale=2 ** -0.5, softmax_logit_scale=torch.zeros(1),
        softmax_history_bias=torch.zeros(1), softmax_history_c=torch.zeros(1),
    )
    # Zero local/history logits give three equally weighted buckets. Derive
    # the state read directly; do not call either preview implementation.
    normalized_q = query * torch.rsqrt(query.square().sum(-1, keepdim=True) + 1e-6)
    history = (normalized_q[0, :, 0] @ state[0, 0].T) * (2 ** -0.5)
    expected = ((value[0, :, 0].sum(0) + history) / 3)[None, :, None]
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-6)
    actual_grads = torch.autograd.grad(output.sum(), (query, value, state), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (query, value, state), retain_graph=True)
    for actual, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)
    gate_grad, = torch.autograd.grad(output.sum(), (gate,), allow_unused=True)
    assert gate_grad is None or torch.count_nonzero(gate_grad) == 0


def test_training_preview_is_invariant_to_preview_gate(tiny_config, tiny_batch):
    torch.manual_seed(41)
    module = CloverGatedDeltaNet(tiny_config)
    hidden = torch.randn(1, 32, tiny_config.hidden_size)
    layout = TrainingLayout(tiny_batch['token_roles'], tiny_batch['position_ids'], 4, gdn_backend='torch')
    original = gdn_forward(module, hidden, layout)

    def change_only_preview_gate(_module, _args, output):
        return torch.cat((output[:, :layout.length] + 7, output[:, layout.length:]), dim=1)

    hook = module.in_proj_a.register_forward_hook(change_only_preview_gate)
    try:
        changed = gdn_forward(module, hidden, layout)
    finally:
        hook.remove()
    torch.testing.assert_close(original, changed, atol=0, rtol=0)


def test_preview_has_no_noisy_gate_gradient_but_reads_decaying_clean_state(tiny_config, tiny_batch):
    torch.manual_seed(43)
    module = CloverGatedDeltaNet(tiny_config)
    hidden = torch.randn(1, 32, tiny_config.hidden_size, requires_grad=True)
    layout = TrainingLayout(tiny_batch['token_roles'], tiny_batch['position_ids'], 4, gdn_backend='torch')
    captured = []

    def capture_gate(_module, _args, output):
        output.retain_grad()
        captured.append(output)

    hook = module.in_proj_a.register_forward_hook(capture_gate)
    try:
        output = gdn_forward(module, hidden, layout)
        # Supervise only preview: clean gate gradients must come through the
        # state the preview reads, not directly from clean output supervision.
        (output[:, :layout.length] * torch.randn_like(output[:, :layout.length])).sum().backward()
    finally:
        hook.remove()
    gradient = captured[0].grad
    assert gradient is not None
    assert torch.count_nonzero(gradient[:, :layout.length]) == 0
    assert gradient[:, layout.length:].abs().sum() > 0


def test_clean_recurrence_still_decays_existing_state():
    query = torch.ones(1, 2, 1, 1)
    key = torch.ones_like(query)
    value = torch.tensor([[[[2.]], [[0.]]]])
    beta = torch.tensor([[[1.], [0.]]])  # second token only decays; no update
    initial = torch.zeros(1, 1, 1, 1)
    valid = torch.ones(1, 2, dtype=torch.bool)
    ordinary, ordinary_state = recurrent_commit_torch(query, key, value, torch.zeros_like(beta), beta, initial, valid)
    gate = torch.tensor([[[0.], [math.log(0.5)]]])
    decayed, decayed_state = recurrent_commit_torch(query, key, value, gate, beta, initial, valid)
    torch.testing.assert_close(decayed[:, 0], ordinary[:, 0])
    torch.testing.assert_close(decayed[:, 1], ordinary[:, 1] * 0.5)
    torch.testing.assert_close(decayed_state, ordinary_state * 0.5)
