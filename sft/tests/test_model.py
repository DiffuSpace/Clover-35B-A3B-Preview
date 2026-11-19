from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from clover_sft.data import IGNORE_INDEX, Role
from clover_sft.model import CloverForSFT
from clover_sft.modeling_clover_moe import CloverMoeModelLM


def inference_reference_loss(model, batch):
    """Teacher-force blocks through the locally corrected inference API.

    There is no xt||x0 layout, training scan, training convolution, or custom
    attention mask in this oracle. Gradients flow through the clean caches.
    """
    roles = batch["token_roles"][0]
    block = model.config.block_size
    numerator = model.lm_head.weight.new_zeros((), dtype=torch.float32)
    denominator = (batch["labels"] != IGNORE_INDEX).sum()
    for start in range(0, roles.numel(), block):
        selected = batch["labels"][:, start:start + block] != IGNORE_INDEX
        if not selected.any():
            continue
        prefix_roles = roles[:start]
        real = (prefix_roles == Role.PROMPT) | (prefix_roles == Role.RESPONSE)
        prefix = batch["input_ids"][:, :start][:, real]
        cache = model.model(prefix, use_cache=True).past_key_values
        output = model(
            batch["noisy_input_ids"][:, start:start + block], past_key_values=cache,
            position_ids=batch["position_ids"][:, start:start + block], preview=True,
        )
        numerator = numerator + F.cross_entropy(
            output.logits[selected].float(), batch["labels"][:, start:start + block][selected], reduction="sum",
        )
    return numerator / denominator


def test_training_matches_corrected_inference_outputs_and_all_gradients(tiny_config, tiny_batch):
    torch.manual_seed(17)
    trained = CloverForSFT(tiny_config, sequence_length=16, gdn_backend="torch")
    # Exercise learned right convolution and all softmax parameters.
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if name.endswith("conv_right_weight"):
                parameter.normal_(std=0.01)
    reference = CloverMoeModelLM(deepcopy(tiny_config))
    reference.load_state_dict(trained.state_dict(), strict=True)
    actual = trained(**tiny_batch).loss
    expected = inference_reference_loss(reference, tiny_batch)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    actual.backward()
    expected.backward()
    for (name, parameter), (other_name, other) in zip(trained.named_parameters(), reference.named_parameters()):
        assert name == other_name
        if parameter.grad is None or other.grad is None:
            assert parameter.grad is None and other.grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, other.grad, atol=3e-5, rtol=3e-4, msg=name)


@pytest.mark.parametrize("aux_coefficient", [0., 0.03])
def test_non_reentrant_recompute_preserves_loss_and_gradients(tiny_config, tiny_batch, aux_coefficient):
    torch.manual_seed(23)
    tiny_config.router_aux_loss_coef = aux_coefficient
    ordinary = CloverForSFT(tiny_config, sequence_length=16, gdn_backend="torch")
    recomputed = deepcopy(ordinary)
    recomputed.gradient_checkpointing_enable()
    losses = []
    for model in (ordinary, recomputed):
        model.train()
        loss = model(**tiny_batch).loss
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(*losses, atol=0, rtol=0)
    for (name, parameter), (_, other) in zip(ordinary.named_parameters(), recomputed.named_parameters()):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=0, rtol=0, msg=name)


def test_model_adds_aux_once_and_router_gradients_remain_connected(tiny_config, tiny_batch):
    model = CloverForSFT(tiny_config, sequence_length=16, gdn_backend="torch", router_aux_loss_coef=0.03)
    output = model(**tiny_batch)
    torch.testing.assert_close(output.loss, output.denoising_loss + 0.03 * output.aux_loss)
    torch.testing.assert_close(output.router_aux_loss, 0.03 * output.aux_loss)
    gates = [layer.mlp.gate.weight for layer in model.model.layers]
    gradients = torch.autograd.grad(output.aux_loss, gates)
    assert all(gradient.isfinite().all() and gradient.abs().sum() > 0 for gradient in gradients)
    # Mask sampling must not change the aux population or normalization.
    changed = deepcopy(tiny_batch)
    changed["labels"].fill_(IGNORE_INDEX)
    empty_ce = model(**changed)
    torch.testing.assert_close(empty_ce.aux_loss, output.aux_loss, atol=0, rtol=0)
    assert empty_ce.denoising_loss.item() == 0


def test_aux_loss_is_sum_of_independent_layer_statistics(tiny_config, tiny_batch):
    model = CloverForSFT(tiny_config, sequence_length=16, gdn_backend="torch", router_aux_loss_coef=0.03)
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.zero_()
    # Disable coreset filtering: uniform p gives E sum_e f_e p_e = 1 per layer.
    model.config.moe_block_coreset_enabled = False
    output = model(**tiny_batch)
    torch.testing.assert_close(output.aux_loss, torch.tensor(float(tiny_config.num_hidden_layers)))


def test_aux_population_includes_noisy_postpad_and_real_clean_tokens_only(tiny_config, tiny_batch, monkeypatch):
    from clover_sft import moe

    original = moe.load_balancing_loss
    observed = []

    def inspect_population(probabilities, ids, valid):
        # Fixture: prompt 0:3, pre-pad 3, response/EOS 4:9, post-pad 9:12,
        # batch pad 12:16. The second physical stream starts at position 16.
        assert valid[0].nonzero(as_tuple=True)[0].tolist() == [
            4, 5, 6, 7, 8, 9, 10, 11, 16, 17, 18, 20, 21, 22, 23, 24,
        ]
        observed.append(True)
        return original(probabilities, ids, valid)

    monkeypatch.setattr(moe, 'load_balancing_loss', inspect_population)
    model = CloverForSFT(tiny_config, sequence_length=16, gdn_backend='torch', router_aux_loss_coef=0.03)
    model(**tiny_batch)
    assert len(observed) == tiny_config.num_hidden_layers


def test_aux_default_override_and_invalid_coefficients(tiny_config, tiny_batch):
    del tiny_config.router_aux_loss_coef
    model = CloverForSFT(tiny_config, sequence_length=16, gdn_backend="torch")
    assert model.router_aux_loss_coef == 1e-3
    assert model.config.router_aux_loss_coef == 1e-3
    model.router_aux_loss_coef = 0
    assert model(**tiny_batch).aux_loss is None
    assert model(**tiny_batch, output_router_logits=True).aux_loss is not None
    for invalid in (-1., float('inf'), float('nan')):
        model.router_aux_loss_coef = invalid
        with pytest.raises(ValueError, match="finite and nonnegative"):
            model(**tiny_batch)


def test_future_clean_and_noisy_blocks_cannot_change_an_earlier_objective(tiny_config, tiny_batch):
    torch.manual_seed(71)
    model = CloverForSFT(tiny_config, sequence_length=16, gdn_backend='torch')
    batch = deepcopy(tiny_batch)
    batch['labels'][:, 8:] = IGNORE_INDEX
    assert batch['labels'].ne(IGNORE_INDEX).any()
    expected = model(**batch).loss
    changed = deepcopy(batch)
    # The second response block follows the only supervised block. Perturb
    # both streams there, leaving all labels and token roles fixed.
    changed['input_ids'][:, 8:12] = torch.tensor([[31, 32, 33, 34]])
    changed['noisy_input_ids'][:, 8:12] = torch.tensor([[35, 36, 37, 38]])
    torch.testing.assert_close(model(**changed).loss, expected, atol=0, rtol=0)


def test_export_loads_as_native_clover_without_training_dependencies(tmp_path, tiny_config):
    model = CloverForSFT(tiny_config, sequence_length=16, gdn_backend="torch")
    model.save_pretrained(tmp_path)
    source = Path(__file__).resolve().parents[1] / "clover_sft" / "modeling_clover_moe.py"
    assert (tmp_path / source.name).read_bytes() == source.read_bytes()
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True)
    assert loaded.__class__.__name__ == "CloverMoeModelLM"
    assert loaded.state_dict().keys() == model.state_dict().keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[name], atol=0, rtol=0)
    # Check behavior after native reload, not just tensor names/weights. Hold
    # the clean cache fixed while changing only the preview's gate values.
    with torch.no_grad():
        cache = loaded.model(torch.tensor([[10, 11, 12]]), use_cache=True).past_key_values
        tokens = torch.tensor([[20, 63, 22, 63]])
        expected = loaded(tokens, past_key_values=cache, preview=True).logits
        for layer in loaded.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.dt_bias.add_(7)
        actual = loaded(tokens, past_key_values=cache, preview=True).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
