"""The actual ms-swift loss adapter must preserve token-weighted accumulation."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer

from clover_sft.data import IGNORE_INDEX, align_sample
from clover_sft.model import CloverForSFT
from clover_sft.swift_plugin import CloverTemplate


def trainer_context(model, *, steps=2, ranks=1, average_tokens=True):
    """Use the real Swift compute_loss path with a minimal orchestration context."""
    template = object.__new__(CloverTemplate)
    template.sequence_parallel_size = 1
    return SimpleNamespace(
        model=model, template=template, label_smoother=None, model_accepts_loss_kwargs=True,
        current_gradient_accumulation_steps=steps,
        args=SimpleNamespace(
            average_tokens_across_devices=average_tokens, enable_dft_loss=False,
            enable_channel_loss=False, past_index=-1, tuner_backend='peft',
        ),
        accelerator=SimpleNamespace(num_processes=ranks),
        custom_metrics={'train': {'aux_loss': SimpleNamespace(update=lambda value: None)},
                        'eval': {'aux_loss': SimpleNamespace(update=lambda value: None)}},
    )


@pytest.mark.parametrize("average_tokens,ranks", [(False, 3), (True, 1), (True, 3)])
def test_swift_aux_is_microbatch_mean_independent_of_ce_token_counts(
    tiny_config, tiny_batch, average_tokens, ranks,
):
    torch.manual_seed(41)
    actual_model = CloverForSFT(tiny_config, sequence_length=16, gdn_backend='torch', router_aux_loss_coef=0.1)
    with torch.no_grad():
        for layer in actual_model.model.layers:
            layer.mlp.gate.weight.mul_(10)
    reference = deepcopy(actual_model)
    first, second = deepcopy(tiny_batch), deepcopy(tiny_batch)
    selected = second['labels'][0].ne(IGNORE_INDEX).nonzero(as_tuple=True)[0]
    second['labels'][0, selected[1:]] = IGNORE_INDEX
    second['input_ids'][0, :3] = torch.tensor([30, 31, 32])
    second['noisy_input_ids'][0, 4:8] = torch.tensor([33, 34, 35, 36])
    total = sum(batch['labels'].ne(IGNORE_INDEX).sum() for batch in (first, second))
    denominator = total * ranks if average_tokens else total
    trainer = trainer_context(actual_model, ranks=ranks, average_tokens=average_tokens)
    actual, expected = 0, 0
    auxiliary = []
    for batch in (first, second):
        loss = Seq2SeqTrainer.compute_loss(trainer, actual_model, dict(batch), num_items_in_batch=denominator)
        output = reference(**batch)
        weight = batch['labels'].ne(IGNORE_INDEX).sum() / denominator
        ce = output.denoising_loss * weight * (ranks if average_tokens else 1)
        independent = ce + 0.1 * output.aux_loss / 2
        auxiliary.append(output.aux_loss.detach())
        loss.backward()
        independent.backward()
        actual = actual + loss.detach()
        expected = expected + independent.detach()
    assert (auxiliary[0] - auxiliary[1]).abs() > 1e-3
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    for (name, parameter), (_, other) in zip(actual_model.named_parameters(), reference.named_parameters()):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_swift_aux_empty_ce_and_eval_are_not_scaled_by_training_accumulation(tiny_config, tiny_batch):
    model = CloverForSFT(tiny_config, sequence_length=16, gdn_backend='torch', router_aux_loss_coef=0.1).eval()
    batch = deepcopy(tiny_batch)
    batch['labels'].fill_(IGNORE_INDEX)
    trainer = trainer_context(model, steps=99, ranks=2)
    actual = Seq2SeqTrainer.compute_loss(trainer, model, batch, num_items_in_batch=torch.tensor(0))
    expected = 0.1 * model(**batch).aux_loss
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_noise_streams_are_rank_independent_reproducible_and_separate_from_model_rng(monkeypatch):
    sample = align_sample([10, 11, 12], [20] * 60 + [2], block_size=4, pad_id=0, length=128)
    monkeypatch.setattr(torch.distributed, 'is_initialized', lambda: True)

    def draw(rank, disturb_model_rng=False):
        monkeypatch.setattr(torch.distributed, 'get_rank', lambda: rank)
        torch.manual_seed(43)
        template = object.__new__(CloverTemplate)
        template.mode = 'train'
        template.config = SimpleNamespace(mask_token_id=63)
        before = torch.get_rng_state()
        first = template._data_collator([sample])['noisy_input_ids']
        assert torch.equal(torch.get_rng_state(), before)
        if disturb_model_rng:
            torch.rand(100)  # model / sampler draws must not affect later noise
        second = template._data_collator([sample])['noisy_input_ids']
        return first, second

    first, second = draw(0)
    repeat_first, repeat_second = draw(0, disturb_model_rng=True)
    other_first, _ = draw(1)
    assert torch.equal(first, repeat_first) and torch.equal(second, repeat_second)
    assert not torch.equal(first, second)
    assert not torch.equal(first, other_first)


def test_swift_accumulation_matches_combined_masked_token_objective(tiny_config, tiny_batch):
    torch.manual_seed(37)
    combined_model = CloverForSFT(tiny_config, sequence_length=16, gdn_backend="torch")
    accumulated_model = deepcopy(combined_model)
    first = deepcopy(tiny_batch)
    second = deepcopy(tiny_batch)
    # Keep a different number of denoising targets in each microbatch. A mean
    # of per-microbatch means would give the wrong gradients in this case.
    selected = second["labels"][0].ne(IGNORE_INDEX).nonzero(as_tuple=True)[0]
    assert selected.numel() > 1
    second["labels"][0, selected[1:]] = IGNORE_INDEX
    combined = {key: torch.cat((first[key], second[key])) for key in first}
    expected = combined_model(**combined).loss
    expected.backward()
    total = combined["labels"].ne(IGNORE_INDEX).sum()
    template = object.__new__(CloverTemplate)
    actual = 0
    for microbatch in (first, second):
        loss = template.compute_sft_loss(accumulated_model, microbatch, num_items_in_batch=total).loss
        actual = actual + loss.detach()
        loss.backward()
    torch.testing.assert_close(actual, expected.detach(), atol=2e-6, rtol=2e-6)
    for (name, parameter), (_, other) in zip(combined_model.named_parameters(), accumulated_model.named_parameters()):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, other.grad, atol=2e-6, rtol=2e-4, msg=name)


def test_swift_asset_copy_does_not_restore_stale_release_code(tmp_path):
    from pathlib import Path

    from swift.model import MODEL_MAPPING, save_checkpoint

    source = Path(__file__).resolve().parents[1] / 'clover_sft' / 'modeling_clover_moe.py'
    base = tmp_path / 'base'
    output = tmp_path / 'output'
    base.mkdir()
    output.mkdir()
    (base / source.name).write_text('# stale released inference implementation\n')
    (output / source.name).write_bytes(source.read_bytes())

    class Processor:
        def save_pretrained(self, output_dir):
            pass

    # Exercise ms-swift's actual post-save asset-copy step with our model
    # registration, after CloverForSFT has exported corrected inference code.
    save_checkpoint(
        None, Processor(), str(output), model_dirs=[str(base)],
        additional_saved_files=MODEL_MAPPING['clover_sft'].additional_saved_files,
    )
    assert (output / source.name).read_bytes() == source.read_bytes()
