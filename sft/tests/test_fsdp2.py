"""Two-rank regression for token-dependent CE chunk counts and grouped tails.

Run with two visible CUDA GPUs. The worker also compares every gradient to
ordinary data-parallel accumulation, with and without layer recomputation.
"""

import datetime
import os
import signal
import subprocess
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from swift.trainers.seq2seq_trainer import Seq2SeqTrainer
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from clover_sft.configuration_clover_moe import CloverMoeConfig
from clover_sft.data import IGNORE_INDEX, align_sample, collate_samples
from clover_sft.model import CloverForSFT
from clover_sft.swift_plugin import CloverTemplate


def _worker():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        'nccl', timeout=datetime.timedelta(seconds=60), device_id=torch.device('cuda', rank),
    )
    config = CloverMoeConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=16,
        shared_expert_intermediate_size=16, block_size=4,
        mask_token_id=63, eos_token_id=2, bos_token_id=2, pad_token_id=0,
    )
    sample = align_sample([10, 11, 12], [20, 21, 22, 23, 2], block_size=4, pad_id=0, length=16)
    batches = []
    for count in ([1, 3] if rank == 0 else [5, 7]):
        batch = collate_samples([sample], mask_id=63, generator=torch.Generator().manual_seed(0))
        selected = torch.zeros_like(batch['labels'], dtype=torch.bool)
        selected[:, 4:4 + count] = True
        batch['labels'] = torch.tensor([sample['labels']]).masked_fill(~selected, IGNORE_INDEX)
        batch['noisy_input_ids'] = batch['input_ids'].masked_fill(selected, 63)
        batches.append({key: value.cuda() for key, value in batch.items()})
    total = sum(batch['labels'].ne(IGNORE_INDEX).sum() for batch in batches)
    dist.all_reduce(total)
    for coefficient, recompute, sync_each_batch in product((0., 0.03), (False, True), (False, True)):
        torch.manual_seed(37)
        model = CloverForSFT(
            deepcopy(config), sequence_length=16, gdn_backend='torch', ce_chunk_size=2,
            router_aux_loss_coef=coefficient,
        ).cuda()
        reference = deepcopy(model)
        policy = {'mp_policy': MixedPrecisionPolicy(), 'reshard_after_forward': True}
        for index, layer in enumerate(model.model.layers):
            if recompute:
                layer = checkpoint_wrapper(layer, preserve_rng_state=False)
                model.model.layers[index] = layer
            fully_shard(layer, **policy)
        fully_shard(model.model.embed_tokens, **policy)
        # Match Accelerate's FSDP2 grouping of final norm and output head.
        fully_shard([model.model.norm, model.lm_head], **{**policy, 'reshard_after_forward': False})
        fully_shard(model, mp_policy=MixedPrecisionPolicy())
        template = object.__new__(CloverTemplate)
        template.sequence_parallel_size = 1
        trainer = SimpleNamespace(
            model=model, template=template, label_smoother=None, model_accepts_loss_kwargs=True,
            current_gradient_accumulation_steps=len(batches),
            args=SimpleNamespace(
                average_tokens_across_devices=True, enable_dft_loss=False,
                enable_channel_loss=False, past_index=-1, tuner_backend='peft',
            ),
            accelerator=SimpleNamespace(num_processes=dist.get_world_size()),
            custom_metrics={'train': {'aux_loss': SimpleNamespace(update=lambda value: None)}},
        )
        for index, batch in enumerate(batches):
            local_count = batch['labels'].ne(IGNORE_INDEX).sum()
            model.set_requires_gradient_sync(sync_each_batch or index == len(batches) - 1)
            model.set_is_last_backward(index == len(batches) - 1)
            actual = Seq2SeqTrainer.compute_loss(trainer, model, dict(batch), num_items_in_batch=total)
            output = reference(**batch)
            expected = output.denoising_loss * local_count / total * dist.get_world_size()
            if coefficient:
                expected = expected + coefficient * output.aux_loss / len(batches)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
            actual.backward()
            expected.backward()
        gradients = {
            name.replace('._checkpoint_wrapped_module', ''): parameter.grad
            for name, parameter in model.named_parameters()
        }
        for name, parameter in reference.named_parameters():
            assert parameter.grad is not None, name
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(dist.get_world_size())
            actual_grad = gradients[name]
            assert actual_grad is not None, name
            # Compare each local DTensor shard to the corresponding part of
            # the explicitly averaged, unsharded reference gradient.
            dimension = actual_grad.placements[0].dim
            size = parameter.grad.shape[dimension]
            chunk = (size + dist.get_world_size() - 1) // dist.get_world_size()
            offset = min(rank * chunk, size)
            expected_shard = parameter.grad.narrow(dimension, offset, min(chunk, size - offset))
            torch.testing.assert_close(
                actual_grad.to_local(), expected_shard, atol=3e-6, rtol=3e-4, msg=name,
            )
        print(
            f'rank={rank} aux={coefficient} recompute={recompute} sync_each_batch={sync_each_batch}: '
            'all gradients match token-weighted DP accumulation', flush=True,
        )
        del model, reference
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='FSDP2 regression requires two CUDA GPUs')
def test_fsdp2_unequal_ce_chunks_and_accumulation():
    root = str(Path(__file__).resolve().parents[1])
    env = dict(
        os.environ, OMP_NUM_THREADS='1', TORCH_DISTRIBUTED_DEBUG='DETAIL',
        PYTHONPATH=os.pathsep.join(filter(None, (root, os.environ.get('PYTHONPATH')))),
    )
    command = [
        sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
        str(Path(__file__).resolve()), '--worker',
    ]
    process = subprocess.Popen(
        command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()
        pytest.fail(f'FSDP2 worker timed out:\n{output}')
    assert process.returncode == 0, output
    assert output.count('all gradients match token-weighted DP accumulation') == 16, output


if __name__ == '__main__' and '--worker' in sys.argv:
    _worker()
