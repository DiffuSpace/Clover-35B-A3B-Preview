"""External ms-swift registration; no changes to the framework are required."""

import json
from pathlib import Path

import torch
from swift.model import Model, ModelGroup, ModelLoader, ModelMeta, register_model
from swift.template import Template, TemplateMeta, register_template
from swift.template.base import MaxLengthError

from clover_sft.configuration_clover_moe import CloverMoeConfig
from clover_sft.data import SEQUENCE_LENGTH, collate_samples, encode_single_turn
from clover_sft.model import CloverForSFT


class CloverLoader(ModelLoader):
    def get_config(self, model_dir):
        serialized = json.loads((Path(model_dir) / "config.json").read_text())
        if serialized.get("model_type") != "clover_moe" or serialized.get("architectures") != ["CloverMoeModelLM"]:
            raise ValueError("Load a native Clover checkpoint, not a Qwen base checkpoint")
        return CloverMoeConfig.from_pretrained(model_dir)

    def get_model(self, model_dir, config, processor, model_kwargs):
        self.auto_model_cls = CloverForSFT
        return super().get_model(model_dir, config, processor, model_kwargs)


class CloverTemplate(Template):
    def _encode(self, inputs):
        if self.packing or self.padding_free or self.sequence_parallel_size != 1:
            raise ValueError("Clover SFT requires packing=false, padding_free=false and sequence_parallel_size=1")
        if self.max_length != SEQUENCE_LENGTH:
            raise ValueError("Clover SFT uses fixed max_length=4096 (physical length=8192)")
        if inputs.tools or inputs.images or inputs.videos or inputs.audios:
            raise ValueError("Clover SFT supports text-only single-turn examples")
        try:
            return encode_single_turn(
                self.tokenizer, inputs.messages, block_size=self.config.block_size,
                system=inputs.system, length=SEQUENCE_LENGTH,
            )
        except ValueError as error:
            if "exceeds fixed length" in str(error):
                raise MaxLengthError(str(error)) from error
            raise

    def _data_collator(self, batch, *, padding_to=None):
        if self.is_training:
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            # Trainer seeds model initialization identically on every rank.
            # Give noise a separate stream; worker initial seeds also differ.
            seed = (torch.initial_seed() + 1_000_003 * rank) % (2**63)
            if getattr(self, "_noise_seed", None) != seed:
                self._noise_seed = seed
                self._noise_generator = torch.Generator().manual_seed(seed)
            generator = self._noise_generator
        else:
            generator = torch.Generator().manual_seed(0)
        return collate_samples(batch, mask_id=self.config.mask_token_id, generator=generator)

    def compute_sft_loss(self, model, inputs, num_items_in_batch=None, trainer=None):
        if "labels" not in inputs:
            raise ValueError("Use the Clover model loss without AR label smoothing, custom losses or loss scaling")
        outputs = model(**inputs)
        if num_items_in_batch is None:
            return outputs
        # Diffusion targets are same-position labels, with no AR shift.
        count = inputs["labels"].ne(-100).sum()
        denominator = torch.as_tensor(num_items_in_batch, device=count.device).clamp_min(1)
        outputs.loss = outputs.denoising_loss * (count / denominator)
        if outputs.router_aux_loss is not None:
            if trainer is None:
                raise ValueError("Router aux normalization requires the Trainer accumulation/DP context")
            steps = trainer.current_gradient_accumulation_steps if model.training else 1
            # Swift multiplies the entire loss by DP size when its denominator
            # includes all ranks. Undo that multiplier for the local aux mean.
            ranks = (
                trainer.accelerator.num_processes
                if trainer.args.average_tokens_across_devices and trainer.model_accepts_loss_kwargs else 1
            )
            outputs.loss = outputs.loss + outputs.router_aux_loss / (steps * ranks)
        return outputs


register_template(TemplateMeta(
    "clover_sft", prefix=[], prompt=["{{QUERY}}"], chat_sep=None,
    system_prefix=["{{SYSTEM}}"], template_cls=CloverTemplate,
))
register_model(ModelMeta(
    "clover_sft", [ModelGroup([Model(hf_model_id="Diffuspace/Clover-35B-A3B-Preview")])],
    loader=CloverLoader, template="clover_sft", architectures=["CloverMoeModelLM"],
    # CloverForSFT.save_pretrained owns inference-source export. Copying these
    # files again from model_dir would restore the upstream preview-decay bug.
    additional_saved_files=[],
))
