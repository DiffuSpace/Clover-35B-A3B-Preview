"""Training extension of native Clover with corrected read-only preview semantics."""

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import MoeCausalLMOutputWithPast

from .attention import TrainingLayout, attention_forward, build_block_mask
from .data import IGNORE_INDEX, SEQUENCE_LENGTH
from .gdn import gdn_forward
from .loss import CloverDenoisingHead
from .modeling_clover_moe import (
    CloverDecoderLayer,
    CloverMoeModel,
    CloverMoeModelLM,
    CloverMoePreTrainedModel,
    CloverRMSNorm,
)
from .moe import moe_forward


@dataclass
class CloverSFTOutput(MoeCausalLMOutputWithPast):
    """Keep CE and weighted router loss separate for Trainer normalization."""

    denoising_loss: torch.Tensor | None = None
    router_aux_loss: torch.Tensor | None = None


class CloverTrainingDecoderLayer(CloverDecoderLayer):
    def forward(self, x, *args, training_layout=None, replay=None, compute_aux_loss=False, **kwargs):
        if training_layout is None:
            return super().forward(x, *args, **kwargs)
        hidden = self.input_layernorm(x)
        if self.preview_type == "linear_attention":
            hidden = gdn_forward(self.linear_attn, hidden, training_layout)
        else:
            hidden = attention_forward(self.self_attn, hidden, training_layout)
        x = x + hidden
        routed = moe_forward(
            self.mlp, self.post_attention_layernorm(x), training_layout, replay,
            compute_aux_loss=compute_aux_loss,
        )
        if compute_aux_loss:
            hidden, aux_loss = routed
            return x + hidden, aux_loss
        return x + routed


class CloverTrainingModel(CloverMoeModel):
    def __init__(self, config):
        CloverMoePreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [CloverTrainingDecoderLayer(config, index) for index in range(config.num_hidden_layers)]
        )
        self.norm = CloverRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_init()


class CloverForSFT(CloverMoeModelLM):
    supports_gradient_checkpointing = True
    _no_split_modules = ["CloverTrainingDecoderLayer"]
    # Swift's template applies accumulated/DP token-count normalization.
    accepts_loss_kwargs = True

    def __init__(
        self, config, sequence_length=SEQUENCE_LENGTH, gdn_backend="fla", ce_chunk_size=128,
        router_aux_loss_coef=None,
    ):
        CloverMoePreTrainedModel.__init__(self, config)
        self.model = CloverTrainingModel(config)
        self.lm_head = CloverDenoisingHead(config.hidden_size, config.vocab_size, bias=False)
        self.sequence_length = sequence_length
        self.gdn_backend = gdn_backend
        self.ce_chunk_size = ce_chunk_size
        self.router_aux_loss_coef = float(
            getattr(config, "router_aux_loss_coef", 1e-3) if router_aux_loss_coef is None else router_aux_loss_coef
        )
        config.router_aux_loss_coef = self.router_aux_loss_coef
        self.gradient_checkpointing = False
        self.post_init()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        options = dict(gradient_checkpointing_kwargs or {})
        if options.get("use_reentrant", False):
            raise ValueError("Clover SFT requires non-reentrant gradient checkpointing")
        options["use_reentrant"] = False
        return super().gradient_checkpointing_enable(options)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        requested_info = kwargs.pop("output_loading_info", False)
        model, info = super().from_pretrained(*args, output_loading_info=True, **kwargs)
        errors = {name: info.get(name) for name in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")}
        if any(errors.values()):
            raise ValueError(f"Native Clover SFT requires complete compatible trained weights: {errors}")
        return (model, info) if requested_info else model

    def forward(
        self, input_ids=None, attention_mask=None, position_ids=None, labels=None,
        noisy_input_ids=None, token_roles=None, output_router_logits=None, **kwargs,
    ):
        if labels is None:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, **kwargs,
            )
        coefficient = self.router_aux_loss_coef
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("router_aux_loss_coef must be finite and nonnegative")
        # Swift requests router outputs to enable auxiliary-loss reporting.
        # Return scalar statistics without retaining all layers' router logits.
        compute_aux_loss = coefficient > 0 or bool(output_router_logits)
        if input_ids.shape[1] != self.sequence_length:
            raise ValueError(f"Expected fixed semantic length {self.sequence_length}, got {input_ids.shape[1]}")
        if noisy_input_ids is None or token_roles is None or position_ids is None:
            raise ValueError("Training requires the Clover collator's noise, roles and semantic positions")
        layout = TrainingLayout(token_roles, position_ids, self.config.block_size, gdn_backend=self.gdn_backend)
        if input_ids.is_cuda and "full_attention" in self.config.layer_types:
            layout.block_mask = build_block_mask(layout)
        physical_ids = torch.cat((noisy_input_ids, input_ids), dim=1)
        hidden = self.model.embed_tokens(physical_ids)
        aux_loss = None
        for layer in self.model.layers:
            # Each checkpoint owns its first-forward routing decisions; the
            # closure survives until backward and is never reused next step.
            replay = {}
            if self.gradient_checkpointing and self.training:
                output = checkpoint(
                    layer, hidden, training_layout=layout, replay=replay,
                    compute_aux_loss=compute_aux_loss,
                    use_reentrant=False, preserve_rng_state=True,
                )
            else:
                output = layer(hidden, training_layout=layout, replay=replay, compute_aux_loss=compute_aux_loss)
            if compute_aux_loss:
                hidden, layer_aux = output
                aux_loss = layer_aux if aux_loss is None else aux_loss + layer_aux
            else:
                hidden = output
        hidden = self.model.norm(hidden[:, :self.sequence_length])
        selected = labels != IGNORE_INDEX
        targets = labels[selected]
        hidden = hidden[selected]
        denoising_loss = self.lm_head(hidden, targets, chunk_size=self.ce_chunk_size)
        router_aux_loss = coefficient * aux_loss if coefficient > 0 and aux_loss is not None else None
        loss = denoising_loss if router_aux_loss is None else denoising_loss + router_aux_loss
        return CloverSFTOutput(
            loss=loss, logits=None, aux_loss=aux_loss,
            denoising_loss=denoising_loss, router_aux_loss=router_aux_loss,
        )

    def save_pretrained(self, save_directory, *args, **kwargs):
        """Export native weights with the corrected standalone inference code."""
        result = super().save_pretrained(save_directory, *args, **kwargs)
        if kwargs.get("is_main_process", True):
            root = Path(save_directory)
            self.config.architectures = ["CloverMoeModelLM"]
            self.config.auto_map = {
                "AutoConfig": "configuration_clover_moe.CloverMoeConfig",
                "AutoModel": "modeling_clover_moe.CloverMoeModel",
                "AutoModelForCausalLM": "modeling_clover_moe.CloverMoeModelLM",
            }
            self.config.save_pretrained(root)
            for name in ("configuration_clover_moe.py", "modeling_clover_moe.py", "sampling_clover_moe.py"):
                shutil.copyfile(Path(__file__).parent / name, root / name)
        return result
