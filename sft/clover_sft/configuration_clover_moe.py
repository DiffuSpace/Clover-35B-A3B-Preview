# Derived from Qwen/Qwen3.6-35B-A3B configuration.
# Upstream model configuration: Copyright 2026 Alibaba Cloud.
# Modified for Clover: independent text-only MoE inference configuration.
# SPDX-License-Identifier: Apache-2.0
"""Clover hybrid-attention MoE inference-only configuration."""

from copy import deepcopy
from typing import Any

from transformers.configuration_utils import PretrainedConfig


class CloverMoeConfig(PretrainedConfig):
    """Configuration for the Clover hybrid-attention MoE decoder.

    Args:
        attention_bias (bool, defaults to False):
            Whether full-attention projections include biases.
        attention_dropout (float, defaults to 0.0):
            Full-attention dropout metadata; inference uses zero dropout.
        attn_output_gate (bool, defaults to True):
            Enable gated full-attention output; Clover requires True.
        block_size (int, defaults to 32):
            Maximum denoising block length; generation overrides must divide it.
        bos_token_id (int | None, defaults to 248044):
            Beginning-of-sequence token ID, or None.
        dtype (str | None, defaults to 'bfloat16'):
            Preferred model floating-point dtype stored in the configuration.
        eos_token_id (int | list[int] | None, defaults to 248044):
            End-of-sequence token ID or IDs; generation configuration may override it.
        full_attention_interval (int, defaults to 4):
            Use full attention at every Nth layer when layer_types is omitted.
        head_dim (int, defaults to 256):
            Dimension of each full-attention head.
        hidden_act (str, defaults to 'silu'):
            Feed-forward activation; Clover requires silu.
        hidden_size (int, defaults to 2048):
            Hidden-state and embedding dimension.
        initializer_range (float, defaults to 0.02):
            Standard deviation for random weight initialization; loaded weights override it.
        linear_conv_kernel_dim (int, defaults to 4):
            Depthwise convolution kernel width in each GDN layer.
        linear_key_head_dim (int, defaults to 128):
            Dimension of each GDN query/key head.
        linear_num_key_heads (int, defaults to 16):
            Number of GDN query/key heads.
        linear_num_value_heads (int, defaults to 32):
            Number of GDN value heads; must be divisible by key heads.
        linear_value_head_dim (int, defaults to 128):
            Dimension of each GDN value head.
        mamba_ssm_dtype (str, defaults to 'float32'):
            Clean recurrent-state dtype; Clover requires float32.
        mask_token_id (int, defaults to 151669):
            Diffusion-mask token ID within the vocabulary.
        max_position_embeddings (int, defaults to 131072):
            Maximum supported sequence position count.
        moe_block_coreset_enabled (bool, defaults to True):
            Restrict preview routing to a block-level expert coreset.
        moe_block_coreset_top_p (float, defaults to 0.9):
            Cumulative routing-mass threshold for selecting the expert coreset.
        moe_intermediate_size (int, defaults to 512):
            Intermediate dimension of each routed expert.
        num_attention_heads (int, defaults to 16):
            Number of full-attention query heads.
        num_experts (int, defaults to 256):
            Number of routed experts per MoE layer.
        num_experts_per_tok (int, defaults to 8):
            Number of routed experts selected per token.
        num_hidden_layers (int, defaults to 40):
            Number of decoder layers.
        num_key_value_heads (int, defaults to 2):
            Number of full-attention key/value heads.
        output_router_logits (bool, defaults to False):
            Router-output metadata; this inference implementation returns no router logits.
        pad_token_id (int | None, defaults to None):
            Padding token ID, or None; generation configuration may override it.
        partial_rotary_factor (float, defaults to 0.25):
            Fraction of each full-attention head rotated by ordinary RoPE.
        rms_norm_eps (float, defaults to 1e-06):
            Epsilon used by RMS normalization.
        rope_parameters (dict[str, Any] | None, defaults to None):
            RoPE metadata. None uses default RoPE with theta=10000000 and partial factor=0.25.
        shared_expert_intermediate_size (int, defaults to 512):
            Intermediate dimension of the shared expert.
        tie_word_embeddings (bool, defaults to False):
            Whether input embeddings and the LM output head share weights.
        use_cache (bool, defaults to True):
            Default forward setting for returning clean-prefix cache states.
        vocab_size (int, defaults to 248320):
            Number of embedding and output vocabulary entries.
        num_shared_experts (int, defaults to 1):
            Number of shared experts per layer; Clover supports zero or one.
        attention_backend (str, defaults to 'sdpa'):
            PyTorch full-attention implementation: sdpa or eager.
        denoising_config (dict[str, Any] | None, defaults to None):
            Denoising settings; omitted entries default to algorithm=low_confidence and threshold=0.95.
        layer_types (list[str] | None, defaults to None):
            Per-layer linear_attention/full_attention list. None derives it from full_attention_interval.
        **kwargs:
            Additional Hugging Face configuration metadata. Existing fixed-architecture
            compatibility checks and removed-field handling remain unchanged.
    """

    model_type = "clover_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        attn_output_gate: bool = True,
        block_size: int = 32,
        bos_token_id: int | None = 248044,
        dtype: str | None = "bfloat16",
        eos_token_id: int | list[int] | None = 248044,
        full_attention_interval: int = 4,
        head_dim: int = 256,
        hidden_act: str = "silu",
        hidden_size: int = 2048,
        initializer_range: float = 0.02,
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 32,
        linear_value_head_dim: int = 128,
        mamba_ssm_dtype: str = "float32",
        mask_token_id: int = 151669,
        max_position_embeddings: int = 131072,
        moe_block_coreset_enabled: bool = True,
        moe_block_coreset_top_p: float = 0.9,
        moe_intermediate_size: int = 512,
        num_attention_heads: int = 16,
        num_experts: int = 256,
        num_experts_per_tok: int = 8,
        num_hidden_layers: int = 40,
        num_key_value_heads: int = 2,
        output_router_logits: bool = False,
        pad_token_id: int | None = None,
        partial_rotary_factor: float = 0.25,
        rms_norm_eps: float = 1e-06,
        rope_parameters: dict[str, Any] | None = None,
        shared_expert_intermediate_size: int = 512,
        tie_word_embeddings: bool = False,
        use_cache: bool = True,
        vocab_size: int = 248320,
        num_shared_experts: int = 1,
        attention_backend: str = "sdpa",
        denoising_config: dict[str, Any] | None = None,
        layer_types: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        values = {
            "attention_bias": attention_bias,
            "attention_dropout": attention_dropout,
            "attn_output_gate": attn_output_gate,
            "block_size": block_size,
            "bos_token_id": bos_token_id,
            "dtype": dtype,
            "eos_token_id": eos_token_id,
            "full_attention_interval": full_attention_interval,
            "head_dim": head_dim,
            "hidden_act": hidden_act,
            "hidden_size": hidden_size,
            "initializer_range": initializer_range,
            "linear_conv_kernel_dim": linear_conv_kernel_dim,
            "linear_key_head_dim": linear_key_head_dim,
            "linear_num_key_heads": linear_num_key_heads,
            "linear_num_value_heads": linear_num_value_heads,
            "linear_value_head_dim": linear_value_head_dim,
            "mamba_ssm_dtype": mamba_ssm_dtype,
            "mask_token_id": mask_token_id,
            "max_position_embeddings": max_position_embeddings,
            "moe_block_coreset_enabled": moe_block_coreset_enabled,
            "moe_block_coreset_top_p": moe_block_coreset_top_p,
            "moe_intermediate_size": moe_intermediate_size,
            "num_attention_heads": num_attention_heads,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_key_value_heads,
            "output_router_logits": output_router_logits,
            "pad_token_id": pad_token_id,
            "partial_rotary_factor": partial_rotary_factor,
            "rms_norm_eps": rms_norm_eps,
            "rope_parameters": (
                {
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default",
                }
                if rope_parameters is None
                else deepcopy(rope_parameters)
            ),
            "shared_expert_intermediate_size": shared_expert_intermediate_size,
            "tie_word_embeddings": tie_word_embeddings,
            "use_cache": use_cache,
            "vocab_size": vocab_size,
            "num_shared_experts": num_shared_experts,
            "attention_backend": attention_backend,
            "denoising_config": {
                "algorithm": "low_confidence",
                "threshold": 0.95,
                **(deepcopy(denoising_config) if denoising_config is not None else {}),
            },
        }
        values = deepcopy(values)
        values.update(deepcopy(kwargs))
        fixed_architecture = {
            "gdn_preview_style": "softmax_blocksize",
            "gdn_conv_mode": "independent_symmetric",
            "gdn_logit_scale_init": 10.0,
            "gdn_history_bias_init": 5.0,
            "gdn_history_c_init": 1.0,
            "gdn_reference_block_size": 32.0,
            "attention_pattern": "causal_prefix_bidirectional_block",
            "logits_alignment": "same_position",
        }
        for key, expected in fixed_architecture.items():
            if key in values and values[key] != expected:
                raise ValueError(
                    f"{key} is fixed by the Clover architecture: {expected!r}"
                )
            values.pop(key, None)
        values.pop("max_denoising_steps", None)
        values.pop("kernel_backend", None)
        values["model_type"] = self.model_type
        values["architectures"] = ["CloverMoeModelLM"]
        values["auto_map"] = {
            "AutoConfig": "configuration_clover_moe.CloverMoeConfig",
            "AutoModel": "modeling_clover_moe.CloverMoeModel",
            "AutoModelForCausalLM": "modeling_clover_moe.CloverMoeModelLM",
        }
        values["denoising_config"].pop("max_denoising_steps", None)
        if layer_types is None:
            interval = values["full_attention_interval"]
            if not isinstance(interval, int) or interval <= 0:
                raise ValueError("full_attention_interval must be positive")
            values["layer_types"] = [
                "full_attention" if (i + 1) % interval == 0 else "linear_attention"
                for i in range(values["num_hidden_layers"])
            ]
        else:
            values["layer_types"] = deepcopy(layer_types)
        if len(values["layer_types"]) != values["num_hidden_layers"]:
            raise ValueError("layer_types must contain one entry per hidden layer")
        if set(values["layer_types"]) - {"full_attention", "linear_attention"}:
            raise ValueError("Unknown attention layer type")
        if not 1 <= values["num_experts_per_tok"] <= values["num_experts"]:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")
        if values["num_shared_experts"] not in (0, 1):
            raise ValueError("Clover supports zero or one shared expert per layer")
        if values["linear_num_value_heads"] % values["linear_num_key_heads"]:
            raise ValueError("Linear value heads must be divisible by key heads")
        if values["num_attention_heads"] % values["num_key_value_heads"]:
            raise ValueError("Attention heads must be divisible by KV heads")
        rotary_dim = int(values["head_dim"] * values["partial_rotary_factor"])
        if rotary_dim <= 0 or rotary_dim % 2:
            raise ValueError("RoPE rotary_dim must be positive and even")
        if values["attention_backend"] not in ("sdpa", "eager"):
            raise ValueError("attention_backend must be sdpa or eager")
        if values["denoising_config"].get("algorithm") != "low_confidence":
            raise ValueError("Only low_confidence denoising is supported")
        if not 0 < values["moe_block_coreset_top_p"] <= 1:
            raise ValueError("moe_block_coreset_top_p must be in (0,1]")
        if not 0 <= values["mask_token_id"] < values["vocab_size"]:
            raise ValueError("mask_token_id is outside the vocabulary")
        if values["hidden_act"] != "silu" or values["attn_output_gate"] is not True:
            raise ValueError("Clover requires SiLU and gated full attention")
        if values["mamba_ssm_dtype"] != "float32":
            raise ValueError("Clean recurrent state must be float32")
        if values["block_size"] <= 0 or values["linear_conv_kernel_dim"] < 1:
            raise ValueError("Block and convolution sizes must be positive")
        super().__init__(**values)
