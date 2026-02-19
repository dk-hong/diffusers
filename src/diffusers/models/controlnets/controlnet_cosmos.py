# Copyright 2025 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...utils import logging, is_torchvision_available
from ..modeling_utils import ModelMixin
from ..transformers.transformer_cosmos import (
    CosmosPatchEmbed,
    CosmosEmbedding,
    CosmosRotaryPosEmbed,
    CosmosLearnablePositionalEmbed,
    CosmosTransformerBlock,
)

from .controlnet import BaseOutput, zero_module


if is_torchvision_available():
    from torchvision import transforms


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class CosmosControlNetOutput(BaseOutput):
    controlnet_block_samples: Tuple[torch.Tensor]


class CosmosControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    r"""
    A ControlNet model for video-like data used in [Cosmos](https://github.com/NVIDIA/Cosmos).

    Parameters:
        in_channels (`int`, defaults to `17`):
            The number of latent channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        hint_channels

        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each attention head.
        num_layers (`int`, defaults to `18`):
            The number of layers of transformer blocks to use.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        adaln_lora_dim (`int`, defaults to `256`):
            The hidden dimension of the Adaptive LayerNorm LoRA layer.
        max_size (`Tuple[int, int, int]`, defaults to `(128, 240, 240)`):
            The maximum size of the input latent tensors in the temporal, height, and width dimensions.
        patch_size (`Tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            The patch size to use for patchifying the input latent tensors in the temporal, height, and width
            dimensions.
        rope_scale (`Tuple[float, float, float]`, defaults to `(2.0, 1.0, 1.0)`):
            The scaling factor to use for RoPE in the temporal, height, and width dimensions.
        hint_nf

        concat_padding_mask (`bool`, defaults to `True`):
            Whether to concatenate the padding mask to the input latent tensors.
        affine_emb_norm

        extra_pos_embed_type (`str`, *optional*, defaults to `learnable`):
            The type of extra positional embeddings to use. Can be one of `None` or `learnable`.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16 + 1,
        out_channels: int = 16,
        hint_channels: int = 128,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        hint_nf: Tuple[int, int, int, int, int, int, int] = [16, 16, 32, 32, 96, 96, 256],
        concat_padding_mask: bool = True,
        extra_pos_embed_type: Optional[str] = "learnable",
    ) -> None:
        super().__init__()
        hidden_size = num_attention_heads * attention_head_dim

        # 1. Patch Embedding
        patch_embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels, hidden_size, patch_size, bias=False)
        patch_embed_hint_channels = hint_channels + 1 if concat_padding_mask else hint_channels
        self.patch_embed2 = CosmosPatchEmbed(patch_embed_hint_channels, hidden_size, patch_size, bias=False)

        # 2. Positional Embedding
        self.rope = CosmosRotaryPosEmbed(
            hidden_size=attention_head_dim, max_size=max_size, patch_size=patch_size, rope_scale=rope_scale
        )

        self.learnable_pos_embed = None
        if extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=hidden_size,
                max_size=max_size,
                patch_size=patch_size,
            )

        # 3. Time Embedding
        self.time_embed = CosmosEmbedding(hidden_size, hidden_size)

        # 4. Transformer Blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=text_embed_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    qk_norm="rms_norm",
                    out_bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Controlnet Blocks
        self.controlnet_blocks = nn.ModuleList([])
        for idx in range(len(self.transformer_blocks)):
            controlnet_block = nn.Linear(hidden_size, hidden_size)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)

        input_hint_block = [nn.Linear(hidden_size, hint_nf[0]), nn.SiLU()]
        for i in range(len(hint_nf) - 1):
            input_hint_block += [nn.Linear(hint_nf[i], hint_nf[i + 1]), nn.SiLU()]
        input_hint_block.append(zero_module(nn.Linear(hint_nf[-1], hidden_size)))
        self.input_hint_block = nn.Sequential(*input_hint_block)

        self.gradient_checkpointing = False

    def forward(
        self,
        hint: torch.Tensor,
        control_weight: Union[float, torch.Tensor],
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # 1. Encode hint
        if hint.size(1) < self.config.hint_channels:
            padding_channels = self.config.hint_channels - hint.size(1)
            hint_padding = hint.new_zeros(batch_size, padding_channels, num_frames, height, width)
            hint = torch.cat([hint, hint_padding], dim=1)
        elif hint.size(1) > self.config.hint_channels:
            raise ValueError(
                f"Expected hint channels <= {self.config.hint_channels}, but got {hint.size(1)}. "
                "Please check control input channels."
            )

        hint_states = hint
        if self.config.concat_padding_mask:
            if padding_mask is None:
                raise ValueError("`padding_mask` must be provided when `concat_padding_mask=True`.")
            padding_mask = transforms.functional.resize(
                padding_mask, list(hint.shape[-2:]), interpolation=transforms.InterpolationMode.NEAREST
            )
            hint_states = torch.cat(
                [hint, padding_mask.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1
            )
            hidden_states = torch.cat(
                [hidden_states, padding_mask.unsqueeze(2).repeat(batch_size, 1, num_frames, 1, 1)], dim=1
            )

        hint_states = self.patch_embed2(hint_states)
        hint_states = self.input_hint_block(hint_states)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # 2. Generate positional embeddings from pre-patch layout
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.config.extra_pos_embed_type else None
        hidden_states = self.patch_embed(hidden_states)

        # 3. Flatten patchified inputs to sequence
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]
        hint_states = hint_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]

        if isinstance(control_weight, torch.Tensor) and control_weight.ndim >= 2:
            if control_weight.ndim == 4:
                control_weight = control_weight.unsqueeze(1)
            if control_weight.ndim != 5:
                raise ValueError(
                    f"Expected control weight tensor to have shape [B, 1, T, H, W] (or [B, T, H, W]), got {tuple(control_weight.shape)}"
                )
            if control_weight.shape[-2:] != (height, width):
                raise ValueError(
                    f"Expected spatial control weight map shape (*, *, *, {height}, {width}), got {tuple(control_weight.shape)}"
                )
            control_weight = control_weight.permute(0, 2, 1, 3, 4).reshape(-1, control_weight.shape[1], height, width)
            control_weight = F.interpolate(
                control_weight,
                size=(post_patch_height, post_patch_width),
                mode="nearest",
            )
            control_weight = control_weight.reshape(batch_size, num_frames, -1, post_patch_height, post_patch_width)
            control_weight = control_weight.permute(0, 2, 1, 3, 4)
            if control_weight.shape[-3] != num_frames:
                raise ValueError(
                    f"Expected temporal control weight map length {num_frames}, got {control_weight.shape[-3]}"
                )
            if p_t > 1:
                control_weight = control_weight.unflatten(-3, (post_patch_num_frames, p_t)).mean(dim=-3)
            control_weight = control_weight.flatten(-3, -1).transpose(1, 2).type_as(hidden_states)

        # 4. Timestep embeddings
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (batch_size, 1, num_frames, 1, 1), (
                f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            )
            timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
            # We can do this because num_frames == post_patch_num_frames, as p_t is 1
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )  # [BT, C] -> [B, T, 1, 1, C] -> [B, T, H, W, C] -> [B, THW, C]
        else:
            assert False

        # 5. Transformer blocks
        controlnet_block_res_samples = ()
        for block, controlnet_block in zip(self.transformer_blocks, self.controlnet_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    extra_pos_emb=extra_pos_emb,
                    attention_mask=attention_mask,
                )
            if hint_states is not None:
                hidden_states += hint_states
                hint_states = None
            control_feat = controlnet_block(hidden_states)
            hint_val = control_feat * control_weight
            controlnet_block_res_samples = controlnet_block_res_samples + (hint_val,)

        if not return_dict:
            return (controlnet_block_res_samples,)

        return CosmosControlNetOutput(controlnet_block_samples=controlnet_block_res_samples)


class CosmosMultiControlNetModel(ModelMixin):
    r"""
    `CosmosControlNetModel` wrapper class for Multi-CosmosControlNet

    This module is a wrapper for multiple instance of the `CosmosControlNetModel`. The `forward()` API is designed to be compatible wit `CosmosControlNetModel`.

    Args:
        controlnets (`List[CosmosControlNetModel]`):
            Provides additional conditioning to the transformer during the denoising process. You must set multiple
            `CosmosControlNetModel` as a list.
    """

    def __init__(self, controlnets):
        super().__init__()
        self.nets = nn.ModuleList(controlnets)

    def forward(
        self,
        hint: Union[torch.Tensor, List[torch.Tensor]],
        control_weight: Union[float, torch.Tensor, List[float], List[torch.Tensor]],
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fps: Optional[int] = None,
        padding_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if not isinstance(hint, (list, tuple)):
            raise ValueError("For `CosmosMultiControlNetModel`, `hint` must be provided as a list or tuple.")

        if isinstance(control_weight, (float, int)):
            control_weight = [float(control_weight)] * len(self.nets)
        elif isinstance(control_weight, torch.Tensor) and control_weight.ndim == 0:
            control_weight = [float(control_weight)] * len(self.nets)
        elif isinstance(control_weight, torch.Tensor) and control_weight.ndim == 1:
            control_weight = [float(w) for w in control_weight]

        if len(hint) != len(self.nets):
            raise ValueError(f"Expected {len(self.nets)} hints, got {len(hint)}.")
        if not isinstance(control_weight, (list, tuple)) or len(control_weight) != len(self.nets):
            raise ValueError(f"Expected {len(self.nets)} control weights, got {len(control_weight)}.")

        for i, (target_hint, scale, controlnet) in enumerate(zip(hint, control_weight, self.nets)):
            block_samples = controlnet(
                hint=target_hint,
                control_weight=scale,
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                fps=fps,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]

            # merge samples
            if i == 0:
                control_block_samples = block_samples
            else:
                control_block_samples = [
                    control_block_sample + block_sample
                    for control_block_sample, block_sample in zip(control_block_samples, block_samples)
                ]
                control_block_samples = tuple(control_block_samples)

        if not return_dict:
            return (control_block_samples,)

        return CosmosControlNetOutput(controlnet_block_samples=control_block_samples)
