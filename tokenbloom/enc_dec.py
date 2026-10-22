# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import Attention, Block, LayerScale, Mlp
from typing import Any as Options


class DecoderBlock(nn.Module):
    """
    Decoder block for encoder-decoder architecture.
    Works like a transformer decoder layer except the keys/values are provided by the encoder (already normalized).
    """

    class SelfAttnBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool,
            qk_norm: bool,
            flex_attn_score_mod=None,
        ):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.gs_self_attn = Attention(
                dim,
                num_heads,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                fused_attn=True,
                flex_attn_score_mod=flex_attn_score_mod,
            )
            # The Fourier XYZ encoder is shared by ATSplat. These projections
            # are deliberately local to a self-attention block, so position
            # enters Q/K but never V or the persistent content state.
            self.q_pos_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_pos_proj = nn.Linear(dim, dim, bias=qkv_bias)

        def forward(
            self,
            gs_tokens: torch.Tensor,
            token_mask: torch.Tensor | None = None,
            self_attn_position: torch.Tensor | None = None,
        ) -> torch.Tensor:
            queries_normed = self.norm(gs_tokens)
            attention_mask = None if token_mask is None else token_mask[:, None, None, :]
            q_pos = k_pos = None
            if self_attn_position is not None:
                if self_attn_position.shape != gs_tokens.shape:
                    raise ValueError(
                        "self_attn_position must match gs_tokens shape: "
                        f"{tuple(self_attn_position.shape)} vs {tuple(gs_tokens.shape)}"
                    )
                q_pos = self.q_pos_proj(self_attn_position.to(queries_normed.dtype))
                k_pos = self.k_pos_proj(self_attn_position.to(queries_normed.dtype))
            output = self.gs_self_attn(
                queries_normed,
                attn_mask=attention_mask,
                q_pos=q_pos,
                k_pos=k_pos,
            )
            if token_mask is not None:
                output = output * token_mask.unsqueeze(-1).to(output.dtype)
            return output

    class CrossAttnBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            qkv_bias: bool,
            q_norm: bool,
            geometric_bias: bool = False,
            geo_sigma0: float = 0.1,
            geo_bias_min: float = -20.0,
            geo_bandwidth2_min: float = 1e-6,
            geo_gamma_init: float = 0.1,
        ):
            super().__init__()
            self.num_heads = num_heads
            self.gs_token_norm = nn.LayerNorm(dim)
            self.q_norm = nn.LayerNorm(dim // num_heads) if q_norm else nn.Identity()
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.out_proj = nn.Linear(dim, dim)
            self.geometric_bias = bool(geometric_bias)
            self.geo_sigma0 = float(geo_sigma0)
            self.geo_bias_min = float(geo_bias_min)
            self.geo_bandwidth2_min = float(geo_bandwidth2_min)
            if self.geometric_bias:
                self.gamma_raw = nn.Parameter(
                    torch.tensor(math.log(math.expm1(float(geo_gamma_init))))
                )
            else:
                self.register_parameter("gamma_raw", None)

        def _anchor_to_ray_bias(
            self,
            anchor_positions: torch.Tensor,
            anchor_radii: torch.Tensor,
            image_rays: torch.Tensor,
        ) -> torch.Tensor:
            """Return point-to-Plucker-ray bias with shape [B,N,M]."""
            if anchor_positions.ndim != 3 or anchor_positions.shape[-1] != 3:
                raise ValueError("anchor_positions must have shape [B,N,3]")
            if anchor_radii.shape != (*anchor_positions.shape[:2], 1):
                raise ValueError("anchor_radii must have shape [B,N,1]")
            if image_rays.ndim != 3 or image_rays.shape[-1] != 6:
                raise ValueError("image_rays must have shape [B,M,6]")
            if image_rays.shape[0] != anchor_positions.shape[0]:
                raise ValueError("anchor and ray batch dimensions must match")

            positions = anchor_positions.float()
            moments = image_rays[..., :3].float()
            directions = image_rays[..., 3:].float()
            direction_norm = torch.linalg.vector_norm(
                directions, dim=-1, keepdim=True
            ).clamp_min(torch.finfo(torch.float32).eps)
            directions = directions / direction_norm
            moments = moments / direction_norm
            displacement = (
                torch.cross(
                    positions[:, :, None, :],
                    directions[:, None, :, :],
                    dim=-1,
                )
                - moments[:, None, :, :]
            )
            distance_squared = displacement.square().sum(dim=-1)
            bandwidth_squared = (
                self.geo_sigma0 * anchor_radii.float()
            ).square().clamp_min(self.geo_bandwidth2_min)
            return (-0.5 * distance_squared / bandwidth_squared).clamp(
                self.geo_bias_min, 0.0
            )

        def forward(
            self,
            gs_tokens: torch.Tensor,
            keys: torch.Tensor,
            values: torch.Tensor,
            anchor_positions: torch.Tensor | None = None,
            anchor_radii: torch.Tensor | None = None,
            image_rays: torch.Tensor | None = None,
        ) -> torch.Tensor:
            gs_tokens_normed = self.gs_token_norm(gs_tokens)
            q = rearrange(self.q_proj(gs_tokens_normed), "b n (h d) -> b h n d", h=self.num_heads)
            q = self.q_norm(q)

            attention_bias = None
            if self.geometric_bias:
                if (
                    anchor_positions is None
                    or anchor_radii is None
                    or image_rays is None
                ):
                    raise ValueError(
                        "geometric cross-attention requires anchor positions, "
                        "anchor radii, and image rays"
                    )
                geo_bias = self._anchor_to_ray_bias(
                    anchor_positions, anchor_radii, image_rays
                )
                gamma = F.softplus(self.gamma_raw)
                attention_bias = (gamma * geo_bias)[:, None].to(q.dtype)

            cross_attn_output = F.scaled_dot_product_attention(
                q, keys, values, attn_mask=attention_bias
            )
            cross_attn_output = rearrange(cross_attn_output, "b h n d -> b n (h d)")
            return self.out_proj(cross_attn_output)

    class MlpBlock(nn.Module):
        def __init__(self, dim: int, mlp_ratio: float, ffn_bias: bool):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.mlp = Mlp(dim, int(dim * mlp_ratio), dim, bias=ffn_bias)

        def forward(self, gs_tokens: torch.Tensor) -> torch.Tensor:
            return self.mlp(self.norm(gs_tokens))

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        ffn_bias: bool,
        qk_norm: bool,
        init_values: float | None = None,
        attn_score_mod=None,
        cross_attn_geometric_bias: bool = False,
        cross_attn_geo_sigma0: float = 0.1,
        cross_attn_geo_bias_min: float = -20.0,
        cross_attn_geo_bandwidth2_min: float = 1e-6,
        cross_attn_geo_gamma_init: float = 0.1,
    ):
        super().__init__()

        def make_scale() -> nn.Module:
            return LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.gs_self_attn = DecoderBlock.SelfAttnBlock(
            dim,
            num_heads,
            qkv_bias,
            qk_norm=qk_norm,
            flex_attn_score_mod=attn_score_mod,
        )
        self.gs_self_attn_scale = make_scale()

        self.gs_cross_attn = DecoderBlock.CrossAttnBlock(
            dim,
            num_heads,
            qkv_bias,
            q_norm=qk_norm,
            geometric_bias=cross_attn_geometric_bias,
            geo_sigma0=cross_attn_geo_sigma0,
            geo_bias_min=cross_attn_geo_bias_min,
            geo_bandwidth2_min=cross_attn_geo_bandwidth2_min,
            geo_gamma_init=cross_attn_geo_gamma_init,
        )
        self.gs_cross_attn_scale = make_scale()
        self.mlp = DecoderBlock.MlpBlock(dim, mlp_ratio, ffn_bias)
        self.mlp_scale = make_scale()

    def forward(
        self,
        gs_tokens: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        self_attn_position: torch.Tensor | None = None,
        cross_attn_anchor_positions: torch.Tensor | None = None,
        cross_attn_anchor_radii: torch.Tensor | None = None,
        cross_attn_image_rays: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        intermediates: dict[str, torch.Tensor] = {}
        mask = None if token_mask is None else token_mask.unsqueeze(-1).to(gs_tokens.dtype)
        if (
            cross_attn_anchor_positions is None
            and cross_attn_anchor_radii is None
            and cross_attn_image_rays is None
        ):
            cross_attn_output = self.gs_cross_attn(gs_tokens, keys, values)
        else:
            cross_attn_output = self.gs_cross_attn(
                gs_tokens,
                keys,
                values,
                anchor_positions=cross_attn_anchor_positions,
                anchor_radii=cross_attn_anchor_radii,
                image_rays=cross_attn_image_rays,
            )
        gs_tokens = gs_tokens + self.gs_cross_attn_scale(cross_attn_output)
        if mask is not None:
            gs_tokens = gs_tokens * mask
        if return_intermediates:
            intermediates["after_ca"] = gs_tokens
        if return_intermediates:
            # Position is passed separately to Q/K below; token content has
            # not been modified by absolute XYZ PE.
            intermediates["before_sa"] = gs_tokens
        gs_tokens = gs_tokens + self.gs_self_attn_scale(
            self.gs_self_attn(
                gs_tokens,
                token_mask=token_mask,
                self_attn_position=self_attn_position,
            )
        )
        if return_intermediates:
            intermediates["after_sa"] = gs_tokens
        gs_tokens = gs_tokens + self.mlp_scale(self.mlp(gs_tokens))
        if mask is not None:
            gs_tokens = gs_tokens * mask

        if return_intermediates:
            intermediates["after_ffn"] = gs_tokens
            return gs_tokens, intermediates
        return gs_tokens

class EncDecBackbone(nn.Module):
    """
    An encoder-decoder backbone for the EncDec architecture.

    Encoder: a stack of ViT blocks which produce a latent representation. This is followed by a key-value projection to produce a key and value for the decoder.

    Decoder: a stack of transformer decoder layers which attend from GS tokens to the encoder output and among themselves.
    """

    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        self.encoder = nn.Sequential(
            *[
                Block(
                    self.opt.enc_embed_dim,
                    self.opt.enc_num_heads,
                    self.opt.mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    init_values=0.01,
                    qk_norm=True,
                    rope=None,
                    flex_attn_block_mask=None,
                )
                for _ in range(self.opt.enc_depth)
            ]
        )

        self.use_multiscale = self.opt.use_multiscale_encoder
        self.use_latent_bottleneck = self.opt.use_latent_bottleneck

        if self.use_multiscale:
            self.multiscale_layers = self.opt.multiscale_encoder_layers
            self.multiscale_norms = nn.ModuleList(
                [nn.LayerNorm(self.opt.enc_embed_dim) for _ in self.multiscale_layers]
            )
            encoder_feature_dim = self.opt.enc_embed_dim * len(self.multiscale_layers)
        else:
            self.encoder_norm = nn.LayerNorm(self.opt.enc_embed_dim)
            encoder_feature_dim = self.opt.enc_embed_dim

        if not self.use_latent_bottleneck:
            self.kv_proj = nn.Linear(
                encoder_feature_dim, self.opt.enc_embed_dim * 2, bias=True
            )

        self.k_proj_norm = nn.LayerNorm(self.opt.enc_embed_dim // self.opt.enc_num_heads)

        if self.use_latent_bottleneck:
            self.latents = nn.Parameter(
                torch.randn(self.opt.num_latents, self.opt.enc_embed_dim) * 0.02
            )
            self.latents._no_weight_decay = True

            self.latent_feature_kv_proj = nn.Linear(
                encoder_feature_dim, self.opt.enc_embed_dim * 2, bias=True
            )
            self.latent_feature_k_norm = nn.LayerNorm(
                self.opt.enc_embed_dim // self.opt.enc_num_heads
            )
            self.latent_blocks = nn.ModuleList(
                [
                    DecoderBlock(
                        self.opt.enc_embed_dim,
                        self.opt.enc_num_heads,
                        self.opt.mlp_ratio,
                        qkv_bias=True,
                        ffn_bias=True,
                        init_values=0.01,
                        qk_norm=True,
                    )
                    for _ in range(self.opt.latent_cross_attn_depth)
                ]
            )
            self.latent_kv_proj = nn.Linear(
                self.opt.enc_embed_dim, self.opt.enc_embed_dim * 2, bias=True
            )

        if self.opt.num_dynamic_gs_tokens > 0:
            block_ids = [0] * self.opt.num_gs_tokens + [1] * self.opt.num_dynamic_gs_tokens
            self.register_buffer(
                "_decoder_block_ids",
                torch.tensor(block_ids, dtype=torch.long),
                persistent=False,
            )

            def block_causal_score_mod(score, b, h, q_idx, kv_idx):
                block_ids_tensor = self._decoder_block_ids
                same_block_mask = block_ids_tensor[q_idx] == block_ids_tensor[kv_idx]
                causal_mask = q_idx >= kv_idx
                return torch.where(same_block_mask | causal_mask, score, float("-inf"))

            score_mod = block_causal_score_mod
        else:
            score_mod = None

        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    self.opt.enc_embed_dim,
                    self.opt.enc_num_heads,
                    self.opt.mlp_ratio,
                    qkv_bias=True,
                    ffn_bias=True,
                    init_values=(
                        self.opt.dec_init_values
                        if self.opt.dec_init_values is not None
                        else 5e-3 * self.opt.gs_token_std
                    ),
                    qk_norm=True,
                    attn_score_mod=score_mod,
                )
                for _ in range(self.opt.dec_depth)
            ]
        )

    def _encode_features(self, image_features: torch.Tensor) -> torch.Tensor:
        if self.use_multiscale:
            return torch.cat(self._encode_feature_layers(image_features), dim=-1)
        image_features = self.encoder(image_features)
        return self.encoder_norm(image_features)

    def _encode_feature_layers(
        self,
        image_features: torch.Tensor,
        *,
        normalize: bool = True,
    ) -> list[torch.Tensor]:
        """Return the normalized intermediate readouts used by multiscale fusion."""
        if not self.use_multiscale:
            raise ValueError("intermediate feature readout requires use_multiscale_encoder")
        x = image_features
        features: list[torch.Tensor] = []
        for layer_index, block in enumerate(self.encoder):
            x = block(x)
            if layer_index in self.multiscale_layers:
                features.append(
                    self.multiscale_norms[len(features)](x) if normalize else x
                )
        if len(features) != len(self.multiscale_layers):
            raise RuntimeError(
                f"collected {len(features)} encoder layers, expected {len(self.multiscale_layers)}"
            )
        return features

    def _features_to_kv(
        self, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_feature_keys, image_feature_values = rearrange(
            self.kv_proj(image_features),
            "b n (kv h c) -> kv b h n c",
            kv=2,
            h=self.opt.enc_num_heads,
        )
        image_feature_keys = self.k_proj_norm(image_feature_keys)
        return image_feature_keys, image_feature_values

    def _features_to_scene_latents(
        self, image_features: torch.Tensor
    ) -> torch.Tensor:
        feat_keys, feat_values = rearrange(
            self.latent_feature_kv_proj(image_features),
            "b n (kv h c) -> kv b h n c",
            kv=2,
            h=self.opt.enc_num_heads,
        )
        feat_keys = self.latent_feature_k_norm(feat_keys)

        latents = self.latents.unsqueeze(0).expand(image_features.shape[0], -1, -1)
        for block in self.latent_blocks:
            latents = block(gs_tokens=latents, keys=feat_keys, values=feat_values)

        return latents

    def _latents_to_kv(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent_keys, latent_values = rearrange(
            self.latent_kv_proj(latents),
            "b n (kv h c) -> kv b h n c",
            kv=2,
            h=self.opt.enc_num_heads,
        )
        latent_keys = self.k_proj_norm(latent_keys)
        return latent_keys, latent_values

    def _features_to_kv_via_latents(
        self, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self._features_to_scene_latents(image_features)
        return self._latents_to_kv(latents)

    def _encode_to_kv(
        self, image_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self._encode_features(image_features)
        if self.use_latent_bottleneck:
            return self._features_to_kv_via_latents(image_features)
        return self._features_to_kv(image_features)

