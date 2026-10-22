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
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.init import trunc_normal_

from typing import Any as Options


class BaseActivationHead(nn.Module, ABC):
    """
    Base class for converting tokens to Gaussians.

    This is a minimal interface - subclasses can have completely different
    internal structures and initialization logic.
    """

    def __init__(self, opt: Options):
        super().__init__()
        self.opt = opt

        self.num_gaussians_per_token = self.opt.num_gaussians_per_token

        scale_cap = self.opt.gaussian_scale_cap
        self.scale_shift = 1 - math.log(scale_cap)
        self.scale_cap = scale_cap

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        rays_os: Optional[torch.Tensor] = None,
        rays_ds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert tokens to Gaussians.

        Args:
            x: Input tokens [B, N, C] where N is number of tokens
            rays_os: Optional ray origins [B, V, 3, H, W] for depth prediction
            rays_ds: Optional ray directions [B, V, 3, H, W] for depth prediction

        Returns:
            Gaussians [B, N * num_gaussians_per_token, 14]
        """
        pass


class ClipActivationHead(BaseActivationHead):
    """
    Clip-based activation head with a configurable one- or two-layer readout.
    Supports both xyz and depth prediction modes.
    """

    def __init__(self, opt: Options):
        super().__init__(opt)

        # Set output dimensions based on prediction mode
        self.output_dims = (
            3 + 1 + 3 + 4 + 3
        )  # x, y, z, opacity, scale, rotation, rgb

        if self.opt.gaussian_head_num_layers == 2:
            self.pre_deconv = nn.Sequential(
                nn.Linear(self.opt.enc_embed_dim, self.opt.enc_embed_dim),
                nn.GELU(),
            )
            self.pre_deconv.apply(self._init_weights)
        elif self.opt.gaussian_head_num_layers == 1:
            self.pre_deconv = nn.Identity()
        else:
            raise ValueError(
                "gaussian_head_num_layers must be either 1 or 2, got "
                f"{self.opt.gaussian_head_num_layers}"
            )

        # Final shared readout for all Gaussian components.
        self.deconv = nn.Linear(
            self.opt.enc_embed_dim,
            self.output_dims * self.num_gaussians_per_token,
            bias=True,
        )

        self._init_weights(self.deconv)
        if self.opt.clip_head_z_init is not None:
            bias_val = math.log(1.0 + self.opt.clip_head_z_init)
            with torch.no_grad():
                self.deconv.bias.view(
                    self.num_gaussians_per_token,
                    self.output_dims,
                )[:, 2].fill_(bias_val)

        # Mark parameters to exclude from weight decay
        for param in self.parameters():
            param._no_weight_decay = True

    def _init_weights(self, m):
        """Initialize deconv weights"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.opt.clip_head_readout_std)
            if m.bias is not None:
                m.bias.data.zero_()

    def pos_act(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw position logits to Gaussian offsets."""
        if self.opt.gaussian_position_activation == "linear":
            return x
        if self.opt.gaussian_position_activation == "signed_exp":
            return torch.sign(x) * torch.expm1(torch.abs(x))
        raise ValueError(
            f"Unsupported Gaussian position activation: "
            f"{self.opt.gaussian_position_activation}"
        )

    def scale_act(self, x: torch.Tensor) -> torch.Tensor:
        """Convert raw scale logits to positive Gaussian axis scales."""
        if self.opt.gaussian_scale_parameterization == "ilrm":
            log_scale = torch.clamp(
                x - self.opt.gaussian_scale_log_shift,
                max=self.opt.gaussian_scale_log_max,
            )
            return torch.exp(log_scale)
        if self.opt.gaussian_scale_parameterization == "relative_cap":
            log_scale = x - self.scale_shift
            log_scale_cap = math.log(self.scale_cap)
            return torch.exp(torch.clamp(log_scale, max=log_scale_cap))
        raise ValueError(
            f"Unsupported Gaussian scale parameterization: "
            f"{self.opt.gaussian_scale_parameterization}"
        )

    def opacity_act(self, x: torch.Tensor) -> torch.Tensor:
        """Opacity activation with hard clipping"""
        return torch.sigmoid(x - self.opt.opacity_bias)

    def rot_act(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation normalization"""
        return F.normalize(x, dim=-1)

    def rgb_act(self, x: torch.Tensor) -> torch.Tensor:
        """Parameterize direct RGB or raw degree-0 SH/DC coefficients."""
        if self.opt.gaussian_color_parameterization == "sh0":
            return x
        if self.opt.gaussian_color_parameterization == "rgb":
            return 0.5 * torch.tanh(x) + 0.5
        raise ValueError(
            f"Unsupported Gaussian color parameterization: "
            f"{self.opt.gaussian_color_parameterization}"
        )

    def forward(
        self,
        x: torch.Tensor,
        rays_os: Optional[torch.Tensor] = None,
        rays_ds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Convert tokens to Gaussians using the configured readout MLP.
        Supports both xyz and depth prediction modes.

        Args:
            x: Input tokens [B, N, C]
            rays_os: Ray origins [B, V, 3, H, W] (required for depth prediction)
            rays_ds: Ray directions [B, V, 3, H, W] (required for depth prediction)

        Returns:
            Gaussians [B, N * num_gaussians_per_token, 14]
        """
        B = x.shape[0]

        # Process through deconv
        x = x.permute(0, 2, 1).unsqueeze(-1)  # B, C, N, 1

        x = x.squeeze(-1).permute(0, 2, 1)  # B, N, C
        x = self.deconv(self.pre_deconv(x))  # [B, N, output_dims * P * P]
        x = rearrange(
            x,
            "b n (p c) -> b c (n p)",
            p=self.num_gaussians_per_token,
        )

        x = x.reshape(B, self.output_dims, -1)  # B, output_dims, N * P * P
        x = x.permute(0, 2, 1).contiguous()  # B, N * P * P, output_dims

        # Apply activations based on prediction mode
        pos, rgb, scaling, rotation, opacity = x.split([3, 3, 3, 4, 1], dim=-1)
        pos = self.pos_act(pos)
        

        opacity = self.opacity_act(opacity)
        scale = self.scale_act(scaling)
        rotation = self.rot_act(rotation)
        rgbs = self.rgb_act(rgb)

        gaussians = torch.cat(
            [pos, opacity, scale, rotation, rgbs], dim=-1
        )  # [B, N, 14]

        return gaussians

