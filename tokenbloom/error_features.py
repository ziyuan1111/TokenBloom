"""Full-resolution ReSplat feature errors and exact Gaussian attribution."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18



class FrozenFullResolutionResNet18Error(nn.Module):
    """Return full-resolution ``render - target`` ResNet-18 residuals.

    This follows ReSplat's feature extractor exactly through ``layer2``:
    ``conv1`` (64), ``layer1`` (64), and ``layer2`` (128) are aligned and
    concatenated.  The only resolution choice made here is to align them to
    the original render grid, as required by the Gaussian rasterization VJP.
    """

    output_channels = 64 + 64 + 128
    feature_level_channels = (64, 64, 128)

    def __init__(self, backbone: nn.Module | None = None) -> None:
        super().__init__()
        if backbone is None:
            local_weights = os.environ.get("TOKENBLOOM_RESNET_WEIGHTS")
            if local_weights:
                weights_path = Path(local_weights)
                if not weights_path.is_file():
                    raise FileNotFoundError(
                        f"ResNet-18 weights not found: {weights_path}"
                    )
                backbone = resnet18(weights=None)
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                backbone.load_state_dict(state, strict=True)
            else:
                backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1),
            persistent=False,
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True) -> FrozenFullResolutionResNet18Error:
        """Keep the frozen backbone and its BatchNorm statistics in eval mode."""
        super().train(False)
        return self

    @classmethod
    def channels_for_levels(cls, num_feature_levels: int) -> int:
        if not 1 <= num_feature_levels <= len(cls.feature_level_channels):
            raise ValueError("num_feature_levels must be in [1, 3]")
        return sum(cls.feature_level_channels[:num_feature_levels])

    def _extract(self, images: torch.Tensor, num_feature_levels: int = 3) -> torch.Tensor:
        height, width = images.shape[-2:]
        x = (images - self.image_mean.to(images)) / self.image_std.to(images)
        conv1 = self.conv1(x)
        features = [conv1]
        if num_feature_levels >= 2:
            x = self.maxpool(self.relu(self.bn1(conv1)))
            layer1 = self.layer1(x)
            features.append(layer1)
            if num_feature_levels >= 3:
                features.append(self.layer2(layer1))
        aligned = []
        for feature in features:
            if feature.shape[-2:] != (height, width):
                feature = F.interpolate(
                    feature,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=True,
                )
            aligned.append(feature)
        return torch.cat(aligned, dim=1)

    def forward(
        self,
        rendered_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        target_features: torch.Tensor | None = None,
        num_feature_levels: int = 3,
    ) -> torch.Tensor:
        if rendered_rgb.shape != target_rgb.shape:
            raise ValueError("rendered_rgb and target_rgb must have identical shapes")
        if rendered_rgb.ndim != 5 or rendered_rgb.shape[2] != 3:
            raise ValueError("RGB tensors must have shape [B,V,3,H,W]")
        rendered_features = self.extract_features(
            rendered_rgb, num_feature_levels=num_feature_levels
        )
        if target_features is None:
            target_features = self.extract_features(
                target_rgb, num_feature_levels=num_feature_levels
            )
        elif target_features.shape[2] == self.output_channels:
            target_features = target_features[:, :, : rendered_features.shape[2]]
        if target_features.shape != rendered_features.shape:
            raise ValueError(
                "target_features must have shape "
                f"{tuple(rendered_features.shape)}"
            )
        return (rendered_features - target_features.detach().float()).detach()

    def extract_features(
        self,
        images: torch.Tensor,
        num_feature_levels: int = 3,
    ) -> torch.Tensor:
        """Extract reusable full-resolution features for a batch of views."""
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [B,V,3,H,W]")
        batch, views, _, height, width = images.shape
        channels = self.channels_for_levels(num_feature_levels)
        flat = images.detach().reshape(batch * views, 3, height, width)
        with torch.no_grad(), torch.autocast(
            device_type=flat.device.type,
            enabled=False,
        ):
            features = self._extract(flat.float(), num_feature_levels=num_feature_levels)
        return features.reshape(
            batch, views, channels, height, width
        ).detach()


@dataclass
class FullResolutionGaussianErrorAssignment:
    """Detached full-resolution error observations attributed to Gaussians."""
