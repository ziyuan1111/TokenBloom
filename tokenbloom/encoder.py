"""ATsplat coarse/fine image encoder using a frozen DINOv2-B/14 backbone."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .attention import Block


_DINO_MEAN = (0.485, 0.456, 0.406)
_DINO_STD = (0.229, 0.224, 0.225)


class ATSplatEncoder(nn.Module):
    """Frozen DINO coarse branch plus lightweight per-view fine branch."""

    def __init__(
        self,
        image_size: tuple[int, int] = (256, 256),
        num_input_views: int = 2,
        dim: int = 768,
        num_heads: int = 12,
        coarse_depth: int = 12,
        fine_depth: int = 2,
        mlp_ratio: float = 4.0,
        dino_weights: str | None = None,
        dino_model: Optional[nn.Module] = None,
        depth_min: float = 0.025,
        depth_max: float = 125.0,
        token_radius_init: float = 0.1,
        token_radius_head_init_std: float = 1e-4,
        token_radius_epsilon: float = 1e-6,
        use_multiscale_anchor_fusion: bool = False,
        multiscale_layers: tuple[int, ...] = (5, 7, 9, 11),
    ) -> None:
        super().__init__()
        if image_size != (256, 256):
            raise ValueError(f"ATsplat RE10K encoder requires 256x256 images, got {image_size}")
        if num_input_views != 2:
            raise ValueError(f"ATsplat RE10K encoder requires two input views, got {num_input_views}")
        if dim != 768 or num_heads != 12:
            raise ValueError("ATsplat encoder requires dim=768 and num_heads=12")
        if coarse_depth != 12 or fine_depth != 2:
            raise ValueError("ATsplat encoder requires 12 coarse and 2 fine SA blocks")
        if not 0 < depth_min < depth_max:
            raise ValueError("depth bounds must satisfy 0 < depth_min < depth_max")
        if token_radius_init <= token_radius_epsilon:
            raise ValueError("token_radius_init must exceed token_radius_epsilon")
        if use_multiscale_anchor_fusion:
            if len(multiscale_layers) != 4:
                raise ValueError("multilayer anchor fusion requires exactly four SA layers")
            if tuple(sorted(set(multiscale_layers))) != multiscale_layers:
                raise ValueError("anchor fusion layers must be unique and increasing")
            if multiscale_layers[0] < 0 or multiscale_layers[-1] >= coarse_depth:
                raise ValueError(
                    f"anchor fusion layers must be in [0,{coarse_depth - 1}]"
                )

        self.image_size = image_size
        self.num_input_views = num_input_views
        self.dim = dim
        self.use_multiscale_anchor_fusion = use_multiscale_anchor_fusion
        self.multiscale_layers = multiscale_layers

        self.dino = dino_model if dino_model is not None else self._load_dino(dino_weights)
        self.dino.requires_grad_(False)
        self.dino.eval()

        self.coarse_rgb_patch = nn.Conv2d(3, dim, kernel_size=16, stride=16)
        self.coarse_fusion = nn.Linear(2 * dim, dim)
        self.coarse_image_norm = nn.LayerNorm(dim)
        self.coarse_ray_embed = nn.Conv2d(6, dim, kernel_size=16, stride=16)
        self.coarse_ray_norm = nn.LayerNorm(dim)
        self.coarse_blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=True,
                    init_values=0.01,
                    fused_attn=True,
                    rope=None,
                )
                for _ in range(coarse_depth)
            ]
        )
        # With selected-layer fusion the final coarse readout is diagnostic only;
        # keeping a trainable norm on that dead-end branch creates unused DDP params.
        self.coarse_encoder_norm = (
            nn.Identity()
            if use_multiscale_anchor_fusion
            else nn.LayerNorm(dim)
        )
        self.anchor_multilayer_norms = (
            nn.ModuleList([nn.LayerNorm(dim) for _ in multiscale_layers])
            if use_multiscale_anchor_fusion
            else nn.ModuleList()
        )
        self.anchor_multilayer_fusion = (
            nn.Linear(len(multiscale_layers) * dim, dim)
            if use_multiscale_anchor_fusion
            else None
        )
        self.anchor_multilayer_fusion_norm = (
            nn.LayerNorm(dim)
            if use_multiscale_anchor_fusion
            else nn.Identity()
        )

        self.fine_patch = nn.Conv2d(3, dim, kernel_size=8, stride=8)
        self.fine_image_norm = nn.LayerNorm(dim)
        self.fine_ray_embed = nn.Conv2d(6, dim, kernel_size=8, stride=8)
        self.fine_ray_norm = nn.LayerNorm(dim)
        self.fine_blocks = nn.ModuleList(
            [
                Block(
                    dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_norm=True,
                    init_values=0.01,
                    fused_attn=True,
                    rope=None,
                )
                for _ in range(fine_depth)
            ]
        )
        self.fine_encoder_norm = nn.LayerNorm(dim)

        self.depth_head_norm = nn.LayerNorm(dim)
        # The two output channels are predicted from the same coarse patch
        # feature: log-depth for ray projection and raw radius for the
        # initial token geometry state.
        self.depth_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )
        with torch.no_grad():
            radius_output = self.depth_head[-1]
            nn.init.trunc_normal_(radius_output.weight[1:2], std=token_radius_head_init_std)
            radius_output.bias[1] = math.log(
                math.expm1(token_radius_init - token_radius_epsilon)
            )
        self.register_buffer(
            "dino_mean", torch.tensor(_DINO_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "dino_std", torch.tensor(_DINO_STD).view(1, 3, 1, 1), persistent=False
        )

    @staticmethod
    def _load_dino(weights: str | None) -> nn.Module:
        try:
            from dinov2.hub.backbones import dinov2_vitb14
        except ModuleNotFoundError as exc:
            local_source = Path(__file__).resolve().parents[2] / "dinov2"
            if local_source.is_dir():
                sys.path.insert(0, str(local_source))
                from dinov2.hub.backbones import dinov2_vitb14
            else:
                raise ModuleNotFoundError(
                    "DINOv2 is required. Install the pinned dinov2 dependency before "
                    "running TokenBloom."
                ) from exc

        package_root = Path(__file__).resolve().parents[1]
        weights_path = (
            Path(weights)
            if weights is not None
            else package_root / "weights/dinov2/dinov2_vitb14_pretrain.pth"
        )
        if not weights_path.is_absolute():
            weights_path = package_root / weights_path
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINOv2-B/14 weights not found: {weights_path}")

        model = dinov2_vitb14(pretrained=False)
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        return model

    def train(self, mode: bool = True) -> ATSplatEncoder:
        super().train(mode)
        self.dino.eval()
        return self

    @staticmethod
    def _intrinsics_matrix(intrinsics: torch.Tensor) -> torch.Tensor:
        if intrinsics.ndim == 4 and intrinsics.shape[-2:] == (3, 3):
            return intrinsics
        if intrinsics.ndim == 3 and intrinsics.shape[-1] == 4:
            fx, fy, cx, cy = intrinsics.unbind(dim=-1)
            matrix = torch.zeros(
                *intrinsics.shape[:2],
                3,
                3,
                dtype=intrinsics.dtype,
                device=intrinsics.device,
            )
            matrix[..., 0, 0] = fx
            matrix[..., 1, 1] = fy
            matrix[..., 0, 2] = cx
            matrix[..., 1, 2] = cy
            matrix[..., 2, 2] = 1
            return matrix
        raise ValueError("intrinsics must be [B,V,4] or [B,V,3,3]")

    @classmethod
    def _full_rays(
        cls,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        matrix = cls._intrinsics_matrix(intrinsics)
        y, x = torch.meshgrid(
            torch.arange(height, device=matrix.device, dtype=matrix.dtype),
            torch.arange(width, device=matrix.device, dtype=matrix.dtype),
            indexing="ij",
        )
        pixels = torch.stack((x + 0.5, y + 0.5, torch.ones_like(x)), dim=-1)
        matrix_inv = torch.linalg.inv(matrix.float()).to(matrix.dtype)
        camera_directions = F.normalize(
            torch.einsum("bvij,hwj->bvhwi", matrix_inv, pixels), dim=-1
        )
        directions = F.normalize(
            torch.einsum("bvij,bvhwj->bvhwi", c2w[..., :3, :3], camera_directions),
            dim=-1,
        )
        origins = c2w[..., :3, 3][:, :, None, None, :].expand_as(directions)
        moments = torch.cross(origins, directions, dim=-1)
        plucker = torch.cat((moments, directions), dim=-1)
        return origins, directions, plucker

    @classmethod
    def _patch_rays(
        cls,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
        grid_size: tuple[int, int],
        patch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_height, grid_width = grid_size
        matrix = cls._intrinsics_matrix(intrinsics)
        y, x = torch.meshgrid(
            torch.arange(grid_height, device=matrix.device, dtype=matrix.dtype),
            torch.arange(grid_width, device=matrix.device, dtype=matrix.dtype),
            indexing="ij",
        )
        pixels = torch.stack(
            (
                (x + 0.5) * patch_size,
                (y + 0.5) * patch_size,
                torch.ones_like(x),
            ),
            dim=-1,
        )
        matrix_inv = torch.linalg.inv(matrix.float()).to(matrix.dtype)
        camera_directions = F.normalize(
            torch.einsum("bvij,hwj->bvhwi", matrix_inv, pixels), dim=-1
        )
        directions = F.normalize(
            torch.einsum("bvij,bvhwj->bvhwi", c2w[..., :3, :3], camera_directions),
            dim=-1,
        )
        origins = c2w[..., :3, 3][:, :, None, None, :].expand_as(directions)
        plucker = torch.cat(
            (torch.cross(origins, directions, dim=-1), directions), dim=-1
        )
        return origins, directions, plucker

    def _dino_features(self, images: torch.Tensor) -> torch.Tensor:
        batch, views = images.shape[:2]
        dino_images = F.interpolate(
            rearrange(images, "b v c h w -> (b v) c h w"),
            size=(224, 224),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        dino_images = (dino_images - self.dino_mean.to(dino_images)) / self.dino_std.to(dino_images)
        self.dino.eval()
        with torch.no_grad():
            patch_tokens = self.dino.forward_features(dino_images)["x_norm_patchtokens"]
        if patch_tokens.shape != (batch * views, 256, 768):
            raise RuntimeError(
                "DINOv2-B/14 patch output must be [B*V,256,768], got "
                f"{tuple(patch_tokens.shape)}"
            )
        return rearrange(patch_tokens, "(b v) n c -> b v n c", b=batch, v=views)

    @staticmethod
    def _positive_depth(raw_log_depth: torch.Tensor) -> torch.Tensor:
        """Interpret the unbounded head output as log-depth and exponentiate in FP32."""
        with torch.autocast(device_type=raw_log_depth.device.type, enabled=False):
            return torch.exp(raw_log_depth.float())

    def forward(
        self,
        images: torch.Tensor,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
    ) -> dict[str, torch.Tensor | tuple[int, int]]:
        if images.ndim != 5 or images.shape[2:] != (3, 256, 256):
            raise ValueError(f"images must be [B,V,3,256,256], got {tuple(images.shape)}")
        batch, views = images.shape[:2]
        if views != self.num_input_views:
            raise ValueError(f"expected {self.num_input_views} views, got {views}")
        if c2w.shape != (batch, views, 4, 4):
            raise ValueError(f"c2w must be {(batch, views, 4, 4)}, got {tuple(c2w.shape)}")

        full_plucker = self._full_rays(
            intrinsics, c2w, self.image_size[0], self.image_size[1]
        )[2]
        full_plucker_image = rearrange(
            full_plucker, "b v h w c -> (b v) c h w"
        )
        fine_patch_plucker = rearrange(
            F.avg_pool2d(full_plucker_image, kernel_size=8, stride=8),
            "(b v) c h w -> b (v h w) c",
            b=batch,
            v=views,
        )

        dino_features = self._dino_features(images)
        image_flat = rearrange(images, "b v c h w -> (b v) c h w")
        normalized_images = (
            image_flat - self.dino_mean.to(image_flat)
        ) / self.dino_std.to(image_flat)
        coarse_rgb = rearrange(
            self.coarse_rgb_patch(normalized_images),
            "(b v) c h w -> b v (h w) c",
            b=batch,
            v=views,
        )
        if coarse_rgb.shape != (batch, views, 256, self.dim):
            raise RuntimeError(f"coarse RGB shape mismatch: {tuple(coarse_rgb.shape)}")
        coarse_origins, coarse_directions, coarse_plucker = self._patch_rays(
            intrinsics, c2w, (16, 16), patch_size=16
        )
        coarse_image = self.coarse_image_norm(
            self.coarse_fusion(torch.cat((dino_features, coarse_rgb), dim=-1))
        )
        coarse_ray = self.coarse_ray_norm(
            rearrange(
                self.coarse_ray_embed(full_plucker_image),
                "(b v) c h w -> b v (h w) c",
                b=batch,
                v=views,
            )
        )
        coarse = coarse_image + coarse_ray
        coarse = rearrange(coarse, "b v n c -> b (v n) c")
        if coarse.shape != (batch, views * 256, self.dim):
            raise RuntimeError(f"coarse global token shape mismatch: {tuple(coarse.shape)}")
        coarse_readouts: list[torch.Tensor] = []
        for layer_index, block in enumerate(self.coarse_blocks):
            coarse = block(coarse)
            if (
                self.use_multiscale_anchor_fusion
                and layer_index in self.multiscale_layers
            ):
                coarse_readouts.append(coarse)
        coarse = self.coarse_encoder_norm(coarse)

        if self.anchor_multilayer_fusion is not None:
            if len(coarse_readouts) != 4:
                raise RuntimeError(
                    f"collected {len(coarse_readouts)} coarse SA readouts, expected 4"
                )
            anchor_features = self.anchor_multilayer_fusion_norm(
                self.anchor_multilayer_fusion(
                    torch.cat(
                        [
                            norm(features)
                            for norm, features in zip(
                                self.anchor_multilayer_norms,
                                coarse_readouts,
                                strict=True,
                            )
                        ],
                        dim=-1,
                    )
                )
            )
        else:
            anchor_features = coarse
        anchor_features_per_view = rearrange(
            anchor_features, "b (v n) c -> b v n c", v=views, n=256
        )

        fine_image = rearrange(
            self.fine_patch(normalized_images),
            "(b v) c h w -> b v (h w) c",
            b=batch,
            v=views,
        )
        fine_image = self.fine_image_norm(fine_image)
        fine_ray = self.fine_ray_norm(
            rearrange(
                self.fine_ray_embed(full_plucker_image),
                "(b v) c h w -> b v (h w) c",
                b=batch,
                v=views,
            )
        )
        fine = fine_image + fine_ray
        fine = rearrange(fine, "b v n c -> (b v) n c")
        if fine.shape != (batch * views, 1024, self.dim):
            raise RuntimeError(f"fine per-view token shape mismatch: {tuple(fine.shape)}")
        for block in self.fine_blocks:
            fine = block(fine)
        fine = self.fine_encoder_norm(fine)
        fine = rearrange(fine, "(b v) n c -> b v n c", b=batch, v=views)

        coarse_map = rearrange(
            anchor_features_per_view,
            "b v (h w) c -> (b v) c h w",
            h=16,
            w=16,
        )
        coarse_up = F.interpolate(
            coarse_map, size=(32, 32), mode="bilinear", align_corners=False
        )
        coarse_up = rearrange(
            coarse_up, "(b v) c h w -> b v (h w) c", b=batch, v=views
        )
        if coarse_up.shape != fine.shape:
            raise RuntimeError(
                f"coarse/fine fusion shape mismatch: {tuple(coarse_up.shape)} vs {tuple(fine.shape)}"
            )
        fine_fused = fine + coarse_up

        initial_geometry = self.depth_head(self.depth_head_norm(anchor_features))
        raw_log_depth = initial_geometry[..., 0]
        initial_token_raw_radii = initial_geometry[..., 1:2]
        depths = self._positive_depth(raw_log_depth)
        with torch.autocast(device_type=raw_log_depth.device.type, enabled=False):
            ray_origins = rearrange(
                coarse_origins, "b v h w c -> b (v h w) c"
            ).float()
            ray_directions = F.normalize(
                rearrange(coarse_directions, "b v h w c -> b (v h w) c").float(),
                dim=-1,
            )
            positions = ray_origins + depths.unsqueeze(-1) * ray_directions
        valid_mask = torch.ones(
            batch, views * 256, dtype=torch.bool, device=images.device
        )

        return {
            "coarse_features": coarse,
            "fine_branch_features": rearrange(fine, "b v n c -> b (v n) c"),
            "fine_features": rearrange(fine_fused, "b v n c -> b (v n) c"),
            "fine_ca_features": rearrange(fine_fused, "b v n c -> b (v n) c"),
            "fine_ca_rays": fine_patch_plucker,
            "raw_log_depths": raw_log_depth.unsqueeze(-1),
            "fine_depths": depths.unsqueeze(-1),
            "initial_token_raw_radii": initial_token_raw_radii,
            "coarse_rays": rearrange(
                coarse_plucker, "b v h w c -> b (v h w) c"
            ),
            "token_positions": positions,
            "initial_token_features": anchor_features,
            "initial_token_positions": positions,
            "initial_token_valid_mask": valid_mask,
            "ray_origins": ray_origins,
            "ray_directions": ray_directions,
            "grid_size": (16, 16),
        }
