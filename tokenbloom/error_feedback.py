"""Sparse 3D token sampling for full-resolution error feedback."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .error_features import FrozenFullResolutionResNet18Error
from .token_point_transformer import TokenPlainPointTransformer


@dataclass
class TokenProjectionSamples:
    """Detached per-point observations sampled from all feedback views."""

    point_error: torch.Tensor  # [B,V,N,14,C]
    point_valid: torch.Tensor  # [B,V,N,14]
    local_delta: torch.Tensor  # [B,N,14,3]
    sample_xyz: torch.Tensor  # [B,N,14,3]
    topk_indices: torch.Tensor  # [B,N,8]


class TokenProjectionErrorSampler(nn.Module):
    """Project six radius points and opacity Top-K Gaussian centers."""

    def __init__(
        self,
        num_gaussians_per_token: int,
        num_adaptive_points: int = 8,
        regular_point_scale: float = 0.7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if num_gaussians_per_token < num_adaptive_points:
            raise ValueError(
                "num_gaussians_per_token must be at least num_adaptive_points"
            )
        if num_adaptive_points <= 0:
            raise ValueError("num_adaptive_points must be positive")
        if regular_point_scale <= 0:
            raise ValueError("regular_point_scale must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.num_gaussians_per_token = int(num_gaussians_per_token)
        self.num_adaptive_points = int(num_adaptive_points)
        self.regular_point_scale = float(regular_point_scale)
        self.eps = float(eps)
        self.register_buffer(
            "axis_directions",
            torch.tensor(
                (
                    (1.0, 0.0, 0.0),
                    (-1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, -1.0, 0.0),
                    (0.0, 0.0, 1.0),
                    (0.0, 0.0, -1.0),
                )
            ),
            persistent=False,
        )

    @staticmethod
    def _camera_coordinates(
        points: torch.Tensor,
        cam_view: torch.Tensor,
    ) -> torch.Tensor:
        ones = torch.ones_like(points[..., :1])
        homogeneous = torch.cat((points.float(), ones.float()), dim=-1)
        return torch.einsum(
            "bnc,bvcd->bvnd", homogeneous, cam_view.float()
        )[..., :3]

    @staticmethod
    def _intrinsic_parameters(
        intrinsics: torch.Tensor,
        expected_shape: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if intrinsics.ndim == 3 and intrinsics.shape[-1] == 4:
            fx, fy, cx, cy = intrinsics.float().unbind(dim=-1)
        elif intrinsics.ndim == 4 and intrinsics.shape[-2:] == (3, 3):
            matrix = intrinsics.float()
            fx = matrix[..., 0, 0]
            fy = matrix[..., 1, 1]
            cx = matrix[..., 0, 2]
            cy = matrix[..., 1, 2]
        else:
            raise ValueError(
                "intrinsics must have shape [B,V,4] or [B,V,3,3]"
            )
        if fx.shape != expected_shape:
            raise ValueError(
                f"intrinsics batch/view shape must be {expected_shape}"
            )
        return fx, fy, cx, cy

    def construct_points(
        self,
        stage_gaussians: torch.Tensor,
        token_xyz: torch.Tensor,
        token_radii: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if token_xyz.ndim != 3 or token_xyz.shape[-1] != 3:
            raise ValueError("token_xyz must have shape [B,N,3]")
        if token_radii.shape != (*token_xyz.shape[:2], 1):
            raise ValueError("token_radii must have shape [B,N,1]")
        batch, num_tokens = token_xyz.shape[:2]
        expected_gaussians = num_tokens * self.num_gaussians_per_token
        if (
            stage_gaussians.ndim != 3
            or stage_gaussians.shape[0] != batch
            or stage_gaussians.shape[1] != expected_gaussians
            or stage_gaussians.shape[-1] < 4
        ):
            raise ValueError(
                "stage_gaussians must have shape [B,N*num_gaussians_per_token,D>=4]"
            )

        center = token_xyz.detach()
        radius = token_radii.detach()
        directions = self.axis_directions.to(center)
        regular_points = center[:, :, None, :] + (
            self.regular_point_scale
            * radius[:, :, None, :]
            * directions[None, None]
        )

        grouped = stage_gaussians.detach().reshape(
            batch,
            num_tokens,
            self.num_gaussians_per_token,
            stage_gaussians.shape[-1],
        )
        topk_indices = grouped[..., 3].topk(
            self.num_adaptive_points,
            dim=2,
            largest=True,
            sorted=True,
        ).indices
        gather_index = topk_indices[..., None].expand(-1, -1, -1, 3)
        adaptive_points = torch.gather(
            grouped[..., :3], dim=2, index=gather_index
        )
        sample_xyz = torch.cat((regular_points, adaptive_points), dim=2)
        local_delta = (
            sample_xyz - center[:, :, None, :]
        ) / radius[:, :, None, :].clamp_min(self.eps)
        return sample_xyz.detach(), local_delta.detach(), topk_indices.detach()

    def forward(
        self,
        stage_gaussians: torch.Tensor,
        token_xyz: torch.Tensor,
        token_radii: torch.Tensor,
        token_valid_mask: torch.Tensor,
        error_map: torch.Tensor,
        cam_view: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> TokenProjectionSamples:
        sample_xyz, local_delta, topk_indices = self.construct_points(
            stage_gaussians,
            token_xyz,
            token_radii,
        )
        batch, num_tokens, num_points = sample_xyz.shape[:3]
        if token_valid_mask.shape != (batch, num_tokens):
            raise ValueError("token_valid_mask must have shape [B,N]")
        if error_map.ndim != 5 or error_map.shape[0] != batch:
            raise ValueError("error_map must have shape [B,V,C,H,W]")
        views, channels, height, width = error_map.shape[1:]
        if cam_view.shape != (batch, views, 4, 4):
            raise ValueError("cam_view must have shape [B,V,4,4]")
        fx, fy, cx, cy = self._intrinsic_parameters(
            intrinsics, (batch, views)
        )

        flattened_points = sample_xyz.reshape(batch, num_tokens * num_points, 3)
        camera = self._camera_coordinates(flattened_points, cam_view)
        z = camera[..., 2]
        safe_z = torch.where(
            z > 0,
            z.clamp_min(self.eps),
            torch.ones_like(z),
        )
        u = fx[..., None] * camera[..., 0] / safe_z + cx[..., None]
        v = fy[..., None] * camera[..., 1] / safe_z + cy[..., None]
        uv = torch.stack((u, v), dim=-1).reshape(
            batch, views, num_tokens, num_points, 2
        )
        depth = z.reshape(batch, views, num_tokens, num_points)
        finite = torch.isfinite(uv).all(dim=-1) & torch.isfinite(depth)
        point_valid = (
            finite
            & (depth > 0)
            & (uv[..., 0] >= 0)
            & (uv[..., 0] <= width - 1)
            & (uv[..., 1] >= 0)
            & (uv[..., 1] <= height - 1)
            & token_valid_mask[:, None, :, None].bool()
        )

        safe_uv = torch.where(
            finite[..., None], uv, torch.zeros_like(uv)
        )
        if width > 1:
            grid_x = 2.0 * safe_uv[..., 0] / (width - 1) - 1.0
        else:
            grid_x = torch.zeros_like(safe_uv[..., 0])
        if height > 1:
            grid_y = 2.0 * safe_uv[..., 1] / (height - 1) - 1.0
        else:
            grid_y = torch.zeros_like(safe_uv[..., 1])
        grid = torch.stack((grid_x, grid_y), dim=-1)
        flat_error = error_map.detach().reshape(
            batch * views, channels, height, width
        )
        flat_grid = grid.reshape(batch * views, num_tokens, num_points, 2)
        with torch.autocast(device_type=error_map.device.type, enabled=False):
            sampled = F.grid_sample(
                flat_error.float(),
                flat_grid.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
        point_error = sampled.permute(0, 2, 3, 1).reshape(
            batch, views, num_tokens, num_points, channels
        )
        point_error = point_error.to(error_map.dtype)
        point_error = point_error * point_valid[..., None].to(point_error.dtype)
        return TokenProjectionSamples(
            point_error=point_error.detach(),
            point_valid=point_valid.detach(),
            local_delta=local_delta.detach(),
            sample_xyz=sample_xyz.detach(),
            topk_indices=topk_indices.detach(),
        )


@dataclass
class FullResolutionTokenProjectionFeedbackResult:
    """Feature-only token update and compact detached diagnostics."""

    updated_token_features: torch.Tensor  # [B,N,D]
    token_error: torch.Tensor  # [B,N,256], or [B,N,512] for separate cues
    point_error_mean_magnitude: torch.Tensor  # scalar
    point_valid_ratio: torch.Tensor  # scalar
    view_valid_ratio: torch.Tensor  # scalar
    token_error_mean_magnitude: torch.Tensor  # scalar
    observation_cue: torch.Tensor | None = None  # [B,N,256]
    prior_cue: torch.Tensor | None = None  # [B,N,256], Difix mode only


class FullResolutionTokenProjectionErrorFeedback(nn.Module):
    """Fuse sparse projected image errors into token content features."""

    error_channels = FrozenFullResolutionResNet18Error.output_channels
    point_channels = 256

    def __init__(
        self,
        token_dim: int,
        num_gaussians_per_token: int,
        knn_samples: int = 16,
        point_transformer_blocks: int = 2,
        point_attention_channels: int = 64,
        separate_prior_cue: bool = False,
        eps: float = 1e-6,
        feature_extractor: nn.Module | None = None,
        sampler: nn.Module | None = None,
        point_transformer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError("token_dim must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.token_dim = int(token_dim)
        self.eps = float(eps)
        self.separate_prior_cue = bool(separate_prior_cue)
        self.feature_extractor = (
            FrozenFullResolutionResNet18Error()
            if feature_extractor is None
            else feature_extractor
        )
        self.sampler = (
            TokenProjectionErrorSampler(
                num_gaussians_per_token=num_gaussians_per_token,
                eps=eps,
            )
            if sampler is None
            else sampler
        )
        self.point_mlp = nn.Sequential(
            nn.Linear(self.error_channels + 3, self.point_channels),
            nn.GELU(),
            nn.Linear(self.point_channels, self.point_channels),
            nn.GELU(),
        )
        self.prior_point_mlp = None
        self.three_way_fusion = None
        self.error_to_token = None
        if self.separate_prior_cue:
            self.prior_point_mlp = nn.Sequential(
                nn.Linear(self.error_channels + 3, self.point_channels),
                nn.GELU(),
                nn.Linear(self.point_channels, self.point_channels),
                nn.GELU(),
            )
            fusion_channels = self.token_dim + 2 * self.point_channels
            self.three_way_fusion = nn.Sequential(
                nn.LayerNorm(fusion_channels),
                nn.Linear(fusion_channels, self.token_dim),
                nn.GELU(),
                nn.Linear(self.token_dim, self.token_dim),
            )
            nn.init.zeros_(self.three_way_fusion[-1].weight)
            nn.init.zeros_(self.three_way_fusion[-1].bias)
        else:
            self.error_to_token = nn.Sequential(
                nn.LayerNorm(self.point_channels),
                nn.Linear(self.point_channels, self.token_dim),
                nn.GELU(),
                nn.Linear(self.token_dim, self.token_dim),
            )
            nn.init.zeros_(self.error_to_token[-1].weight)
            nn.init.zeros_(self.error_to_token[-1].bias)
        self.point_transformer = (
            TokenPlainPointTransformer(
                channels=self.token_dim,
                knn_samples=knn_samples,
                num_blocks=point_transformer_blocks,
                attn_proj_channels=point_attention_channels,
            )
            if point_transformer is None
            else point_transformer
        )
        self.point_output = nn.Linear(self.token_dim, self.token_dim)
        nn.init.zeros_(self.point_output.weight)
        nn.init.zeros_(self.point_output.bias)

    def _aggregate_view_group(
        self,
        point_error: torch.Tensor,
        point_valid: torch.Tensor,
        local_delta: torch.Tensor,
        token_valid_mask: torch.Tensor,
        point_mlp: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, views, num_tokens, num_points, channels = point_error.shape
        if channels != self.error_channels:
            raise ValueError(
                f"point error must have {self.error_channels} channels"
            )
        if local_delta.shape != (batch, num_tokens, num_points, 3):
            raise ValueError("local_delta must have shape [B,N,P,3]")
        point_valid = point_valid & token_valid_mask[:, None, :, None]
        valid_float = point_valid[..., None].to(point_error.dtype)
        sample_error = (
            point_error * valid_float
        ).sum(dim=1) / valid_float.sum(dim=1).clamp_min(1)
        sample_valid = point_valid.any(dim=1)
        point_input = torch.cat(
            (sample_error, local_delta.to(point_error.dtype)), dim=-1
        )
        point_features = point_mlp(point_input)
        sample_valid_float = sample_valid[..., None].to(point_features.dtype)
        token_cue = (
            point_features * sample_valid_float
        ).sum(dim=2) / sample_valid_float.sum(dim=2).clamp_min(1)
        view_valid = point_valid.any(dim=3)
        token_cue = token_cue * token_valid_mask[..., None].to(
            token_cue.dtype
        )
        return token_cue, view_valid

    def _aggregate_samples(
        self,
        samples: TokenProjectionSamples,
        token_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._aggregate_view_group(
            samples.point_error,
            samples.point_valid,
            samples.local_delta,
            token_valid_mask,
            self.point_mlp,
        )

    def _aggregate_separate_cues(
        self,
        samples: TokenProjectionSamples,
        token_valid_mask: torch.Tensor,
        context_view_count: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        views = samples.point_error.shape[1]
        if (
            context_view_count is None
            or context_view_count <= 0
            or context_view_count >= views
        ):
            raise ValueError(
                "separate error cues require nonempty context and target view groups"
            )
        if self.prior_point_mlp is None:
            raise RuntimeError("Difix prior Point MLP is not initialized")
        observation_cue, context_valid = self._aggregate_view_group(
            samples.point_error[:, :context_view_count],
            samples.point_valid[:, :context_view_count],
            samples.local_delta,
            token_valid_mask,
            self.point_mlp,
        )
        prior_cue, prior_valid = self._aggregate_view_group(
            samples.point_error[:, context_view_count:],
            samples.point_valid[:, context_view_count:],
            samples.local_delta,
            token_valid_mask,
            self.prior_point_mlp,
        )
        return (
            observation_cue,
            prior_cue,
            torch.cat((context_valid, prior_valid), dim=1),
        )

    def _diagnostics(
        self,
        samples: TokenProjectionSamples,
        token_error: torch.Tensor,
        view_valid: torch.Tensor,
        token_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        point_valid = samples.point_valid & token_valid_mask[:, None, :, None]
        point_magnitude = torch.linalg.vector_norm(
            samples.point_error.float(), dim=-1
        )
        point_count = point_valid.sum().clamp_min(1)
        point_error_mean = (
            point_magnitude * point_valid.to(point_magnitude.dtype)
        ).sum() / point_count
        views = samples.point_valid.shape[1]
        points = samples.point_valid.shape[3]
        valid_tokens = token_valid_mask.sum().clamp_min(1)
        point_valid_ratio = point_valid.sum().float() / (
            valid_tokens.float() * views * points
        )
        view_valid_ratio = view_valid.sum().float() / (
            valid_tokens.float() * views
        )
        token_magnitude = torch.linalg.vector_norm(token_error.float(), dim=-1)
        token_error_mean = (
            token_magnitude * token_valid_mask.to(token_magnitude.dtype)
        ).sum() / valid_tokens
        return (
            point_error_mean.detach(),
            point_valid_ratio.detach(),
            view_valid_ratio.detach(),
            token_error_mean.detach(),
        )

    def forward(
        self,
        token_features: torch.Tensor,
        token_xyz: torch.Tensor,
        token_radii: torch.Tensor,
        token_valid_mask: torch.Tensor,
        stage_gaussians: torch.Tensor,
        rendered_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        cam_view: torch.Tensor,
        intrinsics: torch.Tensor,
        target_features: torch.Tensor | None = None,
        context_view_count: int | None = None,
    ) -> FullResolutionTokenProjectionFeedbackResult:
        if token_features.ndim != 3 or token_features.shape[-1] != self.token_dim:
            raise ValueError("token_features must have shape [B,N,token_dim]")
        if token_xyz.shape != (*token_features.shape[:2], 3):
            raise ValueError("token_xyz must have shape [B,N,3]")
        if token_radii.shape != (*token_features.shape[:2], 1):
            raise ValueError("token_radii must have shape [B,N,1]")
        if token_valid_mask.shape != token_features.shape[:2]:
            raise ValueError("token_valid_mask must have shape [B,N]")

        error_map = self.feature_extractor(
            rendered_rgb,
            target_rgb,
            target_features=target_features,
        )
        samples = self.sampler(
            stage_gaussians,
            token_xyz.detach(),
            token_radii.detach(),
            token_valid_mask,
            error_map.detach(),
            cam_view.detach(),
            intrinsics.detach(),
        )
        mask = token_valid_mask[..., None].to(token_features.dtype)
        prior_cue = None
        if self.separate_prior_cue:
            observation_cue, prior_cue, view_valid = (
                self._aggregate_separate_cues(
                    samples,
                    token_valid_mask,
                    context_view_count,
                )
            )
            if self.three_way_fusion is None:
                raise RuntimeError("three-way error fusion is not initialized")
            fusion_input = torch.cat(
                (
                    token_features,
                    observation_cue.to(token_features.dtype),
                    prior_cue.to(token_features.dtype),
                ),
                dim=-1,
            )
            fused_features = token_features + self.three_way_fusion(fusion_input)
            token_error = torch.cat((observation_cue, prior_cue), dim=-1)
        else:
            observation_cue, view_valid = self._aggregate_samples(
                samples, token_valid_mask
            )
            if self.error_to_token is None:
                raise RuntimeError("context error fusion is not initialized")
            token_error = observation_cue
            fused_features = token_features + self.error_to_token(
                token_error.to(token_features.dtype)
            )
        fused_features = fused_features * mask
        point_state = self.point_transformer(
            token_xyz.detach(), fused_features, token_valid_mask
        )
        updated_features = fused_features + self.point_output(point_state)
        updated_features = updated_features * mask
        (
            point_error_mean,
            point_valid_ratio,
            view_valid_ratio,
            token_error_mean,
        ) = self._diagnostics(
            samples, token_error, view_valid, token_valid_mask
        )
        return FullResolutionTokenProjectionFeedbackResult(
            updated_token_features=updated_features,
            token_error=token_error.detach(),
            point_error_mean_magnitude=point_error_mean,
            point_valid_ratio=point_valid_ratio,
            view_valid_ratio=view_valid_ratio,
            token_error_mean_magnitude=token_error_mean,
            observation_cue=observation_cue.detach(),
            prior_cue=None if prior_cue is None else prior_cue.detach(),
        )
