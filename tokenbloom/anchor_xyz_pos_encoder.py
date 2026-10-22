"""Absolute world-XYZ positional encoding for initial anchors."""

from __future__ import annotations

import torch
import torch.nn as nn


class AnchorXYZPosEncoder(nn.Module):
    """Project the Fourier encoding of raw absolute XYZ to token dimension."""

    def __init__(
        self,
        dim: int = 768,
        num_freqs: int = 6,
        radius: float = 10.0,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("positional-encoding dimension must be positive")
        if num_freqs <= 0:
            raise ValueError("num_freqs must be positive")
        if radius <= 0:
            raise ValueError("radius must be positive")

        freqs = torch.logspace(
            0,
            num_freqs - 1,
            num_freqs,
            base=2.0,
        )
        freqs = torch.pi / (2.0 * radius) * freqs
        self.register_buffer("freqs", freqs, persistent=False)

        self.num_freqs = int(num_freqs)
        self.radius = float(radius)
        self.coord_proj = nn.Linear(3 + 6 * num_freqs, dim)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Differentiably encode raw world coordinates shaped ``[B, N, 3]``."""
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must have shape [B,N,3], got {tuple(xyz.shape)}")

        # Preserve the exact coordinate system produced by depth unprojection.
        # Fourier operations run in FP32 while retaining gradients to XYZ.
        xyz_fp32 = xyz.float()
        angles = (xyz_fp32[..., None] * self.freqs.float()).flatten(-2)
        encoding = torch.cat(
            (xyz_fp32, angles.cos(), angles.sin()),
            dim=-1,
        )
        encoding = encoding.to(self.coord_proj.weight.dtype)
        return self.coord_proj(encoding)

