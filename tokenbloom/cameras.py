"""Camera validation and ray construction used by TokenBloom inference."""

from __future__ import annotations

import torch


def validate_camera_tensors(intrinsics: torch.Tensor, camera_to_world: torch.Tensor) -> None:
    """Validate batched 3x3 intrinsics and 4x4 camera-to-world matrices."""
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must have shape [V, 3, 3]")
    if camera_to_world.ndim != 3 or camera_to_world.shape[-2:] != (4, 4):
        raise ValueError("camera_to_world must have shape [V, 4, 4]")
    if intrinsics.shape[0] != camera_to_world.shape[0]:
        raise ValueError("intrinsics and camera_to_world must have the same view count")
    if not torch.isfinite(intrinsics).all() or not torch.isfinite(camera_to_world).all():
        raise ValueError("camera tensors must be finite")
    if (intrinsics[:, 0, 0] <= 0).any() or (intrinsics[:, 1, 1] <= 0).any():
        raise ValueError("camera focal lengths must be positive")


def pixel_rays(
    intrinsics: torch.Tensor,
    camera_to_world: torch.Tensor,
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return world-space ray origins and normalized directions, `[V,H,W,3]`."""
    validate_camera_tensors(intrinsics, camera_to_world)
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    device, dtype = intrinsics.device, intrinsics.dtype
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    homogeneous = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    directions_camera = torch.einsum(
        "vij,hwj->vhwi", torch.linalg.inv(intrinsics), homogeneous
    )
    directions_world = torch.einsum(
        "vij,vhwj->vhwi", camera_to_world[:, :3, :3], directions_camera
    )
    directions_world = torch.nn.functional.normalize(directions_world, dim=-1)
    origins = camera_to_world[:, None, None, :3, 3].expand_as(directions_world)
    return origins, directions_world
