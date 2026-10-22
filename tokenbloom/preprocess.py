"""Image/camera preprocessing shared by TokenBloom demo and RE10K inference."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from .input_types import ModelInput, ModelInputDecoder, ModelInputEncoder


_MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
_STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)


def normalize_ilrm_re10k(c2ws: torch.Tensor) -> torch.Tensor:
    """Apply the selected checkpoint's two-context iLRM camera normalization."""
    c2ws = c2ws.clone()
    context = c2ws[:2]
    scale = context[:, :3, 3].abs().max()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError("invalid RE10K context-camera scale")
    position = context[:, :3, 3].mean(0)
    forward = F.normalize(context[:, :3, 2].mean(0), dim=0)
    down = context[:, :3, 1].mean(0)
    down = F.normalize(down - down.dot(forward) * forward, dim=0)
    right = torch.cross(down, forward, dim=0)
    reference = torch.eye(4, dtype=c2ws.dtype, device=c2ws.device)
    reference[:3, :4] = torch.stack((right, down, forward, position), dim=1)
    c2ws = torch.linalg.inv(reference).unsqueeze(0) @ c2ws
    c2ws[:, :3, 3] /= scale
    return c2ws


def _resize_crop_rgb(images: torch.Tensor, output_size: tuple[int, int]) -> tuple[torch.Tensor, int, int]:
    """DepthSplat-style uint8 Lanczos resize and center crop."""
    _, _, height, width = images.shape
    out_h, out_w = output_size
    factor = max(out_h / height, out_w / width)
    scaled_h, scaled_w = round(height * factor), round(width * factor)
    frames = []
    for image in images:
        array = image.clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        resized = Image.fromarray(array).resize((scaled_w, scaled_h), Image.LANCZOS)
        frames.append(torch.from_numpy(np.array(resized)).permute(2, 0, 1).float() / 255)
    stacked = torch.stack(frames).to(images)
    row, col = (scaled_h - out_h) // 2, (scaled_w - out_w) // 2
    return stacked[..., row : row + out_h, col : col + out_w], scaled_h, scaled_w


def _rays(intrinsics: torch.Tensor, c2ws: torch.Tensor, height: int, width: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    views = intrinsics.shape[0]
    y, x = torch.meshgrid(torch.arange(height, device=intrinsics.device, dtype=intrinsics.dtype), torch.arange(width, device=intrinsics.device, dtype=intrinsics.dtype), indexing="ij")
    pixels = torch.stack((x + 0.5, y + 0.5, torch.ones_like(x)), dim=-1)
    K = torch.zeros(views, 3, 3, device=intrinsics.device, dtype=intrinsics.dtype)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = intrinsics.unbind(-1)[0], intrinsics.unbind(-1)[1], intrinsics.unbind(-1)[2], intrinsics.unbind(-1)[3], 1
    directions = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(K), pixels)
    directions = F.normalize(torch.einsum("vij,vhwj->vhwi", c2ws[:, :3, :3], directions), dim=-1)
    origins = c2ws[:, None, None, :3, 3].expand_as(directions)
    plucker = torch.cat((torch.cross(origins, directions, dim=-1), directions), dim=-1)
    return plucker.permute(0, 3, 1, 2), origins.permute(0, 3, 1, 2), directions.permute(0, 3, 1, 2)


def build_model_input(images: torch.Tensor, intrinsics: torch.Tensor, c2ws: torch.Tensor, output_size: tuple[int, int] = (256, 256)) -> ModelInput:
    """Build the fixed two-context TokenBloom input from RGB `[V,3,H,W]`."""
    if images.ndim != 4 or images.shape[0] < 3 or images.shape[1] != 3:
        raise ValueError("images must have [V>=3,3,H,W]: two context then targets")
    if intrinsics.shape != (images.shape[0], 4) or c2ws.shape != (images.shape[0], 4, 4):
        raise ValueError("intrinsics must be [V,4] and c2ws must be [V,4,4]")
    c2ws = normalize_ilrm_re10k(c2ws)
    height_in, width_in = images.shape[-2:]
    images, scaled_h, scaled_w = _resize_crop_rgb(images, output_size)
    out_h, out_w = output_size
    intrinsics = torch.stack((intrinsics[:, 0] * scaled_w / width_in, intrinsics[:, 1] * scaled_h / height_in, intrinsics[:, 2] * out_w / width_in, intrinsics[:, 3] * out_h / height_in), dim=-1)
    plucker, rays_o, rays_d = _rays(intrinsics, c2ws, out_h, out_w)
    rgb_normalized = (images[:2] - _MEAN.to(images)) / _STD.to(images)
    return ModelInput(
        encoder=ModelInputEncoder(images_rgb=rgb_normalized.unsqueeze(0), plucker=plucker[:2].unsqueeze(0), rays_os=rays_o[:2].unsqueeze(0), rays_ds=rays_d[:2].unsqueeze(0), intrinsics_input=intrinsics[:2].unsqueeze(0), cam_to_world_input=c2ws[:2].unsqueeze(0), images_rgb_unnormalized=images[:2].unsqueeze(0)),
        decoder=ModelInputDecoder(cam_view=torch.linalg.inv(c2ws[2:]).transpose(-1, -2).unsqueeze(0), intrinsics=intrinsics[2:].unsqueeze(0)),
    )
