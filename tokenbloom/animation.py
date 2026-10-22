"""Novel-view camera paths and lightweight animation export."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


def _matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert a proper 3x3 rotation matrix to a normalized xyzw quaternion."""
    trace = matrix.trace()
    if trace > 0:
        scale = torch.sqrt(trace + 1.0) * 2.0
        quaternion = torch.stack((
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
            0.25 * scale,
        ))
    else:
        diagonal = torch.diagonal(matrix)
        index = int(diagonal.argmax())
        if index == 0:
            scale = torch.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = torch.stack((
                0.25 * scale,
                (matrix[0, 1] + matrix[1, 0]) / scale,
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
            ))
        elif index == 1:
            scale = torch.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = torch.stack((
                (matrix[0, 1] + matrix[1, 0]) / scale,
                0.25 * scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
            ))
        else:
            scale = torch.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = torch.stack((
                (matrix[0, 2] + matrix[2, 0]) / scale,
                (matrix[1, 2] + matrix[2, 1]) / scale,
                0.25 * scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ))
    return torch.nn.functional.normalize(quaternion, dim=0)


def _quaternion_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    x, y, z, w = torch.nn.functional.normalize(quaternion, dim=0)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    )).reshape(3, 3)


def _slerp(start: torch.Tensor, end: torch.Tensor, fractions: torch.Tensor) -> torch.Tensor:
    dot = torch.dot(start, end)
    end = torch.where((dot < 0).expand_as(end), -end, end)
    dot = torch.dot(start, end).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    if float(angle.abs()) < 1e-6:
        return torch.nn.functional.normalize((1.0 - fractions[:, None]) * start + fractions[:, None] * end, dim=-1)
    return (
        torch.sin((1.0 - fractions[:, None]) * angle) / torch.sin(angle) * start
        + torch.sin(fractions[:, None] * angle) / torch.sin(angle) * end
    )


def interpolate_camera_path(start_c2w: torch.Tensor, end_c2w: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Return rigid C2W transforms with linear translation and SLERP rotation."""
    if num_frames < 2:
        raise ValueError("num_frames must be at least two")
    if start_c2w.shape != (4, 4) or end_c2w.shape != (4, 4):
        raise ValueError("camera transforms must be [4,4]")
    fractions = torch.linspace(0, 1, num_frames, dtype=start_c2w.dtype, device=start_c2w.device)
    q0, q1 = _matrix_to_quaternion(start_c2w[:3, :3]), _matrix_to_quaternion(end_c2w[:3, :3])
    rotations = torch.stack([_quaternion_to_matrix(q) for q in _slerp(q0, q1, fractions)])
    translations = (1.0 - fractions[:, None]) * start_c2w[:3, 3] + fractions[:, None] * end_c2w[:3, 3]
    path = torch.eye(4, dtype=start_c2w.dtype, device=start_c2w.device).expand(num_frames, -1, -1).clone()
    path[:, :3, :3] = rotations
    path[:, :3, 3] = translations
    path[0], path[-1] = start_c2w, end_c2w
    return path


def local_orbit_camera_path(
    target_c2w: torch.Tensor, num_frames: int, radius: float, cycles: int = 1
) -> torch.Tensor:
    """Make a small, closed translation loop starting at a known-good target pose.

    The camera orientation is deliberately retained: this is a conservative
    parallax path for model demos, rather than a large free-camera orbit that
    can leave the reconstructed scene's valid viewing range.
    """
    if num_frames < 2:
        raise ValueError("num_frames must be at least two")
    if target_c2w.shape != (4, 4):
        raise ValueError("target_c2w must be [4,4]")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if cycles < 1:
        raise ValueError("cycles must be positive")
    phase = 2 * torch.pi * cycles * torch.arange(num_frames, dtype=target_c2w.dtype, device=target_c2w.device) / num_frames
    right = target_c2w[:3, 0]
    up = target_c2w[:3, 1]
    offsets = radius * ((torch.cos(phase) - 1)[:, None] * right + 0.35 * torch.sin(phase)[:, None] * up)
    path = target_c2w.expand(num_frames, -1, -1).clone()
    path[:, :3, 3] += offsets
    return path


def save_gif(frames: Sequence[Image.Image], output_path: Path, fps: int = 12) -> None:
    """Write RGB frames as a browser-friendly looping GIF."""
    if not frames:
        raise ValueError("cannot save an empty animation")
    if fps <= 0:
        raise ValueError("fps must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].convert("RGB").save(
        output_path,
        save_all=True,
        append_images=[frame.convert("RGB") for frame in frames[1:]],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
    )
