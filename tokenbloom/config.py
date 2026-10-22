"""Fixed architecture configuration for the released TokenBloom checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    """Architecture constants for the supported RE10K 14-point checkpoint."""

    num_input_views: int = 2
    image_size: int = 256
    encoder_dim: int = 768
    token_dim: int = 768
    encoder_heads: int = 12
    encoder_depth: int = 12
    initial_tokens: int = 512
    final_tokens: int = 1440
    children_per_token: int = 64
    error_sample_count: int = 14
    point_knn: int = 16
    point_blocks: int = 2
    xyz_pe_num_freqs: int = 6
    xyz_pe_radius: float = 10.0
    dino_weights_relpath: Path = Path("weights/dinov2/dinov2_vitb14_pretrain.pth")
    resnet_weights_relpath: Path = Path("weights/torchvision/resnet18-f37072fd.pth")
    checkpoint_relpath: Path = Path("checkpoints/tokenbloom_re10k.safetensors")
    disable_xformers_by_default: bool = True


DEFAULT_CONFIG = InferenceConfig()
