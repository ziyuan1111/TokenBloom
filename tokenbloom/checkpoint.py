"""Configuration and strict checkpoint loading for the released model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml
from safetensors.torch import load_file

from .model import ATSplatStage1


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class _ReleaseConfigLoader(yaml.SafeLoader):
    """Read the original Tyro YAML without importing the training options class."""


def _construct_options(loader: yaml.Loader, node: yaml.Node) -> dict[str, Any]:
    return loader.construct_mapping(node, deep=True)


def _construct_tuple(loader: yaml.Loader, node: yaml.Node) -> tuple[Any, ...]:
    return tuple(loader.construct_sequence(node, deep=True))


_ReleaseConfigLoader.add_constructor("!dataclass:Options", _construct_options)
_ReleaseConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", _construct_tuple)


def load_model_options(config_path: Path | None = None) -> SimpleNamespace:
    """Load the frozen release architecture config without training options/tyro."""
    path = config_path or PACKAGE_ROOT / "tokenbloom" / "model_config.yaml"
    with path.open(encoding="utf-8") as handle:
        values: dict[str, Any] = yaml.load(handle, Loader=_ReleaseConfigLoader)
    values["dino_weights"] = str(
        PACKAGE_ROOT / "weights/dinov2/dinov2_vitb14_pretrain.pth"
    )
    values["use_difix_target_teacher"] = False
    values["use_pgsr"] = False
    values["fullres_error_attribution_mode"] = "projection14"
    return SimpleNamespace(**values)


def expected_weight_paths(package_root: Path = PACKAGE_ROOT) -> dict[str, Path]:
    return {
        "TokenBloom checkpoint": package_root / "checkpoints/tokenbloom_re10k.safetensors",
        "DINOv2-B/14": package_root / "weights/dinov2/dinov2_vitb14_pretrain.pth",
        "ResNet-18": package_root / "weights/torchvision/resnet18-f37072fd.pth",
    }


def require_runtime_weights(
    checkpoint_path: Path | None = None,
    package_root: Path = PACKAGE_ROOT,
    dino_weights: Path | None = None,
    resnet_weights: Path | None = None,
) -> Path:
    paths = expected_weight_paths(package_root)
    if checkpoint_path is not None:
        paths["TokenBloom checkpoint"] = checkpoint_path
    if dino_weights is not None:
        paths["DINOv2-B/14"] = dino_weights
    if resnet_weights is not None:
        paths["ResNet-18"] = resnet_weights
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing TokenBloom runtime weights:\n" + "\n".join(missing))
    return paths["TokenBloom checkpoint"]


def load_tokenbloom_model(
    checkpoint_path: Path | None = None,
    *,
    device: str | torch.device = "cuda",
    package_root: Path = PACKAGE_ROOT,
    dino_weights: Path | None = None,
    resnet_weights: Path | None = None,
) -> ATSplatStage1:
    """Build the fixed architecture and strictly load its safetensors state dict."""
    if "XFORMERS_DISABLED" not in os.environ:
        os.environ["XFORMERS_DISABLED"] = "1"
    checkpoint_path = None if checkpoint_path is None else checkpoint_path.expanduser().resolve()
    dino_weights = None if dino_weights is None else dino_weights.expanduser().resolve()
    resnet_weights = None if resnet_weights is None else resnet_weights.expanduser().resolve()
    checkpoint = require_runtime_weights(
        checkpoint_path, package_root, dino_weights, resnet_weights
    )
    os.environ.setdefault(
        "TOKENBLOOM_RESNET_WEIGHTS",
        str(resnet_weights or package_root / "weights/torchvision/resnet18-f37072fd.pth"),
    )
    options = load_model_options(package_root / "tokenbloom/model_config.yaml")
    options.dino_weights = str(
        dino_weights or package_root / "weights/dinov2/dinov2_vitb14_pretrain.pth"
    )
    model = ATSplatStage1(options)
    state = load_file(str(checkpoint), device="cpu")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.verify:
        parser.error("only --verify is supported")
    load_tokenbloom_model(args.checkpoint, device=args.device)
    print("TokenBloom checkpoint loaded strictly.")


if __name__ == "__main__":
    _main()
