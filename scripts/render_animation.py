#!/usr/bin/env python3
"""Render a conservative local novel-view loop around the supplied target camera."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tokenbloom.animation import local_orbit_camera_path, save_gif
from tokenbloom.checkpoint import load_tokenbloom_model
from tokenbloom.demo import load_demo_spec
from tokenbloom.preprocess import build_model_input


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dino-weights", type=Path)
    parser.add_argument("--resnet-weights", type=Path)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--motion-fraction", type=float, default=0.20, help="Orbit radius as a fraction of the context-camera baseline")
    parser.add_argument("--cycles", type=int, default=2, help="Number of conservative local loops in the animation")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_tokenbloom_model(args.checkpoint, device=device, dino_weights=args.dino_weights, resnet_weights=args.resnet_weights)
    images, intrinsics, c2ws = load_demo_spec(args.scene, device)
    baseline = torch.linalg.vector_norm(c2ws[1, :3, 3] - c2ws[0, :3, 3]).clamp_min(1e-6)
    path = local_orbit_camera_path(c2ws[2], args.frames, radius=float(baseline * args.motion_fraction), cycles=args.cycles)
    animation_images = torch.cat((images[:2], torch.zeros_like(images[:1]).expand(args.frames, -1, -1, -1).clone()), dim=0)
    animation_intrinsics = torch.cat((intrinsics[:2], intrinsics[:1].expand(args.frames, -1)), dim=0)
    animation_c2ws = torch.cat((c2ws[:2], path), dim=0)
    model_input = build_model_input(animation_images, animation_intrinsics, animation_c2ws)
    reconstruction = model.forward_reconstruction(model_input)
    render = model.render_reconstruction(reconstruction, model_input.decoder)

    output_dir = args.output_dir
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, image in enumerate(render["images_pred"][0].detach().float().clamp(0, 1).cpu()):
        frame = Image.fromarray(image.mul(255).byte().permute(1, 2, 0).numpy())
        frame.save(frame_dir / f"{index:03d}.png")
        frames.append(frame)
    save_gif(frames, output_dir / "novel_view.gif", fps=args.fps)
    print(f"Wrote {len(frames)} frames and {output_dir / 'novel_view.gif'}")


if __name__ == "__main__":
    main()
