"""Standalone two-context novel-view demo helpers."""
from __future__ import annotations
import json
from pathlib import Path
import torch
from PIL import Image
from .preprocess import build_model_input

def _load_rgb(path: Path) -> torch.Tensor:
    image = Image.open(path).convert('RGB')
    return torch.from_numpy(__import__('numpy').array(image)).permute(2,0,1).float().div(255)

def load_demo_spec(path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    spec=json.loads(path.read_text())
    context=spec.get('context', [])
    targets=spec.get('targets', [])
    if len(context)!=2: raise ValueError('demo JSON requires exactly two context views')
    if not targets: raise ValueError('demo JSON requires at least one target view')
    context_images=[_load_rgb((path.parent / v['image']).resolve()) for v in context]
    images=torch.stack([*context_images, *[torch.zeros_like(context_images[0]) for _ in targets]]).to(device)
    views=[*context,*targets]
    intrinsics=torch.tensor([v['intrinsics'] for v in views], dtype=torch.float32, device=device)
    c2ws=torch.tensor([v['camera_to_world'] for v in views], dtype=torch.float32, device=device)
    return images,intrinsics,c2ws

def save_render_outputs(render: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb=render['images_pred'][0].detach().float().clamp(0,1).cpu()
    depth=render['depths_pred'][0,:,0].detach().float().cpu()
    for index,(image,z) in enumerate(zip(rgb,depth)):
        Image.fromarray(image.mul(255).byte().permute(1,2,0).numpy()).save(output_dir/f'render_{index:03d}.png')
        finite=z[torch.isfinite(z)&(z>0)]
        lo,hi=(finite.quantile(.02),finite.quantile(.98)) if finite.numel() else (torch.tensor(0.),torch.tensor(1.))
        normalized=((z-lo)/(hi-lo).clamp_min(1e-6)).clamp(0,1)
        Image.fromarray(normalized.mul(255).byte().numpy()).save(output_dir/f'depth_{index:03d}.png')

@torch.inference_mode()
def run_demo(model, images: torch.Tensor, intrinsics: torch.Tensor, c2ws: torch.Tensor) -> tuple[object, dict]:
    model_input=build_model_input(images,intrinsics,c2ws)
    reconstruction=model.forward_reconstruction(model_input)
    return reconstruction, model.render_reconstruction(reconstruction, model_input.decoder)
