#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tokenbloom.checkpoint import load_tokenbloom_model
from tokenbloom.demo import load_demo_spec, run_demo, save_render_outputs
def main():
    p=argparse.ArgumentParser(description='Render target cameras from two TokenBloom context images.')
    p.add_argument('--scene',required=True,type=Path,help='JSON scene specification')
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--dino-weights',type=Path)
    p.add_argument('--resnet-weights',type=Path)
    p.add_argument('--device',default='cuda')
    p.add_argument('--export-ply',action='store_true')
    a=p.parse_args(); device=torch.device(a.device)
    model=load_tokenbloom_model(a.checkpoint,device=device,dino_weights=a.dino_weights,resnet_weights=a.resnet_weights)
    images,K,c2w=load_demo_spec(a.scene,device); reconstruction,render=run_demo(model,images,K,c2w)
    save_render_outputs(render,a.output_dir)
    if a.export_ply: model.gs.save_ply(reconstruction.gaussians,a.output_dir/'gaussians.ply')
if __name__=='__main__': main()
