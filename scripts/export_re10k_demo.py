#!/usr/bin/env python3
"""Export one local RE10K torch scene into TokenBloom demo assets."""
from __future__ import annotations
import argparse, io, json
from pathlib import Path
import torch
from PIL import Image

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scene-torch',type=Path,default=Path('/root/autodl-tmp/re10k_torch/test/000103.torch'))
    p.add_argument('--output-dir',type=Path,default=Path('TokenBloom/examples/re10k_000103'))
    p.add_argument('--context',type=int,nargs=2,default=(0,20))
    p.add_argument('--target',type=int,default=10)
    a=p.parse_args(); sample=torch.load(a.scene_torch,map_location='cpu',weights_only=False)[0]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    def camera(index):
        row=sample['cameras'][index].float(); image=Image.open(io.BytesIO(sample['images'][index].numpy().tobytes())).convert('RGB')
        width,height=image.size
        K=[float(row[0]*width),float(row[1]*height),float(row[2]*width),float(row[3]*height)]
        w2c=torch.eye(4); w2c[:3,:4]=row[6:].reshape(3,4)
        pose=torch.linalg.inv(w2c)
        return image,K,pose.tolist()
    context=[]
    for slot,index in enumerate(a.context):
        image,K,pose=camera(index); filename=f'context_{slot:03d}.png'; image.save(a.output_dir/filename)
        context.append({'image':filename,'intrinsics':K,'camera_to_world':pose})
    _,K,pose=camera(a.target)
    (a.output_dir/'scene.json').write_text(json.dumps({'context':context,'targets':[{'intrinsics':K,'camera_to_world':pose}]},indent=2))
    print(a.output_dir/'scene.json')
if __name__=='__main__': main()
