# External runtime weights

TokenBloom keeps source code separate from large pretrained binaries. Before
running inference, place these files at the exact paths below:

```text
weights/dinov2/dinov2_vitb14_pretrain.pth
weights/torchvision/resnet18-f37072fd.pth
```

The first file is the frozen DINOv2-B/14 image encoder. The second is the
ImageNet ResNet-18 used only by the 14-point context error-feedback module.
Neither binary is committed to this repository.
