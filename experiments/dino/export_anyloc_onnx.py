#!/usr/bin/env python3
"""Export the AnyLoc backbone (DINOv2 ViT-G/14, value facet of layer L) to ONNX.

The exported graph outputs the *dense value-facet tokens* (1, P, E); the C++ AnyLocExtractor
does the VLAD aggregation with a loaded vocabulary (keeps the graph simple + TRT-friendly).
  input  "image" : (1,3,R,R) float RGB [0,255]
  in-graph       : /255 -> ImageNet-norm -> ViT-G forward -> block[L].attn.qkv value tokens
  output "dense" : (1, P, E)   P=(R/14)^2 patch tokens, E=1536

usage: export_anyloc_onnx.py [--model dinov2_vitg14] [--layer 31] [--res 224] [--out anyloc_vitg14.onnx]
"""
import argparse, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="dinov2_vitg14")
ap.add_argument("--layer", type=int, default=31)
ap.add_argument("--res", type=int, default=224)
ap.add_argument("--out", default="anyloc_vitg14.onnx")
ap.add_argument("--opset", type=int, default=17)
a = ap.parse_args()


class AnyLocDense(nn.Module):
    def __init__(self, backbone, layer, res):
        super().__init__()
        self.backbone = backbone
        self.E = backbone.num_features
        self.P = (res // 14) ** 2
        self._v = None
        def hook(_m, _i, o):
            self._v = o[..., 2 * self.E:3 * self.E]   # value = last third of qkv
        backbone.blocks[layer].attn.qkv.register_forward_hook(hook)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        x = image / 255.0
        x = (x - self.mean) / self.std
        self.backbone.forward_features(x)          # runs blocks; hook captures the value tokens
        return self._v[:, -self.P:]                # (B, P, E) drop the cls token


print(f"loading {a.model} (value facet, layer {a.layer}) ...")
backbone = torch.hub.load("facebookresearch/dinov2", a.model, verbose=False).eval()
model = AnyLocDense(backbone, a.layer, a.res).eval()

# TensorRT has no bicubic Resize; DINOv2 resamples pos-embed with bicubic -> force bilinear.
_orig = F.interpolate
def _bilinear(*args, **kwargs):
    if kwargs.get("mode") == "bicubic":
        kwargs["mode"] = "bilinear"; kwargs["antialias"] = False
    return _orig(*args, **kwargs)
F.interpolate = _bilinear

dummy = torch.randint(0, 256, (1, 3, a.res, a.res)).float()
with torch.no_grad():
    ref = model(dummy)
print("dense value-facet shape:", tuple(ref.shape))

torch.onnx.export(
    model, dummy, a.out,
    input_names=["image"], output_names=["dense"],
    opset_version=a.opset, do_constant_folding=True, dynamic_axes=None)
print("exported", a.out)
