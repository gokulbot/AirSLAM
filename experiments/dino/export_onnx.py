#!/usr/bin/env python3
"""Export a DINOv2 global-descriptor graph to ONNX for the C++/TensorRT reloc path.

The exported graph:
  input  "image"      : (1, 3, R, R) float, RGB in [0, 255]   (C++ feeds a resized image)
  in-graph            : /255 -> ImageNet normalize -> DINOv2 backbone
                        -> mean-pool final patch tokens -> L2-normalize
  output "descriptor" : (1, D) float, unit-norm  (cosine-compare in C++)

usage: export_onnx.py [--model dinov2_vits14] [--res 224] [--out out.onnx] [--opset 17]
"""
import argparse, torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="dinov2_vits14")
ap.add_argument("--res", type=int, default=224)          # multiple of 14
ap.add_argument("--out", default="dinov2_vits14.onnx")
ap.add_argument("--opset", type=int, default=17)
a = ap.parse_args()


class DinoGlobal(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):                    # image: (B,3,R,R) in [0,255]
        x = image / 255.0
        x = (x - self.mean) / self.std
        feats = self.backbone.forward_features(x)
        g = feats["x_norm_patchtokens"].mean(dim=1)   # mean-pool patch tokens
        return F.normalize(g, dim=1)                   # (B, D), unit norm


print(f"loading {a.model} ...")
backbone = torch.hub.load("facebookresearch/dinov2", a.model, verbose=False).eval()
model = DinoGlobal(backbone).eval()

# TensorRT has no cubic Resize; DINOv2 interpolates its positional encoding with
# bicubic. Force bilinear (near-lossless for pos-embed resampling, TRT-supported).
_orig_interp = F.interpolate
def _interp_bilinear(*args, **kwargs):
    if kwargs.get("mode") == "bicubic":
        kwargs["mode"] = "bilinear"
        kwargs["antialias"] = False
    return _orig_interp(*args, **kwargs)
F.interpolate = _interp_bilinear

dummy = torch.randint(0, 256, (1, 3, a.res, a.res)).float()
with torch.no_grad():
    ref = model(dummy)
print("descriptor dim:", ref.shape[1])

torch.onnx.export(
    model, dummy, a.out,
    input_names=["image"], output_names=["descriptor"],
    opset_version=a.opset, do_constant_folding=True,
    dynamic_axes=None,   # fixed 1x3xRxR for a simple TRT engine
)
print("exported", a.out)

# quick numerical self-check with onnxruntime if available
try:
    import onnxruntime as ort, numpy as np
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": dummy.numpy()})[0]
    cos = float((out[0] * ref.numpy()[0]).sum())
    err = float(np.abs(out - ref.numpy()).max())
    print(f"onnxruntime vs torch: cosine={cos:.6f}  max_abs_err={err:.2e}")
except ImportError:
    print("onnxruntime not installed - skipped numerical check")
