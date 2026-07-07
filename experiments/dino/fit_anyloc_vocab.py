#!/usr/bin/env python3
# Fit the AnyLoc VLAD vocabulary (32 x 1536 cluster centers) that the C++ AnyLocExtractor
# loads, and emit a reference descriptor for validating the C++ VLAD against Python.
#   output/anyloc_vocab.bin : [<ii K D>][float32 K*D]  cluster centers
#   output/anyloc_ref.bin   : [<i D>][float32 D]       reference VLAD descriptor for ref image
import torch, numpy as np, struct, os, glob, cv2
import torch.nn.functional as F
from sklearn.cluster import KMeans

REPO="/home/gokul/Projects/airslam/AirSLAM"
SP="/tmp/claude-1000/-home-gokul-Projects-airslam/e347bcbd-9cb5-4c77-97fc-181a4b23279a/scratchpad/anyloc_glasgow"
def resolve(rel):
    p=f"{REPO}/{rel}"
    return os.readlink(p).replace("/catkin_ws/src/air_slam", REPO) if os.path.islink(p) else p
imgs=[resolve(l.strip()) for l in open(f"{SP}/glasgow_kf_manifest.txt") if l.strip()]
print("fit images:", len(imgs))

dev="cuda"
mn=torch.tensor([0.485,0.456,0.406],device=dev).view(1,3,1,1)
sd=torch.tensor([0.229,0.224,0.225],device=dev).view(1,3,1,1)
def limg(p,R=224):                                 # match the C++ deploy pipeline exactly:
    g=cv2.imread(p,cv2.IMREAD_GRAYSCALE)           #   grayscale (add_dino / reloc read grayscale)
    g=cv2.resize(g,(R,R))                          #   cv INTER_LINEAR
    rgb=cv2.cvtColor(g,cv2.COLOR_GRAY2RGB)         #   replicate to 3ch (process_input GRAY2RGB)
    return torch.from_numpy(rgb).float().permute(2,0,1)/255.0
model=torch.hub.load("facebookresearch/dinov2","dinov2_vitg14",verbose=False).to(dev).eval().half()
_orig=F.interpolate                                # bicubic->bilinear pos-embed, as the ONNX export does
def _bil(*a,**k):
    if k.get("mode")=="bicubic": k["mode"]="bilinear"; k["antialias"]=False
    return _orig(*a,**k)
F.interpolate=_bil
E=model.num_features; P=(224//14)**2; cap={}
model.blocks[31].attn.qkv.register_forward_hook(lambda m,i,o: cap.__setitem__("v",o[...,2*E:3*E].float()))
def dense(paths):
    out=[]
    with torch.no_grad():
        for s in range(0,len(paths),8):
            x=torch.stack([limg(p) for p in paths[s:s+8]]).to(dev)
            x=((x-mn)/sd).half(); model.forward_features(x)
            out.append(cap["v"][:,-P:].cpu().numpy())
    return np.concatenate(out)
dm=dense(imgs); D=dm.shape[-1]; print("dense:", dm.shape)

rng=np.random.default_rng(0); flat=dm.reshape(-1,D)
sub=flat[rng.choice(len(flat),min(50000,len(flat)),replace=False)]
C=KMeans(n_clusters=32,n_init=4,random_state=0).fit(sub).cluster_centers_.astype("<f4")
with open(f"{REPO}/output/anyloc_vocab.bin","wb") as f:
    f.write(struct.pack("<ii",C.shape[0],C.shape[1])); f.write(C.tobytes())
print("vocab ->", C.shape)

l2n=lambda a,ax=-1: a/(np.linalg.norm(a,axis=ax,keepdims=True)+1e-8)
def vlad(F):
    a=np.argmin(((F[:,None,:]-C[None])**2).sum(-1),1); V=np.zeros((32,D),np.float32)
    for c in range(32):
        if np.any(a==c): V[c]=(F[a==c]-C[c]).sum(0)
    return l2n(l2n(V,ax=1).ravel())
ref=vlad(dm[0]).astype("<f4")
with open(f"{REPO}/output/anyloc_ref.bin","wb") as f:
    f.write(struct.pack("<i",ref.shape[0])); f.write(ref.tobytes())
print("ref descriptor ->", ref.shape, " for image:", imgs[0])
