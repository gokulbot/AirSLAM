import cv2, numpy as np, sys, os
from ultralytics import YOLO

# T265 cam0: Kannala-Brandt fisheye (distortion_type 2)
K = np.array([[284.9808959961, 0, 425.2443847656],
              [0, 286.1023864746, 398.4675903320],
              [0, 0, 1]], dtype=np.float64)
D = np.array([-7.3047108017e-03, 4.3499931693e-02, -4.1283041239e-02, 7.6524601318e-03], dtype=np.float64)
W, H = 848, 800

print("loading yolov8m-seg ...", flush=True)
model = YOLO("yolov8m-seg.pt")   # medium: best quality -> fair test of whether the DATA is the limiter

# fisheye -> pinhole rectify map. balance=0.3 keeps a chunk of FOV without too much black border
Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (W, H), np.eye(3), balance=0.3)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, (W, H), cv2.CV_16SC2)

outdir = "/tmp/yolo_eval"; os.makedirs(outdir, exist_ok=True)
def summ(r):
    if r.boxes is None or len(r.boxes) == 0: return "NONE"
    return ", ".join(f"{r.names[int(c)]}({float(p):.2f})" for c, p in zip(r.boxes.cls, r.boxes.conf))

for i, f in enumerate(sys.argv[1:]):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)   # T265 is mono
    if img is None: print(f"[{i}] could not read {f}"); continue
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    r_raw = model(bgr, verbose=False)[0]
    und = cv2.remap(bgr, map1, map2, cv2.INTER_LINEAR)
    r_und = model(und, verbose=False)[0]
    cv2.imwrite(f"{outdir}/{i}_raw.jpg", r_raw.plot())
    cv2.imwrite(f"{outdir}/{i}_undist.jpg", r_und.plot())
    print(f"[{i}] {os.path.basename(f)}")
    print(f"    RAW-fisheye : {summ(r_raw)}")
    print(f"    UNDISTORTED : {summ(r_und)}", flush=True)
print("EVALDONE")
