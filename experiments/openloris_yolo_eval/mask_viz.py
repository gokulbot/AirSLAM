import cv2, numpy as np, sys, os
from ultralytics import YOLO
model = YOLO("yolov8m-seg.pt")
COLORS = [(255,80,80),(80,255,80),(80,80,255),(255,255,80),(255,80,255),(80,255,255),(255,160,60)]
outdir="/tmp/mask_viz"; os.makedirs(outdir, exist_ok=True)
for idx, f in enumerate(sys.argv[1:]):
    g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    base = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    r = model(base, verbose=False, conf=0.25)[0]
    H,W = base.shape[:2]
    overlay = base.copy(); person = base.copy()
    if r.masks is not None:
        m = r.masks.data.cpu().numpy()  # (N,h,w) in mask res
        for i,(mk,cls,cf) in enumerate(zip(m, r.boxes.cls, r.boxes.conf)):
            mk = cv2.resize(mk,(W,H))>0.5
            col = COLORS[i%len(COLORS)]
            overlay[mk] = (0.45*np.array(col)+0.55*overlay[mk]).astype(np.uint8)
            name=r.names[int(cls)]
            if name=="person":
                person[mk]=(0.5*np.array((0,0,255))+0.5*person[mk]).astype(np.uint8)
        # draw labels on overlay
        for b,cls,cf in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            x1,y1=int(b[0]),int(b[1])
            cv2.putText(overlay,f"{r.names[int(cls)]} {float(cf):.2f}",(x1,max(y1-4,12)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    def lab(im,t):
        im=im.copy(); cv2.rectangle(im,(0,0),(W,26),(0,0,0),-1); cv2.putText(im,t,(6,19),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1); return im
    combo = np.hstack([lab(base,"original (grayscale fisheye)"), lab(overlay,"all masks"), lab(person,"person mask = dynamic-reject zone")])
    cv2.imwrite(f"{outdir}/frame{idx}_masks.jpg", combo)
    print(f"frame{idx}: saved")
print("DONE")
