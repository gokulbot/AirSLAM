#!/usr/bin/env python3
# Generate an AirSLAM VIO camera config + EuRoC-style dataset layout for an
# OpenLORIS-Scene sequence (T265 dual-fisheye stereo + IMU).
# usage: gen_openloris.py <extracted_seq_dir> <seq_name> <repo_datasets_dir> <configs_camera_dir>
import sys, os, re, shutil
import numpy as np

seq_dir, seq_name, ds_root, cfg_dir = sys.argv[1:5]

def read(p):
    return open(os.path.join(seq_dir, p)).read()

def data_array(block):
    m = re.search(r"data:\s*\[(.*?)\]", block, re.S)
    return [float(x) for x in m.group(1).replace("\n"," ").split(",") if x.strip()]

def sensor_block(text, name):
    # from "name:" to the next top-level "..._frame:" or "..._accelerometer:" etc.
    i = text.index(name + ":")
    j = re.search(r"\n[a-zA-Z0-9_]+:\n", text[i+len(name)+2:])
    return text[i: i+len(name)+2 + (j.start() if j else len(text))]

def intr_dist(sy, frame):
    b = sensor_block(sy, frame)
    intr = data_array(b[:b.index("distortion_coefficients")])   # first data:[] = intrinsics
    dist = data_array(b[b.index("distortion_coefficients"):])
    # OpenLORIS stores intrinsics as [fx, cx, fy, cy] -> AirSLAM wants [fx, fy, cx, cy]
    fx, cx, fy, cy = intr
    return [fx, fy, cx, cy], dist[:4] + [0]

def trans(tm, parent, child):
    # find the entry block with matching parent_frame + child_frame
    for blk in re.split(r"\n\s*-\s*\n", tm):
        if f"parent_frame: {parent}" in blk and f"child_frame: {child}" in blk:
            return np.array(data_array(blk)).reshape(4, 4)
    raise KeyError(f"{parent}->{child}")

sy = read("sensors.yaml"); tm = read("trans_matrix.yaml")
K0, D0 = intr_dist(sy, "t265_fisheye1_optical_frame")
K1, D1 = intr_dist(sy, "t265_fisheye2_optical_frame")
M_f1_acc = trans(tm, "t265_fisheye1_optical_frame", "t265_accelerometer")
M_f1_f2  = trans(tm, "t265_fisheye1_optical_frame", "t265_fisheye2_optical_frame")
Tbc0 = np.linalg.inv(M_f1_acc)          # IMU -> fisheye1
Tbc1 = np.linalg.inv(M_f1_acc) @ M_f1_f2  # IMU -> fisheye2

def yaml_T(T):
    return "\n".join("  - [" + ", ".join(f"{x:.10e}" for x in r) + "]" for r in T)

cfg = f"""%YAML:1.0
# OpenLORIS-Scene {seq_name}, T265 dual-fisheye stereo + IMU (VIO). Auto-generated.
# body = T265 IMU; cam0/cam1 T are IMU->fisheye1/2. intrinsics reordered [fx,cx,fy,cy]->[fx,fy,cx,cy].
image_height: 800
image_width: 848
use_imu: 1
depth_lower_thr: 0.1
depth_upper_thr: 10.0
max_y_diff: 2
distortion_type: 2
cam0:
  intrinsics: [{K0[0]:.10f}, {K0[1]:.10f}, {K0[2]:.10f}, {K0[3]:.10f}]
  distortion_coeffs: [{", ".join(f"{x:.10e}" for x in D0)}]
  T_type: 0
  T:
{yaml_T(Tbc0)}
cam1:
  intrinsics: [{K1[0]:.10f}, {K1[1]:.10f}, {K1[2]:.10f}, {K1[3]:.10f}]
  distortion_coeffs: [{", ".join(f"{x:.10e}" for x in D1)}]
  T_type: 0
  T:
{yaml_T(Tbc1)}
rate_hz: 200
gyroscope_noise_density: 1.6968e-04
gyroscope_random_walk: 1.9393e-05
accelerometer_noise_density: 2.0000e-3
accelerometer_random_walk: 3.0000e-3
g_value: 9.81007
"""
os.makedirs(cfg_dir, exist_ok=True)
cfg_path = os.path.join(cfg_dir, f"openloris_{seq_name}_vio.yaml")
open(cfg_path, "w").write(cfg)
print("wrote", cfg_path)
print("cam0 intrinsics [fx,fy,cx,cy] =", [round(v,2) for v in K0], "(cx~424,cy~400,fx~fy expected)")

# ---- IMU csv (interpolate accel -> gyro timestamps) ----
def load_imu(p):
    ts=[]; v=[]
    for l in open(os.path.join(seq_dir,p)):
        l=l.strip()
        if not l or l.startswith('#'): continue
        c=l.split()
        ts.append(float(c[0])); v.append([float(c[1]),float(c[2]),float(c[3])])
    return np.array(ts), np.array(v)
gt_,gv=load_imu("t265_gyroscope.txt"); at_,av=load_imu("t265_accelerometer.txt")
m=(gt_>=at_.min())&(gt_<=at_.max()); gt2=gt_[m]; gv2=gv[m]
ax=np.interp(gt2,at_,av[:,0]); ay=np.interp(gt2,at_,av[:,1]); az=np.interp(gt2,at_,av[:,2])
mav=os.path.join(ds_root, seq_name, "mav0")
os.makedirs(os.path.join(mav,"imu0"), exist_ok=True)
with open(os.path.join(mav,"imu0","data.csv"),"w") as o:
    o.write("#t,wx,wy,wz,ax,ay,az\n")
    for i in range(len(gt2)):
        o.write(f"{gt2[i]:.6f},{gv2[i,0]:.10f},{gv2[i,1]:.10f},{gv2[i,2]:.10f},{ax[i]:.10f},{ay[i]:.10f},{az[i]:.10f}\n")
print("imu rows:", len(gt2))

# ---- fisheye1/2 -> cam0/cam1, groundtruth ----
for src,dst in [("fisheye1","cam0"),("fisheye2","cam1")]:
    d=os.path.join(mav,dst,"data"); os.makedirs(d,exist_ok=True)
    sd=os.path.join(seq_dir,src)
    for f in os.listdir(sd): shutil.move(os.path.join(sd,f), os.path.join(d,f))
shutil.copy(os.path.join(seq_dir,"groundtruth.txt"), os.path.join(ds_root,seq_name,"groundtruth.txt"))
print("cam0:",len(os.listdir(os.path.join(mav,"cam0","data"))),"cam1:",len(os.listdir(os.path.join(mav,"cam1","data"))))
print("dataroot:", mav)
