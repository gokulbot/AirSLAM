#!/usr/bin/env python3
# Extract a VIODE rosbag -> EuRoC/mav0 layout AirSLAM can read.
#   viode_extract.py <bag> <seq_out_dir>
# Writes: <out>/mav0/cam0/data/*.png, cam1/data/*.png, cam0/seg/*.png (oracle),
#         <out>/mav0/imu0/data.csv, <out>/groundtruth.txt (TUM).
# Timestamps offset to start at 1.5e9 s so AirSLAM's StringTimeToDouble (first 10 digits = seconds) parses correctly.
import rosbag, os, sys
import cv2
from cv_bridge import CvBridge

bag_path, out = sys.argv[1], sys.argv[2]
mav0 = os.path.join(out, "mav0")
for d in ["cam0/data", "cam1/data", "cam0/seg", "imu0"]:
    os.makedirs(os.path.join(mav0, d), exist_ok=True)
bridge = CvBridge()
bag = rosbag.Bag(bag_path)

# topic auto-detect (VIODE uses /cam0/image_raw etc., but be tolerant)
topics = bag.get_type_and_topic_info().topics
def pick(*cands):
    for c in cands:
        if c in topics: return c
    return None
T_C0  = pick("/cam0/image_raw", "/cam0/color/image_raw")
T_C1  = pick("/cam1/image_raw", "/cam1/color/image_raw")
T_SEG = pick("/cam0/segmentation", "/cam0/seg")
T_IMU = pick("/imu0", "/imu")
T_GT  = pick("/odometry", "/ground_truth", "/gt")
print(f"topics: cam0={T_C0} cam1={T_C1} seg={T_SEG} imu={T_IMU} gt={T_GT}")

T0 = None
for _, msg, t in bag.read_messages(topics=[T_C0]):
    T0 = msg.header.stamp.to_sec(); break
OFF = 1500000000.0
def ots(sec): return sec - T0 + OFF
def ns19(sec): return str(int(round(ots(sec) * 1e9)))

imu = ["#t[ns],wx,wy,wz,ax,ay,az"]; gt = ["#t tx ty tz qx qy qz qw"]
n = {"c0": 0, "c1": 0, "seg": 0, "imu": 0, "gt": 0}
for topic, msg, t in bag.read_messages():
    try: ts = msg.header.stamp.to_sec()
    except Exception: ts = t.to_sec()
    if topic == T_C0:
        cv2.imwrite(f"{mav0}/cam0/data/{ns19(ts)}.png", bridge.imgmsg_to_cv2(msg, "bgr8")); n["c0"] += 1
    elif topic == T_C1:
        cv2.imwrite(f"{mav0}/cam1/data/{ns19(ts)}.png", bridge.imgmsg_to_cv2(msg, "bgr8")); n["c1"] += 1
    elif topic == T_SEG:
        cv2.imwrite(f"{mav0}/cam0/seg/{ns19(ts)}.png", bridge.imgmsg_to_cv2(msg, "bgr8")); n["seg"] += 1
    elif topic == T_IMU:
        w, a = msg.angular_velocity, msg.linear_acceleration
        imu.append(f"{ns19(ts)},{w.x},{w.y},{w.z},{a.x},{a.y},{a.z}"); n["imu"] += 1
    elif topic == T_GT:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        gt.append(f"{ots(ts):.9f} {p.x} {p.y} {p.z} {q.x} {q.y} {q.z} {q.w}"); n["gt"] += 1
bag.close()
open(f"{mav0}/imu0/data.csv", "w").write("\n".join(imu) + "\n")
open(f"{out}/groundtruth.txt", "w").write("\n".join(gt) + "\n")
print("extracted:", n)
