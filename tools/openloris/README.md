# OpenLORIS-Scene -> AirSLAM

`gen_openloris.py` converts an extracted OpenLORIS-Scene sequence (T265 dual-fisheye
stereo + IMU) into an AirSLAM VIO camera config + EuRoC-style `mav0/` dataset layout.

    python3 gen_openloris.py <extracted_seq_dir> <seq_name> <datasets_dir> <configs/camera dir>

Handling the two OpenLORIS gotchas:
- **Intrinsics order**: `sensors.yaml` stores `[fx, cx, fy, cy]`; AirSLAM expects
  `[fx, fy, cx, cy]`. The generator reorders them (sanity: cx~W/2, cy~H/2, fx~fy).
- **Stereo pair**: OpenLORIS D435i is color+depth only (no IR stereo). The only
  stereo is the T265 fisheye (`distortion_type: 2`, Kannala-Brandt), rectified
  internally by AirSLAM. body frame = T265 IMU; cam0/cam1 T = IMU->fisheye1/2.
- Builds `imu0/data.csv` by interpolating T265 accel (~67Hz) onto gyro (~200Hz).
