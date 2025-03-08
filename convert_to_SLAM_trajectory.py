import os
import argparse
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

FPS = 30
TIME_STEP = 1 / FPS

args = argparse.ArgumentParser(description='Convert 3RScan trajectory to SLAM format')
args.add_argument('--replica_path', type=Path, required=True, help='Replica directory')
args = args.parse_args()

assert args.replica_path.exists(), f"Path {args.replica_path} does not exist"
assert (args.replica_path / "data").exists(), f"Path {args.replica_path / 'data'} does not exist"

scans = sorted([f for f in os.listdir(args.replica_path / "data") if os.path.isdir(args.replica_path / "data" / f)])
assert len(scans) == 18, f"Replica expected to have 18 scans, found {len(scans)}"

for scan in scans:
    scan_path = args.replica_path / "data" / scan / "sequence"
    dir_list = os.listdir(scan_path)
    pose_list = sorted([f for f in dir_list if f.endswith("pose.txt") and not f.endswith("slam.pose.txt")])
    init_pose = np.loadtxt(args.replica_path / "data" / scan / "sequence" / "frame-000000.pose.txt")

    
    inv_init_pose = np.eye(4)
    inv_init_pose[:3, :3] = init_pose[:3, :3].T
    inv_init_pose[:3, 3] = -inv_init_pose[:3, :3] @ init_pose[:3, 3]
    
    trajectory = []
    for idx, pose_file in enumerate(pose_list):
        timestamp = idx * TIME_STEP
        pose = np.loadtxt(scan_path / pose_file)
        pose = init_pose @ pose
        trans_x, trans_y, trans_z = pose[:3, 3]
        q = Quaternion._from_matrix(matrix=pose[:3, :3], rtol=1e-5, atol=1e-6)
        rot_w, rot_x, rot_y, rot_z = q.elements / q.norm
        trajectory.append(f"{timestamp} {trans_x} {trans_y} {trans_z} {rot_x} {rot_y} {rot_z} {rot_w}")
    
    with open(args.replica_path / "data" / scan / "trajectory_gt.txt", "w") as f:
        f.write("\n".join(trajectory))