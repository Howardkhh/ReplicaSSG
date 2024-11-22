from argparse import ArgumentParser
from pathlib import Path
import json
import os

import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_two_vectors
import cv2
from PIL import Image
import quaternion as qt
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, RotationSpline
from tqdm import tqdm
from noise import pnoise1

from settings import default_sim_settings, make_cfg, default_agent_config

# Semantic rendering problem fix
# https://github.com/facebookresearch/Replica-Dataset/issues/89


class Viewer:
    def __init__(self, sim_settings, args):
        self.args = args

        self.sim_settings = sim_settings
        self.fps = 60
        self.sequence_lenth = 60 # in seconds
        if args.scene == "apartment_0":
            self.sequence_lenth = 150
        self.sim_settings["width"] = 960
        self.sim_settings["height"] = 540
        self.sim_settings["sensor_height"] = 0
        self.display_resolution = (1920, 1080)
        self.depth_shift = 1000

        self.cfg = make_cfg(self.sim_settings)
        self.agent_id = sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = default_agent_config(self.cfg, self.agent_id)
        self.sim = habitat_sim.Simulator(self.cfg)
        self.agent: habitat_sim.Agent = self.sim.agents[self.agent_id]
        self.sim.initialize_agent(self.agent_id)
        self.keys2action = {
            119: "move_forward",
            115: "move_backward",
            97: "move_left",
            100: "move_right",
            120: "move_down",
            122: "move_up",
            82: "look_up",
            84: "look_down",
            81: "turn_left",
            83: "turn_right",
            113: "quit",
            107: "add_keyframe",
        }

        semantic_json_file = args.replica_path / self.sim_settings["scene"] / "habitat" / "info_semantic.json"
        with open(semantic_json_file, "r") as f:
            self.semantic_info = json.load(f)
        
        semantic_classes = self.semantic_info["classes"]
        self.semantic_classes = {item["id"]: item["name"] for item in semantic_classes}
        self.semantic_classes[-1] = "unknown (-1)"
        self.semantic_classes[-2] = "unknown (-2)"
        self.semantic_id2label = self.semantic_info["id_to_label"]

        with open(args.replica_to_vg_path, "r") as f:
            self.replica_to_vg = json.load(f)
        
        if not os.path.exists(args.trajectory_json):
            print(f"Trajectory file {args.trajectory_json} not found. Recording new trajectory.")
            self.trajectory = {args.scene: []}
            self.running = True
            self.record()
        else:
            with open(args.trajectory_json, "r") as f:
                self.trajectory = json.load(f)
            if args.scene not in self.trajectory:
                print(f"Trajectory for scene {args.scene} not found. Recording new trajectory.")
                self.trajectory[args.scene] = []
                self.running = True
                self.record()
        self.render_trajectory()

    def record(self):
        while self.running:
            observations = self.sim.get_sensor_observations()
            color = observations["color_sensor"][..., :3]
            display_color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            display_color = cv2.resize(display_color, self.display_resolution)
            
            cv2.imshow("RGB", display_color)
            key = cv2.waitKey(10) & 0xFF
            for k in self.keys2action:
                if k == key:
                    action = self.keys2action[key]
                    if action == "quit":
                        with open(args.trajectory_json, "w") as f:
                            json.dump(self.trajectory, f, indent=4)
                        self.running = False
                    elif action == "add_keyframe":
                        color_sensor_state = self.agent.get_state().sensor_states["color_sensor"]
                        translation = color_sensor_state.position
                        rotation: qt = color_sensor_state.rotation
                        self.trajectory[args.scene].append({
                            "translation": translation.tolist(),
                            "rotation": [rotation.w, rotation.x, rotation.y, rotation.z]
                        })
                        print(f"Added keyframe at translation: {translation} and rotation: {rotation}")
                    else:
                        self.sim.step(action)
                    break
    
    def render_trajectory(self):
        output_dir = self.args.output_dir / self.args.scene / "sequence"
        os.makedirs(output_dir, exist_ok=True)

        total_frames = self.fps * self.sequence_lenth
        translation_array = np.array([frame["translation"] for frame in self.trajectory[args.scene]])
        rotation_array = np.array([frame["rotation"] for frame in self.trajectory[args.scene]])
        trans_diff_amount = np.linalg.norm(np.diff(translation_array, axis=0), axis=1)
        rot_diff_amount = np.array([2 * np.arccos(np.clip(np.abs(np.dot(rotation_array[i], rotation_array[i+1])), -1.0, 1.0)) for i in range(len(rotation_array) - 1)])

        segment_lengths = 1.0 * trans_diff_amount + 0.5 * rot_diff_amount
        frames_per_keyframe  = np.round(segment_lengths / np.sum(segment_lengths) * total_frames).astype(int)
        frames_per_keyframe[frames_per_keyframe==0] = 1
        frame_times = np.concatenate([[0], np.cumsum(frames_per_keyframe)])
        # for i in range(len(translation_array)):
        #     print(f"keyframe: {frame_times[i]/self.fps}, position: {translation_array[i]}, rotation: {rotation_array[i]}")

        trans_spline = CubicSpline(frame_times, translation_array)
        rotation_array = np.concatenate([rotation_array[:, 1:], rotation_array[:, :1]], axis=1)
        rot_spline = RotationSpline(frame_times, Rotation.from_quat(rotation_array))


        with open(output_dir / "_info.txt", "w") as f:
            f.write(f"m_colorWidth = {self.sim_settings['width']}\n")
            f.write(f"m_colorHeight = {self.sim_settings['height']}\n")
            f.write(f"m_depthWidth = {self.sim_settings['width']}\n")
            f.write(f"m_depthHeight = {self.sim_settings['height']}\n")
            f.write(f"m_depthShift = {self.depth_shift}\n")
            focal_length = (self.sim_settings["width"] / 2) / np.tan(np.deg2rad(self.sim_settings["hfov"]) / 2)
            f.write(f"m_calibrationColorIntrinsic = {focal_length} 0 {self.sim_settings['width'] / 2} 0 0 {focal_length} {self.sim_settings['height'] / 2} 0 0 0 1 0 0 0 0 1\n")
            f.write(f"m_calibrationColorExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")
            f.write(f"m_calibrationDepthIntrinsic = {focal_length} 0 {self.sim_settings['width'] / 2} 0 0 {focal_length} {self.sim_settings['height'] / 2} 0 0 0 1 0 0 0 0 1\n")
            f.write(f"m_calibrationDepthExtrinsic = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n")
            f.write(f"m_frames.size = {frame_times[-1]}\n")

        # https://github.com/facebookresearch/Replica-Dataset/issues/29
        _tmp = np.eye(3, dtype=np.float32)
        UnitY = _tmp[1]
        UnitZ = _tmp[2]

        R_hsim_replica = quat_from_two_vectors(-UnitZ, -UnitY)
        T_hsim_replica = np.eye(4, dtype=np.float32)
        T_hsim_replica[0:3, 0:3] = qt.as_rotation_matrix(R_hsim_replica)

        R_forward_back = quat_from_two_vectors(-UnitZ, UnitZ)
        T_replicac_hsimc = np.eye(4, dtype=np.float32)
        T_replicac_hsimc[0:3, 0:3] = qt.as_rotation_matrix(R_forward_back)

        noise_freq = 0.025
        noise_amp_pos = 0.01
        noise_amp_rot = 0.003

        total_frame_idx = 0
        for frame_idx in tqdm(range(frame_times[-1])):
            position = trans_spline(frame_idx)
            position += np.array([pnoise1(frame_idx * noise_freq) * noise_amp_pos, pnoise1(frame_idx * noise_freq + 1) * noise_amp_pos, pnoise1(frame_idx * noise_freq + 2) * noise_amp_pos])
            rot_noise = Rotation.from_rotvec(np.array([pnoise1(frame_idx * noise_freq + 3), pnoise1(frame_idx * noise_freq + 4), pnoise1(frame_idx * noise_freq + 5)]) * noise_amp_rot)
            rotation = (rot_noise * rot_spline(frame_idx)).as_quat()
            rotation = qt.quaternion(rotation[3], rotation[0], rotation[1], rotation[2])
            agent_state = self.agent.get_state()
            agent_state.position = position
            agent_state.rotation = rotation
            for sensor in agent_state.sensor_states:
                agent_state.sensor_states[sensor].position = position
                agent_state.sensor_states[sensor].rotation = rotation
            self.agent.set_state(agent_state)
            
            observations = self.sim.get_sensor_observations()
            color = observations["color_sensor"][..., :3]
            depth = observations["depth_sensor"]
            semantic = observations["semantic_sensor"]

            Image.fromarray(color).save(f"{output_dir}/frame-{total_frame_idx:06d}.color.jpg")
            Image.fromarray((depth * self.depth_shift).astype(np.int32)).save(f"{output_dir}/frame-{total_frame_idx:06d}.depth.pgm")
            Image.fromarray(np.tile(((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)[:, :, None], (1, 1, 3))).save(f"{output_dir}/frame-{total_frame_idx:06d}.depth.jpg")

            # w2c/c2w naming need to be checked
            w2c = np.eye(4, dtype=np.float32)
            w2c[0:3, 0:3] = qt.as_rotation_matrix(rotation)
            w2c[0:3, 3] = position

            c2w = np.linalg.inv(w2c)
            c2w = T_replicac_hsimc @ c2w @ T_hsim_replica
            w2c = np.linalg.inv(c2w)

            with open(f"{output_dir}/frame-{total_frame_idx:06d}.pose.txt", "w") as f:
                f.write(f"{w2c[0, 0]} {w2c[0, 1]} {w2c[0, 2]} {w2c[0, 3]}\n")
                f.write(f"{w2c[1, 0]} {w2c[1, 1]} {w2c[1, 2]} {w2c[1, 3]}\n")
                f.write(f"{w2c[2, 0]} {w2c[2, 1]} {w2c[2, 2]} {w2c[2, 3]}\n")
                f.write("0 0 0 1\n")
            
            instance_ids = np.unique(semantic)
            with open(f"{output_dir}/frame-{total_frame_idx:06d}.bb.txt", "w") as f:
                for id in instance_ids:
                    coords = np.where(semantic == id)
                    x1, x2 = np.min(coords[1]), np.max(coords[1])
                    y1, y2 = np.min(coords[0]), np.max(coords[0])
                    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
                    f.write(f"{id} {x1} {y1} {x2} {y2}\n")                

            total_frame_idx += 1

        
if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--replica_path", type=Path)
    args.add_argument("--scene", type=str)
    args.add_argument("--replica_to_vg_path", type=Path, default="replica_to_visual_genome.json")
    args.add_argument("--trajectory_json", type=Path, required=True)
    args.add_argument("--output_dir", type=Path, default="output")
    args = args.parse_args()

    sim_settings = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = str(args.replica_path / "replica.scene_dataset_config.json")

    viewer = Viewer(sim_settings, args)