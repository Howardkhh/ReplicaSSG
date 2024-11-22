from argparse import ArgumentParser
from pathlib import Path
import json
import os
import threading

import numpy as np
import habitat_sim
import cv2

from settings import default_sim_settings, make_cfg, default_agent_config

# Semantic rendering problem fix
# https://github.com/facebookresearch/Replica-Dataset/issues/89


class Viewer:
    def __init__(self, sim_settings, args):
        self.sim_settings = sim_settings
        self.fps = 60
        self.sim_settings["width"] = 640
        self.sim_settings["height"] = 360
        self.display_resolution = (1920, 1080)

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
            112: "switch_camera",
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

        if os.path.exists(args.relationship_path):
            with open(args.relationship_path, "r") as f:
                self.relationships = json.load(f)
        else:
            self.relationships = {"scans": []}
        
        for scan in self.relationships["scans"]:
            if scan["scan"] == args.scene:
                self.relationships_cur_scan = scan
                break
        else:
            self.relationships["scans"].append({"scan": args.scene, "relationships": []})
            self.relationships_cur_scan = self.relationships["scans"][-1]
            with open(args.relationship_path, "w") as f:
                json.dump(self.relationships, f, indent=4)

        print(f"Annotating scan {args.scene}")
        print(f"Previous relationships: {self.relationships_cur_scan['relationships']}")
        for rel in self.relationships_cur_scan["relationships"]:
            s, o, r, rel_name = rel
            s_name = self.semantic_classes[self.semantic_id2label[s]]
            o_name = self.semantic_classes[self.semantic_id2label[o]]
            print(s_name, rel_name, o_name)


        self.running = True
        self.input_thread = threading.Thread(target=self.get_relationship_input)
        self.input_thread.start()

    def get_relationship_input(self):
        while self.running:
            relationship = input()
            if relationship == "undo":
                if len(self.relationships_cur_scan["relationships"]) > 0:
                    print(f'Undoing {self.relationships_cur_scan["relationships"].pop()}')
                    with open(args.relationship_path, "w") as f:
                        json.dump(self.relationships, f, indent=4)
                else:
                    print("No relationships to undo")
                continue
            try:
                ann = relationship.split(" ")
                sub = ann[0]
                obj = ann[-1]
                rel = " ".join(ann[1:-1])
                sub, obj = int(sub), int(obj)
            except Exception as e:
                print(e)
                continue
            if rel not in self.replica_to_vg["VisualGenome_rel"]:
                print(f"Invalid relationship: {rel}")
                continue
            if sub >= len(self.semantic_id2label) or self.semantic_classes[self.semantic_id2label[sub]].startswith("unknown"):
                print(f"Invalid subject: {sub}")
                continue
            if obj >= len(self.semantic_id2label) or self.semantic_classes[self.semantic_id2label[obj]].startswith("unknown"):
                print(f"Invalid object: {obj}")
                continue
            rel_idx = self.replica_to_vg["VisualGenome_rel"].index(rel)
            if [sub, obj, rel_idx, rel] in self.relationships_cur_scan["relationships"]:
                print(f"Relationship already exists: {sub} {rel} {obj}")
                continue
            self.relationships_cur_scan["relationships"].append([sub, obj, rel_idx, rel])
            print(f'Added relationship: {sub} ({self.replica_to_vg["Replica2VisualGenome"][self.semantic_classes[self.semantic_id2label[sub]]]}) {rel} {obj} ({self.replica_to_vg["Replica2VisualGenome"][self.semantic_classes[self.semantic_id2label[obj]]]})')
            if rel == "near":
                self.relationships_cur_scan["relationships"].append([obj, sub, rel_idx, rel])
                print(f'Added relationship: {obj} ({self.replica_to_vg["Replica2VisualGenome"][self.semantic_classes[self.semantic_id2label[obj]]]}) {rel} {sub} ({self.replica_to_vg["Replica2VisualGenome"][self.semantic_classes[self.semantic_id2label[sub]]]})')
            if rel == "above":
                self.relationships_cur_scan["relationships"].append([obj, sub, self.replica_to_vg["VisualGenome_rel"].index("under"), "under"])
                print(f'Added relationship: {obj} ({self.replica_to_vg["Replica2VisualGenome"][self.semantic_classes[self.semantic_id2label[obj]]]}) under {sub} ({self.replica_to_vg["Replica2VisualGenome"][self.semantic_classes[self.semantic_id2label[sub]]]})')
            with open(args.relationship_path, "w") as f:
                json.dump(self.relationships, f, indent=4)


    def start(self):
        cur_sensor = 0
        all_sensors = ["color_sensor", "depth_sensor", "semantic_sensor", "color_sensor"]
        semantic_colors = (np.random.rand(1024, 3) * 255).astype(np.uint8)
        while self.running:
            observations = self.sim.get_sensor_observations()
            rgb = observations[all_sensors[cur_sensor]]
            if cur_sensor == 0:
                rgb = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
                rgb = cv2.resize(rgb, self.display_resolution)
            elif cur_sensor == 1:
                rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
                rgb = cv2.resize(rgb, self.display_resolution)
            elif cur_sensor == 2:
                rgb = semantic_colors[rgb]
                rgb = cv2.resize(rgb, self.display_resolution, interpolation=cv2.INTER_NEAREST)
            elif cur_sensor == 3:
                rgb = cv2.cvtColor(rgb[..., :3], cv2.COLOR_RGB2BGR)
                rgb = cv2.resize(rgb, self.display_resolution)
                semantic = observations["semantic_sensor"]
                instance_ids = np.unique(semantic)
                id2xy = {}
                for id in instance_ids:
                    coords = np.where(semantic == id)
                    x1, x2 = np.min(coords[1]), np.max(coords[1])
                    y1, y2 = np.min(coords[0]), np.max(coords[0])
                    x1, x2, y1, y2 = int(x1 * self.display_resolution[0] / self.sim_settings["width"]), int(x2 * self.display_resolution[0] / self.sim_settings["width"]), int(y1 * self.display_resolution[1] / self.sim_settings["height"]), int(y2 * self.display_resolution[1] / self.sim_settings["height"])
                    replica_class = self.semantic_classes[self.semantic_id2label[id]]
                    if self.semantic_id2label[id] > 0:
                        vg_class = self.replica_to_vg["Replica2VisualGenome"][replica_class]
                        if vg_class == "unknown":
                            class_name = f"{replica_class} (unknown VG)"
                            continue
                        else:
                            class_name = f"{vg_class} ({replica_class})"
                    else:
                        class_name = replica_class
                        continue
                    id2xy[id] = (x1, y1)
                    cv2.rectangle(rgb, (x1, y1), (x2, y2), semantic_colors[id].tolist(), 2)
                    cv2.putText(rgb, f"{id}: {class_name}", (x1+5, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, semantic_colors[id].tolist(), 1)
                for rel in self.relationships_cur_scan["relationships"]:
                    sub, obj, rel_idx, rel_name = rel
                    if not(sub in instance_ids and obj in instance_ids):
                        continue
                    sub_xy, obj_xy = id2xy[sub], id2xy[obj]
                    cv2.arrowedLine(rgb, sub_xy, obj_xy, semantic_colors[rel_idx].tolist(), 2)

            cv2.imshow("RGB", rgb)
            key = cv2.waitKey(10) & 0xFF
            for k in self.keys2action:
                if k == key:
                    action = self.keys2action[key]
                    if action == "quit":
                        self.running = False
                    elif action == "switch_camera":
                        cur_sensor = (cur_sensor + 1) % len(all_sensors)
                        print(f"Switched to {all_sensors[cur_sensor]}")
                    else:
                        self.sim.step(action)
                    break

        
if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--replica_path", type=Path)
    args.add_argument("--scene", type=str)
    args.add_argument("--replica_to_vg_path", type=Path, default="replica_to_visual_genome.json")
    args.add_argument("--relationship_path", type=Path, default="relationships.json")
    args = args.parse_args()

    sim_settings = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = str(args.replica_path / "replica.scene_dataset_config.json")

    viewer = Viewer(sim_settings, args).start()