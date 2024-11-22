from argparse import ArgumentParser
from pathlib import Path
import json
import os


args = ArgumentParser()
args.add_argument('--replica_path', type=Path, required=True, help='Replica directory')
args = args.parse_args()

assert args.replica_path.exists(), f"Path {args.replica_path} does not exist"
assert (args.replica_path / "data").exists(), f"Path {args.replica_path / 'data'} does not exist"
assert (args.replica_path / "ReplicaSSG").exists(), f"Path {args.replica_path / 'ReplicaSSG'} does not exist"

object_json_path = args.replica_path / "ReplicaSSG" / "objects.json"

scans = sorted([f for f in os.listdir(args.replica_path / "data") if os.path.isdir(args.replica_path / "data" / f)])
assert len(scans) == 18, f"Replica expected to have 18 scans, found {len(scans)}"

object_json = {"scans": []}
for scan in scans:
    print(f"Processing {scan}")
    objects = []
    with open(args.replica_path / "data" / scan / "habitat" / "info_semantic.json", 'r') as f:
        scan_data = json.load(f)
    id2label = {}
    for obj in scan_data["objects"]:
        id2label[obj["id"]] = obj["class_name"]
    
    for id, cls in enumerate(scan_data["id_to_label"]):
        if id not in id2label:
            label = "unknown"
        else:
            label = id2label[id]
        objects.append({
            "label": label,
            "id": str(id)
        })
    object_json["scans"].append({"scan": scan, "objects": objects})

with open(object_json_path, 'w') as f:
    json.dump(object_json, f, indent=4)