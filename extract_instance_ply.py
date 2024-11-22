from argparse import ArgumentParser
from pathlib import Path
import os

from plyfile import PlyData, PlyElement
import numpy as np

def main(args):
    replica_path = args.replica_path / "data"
    scenes = [scene for scene in os.listdir(replica_path) if os.path.isdir(replica_path / scene)]
    
    for scene in scenes:
        print(f"Processing {scene}")
        input_mesh = PlyData.read(str(replica_path / scene / "habitat" / "mesh_semantic.ply"))

        vertex_pos = np.array(input_mesh['vertex'].data[['x','y','z']].tolist())
        object_id_to_color = {id: np.random.randint(0, 255, 3) for id in np.unique(input_mesh['face']['object_id'])}

        points = []
        for face in input_mesh['face']:
            object_id = face['object_id']
            face_center = np.mean(vertex_pos[face['vertex_indices']], axis=0)
            points.append((*face_center, *object_id_to_color[object_id], object_id))
        points_np = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('objectId', 'u2')])

        PlyData([PlyElement.describe(points_np, 'vertex')], text=True).write(str(replica_path / scene / "labels.instances.annotated.v2.ply"))

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--replica_path", type=Path)
    args = args.parse_args()

    main(args)