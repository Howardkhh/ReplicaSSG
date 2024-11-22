# ReplicaSSG Dataset

The structure of the dataset follows the [3DSSG dataset](https://github.com/3DSSG/3DSSG.github.io).

## Download the [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset)
by utilizing the download script in the repository.
```bash
./download.sh Replica/data
```

## Copy the annotations
```bash
mkdir Replica/ReplicaSSG
cp files/* Replica/ReplicaSSG
```

## Annotation tools
### Install the dependencies
```bash
pip install git+https://github.com/facebookresearch/habitat-sim@v0.2.2
pip install opencv-python pillow scipy tqdm noise
```

The `extract_instance_ply.py` script converts point cloud instance segmentation labels from faces to points, following the format used in 3DSSG.
```bash
python extract_instance_ply.py --replica_path Replica
```

The `extract_objects.py` script extracts the object.json file from the Replica dataset.
```bash
python extract_objects.py --replica_path Replica
```

The `labeler.py` allows for manual labeling of the relationships in the ReplicaSSG dataset.

```
W, A, S, D, Z, X keys for moving the camera forward, left, backward, right, up, down.
Arrow keys for rotating the camera.
P key for switching the camera mode.
Type `object_id relationship_name object_id` in the terminal to label a relationship. For example, `0 on 1`.
Type `undo` in the terminal to remove the last relationship.
The relationships are saved in the `relationships.json` file immediately.
```
```bash
export SCENE="apartment_0"
python labeler.py --replica_path Replica --scene $SCENE --replica_to_vg_path Replica/ReplicaSSG/replica_to_visual_genome.json --relationship_path Replica/ReplicaSSG/relationships.json
```

The `extract_path.py` script extracts the camera path from the Replica dataset given the keyframes.
```
W, A, S, D, Z, X keys for moving the camera forward, left, backward, right, up, down.
Arrow keys for rotating the camera.
K key for adding a keyframe.
Q key for saving the camera path and start rendering.
If there are camera path for the selected scene, the camera path will be loaded instead of starting a new path.
```
```bash
python extract_path.py --replica_path Replica --scene $SCENE --replica_to_vg_path Replica/ReplicaSSG/replica_to_visual_genome.json --trajectory_path Replica/ReplicaSSG/trajectories.json --output_dir Replica/data/
```