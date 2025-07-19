<div align="center">

# ReplicaSSG Dataset
#### FROSS: Faster-than-Real-Time Online 3D Semantic Scene Graph Generation from RGB-D Images [ICCV 2025]

[![Project Page](https://img.shields.io/badge/Project-Page-green)](TODO)
[![Paper](https://img.shields.io/badge/Paper-arXiv-green)](TODO)
[![ICCV](https://img.shields.io/badge/ICCV-2025-steelblue)](TODO)
[![Poster](https://img.shields.io/badge/Poster-PDF-blue)](TODO)
[![Code](https://img.shields.io/badge/Code-FROSS-blue)](https://github.com/Howardkhh/FROSS)

</div>

# About
ReplicaSSG extends the [Replica dataset](https://github.com/facebookresearch/Replica-Dataset) with newly annotated object relationships. The original Replica provides high-quality reconstructions of indoor environments with instance-segmented meshes and photorealistic renderings, and is particularly suitable for SSG evaluation. Although Replica encompasses only 18 scenes, which precludes its use for training purposes, it serves as an effective evaluation benchmark. ReplicaSSG adopts the classification system from [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/index.html), a 2D SG dataset. This label mapping from Replica to Visual Genome facilitates zero-shot transfer from 2D SG models, as this approach eliminates the requirement for additional training on the Replica dataset. The structure of this dataset follows the [3DSSG dataset](https://github.com/3DSSG/3DSSG.github.io). As a result, ReplicaSSG offers high-quality instance-segmented meshes with comprehensive relationship annotations.

# Table of Contents
- [Installation](#installation)
- [Preprocessing](#prerocessing)
  - [1. Download the Replica Dataset](#1-download-the-replica-dataset)
  - [2. Fix dataset config file](#2-fix-dataset-config-file)
  - [3. Copy annotations](#3-copy-annotations)
  - [4. Convert mesh to point cloud for FROSS evaluation](#4-convert-mesh-to-point-cloud-for-fross-evaluation)
  - [5. Render image sequences](#5-render-image-sequences)
  - [6. Convert camera trajectories to ORB-SLAM3 format (optional)](#6-convert-camera-trajectories-to-orb-slam3-format-optional)
- [Annotation tools](#annotation-tools)
  - [1. Convert object annotations to 3RScan format (`files/object.json`)](#1-convert-object-annotations-to-3rscan-format-filesobjectjson)
  - [2. Label relationships (`files/relationships.json`)](#2-label-relationships-filesrelationshipsjson)
  - [3. Extract camera path (`files/trajectories.json`)](#3-extract-camera-path-filestrajectoriesjson)
- [Citation](#citation)

## Installation
```bash
# Tested with Python 3.9, Python 3.10 may not work.
# Install habitat-sim for rendering (newer versions may not be compatible for rendering Replica dataset).
# If the installation fails due to CMake version too high, you can try adding `CMAKE_ARGS="-DCMAKE_POLICY_VERSION_MINIMUM=3.5"` in front of the pip install command.
pip install git+https://github.com/facebookresearch/habitat-sim@v0.2.2
pip install numpy==1.26.4 opencv-python==4.10.0.84 noise pyquaternion plyfile
```

## Preprocessing
#### 1. Download the [Replica Dataset](https://github.com/facebookresearch/Replica-Dataset) by Utilizing the Download Script in the Repository.
```bash
./download.sh Replica/data
```

#### 2. Fix dataset config file
The original Replica dataset config file seems to have incorrect settings for the semantic segmentation camera. See [this issue](https://github.com/facebookresearch/Replica-Dataset/issues/89).
```bash
cp files/replica.scene_dataset_config.json Replica/data/replica.scene_dataset_config.json
```

#### 3. Copy annotations
```bash
mkdir Replica/ReplicaSSG
cp files/* Replica/ReplicaSSG
```

#### 4. Convert mesh to point cloud for FROSS evaluation
The `extract_instance_ply.py` script converts point cloud instance segmentation labels from faces to points, following the format used in 3DSSG.
```bash
python extract_instance_ply.py --replica_path Replica
```

#### 5. Render image sequences
You may need a vnc server to run the renderer in a headless environment.
(For example: `vncserver && export DISPLAY=:1.0`)
```bash
for SCENE in apartment_0 apartment_1 apartment_2 frl_apartment_0 frl_apartment_1 frl_apartment_2 frl_apartment_3 frl_apartment_4 frl_apartment_5 hotel_0 office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2; do
    python extract_path.py --replica_path Replica/data \
        --scene $SCENE \
        --replica_to_vg_path Replica/ReplicaSSG/replica_to_visual_genome.json \
        --trajectory_json Replica/ReplicaSSG/trajectories.json \
        --output_dir Replica/data/
done
```

#### 6. Convert camera trajectories to ORB-SLAM3 format (optional)
This is useful for evaluating ORB-SLAM3 performance on the ReplicaSSG dataset.
```bash
python convert_to_SLAM_trajectory.py --replica_path Replica
```

## Annotation tools
The annotated 3D scene graphs and camera paths are already provided in the `files` directory.

However, if you want to manually annotate the dataset, you can use the provided scripts.

#### 1. Convert object annotations to 3RScan format (`files/object.json`)
```bash
python extract_objects.py --replica_path Replica
```

#### 2. Label relationships (`files/relationships.json`)
The `labeler.py` allows for manual labeling of the relationships in the ReplicaSSG dataset.

- W, A, S, D, Z, X keys for moving the camera forward, left, backward, right, up, down.
- Arrow keys for rotating the camera.
- P key for switching the camera mode.
- Type `object_id relationship_name subject_id` in the terminal to label a relationship. For example, `0 on 1`.
- Type `undo` in the terminal to remove the last relationship.
- The relationships are saved in the `relationships.json` file immediately.

```bash
export SCENE="apartment_0"
python labeler.py --replica_path Replica/data --scene $SCENE --replica_to_vg_path Replica/ReplicaSSG/replica_to_visual_genome.json --relationship_path Replica/ReplicaSSG/relationships.json
```

#### 3. Extract camera path (`files/trajectories.json`)
The `extract_path.py` script enables manual selection of camera path keyframes and renders the corresponding camera trajectory. \
If the `--trajectory_json` file already exists, the script will render the existing camera path instead of starting a new keyframe selection process.

- W, A, S, D, Z, X keys for moving the camera forward, left, backward, right, up, down.
- Arrow keys for rotating the camera.
- K key for adding a keyframe.
- Q key for saving the camera path and start rendering.
- If there are camera path for the selected scene, the camera path will be loaded instead of starting a new path.

```bash
python extract_path.py --replica_path Replica/data --scene $SCENE --replica_to_vg_path Replica/ReplicaSSG/replica_to_visual_genome.json --trajectory_json Replica/ReplicaSSG/trajectories.json --output_dir Replica/data/
```

## Citation

```
@InProceedings{hou2025fross,
    author    = {Hao-Yu Hou, Chun-Yi Lee, Motoharu Sonogashira, and Yasutomo Kawanishi},
    title     = {{FROSS}: {F}aster-than-{R}eal-{T}ime {O}nline 3{D} {S}emantic {S}cene {G}raph {G}eneration from {RGB-D} {I}mages},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025}
}
```
```
@article{replica19arxiv,
  title =   {The {R}eplica Dataset: A Digital Replica of Indoor Spaces},
  author =  {Julian Straub and Thomas Whelan and Lingni Ma and Yufan Chen and Erik Wijmans and Simon Green and Jakob J. Engel and Raul Mur-Artal and Carl Ren and Shobhit Verma and Anton Clarkson and Mingfei Yan and Brian Budge and Yajie Yan and Xiaqing Pan and June Yon and Yuyang Zou and Kimberly Leon and Nigel Carter and Jesus Briales and  Tyler Gillingham and  Elias Mueggler and Luis Pesqueira and Manolis Savva and Dhruv Batra and Hauke M. Strasdat and Renzo De Nardi and Michael Goesele and Steven Lovegrove and Richard Newcombe },
  journal = {arXiv preprint arXiv:1906.05797},
  year =    {2019}
}
```