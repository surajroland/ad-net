"""nuScenes dataset implementation for Sparse4D framework.

This module provides a complete nuScenes dataset loader with support for:
- Multi-view camera data (6 surround cameras)
- 3D bounding box annotations with tracking IDs
- Temporal sequences for 4D object detection
- Camera calibration and ego pose information
- Instance tracking across frames
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt
from PIL import Image

from ...interfaces.data.dataset import (
    CameraParams,
    InstanceAnnotation,
    MultiModalDataset,
    Sample,
    TemporalDataset,
    TemporalSequence,
    register_dataset,
)


@register_dataset("nuscenes")
class NuScenesDataset(TemporalDataset, MultiModalDataset):
    """nuScenes dataset implementation for Sparse4D.

    Supports the standard nuScenes dataset format with multi-view cameras,
    3D annotations, and temporal sequences for 4D object detection.
    """

    # nuScenes camera configuration (6 surround-view cameras)
    CAMERA_NAMES = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_FRONT_LEFT",
        "CAM_BACK",
        "CAM_BACK_RIGHT",
        "CAM_BACK_LEFT",
    ]

    # nuScenes object classes
    CLASS_NAMES = [
        "car",
        "truck",
        "construction_vehicle",
        "bus",
        "trailer",
        "barrier",
        "motorcycle",
        "bicycle",
        "pedestrian",
        "traffic_cone",
    ]

    # Attribute mapping for nuScenes
    ATTRIBUTE_MAPPING = {
        "moving": ["vehicle.moving"],
        "stopped": ["vehicle.stopped"],
        "parked": ["vehicle.parked"],
        "sitting": ["pedestrian.sitting_lying_down"],
        "standing": ["pedestrian.standing"],
        "walking": ["pedestrian.moving"],
    }

    def __init__(
        self,
        data_root: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        sequence_length: int = 1,
        temporal_stride: int = 1,
        load_interval: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize nuScenes dataset.

        Args:
            data_root: Root directory containing nuScenes data
            version: Dataset version ('v1.0-trainval', 'v1.0-test', 'v1.0-mini')
            split: Dataset split ('train', 'val', 'test')
            sequence_length: Number of frames in temporal sequences
            temporal_stride: Stride between frames in sequence
            load_interval: Interval for loading frames
            **kwargs: Additional keyword arguments passed to parent classes

        """
        self.version = version
        self.data_root = data_root

        # Initialize parent classes
        super().__init__(
            data_root=data_root,
            split=split,
            sequence_length=sequence_length,
            temporal_stride=temporal_stride,
            load_interval=load_interval,
            **kwargs,
        )

        # Load nuScenes database
        self._load_nuscenes_db()

        # Build temporal sequences if enabled
        if self.enable_temporal and self.sequence_length > 1:
            self._build_temporal_sequences()

    def _load_dataset_info(self) -> None:
        """Load nuScenes-specific dataset information."""
        self.class_names = self.CLASS_NAMES
        self.camera_names = self.CAMERA_NAMES
        self.num_classes = len(self.CLASS_NAMES)
        self.num_cameras = len(self.CAMERA_NAMES)

        # Class name to ID mapping
        self.class_to_id = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}

    def _load_nuscenes_db(self) -> None:
        """Load nuScenes database files."""
        db_path = os.path.join(self.data_root, self.version)

        # Load all database tables
        self.db = {}
        for table in [
            "scene",
            "sample",
            "sample_data",
            "sample_annotation",
            "ego_pose",
            "calibrated_sensor",
            "sensor",
            "log",
        ]:
            table_path = os.path.join(db_path, f"{table}.json")
            if os.path.exists(table_path):
                with open(table_path, "r") as f:
                    self.db[table] = {item["token"]: item for item in json.load(f)}

        # Load category mapping
        category_path = os.path.join(db_path, "category.json")
        if os.path.exists(category_path):
            with open(category_path, "r") as f:
                categories = json.load(f)
                self.category_mapping = {
                    cat["token"]: cat["name"] for cat in categories
                }

        # Load attribute mapping
        attribute_path = os.path.join(db_path, "attribute.json")
        if os.path.exists(attribute_path):
            with open(attribute_path, "r") as f:
                attributes = json.load(f)
                self.attribute_mapping = {
                    attr["token"]: attr["name"] for attr in attributes
                }

    def _load_annotations(self) -> None:
        """Load and parse nuScenes annotations."""
        # Filter samples by split
        if self.split == "train":
            scene_splits = self._get_train_scenes()
        elif self.split == "val":
            scene_splits = self._get_val_scenes()
        else:  # test
            scene_splits = self._get_test_scenes()

        # Collect samples from selected scenes
        self._sample_tokens: List[str] = []
        self._sample_data: Dict[str, Any] = {}

        for scene_token in scene_splits:
            scene = self.db["scene"][scene_token]

            # Walk through all samples in scene
            sample_token = scene["first_sample_token"]
            while sample_token != "":
                sample = self.db["sample"][sample_token]

                # Apply load interval
                if len(self._sample_tokens) % self.load_interval == 0:
                    self._sample_tokens.append(sample_token)
                    self._sample_data[sample_token] = sample

                sample_token = sample["next"]

        print(f"Loaded {len(self._sample_tokens)} samples for {self.split} split")

    def _get_train_scenes(self) -> List[str]:
        """Get scene tokens for training split."""
        # Load splits file
        splits_path = os.path.join(self.data_root, self.version, "train.txt")
        if os.path.exists(splits_path):
            with open(splits_path, "r") as f:
                scene_names = [line.strip() for line in f.readlines()]
        else:
            # Fallback: use first 70% of scenes
            all_scenes = list(self.db["scene"].keys())
            scene_names = all_scenes[: int(0.7 * len(all_scenes))]

        return [
            token
            for token, scene in self.db["scene"].items()
            if scene["name"] in scene_names
        ]

    def _get_val_scenes(self) -> List[str]:
        """Get scene tokens for validation split."""
        splits_path = os.path.join(self.data_root, self.version, "val.txt")
        if os.path.exists(splits_path):
            with open(splits_path, "r") as f:
                scene_names = [line.strip() for line in f.readlines()]
        else:
            # Fallback: use 70-85% of scenes
            all_scenes = list(self.db["scene"].keys())
            start_idx = int(0.7 * len(all_scenes))
            end_idx = int(0.85 * len(all_scenes))
            scene_names = all_scenes[start_idx:end_idx]

        return [
            token
            for token, scene in self.db["scene"].items()
            if scene["name"] in scene_names
        ]

    def _get_test_scenes(self) -> List[str]:
        """Get scene tokens for test split."""
        splits_path = os.path.join(self.data_root, self.version, "test.txt")
        if os.path.exists(splits_path):
            with open(splits_path, "r") as f:
                scene_names = [line.strip() for line in f.readlines()]
        else:
            # Fallback: use last 15% of scenes
            all_scenes = list(self.db["scene"].keys())
            scene_names = all_scenes[int(0.85 * len(all_scenes)) :]

        return [
            token
            for token, scene in self.db["scene"].items()
            if scene["name"] in scene_names
        ]

    @property
    def sample_ids(self) -> List[str]:
        """List of all sample tokens in dataset"""
        return self._sample_tokens

    def _load_sample_data(self, index: int) -> Sample:
        """Load complete sample data for given index"""
        sample_token = self._sample_tokens[index]
        sample = self._sample_data[sample_token]

        # Load images and camera parameters
        images = {}
        camera_intrinsics = []
        camera_extrinsics = []
        camera_timestamps = []

        for cam_name in self.CAMERA_NAMES:
            # Get camera sample data
            cam_token = sample["data"][cam_name]
            cam_data = self.db["sample_data"][cam_token]

            # Load image
            img_path = os.path.join(self.data_root, cam_data["filename"])
            image = np.array(Image.open(img_path))
            images[cam_name] = image

            # Get calibration
            calib_token = cam_data["calibrated_sensor_token"]
            calibration = self.db["calibrated_sensor"][calib_token]

            # Camera intrinsics (3x3 matrix)
            intrinsic = np.array(calibration["camera_intrinsic"])
            camera_intrinsics.append(intrinsic)

            # Camera extrinsics (4x4 matrix)
            extrinsic = self._build_extrinsic_matrix(
                calibration["translation"], calibration["rotation"]
            )
            camera_extrinsics.append(extrinsic)

            # Timestamp
            camera_timestamps.append(cam_data["timestamp"])

        # Create camera parameters
        camera_params = CameraParams(
            intrinsics=np.array(camera_intrinsics),
            extrinsics=np.array(camera_extrinsics),
            timestamps=np.array(camera_timestamps),
        )

        # Load annotations
        instances = self._load_sample_annotations(sample_token)

        # Get ego pose
        ego_pose_token = sample["data"]["LIDAR_TOP"]  # Use LIDAR timestamp as reference
        ego_data = self.db["sample_data"][ego_pose_token]
        ego_pose = self._get_ego_pose(ego_data["ego_pose_token"])

        # Get sequence information
        sequence_info = self.get_temporal_sequence(sample_token)

        # Create sample
        sample_obj = Sample(
            sample_id=sample_token,
            dataset_name="nuscenes",
            sequence_info=sequence_info,
            images=images,
            camera_params=camera_params,
            instances=instances,
            ego_pose=ego_pose,
            location=self._get_sample_location(sample),
            weather=self._get_sample_weather(sample),
            time_of_day=self._get_sample_time(sample),
        )

        # Load additional modalities if requested
        if self.load_lidar:
            sample_obj.lidar_points = self._load_lidar_data(sample_token)

        if self.load_radar:
            sample_obj.radar_points = self._load_radar_data(sample_token)

        if self.load_depth:
            sample_obj.depth_maps = self._load_depth_data(sample_token)

        return sample_obj

    def _build_extrinsic_matrix(
        self, translation: List[float], rotation: List[float]
    ) -> npt.NDArray[np.float64]:
        """Build 4x4 extrinsic matrix from translation and rotation"""
        from scipy.spatial.transform import Rotation

        # Convert quaternion to rotation matrix
        rot = Rotation.from_quat(rotation)
        rot_matrix = rot.as_matrix()

        # Build 4x4 extrinsic matrix
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rot_matrix
        extrinsic[:3, 3] = translation

        return extrinsic

    def _load_sample_annotations(self, sample_token: str) -> List[InstanceAnnotation]:
        """Load 3D annotations for sample"""
        sample = self.db["sample"][sample_token]
        instances = []

        for ann_token in sample["anns"]:
            ann = self.db["sample_annotation"][ann_token]

            # Get category name
            category_name = self.category_mapping[ann["category_token"]]

            # Skip if not in our class list
            if category_name not in self.class_to_id:
                continue

            # Build 3D bounding box [x, y, z, w, l, h, cos(yaw), sin(yaw), velocity]
            center = ann["translation"]
            size = ann["size"]  # [w, l, h]
            rotation = ann["rotation"]  # quaternion

            # Convert quaternion to yaw angle
            from scipy.spatial.transform import Rotation

            rot = Rotation.from_quat(rotation)
            yaw = rot.as_euler("xyz")[2]  # Get yaw angle

            # Get velocity if available
            velocity = 0.0
            if "velocity" in ann and ann["velocity"] is not None:
                velocity = float(
                    np.linalg.norm(ann["velocity"][:2])
                )  # 2D velocity magnitude

            box_3d = np.array(
                [
                    center[0],
                    center[1],
                    center[2],  # x, y, z
                    size[0],
                    size[1],
                    size[2],  # w, l, h
                    np.cos(yaw),
                    np.sin(yaw),  # cos(yaw), sin(yaw)
                    velocity,  # velocity magnitude
                ]
            )

            # Get attributes
            attributes = {}
            for attr_token in ann["attribute_tokens"]:
                attr_name = self.attribute_mapping[attr_token]
                attributes[attr_name] = True

            # Create instance annotation
            instance = InstanceAnnotation(
                box_3d=box_3d,
                category_id=self.class_to_id[category_name],
                instance_id=ann["instance_token"],
                visibility=ann["visibility_token"],
                attributes=attributes,
                confidence=1.0,  # nuScenes annotations are high confidence
            )

            instances.append(instance)

        return instances

    def _get_ego_pose(self, ego_pose_token: str) -> npt.NDArray[np.float64]:
        """Get 4x4 ego pose matrix"""
        ego_pose = self.db["ego_pose"][ego_pose_token]

        translation = ego_pose["translation"]
        rotation = ego_pose["rotation"]

        return self._build_extrinsic_matrix(translation, rotation)

    def get_camera_calibration(self, sample_token: str) -> CameraParams:
        """Get camera calibration parameters for sample"""
        sample = self._sample_data[sample_token]

        camera_intrinsics = []
        camera_extrinsics = []
        camera_timestamps = []

        for cam_name in self.CAMERA_NAMES:
            cam_token = sample["data"][cam_name]
            cam_data = self.db["sample_data"][cam_token]

            # Get calibration
            calib_token = cam_data["calibrated_sensor_token"]
            calibration = self.db["calibrated_sensor"][calib_token]

            camera_intrinsics.append(np.array(calibration["camera_intrinsic"]))
            camera_extrinsics.append(
                self._build_extrinsic_matrix(
                    calibration["translation"], calibration["rotation"]
                )
            )
            camera_timestamps.append(cam_data["timestamp"])

        return CameraParams(
            intrinsics=np.array(camera_intrinsics),
            extrinsics=np.array(camera_extrinsics),
            timestamps=np.array(camera_timestamps),
        )

    def get_temporal_sequence(self, sample_token: str) -> TemporalSequence:
        """Get temporal sequence information for sample"""
        sample = self._sample_data[sample_token]
        scene_token = sample["scene_token"]
        scene = self.db["scene"][scene_token]

        # Collect all samples in scene
        scene_samples = []
        scene_timestamps = []
        scene_poses = []

        current_token = scene["first_sample_token"]
        while current_token != "":
            current_sample = self.db["sample"][current_token]
            scene_samples.append(current_token)
            scene_timestamps.append(current_sample["timestamp"])

            # Get ego pose
            ego_token = current_sample["data"]["LIDAR_TOP"]
            ego_data = self.db["sample_data"][ego_token]
            ego_pose = self._get_ego_pose(ego_data["ego_pose_token"])
            scene_poses.append(ego_pose)

            current_token = current_sample["next"]

        # Find current sample index
        current_index = scene_samples.index(sample_token)

        # Build sequence around current sample
        start_idx = max(0, current_index - self.sequence_length + 1)
        end_idx = current_index + 1

        sequence_indices = list(range(start_idx, end_idx))
        # TODO: Add frame validation and preprocessing if needed

        return TemporalSequence(
            sequence_id=scene_token,
            frame_indices=sequence_indices,
            timestamps=np.array([scene_timestamps[i] for i in sequence_indices]),
            ego_poses=np.array([scene_poses[i] for i in sequence_indices]),
            frame_count=len(scene_samples),
        )

    def _compute_ego_motion(
        self, prev_pose: npt.NDArray[np.float64], curr_pose: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute ego motion between two poses"""
        # Compute relative transformation
        relative_transform = np.linalg.inv(prev_pose) @ curr_pose

        # Extract translation and rotation
        translation = relative_transform[:3, 3]
        rotation_matrix = relative_transform[:3, :3]

        # Convert rotation matrix to euler angles
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_matrix(rotation_matrix)
        euler = rot.as_euler("xyz")

        # Return [dx, dy, dz, roll, pitch, yaw]
        ego_motion = np.concatenate([translation, euler])
        return ego_motion

    def _track_instances(
        self, sequence: TemporalSequence
    ) -> Dict[int, List[InstanceAnnotation]]:
        """Track instances across temporal sequence"""
        # This is a simplified version - full implementation would use
        # nuScenes instance tokens for perfect tracking
        tracks = defaultdict(list)

        for frame_idx in sequence.frame_indices:
            sample_token = self._sample_tokens[frame_idx]
            instances = self._load_sample_annotations(sample_token)

            for instance in instances:
                tracks[instance.instance_id].append(instance)

        return dict(tracks)

    def _load_lidar_data(self, sample_token: str) -> Optional[npt.NDArray[np.float32]]:
        """Load LiDAR point cloud data."""
        if not self.load_lidar:
            return None

        sample = self._sample_data[sample_token]
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = self.db["sample_data"][lidar_token]

        # Load point cloud file
        pc_path = os.path.join(self.data_root, lidar_data["filename"])
        points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 5)

        # Return [x, y, z, intensity] (drop ring index)
        return points[:, :4]

    def _load_radar_data(self, sample_token: str) -> Optional[npt.NDArray[np.float32]]:
        """Load radar point data."""
        if not self.load_radar:
            return None

        sample = self._sample_data[sample_token]
        radar_points = []

        # nuScenes has 5 radar sensors
        radar_names = [
            "RADAR_FRONT",
            "RADAR_FRONT_LEFT",
            "RADAR_FRONT_RIGHT",
            "RADAR_BACK_LEFT",
            "RADAR_BACK_RIGHT",
        ]

        for radar_name in radar_names:
            if radar_name in sample["data"]:
                radar_token = sample["data"][radar_name]
                radar_data = self.db["sample_data"][radar_token]

                # Load radar file
                radar_path = os.path.join(self.data_root, radar_data["filename"])
                points = np.fromfile(radar_path, dtype=np.float32).reshape(-1, 18)

                # Extract [x, y, z, vx, vy, rcs]
                radar_subset = points[:, [0, 1, 2, 6, 7, 9]]
                radar_points.append(radar_subset)

        if radar_points:
            return np.concatenate(radar_points, axis=0)
        return None

    def _load_depth_data(
        self, sample_token: str
    ) -> Optional[Dict[str, npt.NDArray[np.float32]]]:
        """Load depth maps for all cameras."""
        if not self.load_depth:
            return None

        # nuScenes doesn't provide depth maps by default
        # This would be for custom depth estimation or external depth data
        return None

    def _get_sample_location(self, sample: Dict[str, Any]) -> str:
        """Get sample location information."""
        scene = self.db["scene"][sample["scene_token"]]
        log = self.db["log"][scene["log_token"]]
        return log["location"]

    def _get_sample_weather(self, sample: Dict[str, Any]) -> str:
        """Get sample weather information."""
        # nuScenes doesn't have explicit weather labels
        # This could be inferred from scene description or external data
        return "clear"

    def _get_sample_time(self, sample: Dict[str, Any]) -> str:
        """Get sample time of day."""
        scene = self.db["scene"][sample["scene_token"]]
        description = scene["description"].lower()

        if "night" in description:
            return "night"
        elif "evening" in description or "dusk" in description:
            return "evening"
        else:
            return "day"
