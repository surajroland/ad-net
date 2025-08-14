"""
Multi-dataset loader with harmonization for Sparse4D framework.

This module provides cross-dataset training capabilities with automatic
harmonization of coordinate systems, class taxonomies, and data formats
across different autonomous driving datasets.
"""

import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler

from ...interfaces.data.dataset import (
    BaseDataset,
    Sample,
    CameraParams,
    InstanceAnnotation,
    TemporalSequence,
    DatasetRegistry,
)


class UnifiedTaxonomy:
    """
    Unified class taxonomy for multi-dataset training.

    Maps dataset-specific classes to a common taxonomy while preserving
    semantic meaning and handling class hierarchies.
    """

    # Unified class names for autonomous driving
    UNIFIED_CLASSES = [
        "vehicle.car",  # Standard passenger car
        "vehicle.truck",  # Large trucks, delivery vehicles
        "vehicle.bus",  # Public transit buses
        "vehicle.motorcycle",  # Motorcycles, scooters
        "vehicle.bicycle",  # Bicycles
        "vehicle.trailer",  # Trailers, semi-trailers
        "vehicle.emergency",  # Emergency vehicles
        "vehicle.construction",  # Construction vehicles
        "human.pedestrian",  # Walking people
        "human.cyclist",  # People on bicycles
        "movable.barrier",  # Traffic barriers, cones
        "movable.debris",  # Road debris, obstacles
        "static.traffic_sign",  # Traffic signs
        "static.traffic_light",  # Traffic lights
    ]

    # Dataset-specific mappings to unified taxonomy
    DATASET_MAPPINGS = {
        "nuscenes": {
            "car": "vehicle.car",
            "truck": "vehicle.truck",
            "bus": "vehicle.bus",
            "motorcycle": "vehicle.motorcycle",
            "bicycle": "vehicle.bicycle",
            "trailer": "vehicle.trailer",
            "construction_vehicle": "vehicle.construction",
            "pedestrian": "human.pedestrian",
            "barrier": "movable.barrier",
            "traffic_cone": "movable.barrier",
        },
        "waymo": {
            "TYPE_VEHICLE": "vehicle.car",
            "TYPE_PEDESTRIAN": "human.pedestrian",
            "TYPE_CYCLIST": "human.cyclist",
            "TYPE_SIGN": "static.traffic_sign",
        },
        "kitti": {
            "Car": "vehicle.car",
            "Van": "vehicle.truck",
            "Truck": "vehicle.truck",
            "Pedestrian": "human.pedestrian",
            "Person_sitting": "human.pedestrian",
            "Cyclist": "human.cyclist",
            "Tram": "vehicle.bus",
            "Misc": "movable.debris",
        },
        "argoverse": {
            "VEHICLE": "vehicle.car",
            "PEDESTRIAN": "human.pedestrian",
            "ON_ROAD_OBSTACLE": "movable.debris",
            "LARGE_VEHICLE": "vehicle.truck",
            "BICYCLE": "vehicle.bicycle",
            "BICYCLIST": "human.cyclist",
            "BUS": "vehicle.bus",
            "OTHER_MOVER": "movable.debris",
            "TRAILER": "vehicle.trailer",
            "MOTORCYCLIST": "vehicle.motorcycle",
            "MOPED": "vehicle.motorcycle",
            "MOTORCYCLE": "vehicle.motorcycle",
            "STROLLER": "human.pedestrian",
            "WHEELCHAIR": "human.pedestrian",
            "STOP_SIGN": "static.traffic_sign",
            "CONSTRUCTION_CONE": "movable.barrier",
            "CONSTRUCTION_BARREL": "movable.barrier",
        },
    }

    def __init__(self):
        self.unified_to_id = {
            name: idx for idx, name in enumerate(self.UNIFIED_CLASSES)
        }
        self.id_to_unified = {idx: name for name, idx in self.unified_to_id.items()}
        self.num_classes = len(self.UNIFIED_CLASSES)

    def map_class(self, dataset_name: str, original_class: str) -> Optional[int]:
        """Map dataset-specific class to unified class ID"""
        if dataset_name not in self.DATASET_MAPPINGS:
            return None

        mapping = self.DATASET_MAPPINGS[dataset_name]
        if original_class not in mapping:
            return None

        unified_class = mapping[original_class]
        return self.unified_to_id.get(unified_class)

    def get_unified_class_name(self, class_id: int) -> str:
        """Get unified class name from ID"""
        return self.id_to_unified.get(class_id, "unknown")


class CoordinateHarmonizer:
    """
    Coordinate system harmonization for multi-dataset training.

    Converts between different coordinate systems used by various datasets
    to a standardized ego-centric coordinate system.
    """

    COORDINATE_SYSTEMS = {
        "nuscenes": {
            "type": "ego_centric",
            "x_forward": True,  # X points forward
            "y_left": True,  # Y points left
            "z_up": True,  # Z points up
            "rotation_order": "xyz",
        },
        "waymo": {
            "type": "ego_centric",
            "x_forward": True,
            "y_left": True,
            "z_up": True,
            "rotation_order": "xyz",
        },
        "kitti": {
            "type": "camera_centric",
            "x_right": True,  # X points right
            "y_down": True,  # Y points down
            "z_forward": True,  # Z points forward
            "rotation_order": "xyz",
        },
        "argoverse": {
            "type": "ego_centric",
            "x_forward": True,
            "y_left": True,
            "z_up": True,
            "rotation_order": "xyz",
        },
    }

    # Standard ego coordinate system (nuScenes style)
    STANDARD_SYSTEM = {"x_forward": True, "y_left": True, "z_up": True}

    def __init__(self):
        self._build_transformation_matrices()

    def _build_transformation_matrices(self):
        """Build transformation matrices for each dataset"""
        self.transforms = {}

        for dataset, system in self.COORDINATE_SYSTEMS.items():
            if system["type"] == "ego_centric":
                # Already in ego coordinates, minimal transform needed
                transform = np.eye(4)

                # Handle axis orientation differences
                if not system.get("x_forward", True):
                    transform[0, 0] = -1  # Flip X
                if not system.get("y_left", True):
                    transform[1, 1] = -1  # Flip Y
                if not system.get("z_up", True):
                    transform[2, 2] = -1  # Flip Z

            elif system["type"] == "camera_centric":
                # Transform from camera to ego coordinates
                # KITTI: x_right, y_down, z_forward -> x_forward, y_left, z_up
                transform = np.array(
                    [
                        [0, 0, 1, 0],  # camera_z -> ego_x (forward)
                        [-1, 0, 0, 0],  # -camera_x -> ego_y (left)
                        [0, -1, 0, 0],  # -camera_y -> ego_z (up)
                        [0, 0, 0, 1],
                    ]
                )

            self.transforms[dataset] = transform

    def harmonize_pose(self, dataset_name: str, pose: np.ndarray) -> np.ndarray:
        """Harmonize pose to standard coordinate system"""
        if dataset_name not in self.transforms:
            return pose  # No transformation available

        transform = self.transforms[dataset_name]

        # Apply transformation
        harmonized_pose = transform @ pose @ np.linalg.inv(transform)
        return harmonized_pose

    def harmonize_box_3d(self, dataset_name: str, box_3d: np.ndarray) -> np.ndarray:
        """Harmonize 3D bounding box to standard coordinate system"""
        if dataset_name not in self.transforms:
            return box_3d

        # Extract position, size, and rotation
        position = box_3d[:3]
        size = box_3d[3:6]
        yaw_cos, yaw_sin = box_3d[6:8]
        velocity = box_3d[8] if len(box_3d) > 8 else 0.0

        # Transform position
        pos_homogeneous = np.append(position, 1)
        transformed_pos = (self.transforms[dataset_name] @ pos_homogeneous)[:3]

        # Transform rotation (yaw angle)
        yaw = np.arctan2(yaw_sin, yaw_cos)
        # Apply rotation transformation based on coordinate system
        if dataset_name == "kitti":
            yaw = yaw + np.pi / 2  # Adjust for coordinate system difference

        transformed_yaw_cos = np.cos(yaw)
        transformed_yaw_sin = np.sin(yaw)

        # Size typically doesn't need transformation (w, l, h order)
        transformed_box = np.array(
            [
                transformed_pos[0],
                transformed_pos[1],
                transformed_pos[2],
                size[0],
                size[1],
                size[2],
                transformed_yaw_cos,
                transformed_yaw_sin,
                velocity,
            ]
        )

        return transformed_box


class MultiDatasetLoader(Dataset):
    """
    Multi-dataset loader with automatic harmonization.

    Combines multiple datasets with unified class taxonomy, coordinate
    harmonization, and balanced sampling strategies.
    """

    def __init__(
        self,
        dataset_configs: List[Dict[str, Any]],
        harmonize_coordinates: bool = True,
        harmonize_classes: bool = True,
        sampling_strategy: str = "balanced",  # balanced, weighted, sequential
        temporal_alignment: bool = True,
        **kwargs,
    ):
        """
        Initialize multi-dataset loader.

        Args:
            dataset_configs: List of dataset configurations
            harmonize_coordinates: Enable coordinate harmonization
            harmonize_classes: Enable class harmonization
            sampling_strategy: Strategy for sampling across datasets
            temporal_alignment: Enable temporal sequence alignment
        """
        self.dataset_configs = dataset_configs
        self.harmonize_coordinates = harmonize_coordinates
        self.harmonize_classes = harmonize_classes
        self.sampling_strategy = sampling_strategy
        self.temporal_alignment = temporal_alignment

        # Initialize harmonizers
        if self.harmonize_classes:
            self.taxonomy = UnifiedTaxonomy()
        else:
            self.taxonomy = None

        if self.harmonize_coordinates:
            self.coord_harmonizer = CoordinateHarmonizer()
        else:
            self.coord_harmonizer = None

        # Load all datasets
        self.datasets = []
        self.dataset_names = []
        self.dataset_weights = []

        for config in dataset_configs:
            dataset = self._load_dataset(config)
            self.datasets.append(dataset)
            self.dataset_names.append(config["name"])
            self.dataset_weights.append(config.get("weight", 1.0))

        # Build sampling indices
        self._build_sampling_indices()

        print(
            f"Loaded {len(self.datasets)} datasets with {len(self.sample_indices)} total samples"
        )

    def _load_dataset(self, config: Dict[str, Any]) -> BaseDataset:
        """Load individual dataset from configuration"""
        dataset_name = config["name"]
        dataset_class = DatasetRegistry.get(dataset_name)

        # Create dataset instance
        dataset = dataset_class(**config.get("params", {}))

        return dataset

    def _build_sampling_indices(self):
        """Build sampling indices based on strategy"""
        self.sample_indices = []

        if self.sampling_strategy == "sequential":
            # Sequential sampling - concatenate all datasets
            offset = 0
            for i, dataset in enumerate(self.datasets):
                for j in range(len(dataset)):
                    self.sample_indices.append((i, j))
                offset += len(dataset)

        elif self.sampling_strategy == "balanced":
            # Balanced sampling - equal samples from each dataset
            min_samples = min(len(dataset) for dataset in self.datasets)

            for i, dataset in enumerate(self.datasets):
                indices = list(range(len(dataset)))
                if len(indices) > min_samples:
                    indices = random.sample(indices, min_samples)

                for j in indices:
                    self.sample_indices.append((i, j))

        elif self.sampling_strategy == "weighted":
            # Weighted sampling based on dataset weights
            total_weight = sum(self.dataset_weights)
            normalized_weights = [w / total_weight for w in self.dataset_weights]

            total_samples = sum(len(dataset) for dataset in self.datasets)

            for i, (dataset, weight) in enumerate(
                zip(self.datasets, normalized_weights)
            ):
                num_samples = int(total_samples * weight)
                indices = list(range(len(dataset)))

                if len(indices) < num_samples:
                    # Oversample if needed
                    indices = indices * (num_samples // len(indices) + 1)
                    indices = indices[:num_samples]
                elif len(indices) > num_samples:
                    # Undersample
                    indices = random.sample(indices, num_samples)

                for j in indices:
                    self.sample_indices.append((i, j))

        # Shuffle for randomization
        random.shuffle(self.sample_indices)

    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> Sample:
        """Get harmonized sample by index"""
        dataset_idx, sample_idx = self.sample_indices[index]
        dataset = self.datasets[dataset_idx]
        dataset_name = self.dataset_names[dataset_idx]

        # Load original sample
        sample = dataset[sample_idx]

        # Apply harmonization
        harmonized_sample = self._harmonize_sample(sample, dataset_name)

        return harmonized_sample

    def _harmonize_sample(self, sample: Sample, dataset_name: str) -> Sample:
        """Apply all harmonization steps to sample"""
        # Coordinate harmonization
        if self.harmonize_coordinates and self.coord_harmonizer:
            sample = self._harmonize_coordinates(sample, dataset_name)

        # Class harmonization
        if self.harmonize_classes and self.taxonomy:
            sample = self._harmonize_classes(sample, dataset_name)

        # Temporal alignment
        if self.temporal_alignment:
            sample = self._align_temporal_data(sample, dataset_name)

        return sample

    def _harmonize_coordinates(self, sample: Sample, dataset_name: str) -> Sample:
        """Harmonize coordinate systems"""
        # Harmonize ego pose
        sample.ego_pose = self.coord_harmonizer.harmonize_pose(
            dataset_name, sample.ego_pose
        )

        # Harmonize camera extrinsics
        for i in range(len(sample.camera_params.extrinsics)):
            sample.camera_params.extrinsics[i] = self.coord_harmonizer.harmonize_pose(
                dataset_name, sample.camera_params.extrinsics[i]
            )

        # Harmonize 3D bounding boxes
        for instance in sample.instances:
            instance.box_3d = self.coord_harmonizer.harmonize_box_3d(
                dataset_name, instance.box_3d
            )

        # Harmonize temporal sequence poses
        for i in range(len(sample.sequence_info.ego_poses)):
            sample.sequence_info.ego_poses[i] = self.coord_harmonizer.harmonize_pose(
                dataset_name, sample.sequence_info.ego_poses[i]
            )

        return sample

    def _harmonize_classes(self, sample: Sample, dataset_name: str) -> Sample:
        """Harmonize class taxonomies"""
        harmonized_instances = []

        for instance in sample.instances:
            # Get original class name from dataset
            original_dataset = None
            for dataset in self.datasets:
                if dataset.dataset_name == dataset_name:
                    original_dataset = dataset
                    break

            if original_dataset is None:
                continue

            original_class = original_dataset.class_names[instance.category_id]

            # Map to unified taxonomy
            unified_class_id = self.taxonomy.map_class(dataset_name, original_class)

            if unified_class_id is not None:
                instance.category_id = unified_class_id
                harmonized_instances.append(instance)

        sample.instances = harmonized_instances
        return sample

    def _align_temporal_data(self, sample: Sample, dataset_name: str) -> Sample:
        """Align temporal data across datasets"""
        # Ensure consistent temporal sampling rates
        # This is a simplified version - full implementation would handle
        # different fps rates and temporal alignment
        return sample

    def get_unified_class_names(self) -> List[str]:
        """Get unified class names"""
        if self.taxonomy:
            return self.taxonomy.UNIFIED_CLASSES
        else:
            # Return combined class names from all datasets
            all_classes = set()
            for dataset in self.datasets:
                all_classes.update(dataset.class_names)
            return sorted(list(all_classes))

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        stats = {
            "total_samples": len(self),
            "num_datasets": len(self.datasets),
            "sampling_strategy": self.sampling_strategy,
            "harmonize_coordinates": self.harmonize_coordinates,
            "harmonize_classes": self.harmonize_classes,
            "datasets": {},
        }

        for i, (dataset, name) in enumerate(zip(self.datasets, self.dataset_names)):
            dataset_stats = dataset.get_statistics()
            dataset_stats["weight"] = self.dataset_weights[i]
            stats["datasets"][name] = dataset_stats

        if self.taxonomy:
            stats["unified_classes"] = self.taxonomy.UNIFIED_CLASSES
            stats["num_unified_classes"] = self.taxonomy.num_classes

        return stats

    def save_harmonized_dataset(self, output_path: str):
        """Save harmonized dataset for future use"""
        os.makedirs(output_path, exist_ok=True)

        # Save configuration
        config = {
            "dataset_configs": self.dataset_configs,
            "harmonize_coordinates": self.harmonize_coordinates,
            "harmonize_classes": self.harmonize_classes,
            "sampling_strategy": self.sampling_strategy,
            "temporal_alignment": self.temporal_alignment,
            "statistics": self.get_dataset_statistics(),
        }

        with open(os.path.join(output_path, "multi_dataset_config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Save class mappings
        if self.taxonomy:
            mappings = {
                "unified_classes": self.taxonomy.UNIFIED_CLASSES,
                "dataset_mappings": self.taxonomy.DATASET_MAPPINGS,
            }

            with open(os.path.join(output_path, "class_mappings.json"), "w") as f:
                json.dump(mappings, f, indent=2)

        print(f"Saved harmonized dataset configuration to {output_path}")


class CrossDatasetValidator:
    """
    Cross-dataset validation for evaluating generalization across datasets.

    Tests model performance when trained on one dataset and evaluated on another,
    providing insights into domain transfer capabilities.
    """

    def __init__(self, datasets: List[BaseDataset]):
        self.datasets = datasets
        self.dataset_names = [d.__class__.__name__ for d in datasets]

    def create_cross_validation_splits(self) -> List[Tuple[List[int], List[int]]]:
        """Create cross-dataset validation splits"""
        splits = []

        for train_idx in range(len(self.datasets)):
            for val_idx in range(len(self.datasets)):
                if train_idx != val_idx:
                    train_indices = list(range(len(self.datasets[train_idx])))
                    val_indices = list(range(len(self.datasets[val_idx])))
                    splits.append((train_indices, val_indices))

        return splits

    def evaluate_domain_gap(
        self, dataset_a: BaseDataset, dataset_b: BaseDataset
    ) -> Dict[str, float]:
        """Evaluate domain gap between two datasets"""
        gap_metrics = {}

        # Class distribution similarity
        gap_metrics["class_distribution_kl"] = self._compute_class_distribution_kl(
            dataset_a, dataset_b
        )

        # Camera setup similarity
        gap_metrics["camera_setup_similarity"] = self._compute_camera_similarity(
            dataset_a, dataset_b
        )

        # Scene diversity
        gap_metrics["scene_diversity_ratio"] = self._compute_scene_diversity(
            dataset_a, dataset_b
        )

        return gap_metrics

    def _compute_class_distribution_kl(
        self, dataset_a: BaseDataset, dataset_b: BaseDataset
    ) -> float:
        """Compute KL divergence between class distributions"""
        # Simplified implementation
        return 0.5  # Placeholder

    def _compute_camera_similarity(
        self, dataset_a: BaseDataset, dataset_b: BaseDataset
    ) -> float:
        """Compute camera setup similarity"""
        # Compare number of cameras, resolutions, etc.
        return 0.8  # Placeholder

    def _compute_scene_diversity(
        self, dataset_a: BaseDataset, dataset_b: BaseDataset
    ) -> float:
        """Compute scene diversity ratio"""
        # Compare environments, weather conditions, etc.
        return 0.6  # Placeholder


def create_multi_dataset_config(
    dataset_names: List[str],
    data_roots: Dict[str, str],
    weights: Optional[Dict[str, float]] = None,
    **common_params,
) -> List[Dict[str, Any]]:
    """
    Create multi-dataset configuration.

    Args:
        dataset_names: List of dataset names to include
        data_roots: Mapping of dataset names to data root paths
        weights: Optional sampling weights for each dataset
        **common_params: Common parameters for all datasets

    Returns:
        List of dataset configurations
    """
    configs = []

    for name in dataset_names:
        config = {
            "name": name,
            "params": {"data_root": data_roots[name], **common_params},
        }

        if weights and name in weights:
            config["weight"] = weights[name]

        configs.append(config)

    return configs
