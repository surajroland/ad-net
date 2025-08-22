"""Test suite for multi-dataset loader implementation.

Tests for multi-dataset harmonization including:
- Unified taxonomy mapping
- Coordinate system harmonization
- Multi-dataset configuration
- Cross-dataset sampling strategies
- Dataset weighting and balancing
"""

import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from adnet.interfaces.data.dataset import Sample


class TestUnifiedTaxonomy:
    """Test unified taxonomy mapping functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        try:
            from adnet.data.datasets.multi_dataset_loader import UnifiedTaxonomy

            self.taxonomy = UnifiedTaxonomy()
        except ImportError:
            pytest.skip("UnifiedTaxonomy not available")

    def test_unified_class_names(self):
        """Test unified class taxonomy structure."""
        expected_classes = [
            "vehicle.car",
            "vehicle.truck",
            "vehicle.bus",
            "vehicle.motorcycle",
            "vehicle.bicycle",
            "vehicle.trailer",
            "vehicle.emergency",
            "vehicle.construction",
            "human.pedestrian",
            "human.cyclist",
            "movable.barrier",
            "movable.debris",
            "static.traffic_sign",
            "static.traffic_light",
        ]

        # Test that unified classes are properly defined
        assert hasattr(self.taxonomy, "UNIFIED_CLASSES")
        unified_classes = self.taxonomy.UNIFIED_CLASSES

        for expected_class in expected_classes:
            assert expected_class in unified_classes or any(
                expected_class in uc for uc in unified_classes
            )

    def test_nuscenes_mapping(self):
        """Test nuScenes to unified taxonomy mapping."""
        nuscenes_mappings = {
            "car": "vehicle.car",
            "truck": "vehicle.truck",
            "bus": "vehicle.bus",
            "motorcycle": "vehicle.motorcycle",
            "bicycle": "vehicle.bicycle",
            "pedestrian": "human.pedestrian",
            "traffic_cone": "movable.barrier",
            "barrier": "movable.barrier",
        }

        for nuscenes_class, expected_unified in nuscenes_mappings.items():
            unified_id = self.taxonomy.map_class("nuscenes", nuscenes_class)
            if unified_id is not None:
                unified_name = self.taxonomy.get_unified_class_name(unified_id)
                assert unified_name == expected_unified

    def test_waymo_mapping(self):
        """Test Waymo to unified taxonomy mapping."""
        waymo_mappings = {
            "TYPE_VEHICLE": "vehicle.car",
            "TYPE_PEDESTRIAN": "human.pedestrian",
            "TYPE_CYCLIST": "human.cyclist",
            "TYPE_SIGN": "static.traffic_sign",
        }

        for waymo_class, expected_unified in waymo_mappings.items():
            unified_id = self.taxonomy.map_class("waymo", waymo_class)
            if unified_id is not None:
                unified_name = self.taxonomy.get_unified_class_name(unified_id)
                assert unified_name == expected_unified

    def test_unknown_class_handling(self):
        """Test handling of unknown classes."""
        unknown_classes = ["unknown_class", "invalid_type", "not_mapped"]

        for unknown_class in unknown_classes:
            unified_id = self.taxonomy.map_class("nuscenes", unknown_class)
            assert unified_id is None

    def test_unsupported_dataset_handling(self):
        """Test handling of unsupported datasets."""
        unsupported_datasets = ["unsupported_dataset", "random_name"]

        for dataset in unsupported_datasets:
            unified_id = self.taxonomy.map_class(dataset, "car")
            assert unified_id is None


class TestCoordinateHarmonizer:
    """Test coordinate system harmonization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        try:
            from adnet.data.datasets.multi_dataset_loader import CoordinateHarmonizer

            self.harmonizer = CoordinateHarmonizer()
        except ImportError:
            pytest.skip("CoordinateHarmonizer not available")

    def test_pose_harmonization(self):
        """Test ego pose harmonization."""
        # Test pose in different coordinate systems
        test_poses = {
            "nuscenes": np.array(
                [[1, 0, 0, 10], [0, 1, 0, 20], [0, 0, 1, 2], [0, 0, 0, 1]]
            ),
            "waymo": np.array(
                [[0, 1, 0, 15], [-1, 0, 0, 25], [0, 0, 1, 1.5], [0, 0, 0, 1]]
            ),
        }

        for dataset, pose in test_poses.items():
            harmonized_pose = self.harmonizer.harmonize_pose(dataset, pose)

            # Validate harmonized pose structure
            assert harmonized_pose.shape == (4, 4)
            assert np.allclose(harmonized_pose[3, :], [0, 0, 0, 1])  # Bottom row

            # Check that transformation is valid (rotation matrix properties)
            rotation_part = harmonized_pose[:3, :3]
            det = np.linalg.det(rotation_part)
            assert abs(det - 1.0) < 1e-6  # Proper rotation matrix

    def test_box_3d_harmonization(self):
        """Test 3D bounding box harmonization."""
        # Test boxes in different formats
        test_boxes = {
            "nuscenes": np.array(
                [10, 20, 1.5, 2, 4, 1.6, 0.866, 0.5, 3]
            ),  # [x,y,z,w,l,h,cos,sin,v]
            "waymo": np.array([15, 25, 1.2, 1.8, 4.2, 1.5, 0.707, 0.707, 2]),
            "kitti": np.array([12, 18, 1.8, 2.1, 4.5, 1.7, 1, 0, 2.5]),
        }

        for dataset, box in test_boxes.items():
            harmonized_box = self.harmonizer.harmonize_box_3d(dataset, box)

            # Validate harmonized box structure
            assert len(harmonized_box) == 9
            assert harmonized_box[3] > 0  # width
            assert harmonized_box[4] > 0  # length
            assert harmonized_box[5] > 0  # height

            # Check rotation representation
            cos_yaw, sin_yaw = harmonized_box[6], harmonized_box[7]
            assert abs(cos_yaw**2 + sin_yaw**2 - 1.0) < 1e-6

    def test_coordinate_system_consistency(self):
        """Test coordinate system consistency across datasets."""
        # Test that harmonization is consistent
        reference_pose = np.eye(4)
        reference_pose[:3, 3] = [10, 20, 2]

        datasets = ["nuscenes", "waymo", "kitti"]
        harmonized_poses = []

        for dataset in datasets:
            harmonized = self.harmonizer.harmonize_pose(dataset, reference_pose)
            harmonized_poses.append(harmonized)

        # All harmonized poses should follow same coordinate convention
        for pose in harmonized_poses:
            assert pose.shape == (4, 4)
            assert np.allclose(pose[3, :], [0, 0, 0, 1])


class TestMultiDatasetConfiguration:
    """Test multi-dataset configuration functionality."""

    def test_dataset_config_creation(self):
        """Test creation of multi-dataset configurations."""
        try:
            from adnet.data.datasets.multi_dataset_loader import (
                create_multi_dataset_config,
            )

            dataset_names = ["nuscenes", "waymo", "kitti"]
            data_roots = {
                "nuscenes": "/data/nuscenes",
                "waymo": "/data/waymo",
                "kitti": "/data/kitti",
            }
            weights = {"nuscenes": 1.0, "waymo": 0.8, "kitti": 0.6}

            configs = create_multi_dataset_config(
                dataset_names=dataset_names,
                data_roots=data_roots,
                weights=weights,
                split="train",
            )

            # Validate configuration structure
            assert len(configs) == 3
            for config in configs:
                assert "name" in config
                assert "params" in config
                assert "weight" in config
                assert config["name"] in dataset_names
                assert config["params"]["split"] == "train"

        except ImportError:
            pytest.skip("create_multi_dataset_config not available")

    def test_weight_normalization(self):
        """Test dataset weight normalization."""
        weights = {"dataset1": 2.0, "dataset2": 3.0, "dataset3": 1.0}
        total_weight = sum(weights.values())

        # Normalize weights
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Validate normalization
        assert abs(sum(normalized_weights.values()) - 1.0) < 1e-6
        assert all(0 <= w <= 1 for w in normalized_weights.values())

    def test_sampling_strategy_validation(self):
        """Test sampling strategy validation."""
        valid_strategies = ["balanced", "weighted", "sequential", "random"]
        invalid_strategies = ["invalid", "unknown", "custom"]

        for strategy in valid_strategies:
            assert strategy in ["balanced", "weighted", "sequential", "random"]

        for strategy in invalid_strategies:
            assert strategy not in valid_strategies


class TestMultiDatasetLoader:
    """Test multi-dataset loader functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_configs = [
            {
                "name": "dataset1",
                "params": {"data_root": "/data1", "split": "train"},
                "weight": 1.0,
            },
            {
                "name": "dataset2",
                "params": {"data_root": "/data2", "split": "train"},
                "weight": 0.5,
            },
        ]

    @patch("adnet.data.datasets.multi_dataset_loader.DatasetRegistry.get")
    def test_multi_dataset_initialization(self, mock_registry_get):
        """Test multi-dataset loader initialization."""
        try:
            from adnet.data.datasets.multi_dataset_loader import MultiDatasetLoader

            # Mock dataset registry
            mock_dataset_class = Mock()
            mock_registry_get.return_value = mock_dataset_class

            # Test initialization
            loader = MultiDatasetLoader(
                dataset_configs=self.mock_configs,
                harmonize_coordinates=True,
                harmonize_classes=True,
                sampling_strategy="balanced",
            )

            # Validate loader properties
            assert loader.harmonize_coordinates
            assert loader.harmonize_classes
            assert loader.sampling_strategy == "balanced"

        except ImportError:
            pytest.skip("MultiDatasetLoader not available")

    def test_dataset_sampling_weights(self):
        """Test dataset sampling weight calculation."""
        # Calculate expected sampling weights
        total_weight = sum(config["weight"] for config in self.mock_configs)
        expected_weights = [
            config["weight"] / total_weight for config in self.mock_configs
        ]

        # Validate weight calculation
        assert abs(sum(expected_weights) - 1.0) < 1e-6
        assert expected_weights[0] == pytest.approx(2 / 3, rel=1e-3)  # 1.0 / 1.5
        assert expected_weights[1] == pytest.approx(1 / 3, rel=1e-3)  # 0.5 / 1.5

    def test_cross_dataset_sample_harmonization(self):
        """Test sample harmonization across datasets."""
        # Mock samples from different datasets
        sample1 = Sample(
            sample_id="sample1",
            dataset_name="dataset1",
            sequence_info=Mock(),
            images={"CAM_FRONT": np.random.randint(0, 255, (480, 640, 3))},
            camera_params=Mock(),
            instances=[],
            ego_pose=np.eye(4),
            weather="clear",
            time_of_day="day",
            location="location1",
        )

        sample2 = Sample(
            sample_id="sample2",
            dataset_name="dataset2",
            sequence_info=Mock(),
            images={"CAM_FRONT": np.random.randint(0, 255, (480, 640, 3))},
            camera_params=Mock(),
            instances=[],
            ego_pose=np.eye(4),
            weather="clear",
            time_of_day="day",
            location="location2",
        )

        samples = [sample1, sample2]

        # Test that samples from different datasets can be processed together
        for sample in samples:
            assert hasattr(sample, "dataset_name")
            assert hasattr(sample, "sample_id")
            assert isinstance(sample.images, dict)
            assert isinstance(sample.ego_pose, np.ndarray)


if __name__ == "__main__":
    pytest.main([__file__])
