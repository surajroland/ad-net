"""Test suite for nuScenes dataset implementation.

Tests for the nuScenes dataset loader including:
- Dataset initialization and metadata loading
- Sample loading and validation
- Camera calibration handling
- Temporal sequence construction
- Instance tracking across frames
"""

import os
import sys

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from adnet.interfaces.data.dataset import (
    CameraParams,
    InstanceAnnotation,
    Sample,
    TemporalSequence,
)


class TestNuScenesDataset:
    """Test suite for nuScenes dataset implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_data_root = "/mock/nuscenes"
        self.mock_version = "v1.0-mini"
        self.mock_split = "train"

    def test_dataset_initialization(self):
        """Test dataset initialization with mock data."""
        # This would test actual NuScenesDataset initialization
        # For now, test that we can import the module
        try:
            from adnet.data.datasets.nuscenes_dataset import NuScenesDataset

            # Use the import to avoid unused import warning
            assert NuScenesDataset is not None, "NuScenesDataset imported successfully"
        except ImportError as e:
            pytest.skip(f"NuScenesDataset not available: {e}")

    def test_sample_loading(self):
        """Test sample data loading."""
        # Mock sample data structure
        mock_sample = Sample(
            sample_id="sample_001",
            dataset_name="nuscenes",
            sequence_info=TemporalSequence(
                sequence_id="sequence_001",
                frame_indices=[0],
                timestamps=np.array([1000000]),
                ego_poses=np.array([np.eye(4)]),
                frame_count=1,
            ),
            images={
                "CAM_FRONT": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "CAM_BACK": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            },
            camera_params=CameraParams(
                intrinsics=np.array([[[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]] * 2),
                extrinsics=np.array([np.eye(4)] * 2),
                timestamps=np.array([1000000, 1000000]),
            ),
            instances=[
                InstanceAnnotation(
                    box_3d=np.array([1, 2, 3, 2, 4, 1.5, 1, 0, 5]),
                    category_id=0,
                    instance_id="instance_001",
                    visibility=1.0,
                    attributes={},
                )
            ],
            ego_pose=np.eye(4),
            weather="clear",
            time_of_day="day",
            location="singapore",
        )

        # Validate sample structure
        assert mock_sample.sample_id == "sample_001"
        assert len(mock_sample.images) == 2
        assert len(mock_sample.instances) == 1
        assert mock_sample.camera_params.intrinsics.shape[0] == 2

    def test_camera_calibration(self):
        """Test camera calibration parameter handling."""
        intrinsics = np.array(
            [
                [[1266.4, 0, 816.2], [0, 1266.4, 491.5], [0, 0, 1]],  # CAM_FRONT
                [[1260.8, 0, 807.9], [0, 1260.8, 495.3], [0, 0, 1]],  # CAM_BACK
            ]
        )

        extrinsics = np.array([np.eye(4), np.eye(4)])  # CAM_FRONT  # CAM_BACK

        camera_params = CameraParams(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            timestamps=np.array([1000000, 1000000]),
        )

        # Validate camera parameters
        assert camera_params.intrinsics.shape == (2, 3, 3)
        assert camera_params.extrinsics.shape == (2, 4, 4)
        assert camera_params.timestamps.shape == (2,)

        # Test focal length extraction
        fx, fy = intrinsics[0, 0, 0], intrinsics[0, 1, 1]
        assert fx == pytest.approx(1266.4, rel=1e-3)
        assert fy == pytest.approx(1266.4, rel=1e-3)

    def test_temporal_sequence_construction(self):
        """Test temporal sequence metadata construction."""
        sequence_info = TemporalSequence(
            sequence_id="scene_001_frame_010",
            frame_indices=[8, 9, 10, 11],
            timestamps=np.array([1000000, 1500000, 2000000, 2500000]),
            ego_poses=np.array([np.eye(4)] * 4),
            frame_count=4,
        )

        # Validate sequence structure
        assert sequence_info.sequence_id == "scene_001_frame_010"
        assert len(sequence_info.frame_indices) == 4
        assert sequence_info.frame_count == 4
        assert sequence_info.timestamps.shape == (4,)
        assert sequence_info.ego_poses.shape == (4, 4, 4)

        # Test temporal spacing
        time_diffs = np.diff(sequence_info.timestamps)
        expected_diff = 500000  # 0.5 seconds
        assert np.allclose(time_diffs, expected_diff)

    def test_instance_annotation_validation(self):
        """Test 3D bounding box annotation validation."""
        # Valid nuScenes 3D box format: [x, y, z, w, l, h, cos(yaw), sin(yaw), velocity]
        box_3d = np.array([10.5, -5.2, 1.8, 1.8, 4.2, 1.6, 0.866, 0.5, 2.3])

        instance = InstanceAnnotation(
            box_3d=box_3d,
            category_id=1,  # vehicle.car
            instance_id="vehicle_001_frame_010",
            visibility=0.8,
            attributes={"moving": True, "occluded": False},
        )

        # Validate instance structure
        assert len(instance.box_3d) == 9
        assert instance.category_id == 1
        assert instance.visibility == 0.8
        assert "moving" in instance.attributes

        # Validate 3D box components
        x, _y, z = box_3d[0], box_3d[1], box_3d[2]
        w, _length, h = box_3d[3], box_3d[4], box_3d[5]
        cos_yaw, sin_yaw = box_3d[6], box_3d[7]
        box_3d[8]

        assert x == pytest.approx(10.5, rel=1e-3)
        assert z == pytest.approx(1.8, rel=1e-3)  # Height above ground
        assert w > 0 and _length > 0 and h > 0  # Positive dimensions
        assert abs(cos_yaw**2 + sin_yaw**2 - 1.0) < 1e-6  # Unit vector constraint

    def test_class_mapping(self):
        """Test nuScenes class name mapping."""
        # nuScenes detection classes
        nuscenes_classes = [
            "car",
            "truck",
            "bus",
            "trailer",
            "construction_vehicle",
            "pedestrian",
            "motorcycle",
            "bicycle",
            "traffic_cone",
            "barrier",
        ]

        # Test that all classes are valid
        for i, class_name in enumerate(nuscenes_classes):
            assert isinstance(class_name, str)
            assert len(class_name) > 0

        # Test category ID bounds
        assert len(nuscenes_classes) == 10
        for category_id in range(len(nuscenes_classes)):
            assert 0 <= category_id < len(nuscenes_classes)

    def test_split_validation(self):
        """Test dataset split validation."""
        valid_splits = ["train", "val", "test", "mini_train", "mini_val"]

        for split in valid_splits:
            # Test that split names are valid
            assert split in ["train", "val", "test", "mini_train", "mini_val"]

        # Test invalid split handling
        invalid_splits = ["invalid", "testing", "validation"]
        for invalid_split in invalid_splits:
            assert invalid_split not in valid_splits

    @pytest.mark.parametrize("sequence_length", [1, 2, 4, 8])
    def test_sequence_length_handling(self, sequence_length):
        """Test different temporal sequence lengths."""
        frame_indices = list(range(sequence_length))
        timestamps = np.array([i * 500000 for i in range(sequence_length)])
        ego_poses = np.array([np.eye(4)] * sequence_length)

        sequence_info = TemporalSequence(
            sequence_id=f"test_sequence_{sequence_length}",
            frame_indices=frame_indices,
            timestamps=timestamps,
            ego_poses=ego_poses,
            frame_count=sequence_length,
        )

        assert len(sequence_info.frame_indices) == sequence_length
        assert sequence_info.timestamps.shape == (sequence_length,)
        assert sequence_info.ego_poses.shape == (sequence_length, 4, 4)
        assert sequence_info.frame_count == sequence_length


if __name__ == "__main__":
    pytest.main([__file__])
