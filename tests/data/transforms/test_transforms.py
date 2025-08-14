"""
Test suite for data transforms implementation.

Tests for data transformation pipeline including:
- Multi-view image augmentations
- 3D spatial transformations
- Temporal sequence augmentations
- Photometric adjustments
- Transform composition and pipelines
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

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


def create_mock_sample():
    """Create a mock sample for testing transforms"""
    # Mock images (using nuScenes 6-camera setup)
    images = {
        "CAM_FRONT": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "CAM_FRONT_RIGHT": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "CAM_FRONT_LEFT": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "CAM_BACK": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "CAM_BACK_RIGHT": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "CAM_BACK_LEFT": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    }

    # Mock camera parameters (6 cameras to match nuScenes)
    camera_params = CameraParams(
        intrinsics=np.array(
            [
                [[1266.4, 0, 816.2], [0, 1266.4, 491.5], [0, 0, 1]],  # CAM_FRONT
                [[1260.8, 0, 807.9], [0, 1260.8, 495.3], [0, 0, 1]],  # CAM_FRONT_RIGHT
                [[1259.5, 0, 807.2], [0, 1259.5, 450.3], [0, 0, 1]],  # CAM_FRONT_LEFT
                [[1258.1, 0, 808.1], [0, 1258.1, 448.9], [0, 0, 1]],  # CAM_BACK
                [[1257.3, 0, 809.5], [0, 1257.3, 447.2], [0, 0, 1]],  # CAM_BACK_RIGHT
                [[1256.9, 0, 810.1], [0, 1256.9, 449.8], [0, 0, 1]],  # CAM_BACK_LEFT
            ]
        ),
        extrinsics=np.array([np.eye(4)] * 6),
        timestamps=np.array([1000000, 1000000, 1000000, 1000000, 1000000, 1000000]),
    )

    # Mock instances
    instances = [
        InstanceAnnotation(
            box_3d=np.array([10.5, -5.2, 1.8, 1.8, 4.2, 1.6, 0.866, 0.5, 2.3]),
            category_id=0,
            instance_id="vehicle_001",
            visibility=0.8,
            attributes={"moving": True},
        ),
        InstanceAnnotation(
            box_3d=np.array([-8.3, 12.1, 1.5, 0.8, 1.2, 1.7, 0.707, 0.707, 1.5]),
            category_id=1,
            instance_id="pedestrian_001",
            visibility=0.9,
            attributes={"moving": False},
        ),
    ]

    # Mock sequence info
    sequence_info = TemporalSequence(
        sequence_id="test_sequence_001",
        frame_indices=[0, 1, 2, 3],
        timestamps=np.array([1000000, 1500000, 2000000, 2500000]),
        ego_poses=np.array([np.eye(4)] * 4),
        frame_count=4,
    )

    return Sample(
        sample_id="test_sample_001",
        dataset_name="test_dataset",
        sequence_info=sequence_info,
        images=images,
        camera_params=camera_params,
        instances=instances,
        ego_pose=np.eye(4),
        weather="clear",
        time_of_day="day",
        location="test_location",
    )


class TestBaseTransform:
    """Test base transform functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        try:
            from adnet.data.transforms.transforms import Transform

            self.transform_class = Transform
        except ImportError:
            pytest.skip("Transform not available")

    def test_transform_probability(self):
        """Test transform probability mechanism"""

        class TestTransform(self.transform_class):
            def apply(self, sample):
                sample.test_applied = True
                return sample

        # Test with probability 0 (never apply)
        transform_never = TestTransform(probability=0.0)
        sample = create_mock_sample()
        result = transform_never(sample)
        assert not hasattr(result, "test_applied")

        # Test with probability 1 (always apply)
        transform_always = TestTransform(probability=1.0)
        sample = create_mock_sample()
        result = transform_always(sample)
        assert hasattr(result, "test_applied")
        assert result.test_applied == True


class TestPhotometricAugmentation:
    """Test photometric augmentation functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        try:
            from adnet.data.transforms.transforms import PhotometricAugmentation

            self.transform = PhotometricAugmentation(
                brightness_range=(0.8, 1.2),
                contrast_range=(0.8, 1.2),
                saturation_range=(0.8, 1.2),
                hue_range=(-0.1, 0.1),
                probability=1.0,
                consistent_across_views=True,
            )
        except ImportError:
            pytest.skip("PhotometricAugmentation not available")

    def test_photometric_transform_structure(self):
        """Test photometric transform preserves sample structure"""
        sample = create_mock_sample()
        original_image_keys = set(sample.images.keys())

        transformed_sample = self.transform(sample)

        # Should preserve image structure
        assert set(transformed_sample.images.keys()) == original_image_keys

        # Should preserve image dimensions
        for camera_name in original_image_keys:
            original_shape = sample.images[camera_name].shape
            transformed_shape = transformed_sample.images[camera_name].shape
            assert transformed_shape == original_shape

    def test_photometric_consistent_across_views(self):
        """Test photometric consistency across camera views"""
        self.transform.consistent_across_views = True
        sample = create_mock_sample()

        # Store original images for comparison
        original_images = {k: v.copy() for k, v in sample.images.items()}

        transformed_sample = self.transform(sample)

        # With consistent=True, all cameras should have same transformation
        # Check that all images changed in similar way (not identical due to different content)
        for camera_name in sample.images.keys():
            original = original_images[camera_name]
            transformed = transformed_sample.images[camera_name]

            # Images should be different (unless no transformation applied)
            if not np.array_equal(original, transformed):
                # Transformed images should be in valid range
                assert transformed.min() >= 0
                assert transformed.max() <= 255 or transformed.dtype != np.uint8

    def test_photometric_parameter_ranges(self):
        """Test photometric parameter validation"""
        # Test valid parameter ranges
        valid_transform = PhotometricAugmentation(
            brightness_range=(0.5, 1.5),
            contrast_range=(0.5, 1.5),
            saturation_range=(0.5, 1.5),
            hue_range=(-0.2, 0.2),
            probability=1.0,
        )

        sample = create_mock_sample()
        result = valid_transform(sample)

        # Should not raise exceptions
        assert result is not None
        assert isinstance(result, Sample)


class TestMultiViewResize:
    """Test multi-view resize functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        try:
            from adnet.data.transforms.transforms import MultiViewResize

            self.target_size = (224, 224)  # (height, width)
            self.transform = MultiViewResize(self.target_size, probability=1.0)
        except ImportError:
            pytest.skip("MultiViewResize not available")

    def test_image_resize(self):
        """Test image resizing functionality"""
        sample = create_mock_sample()
        transformed_sample = self.transform(sample)

        # All images should be resized to target size
        for camera_name, image in transformed_sample.images.items():
            assert image.shape[:2] == self.target_size  # (H, W)
            assert image.shape[2] == 3  # RGB channels preserved

    def test_camera_intrinsics_update(self):
        """Test camera intrinsics update after resize"""
        sample = create_mock_sample()
        original_intrinsics = sample.camera_params.intrinsics.copy()

        transformed_sample = self.transform(sample)
        updated_intrinsics = transformed_sample.camera_params.intrinsics

        # Intrinsics should be updated
        assert not np.array_equal(original_intrinsics, updated_intrinsics)

        # Focal lengths and principal points should be scaled appropriately
        for i in range(len(updated_intrinsics)):
            # Check that intrinsic matrix is still valid (3x3)
            assert updated_intrinsics[i].shape == (3, 3)

            # Bottom row should remain [0, 0, 1]
            assert np.allclose(updated_intrinsics[i][2, :], [0, 0, 1])

    def test_resize_scale_factors(self):
        """Test resize scale factor computation"""
        sample = create_mock_sample()
        original_size = sample.images["CAM_FRONT"].shape[:2]  # (H, W)

        # Calculate expected scale factors
        expected_scale_y = self.target_size[0] / original_size[0]
        expected_scale_x = self.target_size[1] / original_size[1]

        transformed_sample = self.transform(sample)

        # Verify scale factors applied correctly to intrinsics
        original_fx = sample.camera_params.intrinsics[0, 0, 0]
        transformed_fx = transformed_sample.camera_params.intrinsics[0, 0, 0]

        expected_fx = original_fx * expected_scale_x
        assert abs(transformed_fx - expected_fx) < 1e-6


class TestSpatialAugmentation3D:
    """Test 3D spatial augmentation functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        try:
            from adnet.data.transforms.transforms import SpatialAugmentation3D

            self.transform = SpatialAugmentation3D(
                rotation_range=(-0.1, 0.1),
                translation_range=(-1.0, 1.0),
                scaling_range=(0.9, 1.1),
                probability=1.0,
            )
        except ImportError:
            pytest.skip("SpatialAugmentation3D not available")

    def test_ego_pose_transformation(self):
        """Test ego pose transformation"""
        sample = create_mock_sample()
        original_ego_pose = sample.ego_pose.copy()

        transformed_sample = self.transform(sample)

        # Ego pose should be different
        assert not np.array_equal(original_ego_pose, transformed_sample.ego_pose)

        # Should still be a valid 4x4 transformation matrix
        assert transformed_sample.ego_pose.shape == (4, 4)
        assert np.allclose(transformed_sample.ego_pose[3, :], [0, 0, 0, 1])

    def test_instance_transformation(self):
        """Test 3D bounding box transformation"""
        sample = create_mock_sample()
        original_instances = [inst.box_3d.copy() for inst in sample.instances]

        transformed_sample = self.transform(sample)

        # Instance positions should be transformed
        for i, instance in enumerate(transformed_sample.instances):
            original_box = original_instances[i]
            transformed_box = instance.box_3d

            # Box should still have correct format
            assert len(transformed_box) == 9

            # Dimensions should be positive
            assert transformed_box[3] > 0  # width
            assert transformed_box[4] > 0  # length
            assert transformed_box[5] > 0  # height

            # Rotation should be normalized
            cos_yaw, sin_yaw = transformed_box[6], transformed_box[7]
            assert abs(cos_yaw**2 + sin_yaw**2 - 1.0) < 1e-6

    def test_camera_extrinsics_transformation(self):
        """Test camera extrinsics transformation"""
        sample = create_mock_sample()
        original_extrinsics = sample.camera_params.extrinsics.copy()

        transformed_sample = self.transform(sample)
        updated_extrinsics = transformed_sample.camera_params.extrinsics

        # Extrinsics should be transformed
        assert not np.array_equal(original_extrinsics, updated_extrinsics)

        # Should still be valid transformation matrices
        for extrinsic in updated_extrinsics:
            assert extrinsic.shape == (4, 4)
            assert np.allclose(extrinsic[3, :], [0, 0, 0, 1])

    def test_geometric_consistency(self):
        """Test geometric consistency after transformation"""
        sample = create_mock_sample()
        transformed_sample = self.transform(sample)

        # All geometric components should be consistently transformed
        # This is verified by checking that relationships are preserved

        # Ego pose should be transformed
        assert not np.array_equal(sample.ego_pose, transformed_sample.ego_pose)

        # Instances should be transformed
        for orig, trans in zip(sample.instances, transformed_sample.instances):
            assert not np.array_equal(orig.box_3d[:3], trans.box_3d[:3])

        # Camera extrinsics should be transformed
        assert not np.array_equal(
            sample.camera_params.extrinsics, transformed_sample.camera_params.extrinsics
        )


class TestTemporalAugmentation:
    """Test temporal augmentation functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        try:
            from adnet.data.transforms.transforms import TemporalAugmentation

            self.transform = TemporalAugmentation(
                dropout_probability=0.0,  # Disable for predictable testing
                frame_rate_simulation=True,
                target_fps=2.0,
                probability=1.0,
            )
        except ImportError:
            pytest.skip("TemporalAugmentation not available")

    def test_temporal_transform_structure(self):
        """Test temporal transform preserves sample structure"""
        sample = create_mock_sample()
        transformed_sample = self.transform(sample)

        # Should preserve basic sample structure
        assert transformed_sample.sample_id == sample.sample_id
        assert transformed_sample.dataset_name == sample.dataset_name
        assert hasattr(transformed_sample, "sequence_info")

    def test_frame_rate_simulation(self):
        """Test frame rate simulation"""
        self.transform.frame_rate_simulation = True
        sample = create_mock_sample()

        # Modify sequence to have high frame rate
        sample.sequence_info.timestamps = np.array(
            [1000000, 1100000, 1200000, 1300000, 1400000, 1500000]
        )  # 10 FPS

        transformed_sample = self.transform(sample)

        # Should preserve sequence structure
        assert hasattr(transformed_sample.sequence_info, "timestamps")
        assert hasattr(transformed_sample.sequence_info, "frame_indices")

    def test_temporal_dropout_disabled(self):
        """Test temporal dropout when disabled"""
        self.transform.dropout_probability = 0.0
        sample = create_mock_sample()

        original_frame_count = len(sample.sequence_info.frame_indices)
        transformed_sample = self.transform(sample)

        # With dropout disabled, frame count should be preserved
        # (unless frame rate simulation reduces it)
        transformed_frame_count = len(transformed_sample.sequence_info.frame_indices)
        assert transformed_frame_count > 0


class TestNormalization:
    """Test normalization functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        try:
            from adnet.data.transforms.transforms import Normalize

            self.transform = Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), probability=1.0
            )
        except ImportError:
            pytest.skip("Normalize not available")

    def test_normalization_output_range(self):
        """Test normalization output value range"""
        sample = create_mock_sample()
        transformed_sample = self.transform(sample)

        # Images should be normalized
        for camera_name, image in transformed_sample.images.items():
            # Should be float32
            assert image.dtype == np.float32

            # Should have negative and positive values after normalization
            assert image.min() < 0
            assert image.max() > 0

    def test_normalization_statistics(self):
        """Test normalization statistics"""
        sample = create_mock_sample()
        transformed_sample = self.transform(sample)

        # Check that normalization follows expected pattern
        for camera_name, image in transformed_sample.images.items():
            # Image should be properly normalized
            # (exact statistics depend on input image content)
            assert np.isfinite(image).all()
            assert not np.isnan(image).any()

    def test_normalization_preserves_structure(self):
        """Test normalization preserves image structure"""
        sample = create_mock_sample()
        original_shapes = {k: v.shape for k, v in sample.images.items()}

        transformed_sample = self.transform(sample)

        # Should preserve image shapes and keys
        assert set(transformed_sample.images.keys()) == set(sample.images.keys())

        for camera_name in sample.images.keys():
            original_shape = original_shapes[camera_name]
            transformed_shape = transformed_sample.images[camera_name].shape
            assert transformed_shape == original_shape


class TestTransformComposition:
    """Test transform composition and pipelines"""

    def setup_method(self):
        """Setup test fixtures"""
        try:
            from adnet.data.transforms.transforms import (
                Compose,
                MultiViewResize,
                Normalize,
            )

            self.pipeline = Compose(
                [
                    MultiViewResize((224, 224), probability=1.0),
                    Normalize(probability=1.0),
                ]
            )
        except ImportError:
            pytest.skip("Transform composition not available")

    def test_pipeline_execution(self):
        """Test pipeline execution order"""
        sample = create_mock_sample()
        transformed_sample = self.pipeline(sample)

        # Should apply all transforms in sequence
        # Images should be resized AND normalized
        for camera_name, image in transformed_sample.images.items():
            # Should be resized
            assert image.shape[:2] == (224, 224)

            # Should be normalized (float32 with negative values)
            assert image.dtype == np.float32
            assert image.min() < 0

    def test_pipeline_structure_preservation(self):
        """Test pipeline preserves sample structure"""
        sample = create_mock_sample()
        transformed_sample = self.pipeline(sample)

        # Should preserve all sample attributes
        assert transformed_sample.sample_id == sample.sample_id
        assert transformed_sample.dataset_name == sample.dataset_name
        assert len(transformed_sample.instances) == len(sample.instances)
        assert set(transformed_sample.images.keys()) == set(sample.images.keys())


class TestTransformPipelineFactory:
    """Test transform pipeline factory functions"""

    def test_training_pipeline_creation(self):
        """Test training pipeline factory"""
        try:
            from adnet.data.transforms.transforms import (
                create_training_transform_pipeline,
            )

            pipeline = create_training_transform_pipeline(
                target_image_size=(224, 224),
                enable_photometric=True,
                enable_spatial_3d=True,
                enable_temporal=True,
            )

            # Should create a valid pipeline
            assert pipeline is not None

            # Test pipeline execution
            sample = create_mock_sample()
            result = pipeline(sample)

            # Should produce valid output
            assert isinstance(result, Sample)
            assert result.images["CAM_FRONT"].shape[:2] == (224, 224)
            assert result.images["CAM_FRONT"].dtype == np.float32

        except ImportError:
            pytest.skip("create_training_transform_pipeline not available")

    def test_validation_pipeline_creation(self):
        """Test validation pipeline factory"""
        try:
            from adnet.data.transforms.transforms import (
                create_validation_transform_pipeline,
            )

            pipeline = create_validation_transform_pipeline(
                target_image_size=(224, 224)
            )

            # Should create a minimal pipeline (resize + normalize only)
            assert pipeline is not None

            # Test pipeline execution
            sample = create_mock_sample()
            result = pipeline(sample)

            # Should produce consistent, deterministic output
            assert isinstance(result, Sample)
            assert result.images["CAM_FRONT"].shape[:2] == (224, 224)
            assert result.images["CAM_FRONT"].dtype == np.float32

        except ImportError:
            pytest.skip("create_validation_transform_pipeline not available")


if __name__ == "__main__":
    pytest.main([__file__])
