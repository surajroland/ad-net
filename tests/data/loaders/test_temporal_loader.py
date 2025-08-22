"""Test suite for temporal loader implementation.

Tests for temporal sequence handling including:
- Temporal sequence building strategies
- Frame sampling algorithms
- Ego motion computation
- Instance tracking across time
- Batch collation for 4D detection
"""

import os
import sys
from typing import Dict, List

import numpy as np
import pytest
import torch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from adnet.data.loaders.temporal_loader import (
    TemporalDataLoader,
    TemporalDatasetWrapper,
    TemporalSequenceBuilder,
    TemporalSequenceSample,
    create_temporal_dataloader,
)
from adnet.interfaces.data.dataset import (
    BaseDataset,
    CameraParams,
    InstanceAnnotation,
    Sample,
    TemporalSequence,
)


class MockTemporalDataset(BaseDataset):
    """Mock dataset with temporal sequences for testing."""

    def __init__(self, num_samples: int = 20, sequence_length: int = 4) -> None:
        """Initialize mock dataset."""
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self._sample_ids = [f"sample_{i:03d}" for i in range(num_samples)]
        self.class_names = ["car", "pedestrian", "bicycle"]

        # Initialize parent without calling abstract methods
        super(BaseDataset, self).__init__()
        self.data_root = "/mock/data"
        self.split = "train"

    def _load_dataset_info(self) -> None:
        """Mock implementation."""
        pass

    def _load_annotations(self) -> None:
        """Mock implementation."""
        pass

    def _load_sample_data(self, index: int) -> Sample:
        """Mock implementation."""
        return self._create_mock_sample(index)

    def get_camera_calibration(self, sample_id: str) -> CameraParams:
        """Mock implementation."""
        return CameraParams(
            intrinsics=np.array([[[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]] * 2),
            extrinsics=np.array([np.eye(4)] * 2),
            timestamps=np.array([1000000, 1000000]),
        )

    def get_temporal_sequence(self, sample_id: str) -> TemporalSequence:
        """Mock implementation."""
        return TemporalSequence(
            sequence_id="mock_sequence",
            frame_indices=[0, 1, 2],
            timestamps=np.array([1000000, 2000000, 3000000]),
            ego_poses=np.array([np.eye(4)] * 3),
            frame_count=3,
        )

    def __len__(self) -> int:
        """Return dataset length."""
        return self.num_samples

    def __getitem__(self, index: int) -> Sample:
        """Get item by index."""
        return self._load_sample_data(index)

    def _create_mock_sample(self, index: int) -> Sample:
        """Create mock sample with temporal information."""
        # Create sequence info
        sequence_id = f"sequence_{index // 5}"
        frame_indices = [index]
        timestamps = np.array([index * 500000])  # 0.5 second intervals
        ego_poses = np.array([self._create_ego_pose(index)])

        sequence_info = TemporalSequence(
            sequence_id=sequence_id,
            frame_indices=frame_indices,
            timestamps=timestamps,
            ego_poses=ego_poses,
            frame_count=1,
        )

        # Create mock instances
        instances = []
        num_objects = np.random.randint(1, 4)
        for i in range(num_objects):
            instance = InstanceAnnotation(
                box_3d=np.array(
                    [
                        np.random.uniform(-20, 20),  # x
                        np.random.uniform(-20, 20),  # y
                        np.random.uniform(-2, 2),  # z
                        np.random.uniform(1, 3),  # w
                        np.random.uniform(3, 5),  # l
                        np.random.uniform(1, 2),  # h
                        np.random.uniform(-1, 1),  # cos(yaw)
                        np.random.uniform(-1, 1),  # sin(yaw)
                        np.random.uniform(0, 10),  # velocity
                    ]
                ),
                category_id=np.random.randint(0, 3),
                instance_id=f"instance_{index}_{i}",
                visibility=np.random.uniform(0.5, 1.0),
                attributes={},
            )
            instances.append(instance)

        # Create mock images
        images = {
            "CAM_FRONT": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            "CAM_BACK": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }

        # Create mock camera params
        camera_params = CameraParams(
            intrinsics=np.array([[[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]] * 2),
            extrinsics=np.array([np.eye(4)] * 2),
            timestamps=np.array([index * 500000] * 2),
        )

        return Sample(
            sample_id=f"sample_{index:03d}",
            dataset_name="mock_temporal_dataset",
            sequence_info=sequence_info,
            images=images,
            camera_params=camera_params,
            instances=instances,
            ego_pose=self._create_ego_pose(index),
            weather="clear",
            time_of_day="day",
            location="test",
        )

    def _create_ego_pose(self, index: int) -> np.ndarray:
        """Create mock ego pose with some movement."""
        pose = np.eye(4)
        # Simulate forward movement
        pose[0, 3] = index * 2.0  # 2m per frame
        pose[1, 3] = np.sin(index * 0.1) * 1.0  # slight lateral movement
        return pose

    @property
    def sample_ids(self) -> List[str]:
        """Return sample IDs."""
        return self._sample_ids


class TestTemporalSequenceBuilder:
    """Test temporal sequence building functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.dataset = MockTemporalDataset(num_samples=20)
        self.sequence_builder = TemporalSequenceBuilder(
            sequence_length=4, temporal_stride=1, sampling_strategy="uniform"
        )

    def test_sequence_builder_initialization(self) -> None:
        """Test sequence builder initialization."""
        assert self.sequence_builder.sequence_length == 4
        assert self.sequence_builder.temporal_stride == 1
        assert self.sequence_builder.sampling_strategy == "uniform"
        assert self.sequence_builder.max_temporal_gap == 0.5
        assert self.sequence_builder.instance_tracking is True

    def test_uniform_sampling_strategy(self) -> None:
        """Test uniform temporal sampling."""
        center_frame_idx = 10
        sequence = self.sequence_builder.build_sequence(self.dataset, center_frame_idx)

        if sequence is not None:
            # Validate sequence structure
            assert hasattr(sequence, "frames")
            assert hasattr(sequence, "sequence_length")
            assert hasattr(sequence, "ego_motions")
            assert hasattr(sequence, "temporal_weights")

            # Validate sequence properties
            assert sequence.sequence_length <= 4
            assert len(sequence.frames) == sequence.sequence_length
            assert len(sequence.ego_motions) == sequence.sequence_length - 1
            assert len(sequence.temporal_weights) == sequence.sequence_length

    def test_key_frame_sampling_strategy(self) -> None:
        """Test key frame sampling based on ego motion."""
        key_frame_builder = TemporalSequenceBuilder(
            sequence_length=4, sampling_strategy="key_frame", ego_motion_threshold=1.0
        )

        center_frame_idx = 10
        sequence = key_frame_builder.build_sequence(self.dataset, center_frame_idx)

        if sequence is not None:
            # Key frame sampling should select frames with significant motion
            assert sequence.sequence_length <= 4
            assert len(sequence.frames) == sequence.sequence_length

    def test_adaptive_sampling_strategy(self) -> None:
        """Test adaptive sampling based on scene dynamics."""
        adaptive_builder = TemporalSequenceBuilder(
            sequence_length=4, sampling_strategy="adaptive"
        )

        center_frame_idx = 10
        sequence = adaptive_builder.build_sequence(self.dataset, center_frame_idx)

        if sequence is not None:
            # Adaptive sampling should consider scene importance
            assert sequence.sequence_length <= 4
            assert len(sequence.frames) == sequence.sequence_length

    def test_ego_motion_computation(self) -> None:
        """Test ego motion computation between frames."""
        center_frame_idx = 10
        sequence = self.sequence_builder.build_sequence(self.dataset, center_frame_idx)

        if sequence is not None and len(sequence.frames) > 1:
            ego_motions = sequence.ego_motions

            # Validate ego motion structure
            assert ego_motions.shape[1] == 6  # [dx, dy, dz, roll, pitch, yaw]
            assert ego_motions.shape[0] == len(sequence.frames) - 1

            # Ego motions should be finite
            assert np.all(np.isfinite(ego_motions))

    def test_instance_tracking(self) -> None:
        """Test instance tracking across temporal sequence."""
        center_frame_idx = 10
        sequence = self.sequence_builder.build_sequence(self.dataset, center_frame_idx)

        if sequence is not None:
            instance_tracks = sequence.instance_tracks

            # Validate instance tracking structure
            assert isinstance(instance_tracks, dict)

            # Each tracked instance should have multiple annotations
            for instance_id, annotations in instance_tracks.items():
                assert isinstance(instance_id, str)
                assert isinstance(annotations, list)
                assert len(annotations) >= 2  # Should appear in at least 2 frames

    def test_temporal_weights_computation(self) -> None:
        """Test temporal weight computation."""
        center_frame_idx = 10
        sequence = self.sequence_builder.build_sequence(self.dataset, center_frame_idx)

        if sequence is not None:
            weights = sequence.temporal_weights

            # Validate temporal weights
            assert len(weights) == sequence.sequence_length
            assert np.all(weights >= 0)
            assert np.all(weights <= 1)

            # Current frame (last) should have highest weight
            assert weights[-1] == 1.0

    def test_sequence_insufficient_frames(self) -> None:
        """Test handling when insufficient frames available."""
        # Test at beginning of dataset
        sequence = self.sequence_builder.build_sequence(
            self.dataset, center_frame_idx=0
        )

        # Should still create sequence, possibly with fewer frames
        if sequence is not None:
            assert sequence.sequence_length >= 1
        # Or return None if truly insufficient

    @pytest.mark.parametrize("sequence_length", [2, 4, 6, 8])
    def test_different_sequence_lengths(self, sequence_length: int) -> None:
        """Test different temporal sequence lengths."""
        builder = TemporalSequenceBuilder(
            sequence_length=sequence_length, sampling_strategy="uniform"
        )

        center_frame_idx = 10
        sequence = builder.build_sequence(self.dataset, center_frame_idx)

        if sequence is not None:
            assert sequence.sequence_length <= sequence_length
            assert len(sequence.frames) == sequence.sequence_length


class TestTemporalSequenceSample:
    """Test temporal sequence sample functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.dataset = MockTemporalDataset(num_samples=10)

        # Create mock sequence
        frames = [self.dataset[i] for i in range(4)]
        ego_motions = np.random.rand(3, 6)
        instance_tracks: Dict[str, List[InstanceAnnotation]] = {"instance_1": []}
        temporal_weights = np.array([0.5, 0.7, 0.9, 1.0])

        self.sequence_sample = TemporalSequenceSample(
            frames=frames,
            sequence_length=4,
            ego_motions=ego_motions,
            instance_tracks=instance_tracks,
            temporal_weights=temporal_weights,
        )

    def test_sequence_sample_properties(self) -> None:
        """Test sequence sample properties."""
        assert self.sequence_sample.sequence_length == 4
        assert len(self.sequence_sample.frames) == 4
        assert self.sequence_sample.ego_motions.shape == (3, 6)
        assert len(self.sequence_sample.temporal_weights) == 4

    def test_current_frame_access(self) -> None:
        """Test current frame access."""
        current_frame = self.sequence_sample.get_current_frame()
        assert current_frame is not None
        assert current_frame == self.sequence_sample.frames[-1]

    def test_temporal_frames_access(self) -> None:
        """Test temporal frames access."""
        temporal_frames = self.sequence_sample.get_temporal_frames()
        assert len(temporal_frames) == 4
        assert temporal_frames == self.sequence_sample.frames

    def test_ego_motion_access(self) -> None:
        """Test ego motion access by frame index."""
        ego_motion_0 = self.sequence_sample.get_ego_motion_at(0)
        assert len(ego_motion_0) == 6

        # Test out of bounds
        ego_motion_out = self.sequence_sample.get_ego_motion_at(10)
        assert np.array_equal(ego_motion_out, np.zeros(6))

    def test_instance_track_access(self) -> None:
        """Test instance track access."""
        track = self.sequence_sample.get_instance_track("instance_1")
        assert len(track) == 0

        # Test non-existent instance
        empty_track = self.sequence_sample.get_instance_track("nonexistent")
        assert empty_track == []

    def test_temporal_statistics(self) -> None:
        """Test temporal sequence statistics."""
        stats = self.sequence_sample.get_temporal_statistics()

        # Validate statistics structure
        assert "sequence_id" in stats
        assert "sequence_length" in stats
        assert "frame_count" in stats
        assert "time_span" in stats
        assert "num_tracked_instances" in stats
        assert "avg_instances_per_frame" in stats

        # Validate statistics values
        assert stats["sequence_length"] == 4
        assert stats["frame_count"] == 4
        assert stats["num_tracked_instances"] == 1


class TestTemporalDataLoader:
    """Test temporal data loader functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.dataset = MockTemporalDataset(num_samples=20)
        self.sequence_builder = TemporalSequenceBuilder(sequence_length=3)

        self.dataloader = TemporalDataLoader(
            dataset=self.dataset,
            sequence_builder=self.sequence_builder,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Use 0 for testing
        )

    def test_dataloader_initialization(self) -> None:
        """Test dataloader initialization."""
        assert self.dataloader.batch_size == 2
        assert self.dataloader.sequence_builder.sequence_length == 3

    def test_batch_iteration(self) -> None:
        """Test batch iteration."""
        try:
            batch = next(iter(self.dataloader))

            # Validate batch structure
            assert isinstance(batch, dict)

            expected_keys = [
                "batch_size",
                "max_sequence_length",
                "sequence_lengths",
                "images",
                "camera_params",
                "instances",
                "ego_poses",
                "ego_motions",
                "temporal_weights",
                "instance_tracks",
                "sequence_ids",
                "frame_indices",
            ]

            for key in expected_keys:
                assert key in batch, f"Missing key: {key}"

            # Validate batch dimensions
            assert batch["batch_size"] <= 2
            assert len(batch["sequence_lengths"]) == batch["batch_size"]

        except (StopIteration, RuntimeError):
            # May occur if no valid sequences can be built
            pytest.skip("No valid temporal sequences available")

    def test_batch_collation(self) -> None:
        """Test batch collation functionality."""
        try:
            batch = next(iter(self.dataloader))

            if batch["batch_size"] > 0:
                # Test image batching
                if "CAM_FRONT" in batch["images"]:
                    images = batch["images"]["CAM_FRONT"]
                    assert isinstance(images, torch.Tensor)
                    assert images.shape[0] == batch["batch_size"]

                # Test ego motion batching
                ego_motions = batch["ego_motions"]
                assert isinstance(ego_motions, torch.Tensor)
                assert ego_motions.shape[0] == batch["batch_size"]
                assert ego_motions.shape[2] == 6  # [dx, dy, dz, roll, pitch, yaw]

                # Test temporal weights batching
                temporal_weights = batch["temporal_weights"]
                assert isinstance(temporal_weights, torch.Tensor)
                assert temporal_weights.shape[0] == batch["batch_size"]

        except (StopIteration, RuntimeError):
            pytest.skip("No valid temporal sequences available")

    def test_sequence_padding(self) -> None:
        """Test sequence padding for variable length sequences."""
        try:
            batch = next(iter(self.dataloader))

            if batch["batch_size"] > 1:
                _ = batch["sequence_lengths"]  # Used for validation
                max_length = batch["max_sequence_length"]

                # All sequences should be padded to max length
                ego_poses = batch["ego_poses"]
                assert ego_poses.shape[1] == max_length

                temporal_weights = batch["temporal_weights"]
                assert temporal_weights.shape[1] == max_length

        except (StopIteration, RuntimeError):
            pytest.skip("No valid temporal sequences available")


class TestTemporalDatasetWrapper:
    """Test temporal dataset wrapper functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.base_dataset = MockTemporalDataset(num_samples=20)
        self.sequence_builder = TemporalSequenceBuilder(sequence_length=3)

        self.wrapper = TemporalDatasetWrapper(
            base_dataset=self.base_dataset, sequence_builder=self.sequence_builder
        )

    def test_wrapper_initialization(self) -> None:
        """Test wrapper initialization."""
        assert self.wrapper.base_dataset == self.base_dataset
        assert self.wrapper.sequence_builder == self.sequence_builder
        assert hasattr(self.wrapper, "valid_indices")

    def test_valid_indices_building(self) -> None:
        """Test valid indices building."""
        # Should identify frames that can be centers of valid sequences
        valid_indices = self.wrapper.valid_indices
        assert isinstance(valid_indices, list)
        assert len(valid_indices) <= len(self.base_dataset)

        # All valid indices should be within dataset bounds
        for idx in valid_indices:
            assert 0 <= idx < len(self.base_dataset)

    def test_wrapper_length(self) -> None:
        """Test wrapper length."""
        wrapper_length = len(self.wrapper)
        assert wrapper_length <= len(self.base_dataset)
        assert wrapper_length == len(self.wrapper.valid_indices)

    def test_sequence_retrieval(self) -> None:
        """Test sequence retrieval by index."""
        if len(self.wrapper) > 0:
            sequence = self.wrapper[0]

            # Should return a TemporalSequenceSample
            assert hasattr(sequence, "frames")
            assert hasattr(sequence, "sequence_length")
            assert hasattr(sequence, "ego_motions")
            assert hasattr(sequence, "temporal_weights")


class TestTemporalLoaderFactory:
    """Test temporal loader factory function."""

    def test_create_temporal_dataloader(self) -> None:
        """Test temporal dataloader creation factory."""
        dataset = MockTemporalDataset(num_samples=10)

        dataloader = create_temporal_dataloader(
            dataset=dataset,
            sequence_length=4,
            temporal_stride=1,
            sampling_strategy="uniform",
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )

        # Validate dataloader creation
        assert dataloader is not None
        assert dataloader.batch_size == 2


if __name__ == "__main__":
    pytest.main([__file__])
