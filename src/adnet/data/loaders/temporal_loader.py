"""
Temporal sequence loader for 4D object detection.

This module provides specialized data loading for temporal sequences,
handling frame sampling, ego motion computation, and instance tracking
across time for Sparse4D's temporal reasoning capabilities.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import random

from ...interfaces.data.dataset import (
    BaseDataset,
    Sample,
    InstanceAnnotation,
    TemporalSequence,
)
from ..transforms.transforms import Transform


class TemporalSequenceSample:
    """
    Container for temporal sequence data used in 4D object detection.

    Contains multiple frames with ego motion, instance tracking,
    and proper temporal alignment for Sparse4D processing.
    """

    def __init__(
        self,
        frames: List[Sample],
        sequence_length: int,
        ego_motions: np.ndarray,
        instance_tracks: Dict[str, List[InstanceAnnotation]],
        temporal_weights: np.ndarray,
    ):
        self.frames = frames
        self.sequence_length = sequence_length
        self.ego_motions = ego_motions  # [T-1, 6] - relative motions between frames
        self.instance_tracks = instance_tracks  # {instance_id: [annotations]}
        self.temporal_weights = temporal_weights  # [T] - importance weights

        # Current frame is always the last one
        self.current_frame = frames[-1] if frames else None

        # Temporal metadata
        self.sequence_id = frames[0].sequence_info.sequence_id if frames else None
        self.frame_indices = [f.sequence_info.frame_indices[-1] for f in frames]
        self.timestamps = np.array([f.sequence_info.timestamps[-1] for f in frames])

    def get_current_frame(self) -> Sample:
        """Get the current (last) frame in sequence"""
        return self.current_frame

    def get_temporal_frames(self) -> List[Sample]:
        """Get all frames in temporal order (oldest to newest)"""
        return self.frames

    def get_ego_motion_at(self, frame_idx: int) -> np.ndarray:
        """Get ego motion from frame_idx to frame_idx+1"""
        if frame_idx >= len(self.ego_motions):
            return np.zeros(6)  # [dx, dy, dz, roll, pitch, yaw]
        return self.ego_motions[frame_idx]

    def get_instance_track(self, instance_id: str) -> List[InstanceAnnotation]:
        """Get tracking history for specific instance"""
        return self.instance_tracks.get(instance_id, [])

    def get_temporal_statistics(self) -> Dict[str, Any]:
        """Get temporal sequence statistics"""
        return {
            "sequence_id": self.sequence_id,
            "sequence_length": self.sequence_length,
            "frame_count": len(self.frames),
            "time_span": (
                self.timestamps[-1] - self.timestamps[0]
                if len(self.timestamps) > 1
                else 0
            ),
            "num_tracked_instances": len(self.instance_tracks),
            "avg_instances_per_frame": np.mean([len(f.instances) for f in self.frames]),
        }


class TemporalSequenceBuilder:
    """
    Builds temporal sequences from individual frames.

    Handles frame sampling strategies, ego motion computation,
    and instance tracking across temporal windows.
    """

    def __init__(
        self,
        sequence_length: int = 4,
        temporal_stride: int = 1,
        sampling_strategy: str = "uniform",  # uniform, key_frame, adaptive
        max_temporal_gap: float = 0.5,  # seconds
        ego_motion_threshold: float = 1.0,  # meters
        instance_tracking: bool = True,
    ):
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride
        self.sampling_strategy = sampling_strategy
        self.max_temporal_gap = max_temporal_gap
        self.ego_motion_threshold = ego_motion_threshold
        self.instance_tracking = instance_tracking

    def build_sequence(
        self, dataset: BaseDataset, center_frame_idx: int
    ) -> Optional[TemporalSequenceSample]:
        """
        Build temporal sequence centered on given frame.

        Args:
            dataset: Dataset containing frames
            center_frame_idx: Index of center frame

        Returns:
            TemporalSequenceSample if successful, None if insufficient frames
        """
        # Get frame indices for sequence
        frame_indices = self._get_sequence_indices(dataset, center_frame_idx)

        if len(frame_indices) < 2:  # Need at least 2 frames for temporal
            return None

        # Load frames
        frames = []
        for idx in frame_indices:
            try:
                frame = dataset[idx]
                frames.append(frame)
            except IndexError:
                continue

        if len(frames) < 2:
            return None

        # Compute ego motions
        ego_motions = self._compute_ego_motions(frames)

        # Track instances if enabled
        instance_tracks = {}
        if self.instance_tracking:
            instance_tracks = self._track_instances(frames)

        # Compute temporal weights
        temporal_weights = self._compute_temporal_weights(frames)

        return TemporalSequenceSample(
            frames=frames,
            sequence_length=len(frames),
            ego_motions=ego_motions,
            instance_tracks=instance_tracks,
            temporal_weights=temporal_weights,
        )

    def _get_sequence_indices(self, dataset: BaseDataset, center_idx: int) -> List[int]:
        """Get frame indices for temporal sequence"""
        if self.sampling_strategy == "uniform":
            return self._uniform_sampling(dataset, center_idx)
        elif self.sampling_strategy == "key_frame":
            return self._key_frame_sampling(dataset, center_idx)
        elif self.sampling_strategy == "adaptive":
            return self._adaptive_sampling(dataset, center_idx)
        else:
            raise ValueError(f"Unknown sampling strategy: {self.sampling_strategy}")

    def _uniform_sampling(self, dataset: BaseDataset, center_idx: int) -> List[int]:
        """Uniform temporal sampling"""
        indices = []

        # Sample backwards from center
        for i in range(self.sequence_length):
            idx = center_idx - (self.sequence_length - 1 - i) * self.temporal_stride
            if idx >= 0:
                indices.append(idx)

        return indices

    def _key_frame_sampling(self, dataset: BaseDataset, center_idx: int) -> List[int]:
        """Key frame sampling based on ego motion"""
        indices = [center_idx]  # Always include center frame

        # Look backwards for frames with significant ego motion
        current_idx = center_idx
        ego_motion_sum = 0.0

        for i in range(self.sequence_length - 1):
            prev_idx = current_idx - self.temporal_stride
            if prev_idx < 0:
                break

            # Check ego motion magnitude
            ego_motion = self._compute_ego_motion_between_frames(
                dataset, prev_idx, current_idx
            )
            motion_magnitude = np.linalg.norm(ego_motion[:3])  # Translation magnitude

            ego_motion_sum += motion_magnitude

            # Include frame if significant motion or force include for sequence length
            if (
                ego_motion_sum >= self.ego_motion_threshold
                or i == self.sequence_length - 2
            ):
                indices.insert(0, prev_idx)  # Insert at beginning for temporal order
                ego_motion_sum = 0.0
                current_idx = prev_idx

        return indices

    def _adaptive_sampling(self, dataset: BaseDataset, center_idx: int) -> List[int]:
        """Adaptive sampling based on scene dynamics"""
        # Get all possible frames within temporal window
        max_lookback = self.sequence_length * self.temporal_stride * 2
        candidate_indices = []

        for i in range(max_lookback):
            idx = center_idx - i
            if idx >= 0:
                candidate_indices.append(idx)
            else:
                break

        if len(candidate_indices) <= self.sequence_length:
            return sorted(candidate_indices)

        # Score frames based on scene dynamics
        frame_scores = []
        for idx in candidate_indices[1:]:  # Skip center frame
            score = self._compute_frame_importance(dataset, idx, center_idx)
            frame_scores.append((score, idx))

        # Select top-k frames plus center
        frame_scores.sort(reverse=True)  # Sort by score descending
        selected_indices = [center_idx]  # Always include center

        for score, idx in frame_scores[: self.sequence_length - 1]:
            selected_indices.append(idx)

        return sorted(selected_indices)  # Return in temporal order

    def _compute_frame_importance(
        self, dataset: BaseDataset, frame_idx: int, center_idx: int
    ) -> float:
        """Compute importance score for frame selection"""
        try:
            frame = dataset[frame_idx]
            center_frame = dataset[center_idx]
        except IndexError:
            return 0.0

        score = 0.0

        # Ego motion contribution
        ego_motion = self._compute_ego_motion_between_frames(
            dataset, frame_idx, center_idx
        )
        motion_score = np.linalg.norm(ego_motion[:3])  # Translation magnitude
        score += motion_score * 0.3

        # Instance overlap contribution
        frame_instances = set(inst.instance_id for inst in frame.instances)
        center_instances = set(inst.instance_id for inst in center_frame.instances)
        overlap_ratio = len(frame_instances & center_instances) / max(
            len(center_instances), 1
        )
        score += overlap_ratio * 0.4

        # Temporal distance penalty (prefer recent frames)
        temporal_distance = abs(center_idx - frame_idx)
        temporal_penalty = 1.0 / (1.0 + temporal_distance * 0.1)
        score *= temporal_penalty

        # Scene complexity (number of objects)
        complexity_score = (
            len(frame.instances) / 20.0
        )  # Normalize by typical max objects
        score += complexity_score * 0.3

        return score

    def _compute_ego_motions(self, frames: List[Sample]) -> np.ndarray:
        """Compute ego motions between consecutive frames"""
        ego_motions = []

        for i in range(len(frames) - 1):
            current_pose = frames[i].ego_pose
            next_pose = frames[i + 1].ego_pose

            # Compute relative transformation
            relative_transform = np.linalg.inv(current_pose) @ next_pose

            # Extract translation and rotation
            translation = relative_transform[:3, 3]
            rotation_matrix = relative_transform[:3, :3]

            # Convert rotation matrix to euler angles
            from scipy.spatial.transform import Rotation

            rotation = Rotation.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler("xyz")

            # Combine into ego motion vector [dx, dy, dz, roll, pitch, yaw]
            ego_motion = np.concatenate([translation, euler_angles])
            ego_motions.append(ego_motion)

        return np.array(ego_motions)

    def _compute_ego_motion_between_frames(
        self, dataset: BaseDataset, frame1_idx: int, frame2_idx: int
    ) -> np.ndarray:
        """Compute ego motion between two specific frames"""
        try:
            frame1 = dataset[frame1_idx]
            frame2 = dataset[frame2_idx]

            pose1 = frame1.ego_pose
            pose2 = frame2.ego_pose

            # Compute relative transformation
            relative_transform = np.linalg.inv(pose1) @ pose2

            # Extract translation and rotation
            translation = relative_transform[:3, 3]
            rotation_matrix = relative_transform[:3, :3]

            # Convert to euler angles
            from scipy.spatial.transform import Rotation

            rotation = Rotation.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler("xyz")

            return np.concatenate([translation, euler_angles])

        except (IndexError, KeyError):
            return np.zeros(6)  # Return zero motion if frames unavailable

    def _track_instances(
        self, frames: List[Sample]
    ) -> Dict[str, List[InstanceAnnotation]]:
        """Track instances across frames in sequence"""
        tracks = defaultdict(list)

        for frame in frames:
            for instance in frame.instances:
                tracks[instance.instance_id].append(instance)

        # Filter tracks that appear in at least 2 frames
        filtered_tracks = {}
        for instance_id, annotations in tracks.items():
            if len(annotations) >= 2:
                filtered_tracks[instance_id] = annotations

        return filtered_tracks

    def _compute_temporal_weights(self, frames: List[Sample]) -> np.ndarray:
        """Compute importance weights for each frame in sequence"""
        weights = np.ones(len(frames))

        # Give higher weight to more recent frames
        for i in range(len(frames)):
            # Linear decay towards older frames
            temporal_factor = (i + 1) / len(frames)
            weights[i] = temporal_factor

        # Give highest weight to current frame (last)
        weights[-1] = 1.0

        return weights


class TemporalDataLoader(DataLoader):
    """
    Specialized DataLoader for temporal sequences.

    Handles batch collation for temporal data and provides
    utilities for working with 4D object detection batches.
    """

    def __init__(
        self,
        dataset: BaseDataset,
        sequence_builder: TemporalSequenceBuilder,
        batch_size: int = 1,
        shuffle: bool = True,
        num_workers: int = 4,
        transform: Optional[Transform] = None,
        **kwargs,
    ):
        self.sequence_builder = sequence_builder
        self.transform = transform

        # Create temporal dataset wrapper
        temporal_dataset = TemporalDatasetWrapper(dataset, sequence_builder, transform)

        super().__init__(
            temporal_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_temporal_batch,
            **kwargs,
        )

    def collate_temporal_batch(
        self, batch: List[TemporalSequenceSample]
    ) -> Dict[str, Any]:
        """
        Collate temporal sequences into batch format.

        Returns:
            Dictionary containing batched temporal data for Sparse4D
        """
        if not batch:
            return {}

        batch_size = len(batch)
        max_sequence_length = max(seq.sequence_length for seq in batch)

        # Initialize batch containers
        batch_data = {
            "batch_size": batch_size,
            "max_sequence_length": max_sequence_length,
            "sequence_lengths": [],
            "images": defaultdict(list),  # {camera_name: [batch_frames]}
            "camera_params": {"intrinsics": [], "extrinsics": [], "timestamps": []},
            "instances": [],
            "ego_poses": [],
            "ego_motions": [],
            "temporal_weights": [],
            "instance_tracks": [],
            "sequence_ids": [],
            "frame_indices": [],
        }

        for seq_idx, sequence in enumerate(batch):
            current_frame = sequence.get_current_frame()

            # Store sequence metadata
            batch_data["sequence_lengths"].append(sequence.sequence_length)
            batch_data["sequence_ids"].append(sequence.sequence_id)
            batch_data["frame_indices"].append(sequence.frame_indices)

            # Collate images (use current frame)
            for camera_name, image in current_frame.images.items():
                batch_data["images"][camera_name].append(image)

            # Collate camera parameters (use current frame)
            batch_data["camera_params"]["intrinsics"].append(
                current_frame.camera_params.intrinsics
            )
            batch_data["camera_params"]["extrinsics"].append(
                current_frame.camera_params.extrinsics
            )
            batch_data["camera_params"]["timestamps"].append(
                current_frame.camera_params.timestamps
            )

            # Collate instances (use current frame)
            batch_data["instances"].append(current_frame.instances)

            # Collate ego poses (all frames in sequence)
            seq_ego_poses = np.array([frame.ego_pose for frame in sequence.frames])
            # Pad to max sequence length
            if len(seq_ego_poses) < max_sequence_length:
                padding = np.tile(
                    seq_ego_poses[-1:], (max_sequence_length - len(seq_ego_poses), 1, 1)
                )
                seq_ego_poses = np.concatenate([seq_ego_poses, padding], axis=0)
            batch_data["ego_poses"].append(seq_ego_poses)

            # Collate ego motions
            seq_ego_motions = sequence.ego_motions
            if len(seq_ego_motions) < max_sequence_length - 1:
                padding = np.zeros((max_sequence_length - 1 - len(seq_ego_motions), 6))
                seq_ego_motions = np.concatenate([seq_ego_motions, padding], axis=0)
            batch_data["ego_motions"].append(seq_ego_motions)

            # Collate temporal weights
            seq_weights = sequence.temporal_weights
            if len(seq_weights) < max_sequence_length:
                padding = np.zeros(max_sequence_length - len(seq_weights))
                seq_weights = np.concatenate([seq_weights, padding])
            batch_data["temporal_weights"].append(seq_weights)

            # Collate instance tracks
            batch_data["instance_tracks"].append(sequence.instance_tracks)

        # Convert to tensors where appropriate
        for camera_name in batch_data["images"]:
            batch_data["images"][camera_name] = torch.stack(
                [
                    torch.from_numpy(img) if isinstance(img, np.ndarray) else img
                    for img in batch_data["images"][camera_name]
                ]
            )

        batch_data["camera_params"]["intrinsics"] = torch.tensor(
            np.array(batch_data["camera_params"]["intrinsics"]), dtype=torch.float32
        )
        batch_data["camera_params"]["extrinsics"] = torch.tensor(
            np.array(batch_data["camera_params"]["extrinsics"]), dtype=torch.float32
        )
        batch_data["camera_params"]["timestamps"] = torch.tensor(
            np.array(batch_data["camera_params"]["timestamps"]), dtype=torch.float64
        )

        batch_data["ego_poses"] = torch.tensor(
            np.array(batch_data["ego_poses"]), dtype=torch.float32
        )
        batch_data["ego_motions"] = torch.tensor(
            np.array(batch_data["ego_motions"]), dtype=torch.float32
        )
        batch_data["temporal_weights"] = torch.tensor(
            np.array(batch_data["temporal_weights"]), dtype=torch.float32
        )

        return batch_data


class TemporalDatasetWrapper(Dataset):
    """Wrapper to convert regular dataset to temporal sequence dataset"""

    def __init__(
        self,
        base_dataset: BaseDataset,
        sequence_builder: TemporalSequenceBuilder,
        transform: Optional[Transform] = None,
    ):
        self.base_dataset = base_dataset
        self.sequence_builder = sequence_builder
        self.transform = transform

        # Build valid sequence indices
        self.valid_indices = self._build_valid_indices()

    def _build_valid_indices(self) -> List[int]:
        """Build list of valid center frame indices for temporal sequences"""
        valid_indices = []

        for i in range(len(self.base_dataset)):
            # Check if we can build a valid sequence centered on this frame
            sequence = self.sequence_builder.build_sequence(self.base_dataset, i)
            if sequence is not None:
                valid_indices.append(i)

        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, index: int) -> TemporalSequenceSample:
        center_frame_idx = self.valid_indices[index]

        # Build temporal sequence
        sequence = self.sequence_builder.build_sequence(
            self.base_dataset, center_frame_idx
        )

        if sequence is None:
            raise IndexError(f"Could not build sequence for index {index}")

        # Apply transforms to current frame if provided
        if self.transform:
            current_frame = sequence.get_current_frame()
            transformed_frame = self.transform(current_frame)
            sequence.frames[-1] = transformed_frame  # Update current frame
            sequence.current_frame = transformed_frame

        return sequence


def create_temporal_dataloader(
    dataset: BaseDataset,
    sequence_length: int = 4,
    temporal_stride: int = 1,
    sampling_strategy: str = "uniform",
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[Transform] = None,
    **kwargs,
) -> TemporalDataLoader:
    """
    Factory function to create temporal data loader.

    Args:
        dataset: Base dataset
        sequence_length: Number of frames in each sequence
        temporal_stride: Stride between frames
        sampling_strategy: Frame sampling strategy
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        transform: Optional transforms to apply

    Returns:
        Configured TemporalDataLoader
    """
    sequence_builder = TemporalSequenceBuilder(
        sequence_length=sequence_length,
        temporal_stride=temporal_stride,
        sampling_strategy=sampling_strategy,
        **kwargs,
    )

    return TemporalDataLoader(
        dataset=dataset,
        sequence_builder=sequence_builder,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        transform=transform,
    )
