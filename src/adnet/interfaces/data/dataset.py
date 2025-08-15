"""
Abstract base classes for dataset interfaces in Sparse4D framework.

This module defines the core interfaces that all dataset implementations must follow,
ensuring consistency across different datasets (nuScenes, Waymo, KITTI, etc.) and
enabling seamless multi-dataset training and evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset


@dataclass
class CameraParams:
    """Camera parameter container for multi-view setup"""

    intrinsics: np.ndarray  # [N_cameras, 3, 3] intrinsic matrices
    extrinsics: np.ndarray  # [N_cameras, 4, 4] extrinsic matrices (camera to ego)
    timestamps: np.ndarray  # [N_cameras] timestamps for each camera
    distortion: Optional[np.ndarray] = None  # [N_cameras, 5] distortion coefficients


@dataclass
class InstanceAnnotation:
    """Single object instance annotation"""

    box_3d: np.ndarray  # [9] - [x, y, z, w, l, h, cos(yaw), sin(yaw), velocity]
    category_id: int  # Object category ID
    instance_id: int  # Unique instance ID for tracking
    visibility: float  # Visibility score [0, 1]
    attributes: Dict[str, Any]  # Additional attributes (difficulty, truncation, etc.)
    confidence: float = 1.0  # Annotation confidence


@dataclass
class TemporalSequence:
    """Temporal sequence metadata for 4D processing"""

    sequence_id: str  # Unique sequence identifier
    frame_indices: List[int]  # Frame indices in sequence
    timestamps: np.ndarray  # [T] absolute timestamps
    ego_poses: np.ndarray  # [T, 4, 4] ego poses in global coordinates
    frame_count: int  # Total frames in sequence


@dataclass
class Sample:
    """Single data sample containing all modalities and annotations"""

    # Core identifiers
    sample_id: str
    dataset_name: str
    sequence_info: TemporalSequence

    # Visual data
    images: Dict[str, np.ndarray]  # {camera_name: image_array}
    camera_params: CameraParams

    # Annotations
    instances: List[InstanceAnnotation]
    ego_pose: np.ndarray  # [4, 4] current ego pose

    # Optional modalities
    lidar_points: Optional[np.ndarray] = None  # [N, 4] - [x, y, z, intensity]
    radar_points: Optional[np.ndarray] = None  # [M, 6] - [x, y, z, vx, vy, rcs]
    depth_maps: Optional[Dict[str, np.ndarray]] = None  # {camera_name: depth_map}

    # Metadata
    weather: str = "clear"
    time_of_day: str = "day"
    location: str = "unknown"


class BaseDataset(Dataset, ABC):
    """
    Abstract base class for all Sparse4D datasets.

    Defines the interface that all dataset implementations must follow,
    ensuring consistency across nuScenes, Waymo, KITTI, Argoverse, etc.
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        sequence_length: int = 1,
        temporal_stride: int = 1,
        load_interval: int = 1,
        **kwargs,
    ):
        """
        Initialize base dataset.

        Args:
            data_root: Root directory containing dataset files
            split: Dataset split ('train', 'val', 'test')
            sequence_length: Number of frames in temporal sequences
            temporal_stride: Stride between frames in sequence
            load_interval: Interval for loading frames (1 = all frames)
        """
        self.data_root = data_root
        self.split = split
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride
        self.load_interval = load_interval

        # Dataset metadata
        self.class_names: List[str] = []
        self.camera_names: List[str] = []
        self.num_classes: int = 0
        self.num_cameras: int = 0

        # Load dataset-specific information
        self._load_dataset_info()
        self._load_annotations()

    @abstractmethod
    def _load_dataset_info(self) -> None:
        """Load dataset-specific information (classes, camera setup, etc.)"""
        pass

    @abstractmethod
    def _load_annotations(self) -> None:
        """Load and parse dataset annotations"""
        pass

    @abstractmethod
    def _load_sample_data(self, index: int) -> Sample:
        """Load complete sample data for given index"""
        pass

    @abstractmethod
    def get_camera_calibration(self, sample_id: str) -> CameraParams:
        """Get camera calibration parameters for sample"""
        pass

    @abstractmethod
    def get_temporal_sequence(self, sample_id: str) -> TemporalSequence:
        """Get temporal sequence information for sample"""
        pass

    def __len__(self) -> int:
        """Return dataset length"""
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Sample:
        """Get single sample by index"""
        return self._load_sample_data(index)

    @property
    @abstractmethod
    def sample_ids(self) -> List[str]:
        """List of all sample IDs in dataset"""
        pass

    def get_class_mapping(self) -> Dict[str, int]:
        """Get mapping from class names to IDs"""
        return {name: idx for idx, name in enumerate(self.class_names)}

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        return {
            "num_samples": len(self),
            "num_classes": self.num_classes,
            "num_cameras": self.num_cameras,
            "class_names": self.class_names,
            "camera_names": self.camera_names,
            "split": self.split,
            "sequence_length": self.sequence_length,
        }


class MultiModalDataset(BaseDataset):
    """
    Extended base class for multi-modal datasets (camera + LiDAR + radar).

    Handles loading and synchronization of multiple sensor modalities
    with proper temporal alignment.
    """

    def __init__(self, **kwargs):
        self.load_lidar = kwargs.pop("load_lidar", False)
        self.load_radar = kwargs.pop("load_radar", False)
        self.load_depth = kwargs.pop("load_depth", False)
        super().__init__(**kwargs)

    @abstractmethod
    def _load_lidar_data(self, sample_id: str) -> Optional[np.ndarray]:
        """Load LiDAR point cloud data"""
        pass

    @abstractmethod
    def _load_radar_data(self, sample_id: str) -> Optional[np.ndarray]:
        """Load radar point data"""
        pass

    @abstractmethod
    def _load_depth_data(self, sample_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Load depth maps for all cameras"""
        pass

    def _synchronize_modalities(self, sample: Sample) -> Sample:
        """Synchronize different sensor modalities to common timestamp"""
        # Implement temporal synchronization logic
        # This is dataset-specific and should be overridden
        return sample


class TemporalDataset(BaseDataset):
    """
    Base class for temporal datasets supporting 4D object detection.

    Handles temporal sequence construction, ego motion computation,
    and instance tracking across frames.
    """

    def __init__(self, **kwargs):
        self.enable_temporal = kwargs.pop("enable_temporal", True)
        self.max_temporal_gap = kwargs.pop("max_temporal_gap", 1.0)  # seconds
        super().__init__(**kwargs)

        if self.enable_temporal:
            self._build_temporal_sequences()

    def _build_temporal_sequences(self) -> None:
        """Build temporal sequences from individual frames"""
        # Group frames into temporal sequences
        # Handle ego motion computation
        # Track instance IDs across frames
        pass

    @abstractmethod
    def _compute_ego_motion(
        self, prev_pose: np.ndarray, curr_pose: np.ndarray
    ) -> np.ndarray:
        """Compute ego motion between two poses"""
        pass

    @abstractmethod
    def _track_instances(
        self, sequence: TemporalSequence
    ) -> Dict[int, List[InstanceAnnotation]]:
        """Track instances across temporal sequence"""
        pass

    def get_temporal_sample(self, index: int) -> List[Sample]:
        """Get temporal sequence of samples"""
        if not self.enable_temporal or self.sequence_length == 1:
            return [self._load_sample_data(index)]

        # Load temporal sequence
        base_sample = self._load_sample_data(index)
        sequence_info = base_sample.sequence_info

        samples = []
        for frame_idx in sequence_info.frame_indices[-self.sequence_length :]:
            sample = self._load_sample_data(frame_idx)
            samples.append(sample)

        return samples


class HarmonizedDataset(BaseDataset):
    """
    Base class for harmonized multi-dataset training.

    Provides coordinate normalization, class mapping, and
    format standardization across different datasets.
    """

    def __init__(self, **kwargs):
        self.coordinate_system = kwargs.pop("coordinate_system", "ego")
        self.unified_classes = kwargs.pop("unified_classes", None)
        super().__init__(**kwargs)

        if self.unified_classes:
            self._setup_class_harmonization()

    def _setup_class_harmonization(self) -> None:
        """Setup class mapping to unified taxonomy"""
        # Load unified class taxonomy
        # Create mapping from dataset-specific to unified classes
        pass

    @abstractmethod
    def _normalize_coordinates(self, sample: Sample) -> Sample:
        """Normalize coordinates to standard coordinate system"""
        pass

    @abstractmethod
    def _harmonize_annotations(self, sample: Sample) -> Sample:
        """Harmonize annotations to unified format"""
        pass

    def _harmonize_sample(self, sample: Sample) -> Sample:
        """Apply all harmonization steps"""
        sample = self._normalize_coordinates(sample)
        sample = self._harmonize_annotations(sample)
        return sample


class DatasetRegistry:
    """Registry for all available dataset implementations"""

    _datasets = {}

    @classmethod
    def register(cls, name: str, dataset_class: type):
        """Register a dataset implementation"""
        cls._datasets[name] = dataset_class

    @classmethod
    def get(cls, name: str) -> type:
        """Get dataset class by name"""
        if name not in cls._datasets:
            raise ValueError(
                f"Dataset '{name}' not registered. Available: {list(cls._datasets.keys())}"
            )
        return cls._datasets[name]

    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered datasets"""
        return list(cls._datasets.keys())


# Decorator for automatic dataset registration
def register_dataset(name: str):
    """Decorator to automatically register dataset classes"""

    def decorator(cls):
        DatasetRegistry.register(name, cls)
        return cls

    return decorator
