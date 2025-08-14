"""
Data transforms and augmentations for Sparse4D framework.

This module provides comprehensive data transformation pipelines including:
- Multi-view image augmentations
- Temporal sequence transformations  
- 3D spatial augmentations
- Camera parameter adjustments
- Instance tracking preservation
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import cv2
from PIL import Image
import torch
import torch.nn.functional as F

from ...interfaces.data.dataset import Sample, CameraParams, InstanceAnnotation


class Transform:
    """Base class for all data transformations"""
    
    def __init__(self, probability: float = 1.0):
        self.probability = probability
    
    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.probability:
            return self.apply(sample)
        return sample
    
    def apply(self, sample: Sample) -> Sample:
        raise NotImplementedError


class Compose:
    """Compose multiple transforms together"""
    
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
    
    def __call__(self, sample: Sample) -> Sample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class MultiViewImageTransform(Transform):
    """Base class for multi-view image transformations"""
    
    def apply_to_images(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply transformation to all camera images"""
        transformed_images = {}
        for camera_name, image in images.items():
            transformed_images[camera_name] = self.apply_to_single_image(image)
        return transformed_images
    
    def apply_to_single_image(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class PhotometricAugmentation(MultiViewImageTransform):
    """
    Photometric augmentations for multi-view images.
    
    Includes brightness, contrast, saturation, and hue adjustments
    while maintaining consistency across camera views.
    """
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        hue_range: Tuple[float, float] = (-0.1, 0.1),
        probability: float = 0.5,
        consistent_across_views: bool = True
    ):
        super().__init__(probability)
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.consistent_across_views = consistent_across_views
    
    def apply(self, sample: Sample) -> Sample:
        if self.consistent_across_views:
            # Use same parameters for all camera views
            brightness_factor = random.uniform(*self.brightness_range)
            contrast_factor = random.uniform(*self.contrast_range)
            saturation_factor = random.uniform(*self.saturation_range)
            hue_factor = random.uniform(*self.hue_range)
            
            params = {
                'brightness': brightness_factor,
                'contrast': contrast_factor,
                'saturation': saturation_factor,
                'hue': hue_factor
            }
        else:
            params = None
        
        transformed_images = {}
        for camera_name, image in sample.images.items():
            if not self.consistent_across_views:
                params = {
                    'brightness': random.uniform(*self.brightness_range),
                    'contrast': random.uniform(*self.contrast_range),
                    'saturation': random.uniform(*self.saturation_range),
                    'hue': random.uniform(*self.hue_range)
                }
            
            transformed_images[camera_name] = self._apply_photometric(image, params)
        
        sample.images = transformed_images
        return sample
    
    def _apply_photometric(self, image: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """Apply photometric transformations to single image"""
        # Convert to PIL for easy manipulation
        if image.dtype == np.uint8:
            pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
        
        # Convert to HSV for hue/saturation adjustment
        hsv_image = pil_image.convert('HSV')
        h, s, v = hsv_image.split()
        
        # Adjust brightness (value channel)
        v = np.array(v, dtype=np.float32)
        v = v * params['brightness']
        v = np.clip(v, 0, 255).astype(np.uint8)
        v = Image.fromarray(v, mode='L')
        
        # Adjust saturation
        s = np.array(s, dtype=np.float32)
        s = s * params['saturation']
        s = np.clip(s, 0, 255).astype(np.uint8)
        s = Image.fromarray(s, mode='L')
        
        # Adjust hue
        h = np.array(h, dtype=np.float32)
        h = h + params['hue'] * 255
        h = np.clip(h, 0, 255).astype(np.uint8)
        h = Image.fromarray(h, mode='L')
        
        # Recombine HSV
        hsv_adjusted = Image.merge('HSV', (h, s, v))
        rgb_adjusted = hsv_adjusted.convert('RGB')
        
        # Apply contrast
        contrast_image = np.array(rgb_adjusted, dtype=np.float32)
        contrast_image = contrast_image * params['contrast']
        contrast_image = np.clip(contrast_image, 0, 255)
        
        return contrast_image.astype(image.dtype)


class MultiViewResize(MultiViewImageTransform):
    """Resize all camera images to target size"""
    
    def __init__(self, target_size: Tuple[int, int], probability: float = 1.0):
        super().__init__(probability)
        self.target_size = target_size  # (height, width)
    
    def apply(self, sample: Sample) -> Sample:
        original_sizes = {}
        transformed_images = {}
        
        for camera_name, image in sample.images.items():
            original_sizes[camera_name] = (image.shape[0], image.shape[1])  # (H, W)
            
            # Resize image
            resized_image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            transformed_images[camera_name] = resized_image
        
        # Update camera intrinsics to account for resize
        sample.images = transformed_images
        sample.camera_params = self._update_camera_intrinsics(
            sample.camera_params, original_sizes, self.target_size
        )
        
        return sample
    
    def _update_camera_intrinsics(
        self, 
        camera_params: CameraParams, 
        original_sizes: Dict[str, Tuple[int, int]], 
        target_size: Tuple[int, int]
    ) -> CameraParams:
        """Update camera intrinsics after resize"""
        updated_intrinsics = []
        
        for i, camera_name in enumerate(['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                                       'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_BACK_LEFT']):
            if camera_name in original_sizes:
                orig_h, orig_w = original_sizes[camera_name]
                target_h, target_w = target_size
                
                # Scale factors
                scale_x = target_w / orig_w
                scale_y = target_h / orig_h
                
                # Update intrinsic matrix
                intrinsic = camera_params.intrinsics[i].copy()
                intrinsic[0, 0] *= scale_x  # fx
                intrinsic[1, 1] *= scale_y  # fy
                intrinsic[0, 2] *= scale_x  # cx
                intrinsic[1, 2] *= scale_y  # cy
                
                updated_intrinsics.append(intrinsic)
            else:
                updated_intrinsics.append(camera_params.intrinsics[i])
        
        camera_params.intrinsics = np.array(updated_intrinsics)
        return camera_params


class SpatialAugmentation3D(Transform):
    """
    3D spatial augmentations for autonomous driving scenes.
    
    Includes rotation, translation, and scaling while maintaining
    geometric consistency across all modalities.
    """
    
    def __init__(
        self,
        rotation_range: Tuple[float, float] = (-np.pi/4, np.pi/4),  # radians
        translation_range: Tuple[float, float] = (-2.0, 2.0),  # meters
        scaling_range: Tuple[float, float] = (0.9, 1.1),
        probability: float = 0.5
    ):
        super().__init__(probability)
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scaling_range = scaling_range
    
    def apply(self, sample: Sample) -> Sample:
        # Generate transformation parameters
        rotation_angle = random.uniform(*self.rotation_range)
        translation_x = random.uniform(*self.translation_range)
        translation_y = random.uniform(*self.translation_range)
        scale_factor = random.uniform(*self.scaling_range)
        
        # Build transformation matrix
        transform_matrix = self._build_transform_matrix(
            rotation_angle, translation_x, translation_y, scale_factor
        )
        
        # Apply to all geometric data
        sample = self._transform_ego_pose(sample, transform_matrix)
        sample = self._transform_instances(sample, transform_matrix)
        sample = self._transform_camera_params(sample, transform_matrix)
        sample = self._transform_point_clouds(sample, transform_matrix)
        
        return sample
    
    def _build_transform_matrix(
        self, rotation: float, tx: float, ty: float, scale: float
    ) -> np.ndarray:
        """Build 4x4 transformation matrix"""
        # Rotation matrix (around Z-axis)
        cos_r, sin_r = np.cos(rotation), np.sin(rotation)
        rotation_matrix = np.array([
            [cos_r, -sin_r, 0],
            [sin_r, cos_r, 0],
            [0, 0, 1]
        ])
        
        # Scale and build full transformation
        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix * scale
        transform[0, 3] = tx
        transform[1, 3] = ty
        
        return transform
    
    def _transform_ego_pose(self, sample: Sample, transform: np.ndarray) -> Sample:
        """Transform ego pose"""
        sample.ego_pose = transform @ sample.ego_pose
        
        # Transform sequence poses
        for i in range(len(sample.sequence_info.ego_poses)):
            sample.sequence_info.ego_poses[i] = transform @ sample.sequence_info.ego_poses[i]
        
        return sample
    
    def _transform_instances(self, sample: Sample, transform: np.ndarray) -> Sample:
        """Transform 3D bounding boxes"""
        transformed_instances = []
        
        for instance in sample.instances:
            box_3d = instance.box_3d.copy()
            
            # Transform center position
            center = np.array([box_3d[0], box_3d[1], box_3d[2], 1.0])
            transformed_center = (transform @ center)[:3]
            
            # Transform rotation
            yaw = np.arctan2(box_3d[7], box_3d[6])  # Current yaw
            
            # Extract rotation from transform matrix
            transform_rotation = np.arctan2(transform[1, 0], transform[0, 0])
            new_yaw = yaw + transform_rotation
            
            # Update box
            box_3d[0:3] = transformed_center
            box_3d[6] = np.cos(new_yaw)
            box_3d[7] = np.sin(new_yaw)
            
            # Scale dimensions
            scale = np.linalg.norm(transform[:3, 0])  # Extract scale factor
            box_3d[3:6] *= scale  # Scale w, l, h
            
            instance.box_3d = box_3d
            transformed_instances.append(instance)
        
        sample.instances = transformed_instances
        return sample
    
    def _transform_camera_params(self, sample: Sample, transform: np.ndarray) -> Sample:
        """Transform camera extrinsics"""
        transformed_extrinsics = []
        
        for extrinsic in sample.camera_params.extrinsics:
            # Apply transformation to camera pose
            transformed_extrinsic = transform @ extrinsic
            transformed_extrinsics.append(transformed_extrinsic)
        
        sample.camera_params.extrinsics = np.array(transformed_extrinsics)
        return sample
    
    def _transform_point_clouds(self, sample: Sample, transform: np.ndarray) -> Sample:
        """Transform point clouds (LiDAR/radar)"""
        if sample.lidar_points is not None:
            points = sample.lidar_points
            # Transform xyz coordinates
            xyz = points[:, :3]
            xyz_homo = np.column_stack([xyz, np.ones(len(xyz))])
            transformed_xyz = (transform @ xyz_homo.T).T[:, :3]
            
            # Update points
            sample.lidar_points = np.column_stack([transformed_xyz, points[:, 3:]])
        
        if sample.radar_points is not None:
            points = sample.radar_points
            # Transform xyz coordinates
            xyz = points[:, :3]
            xyz_homo = np.column_stack([xyz, np.ones(len(xyz))])
            transformed_xyz = (transform @ xyz_homo.T).T[:, :3]
            
            # Update points (keeping velocity components as-is for simplicity)
            sample.radar_points = np.column_stack([transformed_xyz, points[:, 3:]])
        
        return sample


class TemporalAugmentation(Transform):
    """
    Temporal sequence augmentations for 4D object detection.
    
    Includes temporal dropout, sequence reordering, and
    frame rate simulation.
    """
    
    def __init__(
        self,
        dropout_probability: float = 0.1,
        max_temporal_gap: float = 0.2,  # seconds
        frame_rate_simulation: bool = True,
        target_fps: float = 2.0,
        probability: float = 0.3
    ):
        super().__init__(probability)
        self.dropout_probability = dropout_probability
        self.max_temporal_gap = max_temporal_gap
        self.frame_rate_simulation = frame_rate_simulation
        self.target_fps = target_fps
    
    def apply(self, sample: Sample) -> Sample:
        # Apply temporal dropout
        if random.random() < self.dropout_probability:
            sample = self._apply_temporal_dropout(sample)
        
        # Simulate different frame rates
        if self.frame_rate_simulation:
            sample = self._simulate_frame_rate(sample)
        
        return sample
    
    def _apply_temporal_dropout(self, sample: Sample) -> Sample:
        """Randomly drop frames from temporal sequence"""
        sequence_info = sample.sequence_info
        
        if len(sequence_info.frame_indices) <= 1:
            return sample  # Can't dropout from single frame
        
        # Randomly select frames to keep
        keep_indices = []
        for i, frame_idx in enumerate(sequence_info.frame_indices):
            if random.random() > self.dropout_probability:
                keep_indices.append(i)
        
        # Ensure at least one frame remains
        if not keep_indices:
            keep_indices = [len(sequence_info.frame_indices) - 1]  # Keep last frame
        
        # Update sequence info
        sequence_info.frame_indices = [sequence_info.frame_indices[i] for i in keep_indices]
        sequence_info.timestamps = sequence_info.timestamps[keep_indices]
        sequence_info.ego_poses = sequence_info.ego_poses[keep_indices]
        
        return sample
    
    def _simulate_frame_rate(self, sample: Sample) -> Sample:
        """Simulate different frame rates by temporal subsampling"""
        sequence_info = sample.sequence_info
        timestamps = sequence_info.timestamps
        
        if len(timestamps) <= 1:
            return sample
        
        # Calculate current frame rate
        time_span = timestamps[-1] - timestamps[0]
        current_fps = len(timestamps) / (time_span / 1e6)  # Convert microseconds to seconds
        
        # Subsample to target frame rate
        if current_fps > self.target_fps:
            step = int(current_fps / self.target_fps)
            selected_indices = list(range(0, len(timestamps), step))
            
            # Always include the last frame
            if selected_indices[-1] != len(timestamps) - 1:
                selected_indices.append(len(timestamps) - 1)
            
            # Update sequence info
            sequence_info.frame_indices = [sequence_info.frame_indices[i] for i in selected_indices]
            sequence_info.timestamps = sequence_info.timestamps[selected_indices]
            sequence_info.ego_poses = sequence_info.ego_poses[selected_indices]
        
        return sample


class CutMix3D(Transform):
    """
    3D CutMix augmentation for multi-view object detection.
    
    Randomly replaces 3D regions from one sample with another,
    maintaining geometric consistency across views.
    """
    
    def __init__(
        self,
        mix_probability: float = 0.5,
        area_ratio_range: Tuple[float, float] = (0.1, 0.3),
        probability: float = 0.3
    ):
        super().__init__(probability)
        self.mix_probability = mix_probability
        self.area_ratio_range = area_ratio_range
        self._mix_sample = None
    
    def set_mix_sample(self, mix_sample: Sample):
        """Set the sample to mix with"""
        self._mix_sample = mix_sample
    
    def apply(self, sample: Sample) -> Sample:
        if self._mix_sample is None:
            return sample
        
        # Generate random 3D bounding box for mixing
        mix_box = self._generate_mix_region(sample)
        
        # Apply mixing to all modalities
        sample = self._mix_images(sample, self._mix_sample, mix_box)
        sample = self._mix_instances(sample, self._mix_sample, mix_box)
        sample = self._mix_point_clouds(sample, self._mix_sample, mix_box)
        
        return sample
    
    def _generate_mix_region(self, sample: Sample) -> np.ndarray:
        """Generate random 3D region for mixing"""
        # Get scene bounds from existing instances
        if sample.instances:
            centers = np.array([inst.box_3d[:3] for inst in sample.instances])
            min_bounds = centers.min(axis=0) - 10.0
            max_bounds = centers.max(axis=0) + 10.0
        else:
            # Default scene bounds
            min_bounds = np.array([-50.0, -50.0, -5.0])
            max_bounds = np.array([50.0, 50.0, 5.0])
        
        # Random region size
        scene_size = max_bounds - min_bounds
        area_ratio = random.uniform(*self.area_ratio_range)
        region_size = scene_size * np.sqrt(area_ratio)
        
        # Random region center
        region_center = min_bounds + (max_bounds - min_bounds) * np.random.random(3)
        
        # Build region box [x, y, z, w, l, h]
        mix_box = np.concatenate([region_center, region_size])
        return mix_box
    
    def _mix_images(self, sample: Sample, mix_sample: Sample, mix_box: np.ndarray) -> Sample:
        """Mix images based on 3D region projection"""
        # This would require projecting the 3D region to each camera view
        # and performing 2D mixing - simplified implementation
        return sample
    
    def _mix_instances(self, sample: Sample, mix_sample: Sample, mix_box: np.ndarray) -> Sample:
        """Mix instances within the 3D region"""
        # Remove instances in mix region
        filtered_instances = []
        for instance in sample.instances:
            if not self._box_overlaps_region(instance.box_3d, mix_box):
                filtered_instances.append(instance)
        
        # Add instances from mix sample that fall within region
        for instance in mix_sample.instances:
            if self._box_overlaps_region(instance.box_3d, mix_box):
                filtered_instances.append(instance)
        
        sample.instances = filtered_instances
        return sample
    
    def _mix_point_clouds(self, sample: Sample, mix_sample: Sample, mix_box: np.ndarray) -> Sample:
        """Mix point clouds within the 3D region"""
        if sample.lidar_points is not None and mix_sample.lidar_points is not None:
            # Filter points outside mix region
            keep_mask = ~self._points_in_region(sample.lidar_points[:, :3], mix_box)
            filtered_points = sample.lidar_points[keep_mask]
            
            # Add points from mix sample within region
            mix_mask = self._points_in_region(mix_sample.lidar_points[:, :3], mix_box)
            mix_points = mix_sample.lidar_points[mix_mask]
            
            # Combine
            sample.lidar_points = np.vstack([filtered_points, mix_points])
        
        return sample
    
    def _box_overlaps_region(self, box_3d: np.ndarray, region: np.ndarray) -> bool:
        """Check if 3D box overlaps with mix region"""
        box_center = box_3d[:3]
        box_size = box_3d[3:6]
        
        region_center = region[:3]
        region_size = region[3:6]
        
        # Simple AABB overlap check
        box_min = box_center - box_size / 2
        box_max = box_center + box_size / 2
        region_min = region_center - region_size / 2
        region_max = region_center + region_size / 2
        
        return (box_min <= region_max).all() and (region_min <= box_max).all()
    
    def _points_in_region(self, points: np.ndarray, region: np.ndarray) -> np.ndarray:
        """Check which points are within the mix region"""
        region_center = region[:3]
        region_size = region[3:6]
        
        region_min = region_center - region_size / 2
        region_max = region_center + region_size / 2
        
        within_bounds = (
            (points >= region_min) & (points <= region_max)
        ).all(axis=1)
        
        return within_bounds


class Normalize(Transform):
    """Normalize images with dataset statistics"""
    
    def __init__(
        self,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        probability: float = 1.0
    ):
        super().__init__(probability)
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def apply(self, sample: Sample) -> Sample:
        normalized_images = {}
        
        for camera_name, image in sample.images.items():
            # Convert to float and normalize to [0, 1]
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            
            # Apply normalization
            normalized_image = (image - self.mean) / self.std
            normalized_images[camera_name] = normalized_image
        
        sample.images = normalized_images
        return sample


def create_training_transform_pipeline(
    target_image_size: Tuple[int, int] = (448, 800),
    enable_photometric: bool = True,
    enable_spatial_3d: bool = True,
    enable_temporal: bool = True,
    enable_cutmix: bool = False
) -> Compose:
    """Create training transformation pipeline"""
    transforms = []
    
    # Resize images
    transforms.append(MultiViewResize(target_image_size))
    
    # Photometric augmentations
    if enable_photometric:
        transforms.append(PhotometricAugmentation(probability=0.5))
    
    # 3D spatial augmentations
    if enable_spatial_3d:
        transforms.append(SpatialAugmentation3D(probability=0.3))
    
    # Temporal augmentations
    if enable_temporal:
        transforms.append(TemporalAugmentation(probability=0.2))
    
    # CutMix (requires special handling)
    if enable_cutmix:
        transforms.append(CutMix3D(probability=0.1))
    
    # Normalization (always last)
    transforms.append(Normalize())
    
    return Compose(transforms)


def create_validation_transform_pipeline(
    target_image_size: Tuple[int, int] = (448, 800)
) -> Compose:
    """Create validation transformation pipeline (no augmentations)"""
    transforms = [
        MultiViewResize(target_image_size),
        Normalize()
    ]
    
    return Compose(transforms)