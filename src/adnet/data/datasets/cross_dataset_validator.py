"""
Cross-dataset validation framework for Sparse4D.

This module provides comprehensive validation and analysis tools for
evaluating model generalization across different autonomous driving datasets,
including domain gap analysis and cross-domain performance evaluation.
"""

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import jensenshannon

from ...interfaces.data.dataset import BaseDataset
from .multi_dataset_loader import UnifiedTaxonomy


@dataclass
class DomainGapMetrics:
    """Container for domain gap analysis metrics.

    Attributes:
        class_distribution_divergence: Jensen-Shannon divergence between class distributions.
        spatial_distribution_divergence: Divergence in spatial object distributions.
        camera_setup_similarity: Similarity score for camera configurations.
        temporal_characteristics_similarity: Similarity in temporal patterns.
        scene_complexity_ratio: Ratio of scene complexity metrics.
        weather_distribution_divergence: Divergence in weather conditions.
        overall_domain_gap_score: Combined domain gap score (0-1, higher = more different).
    """

    class_distribution_divergence: float
    spatial_distribution_divergence: float
    camera_setup_similarity: float
    temporal_characteristics_similarity: float
    scene_complexity_ratio: float
    weather_distribution_divergence: float
    overall_domain_gap_score: float


@dataclass
class CrossDatasetResults:
    """Container for cross-dataset validation results.

    Attributes:
        source_dataset: Name of the source dataset used for training.
        target_dataset: Name of the target dataset used for evaluation.
        domain_gap_metrics: Computed domain gap analysis metrics.
        performance_metrics: Overall performance metrics (mAP, accuracy, etc.).
        class_specific_performance: Per-class performance breakdown.
        failure_analysis: Analysis of failure modes and risk factors.
        recommendations: List of recommendations for improving transfer performance.
    """

    source_dataset: str
    target_dataset: str
    domain_gap_metrics: DomainGapMetrics
    performance_metrics: Dict[str, float]
    class_specific_performance: Dict[str, Dict[str, float]]
    failure_analysis: Dict[str, Any]
    recommendations: List[str]


class DatasetStatisticsAnalyzer:
    """
    Analyzes and compares statistics across different datasets.

    Provides detailed analysis of class distributions, spatial patterns,
    temporal characteristics, and scene complexity.
    """

    def __init__(self, unified_taxonomy: Optional[UnifiedTaxonomy] = None) -> None:
        self.unified_taxonomy = unified_taxonomy or UnifiedTaxonomy()

    def analyze_dataset(self, dataset: BaseDataset) -> Dict[str, Any]:
        """Perform comprehensive analysis of a single dataset.

        Args:
            dataset: The dataset to analyze.

        Returns:
            Dictionary containing detailed analysis results including class distributions,
            spatial patterns, temporal characteristics, and scene complexity metrics.
        """
        print(f"Analyzing dataset: {dataset.__class__.__name__}")

        stats = {
            "dataset_name": dataset.__class__.__name__,
            "total_samples": len(dataset),
            "class_distribution": self._analyze_class_distribution(dataset),
            "spatial_distribution": self._analyze_spatial_distribution(dataset),
            "temporal_characteristics": self._analyze_temporal_characteristics(dataset),
            "scene_complexity": self._analyze_scene_complexity(dataset),
            "camera_characteristics": self._analyze_camera_characteristics(dataset),
            "weather_distribution": self._analyze_weather_distribution(dataset),
            "instance_tracking": self._analyze_instance_tracking(dataset),
        }

        return stats

    def _analyze_class_distribution(self, dataset: BaseDataset) -> Dict[str, Any]:
        """Analyze object class distribution across the dataset.

        Args:
            dataset: The dataset to analyze.

        Returns:
            Dictionary containing class counts, probabilities, and distribution statistics.
        """
        class_counts = Counter()
        total_instances = 0

        # Sample subset for analysis if dataset is large
        sample_indices = self._get_analysis_sample_indices(dataset, max_samples=1000)

        for idx in sample_indices:
            try:
                sample = dataset[idx]
                for instance in sample.instances:
                    # Map to unified taxonomy if available
                    if hasattr(dataset, "class_names") and instance.category_id < len(
                        dataset.class_names
                    ):
                        class_name = dataset.class_names[instance.category_id]

                        # Map to unified class if taxonomy available
                        if self.unified_taxonomy:
                            unified_id = self.unified_taxonomy.map_class(
                                dataset.__class__.__name__.lower(), class_name
                            )
                            if unified_id is not None:
                                unified_class = (
                                    self.unified_taxonomy.get_unified_class_name(
                                        unified_id
                                    )
                                )
                                class_counts[unified_class] += 1
                            else:
                                class_counts[f"unmapped_{class_name}"] += 1
                        else:
                            class_counts[class_name] += 1

                        total_instances += 1
            except Exception as e:
                print(f"Error processing sample {idx}: {e}")
                continue

        # Convert to probabilities
        class_probabilities = {}
        for class_name, count in class_counts.items():
            class_probabilities[class_name] = count / max(total_instances, 1)

        return {
            "class_counts": dict(class_counts),
            "class_probabilities": class_probabilities,
            "total_instances": total_instances,
            "num_classes": len(class_counts),
            "most_common_classes": class_counts.most_common(5),
        }

    def _analyze_spatial_distribution(self, dataset: BaseDataset) -> Dict[str, Any]:
        """Analyze spatial distribution of objects"""
        spatial_stats = {
            "x_positions": [],
            "y_positions": [],
            "z_positions": [],
            "distances_from_ego": [],
            "object_sizes": [],
        }

        sample_indices = self._get_analysis_sample_indices(dataset, max_samples=500)

        for idx in sample_indices:
            try:
                sample = dataset[idx]
                for instance in sample.instances:
                    box_3d = instance.box_3d

                    # Position statistics
                    spatial_stats["x_positions"].append(box_3d[0])
                    spatial_stats["y_positions"].append(box_3d[1])
                    spatial_stats["z_positions"].append(box_3d[2])

                    # Distance from ego
                    distance = np.sqrt(box_3d[0] ** 2 + box_3d[1] ** 2)
                    spatial_stats["distances_from_ego"].append(distance)

                    # Object size (volume)
                    volume = box_3d[3] * box_3d[4] * box_3d[5]  # w * l * h
                    spatial_stats["object_sizes"].append(volume)
            except Exception:
                continue

        # Compute summary statistics
        summary = {}
        for key, values in spatial_stats.items():
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "percentiles": np.percentile(values, [25, 50, 75, 95]),
                }
            else:
                summary[key] = {"mean": 0, "std": 0, "median": 0, "min": 0, "max": 0}

        return summary

    def _analyze_temporal_characteristics(self, dataset: BaseDataset) -> Dict[str, Any]:
        """Analyze temporal characteristics"""
        temporal_stats = {
            "sequence_lengths": [],
            "frame_rates": [],
            "temporal_gaps": [],
        }

        sample_indices = self._get_analysis_sample_indices(dataset, max_samples=200)

        for idx in sample_indices:
            try:
                sample = dataset[idx]
                seq_info = sample.sequence_info

                # Sequence length
                temporal_stats["sequence_lengths"].append(seq_info.frame_count)

                # Frame rate estimation
                if len(seq_info.timestamps) > 1:
                    time_span = (
                        seq_info.timestamps[-1] - seq_info.timestamps[0]
                    ) / 1e6  # Convert to seconds
                    if time_span > 0:
                        fps = len(seq_info.timestamps) / time_span
                        temporal_stats["frame_rates"].append(fps)

                # Temporal gaps between frames
                for i in range(1, len(seq_info.timestamps)):
                    gap = (seq_info.timestamps[i] - seq_info.timestamps[i - 1]) / 1e6
                    temporal_stats["temporal_gaps"].append(gap)
            except Exception:
                continue

        summary = {}
        for key, values in temporal_stats.items():
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                }

        return summary

    def _analyze_scene_complexity(self, dataset: BaseDataset) -> Dict[str, Any]:
        """Analyze scene complexity metrics"""
        complexity_stats = {
            "objects_per_frame": [],
            "unique_classes_per_frame": [],
            "object_density": [],
            "occlusion_levels": [],
        }

        sample_indices = self._get_analysis_sample_indices(dataset, max_samples=500)

        for idx in sample_indices:
            try:
                sample = dataset[idx]

                # Objects per frame
                num_objects = len(sample.instances)
                complexity_stats["objects_per_frame"].append(num_objects)

                # Unique classes per frame
                unique_classes = len(set(inst.category_id for inst in sample.instances))
                complexity_stats["unique_classes_per_frame"].append(unique_classes)

                # Object density (objects per square meter in detection range)
                detection_area = np.pi * 50**2  # Assume 50m detection radius
                density = num_objects / detection_area
                complexity_stats["object_density"].append(density)

                # Occlusion analysis (simplified)
                if hasattr(sample.instances[0], "visibility") and sample.instances:
                    avg_visibility = np.mean(
                        [inst.visibility for inst in sample.instances]
                    )
                    complexity_stats["occlusion_levels"].append(1.0 - avg_visibility)
            except Exception:
                continue

        summary = {}
        for key, values in complexity_stats.items():
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                }

        return summary

    def _analyze_camera_characteristics(self, dataset: BaseDataset) -> Dict[str, Any]:
        """Analyze camera setup characteristics"""
        camera_stats = {
            "num_cameras": 0,
            "camera_names": [],
            "resolutions": [],
            "focal_lengths": [],
            "field_of_views": [],
        }

        # Get sample to analyze camera setup
        try:
            sample = dataset[0]
            camera_params = sample.camera_params

            camera_stats["num_cameras"] = len(camera_params.intrinsics)

            # Analyze intrinsics
            for i, intrinsic in enumerate(camera_params.intrinsics):
                fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                cx, cy = intrinsic[0, 2], intrinsic[1, 2]

                camera_stats["focal_lengths"].append((fx + fy) / 2)

                # Estimate resolution from principal point
                width, height = cx * 2, cy * 2
                camera_stats["resolutions"].append((width, height))

                # Estimate field of view
                fov_x = 2 * np.arctan(cx / fx) * 180 / np.pi
                fov_y = 2 * np.arctan(cy / fy) * 180 / np.pi
                camera_stats["field_of_views"].append((fov_x, fov_y))

            # Get camera names if available
            if hasattr(dataset, "camera_names"):
                camera_stats["camera_names"] = dataset.camera_names

        except Exception as e:
            print(f"Error analyzing camera characteristics: {e}")

        return camera_stats

    def _analyze_weather_distribution(self, dataset: BaseDataset) -> Dict[str, Any]:
        """Analyze weather and environmental conditions"""
        weather_counts = Counter()
        time_counts = Counter()
        location_counts = Counter()

        sample_indices = self._get_analysis_sample_indices(dataset, max_samples=500)

        for idx in sample_indices:
            try:
                sample = dataset[idx]
                weather_counts[sample.weather] += 1
                time_counts[sample.time_of_day] += 1
                location_counts[sample.location] += 1
            except Exception:
                continue

        total_samples = sum(weather_counts.values())

        return {
            "weather_distribution": {
                k: v / total_samples for k, v in weather_counts.items()
            },
            "time_distribution": {k: v / total_samples for k, v in time_counts.items()},
            "location_distribution": {
                k: v / total_samples for k, v in location_counts.items()
            },
            "weather_diversity": len(weather_counts),
            "temporal_diversity": len(time_counts),
            "spatial_diversity": len(location_counts),
        }

    def _analyze_instance_tracking(self, dataset: BaseDataset) -> Dict[str, Any]:
        """Analyze instance tracking characteristics"""
        # TODO: Implement temporal sequence analysis
        # This would require analyzing instance tracking across temporal frames

        # This would require temporal sequence analysis
        # Simplified implementation for now
        return {
            "avg_track_length": 0,
            "track_continuity_ratio": 0,
            "tracking_quality": "unknown",
        }

    def _get_analysis_sample_indices(
        self, dataset: BaseDataset, max_samples: int = 1000
    ) -> List[int]:
        """Get sample indices for analysis (random sampling if dataset is large)"""
        total_samples = len(dataset)

        if total_samples <= max_samples:
            return list(range(total_samples))
        else:
            # Random sampling
            import random

            return random.sample(range(total_samples), max_samples)


class DomainGapAnalyzer:
    """Analyzes domain gaps between different datasets.

    This class computes various metrics to quantify differences between datasets
    and predict cross-domain transfer performance, including class distribution
    divergence, spatial pattern differences, and camera setup variations.

    Attributes:
        statistics_analyzer: Instance for computing dataset statistics.
    """

    def __init__(self) -> None:
        self.statistics_analyzer = DatasetStatisticsAnalyzer()

    def compute_domain_gap(
        self, source_dataset: BaseDataset, target_dataset: BaseDataset
    ) -> DomainGapMetrics:
        """Compute comprehensive domain gap metrics between two datasets.

        Args:
            source_dataset: The source dataset (typically used for training).
            target_dataset: The target dataset (typically used for evaluation).

        Returns:
            DomainGapMetrics object containing all computed gap metrics.
        """

        # Analyze both datasets
        source_stats = self.statistics_analyzer.analyze_dataset(source_dataset)
        target_stats = self.statistics_analyzer.analyze_dataset(target_dataset)

        # Compute individual gap metrics
        class_divergence = self._compute_class_distribution_divergence(
            source_stats["class_distribution"], target_stats["class_distribution"]
        )

        spatial_divergence = self._compute_spatial_distribution_divergence(
            source_stats["spatial_distribution"], target_stats["spatial_distribution"]
        )

        camera_similarity = self._compute_camera_setup_similarity(
            source_stats["camera_characteristics"],
            target_stats["camera_characteristics"],
        )

        temporal_similarity = self._compute_temporal_characteristics_similarity(
            source_stats["temporal_characteristics"],
            target_stats["temporal_characteristics"],
        )

        complexity_ratio = self._compute_scene_complexity_ratio(
            source_stats["scene_complexity"], target_stats["scene_complexity"]
        )

        weather_divergence = self._compute_weather_distribution_divergence(
            source_stats["weather_distribution"], target_stats["weather_distribution"]
        )

        # Compute overall domain gap score
        overall_gap = self._compute_overall_domain_gap(
            class_divergence,
            spatial_divergence,
            camera_similarity,
            temporal_similarity,
            complexity_ratio,
            weather_divergence,
        )

        return DomainGapMetrics(
            class_distribution_divergence=class_divergence,
            spatial_distribution_divergence=spatial_divergence,
            camera_setup_similarity=camera_similarity,
            temporal_characteristics_similarity=temporal_similarity,
            scene_complexity_ratio=complexity_ratio,
            weather_distribution_divergence=weather_divergence,
            overall_domain_gap_score=overall_gap,
        )

    def _compute_class_distribution_divergence(
        self, source_dist: Dict, target_dist: Dict
    ) -> float:
        """Compute Jensen-Shannon divergence between class distributions"""
        try:
            # Get union of all classes
            all_classes = set(source_dist["class_probabilities"].keys()) | set(
                target_dist["class_probabilities"].keys()
            )

            # Build probability vectors
            source_probs = []
            target_probs = []

            for class_name in sorted(all_classes):
                source_probs.append(
                    source_dist["class_probabilities"].get(class_name, 0)
                )
                target_probs.append(
                    target_dist["class_probabilities"].get(class_name, 0)
                )

            # Compute Jensen-Shannon divergence
            js_divergence = jensenshannon(source_probs, target_probs) ** 2
            return float(js_divergence)

        except Exception:
            return 1.0  # Maximum divergence on error

    def _compute_spatial_distribution_divergence(
        self, source_spatial: Dict, target_spatial: Dict
    ) -> float:
        """Compute divergence in spatial distributions"""
        try:
            divergences = []

            # Compare key spatial metrics
            for metric in [
                "distances_from_ego",
                "x_positions",
                "y_positions",
                "object_sizes",
            ]:
                if metric in source_spatial and metric in target_spatial:
                    source_mean = source_spatial[metric]["mean"]
                    target_mean = target_spatial[metric]["mean"]
                    source_std = source_spatial[metric]["std"]
                    target_std = target_spatial[metric]["std"]

                    # Normalized difference in means and stds
                    mean_diff = abs(source_mean - target_mean) / (
                        abs(source_mean) + abs(target_mean) + 1e-6
                    )
                    std_diff = abs(source_std - target_std) / (
                        source_std + target_std + 1e-6
                    )

                    divergences.append((mean_diff + std_diff) / 2)

            return np.mean(divergences) if divergences else 0.5

        except Exception:
            return 0.5

    def _compute_camera_setup_similarity(
        self, source_cameras: Dict, target_cameras: Dict
    ) -> float:
        """Compute similarity in camera setups"""
        try:
            similarity_score = 0.0

            # Number of cameras similarity
            source_num = source_cameras.get("num_cameras", 0)
            target_num = target_cameras.get("num_cameras", 0)
            if max(source_num, target_num) > 0:
                num_similarity = min(source_num, target_num) / max(
                    source_num, target_num
                )
                similarity_score += num_similarity * 0.3

            # Focal length similarity
            source_focal = source_cameras.get("focal_lengths", [])
            target_focal = target_cameras.get("focal_lengths", [])
            if source_focal and target_focal:
                source_avg_focal = np.mean(source_focal)
                target_avg_focal = np.mean(target_focal)
                focal_similarity = 1.0 - abs(source_avg_focal - target_avg_focal) / max(
                    source_avg_focal, target_avg_focal
                )
                similarity_score += focal_similarity * 0.4

            # Resolution similarity
            source_res = source_cameras.get("resolutions", [])
            target_res = target_cameras.get("resolutions", [])
            if source_res and target_res:
                source_avg_res = np.mean([w * h for w, h in source_res])
                target_avg_res = np.mean([w * h for w, h in target_res])
                res_similarity = 1.0 - abs(source_avg_res - target_avg_res) / max(
                    source_avg_res, target_avg_res
                )
                similarity_score += res_similarity * 0.3

            return min(similarity_score, 1.0)

        except Exception:
            return 0.5

    def _compute_temporal_characteristics_similarity(
        self, source_temporal: Dict, target_temporal: Dict
    ) -> float:
        """Compute similarity in temporal characteristics"""
        try:
            similarity_score = 0.0

            # Frame rate similarity
            if "frame_rates" in source_temporal and "frame_rates" in target_temporal:
                source_fps = source_temporal["frame_rates"]["mean"]
                target_fps = target_temporal["frame_rates"]["mean"]
                fps_similarity = 1.0 - abs(source_fps - target_fps) / max(
                    source_fps, target_fps
                )
                similarity_score += fps_similarity * 0.6

            # Sequence length similarity
            if (
                "sequence_lengths" in source_temporal
                and "sequence_lengths" in target_temporal
            ):
                source_seq = source_temporal["sequence_lengths"]["mean"]
                target_seq = target_temporal["sequence_lengths"]["mean"]
                seq_similarity = 1.0 - abs(source_seq - target_seq) / max(
                    source_seq, target_seq
                )
                similarity_score += seq_similarity * 0.4

            return min(similarity_score, 1.0)

        except Exception:
            return 0.5

    def _compute_scene_complexity_ratio(
        self, source_complexity: Dict, target_complexity: Dict
    ) -> float:
        """Compute ratio of scene complexities"""
        try:
            source_objects = source_complexity.get("objects_per_frame", {}).get(
                "mean", 1
            )
            target_objects = target_complexity.get("objects_per_frame", {}).get(
                "mean", 1
            )

            # Ratio closer to 1.0 means similar complexity
            ratio = min(source_objects, target_objects) / max(
                source_objects, target_objects
            )
            return ratio

        except Exception:
            return 0.5

    def _compute_weather_distribution_divergence(
        self, source_weather: Dict, target_weather: Dict
    ) -> float:
        """Compute divergence in weather distributions"""
        try:
            source_dist = source_weather.get("weather_distribution", {})
            target_dist = target_weather.get("weather_distribution", {})

            if not source_dist or not target_dist:
                return 0.5

            # Get union of weather conditions
            all_weather = set(source_dist.keys()) | set(target_dist.keys())

            source_probs = [source_dist.get(w, 0) for w in sorted(all_weather)]
            target_probs = [target_dist.get(w, 0) for w in sorted(all_weather)]

            # Jensen-Shannon divergence
            js_divergence = jensenshannon(source_probs, target_probs) ** 2
            return float(js_divergence)

        except Exception:
            return 0.5

    def _compute_overall_domain_gap(
        self,
        class_div: float,
        spatial_div: float,
        camera_sim: float,
        temporal_sim: float,
        complexity_ratio: float,
        weather_div: float,
    ) -> float:
        """Compute weighted overall domain gap score"""

        # Weights for different aspects
        weights = {
            "class": 0.25,
            "spatial": 0.20,
            "camera": 0.15,
            "temporal": 0.15,
            "complexity": 0.15,
            "weather": 0.10,
        }

        # Convert similarities to divergences for consistency
        camera_div = 1.0 - camera_sim
        temporal_div = 1.0 - temporal_sim
        complexity_div = 1.0 - complexity_ratio

        # Weighted average
        overall_gap = (
            weights["class"] * class_div
            + weights["spatial"] * spatial_div
            + weights["camera"] * camera_div
            + weights["temporal"] * temporal_div
            + weights["complexity"] * complexity_div
            + weights["weather"] * weather_div
        )

        return overall_gap


class CrossDatasetValidator:
    """
    Main class for cross-dataset validation and analysis.

    Orchestrates domain gap analysis, performance evaluation,
    and provides recommendations for cross-domain training.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        self.output_dir = output_dir
        self.domain_gap_analyzer = DomainGapAnalyzer()
        self.results_history = []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def validate_cross_dataset_transfer(
        self,
        source_datasets: List[BaseDataset],
        target_datasets: List[BaseDataset],
        model_performance_fn: Optional[callable] = None,
    ) -> List[CrossDatasetResults]:
        """
        Perform comprehensive cross-dataset validation.

        Args:
            source_datasets: List of source datasets for training
            target_datasets: List of target datasets for evaluation
            model_performance_fn: Optional function to evaluate model performance

        Returns:
            List of validation results for each source-target pair
        """
        results = []

        for source_dataset in source_datasets:
            for target_dataset in target_datasets:
                if source_dataset == target_dataset:
                    continue  # Skip same dataset

                print(
                    f"Analyzing transfer: {source_dataset.__class__.__name__} → {target_dataset.__class__.__name__}"
                )

                # Compute domain gap
                domain_gap = self.domain_gap_analyzer.compute_domain_gap(
                    source_dataset, target_dataset
                )

                # Evaluate performance if function provided
                performance_metrics = {}
                class_specific_performance = {}
                if model_performance_fn:
                    (
                        performance_metrics,
                        class_specific_performance,
                    ) = model_performance_fn(source_dataset, target_dataset)

                # Analyze failure modes
                failure_analysis = self._analyze_failure_modes(
                    source_dataset, target_dataset, domain_gap
                )

                # Generate recommendations
                recommendations = self._generate_recommendations(
                    domain_gap, performance_metrics
                )

                result = CrossDatasetResults(
                    source_dataset=source_dataset.__class__.__name__,
                    target_dataset=target_dataset.__class__.__name__,
                    domain_gap_metrics=domain_gap,
                    performance_metrics=performance_metrics,
                    class_specific_performance=class_specific_performance,
                    failure_analysis=failure_analysis,
                    recommendations=recommendations,
                )

                results.append(result)
                self.results_history.append(result)

        # Save results if output directory specified
        if self.output_dir:
            self._save_validation_results(results)

        return results

    def _analyze_failure_modes(
        self,
        source_dataset: BaseDataset,
        target_dataset: BaseDataset,
        domain_gap: DomainGapMetrics,
    ) -> Dict[str, Any]:
        """Analyze potential failure modes in cross-dataset transfer"""

        failure_analysis = {
            "high_risk_classes": [],
            "spatial_bias_risk": "low",
            "temporal_mismatch_risk": "low",
            "camera_adaptation_required": False,
            "weather_robustness_issues": [],
        }

        # Identify high-risk classes based on distribution differences
        if domain_gap.class_distribution_divergence > 0.5:
            failure_analysis["high_risk_classes"] = [
                "vehicle.truck",
                "vehicle.construction",
                "movable.barrier",
            ]  # Classes that typically vary across datasets

        # Spatial bias assessment
        if domain_gap.spatial_distribution_divergence > 0.4:
            failure_analysis["spatial_bias_risk"] = "high"
        elif domain_gap.spatial_distribution_divergence > 0.2:
            failure_analysis["spatial_bias_risk"] = "medium"

        # Temporal mismatch assessment
        if domain_gap.temporal_characteristics_similarity < 0.5:
            failure_analysis["temporal_mismatch_risk"] = "high"
        elif domain_gap.temporal_characteristics_similarity < 0.7:
            failure_analysis["temporal_mismatch_risk"] = "medium"

        # Camera adaptation requirements
        if domain_gap.camera_setup_similarity < 0.7:
            failure_analysis["camera_adaptation_required"] = True

        # Weather robustness
        if domain_gap.weather_distribution_divergence > 0.3:
            failure_analysis["weather_robustness_issues"] = [
                "rain",
                "night",
                "fog",
            ]  # Common challenging conditions

        return failure_analysis

    def _generate_recommendations(
        self, domain_gap: DomainGapMetrics, performance_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations for improving cross-dataset transfer"""

        recommendations = []

        # Class imbalance recommendations
        if domain_gap.class_distribution_divergence > 0.4:
            recommendations.append(
                "Apply class rebalancing or data augmentation for underrepresented classes"
            )
            recommendations.append(
                "Consider focal loss or class-weighted loss functions"
            )

        # Spatial domain adaptation
        if domain_gap.spatial_distribution_divergence > 0.3:
            recommendations.append(
                "Use domain adaptation techniques (DANN, CORAL) for spatial distributions"
            )
            recommendations.append("Apply spatial data augmentation during training")

        # Camera setup adaptation
        if domain_gap.camera_setup_similarity < 0.6:
            recommendations.append(
                "Fine-tune camera intrinsic parameters or use camera-agnostic features"
            )
            recommendations.append(
                "Apply multi-scale training to handle resolution differences"
            )

        # Temporal characteristics
        if domain_gap.temporal_characteristics_similarity < 0.5:
            recommendations.append(
                "Adjust temporal sequence length and sampling strategy"
            )
            recommendations.append(
                "Use temporal data augmentation (frame dropout, rate simulation)"
            )

        # Scene complexity adaptation
        if domain_gap.scene_complexity_ratio < 0.7:
            recommendations.append(
                "Gradually increase scene complexity during training (curriculum learning)"
            )
            recommendations.append(
                "Use progressive training from simple to complex scenes"
            )

        # Weather robustness
        if domain_gap.weather_distribution_divergence > 0.3:
            recommendations.append(
                "Apply weather-specific data augmentation (photometric, atmospheric)"
            )
            recommendations.append(
                "Use domain-specific batch normalization for different weather conditions"
            )

        # General recommendations based on overall gap
        if domain_gap.overall_domain_gap_score > 0.6:
            recommendations.append(
                "Consider gradual domain adaptation with intermediate datasets"
            )
            recommendations.append(
                "Use self-supervised pre-training on target domain data"
            )

        return recommendations

    def _save_validation_results(self, results: List[CrossDatasetResults]) -> None:
        """Save validation results to files"""

        # Save summary report
        summary_path = os.path.join(
            self.output_dir, "cross_dataset_validation_summary.json"
        )
        summary_data = {
            "validation_timestamp": pd.Timestamp.now().isoformat(),
            "num_dataset_pairs": len(results),
            "results": [],
        }

        for result in results:
            result_data = {
                "source_dataset": result.source_dataset,
                "target_dataset": result.target_dataset,
                "overall_domain_gap": result.domain_gap_metrics.overall_domain_gap_score,
                "performance_metrics": result.performance_metrics,
                "num_recommendations": len(result.recommendations),
            }
            summary_data["results"].append(result_data)

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Save detailed results
        for result in results:
            filename = f"{result.source_dataset}_to_{result.target_dataset}.json"
            result_path = os.path.join(self.output_dir, filename)

            detailed_data = {
                "source_dataset": result.source_dataset,
                "target_dataset": result.target_dataset,
                "domain_gap_metrics": {
                    "class_distribution_divergence": result.domain_gap_metrics.class_distribution_divergence,
                    "spatial_distribution_divergence": result.domain_gap_metrics.spatial_distribution_divergence,
                    "camera_setup_similarity": result.domain_gap_metrics.camera_setup_similarity,
                    "temporal_characteristics_similarity": result.domain_gap_metrics.temporal_characteristics_similarity,
                    "scene_complexity_ratio": result.domain_gap_metrics.scene_complexity_ratio,
                    "weather_distribution_divergence": result.domain_gap_metrics.weather_distribution_divergence,
                    "overall_domain_gap_score": result.domain_gap_metrics.overall_domain_gap_score,
                },
                "performance_metrics": result.performance_metrics,
                "class_specific_performance": result.class_specific_performance,
                "failure_analysis": result.failure_analysis,
                "recommendations": result.recommendations,
            }

            with open(result_path, "w") as f:
                json.dump(detailed_data, f, indent=2)

        print(f"Validation results saved to {self.output_dir}")

    def generate_visualization_report(self, results: List[CrossDatasetResults]) -> None:
        """Generate visualization report for cross-dataset validation"""

        if not self.output_dir:
            print("No output directory specified for visualizations")
            return

        # Create domain gap heatmap
        self._create_domain_gap_heatmap(results)

        # Create performance comparison plots
        self._create_performance_plots(results)

        # Create recommendation summary
        self._create_recommendation_summary(results)

    def _create_domain_gap_heatmap(self, results: List[CrossDatasetResults]) -> None:
        """Create heatmap of domain gaps between datasets"""

        # Collect data for heatmap
        datasets = set()
        for result in results:
            datasets.add(result.source_dataset)
            datasets.add(result.target_dataset)

        datasets = sorted(list(datasets))
        gap_matrix = np.zeros((len(datasets), len(datasets)))

        for result in results:
            source_idx = datasets.index(result.source_dataset)
            target_idx = datasets.index(result.target_dataset)
            gap_matrix[source_idx, target_idx] = (
                result.domain_gap_metrics.overall_domain_gap_score
            )

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            gap_matrix,
            xticklabels=datasets,
            yticklabels=datasets,
            annot=True,
            cmap="viridis_r",
            cbar_kws={"label": "Domain Gap Score"},
        )
        plt.title("Cross-Dataset Domain Gap Heatmap")
        plt.xlabel("Target Dataset")
        plt.ylabel("Source Dataset")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "domain_gap_heatmap.png"), dpi=300)
        plt.close()

    def _create_performance_plots(self, results: List[CrossDatasetResults]) -> None:
        """Create performance comparison plots"""
        # Implementation would create various performance visualizations
        pass

    def _create_recommendation_summary(self, results: List[CrossDatasetResults]) -> str:
        """Create summary of recommendations across all dataset pairs"""

        all_recommendations = []
        for result in results:
            all_recommendations.extend(result.recommendations)

        recommendation_counts = Counter(all_recommendations)

        # Save recommendation summary
        summary_path = os.path.join(self.output_dir, "recommendation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "most_common_recommendations": recommendation_counts.most_common(
                        10
                    ),
                    "total_unique_recommendations": len(recommendation_counts),
                    "total_recommendations": len(all_recommendations),
                },
                f,
                indent=2,
            )
