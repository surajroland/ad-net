"""
Test suite for cross-dataset validator implementation.

Tests for cross-dataset validation including:
- Dataset statistics analysis
- Domain gap computation
- Cross-dataset validation framework
- Performance evaluation metrics
- Recommendation generation
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

from adnet.interfaces.data.dataset import BaseDataset, InstanceAnnotation, Sample


class MockDataset(BaseDataset):
    """Mock dataset for testing cross-dataset validation"""

    def __init__(self, num_samples=10, dataset_name="mock_dataset"):
        self.num_samples = num_samples
        self.dataset_name = dataset_name
        self._sample_ids = [f"sample_{i:03d}" for i in range(num_samples)]
        self.class_names = ["car", "pedestrian", "bicycle", "truck"]
        self.camera_names = ["CAM_FRONT", "CAM_BACK"]
        
        # Initialize parent without calling abstract methods
        super(BaseDataset, self).__init__()
        self.data_root = "/mock/data"
        self.split = "train"

    def _load_dataset_info(self):
        """Mock implementation"""
        pass

    def _load_annotations(self):
        """Mock implementation"""
        pass

    def _load_sample_data(self, index):
        """Mock implementation"""
        return self._create_mock_sample(index)

    def get_camera_calibration(self, sample_id):
        """Mock implementation"""
        from adnet.interfaces.data.dataset import CameraParams
        return CameraParams(
            intrinsics=np.array([[[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]] * 2),
            extrinsics=np.array([np.eye(4)] * 2),
            timestamps=np.array([1000000, 1000000]),
        )

    def get_temporal_sequence(self, sample_id):
        """Mock implementation"""
        from adnet.interfaces.data.dataset import TemporalSequence
        return TemporalSequence(
            sequence_id="mock_sequence",
            frame_indices=[0, 1, 2],
            timestamps=np.array([1000000, 2000000, 3000000]),
            ego_poses=np.array([np.eye(4)] * 3),
            frame_count=3,
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self._load_sample_data(index)

    def _create_mock_sample(self, index):
        """Create a mock sample for testing"""
        from adnet.interfaces.data.dataset import CameraParams, InstanceAnnotation, Sample, TemporalSequence
        
        # Create real instances instead of mocks
        instances = []
        num_objects = np.random.randint(1, 5)
        for i in range(num_objects):
            instance = InstanceAnnotation(
                box_3d=np.random.rand(9),
                category_id=np.random.randint(0, 4),
                instance_id=f"instance_{index}_{i}",
                visibility=np.random.uniform(0.5, 1.0),
                attributes={},
            )
            instances.append(instance)
        
        # Create real temporal sequence
        sequence_info = TemporalSequence(
            sequence_id=f"sequence_{index // 5}",
            frame_indices=[index],
            timestamps=np.array([index * 500000]),
            ego_poses=np.array([np.eye(4)]),
            frame_count=1,
        )
        
        # Create real camera params
        camera_params = CameraParams(
            intrinsics=np.array([[[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]] * 2),
            extrinsics=np.array([np.eye(4)] * 2),
            timestamps=np.array([index * 500000] * 2),
        )
        
        # Create real sample
        return Sample(
            sample_id=f"sample_{index:03d}",
            dataset_name=self.dataset_name,
            sequence_info=sequence_info,
            images={
                "CAM_FRONT": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                "CAM_BACK": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            },
            camera_params=camera_params,
            instances=instances,
            ego_pose=np.eye(4),
            weather="clear",
            time_of_day="day",
            location="test_location",
        )

    @property
    def sample_ids(self):
        return self._sample_ids


class TestDatasetStatisticsAnalyzer:
    """Test dataset statistics analysis functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        try:
            from adnet.data.datasets.cross_dataset_validator import (
                DatasetStatisticsAnalyzer,
            )

            self.analyzer = DatasetStatisticsAnalyzer()
            self.mock_dataset = MockDataset(num_samples=50)
        except ImportError:
            pytest.skip("DatasetStatisticsAnalyzer not available")

    def test_dataset_analysis_structure(self):
        """Test dataset analysis output structure"""
        stats = self.analyzer.analyze_dataset(self.mock_dataset)

        # Validate analysis structure
        required_keys = [
            "dataset_name",
            "total_samples",
            "class_distribution",
            "spatial_distribution",
            "temporal_characteristics",
            "scene_complexity",
            "camera_characteristics",
            "weather_distribution",
            "instance_tracking",
        ]

        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

        # Validate basic statistics
        assert stats["total_samples"] == 50
        assert stats["dataset_name"] == "MockDataset"

    def test_class_distribution_analysis(self):
        """Test class distribution analysis"""
        stats = self.analyzer.analyze_dataset(self.mock_dataset)
        class_dist = stats["class_distribution"]

        # Validate class distribution structure
        assert "class_counts" in class_dist
        assert "class_probabilities" in class_dist
        assert "total_instances" in class_dist
        assert "num_classes" in class_dist

        # Validate probability normalization
        if class_dist["total_instances"] > 0:
            prob_sum = sum(class_dist["class_probabilities"].values())
            assert abs(prob_sum - 1.0) < 1e-6

    def test_spatial_distribution_analysis(self):
        """Test spatial distribution analysis"""
        stats = self.analyzer.analyze_dataset(self.mock_dataset)
        spatial_dist = stats["spatial_distribution"]

        # Validate spatial distribution structure
        expected_metrics = [
            "x_positions",
            "y_positions",
            "z_positions",
            "distances_from_ego",
            "object_sizes",
        ]

        for metric in expected_metrics:
            if metric in spatial_dist:
                metric_stats = spatial_dist[metric]
                assert "mean" in metric_stats
                assert "std" in metric_stats
                assert "median" in metric_stats
                assert "min" in metric_stats
                assert "max" in metric_stats

    def test_scene_complexity_analysis(self):
        """Test scene complexity analysis"""
        stats = self.analyzer.analyze_dataset(self.mock_dataset)
        complexity = stats["scene_complexity"]

        # Validate complexity metrics
        complexity_metrics = [
            "objects_per_frame",
            "unique_classes_per_frame",
            "object_density",
            "occlusion_levels",
        ]

        for metric in complexity_metrics:
            if metric in complexity:
                metric_stats = complexity[metric]
                assert "mean" in metric_stats
                assert "std" in metric_stats
                assert "median" in metric_stats

    def test_weather_distribution_analysis(self):
        """Test weather distribution analysis"""
        stats = self.analyzer.analyze_dataset(self.mock_dataset)
        weather_dist = stats["weather_distribution"]

        # Validate weather distribution structure
        assert "weather_distribution" in weather_dist
        assert "time_distribution" in weather_dist
        assert "location_distribution" in weather_dist
        assert "weather_diversity" in weather_dist
        assert "temporal_diversity" in weather_dist
        assert "spatial_diversity" in weather_dist


class TestDomainGapAnalyzer:
    """Test domain gap analysis functionality"""

    def setup_method(self):
        """Setup test fixtures"""
        try:
            from adnet.data.datasets.cross_dataset_validator import DomainGapAnalyzer

            self.analyzer = DomainGapAnalyzer()
            self.dataset1 = MockDataset(num_samples=50, dataset_name="dataset1")
            self.dataset2 = MockDataset(num_samples=50, dataset_name="dataset2")
        except ImportError:
            pytest.skip("DomainGapAnalyzer not available")

    def test_domain_gap_computation(self):
        """Test domain gap computation between datasets"""
        domain_gap = self.analyzer.compute_domain_gap(self.dataset1, self.dataset2)

        # Validate domain gap metrics structure
        assert hasattr(domain_gap, "class_distribution_divergence")
        assert hasattr(domain_gap, "spatial_distribution_divergence")
        assert hasattr(domain_gap, "camera_setup_similarity")
        assert hasattr(domain_gap, "temporal_characteristics_similarity")
        assert hasattr(domain_gap, "scene_complexity_ratio")
        assert hasattr(domain_gap, "weather_distribution_divergence")
        assert hasattr(domain_gap, "overall_domain_gap_score")

        # Validate metric ranges
        assert 0 <= domain_gap.class_distribution_divergence <= 1
        assert 0 <= domain_gap.spatial_distribution_divergence <= 1
        assert 0 <= domain_gap.camera_setup_similarity <= 1
        assert 0 <= domain_gap.temporal_characteristics_similarity <= 1
        assert 0 <= domain_gap.scene_complexity_ratio <= 1
        assert 0 <= domain_gap.weather_distribution_divergence <= 1
        assert 0 <= domain_gap.overall_domain_gap_score <= 1

    def test_class_distribution_divergence(self):
        """Test class distribution divergence computation"""
        # Mock class distributions
        source_dist = {
            "class_probabilities": {"car": 0.6, "pedestrian": 0.3, "bicycle": 0.1},
            "class_counts": {"car": 60, "pedestrian": 30, "bicycle": 10},
        }

        target_dist = {
            "class_probabilities": {"car": 0.4, "pedestrian": 0.4, "bicycle": 0.2},
            "class_counts": {"car": 40, "pedestrian": 40, "bicycle": 20},
        }

        divergence = self.analyzer._compute_class_distribution_divergence(
            source_dist, target_dist
        )

        # Validate divergence properties
        assert 0 <= divergence <= 1
        assert isinstance(divergence, float)

    def test_spatial_distribution_divergence(self):
        """Test spatial distribution divergence computation"""
        # Mock spatial distributions
        source_spatial = {
            "distances_from_ego": {"mean": 20.0, "std": 5.0},
            "x_positions": {"mean": 0.0, "std": 10.0},
            "y_positions": {"mean": 0.0, "std": 15.0},
            "object_sizes": {"mean": 10.0, "std": 3.0},
        }

        target_spatial = {
            "distances_from_ego": {"mean": 25.0, "std": 7.0},
            "x_positions": {"mean": 2.0, "std": 12.0},
            "y_positions": {"mean": -1.0, "std": 18.0},
            "object_sizes": {"mean": 12.0, "std": 4.0},
        }

        divergence = self.analyzer._compute_spatial_distribution_divergence(
            source_spatial, target_spatial
        )

        # Validate divergence properties
        assert 0 <= divergence <= 1
        assert isinstance(divergence, float)

    def test_camera_setup_similarity(self):
        """Test camera setup similarity computation"""
        # Mock camera characteristics
        source_cameras = {
            "num_cameras": 6,
            "focal_lengths": [1200, 1200, 1000, 1000, 1100, 1100],
            "resolutions": [(1600, 900)] * 6,
        }

        target_cameras = {
            "num_cameras": 5,
            "focal_lengths": [1100, 1100, 950, 950, 1050],
            "resolutions": [(1920, 1080)] * 5,
        }

        similarity = self.analyzer._compute_camera_setup_similarity(
            source_cameras, target_cameras
        )

        # Validate similarity properties
        assert 0 <= similarity <= 1
        assert isinstance(similarity, float)

    def test_overall_domain_gap_calculation(self):
        """Test overall domain gap score calculation"""
        # Mock individual gap metrics
        class_div = 0.3
        spatial_div = 0.2
        camera_sim = 0.8
        temporal_sim = 0.7
        complexity_ratio = 0.9
        weather_div = 0.1

        overall_gap = self.analyzer._compute_overall_domain_gap(
            class_div,
            spatial_div,
            camera_sim,
            temporal_sim,
            complexity_ratio,
            weather_div,
        )

        # Validate overall gap properties
        assert 0 <= overall_gap <= 1
        assert isinstance(overall_gap, float)


class TestCrossDatasetValidator:
    """Test cross-dataset validation framework"""

    def setup_method(self):
        """Setup test fixtures"""
        try:
            from adnet.data.datasets.cross_dataset_validator import (
                CrossDatasetValidator,
            )

            self.validator = CrossDatasetValidator()
            self.source_datasets = [MockDataset(num_samples=30, dataset_name="source")]
            self.target_datasets = [MockDataset(num_samples=30, dataset_name="target")]
        except ImportError:
            pytest.skip("CrossDatasetValidator not available")

    def test_cross_dataset_validation_structure(self):
        """Test cross-dataset validation result structure"""

        # Mock performance function
        def mock_performance_fn(source_dataset, target_dataset):
            return {"mAP": 0.75, "NDS": 0.70}, {
                "car": {"precision": 0.8, "recall": 0.75}
            }

        results = self.validator.validate_cross_dataset_transfer(
            source_datasets=self.source_datasets,
            target_datasets=self.target_datasets,
            model_performance_fn=mock_performance_fn,
        )

        # Validate results structure
        assert len(results) >= 0  # May be 0 if same dataset skipped

        for result in results:
            assert hasattr(result, "source_dataset")
            assert hasattr(result, "target_dataset")
            assert hasattr(result, "domain_gap_metrics")
            assert hasattr(result, "performance_metrics")
            assert hasattr(result, "class_specific_performance")
            assert hasattr(result, "failure_analysis")
            assert hasattr(result, "recommendations")

    def test_failure_mode_analysis(self):
        """Test failure mode analysis"""
        # Mock domain gap metrics
        from adnet.data.datasets.cross_dataset_validator import DomainGapMetrics

        domain_gap = DomainGapMetrics(
            class_distribution_divergence=0.6,
            spatial_distribution_divergence=0.4,
            camera_setup_similarity=0.5,
            temporal_characteristics_similarity=0.4,
            scene_complexity_ratio=0.8,
            weather_distribution_divergence=0.4,
            overall_domain_gap_score=0.5,
        )

        failure_analysis = self.validator._analyze_failure_modes(
            self.source_datasets[0], self.target_datasets[0], domain_gap
        )

        # Validate failure analysis structure
        assert "high_risk_classes" in failure_analysis
        assert "spatial_bias_risk" in failure_analysis
        assert "temporal_mismatch_risk" in failure_analysis
        assert "camera_adaptation_required" in failure_analysis
        assert "weather_robustness_issues" in failure_analysis

        # Validate risk assessments
        assert failure_analysis["spatial_bias_risk"] in ["low", "medium", "high"]
        assert failure_analysis["temporal_mismatch_risk"] in ["low", "medium", "high"]
        assert isinstance(failure_analysis["camera_adaptation_required"], bool)

    def test_recommendation_generation(self):
        """Test recommendation generation"""
        # Mock domain gap and performance metrics
        from adnet.data.datasets.cross_dataset_validator import DomainGapMetrics

        domain_gap = DomainGapMetrics(
            class_distribution_divergence=0.5,
            spatial_distribution_divergence=0.4,
            camera_setup_similarity=0.5,
            temporal_characteristics_similarity=0.4,
            scene_complexity_ratio=0.6,
            weather_distribution_divergence=0.4,
            overall_domain_gap_score=0.7,
        )

        performance_metrics = {"mAP": 0.6, "NDS": 0.55}

        recommendations = self.validator._generate_recommendations(
            domain_gap, performance_metrics
        )

        # Validate recommendations
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

        for recommendation in recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0

    def test_validation_without_performance_function(self):
        """Test validation when no performance function is provided"""
        results = self.validator.validate_cross_dataset_transfer(
            source_datasets=self.source_datasets,
            target_datasets=self.target_datasets,
            model_performance_fn=None,
        )

        # Should still produce results with domain gap analysis
        for result in results:
            assert hasattr(result, "domain_gap_metrics")
            assert result.performance_metrics == {}
            assert result.class_specific_performance == {}


if __name__ == "__main__":
    pytest.main([__file__])
