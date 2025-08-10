"""
Tests for Load Testing and Performance Regression Detection

This module contains comprehensive tests for the load testing suite and
performance regression detection system, validating 50+ concurrent users,
stress testing, scalability testing, and regression detection capabilities.
"""

import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch

from ..load_test_suite import (
    LoadTestSuite, ScalabilityTestResult, StressTestResult, 
    EnduranceTestResult, run_load_tests
)
from ..performance_regression_detector import (
    PerformanceRegressionDetector, RegressionAlert, 
    RegressionAnalysisResult, analyze_performance_regression
)
from ..performance_benchmark import PerformanceBenchmark, LoadTestResult
from ...component import LightRAGComponent
from ...config.settings import LightRAGConfig


class MockLightRAGComponent:
    """Mock LightRAG component for testing."""
    
    def __init__(self, response_time_base: float = 1.0, 
                 failure_rate: float = 0.0, memory_usage: int = 100):
        self.response_time_base = response_time_base
        self.failure_rate = failure_rate
        self.memory_usage = memory_usage
        self.query_count = 0
        self.concurrent_queries = 0
        self.max_concurrent = 0
    
    async def initialize(self):
        """Initialize mock component."""
        pass
    
    async def cleanup(self):
        """Cleanup mock component."""
        pass
    
    async def query(self, question: str, context: Dict = None):
        """Mock query with configurable response time and failure rate."""
        self.query_count += 1
        self.concurrent_queries += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent_queries)
        
        try:
            # Simulate load-dependent response time
            load_factor = min(2.0, self.concurrent_queries / 10.0)
            response_time = self.response_time_base * (1 + load_factor)
            
            await asyncio.sleep(response_time)
            
            # Simulate failures
            if self.failure_rate > 0 and (self.query_count % int(1/self.failure_rate)) == 0:
                raise Exception("Simulated query failure")
            
            return {
                "answer": f"Mock response to: {question}",
                "confidence_score": 0.85,
                "source_documents": ["mock_doc.pdf"],
                "processing_time": response_time
            }
        finally:
            self.concurrent_queries -= 1


@pytest.fixture
def mock_config():
    """Create mock LightRAG configuration."""
    return LightRAGConfig(
        knowledge_graph_path="test_kg",
        vector_store_path="test_vectors",
        cache_directory="test_cache"
    )


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestLoadTestSuite:
    """Test cases for the load testing suite."""
    
    def test_load_test_suite_initialization(self, mock_config, temp_output_dir):
        """Test load test suite initialization."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        
        assert suite.config == mock_config
        assert suite.output_dir == Path(temp_output_dir)
        assert suite.logger is not None
        assert suite.performance_benchmark is not None
        assert suite.memory_monitor is not None
        
        # Check thresholds
        assert "max_response_time_50_users" in suite.load_test_thresholds
        assert "min_success_rate" in suite.load_test_thresholds
        assert suite.load_test_thresholds["min_success_rate"] == 0.95
        
        # Check test questions
        assert "basic" in suite.test_questions
        assert "complex" in suite.test_questions
        assert "mixed" in suite.test_questions
        assert len(suite.test_questions["mixed"]) >= 10
    
    @pytest.mark.asyncio
    async def test_concurrent_user_test_basic(self, mock_config, temp_output_dir):
        """Test basic concurrent user testing."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        mock_component = MockLightRAGComponent(response_time_base=0.1)
        
        result = await suite.run_concurrent_user_test(
            component=mock_component,
            concurrent_users=5,
            requests_per_user=3,
            question_set="basic"
        )
        
        assert isinstance(result, LoadTestResult)
        assert result.concurrent_users == 5
        assert result.total_requests == 15  # 5 users * 3 requests
        assert result.successful_requests > 0
        assert result.failed_requests == 0
        assert result.average_response_time > 0
        assert result.requests_per_second > 0
        assert mock_component.query_count == 15
    
    @pytest.mark.asyncio
    async def test_concurrent_user_test_50_users(self, mock_config, temp_output_dir):
        """Test 50+ concurrent users requirement."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        mock_component = MockLightRAGComponent(response_time_base=0.05)
        
        result = await suite.run_concurrent_user_test(
            component=mock_component,
            concurrent_users=50,
            requests_per_user=2,
            question_set="mixed"
        )
        
        assert isinstance(result, LoadTestResult)
        assert result.concurrent_users == 50
        assert result.total_requests == 100  # 50 users * 2 requests
        assert result.successful_requests > 0
        assert result.average_response_time > 0
        
        # Verify 50+ concurrent users were actually handled
        assert mock_component.max_concurrent >= 50
        
        # Check performance thresholds
        success_rate = result.successful_requests / result.total_requests
        assert success_rate >= 0.90  # Allow some tolerance for mock testing
    
    @pytest.mark.asyncio
    async def test_concurrent_user_test_with_failures(self, mock_config, temp_output_dir):
        """Test concurrent user testing with simulated failures."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        mock_component = MockLightRAGComponent(
            response_time_base=0.1, 
            failure_rate=0.1  # 10% failure rate
        )
        
        result = await suite.run_concurrent_user_test(
            component=mock_component,
            concurrent_users=10,
            requests_per_user=5,
            question_set="basic"
        )
        
        assert result.total_requests == 50
        assert result.failed_requests > 0
        assert result.successful_requests > 0
        assert len(result.error_details) > 0
        
        success_rate = result.successful_requests / result.total_requests
        assert success_rate < 1.0  # Should have some failures
    
    @pytest.mark.asyncio
    async def test_duration_based_test(self, mock_config, temp_output_dir):
        """Test duration-based load testing."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        mock_component = MockLightRAGComponent(response_time_base=0.05)
        
        # Run for 0.1 minutes (6 seconds) to keep test fast
        result = await suite.run_concurrent_user_test(
            component=mock_component,
            concurrent_users=10,
            test_duration_minutes=0.1,
            question_set="basic"
        )
        
        assert isinstance(result, LoadTestResult)
        assert result.concurrent_users == 10
        assert result.duration_seconds >= 5  # Should run for at least 5 seconds
        assert result.total_requests > 0
        assert result.requests_per_second > 0
    
    @pytest.mark.asyncio
    async def test_scalability_test(self, mock_config, temp_output_dir):
        """Test scalability testing functionality."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        mock_component = MockLightRAGComponent(response_time_base=0.02)
        
        result = await suite.run_scalability_test(
            component=mock_component,
            max_users=30,  # Keep small for test speed
            step_size=10,
            requests_per_user=2
        )
        
        assert isinstance(result, ScalabilityTestResult)
        assert result.user_levels == [10, 20, 30]
        assert len(result.results_by_level) > 0
        assert result.scalability_metrics is not None
        assert result.recommendations is not None
        
        # Check that results exist for each user level
        for user_level in result.user_levels:
            if user_level in result.results_by_level:
                load_result = result.results_by_level[user_level]
                assert load_result.concurrent_users == user_level
    
    @pytest.mark.asyncio
    async def test_scalability_test_with_breaking_point(self, mock_config, temp_output_dir):
        """Test scalability testing with performance degradation."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        
        # Create component that degrades with high load
        mock_component = MockLightRAGComponent(response_time_base=0.1)
        
        # Override query method to simulate degradation
        original_query = mock_component.query
        async def degrading_query(question: str, context: Dict = None):
            # Simulate severe degradation at high concurrency
            if mock_component.concurrent_queries > 15:
                mock_component.response_time_base = 2.0  # Very slow
            return await original_query(question, context)
        
        mock_component.query = degrading_query
        
        result = await suite.run_scalability_test(
            component=mock_component,
            max_users=25,
            step_size=5,
            requests_per_user=2
        )
        
        assert isinstance(result, ScalabilityTestResult)
        assert result.breaking_point is not None
        assert result.breaking_point <= 25
        assert "performance degraded" in " ".join(result.recommendations).lower()
    
    @pytest.mark.asyncio
    async def test_stress_test(self, mock_config, temp_output_dir):
        """Test stress testing functionality."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        mock_component = MockLightRAGComponent(response_time_base=0.02)
        
        result = await suite.run_stress_test(
            component=mock_component,
            duration_minutes=0.2,  # 12 seconds for test speed
            max_concurrent_users=20,
            ramp_up_minutes=0.1  # 6 seconds ramp-up
        )
        
        assert isinstance(result, StressTestResult)
        assert result.duration_minutes == 0.2
        assert result.max_concurrent_users == 20
        assert len(result.load_progression) > 0
        assert result.system_stability is not None
        assert result.recovery_metrics is not None
        
        # Check that load progression shows increasing users
        user_counts = [r.concurrent_users for r in result.load_progression]
        assert max(user_counts) == 20
    
    @pytest.mark.asyncio
    async def test_endurance_test(self, mock_config, temp_output_dir):
        """Test endurance testing functionality."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        mock_component = MockLightRAGComponent(response_time_base=0.02)
        
        result = await suite.run_endurance_test(
            component=mock_component,
            duration_hours=0.05,  # 3 minutes for test speed
            constant_load_users=5,
            sampling_interval_minutes=0.02  # 1.2 seconds
        )
        
        assert isinstance(result, EnduranceTestResult)
        assert result.duration_hours == 0.05
        assert result.constant_load_users == 5
        assert len(result.performance_over_time) > 0
        assert result.memory_leak_analysis is not None
        assert result.performance_degradation is not None
        assert 0 <= result.stability_score <= 1
    
    @pytest.mark.asyncio
    async def test_comprehensive_load_tests(self, mock_config, temp_output_dir):
        """Test comprehensive load testing suite."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        mock_component = MockLightRAGComponent(response_time_base=0.02)
        
        results = await suite.run_comprehensive_load_tests(mock_component)
        
        assert isinstance(results, dict)
        assert "timestamp" in results
        assert "concurrent_user_tests" in results
        assert "scalability_test" in results
        assert "stress_test" in results
        assert "endurance_test" in results
        assert "summary" in results
        assert "recommendations" in results
        
        # Check concurrent user tests
        assert len(results["concurrent_user_tests"]) > 0
        
        # Check summary
        summary = results["summary"]
        assert "max_stable_users" in summary
        assert "peak_throughput" in summary
        assert "system_scalability" in summary
    
    def test_load_test_report_generation(self, mock_config, temp_output_dir):
        """Test load test report generation."""
        suite = LoadTestSuite(config=mock_config, output_dir=temp_output_dir)
        
        # Create mock results
        mock_results = {
            "timestamp": datetime.now().isoformat(),
            "concurrent_user_tests": [
                {
                    "user_count": 50,
                    "result": {
                        "concurrent_users": 50,
                        "total_requests": 100,
                        "successful_requests": 95,
                        "failed_requests": 5,
                        "average_response_time": 2.5,
                        "requests_per_second": 20.0,
                        "percentile_95_response_time": 4.0
                    }
                }
            ],
            "summary": {
                "max_stable_users": 50,
                "peak_throughput": 20.0,
                "system_scalability": "good",
                "stress_test_passed": True,
                "endurance_test_passed": True
            },
            "recommendations": [
                "System handles 50+ concurrent users well",
                "Consider optimizing for higher loads"
            ]
        }
        
        report = suite.generate_load_test_report(mock_results)
        
        assert "LIGHTRAG LOAD TESTING REPORT" in report
        assert "Max Stable Users: 50" in report
        assert "Peak Throughput: 20.00 req/s" in report
        assert "50 Users:" in report
        assert "Success Rate: 95.0%" in report
        assert "RECOMMENDATIONS:" in report


class TestPerformanceRegressionDetector:
    """Test cases for performance regression detection."""
    
    def test_regression_detector_initialization(self, temp_output_dir):
        """Test regression detector initialization."""
        detector = PerformanceRegressionDetector(output_dir=temp_output_dir)
        
        assert detector.output_dir == Path(temp_output_dir)
        assert detector.logger is not None
        assert detector.baselines == {}  # No baselines initially
        assert "minor" in detector.regression_thresholds
        assert "response_time" in detector.metric_configs
    
    def test_baseline_creation_and_update(self, temp_output_dir):
        """Test baseline creation and updates."""
        detector = PerformanceRegressionDetector(output_dir=temp_output_dir)
        
        # Create new baseline
        detector.update_baseline("response_time", 2.5)
        
        assert "response_time" in detector.baselines
        baseline = detector.baselines["response_time"]
        assert baseline.baseline_value == 2.5
        assert baseline.measurement_count == 1
        assert len(baseline.historical_values) == 1
        
        # Update with better performance (should update baseline)
        detector.update_baseline("response_time", 2.0)
        
        baseline = detector.baselines["response_time"]
        assert baseline.baseline_value == 2.0  # Should update to better value
        assert baseline.measurement_count == 2
        assert len(baseline.historical_values) == 2
        
        # Update with worse performance (should not update baseline)
        detector.update_baseline("response_time", 3.0)
        
        baseline = detector.baselines["response_time"]
        assert baseline.baseline_value == 2.0  # Should not update
        assert baseline.measurement_count == 3
        assert len(baseline.historical_values) == 3
    
    def test_regression_detection_no_regression(self, temp_output_dir):
        """Test regression detection with no regression."""
        detector = PerformanceRegressionDetector(output_dir=temp_output_dir)
        
        # Set baseline
        detector.update_baseline("response_time", 2.0)
        detector.update_baseline("memory_usage", 1000.0)
        detector.update_baseline("throughput", 50.0)
        
        # Test with similar or better performance
        current_metrics = {
            "response_time": 2.1,  # Slightly worse but within tolerance
            "memory_usage": 950.0,  # Better
            "throughput": 52.0  # Better
        }
        
        result = detector.analyze_performance_metrics(current_metrics)
        
        assert isinstance(result, RegressionAnalysisResult)
        assert result.total_metrics_analyzed == 3
        assert result.regressions_detected == 0
        assert len(result.alerts) == 0
        assert result.overall_performance_trend in ["stable", "improving"]
    
    def test_regression_detection_with_regressions(self, temp_output_dir):
        """Test regression detection with actual regressions."""
        detector = PerformanceRegressionDetector(output_dir=temp_output_dir)
        
        # Set baselines
        detector.update_baseline("response_time", 2.0)
        detector.update_baseline("memory_usage", 1000.0)
        detector.update_baseline("throughput", 50.0)
        detector.update_baseline("success_rate", 0.98)
        
        # Test with degraded performance
        current_metrics = {
            "response_time": 3.0,  # 50% worse
            "memory_usage": 1500.0,  # 50% worse
            "throughput": 30.0,  # 40% worse
            "success_rate": 0.85  # 13% worse
        }
        
        result = detector.analyze_performance_metrics(current_metrics)
        
        assert result.total_metrics_analyzed == 4
        assert result.regressions_detected > 0
        assert len(result.alerts) > 0
        assert result.overall_performance_trend == "degrading"
        
        # Check alert details
        response_time_alerts = [a for a in result.alerts if a.metric_name == "response_time"]
        assert len(response_time_alerts) > 0
        
        alert = response_time_alerts[0]
        assert alert.current_value == 3.0
        assert alert.baseline_value == 2.0
        assert alert.regression_percentage == 50.0
        assert alert.severity in ["moderate", "severe"]
        assert len(alert.recommended_actions) > 0
    
    def test_severity_classification(self, temp_output_dir):
        """Test regression severity classification."""
        detector = PerformanceRegressionDetector(output_dir=temp_output_dir)
        
        # Set baseline
        detector.update_baseline("response_time", 1.0)
        
        test_cases = [
            (1.15, "minor"),    # 15% regression
            (1.30, "moderate"), # 30% regression
            (1.60, "severe"),   # 60% regression
            (2.50, "critical")  # 150% regression
        ]
        
        for current_value, expected_severity in test_cases:
            result = detector.analyze_performance_metrics({"response_time": current_value})
            
            if result.regressions_detected > 0:
                alert = result.alerts[0]
                assert alert.severity == expected_severity
    
    def test_baseline_persistence(self, temp_output_dir):
        """Test baseline persistence to file."""
        baseline_file = str(Path(temp_output_dir) / "test_baselines.json")
        
        # Create detector and add baselines
        detector1 = PerformanceRegressionDetector(
            baseline_file=baseline_file,
            output_dir=temp_output_dir
        )
        detector1.update_baseline("response_time", 2.5)
        detector1.update_baseline("memory_usage", 1200.0)
        
        # Create new detector and verify baselines loaded
        detector2 = PerformanceRegressionDetector(
            baseline_file=baseline_file,
            output_dir=temp_output_dir
        )
        
        assert "response_time" in detector2.baselines
        assert "memory_usage" in detector2.baselines
        assert detector2.baselines["response_time"].baseline_value == 2.5
        assert detector2.baselines["memory_usage"].baseline_value == 1200.0
    
    def test_regression_report_generation(self, temp_output_dir):
        """Test regression analysis report generation."""
        detector = PerformanceRegressionDetector(output_dir=temp_output_dir)
        
        # Create mock analysis result with regressions
        alerts = [
            RegressionAlert(
                metric_name="response_time",
                current_value=3.0,
                baseline_value=2.0,
                regression_percentage=50.0,
                severity="severe",
                timestamp=datetime.now(),
                description="Response time increased from 2.0 to 3.0 (50% degradation)",
                recommended_actions=["Check database performance", "Review caching"]
            )
        ]
        
        analysis_result = RegressionAnalysisResult(
            timestamp=datetime.now(),
            total_metrics_analyzed=3,
            regressions_detected=1,
            alerts=alerts,
            overall_performance_trend="degrading",
            summary={
                "regression_rate": 0.33,
                "severity_distribution": {"severe": 1},
                "most_affected_metrics": [
                    {
                        "metric": "response_time",
                        "regression_percentage": 50.0,
                        "severity": "severe"
                    }
                ],
                "average_regression": 50.0
            }
        )
        
        report = detector.generate_regression_report(analysis_result)
        
        assert "PERFORMANCE REGRESSION ANALYSIS REPORT" in report
        assert "Regressions Detected: 1" in report
        assert "Overall Trend: Degrading" in report
        assert "SEVERE REGRESSIONS:" in report
        assert "response_time" in report
        assert "50.0% degradation" in report
        assert "Check database performance" in report


@pytest.mark.asyncio
async def test_integration_load_tests_with_regression_detection(temp_output_dir):
    """Test integration of load testing with regression detection."""
    # Create mock component
    mock_component = MockLightRAGComponent(response_time_base=0.05)
    
    # Run load tests
    load_suite = LoadTestSuite(output_dir=temp_output_dir)
    load_results = await load_suite.run_comprehensive_load_tests(mock_component)
    
    # Extract performance metrics from load test results
    performance_metrics = {}
    
    if load_results.get("concurrent_user_tests"):
        # Use 50-user test results if available
        for test in load_results["concurrent_user_tests"]:
            if test["user_count"] == 50:
                result = test["result"]
                performance_metrics.update({
                    "response_time_50_users": result["average_response_time"],
                    "throughput_50_users": result["requests_per_second"],
                    "success_rate_50_users": result["successful_requests"] / result["total_requests"],
                    "memory_usage_50_users": result.get("memory_usage_mb", {}).get("peak", 0)
                })
                break
    
    # Run regression analysis
    if performance_metrics:
        regression_detector = PerformanceRegressionDetector(output_dir=temp_output_dir)
        
        # Update baselines (first run)
        for metric, value in performance_metrics.items():
            regression_detector.update_baseline(metric, value)
        
        # Simulate degraded performance
        degraded_metrics = {
            metric: value * 1.3 if "response_time" in metric or "memory" in metric else value * 0.8
            for metric, value in performance_metrics.items()
        }
        
        regression_result = regression_detector.analyze_performance_metrics(degraded_metrics)
        
        assert isinstance(regression_result, RegressionAnalysisResult)
        assert regression_result.total_metrics_analyzed > 0


@pytest.mark.asyncio
async def test_large_document_collection_stress_test(temp_output_dir):
    """Test stress testing with large document collections simulation."""
    # Create component that simulates large document collection behavior
    class LargeCollectionMockComponent(MockLightRAGComponent):
        def __init__(self):
            super().__init__(response_time_base=0.1)
            self.document_count = 1000  # Simulate 1000 documents
        
        async def query(self, question: str, context: Dict = None):
            # Simulate increased response time with large collections
            collection_factor = min(2.0, self.document_count / 500.0)
            self.response_time_base = 0.1 * collection_factor
            return await super().query(question, context)
    
    mock_component = LargeCollectionMockComponent()
    load_suite = LoadTestSuite(output_dir=temp_output_dir)
    
    # Run stress test
    result = await load_suite.run_stress_test(
        component=mock_component,
        duration_minutes=0.2,  # Short duration for test
        max_concurrent_users=30,
        ramp_up_minutes=0.1
    )
    
    assert isinstance(result, StressTestResult)
    assert len(result.load_progression) > 0
    
    # Verify system handled the stress
    final_result = result.load_progression[-1]
    assert final_result.concurrent_users > 0
    assert final_result.total_requests > 0


def test_scalability_testing_system_limits(temp_output_dir):
    """Test scalability testing to find system limits."""
    # This test validates the scalability testing approach
    # without actually running expensive tests
    
    load_suite = LoadTestSuite(output_dir=temp_output_dir)
    
    # Test scalability metrics calculation
    mock_results = {
        10: LoadTestResult(
            test_name="test_10", concurrent_users=10, total_requests=20,
            successful_requests=20, failed_requests=0, duration_seconds=10.0,
            requests_per_second=2.0, average_response_time=1.0,
            min_response_time=0.8, max_response_time=1.2,
            percentile_95_response_time=1.1, percentile_99_response_time=1.2,
            memory_usage_mb={"peak": 500}, cpu_usage_percent={"average": 30},
            error_details=[]
        ),
        20: LoadTestResult(
            test_name="test_20", concurrent_users=20, total_requests=40,
            successful_requests=40, failed_requests=0, duration_seconds=10.0,
            requests_per_second=4.0, average_response_time=1.5,
            min_response_time=1.2, max_response_time=2.0,
            percentile_95_response_time=1.8, percentile_99_response_time=2.0,
            memory_usage_mb={"peak": 800}, cpu_usage_percent={"average": 50},
            error_details=[]
        ),
        30: LoadTestResult(
            test_name="test_30", concurrent_users=30, total_requests=60,
            successful_requests=55, failed_requests=5, duration_seconds=10.0,
            requests_per_second=5.5, average_response_time=3.0,
            min_response_time=2.0, max_response_time=5.0,
            percentile_95_response_time=4.5, percentile_99_response_time=5.0,
            memory_usage_mb={"peak": 1200}, cpu_usage_percent={"average": 80},
            error_details=["Timeout error", "Connection error"]
        )
    }
    
    # Test scalability analysis
    scalability_metrics = load_suite._analyze_scalability_metrics(mock_results)
    
    assert "response_time_growth_rate" in scalability_metrics
    assert "throughput_efficiency" in scalability_metrics
    assert "max_stable_users" in scalability_metrics
    assert "memory_per_user" in scalability_metrics
    
    # Verify max stable users detection
    max_stable = scalability_metrics["max_stable_users"]
    assert max_stable == 20  # 30 users had failures and high response time
    
    # Test recommendations generation
    recommendations = load_suite._generate_scalability_recommendations(
        mock_results, breaking_point=30, scalability_metrics=scalability_metrics
    )
    
    assert len(recommendations) > 0
    assert any("performance degrades" in rec.lower() for rec in recommendations)


if __name__ == "__main__":
    # Run specific test categories
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "load_only":
        # Run only load testing tests
        pytest.main([__file__ + "::TestLoadTestSuite", "-v"])
    elif len(sys.argv) > 1 and sys.argv[1] == "regression_only":
        # Run only regression detection tests
        pytest.main([__file__ + "::TestPerformanceRegressionDetector", "-v"])
    else:
        # Run all tests
        pytest.main([__file__, "-v"])