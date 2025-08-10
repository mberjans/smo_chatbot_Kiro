"""
Integration Tests for Performance and Load Testing System

This module provides comprehensive integration tests that validate the complete
performance and load testing workflow, including 50+ concurrent users,
stress testing, scalability analysis, and regression detection.
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

from ..load_test_suite import LoadTestSuite, run_load_tests
from ..performance_regression_detector import PerformanceRegressionDetector
from ..performance_benchmark import PerformanceBenchmark
from ..automated_test_runner import AutomatedTestRunner
from ...component import LightRAGComponent
from ...config.settings import LightRAGConfig


class PerformanceTestComponent:
    """
    Realistic test component that simulates actual LightRAG behavior
    with configurable performance characteristics.
    """
    
    def __init__(self, base_response_time: float = 0.5,
                 memory_growth_rate: float = 0.1,
                 failure_threshold: int = 100):
        self.base_response_time = base_response_time
        self.memory_growth_rate = memory_growth_rate
        self.failure_threshold = failure_threshold
        
        self.query_count = 0
        self.concurrent_queries = 0
        self.max_concurrent_reached = 0
        self.total_memory_usage = 100  # MB
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the test component."""
        await asyncio.sleep(0.1)  # Simulate initialization time
        self.is_initialized = True
    
    async def cleanup(self):
        """Cleanup the test component."""
        await asyncio.sleep(0.05)  # Simulate cleanup time
        self.is_initialized = False
    
    async def query(self, question: str, context: Dict = None):
        """
        Simulate realistic query processing with performance characteristics.
        """
        if not self.is_initialized:
            raise RuntimeError("Component not initialized")
        
        self.query_count += 1
        self.concurrent_queries += 1
        self.max_concurrent_reached = max(self.max_concurrent_reached, self.concurrent_queries)
        
        try:
            # Simulate memory growth over time
            self.total_memory_usage += self.memory_growth_rate
            
            # Calculate response time based on load and system state
            load_factor = min(3.0, self.concurrent_queries / 20.0)  # Degrades after 20 concurrent
            memory_factor = min(2.0, self.total_memory_usage / 1000.0)  # Degrades after 1GB
            
            response_time = self.base_response_time * (1 + load_factor + memory_factor * 0.5)
            
            # Simulate processing time
            await asyncio.sleep(response_time)
            
            # Simulate failures under extreme load
            if self.concurrent_queries > self.failure_threshold:
                if self.query_count % 10 == 0:  # 10% failure rate under extreme load
                    raise Exception(f"System overloaded: {self.concurrent_queries} concurrent queries")
            
            # Simulate realistic response
            return {
                "answer": f"Clinical metabolomics response for: {question[:50]}...",
                "confidence_score": max(0.5, 0.95 - (load_factor * 0.1)),
                "source_documents": [f"doc_{self.query_count % 10}.pdf"],
                "entities_used": [{"text": "metabolite", "type": "compound"}],
                "processing_time": response_time,
                "memory_usage_mb": self.total_memory_usage
            }
            
        finally:
            self.concurrent_queries -= 1
    
    async def get_health_status(self):
        """Get component health status."""
        return {
            "status": "healthy" if self.concurrent_queries < self.failure_threshold else "degraded",
            "query_count": self.query_count,
            "concurrent_queries": self.concurrent_queries,
            "memory_usage_mb": self.total_memory_usage,
            "max_concurrent_reached": self.max_concurrent_reached
        }


@pytest.fixture
def performance_config():
    """Create configuration for performance testing."""
    return LightRAGConfig(
        knowledge_graph_path="test_performance_kg",
        vector_store_path="test_performance_vectors",
        cache_directory="test_performance_cache",
        batch_size=16,
        max_concurrent_requests=50
    )


@pytest.fixture
def temp_results_dir():
    """Create temporary directory for test results."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestPerformanceIntegration:
    """Integration tests for performance and load testing system."""
    
    @pytest.mark.asyncio
    async def test_50_plus_concurrent_users_requirement(self, performance_config, temp_results_dir):
        """
        Test that the system can handle 50+ concurrent users as required.
        This validates requirement 8.5 for load testing.
        """
        # Create performance test component
        component = PerformanceTestComponent(
            base_response_time=0.1,
            memory_growth_rate=0.05,
            failure_threshold=75  # Should handle 50+ users before failures
        )
        await component.initialize()
        
        try:
            # Create load test suite
            load_suite = LoadTestSuite(config=performance_config, output_dir=temp_results_dir)
            
            # Test 50 concurrent users
            result_50 = await load_suite.run_concurrent_user_test(
                component=component,
                concurrent_users=50,
                requests_per_user=3,
                question_set="mixed"
            )
            
            # Validate 50-user performance
            assert result_50.concurrent_users == 50
            assert result_50.total_requests == 150
            success_rate_50 = result_50.successful_requests / result_50.total_requests
            
            # Should handle 50 users with high success rate
            assert success_rate_50 >= 0.90, f"50-user success rate too low: {success_rate_50:.2%}"
            assert result_50.average_response_time < 15.0, f"50-user response time too high: {result_50.average_response_time:.2f}s"
            
            # Test 75 concurrent users (stress test)
            result_75 = await load_suite.run_concurrent_user_test(
                component=component,
                concurrent_users=75,
                requests_per_user=2,
                question_set="basic"
            )
            
            # Validate that system degrades gracefully beyond 50 users
            assert result_75.concurrent_users == 75
            success_rate_75 = result_75.successful_requests / result_75.total_requests
            
            # System should still function but may show degradation
            assert success_rate_75 >= 0.70, f"75-user success rate too low: {success_rate_75:.2%}"
            
            # Verify actual concurrent load was achieved
            health_status = await component.get_health_status()
            assert health_status["max_concurrent_reached"] >= 50, "Did not achieve 50+ concurrent users"
            
            print(f"✅ 50-user test: {success_rate_50:.1%} success, {result_50.average_response_time:.2f}s avg response")
            print(f"✅ 75-user test: {success_rate_75:.1%} success, {result_75.average_response_time:.2f}s avg response")
            print(f"✅ Max concurrent reached: {health_status['max_concurrent_reached']}")
            
        finally:
            await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_stress_testing_large_document_collections(self, performance_config, temp_results_dir):
        """
        Test stress testing with large document collections simulation.
        This validates requirement 8.5 for stress testing.
        """
        # Create component that simulates large document collection behavior
        component = PerformanceTestComponent(
            base_response_time=0.2,  # Slower due to large collection
            memory_growth_rate=0.2,  # Higher memory growth
            failure_threshold=60
        )
        await component.initialize()
        
        try:
            load_suite = LoadTestSuite(config=performance_config, output_dir=temp_results_dir)
            
            # Run stress test
            stress_result = await load_suite.run_stress_test(
                component=component,
                duration_minutes=0.5,  # 30 seconds for test speed
                max_concurrent_users=60,
                ramp_up_minutes=0.2  # 12 seconds ramp-up
            )
            
            # Validate stress test results
            assert isinstance(stress_result.load_progression, list)
            assert len(stress_result.load_progression) > 0
            assert stress_result.max_concurrent_users == 60
            
            # Check that system handled progressive load increase
            user_counts = [r.concurrent_users for r in stress_result.load_progression]
            assert max(user_counts) == 60
            assert min(user_counts) < 60  # Should show progression
            
            # Validate system stability metrics
            assert "stable_periods" in stress_result.system_stability
            assert "unstable_periods" in stress_result.system_stability
            
            # Check recovery metrics
            assert stress_result.recovery_metrics is not None
            assert "recovery_successful" in stress_result.recovery_metrics
            
            print(f"✅ Stress test completed: {len(stress_result.load_progression)} phases")
            print(f"✅ System stability: {stress_result.system_stability}")
            print(f"✅ Recovery: {stress_result.recovery_metrics['recovery_successful']}")
            
        finally:
            await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_scalability_testing_system_limits(self, performance_config, temp_results_dir):
        """
        Test scalability testing to find system limits.
        This validates requirement 8.7 for scalability testing.
        """
        # Create component with clear scalability limits
        component = PerformanceTestComponent(
            base_response_time=0.1,
            memory_growth_rate=0.1,
            failure_threshold=45  # Clear limit at 45 users
        )
        await component.initialize()
        
        try:
            load_suite = LoadTestSuite(config=performance_config, output_dir=temp_results_dir)
            
            # Run scalability test
            scalability_result = await load_suite.run_scalability_test(
                component=component,
                max_users=60,
                step_size=15,  # Test at 15, 30, 45, 60 users
                requests_per_user=2
            )
            
            # Validate scalability results
            assert scalability_result.user_levels == [15, 30, 45, 60]
            assert len(scalability_result.results_by_level) > 0
            
            # Should detect breaking point around 45 users
            assert scalability_result.breaking_point is not None
            assert 40 <= scalability_result.breaking_point <= 60
            
            # Validate scalability metrics
            metrics = scalability_result.scalability_metrics
            assert "response_time_growth_rate" in metrics
            assert "throughput_efficiency" in metrics
            assert "max_stable_users" in metrics
            assert "memory_per_user" in metrics
            
            # Check recommendations
            assert len(scalability_result.recommendations) > 0
            recommendations_text = " ".join(scalability_result.recommendations).lower()
            assert "performance" in recommendations_text or "scaling" in recommendations_text
            
            print(f"✅ Scalability test: Breaking point at {scalability_result.breaking_point} users")
            print(f"✅ Max stable users: {metrics['max_stable_users']}")
            print(f"✅ Throughput efficiency: {metrics['throughput_efficiency']:.2f}")
            
        finally:
            await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self, performance_config, temp_results_dir):
        """
        Test performance regression detection system.
        This validates requirement 8.3 for performance regression detection.
        """
        # Create baseline performance component
        baseline_component = PerformanceTestComponent(
            base_response_time=0.1,
            memory_growth_rate=0.05,
            failure_threshold=100
        )
        await baseline_component.initialize()
        
        try:
            load_suite = LoadTestSuite(config=performance_config, output_dir=temp_results_dir)
            regression_detector = PerformanceRegressionDetector(output_dir=temp_results_dir)
            
            # Establish baseline performance
            baseline_result = await load_suite.run_concurrent_user_test(
                component=baseline_component,
                concurrent_users=25,
                requests_per_user=4,
                question_set="mixed"
            )
            
            # Extract baseline metrics
            baseline_metrics = {
                "response_time": baseline_result.average_response_time,
                "throughput": baseline_result.requests_per_second,
                "success_rate": baseline_result.successful_requests / baseline_result.total_requests,
                "memory_usage": baseline_result.memory_usage_mb.get("peak", 0)
            }
            
            # Update baselines
            for metric, value in baseline_metrics.items():
                regression_detector.update_baseline(metric, value)
            
            print(f"✅ Baseline established: {baseline_metrics}")
            
            # Create degraded performance component
            degraded_component = PerformanceTestComponent(
                base_response_time=0.2,  # 2x slower
                memory_growth_rate=0.15,  # 3x more memory growth
                failure_threshold=80  # Lower failure threshold
            )
            await degraded_component.initialize()
            
            try:
                # Test degraded performance
                degraded_result = await load_suite.run_concurrent_user_test(
                    component=degraded_component,
                    concurrent_users=25,
                    requests_per_user=4,
                    question_set="mixed"
                )
                
                # Extract current metrics
                current_metrics = {
                    "response_time": degraded_result.average_response_time,
                    "throughput": degraded_result.requests_per_second,
                    "success_rate": degraded_result.successful_requests / degraded_result.total_requests,
                    "memory_usage": degraded_result.memory_usage_mb.get("peak", 0)
                }
                
                # Analyze for regressions
                regression_result = regression_detector.analyze_performance_metrics(current_metrics)
                
                # Validate regression detection
                assert regression_result.total_metrics_analyzed > 0
                assert regression_result.regressions_detected > 0, "Should detect performance regressions"
                assert len(regression_result.alerts) > 0
                assert regression_result.overall_performance_trend == "degrading"
                
                # Check specific regressions
                response_time_alerts = [a for a in regression_result.alerts if a.metric_name == "response_time"]
                assert len(response_time_alerts) > 0, "Should detect response time regression"
                
                response_time_alert = response_time_alerts[0]
                assert response_time_alert.regression_percentage > 20, "Should detect significant regression"
                assert response_time_alert.severity in ["moderate", "severe", "critical"]
                
                # Generate regression report
                report = regression_detector.generate_regression_report(regression_result)
                assert "PERFORMANCE REGRESSION ANALYSIS REPORT" in report
                assert "REGRESSIONS DETECTED" in report.upper()
                
                print(f"✅ Regression detection: {regression_result.regressions_detected} regressions found")
                print(f"✅ Response time regression: {response_time_alert.regression_percentage:.1f}%")
                print(f"✅ Overall trend: {regression_result.overall_performance_trend}")
                
            finally:
                await degraded_component.cleanup()
                
        finally:
            await baseline_component.cleanup()
    
    @pytest.mark.asyncio
    async def test_comprehensive_performance_testing_workflow(self, performance_config, temp_results_dir):
        """
        Test the complete performance testing workflow integration.
        This validates the entire performance testing system working together.
        """
        # Create realistic test component
        component = PerformanceTestComponent(
            base_response_time=0.15,
            memory_growth_rate=0.08,
            failure_threshold=55
        )
        await component.initialize()
        
        try:
            # Run comprehensive load tests
            results = await run_load_tests(
                config=performance_config,
                output_dir=temp_results_dir,
                max_users=60
            )
            
            # Validate comprehensive results structure
            assert "timestamp" in results
            assert "concurrent_user_tests" in results
            assert "scalability_test" in results
            assert "stress_test" in results
            assert "endurance_test" in results
            assert "summary" in results
            assert "recommendations" in results
            
            # Validate concurrent user tests
            concurrent_tests = results["concurrent_user_tests"]
            assert len(concurrent_tests) > 0
            
            # Find 50+ user test
            user_50_plus_test = None
            for test in concurrent_tests:
                if test["user_count"] >= 50:
                    user_50_plus_test = test
                    break
            
            assert user_50_plus_test is not None, "Should include 50+ user test"
            
            # Validate summary metrics
            summary = results["summary"]
            assert "max_stable_users" in summary
            assert "peak_throughput" in summary
            assert "system_scalability" in summary
            
            # Validate recommendations
            recommendations = results["recommendations"]
            assert len(recommendations) > 0
            
            # Generate and validate report
            load_suite = LoadTestSuite(config=performance_config, output_dir=temp_results_dir)
            report = load_suite.generate_load_test_report(results)
            
            assert "LIGHTRAG LOAD TESTING REPORT" in report
            assert "Max Stable Users:" in report
            assert "Peak Throughput:" in report
            
            print(f"✅ Comprehensive test completed")
            print(f"✅ Max stable users: {summary['max_stable_users']}")
            print(f"✅ Peak throughput: {summary['peak_throughput']:.2f} req/s")
            print(f"✅ System scalability: {summary['system_scalability']}")
            print(f"✅ Recommendations: {len(recommendations)}")
            
        finally:
            await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_endurance_testing_memory_leak_detection(self, performance_config, temp_results_dir):
        """
        Test endurance testing for memory leak detection.
        This validates long-term stability testing capabilities.
        """
        # Create component with memory leak simulation
        component = PerformanceTestComponent(
            base_response_time=0.1,
            memory_growth_rate=0.5,  # Significant memory growth to simulate leak
            failure_threshold=100
        )
        await component.initialize()
        
        try:
            load_suite = LoadTestSuite(config=performance_config, output_dir=temp_results_dir)
            
            # Run endurance test
            endurance_result = await load_suite.run_endurance_test(
                component=component,
                duration_hours=0.1,  # 6 minutes for test speed
                constant_load_users=10,
                sampling_interval_minutes=0.02  # 1.2 seconds between samples
            )
            
            # Validate endurance test results
            assert endurance_result.duration_hours == 0.1
            assert endurance_result.constant_load_users == 10
            assert len(endurance_result.performance_over_time) > 0
            
            # Validate memory leak analysis
            memory_analysis = endurance_result.memory_leak_analysis
            assert "memory_growth_rate_mb_per_hour" in memory_analysis
            assert "memory_increase_percent" in memory_analysis
            assert "potential_memory_leak" in memory_analysis
            
            # Should detect memory leak due to high growth rate
            assert memory_analysis["potential_memory_leak"], "Should detect simulated memory leak"
            
            # Validate performance degradation analysis
            perf_degradation = endurance_result.performance_degradation
            assert "response_time_degradation_percent" in perf_degradation
            
            # Validate stability score
            assert 0 <= endurance_result.stability_score <= 1
            
            print(f"✅ Endurance test: {len(endurance_result.performance_over_time)} samples")
            print(f"✅ Memory leak detected: {memory_analysis['potential_memory_leak']}")
            print(f"✅ Memory growth rate: {memory_analysis['memory_growth_rate_mb_per_hour']:.2f} MB/hour")
            print(f"✅ Stability score: {endurance_result.stability_score:.2f}")
            
        finally:
            await component.cleanup()
    
    def test_performance_test_configuration_validation(self, performance_config, temp_results_dir):
        """
        Test that performance testing components are properly configured.
        """
        # Test load test suite configuration
        load_suite = LoadTestSuite(config=performance_config, output_dir=temp_results_dir)
        
        # Validate thresholds
        assert load_suite.load_test_thresholds["max_response_time_50_users"] > 0
        assert load_suite.load_test_thresholds["min_success_rate"] > 0.9
        assert load_suite.load_test_thresholds["max_memory_usage_per_user"] > 0
        
        # Validate test questions
        assert len(load_suite.test_questions["mixed"]) >= 10
        assert "What is clinical metabolomics?" in load_suite.test_questions["basic"]
        
        # Test regression detector configuration
        regression_detector = PerformanceRegressionDetector(output_dir=temp_results_dir)
        
        # Validate regression thresholds
        assert regression_detector.regression_thresholds["minor"] < regression_detector.regression_thresholds["moderate"]
        assert regression_detector.regression_thresholds["moderate"] < regression_detector.regression_thresholds["severe"]
        assert regression_detector.regression_thresholds["severe"] < regression_detector.regression_thresholds["critical"]
        
        # Validate metric configurations
        assert "response_time" in regression_detector.metric_configs
        assert "memory_usage" in regression_detector.metric_configs
        assert "throughput" in regression_detector.metric_configs
        
        print("✅ All performance testing components properly configured")


@pytest.mark.asyncio
async def test_performance_requirements_validation():
    """
    Validate that the performance testing system meets all specified requirements.
    """
    requirements_met = {
        "50_plus_concurrent_users": False,
        "stress_testing_large_collections": False,
        "performance_regression_detection": False,
        "scalability_testing_system_limits": False
    }
    
    # This would be run as part of the full test suite
    # For now, we validate that the test structure supports all requirements
    
    # Check that test methods exist for each requirement
    test_class = TestPerformanceIntegration
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    required_tests = [
        "test_50_plus_concurrent_users_requirement",
        "test_stress_testing_large_document_collections", 
        "test_scalability_testing_system_limits",
        "test_performance_regression_detection"
    ]
    
    for required_test in required_tests:
        assert required_test in test_methods, f"Missing required test: {required_test}"
        requirements_met[required_test.replace("test_", "").replace("_requirement", "")] = True
    
    # Validate all requirements can be tested
    assert all(requirements_met.values()), f"Some requirements not testable: {requirements_met}"
    
    print("✅ All performance testing requirements have corresponding tests")
    print(f"✅ Requirements coverage: {requirements_met}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-s"])