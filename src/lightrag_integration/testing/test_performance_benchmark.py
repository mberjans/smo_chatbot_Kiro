"""
Tests for Performance Benchmark Module

This module contains unit tests for the performance benchmarking functionality,
validating measurement accuracy, load testing, and regression analysis.
"""

import pytest
import asyncio
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from .performance_benchmark import (
    PerformanceBenchmark,
    MemoryMonitor,
    PerformanceMetrics,
    LoadTestResult,
    BenchmarkSuite,
    run_performance_benchmark
)
from ..component import LightRAGComponent
from ..config.settings import LightRAGConfig


class TestMemoryMonitor:
    """Test cases for the memory monitor."""
    
    def test_memory_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor()
        
        assert not monitor.monitoring
        assert monitor.measurements == []
        assert monitor.monitor_thread is None
    
    def test_get_current_memory(self):
        """Test current memory measurement."""
        monitor = MemoryMonitor()
        memory = monitor.get_current_memory()
        
        assert isinstance(memory, float)
        assert memory > 0  # Should have some memory usage
    
    def test_monitoring_lifecycle(self):
        """Test start/stop monitoring lifecycle."""
        monitor = MemoryMonitor()
        
        # Start monitoring
        monitor.start_monitoring(interval=0.01)  # Very short interval for testing
        assert monitor.monitoring
        
        # Let it collect some measurements
        time.sleep(0.05)
        
        # Stop monitoring
        stats = monitor.stop_monitoring()
        assert not monitor.monitoring
        
        # Check statistics
        assert isinstance(stats, dict)
        assert "current" in stats
        assert "peak" in stats
        assert "average" in stats
        assert "min" in stats
        
        assert stats["current"] > 0
        assert stats["peak"] >= stats["current"]
        assert stats["average"] > 0


class TestPerformanceBenchmark:
    """Test cases for the performance benchmark."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        benchmark = PerformanceBenchmark()
        
        assert benchmark.config is not None
        assert benchmark.memory_monitor is not None
        assert isinstance(benchmark.thresholds, dict)
        assert "max_response_time" in benchmark.thresholds
    
    @pytest.mark.asyncio
    async def test_measure_operation_success(self):
        """Test measuring a successful operation."""
        benchmark = PerformanceBenchmark()
        
        async def test_operation(value):
            await asyncio.sleep(0.01)  # Simulate work
            return value * 2
        
        metrics = await benchmark.measure_operation(
            "test_operation",
            test_operation,
            5
        )
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation_name == "test_operation"
        assert metrics.success is True
        assert metrics.error is None
        assert metrics.duration_seconds > 0
        assert metrics.memory_before_mb > 0
        assert metrics.memory_after_mb > 0
    
    @pytest.mark.asyncio
    async def test_measure_operation_failure(self):
        """Test measuring a failed operation."""
        benchmark = PerformanceBenchmark()
        
        async def failing_operation():
            raise ValueError("Test error")
        
        metrics = await benchmark.measure_operation(
            "failing_operation",
            failing_operation
        )
        
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation_name == "failing_operation"
        assert metrics.success is False
        assert metrics.error == "Test error"
        assert metrics.duration_seconds > 0
    
    @pytest.mark.asyncio
    async def test_benchmark_component_operations(self):
        """Test benchmarking component operations."""
        benchmark = PerformanceBenchmark()
        
        # Create mock component
        mock_component = AsyncMock(spec=LightRAGComponent)
        mock_component.initialize.return_value = None
        mock_component.ingest_documents.return_value = {"processed": 1}
        mock_component.query.return_value = {"answer": "test answer"}
        
        # Test initialization benchmark
        init_metrics = await benchmark.benchmark_component_initialization(mock_component)
        assert init_metrics.operation_name == "component_initialization"
        assert init_metrics.success is True
        
        # Test ingestion benchmark
        ingestion_metrics = await benchmark.benchmark_document_ingestion(
            mock_component, ["test_doc.txt"]
        )
        assert ingestion_metrics.operation_name == "document_ingestion"
        assert ingestion_metrics.success is True
        
        # Test query benchmark
        query_metrics = await benchmark.benchmark_single_query(
            mock_component, "What is clinical metabolomics?"
        )
        assert query_metrics.operation_name == "single_query"
        assert query_metrics.success is True
    
    @pytest.mark.asyncio
    async def test_load_test(self):
        """Test load testing functionality."""
        benchmark = PerformanceBenchmark()
        
        # Create mock component with realistic response times
        mock_component = AsyncMock(spec=LightRAGComponent)
        
        async def mock_query(question):
            await asyncio.sleep(0.01)  # Simulate processing time
            return {
                "answer": f"Answer to: {question}",
                "confidence_score": 0.8,
                "processing_time": 0.01
            }
        
        mock_component.query = mock_query
        
        # Run small load test
        result = await benchmark.run_load_test(
            component=mock_component,
            questions=["Question 1", "Question 2"],
            concurrent_users=2,
            requests_per_user=2,
            ramp_up_time=0.1
        )
        
        assert isinstance(result, LoadTestResult)
        assert result.concurrent_users == 2
        assert result.total_requests == 4
        assert result.successful_requests <= 4
        assert result.failed_requests >= 0
        assert result.duration_seconds > 0
        assert result.requests_per_second >= 0
        assert result.average_response_time >= 0
    
    def test_system_info(self):
        """Test system information collection."""
        benchmark = PerformanceBenchmark()
        system_info = benchmark.get_system_info()
        
        assert isinstance(system_info, dict)
        assert "cpu_count" in system_info
        assert "memory_total_gb" in system_info
        assert "memory_available_gb" in system_info
        assert "python_version" in system_info
        assert "platform" in system_info
        
        assert system_info["cpu_count"] > 0
        assert system_info["memory_total_gb"] > 0
    
    def test_regression_analysis_no_baseline(self):
        """Test regression analysis without baseline."""
        benchmark = PerformanceBenchmark()
        
        # Create mock current results
        from datetime import datetime
        current_results = [
            PerformanceMetrics(
                operation_name="test_op",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.0,
                memory_before_mb=100,
                memory_after_mb=110,
                memory_peak_mb=115,
                cpu_percent=50,
                success=True
            )
        ]
        
        analysis = benchmark.analyze_performance_regression(
            current_results, "nonexistent_baseline.json"
        )
        
        assert analysis["baseline_available"] is False
        assert "message" in analysis
    
    def test_regression_analysis_with_baseline(self):
        """Test regression analysis with baseline."""
        benchmark = PerformanceBenchmark()
        
        # Create mock baseline data
        baseline_data = {
            "individual_metrics": [
                {
                    "operation_name": "test_op",
                    "duration_seconds": 1.0
                }
            ]
        }
        
        # Create mock current results
        from datetime import datetime
        current_results = [
            PerformanceMetrics(
                operation_name="test_op",
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=1.5,  # 50% slower
                memory_before_mb=100,
                memory_after_mb=110,
                memory_peak_mb=115,
                cpu_percent=50,
                success=True
            )
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(baseline_data, f)
            baseline_file = f.name
        
        try:
            analysis = benchmark.analyze_performance_regression(
                current_results, baseline_file
            )
            
            assert analysis["baseline_available"] is True
            assert "operations" in analysis
            assert "test_op" in analysis["operations"]
            
            op_analysis = analysis["operations"]["test_op"]
            assert op_analysis["current_avg"] == 1.5
            assert op_analysis["baseline_avg"] == 1.0
            assert op_analysis["change_percent"] == 50.0
            assert op_analysis["is_regression"] is True
            
        finally:
            Path(baseline_file).unlink()
    
    def test_report_generation(self):
        """Test benchmark report generation."""
        benchmark = PerformanceBenchmark()
        
        # Create mock benchmark suite
        from datetime import datetime
        
        mock_suite = BenchmarkSuite(
            timestamp=datetime.now(),
            system_info={
                "cpu_count": 4,
                "memory_total_gb": 16.0,
                "memory_available_gb": 8.0,
                "platform": "test"
            },
            individual_metrics=[
                PerformanceMetrics(
                    operation_name="test_op",
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=1.0,
                    memory_before_mb=100,
                    memory_after_mb=110,
                    memory_peak_mb=115,
                    cpu_percent=50,
                    success=True
                )
            ],
            load_test_results=[],
            regression_analysis={"baseline_available": False},
            summary={
                "total_operations": 1,
                "successful_operations": 1,
                "failed_operations": 0,
                "average_response_time": 1.0,
                "max_response_time": 1.0,
                "average_memory_usage": 115.0,
                "max_memory_usage": 115.0,
                "performance_thresholds_met": {
                    "response_time": True,
                    "memory_usage": True
                }
            }
        )
        
        report = benchmark.generate_benchmark_report(mock_suite)
        
        assert "LIGHTRAG PERFORMANCE BENCHMARK REPORT" in report
        assert "SYSTEM INFORMATION:" in report
        assert "SUMMARY:" in report
        assert "PERFORMANCE THRESHOLDS:" in report
        assert "INDIVIDUAL OPERATION METRICS:" in report
        assert "test_op: ✅ SUCCESS" in report
    
    def test_save_benchmark_results(self):
        """Test saving benchmark results to JSON."""
        benchmark = PerformanceBenchmark()
        
        # Create mock benchmark suite
        from datetime import datetime
        
        mock_suite = BenchmarkSuite(
            timestamp=datetime.now(),
            system_info={"test": "data"},
            individual_metrics=[],
            load_test_results=[],
            regression_analysis={},
            summary={"test": "summary"}
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
        
        try:
            benchmark.save_benchmark_results(mock_suite, output_file)
            
            # Verify file was created and contains valid JSON
            assert Path(output_file).exists()
            
            with open(output_file) as f:
                loaded_data = json.load(f)
            
            assert "timestamp" in loaded_data
            assert "system_info" in loaded_data
            assert "summary" in loaded_data
            
        finally:
            Path(output_file).unlink()


@pytest.mark.asyncio
async def test_run_performance_benchmark_integration():
    """Test the complete performance benchmark workflow."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test configuration
        config = LightRAGConfig()
        config.papers_directory = str(Path(temp_dir) / "papers")
        config.knowledge_graph_path = str(Path(temp_dir) / "kg")
        config.vector_store_path = str(Path(temp_dir) / "vectors")
        config.cache_directory = str(Path(temp_dir) / "cache")
        
        # Mock the component to avoid actual LightRAG initialization
        original_component_init = LightRAGComponent.__init__
        original_component_initialize = LightRAGComponent.initialize
        original_component_ingest = LightRAGComponent.ingest_documents
        original_component_query = LightRAGComponent.query
        original_component_cleanup = LightRAGComponent.cleanup
        
        def mock_init(self, config):
            self.config = config
            self._initialized = False
        
        async def mock_initialize(self):
            await asyncio.sleep(0.01)  # Simulate initialization time
            self._initialized = True
        
        async def mock_ingest(self, documents):
            await asyncio.sleep(0.02)  # Simulate ingestion time
            return {"processed": len(documents), "success": True}
        
        async def mock_query(self, question, context=None):
            await asyncio.sleep(0.01)  # Simulate query time
            return {
                "answer": f"Mock answer for: {question}",
                "confidence_score": 0.8,
                "source_documents": ["mock_paper.txt"],
                "processing_time": 0.01
            }
        
        async def mock_cleanup(self):
            pass
        
        try:
            # Apply mocks
            LightRAGComponent.__init__ = mock_init
            LightRAGComponent.initialize = mock_initialize
            LightRAGComponent.ingest_documents = mock_ingest
            LightRAGComponent.query = mock_query
            LightRAGComponent.cleanup = mock_cleanup
            
            # Run performance benchmark
            results = await run_performance_benchmark(config)
            
            # Validate results
            assert isinstance(results, BenchmarkSuite)
            assert len(results.individual_metrics) > 0
            assert results.summary["total_operations"] > 0
            assert results.system_info["cpu_count"] > 0
            
        finally:
            # Restore original methods
            LightRAGComponent.__init__ = original_component_init
            LightRAGComponent.initialize = original_component_initialize
            LightRAGComponent.ingest_documents = original_component_ingest
            LightRAGComponent.query = original_component_query
            LightRAGComponent.cleanup = original_component_cleanup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])