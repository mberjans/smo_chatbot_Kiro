"""
Performance Benchmarking for LightRAG Integration

This module provides comprehensive performance testing capabilities including
response time measurement, load testing, memory usage monitoring, and
performance regression detection.
"""

import asyncio
import logging
import time
import psutil
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import gc

from ..component import LightRAGComponent
from ..config.settings import LightRAGConfig
from .clinical_metabolomics_suite import ClinicalMetabolomicsTestSuite, TestQuestion


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation_name: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    cpu_percent: float
    success: bool
    error: Optional[str] = None


@dataclass
class LoadTestResult:
    """Results from a load test run."""
    test_name: str
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    requests_per_second: float
    average_response_time: float
    min_response_time: float
    max_response_time: float
    percentile_95_response_time: float
    percentile_99_response_time: float
    memory_usage_mb: Dict[str, float]
    cpu_usage_percent: Dict[str, float]
    error_details: List[str]


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""
    timestamp: datetime
    system_info: Dict[str, Any]
    individual_metrics: List[PerformanceMetrics]
    load_test_results: List[LoadTestResult]
    regression_analysis: Dict[str, Any]
    summary: Dict[str, Any]


class MemoryMonitor:
    """Monitor memory usage during operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.measurements = []
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    def start_monitoring(self, interval: float = 0.1):
        """Start memory monitoring in background thread."""
        with self.lock:
            if self.monitoring:
                return
            
            self.monitoring = True
            self.measurements = []
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                args=(interval,),
                daemon=True
            )
            self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics."""
        with self.lock:
            if not self.monitoring:
                return {"current": 0, "peak": 0, "average": 0}
            
            self.monitoring = False
        
        # Wait for monitor thread to finish
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.measurements:
            return {"current": 0, "peak": 0, "average": 0}
        
        return {
            "current": self.measurements[-1],
            "peak": max(self.measurements),
            "average": statistics.mean(self.measurements),
            "min": min(self.measurements)
        }
    
    def _monitor_loop(self, interval: float):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                memory_mb = self.process.memory_info().rss / 1024 / 1024
                with self.lock:
                    if self.monitoring:  # Check again inside lock
                        self.measurements.append(memory_mb)
                time.sleep(interval)
            except Exception:
                break
    
    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0


class PerformanceBenchmark:
    """
    Comprehensive performance benchmarking for LightRAG integration.
    
    This class provides methods for measuring response times, conducting load tests,
    monitoring memory usage, and detecting performance regressions.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None):
        """Initialize the performance benchmark."""
        self.config = config or LightRAGConfig.from_env()
        self.logger = logging.getLogger(__name__)
        self.memory_monitor = MemoryMonitor()
        
        # Performance thresholds
        self.thresholds = {
            "max_response_time": 5.0,  # seconds
            "max_memory_usage": 2048,  # MB
            "min_requests_per_second": 1.0,
            "max_error_rate": 0.05  # 5%
        }
    
    async def measure_operation(self, 
                              operation_name: str,
                              operation_func: Callable,
                              *args, **kwargs) -> PerformanceMetrics:
        """
        Measure performance of a single operation.
        
        Args:
            operation_name: Name of the operation being measured
            operation_func: Async function to measure
            *args, **kwargs: Arguments for the operation function
            
        Returns:
            PerformanceMetrics with detailed measurements
        """
        # Start monitoring
        memory_before = self.memory_monitor.get_current_memory()
        self.memory_monitor.start_monitoring()
        
        start_time = datetime.now()
        cpu_before = psutil.cpu_percent()
        
        success = True
        error = None
        
        try:
            # Execute operation
            if asyncio.iscoroutinefunction(operation_func):
                result = await operation_func(*args, **kwargs)
            else:
                result = operation_func(*args, **kwargs)
                
        except Exception as e:
            success = False
            error = str(e)
            result = None
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Stop monitoring and get memory stats
        memory_stats = self.memory_monitor.stop_monitoring()
        memory_after = self.memory_monitor.get_current_memory()
        cpu_after = psutil.cpu_percent()
        
        return PerformanceMetrics(
            operation_name=operation_name,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            memory_before_mb=memory_before,
            memory_after_mb=memory_after,
            memory_peak_mb=memory_stats["peak"],
            cpu_percent=(cpu_before + cpu_after) / 2,
            success=success,
            error=error
        )
    
    async def benchmark_component_initialization(self, 
                                               component: LightRAGComponent) -> PerformanceMetrics:
        """Benchmark component initialization performance."""
        return await self.measure_operation(
            "component_initialization",
            component.initialize
        )
    
    async def benchmark_document_ingestion(self, 
                                         component: LightRAGComponent,
                                         documents: List[str]) -> PerformanceMetrics:
        """Benchmark document ingestion performance."""
        return await self.measure_operation(
            "document_ingestion",
            component.ingest_documents,
            documents
        )
    
    async def benchmark_single_query(self, 
                                   component: LightRAGComponent,
                                   question: str) -> PerformanceMetrics:
        """Benchmark single query performance."""
        return await self.measure_operation(
            "single_query",
            component.query,
            question
        )
    
    async def run_load_test(self, 
                          component: LightRAGComponent,
                          questions: List[str],
                          concurrent_users: int = 10,
                          requests_per_user: int = 5,
                          ramp_up_time: float = 1.0) -> LoadTestResult:
        """
        Run load test with concurrent users.
        
        Args:
            component: LightRAG component to test
            questions: List of questions to ask
            concurrent_users: Number of concurrent users to simulate
            requests_per_user: Number of requests each user makes
            ramp_up_time: Time to ramp up all users (seconds)
            
        Returns:
            LoadTestResult with comprehensive metrics
        """
        self.logger.info(
            f"Starting load test: {concurrent_users} users, "
            f"{requests_per_user} requests each"
        )
        
        # Prepare test data
        total_requests = concurrent_users * requests_per_user
        results = []
        errors = []
        
        # Start system monitoring
        self.memory_monitor.start_monitoring()
        start_time = datetime.now()
        
        async def user_simulation(user_id: int):
            """Simulate a single user's requests."""
            user_results = []
            
            # Stagger user start times for ramp-up
            await asyncio.sleep((user_id / concurrent_users) * ramp_up_time)
            
            for request_id in range(requests_per_user):
                question = questions[request_id % len(questions)]
                request_start = time.time()
                
                try:
                    response = await component.query(question)
                    request_end = time.time()
                    
                    user_results.append({
                        "user_id": user_id,
                        "request_id": request_id,
                        "response_time": request_end - request_start,
                        "success": True,
                        "error": None
                    })
                    
                except Exception as e:
                    request_end = time.time()
                    error_msg = f"User {user_id}, Request {request_id}: {str(e)}"
                    errors.append(error_msg)
                    
                    user_results.append({
                        "user_id": user_id,
                        "request_id": request_id,
                        "response_time": request_end - request_start,
                        "success": False,
                        "error": str(e)
                    })
            
            return user_results
        
        # Run concurrent user simulations
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all results
        for user_result in user_results:
            if isinstance(user_result, Exception):
                errors.append(f"User simulation failed: {str(user_result)}")
            else:
                results.extend(user_result)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Stop monitoring and get system stats
        memory_stats = self.memory_monitor.stop_monitoring()
        
        # Calculate metrics
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_results]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        requests_per_second = len(successful_results) / total_duration if total_duration > 0 else 0
        
        return LoadTestResult(
            test_name=f"load_test_{concurrent_users}users_{requests_per_user}req",
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            duration_seconds=total_duration,
            requests_per_second=requests_per_second,
            average_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            percentile_95_response_time=p95_response_time,
            percentile_99_response_time=p99_response_time,
            memory_usage_mb=memory_stats,
            cpu_usage_percent={"average": psutil.cpu_percent()},
            error_details=errors
        )
    
    async def run_stress_test(self, 
                            component: LightRAGComponent,
                            questions: List[str],
                            duration_minutes: int = 10,
                            max_concurrent_users: int = 50) -> List[LoadTestResult]:
        """
        Run stress test with gradually increasing load.
        
        Args:
            component: LightRAG component to test
            questions: List of questions to ask
            duration_minutes: Total test duration
            max_concurrent_users: Maximum concurrent users to reach
            
        Returns:
            List of LoadTestResult for each load level
        """
        self.logger.info(
            f"Starting stress test: up to {max_concurrent_users} users "
            f"over {duration_minutes} minutes"
        )
        
        results = []
        user_levels = [1, 5, 10, 20, 30, 40, 50]
        user_levels = [u for u in user_levels if u <= max_concurrent_users]
        
        test_duration_per_level = (duration_minutes * 60) / len(user_levels)
        requests_per_user = max(1, int(test_duration_per_level / 10))  # Rough estimate
        
        for user_count in user_levels:
            self.logger.info(f"Testing with {user_count} concurrent users")
            
            try:
                result = await self.run_load_test(
                    component=component,
                    questions=questions,
                    concurrent_users=user_count,
                    requests_per_user=requests_per_user,
                    ramp_up_time=2.0
                )
                results.append(result)
                
                # Brief pause between load levels
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Stress test failed at {user_count} users: {str(e)}")
                break
        
        return results
    
    async def run_memory_stress_test(self, 
                                   component: LightRAGComponent,
                                   large_documents: List[str],
                                   iterations: int = 10) -> List[PerformanceMetrics]:
        """
        Test memory usage with large document processing.
        
        Args:
            component: LightRAG component to test
            large_documents: List of large documents to process
            iterations: Number of iterations to run
            
        Returns:
            List of PerformanceMetrics for each iteration
        """
        self.logger.info(f"Starting memory stress test with {iterations} iterations")
        
        results = []
        
        for i in range(iterations):
            self.logger.info(f"Memory stress test iteration {i+1}/{iterations}")
            
            # Force garbage collection before each iteration
            gc.collect()
            
            # Measure document ingestion
            metrics = await self.benchmark_document_ingestion(component, large_documents)
            metrics.operation_name = f"memory_stress_iteration_{i+1}"
            results.append(metrics)
            
            # Check if memory usage is growing excessively
            if metrics.memory_peak_mb > self.thresholds["max_memory_usage"]:
                self.logger.warning(
                    f"Memory usage exceeded threshold: {metrics.memory_peak_mb:.1f}MB"
                )
        
        return results
    
    def analyze_performance_regression(self, 
                                     current_results: List[PerformanceMetrics],
                                     baseline_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze performance regression compared to baseline.
        
        Args:
            current_results: Current performance metrics
            baseline_file: Path to baseline results JSON file
            
        Returns:
            Dictionary with regression analysis
        """
        if not baseline_file or not Path(baseline_file).exists():
            return {
                "baseline_available": False,
                "message": "No baseline available for comparison"
            }
        
        try:
            with open(baseline_file) as f:
                baseline_data = json.load(f)
            
            baseline_metrics = baseline_data.get("individual_metrics", [])
            
            if not baseline_metrics:
                return {
                    "baseline_available": False,
                    "message": "No baseline metrics found"
                }
            
            # Group metrics by operation name
            current_by_operation = {}
            baseline_by_operation = {}
            
            for metric in current_results:
                op_name = metric.operation_name
                if op_name not in current_by_operation:
                    current_by_operation[op_name] = []
                current_by_operation[op_name].append(metric.duration_seconds)
            
            for metric_data in baseline_metrics:
                op_name = metric_data["operation_name"]
                if op_name not in baseline_by_operation:
                    baseline_by_operation[op_name] = []
                baseline_by_operation[op_name].append(metric_data["duration_seconds"])
            
            # Compare operations
            regression_analysis = {
                "baseline_available": True,
                "operations": {},
                "overall_regression": False,
                "significant_regressions": []
            }
            
            for op_name in current_by_operation:
                if op_name not in baseline_by_operation:
                    continue
                
                current_times = current_by_operation[op_name]
                baseline_times = baseline_by_operation[op_name]
                
                current_avg = statistics.mean(current_times)
                baseline_avg = statistics.mean(baseline_times)
                
                change_percent = ((current_avg - baseline_avg) / baseline_avg) * 100
                
                # Consider >20% increase as significant regression
                is_regression = change_percent > 20
                
                regression_analysis["operations"][op_name] = {
                    "current_avg": current_avg,
                    "baseline_avg": baseline_avg,
                    "change_percent": change_percent,
                    "is_regression": is_regression
                }
                
                if is_regression:
                    regression_analysis["overall_regression"] = True
                    regression_analysis["significant_regressions"].append({
                        "operation": op_name,
                        "change_percent": change_percent,
                        "current_time": current_avg,
                        "baseline_time": baseline_avg
                    })
            
            return regression_analysis
            
        except Exception as e:
            return {
                "baseline_available": False,
                "error": f"Failed to analyze regression: {str(e)}"
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for benchmark context."""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "memory_available_gb": psutil.virtual_memory().available / 1024 / 1024 / 1024,
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            "platform": psutil.sys.platform
        }
    
    async def run_comprehensive_benchmark(self, 
                                        component: LightRAGComponent,
                                        test_documents: List[str],
                                        test_questions: List[str],
                                        baseline_file: Optional[str] = None) -> BenchmarkSuite:
        """
        Run comprehensive performance benchmark suite.
        
        Args:
            component: LightRAG component to benchmark
            test_documents: Documents for ingestion testing
            test_questions: Questions for query testing
            baseline_file: Optional baseline for regression analysis
            
        Returns:
            BenchmarkSuite with complete results
        """
        self.logger.info("Starting comprehensive performance benchmark")
        
        individual_metrics = []
        load_test_results = []
        
        # 1. Component initialization benchmark
        self.logger.info("Benchmarking component initialization")
        init_metrics = await self.benchmark_component_initialization(component)
        individual_metrics.append(init_metrics)
        
        # 2. Document ingestion benchmark
        if test_documents:
            self.logger.info("Benchmarking document ingestion")
            ingestion_metrics = await self.benchmark_document_ingestion(component, test_documents)
            individual_metrics.append(ingestion_metrics)
        
        # 3. Single query benchmarks
        self.logger.info("Benchmarking individual queries")
        for i, question in enumerate(test_questions[:5]):  # Test first 5 questions
            query_metrics = await self.benchmark_single_query(component, question)
            query_metrics.operation_name = f"query_{i+1}"
            individual_metrics.append(query_metrics)
        
        # 4. Load testing
        self.logger.info("Running load tests")
        for user_count in [1, 5, 10]:
            load_result = await self.run_load_test(
                component=component,
                questions=test_questions[:3],  # Use first 3 questions
                concurrent_users=user_count,
                requests_per_user=3
            )
            load_test_results.append(load_result)
        
        # 5. Regression analysis
        regression_analysis = self.analyze_performance_regression(
            individual_metrics, baseline_file
        )
        
        # 6. Generate summary
        successful_metrics = [m for m in individual_metrics if m.success]
        
        summary = {
            "total_operations": len(individual_metrics),
            "successful_operations": len(successful_metrics),
            "failed_operations": len(individual_metrics) - len(successful_metrics),
            "average_response_time": statistics.mean([m.duration_seconds for m in successful_metrics]) if successful_metrics else 0,
            "max_response_time": max([m.duration_seconds for m in successful_metrics]) if successful_metrics else 0,
            "average_memory_usage": statistics.mean([m.memory_peak_mb for m in successful_metrics]) if successful_metrics else 0,
            "max_memory_usage": max([m.memory_peak_mb for m in successful_metrics]) if successful_metrics else 0,
            "performance_thresholds_met": {
                "response_time": all(m.duration_seconds <= self.thresholds["max_response_time"] for m in successful_metrics),
                "memory_usage": all(m.memory_peak_mb <= self.thresholds["max_memory_usage"] for m in successful_metrics)
            }
        }
        
        return BenchmarkSuite(
            timestamp=datetime.now(),
            system_info=self.get_system_info(),
            individual_metrics=individual_metrics,
            load_test_results=load_test_results,
            regression_analysis=regression_analysis,
            summary=summary
        )
    
    def generate_benchmark_report(self, 
                                benchmark_suite: BenchmarkSuite,
                                output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive benchmark report.
        
        Args:
            benchmark_suite: Benchmark results
            output_file: Optional file to save report
            
        Returns:
            Report content as string
        """
        report_lines = [
            "=" * 80,
            "LIGHTRAG PERFORMANCE BENCHMARK REPORT",
            "=" * 80,
            f"Timestamp: {benchmark_suite.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SYSTEM INFORMATION:",
            f"  CPU Cores: {benchmark_suite.system_info['cpu_count']}",
            f"  Total Memory: {benchmark_suite.system_info['memory_total_gb']:.1f} GB",
            f"  Available Memory: {benchmark_suite.system_info['memory_available_gb']:.1f} GB",
            f"  Platform: {benchmark_suite.system_info['platform']}",
            "",
            "SUMMARY:",
            f"  Total Operations: {benchmark_suite.summary['total_operations']}",
            f"  Successful: {benchmark_suite.summary['successful_operations']}",
            f"  Failed: {benchmark_suite.summary['failed_operations']}",
            f"  Average Response Time: {benchmark_suite.summary['average_response_time']:.3f}s",
            f"  Max Response Time: {benchmark_suite.summary['max_response_time']:.3f}s",
            f"  Average Memory Usage: {benchmark_suite.summary['average_memory_usage']:.1f} MB",
            f"  Max Memory Usage: {benchmark_suite.summary['max_memory_usage']:.1f} MB",
            "",
            "PERFORMANCE THRESHOLDS:",
            f"  Response Time (<{self.thresholds['max_response_time']}s): {'✅ PASS' if benchmark_suite.summary['performance_thresholds_met']['response_time'] else '❌ FAIL'}",
            f"  Memory Usage (<{self.thresholds['max_memory_usage']}MB): {'✅ PASS' if benchmark_suite.summary['performance_thresholds_met']['memory_usage'] else '❌ FAIL'}",
            "",
            "INDIVIDUAL OPERATION METRICS:",
            "-" * 50
        ]
        
        for metric in benchmark_suite.individual_metrics:
            status = "✅ SUCCESS" if metric.success else "❌ FAILED"
            report_lines.extend([
                f"{metric.operation_name}: {status}",
                f"  Duration: {metric.duration_seconds:.3f}s",
                f"  Memory Peak: {metric.memory_peak_mb:.1f} MB",
                f"  CPU Usage: {metric.cpu_percent:.1f}%",
            ])
            
            if metric.error:
                report_lines.append(f"  Error: {metric.error}")
            
            report_lines.append("")
        
        if benchmark_suite.load_test_results:
            report_lines.extend([
                "LOAD TEST RESULTS:",
                "-" * 50
            ])
            
            for load_result in benchmark_suite.load_test_results:
                error_rate = load_result.failed_requests / load_result.total_requests
                report_lines.extend([
                    f"{load_result.test_name}:",
                    f"  Concurrent Users: {load_result.concurrent_users}",
                    f"  Total Requests: {load_result.total_requests}",
                    f"  Success Rate: {(1-error_rate):.1%}",
                    f"  Requests/Second: {load_result.requests_per_second:.2f}",
                    f"  Avg Response Time: {load_result.average_response_time:.3f}s",
                    f"  95th Percentile: {load_result.percentile_95_response_time:.3f}s",
                    f"  99th Percentile: {load_result.percentile_99_response_time:.3f}s",
                    f"  Memory Peak: {load_result.memory_usage_mb['peak']:.1f} MB",
                    ""
                ])
        
        if benchmark_suite.regression_analysis.get("baseline_available"):
            report_lines.extend([
                "REGRESSION ANALYSIS:",
                "-" * 50
            ])
            
            if benchmark_suite.regression_analysis["overall_regression"]:
                report_lines.append("❌ PERFORMANCE REGRESSION DETECTED")
                
                for regression in benchmark_suite.regression_analysis["significant_regressions"]:
                    report_lines.append(
                        f"  {regression['operation']}: "
                        f"{regression['change_percent']:+.1f}% "
                        f"({regression['baseline_time']:.3f}s → {regression['current_time']:.3f}s)"
                    )
            else:
                report_lines.append("✅ NO SIGNIFICANT REGRESSION DETECTED")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report_content)
            self.logger.info(f"Benchmark report saved to {output_file}")
        
        return report_content
    
    def save_benchmark_results(self, 
                             benchmark_suite: BenchmarkSuite,
                             output_file: str) -> None:
        """Save benchmark results as JSON for analysis."""
        # Convert dataclass to dict for JSON serialization
        result_dict = asdict(benchmark_suite)
        
        # Convert datetime objects to strings
        result_dict['timestamp'] = benchmark_suite.timestamp.isoformat()
        
        for metric in result_dict['individual_metrics']:
            metric['start_time'] = datetime.fromisoformat(metric['start_time']).isoformat() if isinstance(metric['start_time'], str) else metric['start_time'].isoformat()
            metric['end_time'] = datetime.fromisoformat(metric['end_time']).isoformat() if isinstance(metric['end_time'], str) else metric['end_time'].isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Benchmark results saved to {output_file}")


async def run_performance_benchmark(config: Optional[LightRAGConfig] = None) -> BenchmarkSuite:
    """
    Run comprehensive performance benchmark for LightRAG.
    
    This is the main entry point for performance testing.
    
    Args:
        config: Optional LightRAG configuration
        
    Returns:
        BenchmarkSuite with complete results
    """
    # Create benchmark instance
    benchmark = PerformanceBenchmark(config)
    
    # Create test suite for questions
    test_suite = ClinicalMetabolomicsTestSuite()
    test_questions = [q.question for q in test_suite.test_questions[:10]]  # First 10 questions
    
    # Create temporary test environment
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up test configuration
        test_config = config or LightRAGConfig.from_env()
        test_config.papers_directory = str(Path(temp_dir) / "papers")
        test_config.knowledge_graph_path = str(Path(temp_dir) / "kg")
        test_config.vector_store_path = str(Path(temp_dir) / "vectors")
        test_config.cache_directory = str(Path(temp_dir) / "cache")
        
        # Create test papers
        test_documents = test_suite.create_test_papers_dataset(test_config.papers_directory)
        
        # Initialize component
        component = LightRAGComponent(test_config)
        
        try:
            # Run comprehensive benchmark
            results = await benchmark.run_comprehensive_benchmark(
                component=component,
                test_documents=test_documents,
                test_questions=test_questions
            )
            
            return results
            
        finally:
            await component.cleanup()


if __name__ == "__main__":
    # Run performance benchmark
    async def main():
        logging.basicConfig(level=logging.INFO)
        results = await run_performance_benchmark()
        
        # Generate and print report
        benchmark = PerformanceBenchmark()
        report = benchmark.generate_benchmark_report(results)
        print(report)
        
        # Save results
        benchmark.save_benchmark_results(results, "performance_benchmark_results.json")
    
    asyncio.run(main())