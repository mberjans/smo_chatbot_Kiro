"""
Enhanced Load Testing Suite for LightRAG Integration

This module provides comprehensive load testing capabilities including
concurrent user simulation, stress testing, scalability testing, and
performance regression detection for 50+ concurrent users.
"""

import asyncio
import logging
import time
import psutil
import statistics
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import json
import threading
import gc
import aiohttp
import numpy as np

from ..component import LightRAGComponent
from ..config.settings import LightRAGConfig
from .performance_benchmark import PerformanceBenchmark, MemoryMonitor, LoadTestResult
from ..utils.logging import setup_logger


@dataclass
class ScalabilityTestResult:
    """Results from scalability testing."""
    test_name: str
    user_levels: List[int]
    results_by_level: Dict[int, LoadTestResult]
    breaking_point: Optional[int]  # User level where system starts failing
    scalability_metrics: Dict[str, Any]
    recommendations: List[str]


@dataclass
class StressTestResult:
    """Results from stress testing."""
    test_name: str
    duration_minutes: int
    max_concurrent_users: int
    load_progression: List[LoadTestResult]
    system_stability: Dict[str, Any]
    resource_exhaustion_point: Optional[Dict[str, Any]]
    recovery_metrics: Dict[str, Any]


@dataclass
class EnduranceTestResult:
    """Results from endurance testing."""
    test_name: str
    duration_hours: float
    constant_load_users: int
    performance_over_time: List[Dict[str, Any]]
    memory_leak_analysis: Dict[str, Any]
    performance_degradation: Dict[str, Any]
    stability_score: float


class LoadTestSuite:
    """
    Enhanced load testing suite for LightRAG integration.
    
    This class provides comprehensive load testing capabilities including
    concurrent user simulation up to 50+ users, stress testing, scalability
    analysis, and performance regression detection.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the load test suite.
        
        Args:
            config: Optional LightRAG configuration
            output_dir: Directory for test outputs and reports
        """
        self.config = config or LightRAGConfig.from_env()
        self.output_dir = Path(output_dir or "load_test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("load_test_suite",
                                 log_file=str(self.output_dir / "load_tests.log"))
        
        self.performance_benchmark = PerformanceBenchmark(config)
        self.memory_monitor = MemoryMonitor()
        
        # Enhanced thresholds for load testing
        self.load_test_thresholds = {
            "max_response_time_50_users": 15.0,  # seconds
            "max_response_time_100_users": 30.0,  # seconds
            "min_success_rate": 0.95,  # 95%
            "max_memory_usage_per_user": 50,  # MB per user
            "max_cpu_usage": 80,  # percent
            "max_error_rate": 0.05  # 5%
        }
        
        # Test question sets for different scenarios
        self.test_questions = self._create_test_question_sets()
    
    def _create_test_question_sets(self) -> Dict[str, List[str]]:
        """Create different sets of test questions for various scenarios."""
        return {
            "basic": [
                "What is clinical metabolomics?",
                "What are metabolites?",
                "How is NMR used in metabolomics?",
                "What are biomarkers?",
                "What is mass spectrometry?"
            ],
            "complex": [
                "How can metabolomics be used for disease diagnosis?",
                "What are the challenges in clinical metabolomics implementation?",
                "Explain the workflow of a typical metabolomics study.",
                "What is the role of metabolomics in personalized medicine?",
                "How do you validate metabolomic biomarkers?"
            ],
            "technical": [
                "What are the differences between targeted and untargeted metabolomics?",
                "How do you handle data preprocessing in metabolomics?",
                "What statistical methods are used in metabolomics analysis?",
                "How do you ensure reproducibility in metabolomics studies?",
                "What are the quality control measures in metabolomics?"
            ],
            "mixed": [
                "What is clinical metabolomics?",
                "How can metabolomics be used for disease diagnosis?",
                "What are the differences between targeted and untargeted metabolomics?",
                "What are metabolites?",
                "What are the challenges in clinical metabolomics implementation?",
                "How is NMR used in metabolomics?",
                "What statistical methods are used in metabolomics analysis?",
                "What are biomarkers?",
                "What is the role of metabolomics in personalized medicine?",
                "What is mass spectrometry?"
            ]
        }
    
    async def run_concurrent_user_test(self, 
                                     component: LightRAGComponent,
                                     concurrent_users: int,
                                     requests_per_user: int = 10,
                                     test_duration_minutes: Optional[int] = None,
                                     question_set: str = "mixed") -> LoadTestResult:
        """
        Run concurrent user test with specified parameters.
        
        Args:
            component: LightRAG component to test
            concurrent_users: Number of concurrent users
            requests_per_user: Requests per user (ignored if test_duration_minutes is set)
            test_duration_minutes: Run for fixed duration instead of fixed requests
            question_set: Question set to use ("basic", "complex", "technical", "mixed")
            
        Returns:
            LoadTestResult with detailed metrics
        """
        self.logger.info(
            f"Starting concurrent user test: {concurrent_users} users, "
            f"{'duration=' + str(test_duration_minutes) + 'min' if test_duration_minutes else 'requests=' + str(requests_per_user)}"
        )
        
        questions = self.test_questions.get(question_set, self.test_questions["mixed"])
        
        if test_duration_minutes:
            return await self._run_duration_based_test(
                component, concurrent_users, test_duration_minutes, questions
            )
        else:
            return await self.performance_benchmark.run_load_test(
                component=component,
                questions=questions,
                concurrent_users=concurrent_users,
                requests_per_user=requests_per_user,
                ramp_up_time=min(10.0, concurrent_users * 0.2)  # Scale ramp-up time
            )
    
    async def _run_duration_based_test(self, 
                                     component: LightRAGComponent,
                                     concurrent_users: int,
                                     duration_minutes: int,
                                     questions: List[str]) -> LoadTestResult:
        """Run load test for a fixed duration."""
        self.logger.info(f"Running duration-based test: {duration_minutes} minutes")
        
        results = []
        errors = []
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        # Start system monitoring
        self.memory_monitor.start_monitoring()
        
        async def user_simulation(user_id: int):
            """Simulate a single user for the duration."""
            user_results = []
            request_count = 0
            
            # Stagger user start times
            await asyncio.sleep((user_id / concurrent_users) * 10)
            
            while datetime.now() < end_time:
                question = random.choice(questions)
                request_start = time.time()
                
                try:
                    response = await component.query(question)
                    request_end = time.time()
                    
                    user_results.append({
                        "user_id": user_id,
                        "request_id": request_count,
                        "response_time": request_end - request_start,
                        "success": True,
                        "error": None,
                        "timestamp": datetime.now()
                    })
                    
                except Exception as e:
                    request_end = time.time()
                    error_msg = f"User {user_id}, Request {request_count}: {str(e)}"
                    errors.append(error_msg)
                    
                    user_results.append({
                        "user_id": user_id,
                        "request_id": request_count,
                        "response_time": request_end - request_start,
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now()
                    })
                
                request_count += 1
                
                # Brief pause between requests
                await asyncio.sleep(random.uniform(0.5, 2.0))
            
            return user_results
        
        # Run concurrent user simulations
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for user_result in user_results:
            if isinstance(user_result, Exception):
                errors.append(f"User simulation failed: {str(user_result)}")
            else:
                results.extend(user_result)
        
        actual_end_time = datetime.now()
        total_duration = (actual_end_time - start_time).total_seconds()
        
        # Stop monitoring
        memory_stats = self.memory_monitor.stop_monitoring()
        
        # Calculate metrics
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        response_times = [r["response_time"] for r in successful_results]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            p95_response_time = p99_response_time = 0
        
        requests_per_second = len(successful_results) / total_duration if total_duration > 0 else 0
        
        return LoadTestResult(
            test_name=f"duration_test_{concurrent_users}users_{duration_minutes}min",
            concurrent_users=concurrent_users,
            total_requests=len(results),
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
    
    async def run_scalability_test(self, 
                                 component: LightRAGComponent,
                                 max_users: int = 100,
                                 step_size: int = 10,
                                 requests_per_user: int = 5) -> ScalabilityTestResult:
        """
        Run scalability test to find system limits.
        
        Args:
            component: LightRAG component to test
            max_users: Maximum number of users to test
            step_size: Increment between user levels
            requests_per_user: Requests per user at each level
            
        Returns:
            ScalabilityTestResult with analysis
        """
        self.logger.info(f"Starting scalability test: up to {max_users} users")
        
        user_levels = list(range(step_size, max_users + 1, step_size))
        results_by_level = {}
        breaking_point = None
        
        for user_count in user_levels:
            self.logger.info(f"Testing scalability with {user_count} users")
            
            try:
                result = await self.run_concurrent_user_test(
                    component=component,
                    concurrent_users=user_count,
                    requests_per_user=requests_per_user,
                    question_set="mixed"
                )
                
                results_by_level[user_count] = result
                
                # Check if system is starting to fail
                success_rate = result.successful_requests / result.total_requests
                avg_response_time = result.average_response_time
                
                if (success_rate < self.load_test_thresholds["min_success_rate"] or
                    avg_response_time > self.load_test_thresholds["max_response_time_50_users"]):
                    
                    if breaking_point is None:
                        breaking_point = user_count
                        self.logger.warning(f"System performance degraded at {user_count} users")
                
                # Brief pause between tests
                await asyncio.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Scalability test failed at {user_count} users: {str(e)}")
                if breaking_point is None:
                    breaking_point = user_count
                break
        
        # Analyze scalability metrics
        scalability_metrics = self._analyze_scalability_metrics(results_by_level)
        recommendations = self._generate_scalability_recommendations(
            results_by_level, breaking_point, scalability_metrics
        )
        
        return ScalabilityTestResult(
            test_name=f"scalability_test_up_to_{max_users}users",
            user_levels=user_levels,
            results_by_level=results_by_level,
            breaking_point=breaking_point,
            scalability_metrics=scalability_metrics,
            recommendations=recommendations
        )
    
    def _analyze_scalability_metrics(self, 
                                   results_by_level: Dict[int, LoadTestResult]) -> Dict[str, Any]:
        """Analyze scalability metrics from test results."""
        if not results_by_level:
            return {}
        
        user_counts = sorted(results_by_level.keys())
        response_times = [results_by_level[u].average_response_time for u in user_counts]
        throughputs = [results_by_level[u].requests_per_second for u in user_counts]
        success_rates = [results_by_level[u].successful_requests / results_by_level[u].total_requests for u in user_counts]
        memory_usage = [results_by_level[u].memory_usage_mb.get("peak", 0) for u in user_counts]
        
        # Calculate scalability coefficients
        response_time_growth = self._calculate_growth_rate(user_counts, response_times)
        throughput_efficiency = self._calculate_throughput_efficiency(user_counts, throughputs)
        
        return {
            "response_time_growth_rate": response_time_growth,
            "throughput_efficiency": throughput_efficiency,
            "max_stable_users": self._find_max_stable_users(results_by_level),
            "linear_scalability_limit": self._find_linear_scalability_limit(user_counts, throughputs),
            "memory_per_user": self._calculate_memory_per_user(user_counts, memory_usage),
            "performance_degradation_point": self._find_performance_degradation_point(results_by_level)
        }
    
    def _calculate_growth_rate(self, x_values: List[int], y_values: List[float]) -> float:
        """Calculate growth rate using linear regression."""
        if len(x_values) < 2:
            return 0.0
        
        try:
            # Simple linear regression
            n = len(x_values)
            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except:
            return 0.0
    
    def _calculate_throughput_efficiency(self, user_counts: List[int], 
                                       throughputs: List[float]) -> float:
        """Calculate throughput efficiency (how well throughput scales with users)."""
        if len(user_counts) < 2:
            return 1.0
        
        # Ideal throughput would scale linearly with users
        # Efficiency = actual_throughput / ideal_throughput
        base_throughput_per_user = throughputs[0] / user_counts[0] if user_counts[0] > 0 else 0
        
        efficiencies = []
        for i, (users, throughput) in enumerate(zip(user_counts, throughputs)):
            if users > 0 and base_throughput_per_user > 0:
                ideal_throughput = base_throughput_per_user * users
                efficiency = throughput / ideal_throughput if ideal_throughput > 0 else 0
                efficiencies.append(efficiency)
        
        return statistics.mean(efficiencies) if efficiencies else 1.0
    
    def _find_max_stable_users(self, results_by_level: Dict[int, LoadTestResult]) -> int:
        """Find maximum number of users with stable performance."""
        stable_users = 0
        
        for user_count in sorted(results_by_level.keys()):
            result = results_by_level[user_count]
            success_rate = result.successful_requests / result.total_requests
            
            if (success_rate >= self.load_test_thresholds["min_success_rate"] and
                result.average_response_time <= self.load_test_thresholds["max_response_time_50_users"]):
                stable_users = user_count
            else:
                break
        
        return stable_users
    
    def _find_linear_scalability_limit(self, user_counts: List[int], 
                                     throughputs: List[float]) -> int:
        """Find the point where throughput stops scaling linearly."""
        if len(user_counts) < 3:
            return user_counts[-1] if user_counts else 0
        
        # Look for the point where throughput growth significantly slows
        for i in range(2, len(user_counts)):
            prev_growth = (throughputs[i-1] - throughputs[i-2]) / (user_counts[i-1] - user_counts[i-2])
            curr_growth = (throughputs[i] - throughputs[i-1]) / (user_counts[i] - user_counts[i-1])
            
            # If growth rate drops by more than 50%, consider this the limit
            if curr_growth < prev_growth * 0.5:
                return user_counts[i-1]
        
        return user_counts[-1]
    
    def _calculate_memory_per_user(self, user_counts: List[int], 
                                 memory_usage: List[float]) -> float:
        """Calculate average memory usage per user."""
        if not user_counts or not memory_usage:
            return 0.0
        
        memory_per_user_values = []
        for users, memory in zip(user_counts, memory_usage):
            if users > 0:
                memory_per_user_values.append(memory / users)
        
        return statistics.mean(memory_per_user_values) if memory_per_user_values else 0.0
    
    def _find_performance_degradation_point(self, 
                                          results_by_level: Dict[int, LoadTestResult]) -> Optional[int]:
        """Find the point where performance starts to degrade significantly."""
        user_counts = sorted(results_by_level.keys())
        
        if len(user_counts) < 2:
            return None
        
        baseline_response_time = results_by_level[user_counts[0]].average_response_time
        
        for user_count in user_counts[1:]:
            result = results_by_level[user_count]
            
            # If response time increases by more than 100% from baseline
            if result.average_response_time > baseline_response_time * 2:
                return user_count
        
        return None    

    def _generate_scalability_recommendations(self, 
                                            results_by_level: Dict[int, LoadTestResult],
                                            breaking_point: Optional[int],
                                            scalability_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on scalability test results."""
        recommendations = []
        
        max_stable_users = scalability_metrics.get("max_stable_users", 0)
        throughput_efficiency = scalability_metrics.get("throughput_efficiency", 1.0)
        
        if breaking_point:
            recommendations.append(f"System performance degrades at {breaking_point} concurrent users")
            recommendations.append(f"Consider horizontal scaling beyond {max_stable_users} users")
        
        if throughput_efficiency < 0.7:
            recommendations.append("Poor throughput efficiency detected - investigate bottlenecks")
            recommendations.append("Consider optimizing database queries and caching strategies")
        
        if scalability_metrics.get("memory_per_user", 0) > self.load_test_thresholds["max_memory_usage_per_user"]:
            recommendations.append("High memory usage per user - optimize memory management")
        
        if max_stable_users < 50:
            recommendations.append("System cannot handle 50+ concurrent users stably")
            recommendations.append("Performance optimization required before production deployment")
        
        return recommendations
    
    async def run_stress_test(self, 
                            component: LightRAGComponent,
                            duration_minutes: int = 30,
                            max_concurrent_users: int = 100,
                            ramp_up_minutes: int = 10) -> StressTestResult:
        """
        Run stress test with gradually increasing load.
        
        Args:
            component: LightRAG component to test
            duration_minutes: Total test duration
            max_concurrent_users: Maximum concurrent users to reach
            ramp_up_minutes: Time to ramp up to max users
            
        Returns:
            StressTestResult with comprehensive analysis
        """
        self.logger.info(
            f"Starting stress test: {max_concurrent_users} users over {duration_minutes} minutes"
        )
        
        # Calculate load progression
        user_levels = self._calculate_stress_test_progression(
            max_concurrent_users, duration_minutes, ramp_up_minutes
        )
        
        load_progression = []
        resource_exhaustion_point = None
        system_stability = {"stable_periods": 0, "unstable_periods": 0}
        
        for i, (user_count, test_duration) in enumerate(user_levels):
            self.logger.info(f"Stress test phase {i+1}: {user_count} users for {test_duration:.1f} minutes")
            
            try:
                result = await self.run_concurrent_user_test(
                    component=component,
                    concurrent_users=user_count,
                    test_duration_minutes=test_duration,
                    question_set="mixed"
                )
                
                load_progression.append(result)
                
                # Analyze system stability
                success_rate = result.successful_requests / result.total_requests
                if success_rate >= 0.95 and result.average_response_time <= 10.0:
                    system_stability["stable_periods"] += 1
                else:
                    system_stability["unstable_periods"] += 1
                
                # Check for resource exhaustion
                if (success_rate < 0.8 or 
                    result.memory_usage_mb.get("peak", 0) > 4096 or
                    result.average_response_time > 30.0):
                    
                    if resource_exhaustion_point is None:
                        resource_exhaustion_point = {
                            "user_count": user_count,
                            "success_rate": success_rate,
                            "memory_usage_mb": result.memory_usage_mb.get("peak", 0),
                            "response_time": result.average_response_time
                        }
                
                # Brief recovery period between phases
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Stress test failed at {user_count} users: {str(e)}")
                break
        
        # Test recovery after stress
        recovery_metrics = await self._test_system_recovery(component)
        
        return StressTestResult(
            test_name=f"stress_test_{max_concurrent_users}users_{duration_minutes}min",
            duration_minutes=duration_minutes,
            max_concurrent_users=max_concurrent_users,
            load_progression=load_progression,
            system_stability=system_stability,
            resource_exhaustion_point=resource_exhaustion_point,
            recovery_metrics=recovery_metrics
        )
    
    def _calculate_stress_test_progression(self, 
                                         max_users: int,
                                         total_duration: int,
                                         ramp_up_duration: int) -> List[Tuple[int, float]]:
        """Calculate stress test load progression."""
        phases = []
        
        # Ramp-up phase
        ramp_steps = 5
        for i in range(ramp_steps):
            user_count = int((i + 1) * max_users / ramp_steps)
            phase_duration = ramp_up_duration / ramp_steps
            phases.append((user_count, phase_duration))
        
        # Sustained load phase
        sustained_duration = total_duration - ramp_up_duration
        if sustained_duration > 0:
            phases.append((max_users, sustained_duration))
        
        return phases
    
    async def _test_system_recovery(self, component: LightRAGComponent) -> Dict[str, Any]:
        """Test system recovery after stress test."""
        self.logger.info("Testing system recovery after stress")
        
        # Wait for system to settle
        await asyncio.sleep(60)
        
        # Run a simple test to check recovery
        try:
            recovery_start = time.time()
            response = await component.query("What is clinical metabolomics?")
            recovery_time = time.time() - recovery_start
            
            return {
                "recovery_successful": True,
                "recovery_response_time": recovery_time,
                "system_responsive": recovery_time < 10.0
            }
            
        except Exception as e:
            return {
                "recovery_successful": False,
                "error": str(e),
                "system_responsive": False
            }
    
    async def run_endurance_test(self, 
                               component: LightRAGComponent,
                               duration_hours: float = 2.0,
                               constant_load_users: int = 20,
                               sampling_interval_minutes: int = 10) -> EnduranceTestResult:
        """
        Run endurance test to detect memory leaks and performance degradation.
        
        Args:
            component: LightRAG component to test
            duration_hours: Test duration in hours
            constant_load_users: Constant number of users
            sampling_interval_minutes: Interval for performance sampling
            
        Returns:
            EnduranceTestResult with long-term analysis
        """
        self.logger.info(
            f"Starting endurance test: {constant_load_users} users for {duration_hours} hours"
        )
        
        performance_over_time = []
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        sample_count = 0
        
        while datetime.now() < end_time:
            sample_start = datetime.now()
            
            # Run a short load test sample
            try:
                result = await self.run_concurrent_user_test(
                    component=component,
                    concurrent_users=constant_load_users,
                    test_duration_minutes=sampling_interval_minutes,
                    question_set="mixed"
                )
                
                performance_sample = {
                    "sample_number": sample_count,
                    "timestamp": sample_start.isoformat(),
                    "elapsed_hours": (sample_start - start_time).total_seconds() / 3600,
                    "response_time": result.average_response_time,
                    "success_rate": result.successful_requests / result.total_requests,
                    "memory_usage_mb": result.memory_usage_mb.get("peak", 0),
                    "requests_per_second": result.requests_per_second
                }
                
                performance_over_time.append(performance_sample)
                sample_count += 1
                
                self.logger.info(
                    f"Endurance sample {sample_count}: "
                    f"Response time: {result.average_response_time:.2f}s, "
                    f"Memory: {result.memory_usage_mb.get('peak', 0):.1f}MB"
                )
                
            except Exception as e:
                self.logger.error(f"Endurance test sample failed: {str(e)}")
                performance_sample = {
                    "sample_number": sample_count,
                    "timestamp": sample_start.isoformat(),
                    "elapsed_hours": (sample_start - start_time).total_seconds() / 3600,
                    "error": str(e)
                }
                performance_over_time.append(performance_sample)
                sample_count += 1
        
        # Analyze results
        memory_leak_analysis = self._analyze_memory_leaks(performance_over_time)
        performance_degradation = self._analyze_performance_degradation(performance_over_time)
        stability_score = self._calculate_stability_score(performance_over_time)
        
        return EnduranceTestResult(
            test_name=f"endurance_test_{constant_load_users}users_{duration_hours}h",
            duration_hours=duration_hours,
            constant_load_users=constant_load_users,
            performance_over_time=performance_over_time,
            memory_leak_analysis=memory_leak_analysis,
            performance_degradation=performance_degradation,
            stability_score=stability_score
        )
    
    def _analyze_memory_leaks(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze memory usage over time to detect leaks."""
        memory_values = []
        time_values = []
        
        for sample in performance_data:
            if "memory_usage_mb" in sample and "elapsed_hours" in sample:
                memory_values.append(sample["memory_usage_mb"])
                time_values.append(sample["elapsed_hours"])
        
        if len(memory_values) < 3:
            return {"insufficient_data": True}
        
        # Calculate memory growth rate
        memory_growth_rate = self._calculate_growth_rate(time_values, memory_values)
        
        # Detect significant memory increase
        initial_memory = statistics.mean(memory_values[:3])
        final_memory = statistics.mean(memory_values[-3:])
        memory_increase_percent = ((final_memory - initial_memory) / initial_memory) * 100
        
        return {
            "memory_growth_rate_mb_per_hour": memory_growth_rate,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_percent": memory_increase_percent,
            "potential_memory_leak": memory_increase_percent > 20,  # >20% increase
            "memory_stability": memory_increase_percent < 10  # <10% is stable
        }
    
    def _analyze_performance_degradation(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance degradation over time."""
        response_times = []
        success_rates = []
        throughputs = []
        
        for sample in performance_data:
            if "response_time" in sample:
                response_times.append(sample["response_time"])
            if "success_rate" in sample:
                success_rates.append(sample["success_rate"])
            if "requests_per_second" in sample:
                throughputs.append(sample["requests_per_second"])
        
        if not response_times:
            return {"insufficient_data": True}
        
        # Calculate degradation metrics
        initial_response_time = statistics.mean(response_times[:3]) if len(response_times) >= 3 else response_times[0]
        final_response_time = statistics.mean(response_times[-3:]) if len(response_times) >= 3 else response_times[-1]
        
        response_time_degradation = ((final_response_time - initial_response_time) / initial_response_time) * 100
        
        return {
            "initial_response_time": initial_response_time,
            "final_response_time": final_response_time,
            "response_time_degradation_percent": response_time_degradation,
            "significant_degradation": response_time_degradation > 50,  # >50% slower
            "performance_stable": response_time_degradation < 20  # <20% change is stable
        }
    
    def _calculate_stability_score(self, performance_data: List[Dict[str, Any]]) -> float:
        """Calculate overall stability score (0-1)."""
        if not performance_data:
            return 0.0
        
        stability_factors = []
        
        # Success rate stability
        success_rates = [s.get("success_rate", 0) for s in performance_data if "success_rate" in s]
        if success_rates:
            avg_success_rate = statistics.mean(success_rates)
            success_rate_stability = min(1.0, avg_success_rate / 0.95)  # Target 95%
            stability_factors.append(success_rate_stability)
        
        # Response time stability
        response_times = [s.get("response_time", 0) for s in performance_data if "response_time" in s]
        if response_times:
            response_time_cv = statistics.stdev(response_times) / statistics.mean(response_times)
            response_time_stability = max(0.0, 1.0 - response_time_cv)  # Lower CV is better
            stability_factors.append(response_time_stability)
        
        # Error rate stability
        error_count = sum(1 for s in performance_data if "error" in s)
        error_rate = error_count / len(performance_data)
        error_stability = max(0.0, 1.0 - error_rate * 10)  # Penalize errors heavily
        stability_factors.append(error_stability)
        
        return statistics.mean(stability_factors) if stability_factors else 0.0
    
    async def run_comprehensive_load_tests(self, 
                                         component: LightRAGComponent) -> Dict[str, Any]:
        """
        Run comprehensive load testing suite.
        
        Args:
            component: LightRAG component to test
            
        Returns:
            Dictionary with all load test results
        """
        self.logger.info("Starting comprehensive load testing suite")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "concurrent_user_tests": [],
            "scalability_test": None,
            "stress_test": None,
            "endurance_test": None,
            "summary": {},
            "recommendations": []
        }
        
        try:
            # 1. Concurrent user tests at different levels
            self.logger.info("Running concurrent user tests")
            user_levels = [10, 25, 50, 75, 100]
            
            for user_count in user_levels:
                try:
                    result = await self.run_concurrent_user_test(
                        component=component,
                        concurrent_users=user_count,
                        requests_per_user=10,
                        question_set="mixed"
                    )
                    results["concurrent_user_tests"].append({
                        "user_count": user_count,
                        "result": asdict(result)
                    })
                    
                    self.logger.info(f"Completed {user_count} user test")
                    
                except Exception as e:
                    self.logger.error(f"Failed {user_count} user test: {str(e)}")
                    break
            
            # 2. Scalability test
            self.logger.info("Running scalability test")
            try:
                scalability_result = await self.run_scalability_test(
                    component=component,
                    max_users=100,
                    step_size=20,
                    requests_per_user=5
                )
                results["scalability_test"] = asdict(scalability_result)
            except Exception as e:
                self.logger.error(f"Scalability test failed: {str(e)}")
            
            # 3. Stress test
            self.logger.info("Running stress test")
            try:
                stress_result = await self.run_stress_test(
                    component=component,
                    duration_minutes=20,
                    max_concurrent_users=75,
                    ramp_up_minutes=5
                )
                results["stress_test"] = asdict(stress_result)
            except Exception as e:
                self.logger.error(f"Stress test failed: {str(e)}")
            
            # 4. Short endurance test (for comprehensive suite)
            self.logger.info("Running short endurance test")
            try:
                endurance_result = await self.run_endurance_test(
                    component=component,
                    duration_hours=0.5,  # 30 minutes for comprehensive suite
                    constant_load_users=25,
                    sampling_interval_minutes=5
                )
                results["endurance_test"] = asdict(endurance_result)
            except Exception as e:
                self.logger.error(f"Endurance test failed: {str(e)}")
            
            # Generate summary and recommendations
            results["summary"] = self._generate_load_test_summary(results)
            results["recommendations"] = self._generate_load_test_recommendations(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Comprehensive load tests failed: {str(e)}")
            results["error"] = str(e)
            return results
    
    def _generate_load_test_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of load test results."""
        summary = {
            "max_stable_users": 0,
            "peak_throughput": 0.0,
            "average_response_time_50_users": None,
            "system_scalability": "unknown",
            "stress_test_passed": False,
            "endurance_test_passed": False
        }
        
        # Analyze concurrent user tests
        for test in results.get("concurrent_user_tests", []):
            user_count = test["user_count"]
            result = test["result"]
            success_rate = result["successful_requests"] / result["total_requests"]
            
            if success_rate >= 0.95 and result["average_response_time"] <= 10.0:
                summary["max_stable_users"] = max(summary["max_stable_users"], user_count)
            
            if result["requests_per_second"] > summary["peak_throughput"]:
                summary["peak_throughput"] = result["requests_per_second"]
            
            if user_count == 50:
                summary["average_response_time_50_users"] = result["average_response_time"]
        
        # Analyze scalability test
        if results.get("scalability_test"):
            scalability = results["scalability_test"]
            if scalability["breaking_point"]:
                if scalability["breaking_point"] >= 50:
                    summary["system_scalability"] = "good"
                elif scalability["breaking_point"] >= 25:
                    summary["system_scalability"] = "moderate"
                else:
                    summary["system_scalability"] = "poor"
            else:
                summary["system_scalability"] = "excellent"
        
        # Analyze stress test
        if results.get("stress_test"):
            stress = results["stress_test"]
            if not stress.get("resource_exhaustion_point"):
                summary["stress_test_passed"] = True
        
        # Analyze endurance test
        if results.get("endurance_test"):
            endurance = results["endurance_test"]
            if endurance["stability_score"] >= 0.8:
                summary["endurance_test_passed"] = True
        
        return summary
    
    def _generate_load_test_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on load test results."""
        recommendations = []
        summary = results.get("summary", {})
        
        max_stable_users = summary.get("max_stable_users", 0)
        
        if max_stable_users < 50:
            recommendations.append(f"System only supports {max_stable_users} concurrent users stably")
            recommendations.append("Performance optimization required for production deployment")
            recommendations.append("Consider horizontal scaling or caching improvements")
        elif max_stable_users >= 100:
            recommendations.append("Excellent scalability - system handles 100+ concurrent users")
        
        if summary.get("average_response_time_50_users", 0) > 10:
            recommendations.append("Response times too high at 50 users - optimize query processing")
        
        if summary.get("system_scalability") == "poor":
            recommendations.append("Poor scalability detected - investigate bottlenecks")
            recommendations.append("Consider database optimization and connection pooling")
        
        if not summary.get("stress_test_passed"):
            recommendations.append("System failed stress test - improve error handling and resource management")
        
        if not summary.get("endurance_test_passed"):
            recommendations.append("Stability issues detected in endurance test")
            recommendations.append("Investigate potential memory leaks and performance degradation")
        
        return recommendations
    
    def generate_load_test_report(self, results: Dict[str, Any],
                                output_file: Optional[str] = None) -> str:
        """Generate comprehensive load test report."""
        report_lines = [
            "=" * 80,
            "LIGHTRAG LOAD TESTING REPORT",
            "=" * 80,
            f"Timestamp: {results['timestamp']}",
            ""
        ]
        
        # Summary
        summary = results.get("summary", {})
        report_lines.extend([
            "SUMMARY:",
            f"  Max Stable Users: {summary.get('max_stable_users', 0)}",
            f"  Peak Throughput: {summary.get('peak_throughput', 0):.2f} req/s",
            f"  System Scalability: {summary.get('system_scalability', 'unknown').title()}",
            f"  Stress Test: {'✅ PASSED' if summary.get('stress_test_passed') else '❌ FAILED'}",
            f"  Endurance Test: {'✅ PASSED' if summary.get('endurance_test_passed') else '❌ FAILED'}",
            ""
        ])
        
        # Concurrent user test results
        if results.get("concurrent_user_tests"):
            report_lines.extend([
                "CONCURRENT USER TEST RESULTS:",
                "-" * 40
            ])
            
            for test in results["concurrent_user_tests"]:
                user_count = test["user_count"]
                result = test["result"]
                success_rate = result["successful_requests"] / result["total_requests"]
                
                status = "✅ STABLE" if success_rate >= 0.95 and result["average_response_time"] <= 10.0 else "❌ UNSTABLE"
                
                report_lines.extend([
                    f"{user_count} Users: {status}",
                    f"  Success Rate: {success_rate:.1%}",
                    f"  Avg Response Time: {result['average_response_time']:.3f}s",
                    f"  Throughput: {result['requests_per_second']:.2f} req/s",
                    f"  95th Percentile: {result['percentile_95_response_time']:.3f}s",
                    ""
                ])
        
        # Recommendations
        if results.get("recommendations"):
            report_lines.extend([
                "RECOMMENDATIONS:",
                "-" * 20
            ])
            
            for recommendation in results["recommendations"]:
                report_lines.append(f"  • {recommendation}")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report_content)
            self.logger.info(f"Load test report saved to {output_file}")
        
        return report_content


# Convenience function for running load tests
async def run_load_tests(config: Optional[LightRAGConfig] = None,
                       output_dir: Optional[str] = None,
                       max_users: int = 100) -> Dict[str, Any]:
    """
    Run comprehensive load testing suite.
    
    Args:
        config: Optional LightRAG configuration
        output_dir: Optional output directory for results
        max_users: Maximum number of concurrent users to test
        
    Returns:
        Dictionary with load test results
    """
    load_test_suite = LoadTestSuite(config=config, output_dir=output_dir)
    
    # Create and initialize component
    component = LightRAGComponent(config or LightRAGConfig.from_env())
    await component.initialize()
    
    try:
        results = await load_test_suite.run_comprehensive_load_tests(component)
        return results
    finally:
        await component.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LightRAG Load Testing Suite")
    parser.add_argument("--output-dir", default="load_test_results",
                       help="Output directory for test results")
    parser.add_argument("--max-users", type=int, default=100,
                       help="Maximum number of concurrent users to test")
    parser.add_argument("--test-type", choices=["concurrent", "scalability", "stress", "endurance", "comprehensive"],
                       default="comprehensive", help="Type of load test to run")
    parser.add_argument("--save-results", action="store_true",
                       help="Save detailed results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        try:
            # Run load tests
            results = await run_load_tests(
                output_dir=args.output_dir,
                max_users=args.max_users
            )
            
            # Generate report
            load_test_suite = LoadTestSuite(output_dir=args.output_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            report = load_test_suite.generate_load_test_report(
                results,
                str(load_test_suite.output_dir / f"load_test_report_{timestamp}.txt")
            )
            print(report)
            
            # Save results if requested
            if args.save_results:
                with open(load_test_suite.output_dir / f"load_test_results_{timestamp}.json", 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            # Exit with appropriate code
            summary = results.get("summary", {})
            if (summary.get("max_stable_users", 0) >= 50 and
                summary.get("stress_test_passed", False)):
                print("\n🎉 LOAD TESTS PASSED!")
                exit(0)
            else:
                print("\n💥 LOAD TESTS FAILED!")
                exit(1)
                
        except Exception as e:
            print(f"Load tests failed: {str(e)}")
            exit(1)
    
    asyncio.run(main())