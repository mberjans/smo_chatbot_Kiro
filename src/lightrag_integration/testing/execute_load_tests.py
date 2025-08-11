"""
Load Testing Execution Script

This script executes comprehensive load testing to meet the requirements
of task 15.2, including testing with 50+ concurrent users, stress testing,
and performance regression detection.
"""

import asyncio
import logging
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .load_test_suite import LoadTestSuite
from .performance_regression_detector import PerformanceRegressionDetector, detect_performance_regressions
from ..component import LightRAGComponent
from ..config.settings import LightRAGConfig
from ..utils.logging import setup_logger


class LoadTestExecutor:
    """
    Load test executor that runs comprehensive performance and load testing
    to validate system capabilities under various load conditions.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the load test executor.
        
        Args:
            config: Optional LightRAG configuration
            output_dir: Directory for test outputs and reports
        """
        self.config = config or LightRAGConfig.from_env()
        self.output_dir = Path(output_dir or "load_test_execution_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("load_test_executor",
                                 log_file=str(self.output_dir / "load_test_execution.log"))
        
        self.load_test_suite = LoadTestSuite(self.config, str(self.output_dir / "load_tests"))
        self.regression_detector = PerformanceRegressionDetector(
            baseline_dir=str(self.output_dir / "baselines"),
            output_dir=str(self.output_dir / "regression_analysis")
        )
    
    async def execute_comprehensive_load_tests(self, 
                                             max_concurrent_users: int = 75,
                                             include_endurance_tests: bool = False) -> Dict[str, Any]:
        """
        Execute comprehensive load testing suite.
        
        Args:
            max_concurrent_users: Maximum concurrent users to test (must be >= 50)
            include_endurance_tests: Whether to include long-running endurance tests
            
        Returns:
            Comprehensive load test results
        """
        if max_concurrent_users < 50:
            raise ValueError("max_concurrent_users must be at least 50 to meet requirements")
        
        test_start = datetime.now()
        self.logger.info(f"Starting comprehensive load testing with up to {max_concurrent_users} users")
        
        results = {
            "timestamp": test_start.isoformat(),
            "max_concurrent_users_tested": max_concurrent_users,
            "test_results": {},
            "performance_metrics": {},
            "regression_analysis": {},
            "requirements_validation": {},
            "overall_success": False,
            "recommendations": []
        }
        
        try:
            # Initialize LightRAG component
            component = LightRAGComponent(self.config)
            await component.initialize()
            
            # 1. Concurrent User Load Test (50+ users requirement)
            self.logger.info(f"Running concurrent user test with {max_concurrent_users} users")
            concurrent_test_result = await self.load_test_suite.run_concurrent_user_test(
                component=component,
                concurrent_users=max_concurrent_users,
                requests_per_user=10,
                question_set="mixed"
            )
            results["test_results"]["concurrent_user_test"] = {
                "result": concurrent_test_result.__dict__,
                "meets_50_user_requirement": max_concurrent_users >= 50,
                "success_rate": concurrent_test_result.successful_requests / concurrent_test_result.total_requests,
                "avg_response_time": concurrent_test_result.average_response_time,
                "p95_response_time": concurrent_test_result.percentile_95_response_time
            }
            
            # 2. Scalability Test
            self.logger.info("Running scalability test")
            scalability_result = await self.load_test_suite.run_scalability_test(
                component=component,
                max_users=max_concurrent_users,
                step_size=10,
                requests_per_user=5
            )
            results["test_results"]["scalability_test"] = {
                "result": scalability_result.__dict__,
                "breaking_point": scalability_result.breaking_point,
                "max_stable_users": scalability_result.scalability_metrics.get("max_stable_users", 0),
                "throughput_efficiency": scalability_result.scalability_metrics.get("throughput_efficiency", 0)
            }
            
            # 3. Stress Test
            self.logger.info("Running stress test")
            stress_result = await self.load_test_suite.run_stress_test(
                component=component,
                duration_minutes=20,
                max_concurrent_users=max_concurrent_users,
                ramp_up_minutes=5
            )
            results["test_results"]["stress_test"] = {
                "result": stress_result.__dict__,
                "system_stability": stress_result.system_stability,
                "resource_exhaustion_point": stress_result.resource_exhaustion_point,
                "recovery_successful": stress_result.recovery_metrics.get("recovery_successful", False)
            }
            
            # 4. Endurance Test (optional)
            if include_endurance_tests:
                self.logger.info("Running endurance test")
                endurance_result = await self.load_test_suite.run_endurance_test(
                    component=component,
                    duration_hours=1.0,
                    constant_load_users=min(25, max_concurrent_users // 3)
                )
                results["test_results"]["endurance_test"] = {
                    "result": endurance_result.__dict__,
                    "stability_score": endurance_result.stability_score,
                    "memory_leak_detected": endurance_result.memory_leak_analysis.get("leak_detected", False)
                }
            
            # 5. Extract Performance Metrics
            performance_metrics = self._extract_performance_metrics(results["test_results"])
            results["performance_metrics"] = performance_metrics
            
            # 6. Performance Regression Analysis
            self.logger.info("Running performance regression analysis")
            test_run_id = f"load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            regression_report = self.regression_detector.detect_regressions(
                performance_metrics, test_run_id
            )
            results["regression_analysis"] = {
                "report": regression_report.__dict__,
                "regressions_detected": regression_report.overall_regression_detected,
                "regression_count": regression_report.regressions_detected,
                "recommendations": regression_report.recommendations
            }
            
            # 7. Validate Requirements
            requirements_validation = self._validate_load_test_requirements(
                results["test_results"], performance_metrics
            )
            results["requirements_validation"] = requirements_validation
            
            # 8. Overall Success Assessment
            overall_success = self._assess_overall_success(
                requirements_validation, regression_report
            )
            results["overall_success"] = overall_success
            
            # 9. Generate Recommendations
            recommendations = self._generate_load_test_recommendations(
                results["test_results"], performance_metrics, regression_report, overall_success
            )
            results["recommendations"] = recommendations
            
            # Cleanup
            await component.cleanup()
            
            total_duration = (datetime.now() - test_start).total_seconds()
            results["total_duration_seconds"] = total_duration
            
            self.logger.info(
                f"Comprehensive load testing completed: "
                f"{'SUCCESS' if overall_success else 'FAILED'} "
                f"in {total_duration:.2f}s"
            )
            
            # Save results
            await self._save_load_test_results(results)
            
            return results
            
        except Exception as e:
            error_msg = f"Load testing execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            results["error"] = error_msg
            results["total_duration_seconds"] = (datetime.now() - test_start).total_seconds()
            return results
    
    def _extract_performance_metrics(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key performance metrics from test results."""
        metrics = {}
        
        # Concurrent user test metrics
        if "concurrent_user_test" in test_results:
            concurrent_result = test_results["concurrent_user_test"]["result"]
            metrics.update({
                "concurrent_user_avg_response_time": concurrent_result["average_response_time"],
                "concurrent_user_p95_response_time": concurrent_result["percentile_95_response_time"],
                "concurrent_user_p99_response_time": concurrent_result["percentile_99_response_time"],
                "concurrent_user_throughput": concurrent_result["requests_per_second"],
                "concurrent_user_success_rate": concurrent_result["successful_requests"] / concurrent_result["total_requests"],
                "max_concurrent_users_tested": concurrent_result["concurrent_users"]
            })
            
            # Memory and CPU metrics
            if "memory_usage_mb" in concurrent_result:
                memory_stats = concurrent_result["memory_usage_mb"]
                metrics["max_memory_usage"] = memory_stats.get("peak", 0)
                metrics["avg_memory_usage"] = memory_stats.get("average", 0)
            
            if "cpu_usage_percent" in concurrent_result:
                cpu_stats = concurrent_result["cpu_usage_percent"]
                metrics["avg_cpu_usage"] = cpu_stats.get("average", 0)
                metrics["max_cpu_usage"] = cpu_stats.get("peak", 0)
        
        # Scalability test metrics
        if "scalability_test" in test_results:
            scalability_result = test_results["scalability_test"]
            metrics.update({
                "scalability_breaking_point": scalability_result["breaking_point"],
                "max_stable_users": scalability_result["max_stable_users"],
                "throughput_efficiency": scalability_result["throughput_efficiency"]
            })
        
        # Stress test metrics
        if "stress_test" in test_results:
            stress_result = test_results["stress_test"]
            stability = stress_result["system_stability"]
            metrics.update({
                "stress_test_stability_score": stability.get("stable_periods", 0) / max(
                    stability.get("stable_periods", 0) + stability.get("unstable_periods", 0), 1
                ),
                "stress_test_recovery_successful": stress_result["recovery_successful"]
            })
        
        # Endurance test metrics
        if "endurance_test" in test_results:
            endurance_result = test_results["endurance_test"]
            metrics.update({
                "endurance_stability_score": endurance_result["stability_score"],
                "memory_leak_detected": endurance_result["memory_leak_detected"]
            })
        
        return metrics
    
    def _validate_load_test_requirements(self, test_results: Dict[str, Any],
                                       performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that load test results meet the specified requirements."""
        validation = {
            "requirement_8_3_concurrent_users": False,  # Handle at least 50 concurrent users
            "requirement_8_5_success_rate": False,     # Success rate >= 95%
            "requirement_8_7_response_time": False,    # Response time <= 5s for 95% of requests
            "stress_testing_completed": False,
            "scalability_testing_completed": False,
            "regression_detection_functional": False
        }
        
        # Requirement 8.3: Handle at least 50 concurrent users
        max_users_tested = performance_metrics.get("max_concurrent_users_tested", 0)
        max_stable_users = performance_metrics.get("max_stable_users", 0)
        validation["requirement_8_3_concurrent_users"] = (
            max_users_tested >= 50 and max_stable_users >= 50
        )
        
        # Requirement 8.5: Success rate >= 95%
        success_rate = performance_metrics.get("concurrent_user_success_rate", 0)
        validation["requirement_8_5_success_rate"] = success_rate >= 0.95
        
        # Requirement 8.7: Response time <= 5 seconds for 95% of requests
        p95_response_time = performance_metrics.get("concurrent_user_p95_response_time", float('inf'))
        validation["requirement_8_7_response_time"] = p95_response_time <= 5.0
        
        # Stress testing completed
        validation["stress_testing_completed"] = "stress_test" in test_results
        
        # Scalability testing completed
        validation["scalability_testing_completed"] = "scalability_test" in test_results
        
        # Regression detection functional
        validation["regression_detection_functional"] = True  # Validated by successful execution
        
        return validation
    
    def _assess_overall_success(self, requirements_validation: Dict[str, Any],
                              regression_report) -> bool:
        """Assess overall success of load testing."""
        # All critical requirements must be met
        critical_requirements = [
            "requirement_8_3_concurrent_users",
            "requirement_8_5_success_rate", 
            "requirement_8_7_response_time",
            "stress_testing_completed",
            "scalability_testing_completed"
        ]
        
        requirements_met = all(
            requirements_validation.get(req, False) for req in critical_requirements
        )
        
        # No critical performance regressions
        no_critical_regressions = not any(
            result.is_regression and result.severity == "critical"
            for result in regression_report.regression_results
        )
        
        return requirements_met and no_critical_regressions
    
    def _generate_load_test_recommendations(self, test_results: Dict[str, Any],
                                          performance_metrics: Dict[str, Any],
                                          regression_report,
                                          overall_success: bool) -> List[str]:
        """Generate recommendations based on load test results."""
        recommendations = []
        
        if overall_success:
            recommendations.extend([
                "✅ All load testing requirements successfully met",
                "✅ System handles 50+ concurrent users with acceptable performance",
                "✅ No critical performance regressions detected",
                "System is ready for production deployment under expected load"
            ])
        else:
            recommendations.append("❌ Load testing requirements not fully met")
        
        # Specific performance recommendations
        max_stable_users = performance_metrics.get("max_stable_users", 0)
        if max_stable_users < 50:
            recommendations.append(f"⚠️ System only stable up to {max_stable_users} users - optimization required")
        
        success_rate = performance_metrics.get("concurrent_user_success_rate", 0)
        if success_rate < 0.95:
            recommendations.append(f"⚠️ Success rate ({success_rate:.1%}) below 95% requirement")
        
        p95_response_time = performance_metrics.get("concurrent_user_p95_response_time", 0)
        if p95_response_time > 5.0:
            recommendations.append(f"⚠️ P95 response time ({p95_response_time:.2f}s) exceeds 5s requirement")
        
        # Scalability recommendations
        breaking_point = performance_metrics.get("scalability_breaking_point")
        if breaking_point and breaking_point < 75:
            recommendations.append(f"⚠️ Performance degrades at {breaking_point} users - consider horizontal scaling")
        
        throughput_efficiency = performance_metrics.get("throughput_efficiency", 1.0)
        if throughput_efficiency < 0.7:
            recommendations.append("⚠️ Poor throughput scaling efficiency - investigate bottlenecks")
        
        # Memory and resource recommendations
        max_memory = performance_metrics.get("max_memory_usage", 0)
        if max_memory > 2048:  # 2GB threshold
            recommendations.append(f"⚠️ High memory usage ({max_memory:.0f}MB) - optimize memory management")
        
        # Regression-based recommendations
        if regression_report.overall_regression_detected:
            critical_regressions = [r for r in regression_report.regression_results 
                                  if r.is_regression and r.severity in ["critical", "severe"]]
            if critical_regressions:
                recommendations.append(f"❌ {len(critical_regressions)} critical/severe regressions detected")
                recommendations.append("Address performance regressions before deployment")
        
        # Stress test recommendations
        if "stress_test" in test_results:
            stress_result = test_results["stress_test"]
            if not stress_result["recovery_successful"]:
                recommendations.append("⚠️ System recovery after stress test failed - investigate stability")
        
        # General optimization recommendations
        if not overall_success:
            recommendations.extend([
                "Consider implementing caching strategies to improve response times",
                "Optimize database queries and connection pooling",
                "Implement request queuing and rate limiting for better load handling",
                "Monitor system resources and implement auto-scaling if needed"
            ])
        
        return recommendations
    
    async def _save_load_test_results(self, results: Dict[str, Any]) -> None:
        """Save load test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_file = self.output_dir / f"load_test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            import json
            json.dump(results, f, indent=2, default=str)
        
        # Save executive summary
        summary_file = self.output_dir / f"load_test_summary_{timestamp}.txt"
        summary_content = self._generate_load_test_summary(results)
        summary_file.write_text(summary_content)
        
        # Save requirements validation report
        validation_file = self.output_dir / f"requirements_validation_{timestamp}.txt"
        validation_content = self._generate_requirements_validation_report(results)
        validation_file.write_text(validation_content)
        
        self.logger.info(f"Load test results saved to {self.output_dir}")
    
    def _generate_load_test_summary(self, results: Dict[str, Any]) -> str:
        """Generate executive summary of load test results."""
        lines = [
            "LOAD TESTING EXECUTION SUMMARY",
            "=" * 35,
            f"Test Date: {results['timestamp']}",
            f"Duration: {results.get('total_duration_seconds', 0):.2f} seconds",
            f"Max Concurrent Users Tested: {results['max_concurrent_users_tested']}",
            f"Overall Status: {'✅ SUCCESS' if results['overall_success'] else '❌ FAILED'}",
            "",
            "PERFORMANCE METRICS:",
            "-" * 20
        ]
        
        metrics = results["performance_metrics"]
        lines.extend([
            f"  Average Response Time: {metrics.get('concurrent_user_avg_response_time', 0):.2f}s",
            f"  P95 Response Time: {metrics.get('concurrent_user_p95_response_time', 0):.2f}s",
            f"  Success Rate: {metrics.get('concurrent_user_success_rate', 0):.1%}",
            f"  Throughput: {metrics.get('concurrent_user_throughput', 0):.1f} req/s",
            f"  Max Stable Users: {metrics.get('max_stable_users', 0)}",
            f"  Memory Usage: {metrics.get('max_memory_usage', 0):.0f}MB",
            ""
        ])
        
        # Requirements validation
        validation = results["requirements_validation"]
        lines.extend([
            "REQUIREMENTS VALIDATION:",
            "-" * 25,
            f"  50+ Concurrent Users: {'✅ PASS' if validation.get('requirement_8_3_concurrent_users', False) else '❌ FAIL'}",
            f"  95% Success Rate: {'✅ PASS' if validation.get('requirement_8_5_success_rate', False) else '❌ FAIL'}",
            f"  5s P95 Response Time: {'✅ PASS' if validation.get('requirement_8_7_response_time', False) else '❌ FAIL'}",
            f"  Stress Testing: {'✅ COMPLETE' if validation.get('stress_testing_completed', False) else '❌ INCOMPLETE'}",
            f"  Scalability Testing: {'✅ COMPLETE' if validation.get('scalability_testing_completed', False) else '❌ INCOMPLETE'}",
            ""
        ])
        
        # Recommendations
        lines.extend([
            "KEY RECOMMENDATIONS:",
            "-" * 20
        ])
        
        for recommendation in results["recommendations"][:5]:  # Top 5 recommendations
            lines.append(f"  • {recommendation}")
        
        return "\n".join(lines)
    
    def _generate_requirements_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed requirements validation report."""
        lines = [
            "LOAD TESTING REQUIREMENTS VALIDATION REPORT",
            "=" * 45,
            f"Test Execution Date: {results['timestamp']}",
            f"Overall Validation Status: {'✅ ALL REQUIREMENTS MET' if results['overall_success'] else '❌ REQUIREMENTS NOT MET'}",
            "",
            "TASK 15.2 REQUIREMENTS VALIDATION:",
            "-" * 35
        ]
        
        validation = results["requirements_validation"]
        requirements = [
            ("requirement_8_3_concurrent_users", "Implement load testing for 50+ concurrent users"),
            ("requirement_8_5_success_rate", "Achieve 95% success rate under load"),
            ("requirement_8_7_response_time", "Maintain P95 response time <= 5 seconds"),
            ("stress_testing_completed", "Add stress testing for large document collections"),
            ("scalability_testing_completed", "Write scalability testing for system limits"),
            ("regression_detection_functional", "Create performance regression detection")
        ]
        
        for req_key, req_description in requirements:
            status = validation.get(req_key, False)
            status_icon = "✅" if status else "❌"
            lines.append(f"  {status_icon} {req_description}: {'PASS' if status else 'FAIL'}")
        
        lines.extend([
            "",
            "DETAILED PERFORMANCE ANALYSIS:",
            "-" * 32
        ])
        
        metrics = results["performance_metrics"]
        
        # Concurrent user analysis
        max_users = metrics.get("max_concurrent_users_tested", 0)
        stable_users = metrics.get("max_stable_users", 0)
        lines.extend([
            f"Concurrent User Testing:",
            f"  • Maximum users tested: {max_users}",
            f"  • Maximum stable users: {stable_users}",
            f"  • Meets 50+ user requirement: {'Yes' if stable_users >= 50 else 'No'}",
            ""
        ])
        
        # Performance analysis
        success_rate = metrics.get("concurrent_user_success_rate", 0)
        avg_response = metrics.get("concurrent_user_avg_response_time", 0)
        p95_response = metrics.get("concurrent_user_p95_response_time", 0)
        throughput = metrics.get("concurrent_user_throughput", 0)
        
        lines.extend([
            f"Performance Analysis:",
            f"  • Success rate: {success_rate:.1%} (requirement: ≥95%)",
            f"  • Average response time: {avg_response:.2f}s",
            f"  • P95 response time: {p95_response:.2f}s (requirement: ≤5s)",
            f"  • Throughput: {throughput:.1f} requests/second",
            ""
        ])
        
        # Scalability analysis
        breaking_point = metrics.get("scalability_breaking_point")
        efficiency = metrics.get("throughput_efficiency", 0)
        lines.extend([
            f"Scalability Analysis:",
            f"  • Performance breaking point: {breaking_point if breaking_point else 'Not reached'}",
            f"  • Throughput efficiency: {efficiency:.1%}",
            ""
        ])
        
        # Regression analysis
        regression_analysis = results["regression_analysis"]
        lines.extend([
            f"Regression Analysis:",
            f"  • Regressions detected: {'Yes' if regression_analysis.get('regressions_detected', False) else 'No'}",
            f"  • Regression count: {regression_analysis.get('regression_count', 0)}",
            ""
        ])
        
        # Final assessment
        lines.extend([
            "FINAL ASSESSMENT:",
            "-" * 17
        ])
        
        if results["overall_success"]:
            lines.extend([
                "✅ All load testing requirements have been successfully implemented and validated.",
                "✅ System demonstrates capability to handle 50+ concurrent users.",
                "✅ Performance metrics meet or exceed specified thresholds.",
                "✅ Comprehensive testing suite is ready for production use.",
                "",
                "RECOMMENDATION: Proceed with deployment - load testing requirements satisfied."
            ])
        else:
            failed_requirements = [req for req, status in validation.items() if not status]
            lines.extend([
                "❌ Load testing requirements validation failed.",
                f"❌ Failed requirements: {', '.join(failed_requirements)}",
                "",
                "RECOMMENDATION: Address failed requirements before deployment."
            ])
        
        return "\n".join(lines)


# Convenience function for load test execution
async def execute_load_tests(config: Optional[LightRAGConfig] = None,
                           output_dir: Optional[str] = None,
                           max_concurrent_users: int = 75,
                           include_endurance_tests: bool = False) -> Dict[str, Any]:
    """
    Execute comprehensive load testing.
    
    Args:
        config: Optional LightRAG configuration
        output_dir: Optional output directory
        max_concurrent_users: Maximum concurrent users to test (must be >= 50)
        include_endurance_tests: Whether to include endurance tests
        
    Returns:
        Load test results dictionary
    """
    executor = LoadTestExecutor(config=config, output_dir=output_dir)
    return await executor.execute_comprehensive_load_tests(
        max_concurrent_users=max_concurrent_users,
        include_endurance_tests=include_endurance_tests
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightRAG Load Test Executor")
    parser.add_argument("--output-dir", default="load_test_execution_results",
                       help="Output directory for test results")
    parser.add_argument("--max-users", type=int, default=75,
                       help="Maximum concurrent users to test (minimum 50)")
    parser.add_argument("--include-endurance", action="store_true",
                       help="Include endurance tests")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.max_users < 50:
        print("Error: max-users must be at least 50 to meet requirements")
        sys.exit(1)
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        try:
            results = await execute_load_tests(
                output_dir=args.output_dir,
                max_concurrent_users=args.max_users,
                include_endurance_tests=args.include_endurance
            )
            
            # Print executive summary
            executor = LoadTestExecutor(output_dir=args.output_dir)
            summary = executor._generate_load_test_summary(results)
            print(summary)
            
            # Exit with appropriate code
            if results["overall_success"]:
                print("\n🎉 ALL LOAD TESTING REQUIREMENTS MET!")
                sys.exit(0)
            else:
                print("\n💥 LOAD TESTING REQUIREMENTS NOT MET!")
                sys.exit(1)
                
        except Exception as e:
            print(f"Load test execution failed: {str(e)}")
            sys.exit(1)
    
    asyncio.run(main())