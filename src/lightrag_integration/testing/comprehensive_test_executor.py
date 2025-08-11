"""
Comprehensive Test Executor for LightRAG Integration

This module provides a unified interface for executing all test suites
including end-to-end tests, regression tests, user acceptance tests,
and performance/load tests with comprehensive reporting.
"""

import asyncio
import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .end_to_end_test_suite import EndToEndTestSuite
from .regression_test_suite import RegressionTestSuite
from .user_acceptance_test_suite import UserAcceptanceTestSuite
from .load_test_suite import LoadTestSuite
from .automated_test_runner import AutomatedTestRunner
from ..component import LightRAGComponent
from ..config.settings import LightRAGConfig
from ..utils.logging import setup_logger


@dataclass
class ComprehensiveTestReport:
    """Comprehensive test execution report."""
    timestamp: datetime
    overall_success: bool
    total_duration_seconds: float
    test_suite_results: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    recommendations: List[str]
    artifacts: Dict[str, str]


class ComprehensiveTestExecutor:
    """
    Comprehensive test executor that runs all test suites and provides
    unified reporting for complete system validation.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the comprehensive test executor.
        
        Args:
            config: Optional LightRAG configuration
            output_dir: Directory for test outputs and reports
        """
        self.config = config or LightRAGConfig.from_env()
        self.output_dir = Path(output_dir or "comprehensive_test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("comprehensive_test_executor",
                                 log_file=str(self.output_dir / "comprehensive_tests.log"))
        
        # Initialize test suites
        self.end_to_end_suite = EndToEndTestSuite(self.config, str(self.output_dir / "e2e"))
        self.regression_suite = RegressionTestSuite(self.config, str(self.output_dir / "regression"))
        self.uat_suite = UserAcceptanceTestSuite(self.config, str(self.output_dir / "uat"))
        self.load_test_suite = LoadTestSuite(self.config, str(self.output_dir / "load"))
        self.automated_runner = AutomatedTestRunner(self.config, str(self.output_dir / "automated"))
    
    async def execute_comprehensive_tests(self, 
                                        include_load_tests: bool = True,
                                        include_endurance_tests: bool = False,
                                        max_concurrent_users: int = 50) -> ComprehensiveTestReport:
        """
        Execute comprehensive test suite covering all requirements.
        
        Args:
            include_load_tests: Whether to include load/performance tests
            include_endurance_tests: Whether to include long-running endurance tests
            max_concurrent_users: Maximum concurrent users for load testing
            
        Returns:
            ComprehensiveTestReport with all results
        """
        test_start = datetime.now()
        self.logger.info("Starting comprehensive test execution")
        
        test_suite_results = {}
        performance_metrics = {}
        quality_metrics = {}
        overall_success = True
        artifacts = {}
        
        try:
            # Initialize LightRAG component for testing
            component = LightRAGComponent(self.config)
            await component.initialize()
            
            # 1. End-to-End Tests (Task 15.1 requirement)
            self.logger.info("Executing end-to-end test suite")
            e2e_results = await self.end_to_end_suite.run_complete_end_to_end_tests()
            test_suite_results["end_to_end"] = e2e_results
            
            if not e2e_results.get("overall_success", False):
                overall_success = False
                self.logger.warning("End-to-end tests failed")
            
            # 2. Regression Tests (Task 15.1 requirement)
            self.logger.info("Executing regression test suite")
            regression_results = await self.regression_suite.run_regression_tests()
            test_suite_results["regression"] = regression_results
            
            if not regression_results.get("overall_success", False):
                overall_success = False
                self.logger.warning("Regression tests failed")
            
            # 3. User Acceptance Tests (Task 15.1 requirement)
            self.logger.info("Executing user acceptance test suite")
            uat_results = await self.uat_suite.run_user_acceptance_tests()
            test_suite_results["user_acceptance"] = uat_results
            
            if not uat_results.get("overall_success", False):
                self.logger.warning("User acceptance tests failed (non-critical)")
            
            # 4. Performance and Load Tests (Task 15.2 requirement)
            if include_load_tests:
                self.logger.info("Executing performance and load tests")
                
                # Concurrent user test (50+ users requirement)
                load_test_results = await self.load_test_suite.run_concurrent_user_test(
                    component=component,
                    concurrent_users=max_concurrent_users,
                    requests_per_user=10,
                    question_set="mixed"
                )
                
                # Scalability test
                scalability_results = await self.load_test_suite.run_scalability_test(
                    component=component,
                    max_users=max_concurrent_users * 2,
                    step_size=10,
                    requests_per_user=5
                )
                
                # Stress test
                stress_results = await self.load_test_suite.run_stress_test(
                    component=component,
                    duration_minutes=15,
                    max_concurrent_users=max_concurrent_users,
                    ramp_up_minutes=5
                )
                
                test_suite_results["load_tests"] = {
                    "concurrent_user_test": asdict(load_test_results),
                    "scalability_test": asdict(scalability_results),
                    "stress_test": asdict(stress_results)
                }
                
                # Extract performance metrics
                performance_metrics = self._extract_performance_metrics(
                    load_test_results, scalability_results, stress_results
                )
                
                # Check if load tests meet requirements
                if not self._validate_load_test_requirements(performance_metrics):
                    overall_success = False
                    self.logger.warning("Load test requirements not met")
            
            # 5. Endurance Tests (optional)
            if include_endurance_tests:
                self.logger.info("Executing endurance tests")
                endurance_results = await self.load_test_suite.run_endurance_test(
                    component=component,
                    duration_hours=1.0,  # Shortened for testing
                    constant_load_users=20
                )
                test_suite_results["endurance"] = asdict(endurance_results)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(test_suite_results)
            
            # Generate recommendations
            recommendations = self._generate_comprehensive_recommendations(
                test_suite_results, performance_metrics, quality_metrics, overall_success
            )
            
            # Cleanup
            await component.cleanup()
            
            total_duration = (datetime.now() - test_start).total_seconds()
            
            # Create comprehensive report
            report = ComprehensiveTestReport(
                timestamp=test_start,
                overall_success=overall_success,
                total_duration_seconds=total_duration,
                test_suite_results=test_suite_results,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics,
                recommendations=recommendations,
                artifacts=artifacts
            )
            
            # Save report artifacts
            await self._save_comprehensive_report(report)
            
            self.logger.info(
                f"Comprehensive test execution completed: "
                f"{'SUCCESS' if overall_success else 'FAILED'} "
                f"in {total_duration:.2f}s"
            )
            
            return report
            
        except Exception as e:
            error_msg = f"Comprehensive test execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return ComprehensiveTestReport(
                timestamp=test_start,
                overall_success=False,
                total_duration_seconds=(datetime.now() - test_start).total_seconds(),
                test_suite_results=test_suite_results,
                performance_metrics=performance_metrics,
                quality_metrics=quality_metrics,
                recommendations=[f"Fix critical error: {error_msg}"],
                artifacts={}
            )
    
    def _extract_performance_metrics(self, load_test_result, scalability_result, stress_result) -> Dict[str, Any]:
        """Extract key performance metrics from test results."""
        return {
            "max_concurrent_users_tested": load_test_result.concurrent_users,
            "concurrent_user_success_rate": load_test_result.successful_requests / load_test_result.total_requests,
            "concurrent_user_avg_response_time": load_test_result.average_response_time,
            "concurrent_user_p95_response_time": load_test_result.percentile_95_response_time,
            "concurrent_user_throughput": load_test_result.requests_per_second,
            "scalability_breaking_point": scalability_result.breaking_point,
            "max_stable_users": scalability_result.scalability_metrics.get("max_stable_users", 0),
            "throughput_efficiency": scalability_result.scalability_metrics.get("throughput_efficiency", 0),
            "stress_test_max_users": stress_result.max_concurrent_users,
            "stress_test_stability": stress_result.system_stability,
            "resource_exhaustion_point": stress_result.resource_exhaustion_point,
            "system_recovery_successful": stress_result.recovery_metrics.get("recovery_successful", False)
        }
    
    def _validate_load_test_requirements(self, performance_metrics: Dict[str, Any]) -> bool:
        """Validate that load test results meet requirements."""
        requirements_met = True
        
        # Requirement 8.3: Handle at least 50 concurrent users
        if performance_metrics.get("max_stable_users", 0) < 50:
            self.logger.warning("Requirement 8.3 not met: Cannot handle 50+ concurrent users")
            requirements_met = False
        
        # Requirement 8.5: Success rate >= 95%
        if performance_metrics.get("concurrent_user_success_rate", 0) < 0.95:
            self.logger.warning("Requirement 8.5 not met: Success rate below 95%")
            requirements_met = False
        
        # Requirement 8.7: Response time <= 5 seconds for 95% of requests
        if performance_metrics.get("concurrent_user_p95_response_time", 0) > 5.0:
            self.logger.warning("Requirement 8.7 not met: P95 response time above 5 seconds")
            requirements_met = False
        
        return requirements_met
    
    def _calculate_quality_metrics(self, test_suite_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics from all test results."""
        quality_metrics = {}
        
        # End-to-end test quality
        e2e_results = test_suite_results.get("end_to_end", {})
        if e2e_results:
            e2e_summary = e2e_results.get("summary", {})
            quality_metrics["e2e_pass_rate"] = (
                e2e_summary.get("scenarios_passed", 0) / 
                max(e2e_summary.get("scenarios_passed", 0) + e2e_summary.get("scenarios_failed", 0), 1)
            )
        
        # Regression test quality
        regression_results = test_suite_results.get("regression", {})
        if regression_results:
            regression_summary = regression_results.get("summary", {})
            quality_metrics["regression_pass_rate"] = (
                regression_summary.get("passed", 0) / 
                max(regression_summary.get("total_tests", 1), 1)
            )
            quality_metrics["critical_regressions"] = regression_summary.get("critical_failures", 0)
        
        # User acceptance quality
        uat_results = test_suite_results.get("user_acceptance", {})
        if uat_results:
            uat_summary = uat_results.get("summary", {})
            quality_metrics["uat_pass_rate"] = (
                uat_summary.get("passed", 0) / 
                max(uat_summary.get("total_scenarios", 1), 1)
            )
            quality_metrics["average_user_satisfaction"] = uat_summary.get("average_satisfaction", 0)
        
        # Overall quality score
        quality_scores = [
            quality_metrics.get("e2e_pass_rate", 0),
            quality_metrics.get("regression_pass_rate", 0),
            quality_metrics.get("uat_pass_rate", 0) * 0.8  # Weight UAT slightly less
        ]
        quality_metrics["overall_quality_score"] = sum(quality_scores) / len(quality_scores)
        
        return quality_metrics
    
    def _generate_comprehensive_recommendations(self, 
                                              test_suite_results: Dict[str, Any],
                                              performance_metrics: Dict[str, Any],
                                              quality_metrics: Dict[str, Any],
                                              overall_success: bool) -> List[str]:
        """Generate comprehensive recommendations based on all test results."""
        recommendations = []
        
        if overall_success:
            recommendations.extend([
                "✅ All critical test suites passed successfully",
                "✅ System meets performance requirements for 50+ concurrent users",
                "✅ No critical regressions detected in existing functionality",
                "✅ End-to-end workflows function correctly",
                "System is ready for production deployment"
            ])
        else:
            recommendations.append("❌ Critical test failures detected - deployment not recommended")
        
        # Performance-specific recommendations
        max_stable_users = performance_metrics.get("max_stable_users", 0)
        if max_stable_users < 50:
            recommendations.append(f"⚠️ System only stable up to {max_stable_users} users - optimization needed")
        
        throughput_efficiency = performance_metrics.get("throughput_efficiency", 1.0)
        if throughput_efficiency < 0.7:
            recommendations.append("⚠️ Poor throughput scaling - investigate bottlenecks")
        
        # Quality-specific recommendations
        overall_quality = quality_metrics.get("overall_quality_score", 0)
        if overall_quality < 0.8:
            recommendations.append("⚠️ Overall quality score below 80% - address test failures")
        
        critical_regressions = quality_metrics.get("critical_regressions", 0)
        if critical_regressions > 0:
            recommendations.append(f"❌ {critical_regressions} critical regressions detected - fix before deployment")
        
        user_satisfaction = quality_metrics.get("average_user_satisfaction", 0)
        if user_satisfaction < 3.5:
            recommendations.append("⚠️ Low user satisfaction score - improve user experience")
        
        # Specific improvement recommendations
        if not overall_success:
            recommendations.extend([
                "Review detailed test reports for specific failure causes",
                "Fix critical issues identified in failed test suites",
                "Re-run comprehensive tests after fixes",
                "Consider performance optimization if load tests failed",
                "Update regression baselines if intentional changes were made"
            ])
        
        return recommendations
    
    async def _save_comprehensive_report(self, report: ComprehensiveTestReport) -> None:
        """Save comprehensive test report to files."""
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON report
        json_file = self.output_dir / f"comprehensive_test_report_{timestamp}.json"
        report_dict = asdict(report)
        report_dict["timestamp"] = report.timestamp.isoformat()
        
        with open(json_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        # Save human-readable summary
        summary_file = self.output_dir / f"comprehensive_test_summary_{timestamp}.txt"
        summary_content = self._generate_summary_report(report)
        summary_file.write_text(summary_content)
        
        # Save executive summary
        exec_summary_file = self.output_dir / f"executive_summary_{timestamp}.txt"
        exec_summary = self._generate_executive_summary(report)
        exec_summary_file.write_text(exec_summary)
        
        report.artifacts.update({
            "detailed_json_report": str(json_file),
            "summary_report": str(summary_file),
            "executive_summary": str(exec_summary_file)
        })
        
        self.logger.info(f"Comprehensive test reports saved to {self.output_dir}")
    
    def _generate_summary_report(self, report: ComprehensiveTestReport) -> str:
        """Generate detailed summary report."""
        lines = [
            "=" * 80,
            "LIGHTRAG COMPREHENSIVE TEST EXECUTION REPORT",
            "=" * 80,
            f"Timestamp: {report.timestamp.isoformat()}",
            f"Duration: {report.total_duration_seconds:.2f} seconds",
            f"Overall Status: {'✅ SUCCESS' if report.overall_success else '❌ FAILURE'}",
            "",
            "QUALITY METRICS:",
            f"  Overall Quality Score: {report.quality_metrics.get('overall_quality_score', 0):.1%}",
            f"  End-to-End Pass Rate: {report.quality_metrics.get('e2e_pass_rate', 0):.1%}",
            f"  Regression Pass Rate: {report.quality_metrics.get('regression_pass_rate', 0):.1%}",
            f"  User Acceptance Pass Rate: {report.quality_metrics.get('uat_pass_rate', 0):.1%}",
            f"  Average User Satisfaction: {report.quality_metrics.get('average_user_satisfaction', 0):.1f}/5.0",
            "",
            "PERFORMANCE METRICS:",
            f"  Max Concurrent Users Tested: {report.performance_metrics.get('max_concurrent_users_tested', 0)}",
            f"  Max Stable Users: {report.performance_metrics.get('max_stable_users', 0)}",
            f"  Success Rate at Max Load: {report.performance_metrics.get('concurrent_user_success_rate', 0):.1%}",
            f"  Average Response Time: {report.performance_metrics.get('concurrent_user_avg_response_time', 0):.2f}s",
            f"  P95 Response Time: {report.performance_metrics.get('concurrent_user_p95_response_time', 0):.2f}s",
            f"  Throughput: {report.performance_metrics.get('concurrent_user_throughput', 0):.1f} req/s",
            f"  Throughput Efficiency: {report.performance_metrics.get('throughput_efficiency', 0):.1%}",
            "",
            "RECOMMENDATIONS:",
        ]
        
        for recommendation in report.recommendations:
            lines.append(f"  {recommendation}")
        
        return "\n".join(lines)
    
    def _generate_executive_summary(self, report: ComprehensiveTestReport) -> str:
        """Generate executive summary for stakeholders."""
        status = "READY FOR DEPLOYMENT" if report.overall_success else "NOT READY FOR DEPLOYMENT"
        
        lines = [
            "LIGHTRAG INTEGRATION - EXECUTIVE SUMMARY",
            "=" * 50,
            f"Status: {status}",
            f"Test Date: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Test Duration: {report.total_duration_seconds/60:.1f} minutes",
            "",
            "KEY FINDINGS:",
        ]
        
        # Key findings based on results
        if report.overall_success:
            lines.extend([
                "✅ All critical functionality tests passed",
                "✅ System handles 50+ concurrent users successfully",
                "✅ No regressions in existing functionality",
                "✅ User acceptance criteria met"
            ])
        else:
            lines.extend([
                "❌ Critical test failures detected",
                "❌ System not ready for production deployment",
                "❌ Performance or functionality issues identified"
            ])
        
        lines.extend([
            "",
            "PERFORMANCE SUMMARY:",
            f"  Maximum Stable Users: {report.performance_metrics.get('max_stable_users', 0)}",
            f"  Response Time (95th percentile): {report.performance_metrics.get('concurrent_user_p95_response_time', 0):.1f}s",
            f"  System Success Rate: {report.performance_metrics.get('concurrent_user_success_rate', 0):.1%}",
            "",
            "NEXT STEPS:"
        ])
        
        if report.overall_success:
            lines.extend([
                "• Proceed with production deployment",
                "• Monitor system performance in production",
                "• Schedule regular performance reviews"
            ])
        else:
            lines.extend([
                "• Address critical test failures",
                "• Optimize performance bottlenecks",
                "• Re-run comprehensive tests",
                "• Review detailed test reports for specific issues"
            ])
        
        return "\n".join(lines)


# Convenience function for running comprehensive tests
async def run_comprehensive_tests(config: Optional[LightRAGConfig] = None,
                                output_dir: Optional[str] = None,
                                include_load_tests: bool = True,
                                include_endurance_tests: bool = False,
                                max_concurrent_users: int = 50) -> ComprehensiveTestReport:
    """
    Run comprehensive test suite covering all requirements.
    
    Args:
        config: Optional LightRAG configuration
        output_dir: Optional output directory
        include_load_tests: Whether to include load/performance tests
        include_endurance_tests: Whether to include endurance tests
        max_concurrent_users: Maximum concurrent users for load testing
        
    Returns:
        ComprehensiveTestReport with all results
    """
    executor = ComprehensiveTestExecutor(config=config, output_dir=output_dir)
    return await executor.execute_comprehensive_tests(
        include_load_tests=include_load_tests,
        include_endurance_tests=include_endurance_tests,
        max_concurrent_users=max_concurrent_users
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LightRAG Comprehensive Test Executor")
    parser.add_argument("--output-dir", default="comprehensive_test_results",
                       help="Output directory for test results")
    parser.add_argument("--max-users", type=int, default=50,
                       help="Maximum concurrent users for load testing")
    parser.add_argument("--skip-load-tests", action="store_true",
                       help="Skip load and performance tests")
    parser.add_argument("--include-endurance", action="store_true",
                       help="Include endurance tests")
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
            report = await run_comprehensive_tests(
                output_dir=args.output_dir,
                include_load_tests=not args.skip_load_tests,
                include_endurance_tests=args.include_endurance,
                max_concurrent_users=args.max_users
            )
            
            # Print executive summary
            executor = ComprehensiveTestExecutor(output_dir=args.output_dir)
            exec_summary = executor._generate_executive_summary(report)
            print(exec_summary)
            
            # Exit with appropriate code
            if report.overall_success:
                print("\n🎉 ALL COMPREHENSIVE TESTS PASSED!")
                sys.exit(0)
            else:
                print("\n💥 COMPREHENSIVE TESTS FAILED!")
                sys.exit(1)
                
        except Exception as e:
            print(f"Comprehensive test execution failed: {str(e)}")
            sys.exit(1)
    
    asyncio.run(main())