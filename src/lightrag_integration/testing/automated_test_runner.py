"""
Automated Test Execution and Reporting System

This module provides automated test execution that combines end-to-end tests,
regression tests, and user acceptance tests with comprehensive reporting
and integration with CI/CD systems.
"""

import asyncio
import logging
import json
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .end_to_end_test_suite import EndToEndTestSuite, run_end_to_end_tests
from .regression_test_suite import RegressionTestSuite, run_regression_tests
from .user_acceptance_test_suite import UserAcceptanceTestSuite, run_user_acceptance_tests
from .test_runner import MVPTestRunner
from ..config.settings import LightRAGConfig
from ..utils.logging import setup_logger


@dataclass
class TestSuiteResult:
    """Result of a test suite execution."""
    suite_name: str
    success: bool
    duration_seconds: float
    test_count: int
    passed_count: int
    failed_count: int
    error_message: Optional[str]
    timestamp: datetime


@dataclass
class AutomatedTestReport:
    """Comprehensive automated test report."""
    timestamp: datetime
    overall_success: bool
    total_duration_seconds: float
    suite_results: List[TestSuiteResult]
    summary: Dict[str, Any]
    recommendations: List[str]
    artifacts: Dict[str, str]  # File paths to generated artifacts


class AutomatedTestRunner:
    """
    Automated test runner that executes all test suites and generates
    comprehensive reports for CI/CD integration.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the automated test runner.
        
        Args:
            config: Optional LightRAG configuration
            output_dir: Directory for test outputs and reports
        """
        self.config = config or LightRAGConfig.from_env()
        self.output_dir = Path(output_dir or "automated_test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("automated_test_runner",
                                 log_file=str(self.output_dir / "automated_tests.log"))
        
        # Test suite configurations
        self.test_suites = {
            "mvp_validation": {
                "name": "MVP Validation",
                "runner_class": MVPTestRunner,
                "enabled": True,
                "critical": True
            },
            "end_to_end": {
                "name": "End-to-End Tests",
                "runner_class": EndToEndTestSuite,
                "enabled": True,
                "critical": True
            },
            "regression": {
                "name": "Regression Tests",
                "runner_class": RegressionTestSuite,
                "enabled": True,
                "critical": True
            },
            "user_acceptance": {
                "name": "User Acceptance Tests",
                "runner_class": UserAcceptanceTestSuite,
                "enabled": True,
                "critical": False
            }
        }
    
    async def run_all_tests(self, 
                          include_suites: Optional[List[str]] = None,
                          exclude_suites: Optional[List[str]] = None,
                          fail_fast: bool = False) -> AutomatedTestReport:
        """
        Run all configured test suites.
        
        Args:
            include_suites: List of suite names to include (None for all)
            exclude_suites: List of suite names to exclude
            fail_fast: Stop execution on first critical failure
            
        Returns:
            Comprehensive automated test report
        """
        test_start = datetime.now()
        self.logger.info("Starting automated test execution")
        
        suite_results = []
        overall_success = True
        artifacts = {}
        
        # Determine which suites to run
        suites_to_run = self._determine_suites_to_run(include_suites, exclude_suites)
        
        try:
            for suite_key in suites_to_run:
                suite_config = self.test_suites[suite_key]
                
                self.logger.info(f"Running test suite: {suite_config['name']}")
                
                suite_result = await self._run_test_suite(suite_key, suite_config)
                suite_results.append(suite_result)
                
                # Update overall success
                if not suite_result.success:
                    if suite_config["critical"]:
                        overall_success = False
                        if fail_fast:
                            self.logger.warning(f"Critical test suite failed, stopping execution")
                            break
                
                self.logger.info(
                    f"Suite {suite_config['name']} completed: "
                    f"{'SUCCESS' if suite_result.success else 'FAILED'} "
                    f"({suite_result.passed_count}/{suite_result.test_count} passed)"
                )
            
            # Generate comprehensive report
            total_duration = (datetime.now() - test_start).total_seconds()
            
            report = AutomatedTestReport(
                timestamp=test_start,
                overall_success=overall_success,
                total_duration_seconds=total_duration,
                suite_results=suite_results,
                summary=self._generate_summary(suite_results),
                recommendations=self._generate_recommendations(suite_results, overall_success),
                artifacts=artifacts
            )
            
            # Save report artifacts
            await self._save_report_artifacts(report)
            
            self.logger.info(
                f"Automated test execution completed: "
                f"{'SUCCESS' if overall_success else 'FAILED'} "
                f"in {total_duration:.2f}s"
            )
            
            return report
            
        except Exception as e:
            error_msg = f"Automated test execution failed: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Create error report
            return AutomatedTestReport(
                timestamp=test_start,
                overall_success=False,
                total_duration_seconds=(datetime.now() - test_start).total_seconds(),
                suite_results=suite_results,
                summary={"error": error_msg},
                recommendations=["Fix critical system error before proceeding"],
                artifacts={}
            )
    
    def _determine_suites_to_run(self, 
                               include_suites: Optional[List[str]],
                               exclude_suites: Optional[List[str]]) -> List[str]:
        """Determine which test suites to run based on include/exclude lists."""
        suites_to_run = []
        
        for suite_key, suite_config in self.test_suites.items():
            if not suite_config["enabled"]:
                continue
            
            if include_suites and suite_key not in include_suites:
                continue
            
            if exclude_suites and suite_key in exclude_suites:
                continue
            
            suites_to_run.append(suite_key)
        
        return suites_to_run
    
    async def _run_test_suite(self, suite_key: str, suite_config: Dict[str, Any]) -> TestSuiteResult:
        """Run a single test suite and return results."""
        suite_start = datetime.now()
        
        try:
            if suite_key == "mvp_validation":
                runner = MVPTestRunner(self.config, str(self.output_dir / "mvp"))
                results = await runner.run_complete_mvp_validation()
                
                success = results.get("overall_mvp_status", False)
                test_count = 2  # Validation + Performance
                passed_count = 2 if success else 0
                failed_count = 0 if success else 2
                
            elif suite_key == "end_to_end":
                results = await run_end_to_end_tests(
                    config=self.config,
                    output_dir=str(self.output_dir / "e2e")
                )
                
                success = results.get("overall_success", False)
                test_count = len(results.get("test_scenarios", []))
                passed_count = results.get("summary", {}).get("scenarios_passed", 0)
                failed_count = results.get("summary", {}).get("scenarios_failed", 0)
                
            elif suite_key == "regression":
                results = await run_regression_tests(
                    config=self.config,
                    output_dir=str(self.output_dir / "regression")
                )
                
                success = results.get("overall_success", False)
                test_count = results.get("summary", {}).get("total_tests", 0)
                passed_count = results.get("summary", {}).get("passed", 0)
                failed_count = results.get("summary", {}).get("failed", 0)
                
            elif suite_key == "user_acceptance":
                results = await run_user_acceptance_tests(
                    config=self.config,
                    output_dir=str(self.output_dir / "uat")
                )
                
                success = results.get("overall_success", False)
                test_count = results.get("summary", {}).get("total_scenarios", 0)
                passed_count = results.get("summary", {}).get("passed", 0)
                failed_count = results.get("summary", {}).get("failed", 0)
                
            else:
                raise ValueError(f"Unknown test suite: {suite_key}")
            
            return TestSuiteResult(
                suite_name=suite_config["name"],
                success=success,
                duration_seconds=(datetime.now() - suite_start).total_seconds(),
                test_count=test_count,
                passed_count=passed_count,
                failed_count=failed_count,
                error_message=results.get("error"),
                timestamp=suite_start
            )
            
        except Exception as e:
            return TestSuiteResult(
                suite_name=suite_config["name"],
                success=False,
                duration_seconds=(datetime.now() - suite_start).total_seconds(),
                test_count=0,
                passed_count=0,
                failed_count=1,
                error_message=str(e),
                timestamp=suite_start
            )
    
    def _generate_summary(self, suite_results: List[TestSuiteResult]) -> Dict[str, Any]:
        """Generate summary statistics from suite results."""
        total_tests = sum(result.test_count for result in suite_results)
        total_passed = sum(result.passed_count for result in suite_results)
        total_failed = sum(result.failed_count for result in suite_results)
        total_suites = len(suite_results)
        passed_suites = sum(1 for result in suite_results if result.success)
        failed_suites = total_suites - passed_suites
        
        return {
            "total_test_suites": total_suites,
            "passed_test_suites": passed_suites,
            "failed_test_suites": failed_suites,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "overall_pass_rate": total_passed / total_tests if total_tests > 0 else 0.0,
            "suite_pass_rate": passed_suites / total_suites if total_suites > 0 else 0.0
        }
    
    def _generate_recommendations(self, suite_results: List[TestSuiteResult],
                                overall_success: bool) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if overall_success:
            recommendations.extend([
                "✅ All critical tests passed successfully",
                "System is ready for deployment",
                "Consider running additional performance tests under load",
                "Monitor system metrics in production environment"
            ])
        else:
            recommendations.append("❌ Critical test failures detected")
            
            # Identify specific issues
            failed_suites = [result for result in suite_results if not result.success]
            
            for failed_suite in failed_suites:
                if failed_suite.error_message:
                    recommendations.append(f"Fix error in {failed_suite.suite_name}: {failed_suite.error_message}")
                else:
                    recommendations.append(f"Address test failures in {failed_suite.suite_name}")
            
            recommendations.extend([
                "Review failed test details in individual suite reports",
                "Fix critical issues before proceeding with deployment",
                "Consider rolling back recent changes if regressions detected"
            ])
        
        return recommendations
    
    async def _save_report_artifacts(self, report: AutomatedTestReport) -> None:
        """Save report artifacts to files."""
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save comprehensive report
        report_file = self.output_dir / f"automated_test_report_{timestamp}.txt"
        report_content = self._generate_text_report(report)
        report_file.write_text(report_content)
        report.artifacts["text_report"] = str(report_file)
        
        # Save JSON results
        json_file = self.output_dir / f"automated_test_results_{timestamp}.json"
        json_content = asdict(report)
        # Convert datetime objects to strings for JSON serialization
        json_content["timestamp"] = report.timestamp.isoformat()
        for suite_result in json_content["suite_results"]:
            suite_result["timestamp"] = suite_result["timestamp"].isoformat() if hasattr(suite_result["timestamp"], "isoformat") else suite_result["timestamp"]
        
        with open(json_file, 'w') as f:
            json.dump(json_content, f, indent=2, default=str)
        report.artifacts["json_results"] = str(json_file)
        
        # Save CI/CD compatible results
        cicd_file = self.output_dir / f"test_results_{timestamp}.xml"
        cicd_content = self._generate_junit_xml(report)
        cicd_file.write_text(cicd_content)
        report.artifacts["junit_xml"] = str(cicd_file)
        
        self.logger.info(f"Report artifacts saved to {self.output_dir}")
    
    def _generate_text_report(self, report: AutomatedTestReport) -> str:
        """Generate human-readable text report."""
        report_lines = [
            "=" * 80,
            "LIGHTRAG AUTOMATED TEST EXECUTION REPORT",
            "=" * 80,
            f"Timestamp: {report.timestamp.isoformat()}",
            f"Total Duration: {report.total_duration_seconds:.2f} seconds",
            f"Overall Status: {'✅ SUCCESS' if report.overall_success else '❌ FAILURE'}",
            ""
        ]
        
        # Summary
        summary = report.summary
        if "error" not in summary:
            report_lines.extend([
                "SUMMARY:",
                f"  Test Suites: {summary['passed_test_suites']}/{summary['total_test_suites']} passed",
                f"  Individual Tests: {summary['total_passed']}/{summary['total_tests']} passed",
                f"  Suite Pass Rate: {summary['suite_pass_rate']:.1%}",
                f"  Overall Pass Rate: {summary['overall_pass_rate']:.1%}",
                ""
            ])
        
        # Suite results
        report_lines.append("TEST SUITE RESULTS:")
        report_lines.append("-" * 40)
        
        for suite_result in report.suite_results:
            status = "✅ PASSED" if suite_result.success else "❌ FAILED"
            
            report_lines.extend([
                f"Suite: {suite_result.suite_name}",
                f"  Status: {status}",
                f"  Duration: {suite_result.duration_seconds:.2f}s",
                f"  Tests: {suite_result.passed_count}/{suite_result.test_count} passed"
            ])
            
            if suite_result.error_message:
                report_lines.append(f"  Error: {suite_result.error_message}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 20
        ])
        
        for recommendation in report.recommendations:
            report_lines.append(f"  {recommendation}")
        
        report_lines.extend([
            "",
            "ARTIFACTS:",
            "-" * 10
        ])
        
        for artifact_type, artifact_path in report.artifacts.items():
            report_lines.append(f"  {artifact_type}: {artifact_path}")
        
        return "\n".join(report_lines)
    
    def _generate_junit_xml(self, report: AutomatedTestReport) -> str:
        """Generate JUnit XML format for CI/CD integration."""
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuites name="LightRAG Tests" tests="{report.summary.get("total_tests", 0)}" '
            f'failures="{report.summary.get("total_failed", 0)}" '
            f'time="{report.total_duration_seconds:.3f}">'
        ]
        
        for suite_result in report.suite_results:
            xml_lines.extend([
                f'  <testsuite name="{suite_result.suite_name}" '
                f'tests="{suite_result.test_count}" '
                f'failures="{suite_result.failed_count}" '
                f'time="{suite_result.duration_seconds:.3f}">',
            ])
            
            # Add individual test cases (simplified)
            for i in range(suite_result.test_count):
                test_name = f"test_{i+1}"
                if i < suite_result.passed_count:
                    xml_lines.append(f'    <testcase name="{test_name}" time="1.0"/>')
                else:
                    xml_lines.extend([
                        f'    <testcase name="{test_name}" time="1.0">',
                        f'      <failure message="Test failed">{suite_result.error_message or "Test failed"}</failure>',
                        '    </testcase>'
                    ])
            
            xml_lines.append('  </testsuite>')
        
        xml_lines.append('</testsuites>')
        
        return "\n".join(xml_lines)
    
    def generate_ci_cd_summary(self, report: AutomatedTestReport) -> Dict[str, Any]:
        """Generate CI/CD compatible summary."""
        return {
            "success": report.overall_success,
            "timestamp": report.timestamp.isoformat(),
            "duration_seconds": report.total_duration_seconds,
            "total_tests": report.summary.get("total_tests", 0),
            "passed_tests": report.summary.get("total_passed", 0),
            "failed_tests": report.summary.get("total_failed", 0),
            "pass_rate": report.summary.get("overall_pass_rate", 0.0),
            "artifacts": report.artifacts,
            "recommendations": report.recommendations[:3]  # Top 3 recommendations
        }


# Convenience function for running automated tests
async def run_automated_tests(config: Optional[LightRAGConfig] = None,
                            output_dir: Optional[str] = None,
                            include_suites: Optional[List[str]] = None,
                            exclude_suites: Optional[List[str]] = None,
                            fail_fast: bool = False) -> AutomatedTestReport:
    """
    Run automated test suite.
    
    Args:
        config: Optional LightRAG configuration
        output_dir: Optional output directory for results
        include_suites: List of suite names to include
        exclude_suites: List of suite names to exclude
        fail_fast: Stop on first critical failure
        
    Returns:
        Automated test report
    """
    runner = AutomatedTestRunner(config=config, output_dir=output_dir)
    return await runner.run_all_tests(
        include_suites=include_suites,
        exclude_suites=exclude_suites,
        fail_fast=fail_fast
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightRAG Automated Test Runner")
    parser.add_argument("--output-dir", default="automated_test_results",
                       help="Output directory for test results")
    parser.add_argument("--include-suites", nargs="+",
                       choices=["mvp_validation", "end_to_end", "regression", "user_acceptance"],
                       help="Test suites to include")
    parser.add_argument("--exclude-suites", nargs="+",
                       choices=["mvp_validation", "end_to_end", "regression", "user_acceptance"],
                       help="Test suites to exclude")
    parser.add_argument("--fail-fast", action="store_true",
                       help="Stop on first critical failure")
    parser.add_argument("--ci-cd", action="store_true",
                       help="Output CI/CD compatible summary")
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
            # Run automated tests
            report = await run_automated_tests(
                output_dir=args.output_dir,
                include_suites=args.include_suites,
                exclude_suites=args.exclude_suites,
                fail_fast=args.fail_fast
            )
            
            if args.ci_cd:
                # Output CI/CD compatible summary
                runner = AutomatedTestRunner(output_dir=args.output_dir)
                summary = runner.generate_ci_cd_summary(report)
                print(json.dumps(summary, indent=2))
            else:
                # Output human-readable report
                runner = AutomatedTestRunner(output_dir=args.output_dir)
                text_report = runner._generate_text_report(report)
                print(text_report)
            
            # Exit with appropriate code
            if report.overall_success:
                print("\n🎉 ALL AUTOMATED TESTS PASSED!")
                sys.exit(0)
            else:
                print("\n💥 AUTOMATED TESTS FAILED!")
                sys.exit(1)
                
        except Exception as e:
            print(f"Automated test execution failed: {str(e)}")
            sys.exit(1)
    
    asyncio.run(main())