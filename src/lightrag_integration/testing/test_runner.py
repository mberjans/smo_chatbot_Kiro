"""
Comprehensive Test Runner for LightRAG MVP Validation

This module provides a unified test runner that combines clinical metabolomics
validation tests with performance benchmarking for complete MVP testing.
"""

import asyncio
import logging
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json

from .clinical_metabolomics_suite import (
    ClinicalMetabolomicsTestSuite,
    run_mvp_validation_test,
    TestSuiteResult
)
from .performance_benchmark import (
    PerformanceBenchmark,
    run_performance_benchmark,
    BenchmarkSuite
)
from ..config.settings import LightRAGConfig
from ..utils.logging import setup_logger


class MVPTestRunner:
    """
    Comprehensive test runner for LightRAG MVP validation.
    
    This class combines clinical metabolomics validation tests with performance
    benchmarking to provide complete MVP testing capabilities.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None, 
                 output_dir: Optional[str] = None):
        """
        Initialize the MVP test runner.
        
        Args:
            config: Optional LightRAG configuration
            output_dir: Directory for test outputs and reports
        """
        self.config = config or LightRAGConfig.from_env()
        self.output_dir = Path(output_dir or "mvp_test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("mvp_test_runner", 
                                 log_file=str(self.output_dir / "test_runner.log"))
        
        # MVP success criteria
        self.mvp_criteria = {
            # Functional criteria
            "minimum_pass_rate": 0.85,  # 85% of validation tests must pass
            "minimum_accuracy": 0.75,   # Average accuracy must be 75%
            "minimum_confidence": 0.6,  # Average confidence must be 60%
            
            # Performance criteria
            "maximum_response_time": 5.0,  # Average response time under 5 seconds
            "maximum_memory_usage": 2048,  # Peak memory under 2GB
            "minimum_throughput": 1.0,     # At least 1 request per second
            
            # Core functionality
            "clinical_metabolomics_question_pass": True,  # Must answer core question correctly
        }
    
    async def run_complete_mvp_validation(self) -> Dict[str, Any]:
        """
        Run complete MVP validation including both functional and performance tests.
        
        Returns:
            Dictionary with comprehensive MVP validation results
        """
        mvp_start = datetime.now()
        self.logger.info("Starting complete MVP validation")
        
        results = {
            "timestamp": mvp_start.isoformat(),
            "validation_tests": None,
            "performance_benchmark": None,
            "mvp_criteria_evaluation": None,
            "overall_mvp_status": False,
            "duration_seconds": 0,
            "errors": []
        }
        
        try:
            # 1. Run clinical metabolomics validation tests
            self.logger.info("Running clinical metabolomics validation tests")
            try:
                validation_results = await run_mvp_validation_test(self.config)
                results["validation_tests"] = validation_results
                self.logger.info(
                    f"Validation tests completed: {validation_results.passed_questions}/"
                    f"{validation_results.total_questions} passed"
                )
            except Exception as e:
                error_msg = f"Validation tests failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                results["errors"].append(error_msg)
            
            # 2. Run performance benchmark
            self.logger.info("Running performance benchmark")
            try:
                benchmark_results = await run_performance_benchmark(self.config)
                results["performance_benchmark"] = benchmark_results
                self.logger.info(
                    f"Performance benchmark completed: "
                    f"{benchmark_results.summary['successful_operations']}/"
                    f"{benchmark_results.summary['total_operations']} operations successful"
                )
            except Exception as e:
                error_msg = f"Performance benchmark failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                results["errors"].append(error_msg)
            
            # 3. Evaluate MVP criteria
            mvp_evaluation = self._evaluate_mvp_criteria(
                results["validation_tests"],
                results["performance_benchmark"]
            )
            results["mvp_criteria_evaluation"] = mvp_evaluation
            results["overall_mvp_status"] = mvp_evaluation["overall_success"]
            
            # 4. Calculate total duration
            results["duration_seconds"] = (datetime.now() - mvp_start).total_seconds()
            
            # 5. Log final status
            status = "SUCCESS" if results["overall_mvp_status"] else "FAILURE"
            self.logger.info(
                f"MVP validation completed with {status} in "
                f"{results['duration_seconds']:.2f} seconds"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"MVP validation failed with unexpected error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            results["errors"].append(error_msg)
            results["duration_seconds"] = (datetime.now() - mvp_start).total_seconds()
            return results
    
    def _evaluate_mvp_criteria(self, 
                             validation_results: Optional[TestSuiteResult],
                             benchmark_results: Optional[BenchmarkSuite]) -> Dict[str, Any]:
        """
        Evaluate MVP success criteria against test results.
        
        Args:
            validation_results: Clinical metabolomics validation results
            benchmark_results: Performance benchmark results
            
        Returns:
            Dictionary with MVP criteria evaluation
        """
        criteria_evaluation = {
            "overall_success": False,
            "functional_criteria": {},
            "performance_criteria": {},
            "core_functionality": {},
            "failed_criteria": []
        }
        
        # Evaluate functional criteria
        if validation_results:
            pass_rate = validation_results.passed_questions / validation_results.total_questions
            
            criteria_evaluation["functional_criteria"] = {
                "pass_rate": {
                    "value": pass_rate,
                    "threshold": self.mvp_criteria["minimum_pass_rate"],
                    "passed": pass_rate >= self.mvp_criteria["minimum_pass_rate"]
                },
                "accuracy": {
                    "value": validation_results.average_accuracy,
                    "threshold": self.mvp_criteria["minimum_accuracy"],
                    "passed": validation_results.average_accuracy >= self.mvp_criteria["minimum_accuracy"]
                },
                "confidence": {
                    "value": validation_results.average_confidence,
                    "threshold": self.mvp_criteria["minimum_confidence"],
                    "passed": validation_results.average_confidence >= self.mvp_criteria["minimum_confidence"]
                }
            }
            
            # Check core clinical metabolomics question
            core_question_passed = any(
                result.passed and "clinical metabolomics" in result.question.lower()
                for result in validation_results.results
            )
            
            criteria_evaluation["core_functionality"]["clinical_metabolomics_question"] = {
                "passed": core_question_passed,
                "required": self.mvp_criteria["clinical_metabolomics_question_pass"]
            }
        
        # Evaluate performance criteria
        if benchmark_results:
            criteria_evaluation["performance_criteria"] = {
                "response_time": {
                    "value": benchmark_results.summary["average_response_time"],
                    "threshold": self.mvp_criteria["maximum_response_time"],
                    "passed": benchmark_results.summary["average_response_time"] <= self.mvp_criteria["maximum_response_time"]
                },
                "memory_usage": {
                    "value": benchmark_results.summary["max_memory_usage"],
                    "threshold": self.mvp_criteria["maximum_memory_usage"],
                    "passed": benchmark_results.summary["max_memory_usage"] <= self.mvp_criteria["maximum_memory_usage"]
                }
            }
            
            # Calculate throughput from load test results
            if benchmark_results.load_test_results:
                max_throughput = max(
                    result.requests_per_second 
                    for result in benchmark_results.load_test_results
                )
                criteria_evaluation["performance_criteria"]["throughput"] = {
                    "value": max_throughput,
                    "threshold": self.mvp_criteria["minimum_throughput"],
                    "passed": max_throughput >= self.mvp_criteria["minimum_throughput"]
                }
        
        # Determine overall success
        all_criteria = []
        
        # Collect all criteria results
        for category in ["functional_criteria", "performance_criteria"]:
            if category in criteria_evaluation:
                for criterion_name, criterion_data in criteria_evaluation[category].items():
                    all_criteria.append((f"{category}.{criterion_name}", criterion_data["passed"]))
        
        # Core functionality is mandatory
        if "core_functionality" in criteria_evaluation:
            for criterion_name, criterion_data in criteria_evaluation["core_functionality"].items():
                all_criteria.append((f"core_functionality.{criterion_name}", criterion_data["passed"]))
        
        # Overall success requires all criteria to pass
        criteria_evaluation["overall_success"] = all(passed for _, passed in all_criteria)
        
        # Track failed criteria
        criteria_evaluation["failed_criteria"] = [
            name for name, passed in all_criteria if not passed
        ]
        
        return criteria_evaluation
    
    def generate_mvp_report(self, mvp_results: Dict[str, Any], 
                          output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive MVP validation report.
        
        Args:
            mvp_results: MVP validation results
            output_file: Optional file to save report
            
        Returns:
            Report content as string
        """
        report_lines = [
            "=" * 80,
            "LIGHTRAG MVP VALIDATION REPORT",
            "=" * 80,
            f"Timestamp: {mvp_results['timestamp']}",
            f"Duration: {mvp_results['duration_seconds']:.2f} seconds",
            f"Overall Status: {'✅ MVP PASSED' if mvp_results['overall_mvp_status'] else '❌ MVP FAILED'}",
            ""
        ]
        
        # Validation tests summary
        if mvp_results["validation_tests"]:
            validation = mvp_results["validation_tests"]
            pass_rate = validation.passed_questions / validation.total_questions
            
            report_lines.extend([
                "CLINICAL METABOLOMICS VALIDATION:",
                f"  Total Questions: {validation.total_questions}",
                f"  Passed: {validation.passed_questions} ({pass_rate:.1%})",
                f"  Failed: {validation.failed_questions}",
                f"  Average Accuracy: {validation.average_accuracy:.3f}",
                f"  Average Confidence: {validation.average_confidence:.3f}",
                f"  Average Response Time: {validation.average_processing_time:.3f}s",
                ""
            ])
        
        # Performance benchmark summary
        if mvp_results["performance_benchmark"]:
            benchmark = mvp_results["performance_benchmark"]
            
            report_lines.extend([
                "PERFORMANCE BENCHMARK:",
                f"  Total Operations: {benchmark.summary['total_operations']}",
                f"  Successful: {benchmark.summary['successful_operations']}",
                f"  Failed: {benchmark.summary['failed_operations']}",
                f"  Average Response Time: {benchmark.summary['average_response_time']:.3f}s",
                f"  Max Response Time: {benchmark.summary['max_response_time']:.3f}s",
                f"  Average Memory Usage: {benchmark.summary['average_memory_usage']:.1f} MB",
                f"  Max Memory Usage: {benchmark.summary['max_memory_usage']:.1f} MB",
                ""
            ])
            
            if benchmark.load_test_results:
                best_load_test = max(
                    benchmark.load_test_results,
                    key=lambda x: x.requests_per_second
                )
                report_lines.extend([
                    "LOAD TEST (Best Performance):",
                    f"  Concurrent Users: {best_load_test.concurrent_users}",
                    f"  Requests/Second: {best_load_test.requests_per_second:.2f}",
                    f"  Success Rate: {(best_load_test.successful_requests/best_load_test.total_requests):.1%}",
                    f"  95th Percentile Response: {best_load_test.percentile_95_response_time:.3f}s",
                    ""
                ])
        
        # MVP criteria evaluation
        if mvp_results["mvp_criteria_evaluation"]:
            evaluation = mvp_results["mvp_criteria_evaluation"]
            
            report_lines.extend([
                "MVP CRITERIA EVALUATION:",
                "-" * 40
            ])
            
            # Functional criteria
            if "functional_criteria" in evaluation:
                report_lines.append("Functional Criteria:")
                for criterion, data in evaluation["functional_criteria"].items():
                    status = "✅ PASS" if data["passed"] else "❌ FAIL"
                    report_lines.append(
                        f"  {criterion}: {status} "
                        f"({data['value']:.3f} vs {data['threshold']:.3f})"
                    )
                report_lines.append("")
            
            # Performance criteria
            if "performance_criteria" in evaluation:
                report_lines.append("Performance Criteria:")
                for criterion, data in evaluation["performance_criteria"].items():
                    status = "✅ PASS" if data["passed"] else "❌ FAIL"
                    report_lines.append(
                        f"  {criterion}: {status} "
                        f"({data['value']:.3f} vs {data['threshold']:.3f})"
                    )
                report_lines.append("")
            
            # Core functionality
            if "core_functionality" in evaluation:
                report_lines.append("Core Functionality:")
                for criterion, data in evaluation["core_functionality"].items():
                    status = "✅ PASS" if data["passed"] else "❌ FAIL"
                    report_lines.append(f"  {criterion}: {status}")
                report_lines.append("")
            
            # Failed criteria
            if evaluation["failed_criteria"]:
                report_lines.extend([
                    "FAILED CRITERIA:",
                    "-" * 20
                ])
                for criterion in evaluation["failed_criteria"]:
                    report_lines.append(f"  ❌ {criterion}")
                report_lines.append("")
        
        # Errors
        if mvp_results["errors"]:
            report_lines.extend([
                "ERRORS ENCOUNTERED:",
                "-" * 20
            ])
            for error in mvp_results["errors"]:
                report_lines.append(f"  • {error}")
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 20
        ])
        
        if mvp_results["overall_mvp_status"]:
            report_lines.extend([
                "✅ MVP validation successful! The LightRAG integration meets all",
                "   required criteria and is ready for the next development phase.",
                "",
                "Next steps:",
                "  • Proceed with task 8: Integrate with existing Chainlit interface",
                "  • Begin implementation of intelligent query routing system",
                "  • Consider performance optimizations for production deployment"
            ])
        else:
            report_lines.extend([
                "❌ MVP validation failed. Address the following issues before",
                "   proceeding to the next development phase:",
                ""
            ])
            
            if mvp_results["mvp_criteria_evaluation"]:
                failed_criteria = mvp_results["mvp_criteria_evaluation"]["failed_criteria"]
                for criterion in failed_criteria:
                    report_lines.append(f"  • Fix {criterion}")
            
            report_lines.extend([
                "",
                "Consider:",
                "  • Reviewing and improving test question accuracy",
                "  • Optimizing performance bottlenecks",
                "  • Enhancing error handling and robustness",
                "  • Adding more comprehensive test coverage"
            ])
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report_content)
            self.logger.info(f"MVP report saved to {output_file}")
        
        return report_content
    
    def save_mvp_results(self, mvp_results: Dict[str, Any], output_file: str) -> None:
        """Save MVP results as JSON for analysis."""
        # Convert complex objects to serializable format
        serializable_results = mvp_results.copy()
        
        # Convert TestSuiteResult to dict if present
        if serializable_results["validation_tests"]:
            from dataclasses import asdict
            validation_dict = asdict(serializable_results["validation_tests"])
            validation_dict["timestamp"] = serializable_results["validation_tests"].timestamp.isoformat()
            serializable_results["validation_tests"] = validation_dict
        
        # Convert BenchmarkSuite to dict if present
        if serializable_results["performance_benchmark"]:
            from dataclasses import asdict
            benchmark_dict = asdict(serializable_results["performance_benchmark"])
            benchmark_dict["timestamp"] = serializable_results["performance_benchmark"].timestamp.isoformat()
            
            # Convert datetime objects in metrics
            for metric in benchmark_dict["individual_metrics"]:
                metric["start_time"] = metric["start_time"].isoformat() if hasattr(metric["start_time"], "isoformat") else metric["start_time"]
                metric["end_time"] = metric["end_time"].isoformat() if hasattr(metric["end_time"], "isoformat") else metric["end_time"]
            
            serializable_results["performance_benchmark"] = benchmark_dict
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"MVP results saved to {output_file}")


async def main():
    """Main entry point for MVP test runner."""
    parser = argparse.ArgumentParser(description="LightRAG MVP Validation Test Runner")
    parser.add_argument("--output-dir", default="mvp_test_results", 
                       help="Output directory for results")
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
    
    # Create test runner
    runner = MVPTestRunner(output_dir=args.output_dir)
    
    try:
        # Run complete MVP validation
        results = await runner.run_complete_mvp_validation()
        
        # Generate timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate and display report
        report = runner.generate_mvp_report(
            results, 
            str(runner.output_dir / f"mvp_report_{timestamp}.txt")
        )
        print(report)
        
        # Save detailed results if requested
        if args.save_results:
            runner.save_mvp_results(
                results, 
                str(runner.output_dir / f"mvp_results_{timestamp}.json")
            )
        
        # Exit with appropriate code
        if results["overall_mvp_status"]:
            print("\n🎉 MVP VALIDATION SUCCESSFUL!")
            sys.exit(0)
        else:
            print("\n💥 MVP VALIDATION FAILED!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nMVP validation interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"MVP validation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())