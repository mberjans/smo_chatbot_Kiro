"""
Automated Testing Pipeline for LightRAG MVP Validation

This module provides automated testing capabilities for continuous validation
of the LightRAG integration, including scheduled testing, reporting, and
integration with CI/CD systems.
"""

import asyncio
import logging
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from .clinical_metabolomics_suite import (
    ClinicalMetabolomicsTestSuite,
    run_mvp_validation_test,
    TestSuiteResult
)
from ..config.settings import LightRAGConfig
from ..utils.logging import setup_logger


class AutomatedTestingPipeline:
    """
    Automated testing pipeline for LightRAG MVP validation.
    
    This class provides methods for running automated tests, generating reports,
    and integrating with continuous integration systems.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None, 
                 output_dir: Optional[str] = None):
        """
        Initialize the automated testing pipeline.
        
        Args:
            config: Optional LightRAG configuration
            output_dir: Directory for test outputs and reports
        """
        self.config = config or LightRAGConfig.from_env()
        self.output_dir = Path(output_dir or "test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("automated_pipeline", 
                                 log_file=str(self.output_dir / "pipeline.log"))
        
        # Test configuration
        self.success_criteria = {
            "minimum_pass_rate": 0.85,  # 85% of tests must pass
            "minimum_accuracy": 0.75,   # Average accuracy must be 75%
            "minimum_confidence": 0.6,  # Average confidence must be 60%
            "maximum_response_time": 5.0  # Average response time under 5 seconds
        }
    
    async def run_validation_pipeline(self, 
                                    save_results: bool = True,
                                    generate_report: bool = True) -> Dict[str, Any]:
        """
        Run the complete validation pipeline.
        
        Args:
            save_results: Whether to save detailed results
            generate_report: Whether to generate human-readable report
            
        Returns:
            Dictionary with pipeline results and status
        """
        pipeline_start = datetime.now()
        self.logger.info("Starting automated validation pipeline")
        
        try:
            # Run MVP validation test
            self.logger.info("Running MVP validation test suite")
            test_results = await run_mvp_validation_test(self.config)
            
            # Evaluate success criteria
            success_evaluation = self._evaluate_success_criteria(test_results)
            
            # Generate timestamp for this run
            timestamp = pipeline_start.strftime("%Y%m%d_%H%M%S")
            
            # Save results if requested
            if save_results:
                results_file = self.output_dir / f"validation_results_{timestamp}.json"
                test_suite = ClinicalMetabolomicsTestSuite()
                test_suite.save_results_json(test_results, str(results_file))
                self.logger.info(f"Results saved to {results_file}")
            
            # Generate report if requested
            report_content = None
            if generate_report:
                report_file = self.output_dir / f"validation_report_{timestamp}.txt"
                test_suite = ClinicalMetabolomicsTestSuite()
                report_content = test_suite.generate_report(test_results, str(report_file))
                self.logger.info(f"Report saved to {report_file}")
            
            # Create pipeline summary
            pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
            
            pipeline_result = {
                "timestamp": pipeline_start.isoformat(),
                "duration_seconds": pipeline_duration,
                "success": success_evaluation["overall_success"],
                "test_results_summary": {
                    "total_questions": test_results.total_questions,
                    "passed_questions": test_results.passed_questions,
                    "failed_questions": test_results.failed_questions,
                    "pass_rate": test_results.passed_questions / test_results.total_questions,
                    "average_accuracy": test_results.average_accuracy,
                    "average_confidence": test_results.average_confidence,
                    "average_processing_time": test_results.average_processing_time
                },
                "success_criteria_evaluation": success_evaluation,
                "files_generated": {
                    "results_file": str(results_file) if save_results else None,
                    "report_file": str(report_file) if generate_report else None
                }
            }
            
            # Log summary
            status = "SUCCESS" if success_evaluation["overall_success"] else "FAILURE"
            self.logger.info(
                f"Pipeline completed with {status} in {pipeline_duration:.2f}s. "
                f"Pass rate: {pipeline_result['test_results_summary']['pass_rate']:.1%}, "
                f"Accuracy: {test_results.average_accuracy:.3f}"
            )
            
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
            
            return {
                "timestamp": pipeline_start.isoformat(),
                "duration_seconds": (datetime.now() - pipeline_start).total_seconds(),
                "success": False,
                "error": str(e),
                "test_results_summary": None,
                "success_criteria_evaluation": None,
                "files_generated": None
            }
    
    def _evaluate_success_criteria(self, test_results: TestSuiteResult) -> Dict[str, Any]:
        """
        Evaluate test results against success criteria.
        
        Args:
            test_results: Test suite results to evaluate
            
        Returns:
            Dictionary with evaluation results
        """
        pass_rate = test_results.passed_questions / test_results.total_questions
        
        criteria_results = {
            "pass_rate": {
                "value": pass_rate,
                "threshold": self.success_criteria["minimum_pass_rate"],
                "passed": pass_rate >= self.success_criteria["minimum_pass_rate"]
            },
            "accuracy": {
                "value": test_results.average_accuracy,
                "threshold": self.success_criteria["minimum_accuracy"],
                "passed": test_results.average_accuracy >= self.success_criteria["minimum_accuracy"]
            },
            "confidence": {
                "value": test_results.average_confidence,
                "threshold": self.success_criteria["minimum_confidence"],
                "passed": test_results.average_confidence >= self.success_criteria["minimum_confidence"]
            },
            "response_time": {
                "value": test_results.average_processing_time,
                "threshold": self.success_criteria["maximum_response_time"],
                "passed": test_results.average_processing_time <= self.success_criteria["maximum_response_time"]
            }
        }
        
        # Overall success requires all criteria to pass
        overall_success = all(criteria["passed"] for criteria in criteria_results.values())
        
        return {
            "overall_success": overall_success,
            "criteria": criteria_results,
            "failed_criteria": [
                name for name, criteria in criteria_results.items() 
                if not criteria["passed"]
            ]
        }
    
    async def run_continuous_validation(self, 
                                      interval_hours: int = 24,
                                      max_runs: Optional[int] = None) -> None:
        """
        Run continuous validation at specified intervals.
        
        Args:
            interval_hours: Hours between validation runs
            max_runs: Maximum number of runs (None for unlimited)
        """
        self.logger.info(
            f"Starting continuous validation with {interval_hours}h intervals"
            + (f" (max {max_runs} runs)" if max_runs else "")
        )
        
        run_count = 0
        
        while max_runs is None or run_count < max_runs:
            try:
                # Run validation pipeline
                result = await self.run_validation_pipeline()
                
                run_count += 1
                status = "SUCCESS" if result["success"] else "FAILURE"
                
                self.logger.info(
                    f"Continuous validation run {run_count} completed with {status}"
                )
                
                # If not the last run, wait for next interval
                if max_runs is None or run_count < max_runs:
                    self.logger.info(f"Waiting {interval_hours} hours until next run")
                    await asyncio.sleep(interval_hours * 3600)
                    
            except KeyboardInterrupt:
                self.logger.info("Continuous validation interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error in continuous validation: {str(e)}", exc_info=True)
                # Wait before retrying
                await asyncio.sleep(300)  # 5 minutes
    
    def generate_trend_report(self, days: int = 7) -> str:
        """
        Generate a trend report from recent test results.
        
        Args:
            days: Number of days to include in trend analysis
            
        Returns:
            Trend report as string
        """
        # Find recent result files
        cutoff_date = datetime.now() - timedelta(days=days)
        result_files = []
        
        for file_path in self.output_dir.glob("validation_results_*.json"):
            try:
                # Extract timestamp from filename
                timestamp_str = file_path.stem.split("_", 2)[-1]
                file_date = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                
                if file_date >= cutoff_date:
                    result_files.append((file_date, file_path))
            except (ValueError, IndexError):
                continue
        
        if not result_files:
            return f"No validation results found in the last {days} days."
        
        # Sort by date
        result_files.sort(key=lambda x: x[0])
        
        # Load and analyze results
        trend_data = []
        for date, file_path in result_files:
            try:
                with open(file_path) as f:
                    data = json.load(f)
                
                trend_data.append({
                    "date": date,
                    "pass_rate": data["passed_questions"] / data["total_questions"],
                    "accuracy": data["average_accuracy"],
                    "confidence": data["average_confidence"],
                    "response_time": data["average_processing_time"]
                })
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {str(e)}")
                continue
        
        if not trend_data:
            return f"No valid validation results found in the last {days} days."
        
        # Generate trend report
        report_lines = [
            f"VALIDATION TREND REPORT ({days} days)",
            "=" * 50,
            f"Period: {trend_data[0]['date'].strftime('%Y-%m-%d')} to {trend_data[-1]['date'].strftime('%Y-%m-%d')}",
            f"Total Runs: {len(trend_data)}",
            "",
            "METRICS TRENDS:",
        ]
        
        # Calculate trends
        if len(trend_data) > 1:
            first = trend_data[0]
            last = trend_data[-1]
            
            metrics = ["pass_rate", "accuracy", "confidence", "response_time"]
            for metric in metrics:
                first_val = first[metric]
                last_val = last[metric]
                change = last_val - first_val
                change_pct = (change / first_val) * 100 if first_val != 0 else 0
                
                trend_symbol = "↑" if change > 0 else "↓" if change < 0 else "→"
                
                report_lines.append(
                    f"  {metric.replace('_', ' ').title()}: "
                    f"{first_val:.3f} → {last_val:.3f} "
                    f"({change:+.3f}, {change_pct:+.1f}%) {trend_symbol}"
                )
        
        report_lines.extend([
            "",
            "RECENT RESULTS:",
            "-" * 30
        ])
        
        # Show last 5 results
        for data in trend_data[-5:]:
            report_lines.append(
                f"{data['date'].strftime('%Y-%m-%d %H:%M')}: "
                f"Pass={data['pass_rate']:.1%}, "
                f"Acc={data['accuracy']:.3f}, "
                f"Conf={data['confidence']:.3f}, "
                f"Time={data['response_time']:.2f}s"
            )
        
        return "\n".join(report_lines)
    
    def cleanup_old_results(self, days: int = 30) -> int:
        """
        Clean up old test result files.
        
        Args:
            days: Age threshold for cleanup (files older than this are removed)
            
        Returns:
            Number of files removed
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        removed_count = 0
        
        for file_path in self.output_dir.glob("validation_*"):
            try:
                # Check file modification time
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                
                if file_mtime < cutoff_date:
                    file_path.unlink()
                    removed_count += 1
                    self.logger.info(f"Removed old result file: {file_path}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to remove {file_path}: {str(e)}")
        
        self.logger.info(f"Cleanup completed: removed {removed_count} old files")
        return removed_count


async def main():
    """Main entry point for automated testing pipeline."""
    parser = argparse.ArgumentParser(description="LightRAG Automated Testing Pipeline")
    parser.add_argument("--mode", choices=["single", "continuous", "trend"], 
                       default="single", help="Testing mode")
    parser.add_argument("--output-dir", default="test_results", 
                       help="Output directory for results")
    parser.add_argument("--interval", type=int, default=24, 
                       help="Hours between continuous runs")
    parser.add_argument("--max-runs", type=int, 
                       help="Maximum runs for continuous mode")
    parser.add_argument("--trend-days", type=int, default=7, 
                       help="Days for trend analysis")
    parser.add_argument("--cleanup-days", type=int, default=30, 
                       help="Age threshold for cleanup")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create pipeline
    pipeline = AutomatedTestingPipeline(output_dir=args.output_dir)
    
    try:
        if args.mode == "single":
            # Run single validation
            result = await pipeline.run_validation_pipeline()
            
            if result["success"]:
                print("✅ Validation PASSED")
                sys.exit(0)
            else:
                print("❌ Validation FAILED")
                if "error" in result:
                    print(f"Error: {result['error']}")
                sys.exit(1)
                
        elif args.mode == "continuous":
            # Run continuous validation
            await pipeline.run_continuous_validation(
                interval_hours=args.interval,
                max_runs=args.max_runs
            )
            
        elif args.mode == "trend":
            # Generate trend report
            report = pipeline.generate_trend_report(days=args.trend_days)
            print(report)
            
            # Optional cleanup
            if args.cleanup_days:
                pipeline.cleanup_old_results(days=args.cleanup_days)
                
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())