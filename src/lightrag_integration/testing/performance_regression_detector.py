"""
Performance Regression Detection System

This module provides automated detection of performance regressions by
comparing current performance metrics against historical baselines and
detecting significant degradations in system performance.
"""

import json
import logging
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from ..utils.logging import setup_logger


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    metric_name: str
    baseline_value: float
    acceptable_variance: float  # Percentage variance allowed
    measurement_count: int
    last_updated: datetime
    historical_values: List[float]


@dataclass
class RegressionAlert:
    """Performance regression alert."""
    metric_name: str
    current_value: float
    baseline_value: float
    regression_percentage: float
    severity: str  # "minor", "moderate", "severe", "critical"
    timestamp: datetime
    description: str
    recommended_actions: List[str]


@dataclass
class RegressionAnalysisResult:
    """Result of regression analysis."""
    timestamp: datetime
    total_metrics_analyzed: int
    regressions_detected: int
    alerts: List[RegressionAlert]
    overall_performance_trend: str  # "improving", "stable", "degrading"
    summary: Dict[str, Any]


class PerformanceRegressionDetector:
    """
    Automated performance regression detection system.
    
    This class analyzes performance metrics against historical baselines
    to detect regressions and generate alerts for performance degradations.
    """
    
    def __init__(self, baseline_file: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the regression detector.
        
        Args:
            baseline_file: Path to baseline metrics file
            output_dir: Directory for regression reports
        """
        self.output_dir = Path(output_dir or "regression_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_file = baseline_file or str(self.output_dir / "performance_baselines.json")
        
        self.logger = setup_logger("performance_regression_detector",
                                 log_file=str(self.output_dir / "regression_detector.log"))
        
        # Load existing baselines
        self.baselines = self._load_baselines()
        
        # Regression thresholds
        self.regression_thresholds = {
            "minor": 10.0,      # 10% degradation
            "moderate": 25.0,   # 25% degradation
            "severe": 50.0,     # 50% degradation
            "critical": 100.0   # 100% degradation
        }
        
        # Metric configurations
        self.metric_configs = {
            "response_time": {
                "direction": "lower_is_better",
                "acceptable_variance": 15.0,  # 15%
                "critical_threshold": 10.0    # 10 seconds
            },
            "memory_usage": {
                "direction": "lower_is_better", 
                "acceptable_variance": 20.0,  # 20%
                "critical_threshold": 4096    # 4GB
            },
            "throughput": {
                "direction": "higher_is_better",
                "acceptable_variance": 10.0,  # 10%
                "critical_threshold": 0.5     # 0.5 req/s
            },
            "success_rate": {
                "direction": "higher_is_better",
                "acceptable_variance": 2.0,   # 2%
                "critical_threshold": 0.95    # 95%
            },
            "cpu_usage": {
                "direction": "lower_is_better",
                "acceptable_variance": 15.0,  # 15%
                "critical_threshold": 90.0    # 90%
            }
        }
    
    def _load_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Load performance baselines from file."""
        if not Path(self.baseline_file).exists():
            self.logger.info("No baseline file found, starting with empty baselines")
            return {}
        
        try:
            with open(self.baseline_file) as f:
                baseline_data = json.load(f)
            
            baselines = {}
            for metric_name, data in baseline_data.items():
                baselines[metric_name] = PerformanceBaseline(
                    metric_name=data["metric_name"],
                    baseline_value=data["baseline_value"],
                    acceptable_variance=data["acceptable_variance"],
                    measurement_count=data["measurement_count"],
                    last_updated=datetime.fromisoformat(data["last_updated"]),
                    historical_values=data["historical_values"]
                )
            
            self.logger.info(f"Loaded {len(baselines)} performance baselines")
            return baselines
            
        except Exception as e:
            self.logger.error(f"Failed to load baselines: {str(e)}")
            return {}
    
    def _save_baselines(self) -> None:
        """Save performance baselines to file."""
        try:
            baseline_data = {}
            for metric_name, baseline in self.baselines.items():
                baseline_dict = asdict(baseline)
                baseline_dict["last_updated"] = baseline.last_updated.isoformat()
                baseline_data[metric_name] = baseline_dict
            
            with open(self.baseline_file, 'w') as f:
                json.dump(baseline_data, f, indent=2)
            
            self.logger.info(f"Saved {len(self.baselines)} performance baselines")
            
        except Exception as e:
            self.logger.error(f"Failed to save baselines: {str(e)}")
    
    def update_baseline(self, metric_name: str, value: float,
                       force_update: bool = False) -> None:
        """
        Update baseline for a performance metric.
        
        Args:
            metric_name: Name of the performance metric
            value: New measurement value
            force_update: Force update even if value is worse
        """
        config = self.metric_configs.get(metric_name, {})
        acceptable_variance = config.get("acceptable_variance", 15.0)
        
        if metric_name not in self.baselines:
            # Create new baseline
            self.baselines[metric_name] = PerformanceBaseline(
                metric_name=metric_name,
                baseline_value=value,
                acceptable_variance=acceptable_variance,
                measurement_count=1,
                last_updated=datetime.now(),
                historical_values=[value]
            )
            self.logger.info(f"Created new baseline for {metric_name}: {value}")
            
        else:
            baseline = self.baselines[metric_name]
            
            # Add to historical values
            baseline.historical_values.append(value)
            baseline.measurement_count += 1
            baseline.last_updated = datetime.now()
            
            # Keep only recent historical values (last 100)
            if len(baseline.historical_values) > 100:
                baseline.historical_values = baseline.historical_values[-100:]
            
            # Update baseline if performance improved or force update
            should_update = force_update
            
            if not force_update:
                direction = config.get("direction", "lower_is_better")
                if direction == "lower_is_better":
                    should_update = value < baseline.baseline_value
                else:
                    should_update = value > baseline.baseline_value
            
            if should_update:
                old_value = baseline.baseline_value
                baseline.baseline_value = value
                self.logger.info(
                    f"Updated baseline for {metric_name}: {old_value} -> {value}"
                )
        
        # Save updated baselines
        self._save_baselines()
    
    def analyze_performance_metrics(self, 
                                  current_metrics: Dict[str, float]) -> RegressionAnalysisResult:
        """
        Analyze current performance metrics for regressions.
        
        Args:
            current_metrics: Dictionary of current performance metrics
            
        Returns:
            RegressionAnalysisResult with detected regressions
        """
        self.logger.info(f"Analyzing {len(current_metrics)} performance metrics")
        
        alerts = []
        metrics_analyzed = 0
        regressions_detected = 0
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.baselines:
                self.logger.warning(f"No baseline found for metric: {metric_name}")
                continue
            
            metrics_analyzed += 1
            baseline = self.baselines[metric_name]
            
            # Detect regression
            regression_alert = self._detect_regression(
                metric_name, current_value, baseline
            )
            
            if regression_alert:
                alerts.append(regression_alert)
                regressions_detected += 1
                
                self.logger.warning(
                    f"Regression detected in {metric_name}: "
                    f"{regression_alert.regression_percentage:.1f}% degradation"
                )
        
        # Determine overall performance trend
        overall_trend = self._determine_overall_trend(current_metrics)
        
        # Generate summary
        summary = self._generate_regression_summary(alerts, metrics_analyzed)
        
        return RegressionAnalysisResult(
            timestamp=datetime.now(),
            total_metrics_analyzed=metrics_analyzed,
            regressions_detected=regressions_detected,
            alerts=alerts,
            overall_performance_trend=overall_trend,
            summary=summary
        )
    
    def _detect_regression(self, metric_name: str, current_value: float,
                         baseline: PerformanceBaseline) -> Optional[RegressionAlert]:
        """Detect if a metric shows performance regression."""
        config = self.metric_configs.get(metric_name, {})
        direction = config.get("direction", "lower_is_better")
        
        # Calculate regression percentage
        if direction == "lower_is_better":
            # For metrics where lower is better (response time, memory usage)
            if current_value <= baseline.baseline_value:
                return None  # No regression, performance improved or stayed same
            
            regression_percentage = ((current_value - baseline.baseline_value) / 
                                   baseline.baseline_value) * 100
        else:
            # For metrics where higher is better (throughput, success rate)
            if current_value >= baseline.baseline_value:
                return None  # No regression, performance improved or stayed same
            
            regression_percentage = ((baseline.baseline_value - current_value) / 
                                   baseline.baseline_value) * 100
        
        # Check if regression exceeds acceptable variance
        if regression_percentage <= baseline.acceptable_variance:
            return None  # Within acceptable variance
        
        # Determine severity
        severity = self._determine_severity(regression_percentage)
        
        # Generate description and recommendations
        description = self._generate_regression_description(
            metric_name, current_value, baseline.baseline_value, 
            regression_percentage, direction
        )
        
        recommendations = self._generate_regression_recommendations(
            metric_name, severity, regression_percentage
        )
        
        return RegressionAlert(
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline.baseline_value,
            regression_percentage=regression_percentage,
            severity=severity,
            timestamp=datetime.now(),
            description=description,
            recommended_actions=recommendations
        )
    
    def _determine_severity(self, regression_percentage: float) -> str:
        """Determine severity level based on regression percentage."""
        if regression_percentage >= self.regression_thresholds["critical"]:
            return "critical"
        elif regression_percentage >= self.regression_thresholds["severe"]:
            return "severe"
        elif regression_percentage >= self.regression_thresholds["moderate"]:
            return "moderate"
        else:
            return "minor"
    
    def _generate_regression_description(self, metric_name: str, current_value: float,
                                       baseline_value: float, regression_percentage: float,
                                       direction: str) -> str:
        """Generate human-readable regression description."""
        if direction == "lower_is_better":
            return (f"{metric_name} increased from {baseline_value:.3f} to {current_value:.3f} "
                   f"({regression_percentage:.1f}% degradation)")
        else:
            return (f"{metric_name} decreased from {baseline_value:.3f} to {current_value:.3f} "
                   f"({regression_percentage:.1f}% degradation)")
    
    def _generate_regression_recommendations(self, metric_name: str, severity: str,
                                           regression_percentage: float) -> List[str]:
        """Generate recommendations based on regression type and severity."""
        recommendations = []
        
        # General recommendations based on severity
        if severity == "critical":
            recommendations.extend([
                "CRITICAL: Immediate investigation required",
                "Consider rolling back recent changes",
                "Alert development team immediately"
            ])
        elif severity == "severe":
            recommendations.extend([
                "Urgent investigation needed",
                "Review recent code changes and deployments",
                "Consider performance optimization"
            ])
        elif severity == "moderate":
            recommendations.extend([
                "Schedule performance investigation",
                "Monitor trend over next few measurements"
            ])
        else:
            recommendations.append("Monitor for continued degradation")
        
        # Metric-specific recommendations
        if metric_name == "response_time":
            recommendations.extend([
                "Check database query performance",
                "Review caching effectiveness",
                "Analyze system resource usage"
            ])
        elif metric_name == "memory_usage":
            recommendations.extend([
                "Investigate potential memory leaks",
                "Review object lifecycle management",
                "Check garbage collection efficiency"
            ])
        elif metric_name == "throughput":
            recommendations.extend([
                "Analyze system bottlenecks",
                "Check connection pool settings",
                "Review concurrent processing limits"
            ])
        elif metric_name == "success_rate":
            recommendations.extend([
                "Investigate error patterns",
                "Review error handling logic",
                "Check external dependency health"
            ])
        
        return recommendations
    
    def _determine_overall_trend(self, current_metrics: Dict[str, float]) -> str:
        """Determine overall performance trend."""
        trend_scores = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.baselines:
                continue
            
            baseline = self.baselines[metric_name]
            config = self.metric_configs.get(metric_name, {})
            direction = config.get("direction", "lower_is_better")
            
            # Calculate trend score (-1 to 1, where 1 is improvement)
            if direction == "lower_is_better":
                if baseline.baseline_value > 0:
                    trend_score = (baseline.baseline_value - current_value) / baseline.baseline_value
                else:
                    trend_score = 0
            else:
                if baseline.baseline_value > 0:
                    trend_score = (current_value - baseline.baseline_value) / baseline.baseline_value
                else:
                    trend_score = 0
            
            # Cap trend score at -1 to 1
            trend_score = max(-1, min(1, trend_score))
            trend_scores.append(trend_score)
        
        if not trend_scores:
            return "unknown"
        
        average_trend = statistics.mean(trend_scores)
        
        if average_trend > 0.1:
            return "improving"
        elif average_trend < -0.1:
            return "degrading"
        else:
            return "stable"
    
    def _generate_regression_summary(self, alerts: List[RegressionAlert],
                                   metrics_analyzed: int) -> Dict[str, Any]:
        """Generate summary of regression analysis."""
        if not alerts:
            return {
                "regression_rate": 0.0,
                "severity_distribution": {},
                "most_affected_metrics": [],
                "average_regression": 0.0
            }
        
        # Calculate severity distribution
        severity_counts = {}
        for alert in alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
        
        # Find most affected metrics
        most_affected = sorted(alerts, key=lambda x: x.regression_percentage, reverse=True)[:5]
        
        # Calculate average regression
        average_regression = statistics.mean([alert.regression_percentage for alert in alerts])
        
        return {
            "regression_rate": len(alerts) / metrics_analyzed if metrics_analyzed > 0 else 0.0,
            "severity_distribution": severity_counts,
            "most_affected_metrics": [
                {
                    "metric": alert.metric_name,
                    "regression_percentage": alert.regression_percentage,
                    "severity": alert.severity
                }
                for alert in most_affected
            ],
            "average_regression": average_regression
        }
    
    def generate_regression_report(self, analysis_result: RegressionAnalysisResult,
                                 output_file: Optional[str] = None) -> str:
        """Generate comprehensive regression analysis report."""
        report_lines = [
            "=" * 80,
            "PERFORMANCE REGRESSION ANALYSIS REPORT",
            "=" * 80,
            f"Timestamp: {analysis_result.timestamp.isoformat()}",
            f"Metrics Analyzed: {analysis_result.total_metrics_analyzed}",
            f"Regressions Detected: {analysis_result.regressions_detected}",
            f"Overall Trend: {analysis_result.overall_performance_trend.title()}",
            ""
        ]
        
        # Summary
        summary = analysis_result.summary
        if analysis_result.regressions_detected > 0:
            report_lines.extend([
                "REGRESSION SUMMARY:",
                f"  Regression Rate: {summary['regression_rate']:.1%}",
                f"  Average Regression: {summary['average_regression']:.1f}%",
                ""
            ])
            
            # Severity distribution
            if summary["severity_distribution"]:
                report_lines.append("  Severity Distribution:")
                for severity, count in summary["severity_distribution"].items():
                    report_lines.append(f"    {severity.title()}: {count}")
                report_lines.append("")
        
        # Detailed alerts
        if analysis_result.alerts:
            report_lines.extend([
                "REGRESSION ALERTS:",
                "-" * 40
            ])
            
            # Group alerts by severity
            alerts_by_severity = {}
            for alert in analysis_result.alerts:
                if alert.severity not in alerts_by_severity:
                    alerts_by_severity[alert.severity] = []
                alerts_by_severity[alert.severity].append(alert)
            
            # Display alerts by severity (critical first)
            severity_order = ["critical", "severe", "moderate", "minor"]
            
            for severity in severity_order:
                if severity not in alerts_by_severity:
                    continue
                
                report_lines.append(f"\n{severity.upper()} REGRESSIONS:")
                
                for alert in alerts_by_severity[severity]:
                    icon = "🚨" if severity == "critical" else "⚠️" if severity == "severe" else "⚡"
                    
                    report_lines.extend([
                        f"{icon} {alert.metric_name}:",
                        f"  {alert.description}",
                        f"  Severity: {alert.severity.title()}",
                        f"  Current: {alert.current_value:.3f}",
                        f"  Baseline: {alert.baseline_value:.3f}",
                        ""
                    ])
                    
                    if alert.recommended_actions:
                        report_lines.append("  Recommended Actions:")
                        for action in alert.recommended_actions:
                            report_lines.append(f"    • {action}")
                        report_lines.append("")
        else:
            report_lines.extend([
                "✅ NO PERFORMANCE REGRESSIONS DETECTED",
                "",
                "All metrics are performing within acceptable variance of baselines."
            ])
        
        # Overall recommendations
        report_lines.extend([
            "OVERALL RECOMMENDATIONS:",
            "-" * 30
        ])
        
        if analysis_result.regressions_detected == 0:
            report_lines.extend([
                "• Continue monitoring performance metrics",
                "• Consider updating baselines if consistent improvements observed",
                "• Maintain current performance optimization practices"
            ])
        else:
            critical_count = sum(1 for alert in analysis_result.alerts if alert.severity == "critical")
            severe_count = sum(1 for alert in analysis_result.alerts if alert.severity == "severe")
            
            if critical_count > 0:
                report_lines.extend([
                    f"• URGENT: Address {critical_count} critical regression(s) immediately",
                    "• Consider emergency rollback if recent deployment caused regressions",
                    "• Implement immediate monitoring and alerting"
                ])
            
            if severe_count > 0:
                report_lines.extend([
                    f"• High priority: Investigate {severe_count} severe regression(s)",
                    "• Schedule performance optimization work",
                    "• Review recent changes for performance impact"
                ])
            
            report_lines.extend([
                "• Increase monitoring frequency until regressions resolved",
                "• Consider performance testing before future deployments",
                "• Update performance baselines after fixes implemented"
            ])
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report_content)
            self.logger.info(f"Regression analysis report saved to {output_file}")
        
        return report_content
    
    def save_analysis_results(self, analysis_result: RegressionAnalysisResult,
                            output_file: str) -> None:
        """Save regression analysis results as JSON."""
        # Convert to serializable format
        result_dict = asdict(analysis_result)
        result_dict["timestamp"] = analysis_result.timestamp.isoformat()
        
        for alert in result_dict["alerts"]:
            alert["timestamp"] = alert["timestamp"].isoformat() if hasattr(alert["timestamp"], "isoformat") else alert["timestamp"]
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        self.logger.info(f"Regression analysis results saved to {output_file}")


# Convenience function for regression analysis
def analyze_performance_regression(current_metrics: Dict[str, float],
                                 baseline_file: Optional[str] = None,
                                 output_dir: Optional[str] = None) -> RegressionAnalysisResult:
    """
    Analyze performance metrics for regressions.
    
    Args:
        current_metrics: Dictionary of current performance metrics
        baseline_file: Path to baseline metrics file
        output_dir: Directory for regression reports
        
    Returns:
        RegressionAnalysisResult with analysis
    """
    detector = PerformanceRegressionDetector(
        baseline_file=baseline_file,
        output_dir=output_dir
    )
    
    return detector.analyze_performance_metrics(current_metrics)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Regression Detector")
    parser.add_argument("--metrics-file", required=True,
                       help="JSON file with current performance metrics")
    parser.add_argument("--baseline-file",
                       help="JSON file with performance baselines")
    parser.add_argument("--output-dir", default="regression_analysis",
                       help="Output directory for analysis results")
    parser.add_argument("--update-baselines", action="store_true",
                       help="Update baselines with current metrics")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Load current metrics
        with open(args.metrics_file) as f:
            current_metrics = json.load(f)
        
        # Create detector
        detector = PerformanceRegressionDetector(
            baseline_file=args.baseline_file,
            output_dir=args.output_dir
        )
        
        # Update baselines if requested
        if args.update_baselines:
            for metric_name, value in current_metrics.items():
                detector.update_baseline(metric_name, value)
            print("Baselines updated.")
        
        # Analyze for regressions
        analysis_result = detector.analyze_performance_metrics(current_metrics)
        
        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = detector.generate_regression_report(
            analysis_result,
            str(detector.output_dir / f"regression_report_{timestamp}.txt")
        )
        print(report)
        
        # Save results
        detector.save_analysis_results(
            analysis_result,
            str(detector.output_dir / f"regression_analysis_{timestamp}.json")
        )
        
        # Exit with appropriate code
        if analysis_result.regressions_detected == 0:
            print("\n✅ NO REGRESSIONS DETECTED!")
            exit(0)
        else:
            critical_regressions = sum(
                1 for alert in analysis_result.alerts 
                if alert.severity == "critical"
            )
            if critical_regressions > 0:
                print(f"\n🚨 {critical_regressions} CRITICAL REGRESSIONS DETECTED!")
                exit(2)
            else:
                print(f"\n⚠️ {analysis_result.regressions_detected} REGRESSIONS DETECTED!")
                exit(1)
                
    except Exception as e:
        print(f"Regression analysis failed: {str(e)}")
        exit(1)