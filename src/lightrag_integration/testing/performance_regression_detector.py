"""
Performance Regression Detection System

This module provides automated detection of performance regressions by
comparing current test results with historical baselines and identifying
significant performance degradations.
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
    measurement_unit: str
    last_updated: datetime
    sample_count: int
    confidence_interval: Tuple[float, float]


@dataclass
class RegressionDetectionResult:
    """Result of regression detection analysis."""
    metric_name: str
    current_value: float
    baseline_value: float
    variance_percentage: float
    is_regression: bool
    severity: str  # "minor", "moderate", "severe", "critical"
    confidence_level: float
    recommendation: str


@dataclass
class PerformanceRegressionReport:
    """Comprehensive performance regression report."""
    timestamp: datetime
    test_run_id: str
    overall_regression_detected: bool
    total_metrics_analyzed: int
    regressions_detected: int
    regression_results: List[RegressionDetectionResult]
    performance_trends: Dict[str, Any]
    recommendations: List[str]


class PerformanceRegressionDetector:
    """
    Performance regression detector that analyzes test results against
    historical baselines to identify performance degradations.
    """
    
    def __init__(self, baseline_dir: Optional[str] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the performance regression detector.
        
        Args:
            baseline_dir: Directory containing performance baselines
            output_dir: Directory for regression analysis outputs
        """
        self.baseline_dir = Path(baseline_dir or "performance_baselines")
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        
        self.output_dir = Path(output_dir or "regression_analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("performance_regression_detector",
                                 log_file=str(self.output_dir / "regression_detection.log"))
        
        # Regression detection thresholds
        self.regression_thresholds = {
            "response_time": {
                "minor": 0.10,      # 10% increase
                "moderate": 0.25,   # 25% increase
                "severe": 0.50,     # 50% increase
                "critical": 1.00    # 100% increase
            },
            "throughput": {
                "minor": -0.10,     # 10% decrease
                "moderate": -0.25,  # 25% decrease
                "severe": -0.50,    # 50% decrease
                "critical": -0.75   # 75% decrease
            },
            "success_rate": {
                "minor": -0.02,     # 2% decrease
                "moderate": -0.05,  # 5% decrease
                "severe": -0.10,    # 10% decrease
                "critical": -0.20   # 20% decrease
            },
            "memory_usage": {
                "minor": 0.15,      # 15% increase
                "moderate": 0.30,   # 30% increase
                "severe": 0.60,     # 60% increase
                "critical": 1.20    # 120% increase
            },
            "cpu_usage": {
                "minor": 0.15,      # 15% increase
                "moderate": 0.30,   # 30% increase
                "severe": 0.60,     # 60% increase
                "critical": 1.20    # 120% increase
            }
        }
        
        # Load existing baselines
        self.baselines = self._load_baselines()
    
    def _load_baselines(self) -> Dict[str, PerformanceBaseline]:
        """Load performance baselines from storage."""
        baselines = {}
        baseline_file = self.baseline_dir / "performance_baselines.json"
        
        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    baseline_data = json.load(f)
                
                for metric_name, data in baseline_data.items():
                    baselines[metric_name] = PerformanceBaseline(
                        metric_name=data["metric_name"],
                        baseline_value=data["baseline_value"],
                        acceptable_variance=data["acceptable_variance"],
                        measurement_unit=data["measurement_unit"],
                        last_updated=datetime.fromisoformat(data["last_updated"]),
                        sample_count=data["sample_count"],
                        confidence_interval=tuple(data["confidence_interval"])
                    )
                
                self.logger.info(f"Loaded {len(baselines)} performance baselines")
                
            except Exception as e:
                self.logger.warning(f"Failed to load baselines: {e}")
        
        return baselines
    
    def _save_baselines(self) -> None:
        """Save performance baselines to storage."""
        baseline_file = self.baseline_dir / "performance_baselines.json"
        
        baseline_data = {}
        for metric_name, baseline in self.baselines.items():
            baseline_dict = asdict(baseline)
            baseline_dict["last_updated"] = baseline.last_updated.isoformat()
            baseline_data[metric_name] = baseline_dict
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2)
        
        self.logger.info(f"Saved {len(self.baselines)} performance baselines")
    
    def update_baselines(self, performance_metrics: Dict[str, Any],
                        test_run_id: str) -> None:
        """
        Update performance baselines with new test results.
        
        Args:
            performance_metrics: Performance metrics from test run
            test_run_id: Identifier for the test run
        """
        self.logger.info(f"Updating baselines with results from {test_run_id}")
        
        # Define metric mappings and units
        metric_mappings = {
            "concurrent_user_avg_response_time": ("response_time", "seconds", 0.15),
            "concurrent_user_p95_response_time": ("p95_response_time", "seconds", 0.20),
            "concurrent_user_throughput": ("throughput", "requests/second", 0.10),
            "concurrent_user_success_rate": ("success_rate", "percentage", 0.02),
            "max_memory_usage": ("memory_usage", "MB", 0.20),
            "avg_cpu_usage": ("cpu_usage", "percentage", 0.15)
        }
        
        for metric_key, (metric_name, unit, variance) in metric_mappings.items():
            if metric_key in performance_metrics:
                current_value = performance_metrics[metric_key]
                
                if metric_name in self.baselines:
                    # Update existing baseline
                    baseline = self.baselines[metric_name]
                    
                    # Use exponential moving average for baseline updates
                    alpha = 0.2  # Smoothing factor
                    new_baseline_value = (alpha * current_value + 
                                        (1 - alpha) * baseline.baseline_value)
                    
                    # Update confidence interval
                    samples = [baseline.baseline_value, current_value]
                    if baseline.sample_count > 1:
                        # Approximate confidence interval update
                        ci_lower = min(baseline.confidence_interval[0], current_value * 0.95)
                        ci_upper = max(baseline.confidence_interval[1], current_value * 1.05)
                    else:
                        ci_lower = current_value * 0.95
                        ci_upper = current_value * 1.05
                    
                    self.baselines[metric_name] = PerformanceBaseline(
                        metric_name=metric_name,
                        baseline_value=new_baseline_value,
                        acceptable_variance=variance,
                        measurement_unit=unit,
                        last_updated=datetime.now(),
                        sample_count=baseline.sample_count + 1,
                        confidence_interval=(ci_lower, ci_upper)
                    )
                    
                else:
                    # Create new baseline
                    self.baselines[metric_name] = PerformanceBaseline(
                        metric_name=metric_name,
                        baseline_value=current_value,
                        acceptable_variance=variance,
                        measurement_unit=unit,
                        last_updated=datetime.now(),
                        sample_count=1,
                        confidence_interval=(current_value * 0.95, current_value * 1.05)
                    )
        
        # Save updated baselines
        self._save_baselines()
    
    def detect_regressions(self, performance_metrics: Dict[str, Any],
                          test_run_id: str) -> PerformanceRegressionReport:
        """
        Detect performance regressions in current test results.
        
        Args:
            performance_metrics: Performance metrics from current test run
            test_run_id: Identifier for the test run
            
        Returns:
            PerformanceRegressionReport with regression analysis
        """
        self.logger.info(f"Detecting regressions for test run {test_run_id}")
        
        regression_results = []
        overall_regression_detected = False
        
        # Analyze each metric
        for metric_name, baseline in self.baselines.items():
            # Map baseline metric names to performance metric keys
            metric_key = self._map_baseline_to_metric_key(metric_name)
            
            if metric_key and metric_key in performance_metrics:
                current_value = performance_metrics[metric_key]
                
                regression_result = self._analyze_metric_regression(
                    metric_name, current_value, baseline
                )
                
                regression_results.append(regression_result)
                
                if regression_result.is_regression:
                    overall_regression_detected = True
                    
                    if regression_result.severity in ["severe", "critical"]:
                        self.logger.warning(
                            f"{regression_result.severity.upper()} regression detected in {metric_name}: "
                            f"{regression_result.variance_percentage:.1f}% change"
                        )
        
        # Analyze performance trends
        performance_trends = self._analyze_performance_trends(performance_metrics)
        
        # Generate recommendations
        recommendations = self._generate_regression_recommendations(
            regression_results, performance_trends
        )
        
        report = PerformanceRegressionReport(
            timestamp=datetime.now(),
            test_run_id=test_run_id,
            overall_regression_detected=overall_regression_detected,
            total_metrics_analyzed=len(regression_results),
            regressions_detected=sum(1 for r in regression_results if r.is_regression),
            regression_results=regression_results,
            performance_trends=performance_trends,
            recommendations=recommendations
        )
        
        # Save regression report
        self._save_regression_report(report)
        
        return report
    
    def _map_baseline_to_metric_key(self, baseline_metric_name: str) -> Optional[str]:
        """Map baseline metric names to performance metric keys."""
        mapping = {
            "response_time": "concurrent_user_avg_response_time",
            "p95_response_time": "concurrent_user_p95_response_time",
            "throughput": "concurrent_user_throughput",
            "success_rate": "concurrent_user_success_rate",
            "memory_usage": "max_memory_usage",
            "cpu_usage": "avg_cpu_usage"
        }
        return mapping.get(baseline_metric_name)
    
    def _analyze_metric_regression(self, metric_name: str, current_value: float,
                                 baseline: PerformanceBaseline) -> RegressionDetectionResult:
        """Analyze a single metric for regression."""
        # Calculate variance percentage
        if baseline.baseline_value != 0:
            variance_percentage = (current_value - baseline.baseline_value) / baseline.baseline_value
        else:
            variance_percentage = 0.0
        
        # Determine if this is a regression based on metric type
        is_regression = False
        severity = "none"
        
        # Get thresholds for this metric type
        metric_type = self._get_metric_type(metric_name)
        thresholds = self.regression_thresholds.get(metric_type, self.regression_thresholds["response_time"])
        
        # Check for regression
        if metric_type in ["throughput", "success_rate"]:
            # For throughput and success rate, negative changes are regressions
            if variance_percentage <= thresholds["critical"]:
                is_regression = True
                severity = "critical"
            elif variance_percentage <= thresholds["severe"]:
                is_regression = True
                severity = "severe"
            elif variance_percentage <= thresholds["moderate"]:
                is_regression = True
                severity = "moderate"
            elif variance_percentage <= thresholds["minor"]:
                is_regression = True
                severity = "minor"
        else:
            # For response time, memory, CPU, positive changes are regressions
            if variance_percentage >= thresholds["critical"]:
                is_regression = True
                severity = "critical"
            elif variance_percentage >= thresholds["severe"]:
                is_regression = True
                severity = "severe"
            elif variance_percentage >= thresholds["moderate"]:
                is_regression = True
                severity = "moderate"
            elif variance_percentage >= thresholds["minor"]:
                is_regression = True
                severity = "minor"
        
        # Calculate confidence level
        confidence_level = self._calculate_confidence_level(
            current_value, baseline, variance_percentage
        )
        
        # Generate recommendation
        recommendation = self._generate_metric_recommendation(
            metric_name, variance_percentage, severity, baseline.measurement_unit
        )
        
        return RegressionDetectionResult(
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline.baseline_value,
            variance_percentage=variance_percentage,
            is_regression=is_regression,
            severity=severity,
            confidence_level=confidence_level,
            recommendation=recommendation
        )
    
    def _get_metric_type(self, metric_name: str) -> str:
        """Determine the type of metric for threshold selection."""
        if "response_time" in metric_name or "latency" in metric_name:
            return "response_time"
        elif "throughput" in metric_name or "requests_per_second" in metric_name:
            return "throughput"
        elif "success_rate" in metric_name or "error_rate" in metric_name:
            return "success_rate"
        elif "memory" in metric_name:
            return "memory_usage"
        elif "cpu" in metric_name:
            return "cpu_usage"
        else:
            return "response_time"  # Default
    
    def _calculate_confidence_level(self, current_value: float,
                                  baseline: PerformanceBaseline,
                                  variance_percentage: float) -> float:
        """Calculate confidence level for regression detection."""
        # Simple confidence calculation based on sample count and variance
        base_confidence = min(0.9, baseline.sample_count / 10.0)  # More samples = higher confidence
        
        # Adjust based on magnitude of change
        magnitude_factor = min(1.0, abs(variance_percentage) * 2)  # Larger changes = higher confidence
        
        confidence = base_confidence * magnitude_factor
        return max(0.1, min(0.95, confidence))
    
    def _generate_metric_recommendation(self, metric_name: str, variance_percentage: float,
                                      severity: str, unit: str) -> str:
        """Generate recommendation for a specific metric regression."""
        if severity == "none":
            return f"{metric_name} is within acceptable performance range"
        
        change_direction = "increased" if variance_percentage > 0 else "decreased"
        change_magnitude = abs(variance_percentage) * 100
        
        recommendations = {
            "minor": f"Minor performance change detected ({change_magnitude:.1f}% {change_direction}). Monitor trend.",
            "moderate": f"Moderate performance regression ({change_magnitude:.1f}% {change_direction}). Investigate potential causes.",
            "severe": f"Severe performance regression ({change_magnitude:.1f}% {change_direction}). Immediate investigation required.",
            "critical": f"Critical performance regression ({change_magnitude:.1f}% {change_direction}). System may be unstable."
        }
        
        base_recommendation = recommendations.get(severity, "Performance change detected")
        
        # Add metric-specific recommendations
        if "response_time" in metric_name and variance_percentage > 0:
            base_recommendation += " Consider optimizing query processing or caching."
        elif "throughput" in metric_name and variance_percentage < 0:
            base_recommendation += " Check for bottlenecks in request processing."
        elif "memory" in metric_name and variance_percentage > 0:
            base_recommendation += " Investigate memory leaks or inefficient memory usage."
        elif "success_rate" in metric_name and variance_percentage < 0:
            base_recommendation += " Check error logs for increased failure rates."
        
        return base_recommendation
    
    def _analyze_performance_trends(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance trends from historical data."""
        trends = {}
        
        # Load historical performance data
        history_file = self.output_dir / "performance_history.json"
        historical_data = []
        
        if history_file.exists():
            try:
                with open(history_file) as f:
                    historical_data = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load performance history: {e}")
        
        # Add current metrics to history
        current_entry = {
            "timestamp": datetime.now().isoformat(),
            "metrics": performance_metrics
        }
        historical_data.append(current_entry)
        
        # Keep only last 30 entries
        historical_data = historical_data[-30:]
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(historical_data, f, indent=2)
        
        # Analyze trends if we have enough data
        if len(historical_data) >= 5:
            for metric_key in performance_metrics.keys():
                values = []
                timestamps = []
                
                for entry in historical_data:
                    if metric_key in entry["metrics"]:
                        values.append(entry["metrics"][metric_key])
                        timestamps.append(datetime.fromisoformat(entry["timestamp"]))
                
                if len(values) >= 5:
                    # Calculate trend
                    trend_slope = self._calculate_trend_slope(values)
                    trend_direction = "improving" if trend_slope < 0 else "degrading" if trend_slope > 0 else "stable"
                    
                    # Calculate volatility
                    volatility = statistics.stdev(values) / statistics.mean(values) if statistics.mean(values) != 0 else 0
                    
                    trends[metric_key] = {
                        "trend_direction": trend_direction,
                        "trend_slope": trend_slope,
                        "volatility": volatility,
                        "recent_average": statistics.mean(values[-5:]),
                        "overall_average": statistics.mean(values)
                    }
        
        return trends
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate trend slope using linear regression."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression
        sum_x = sum(x_values)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_values, values))
        sum_x2 = sum(x * x for x in x_values)
        
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except ZeroDivisionError:
            return 0.0
    
    def _generate_regression_recommendations(self, regression_results: List[RegressionDetectionResult],
                                           performance_trends: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on regression analysis."""
        recommendations = []
        
        # Count regressions by severity
        severity_counts = {"critical": 0, "severe": 0, "moderate": 0, "minor": 0}
        for result in regression_results:
            if result.is_regression:
                severity_counts[result.severity] += 1
        
        # Generate severity-based recommendations
        if severity_counts["critical"] > 0:
            recommendations.append(f"🚨 {severity_counts['critical']} critical performance regressions detected - immediate action required")
            recommendations.append("Consider rolling back recent changes or emergency performance fixes")
        
        if severity_counts["severe"] > 0:
            recommendations.append(f"⚠️ {severity_counts['severe']} severe performance regressions detected")
            recommendations.append("Schedule immediate performance investigation and optimization")
        
        if severity_counts["moderate"] > 0:
            recommendations.append(f"📊 {severity_counts['moderate']} moderate performance regressions detected")
            recommendations.append("Plan performance optimization in next development cycle")
        
        if severity_counts["minor"] > 0:
            recommendations.append(f"📈 {severity_counts['minor']} minor performance changes detected - monitor trends")
        
        # Trend-based recommendations
        degrading_trends = [k for k, v in performance_trends.items() 
                          if v.get("trend_direction") == "degrading"]
        
        if degrading_trends:
            recommendations.append(f"📉 Degrading performance trends detected in: {', '.join(degrading_trends)}")
            recommendations.append("Consider proactive performance optimization to prevent future regressions")
        
        # High volatility recommendations
        volatile_metrics = [k for k, v in performance_trends.items() 
                          if v.get("volatility", 0) > 0.2]
        
        if volatile_metrics:
            recommendations.append(f"🔄 High performance volatility in: {', '.join(volatile_metrics)}")
            recommendations.append("Investigate causes of performance instability")
        
        # General recommendations
        if not any(result.is_regression for result in regression_results):
            recommendations.append("✅ No performance regressions detected - system performance is stable")
        else:
            recommendations.extend([
                "Review detailed regression analysis for specific optimization targets",
                "Update performance baselines after addressing regressions",
                "Consider implementing performance monitoring alerts"
            ])
        
        return recommendations
    
    def _save_regression_report(self, report: PerformanceRegressionReport) -> None:
        """Save regression report to file."""
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"regression_report_{timestamp}.json"
        
        # Convert report to dict for JSON serialization
        report_dict = asdict(report)
        report_dict["timestamp"] = report.timestamp.isoformat()
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"Regression report saved to {report_file}")
    
    def generate_regression_summary(self, report: PerformanceRegressionReport) -> str:
        """Generate human-readable regression summary."""
        lines = [
            "PERFORMANCE REGRESSION ANALYSIS REPORT",
            "=" * 50,
            f"Test Run: {report.test_run_id}",
            f"Analysis Time: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Overall Status: {'🚨 REGRESSIONS DETECTED' if report.overall_regression_detected else '✅ NO REGRESSIONS'}",
            f"Metrics Analyzed: {report.total_metrics_analyzed}",
            f"Regressions Found: {report.regressions_detected}",
            "",
            "REGRESSION DETAILS:",
            "-" * 20
        ]
        
        for result in report.regression_results:
            if result.is_regression:
                severity_icon = {
                    "critical": "🚨",
                    "severe": "⚠️",
                    "moderate": "📊",
                    "minor": "📈"
                }.get(result.severity, "📊")
                
                lines.extend([
                    f"{severity_icon} {result.metric_name.upper()} REGRESSION",
                    f"  Current: {result.current_value:.3f}",
                    f"  Baseline: {result.baseline_value:.3f}",
                    f"  Change: {result.variance_percentage:.1%}",
                    f"  Severity: {result.severity}",
                    f"  Confidence: {result.confidence_level:.1%}",
                    f"  Recommendation: {result.recommendation}",
                    ""
                ])
        
        lines.extend([
            "RECOMMENDATIONS:",
            "-" * 15
        ])
        
        for recommendation in report.recommendations:
            lines.append(f"• {recommendation}")
        
        return "\n".join(lines)


# Convenience function for regression detection
def detect_performance_regressions(performance_metrics: Dict[str, Any],
                                 test_run_id: str,
                                 baseline_dir: Optional[str] = None,
                                 output_dir: Optional[str] = None,
                                 update_baselines: bool = False) -> PerformanceRegressionReport:
    """
    Detect performance regressions in test results.
    
    Args:
        performance_metrics: Performance metrics from test run
        test_run_id: Identifier for the test run
        baseline_dir: Directory containing baselines
        output_dir: Directory for outputs
        update_baselines: Whether to update baselines with current results
        
    Returns:
        PerformanceRegressionReport with analysis
    """
    detector = PerformanceRegressionDetector(baseline_dir=baseline_dir, output_dir=output_dir)
    
    # Detect regressions
    report = detector.detect_regressions(performance_metrics, test_run_id)
    
    # Update baselines if requested and no critical regressions
    if update_baselines:
        critical_regressions = [r for r in report.regression_results 
                              if r.is_regression and r.severity == "critical"]
        if not critical_regressions:
            detector.update_baselines(performance_metrics, test_run_id)
        else:
            detector.logger.warning("Skipping baseline update due to critical regressions")
    
    return report