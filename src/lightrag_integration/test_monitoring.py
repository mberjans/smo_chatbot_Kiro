"""
Tests for Monitoring and Alerting System

This module tests the comprehensive monitoring, metrics collection,
performance tracking, and alerting capabilities.
"""

import asyncio
import pytest
import time
import tempfile
import json
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from .monitoring import (
    MetricsCollector,
    AlertManager,
    PerformanceMonitor,
    AlertSeverity,
    MetricType,
    MetricValue,
    Alert,
    PerformanceMetrics,
    SystemMetrics,
    timer,
    monitor_performance,
    log_alert_handler,
    file_alert_handler
)


class TestMetricsCollector:
    """Test cases for the MetricsCollector class."""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create a MetricsCollector instance for testing."""
        return MetricsCollector(max_metrics_history=100)
    
    def test_metrics_collector_initialization(self, metrics_collector):
        """Test MetricsCollector initialization."""
        assert metrics_collector.max_metrics_history == 100
        assert len(metrics_collector.metrics_history) == 0
        assert len(metrics_collector.current_metrics) == 0
        assert len(metrics_collector.performance_metrics) == 0
    
    def test_increment_counter(self, metrics_collector):
        """Test counter increment functionality."""
        # First increment
        metrics_collector.increment_counter("test_counter", 5)
        
        assert "test_counter" in metrics_collector.current_metrics
        assert metrics_collector.current_metrics["test_counter"].value == 5
        assert metrics_collector.current_metrics["test_counter"].metric_type == MetricType.COUNTER
        
        # Second increment
        metrics_collector.increment_counter("test_counter", 3)
        assert metrics_collector.current_metrics["test_counter"].value == 8
        
        # Check history
        assert len(metrics_collector.metrics_history) == 2
    
    def test_set_gauge(self, metrics_collector):
        """Test gauge setting functionality."""
        metrics_collector.set_gauge("test_gauge", 42.5)
        
        assert "test_gauge" in metrics_collector.current_metrics
        assert metrics_collector.current_metrics["test_gauge"].value == 42.5
        assert metrics_collector.current_metrics["test_gauge"].metric_type == MetricType.GAUGE
        
        # Update gauge value
        metrics_collector.set_gauge("test_gauge", 100.0)
        assert metrics_collector.current_metrics["test_gauge"].value == 100.0
        
        # Check history
        assert len(metrics_collector.metrics_history) == 2
    
    def test_record_histogram(self, metrics_collector):
        """Test histogram recording functionality."""
        values = [1.0, 2.5, 3.2, 1.8, 4.1]
        
        for value in values:
            metrics_collector.record_histogram("test_histogram", value)
        
        # Histogram values are not stored in current_metrics
        assert "test_histogram" not in metrics_collector.current_metrics
        
        # But they are in history
        assert len(metrics_collector.metrics_history) == 5
        
        # All should be histogram type
        histogram_metrics = [m for m in metrics_collector.metrics_history if m.metric_type == MetricType.HISTOGRAM]
        assert len(histogram_metrics) == 5
    
    def test_record_timing(self, metrics_collector):
        """Test timing recording functionality."""
        # Record some timings
        timings = [0.1, 0.2, 0.15, 0.3, 0.05]
        
        for i, timing in enumerate(timings):
            success = i < 4  # Last one is a failure
            metrics_collector.record_timing("test_operation", timing, success)
        
        # Check performance metrics
        assert "test_operation" in metrics_collector.performance_metrics
        perf_metrics = metrics_collector.performance_metrics["test_operation"]
        
        assert perf_metrics.total_calls == 5
        assert perf_metrics.successful_calls == 4
        assert perf_metrics.failed_calls == 1
        assert perf_metrics.min_duration == 0.05
        assert perf_metrics.max_duration == 0.3
        assert abs(perf_metrics.avg_duration - 0.16) < 0.01  # Average of timings
        
        # Check that timing metrics were also recorded
        timing_metrics = [m for m in metrics_collector.metrics_history if m.metric_type == MetricType.TIMER]
        assert len(timing_metrics) == 5
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.Process')
    def test_collect_system_metrics(self, mock_process, mock_disk, mock_memory, mock_cpu, metrics_collector):
        """Test system metrics collection."""
        # Mock system information
        mock_cpu.return_value = 45.2
        
        mock_memory_info = Mock()
        mock_memory_info.percent = 67.8
        mock_memory_info.used = 8 * 1024 * 1024 * 1024  # 8GB in bytes
        mock_memory.return_value = mock_memory_info
        
        mock_disk_info = Mock()
        mock_disk_info.used = 100 * 1024 * 1024 * 1024  # 100GB
        mock_disk_info.total = 500 * 1024 * 1024 * 1024  # 500GB
        mock_disk_info.free = 400 * 1024 * 1024 * 1024   # 400GB
        mock_disk.return_value = mock_disk_info
        
        mock_process_info = Mock()
        mock_process_info.num_threads.return_value = 12
        mock_process_info.open_files.return_value = [Mock()] * 25  # 25 open files
        mock_process_info.net_connections.return_value = [Mock()] * 8  # 8 connections
        mock_process_info.connections.return_value = [Mock()] * 8  # 8 connections (fallback)
        mock_process.return_value = mock_process_info
        
        # Collect metrics
        system_metrics = metrics_collector.collect_system_metrics()
        
        assert system_metrics.cpu_usage_percent == 45.2
        assert system_metrics.memory_usage_percent == 67.8
        assert abs(system_metrics.memory_usage_mb - 8192) < 1  # 8GB in MB
        assert abs(system_metrics.disk_usage_percent - 20.0) < 1  # 100/500 * 100
        assert abs(system_metrics.disk_free_gb - 400) < 1  # 400GB
        assert system_metrics.active_threads == 12
        assert system_metrics.open_files == 25
        assert system_metrics.network_connections == 8
        
        # Check that system metrics were stored
        assert len(metrics_collector.system_metrics_history) == 1
    
    def test_get_metrics_history_filtering(self, metrics_collector):
        """Test metrics history filtering functionality."""
        # Add various metrics
        metrics_collector.increment_counter("counter1", 1)
        metrics_collector.set_gauge("gauge1", 10.0)
        metrics_collector.increment_counter("counter2", 2)
        metrics_collector.record_histogram("histogram1", 5.0)
        
        # Get all history
        all_history = metrics_collector.get_metrics_history()
        assert len(all_history) == 4
        
        # Filter by metric name
        counter1_history = metrics_collector.get_metrics_history(metric_name="counter1")
        assert len(counter1_history) == 1
        assert counter1_history[0].name == "counter1"
        
        # Filter by time (should get all since they're recent)
        recent_history = metrics_collector.get_metrics_history(since=datetime.now() - timedelta(minutes=1))
        assert len(recent_history) == 4
        
        # Filter by time (should get none since they're too recent)
        old_history = metrics_collector.get_metrics_history(since=datetime.now() + timedelta(minutes=1))
        assert len(old_history) == 0
    
    def test_reset_metrics(self, metrics_collector):
        """Test metrics reset functionality."""
        # Add some metrics
        metrics_collector.increment_counter("test_counter", 5)
        metrics_collector.set_gauge("test_gauge", 10.0)
        metrics_collector.record_timing("test_operation", 0.1, True)
        
        # Verify metrics exist
        assert len(metrics_collector.current_metrics) > 0
        assert len(metrics_collector.performance_metrics) > 0
        assert len(metrics_collector.metrics_history) > 0
        
        # Reset metrics
        metrics_collector.reset_metrics()
        
        # Verify metrics are cleared
        assert len(metrics_collector.current_metrics) == 0
        assert len(metrics_collector.performance_metrics) == 0
        assert len(metrics_collector.metrics_history) == 0


class TestAlertManager:
    """Test cases for the AlertManager class."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create an AlertManager instance for testing."""
        config = {
            'thresholds': {
                'cpu_usage': {
                    'high': {'value': 80.0, 'operator': 'gt', 'severity': 'warning'},
                    'critical': {'value': 95.0, 'operator': 'gt', 'severity': 'critical'}
                },
                'test_operation_avg_response_time': {
                    'slow': {'value': 1.0, 'operator': 'gt', 'severity': 'warning'}
                },
                'system_memory_usage': {
                    'high': {'value': 85.0, 'operator': 'gt', 'severity': 'error'}
                }
            }
        }
        return AlertManager(config)
    
    def test_alert_manager_initialization(self, alert_manager):
        """Test AlertManager initialization."""
        assert len(alert_manager.active_alerts) == 0
        assert len(alert_manager.alert_history) == 0
        assert len(alert_manager.notification_handlers) == 0
        assert 'thresholds' in alert_manager.config
    
    def test_create_alert(self, alert_manager):
        """Test alert creation functionality."""
        alert = alert_manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            component="test_component",
            metric_name="test_metric",
            metric_value=75.0,
            threshold=70.0
        )
        
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test alert"
        assert alert.component == "test_component"
        assert alert.metric_name == "test_metric"
        assert alert.metric_value == 75.0
        assert alert.threshold == 70.0
        assert not alert.resolved
        
        # Check that alert is stored
        assert alert.alert_id in alert_manager.active_alerts
        assert len(alert_manager.alert_history) == 1
    
    def test_resolve_alert(self, alert_manager):
        """Test alert resolution functionality."""
        # Create an alert
        alert = alert_manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            component="test_component"
        )
        
        alert_id = alert.alert_id
        
        # Verify alert is active
        assert alert_id in alert_manager.active_alerts
        assert not alert.resolved
        
        # Resolve the alert
        success = alert_manager.resolve_alert(alert_id, "Issue resolved")
        
        assert success
        assert alert_id not in alert_manager.active_alerts
        assert alert.resolved
        assert alert.resolution_timestamp is not None
        assert alert.metadata['resolution_message'] == "Issue resolved"
    
    def test_check_metric_thresholds(self, alert_manager):
        """Test metric threshold checking."""
        # Create a metric that violates threshold
        metric = MetricValue(
            name="cpu_usage",
            value=85.0,  # Above warning threshold of 80
            metric_type=MetricType.GAUGE
        )
        
        alerts = alert_manager.check_metric_thresholds(metric)
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.severity == AlertSeverity.WARNING
        assert alert.metric_name == "cpu_usage"
        assert alert.metric_value == 85.0
        assert alert.threshold == 80.0
        
        # Create a metric that violates critical threshold
        critical_metric = MetricValue(
            name="cpu_usage",
            value=97.0,  # Above critical threshold of 95
            metric_type=MetricType.GAUGE
        )
        
        critical_alerts = alert_manager.check_metric_thresholds(critical_metric)
        
        assert len(critical_alerts) == 2  # Both warning and critical thresholds
        severities = [a.severity for a in critical_alerts]
        assert AlertSeverity.WARNING in severities
        assert AlertSeverity.CRITICAL in severities
    
    def test_check_performance_thresholds(self, alert_manager):
        """Test performance threshold checking."""
        # Create performance metrics that violate threshold
        perf_metrics = PerformanceMetrics(
            operation_name="test_operation",
            total_calls=100,
            successful_calls=95,
            failed_calls=5,
            avg_duration=1.5  # Above threshold of 1.0
        )
        
        alerts = alert_manager.check_performance_thresholds(perf_metrics)
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.severity == AlertSeverity.WARNING
        assert "test_operation" in alert.title
        assert alert.metric_value == 1.5
        assert alert.threshold == 1.0
    
    def test_check_system_thresholds(self, alert_manager):
        """Test system threshold checking."""
        # Create system metrics that violate threshold
        system_metrics = SystemMetrics(
            cpu_usage_percent=75.0,  # Below threshold
            memory_usage_percent=90.0,  # Above threshold of 85
            disk_usage_percent=50.0,
            active_threads=10,
            open_files=20
        )
        
        alerts = alert_manager.check_system_thresholds(system_metrics)
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.severity == AlertSeverity.ERROR
        assert "memory_usage" in alert.title
        assert alert.metric_value == 90.0
        assert alert.threshold == 85.0
    
    def test_notification_handlers(self, alert_manager):
        """Test notification handler functionality."""
        # Add a mock notification handler
        mock_handler = Mock()
        mock_handler.__name__ = "mock_handler"  # Add __name__ attribute
        alert_manager.add_notification_handler(mock_handler)
        
        # Create an alert
        alert = alert_manager.create_alert(
            severity=AlertSeverity.ERROR,
            title="Test Alert",
            message="Test message",
            component="test"
        )
        
        # Verify handler was called
        mock_handler.assert_called_once_with(alert)
    
    def test_get_alert_statistics(self, alert_manager):
        """Test alert statistics generation."""
        # Create various alerts
        alert1 = alert_manager.create_alert(AlertSeverity.WARNING, "Warning 1", "Message", "comp1")
        alert2 = alert_manager.create_alert(AlertSeverity.ERROR, "Error 1", "Message", "comp1")
        alert3 = alert_manager.create_alert(AlertSeverity.CRITICAL, "Critical 1", "Message", "comp2")
        
        stats = alert_manager.get_alert_statistics()
        
        # Check the actual number of active alerts
        active_count = len(alert_manager.get_active_alerts())
        assert stats['active_alerts'] == active_count
        assert stats['total_alerts'] == 3
        assert stats['critical_alerts'] == 1
        assert stats['error_alerts'] == 1
        assert 'comp1' in stats['alerts_by_component']
        assert 'comp2' in stats['alerts_by_component']
        assert stats['alerts_by_component']['comp1'] == 2
        assert stats['alerts_by_component']['comp2'] == 1


class TestPerformanceMonitor:
    """Test cases for the PerformanceMonitor class."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a PerformanceMonitor instance for testing."""
        config = {
            'monitoring_interval': 0.1,  # Short interval for testing
            'alerting': {
                'thresholds': {
                    'test_operation_avg_response_time': {
                        'slow': {'value': 0.5, 'operator': 'gt', 'severity': 'warning'}
                    }
                }
            }
        }
        return PerformanceMonitor(config)
    
    def test_performance_monitor_initialization(self, performance_monitor):
        """Test PerformanceMonitor initialization."""
        assert performance_monitor.metrics_collector is not None
        assert performance_monitor.alert_manager is not None
        assert not performance_monitor.monitoring_active
        assert performance_monitor.monitoring_interval == 0.1
    
    def test_timer_functionality(self, performance_monitor):
        """Test timer start/end functionality."""
        # Start timer
        timer_id = performance_monitor.start_timer("test_operation")
        assert timer_id in performance_monitor.operation_timers
        
        # Wait a bit
        time.sleep(0.01)
        
        # End timer
        duration = performance_monitor.end_timer(timer_id, "test_operation", True)
        
        assert duration > 0
        assert timer_id not in performance_monitor.operation_timers
        
        # Check that metrics were recorded
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "test_operation" in perf_metrics
        assert perf_metrics["test_operation"].total_calls == 1
        assert perf_metrics["test_operation"].successful_calls == 1
    
    def test_record_operation(self, performance_monitor):
        """Test operation recording functionality."""
        # Record successful operation
        performance_monitor.record_operation("test_op", 0.1, True)
        
        # Record failed operation
        performance_monitor.record_operation("test_op", 0.2, False)
        
        # Check metrics
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "test_op" in perf_metrics
        
        pm = perf_metrics["test_op"]
        assert pm.total_calls == 2
        assert pm.successful_calls == 1
        assert pm.failed_calls == 1
        
        # Check counters
        current_metrics = performance_monitor.metrics_collector.get_current_metrics()
        assert current_metrics["test_op_total"].value == 2
        assert current_metrics["test_op_success"].value == 1
        assert current_metrics["test_op_error"].value == 1
    
    def test_metric_operations(self, performance_monitor):
        """Test various metric operations."""
        # Set gauge
        performance_monitor.set_metric("test_gauge", 42.0, {"tag": "value"})
        
        # Increment counter
        performance_monitor.increment_counter("test_counter", 5, {"env": "test"})
        
        # Record histogram
        performance_monitor.record_histogram("test_histogram", 1.5, {"bucket": "small"})
        
        # Verify metrics were recorded
        current_metrics = performance_monitor.metrics_collector.get_current_metrics()
        assert current_metrics["test_gauge"].value == 42.0
        assert current_metrics["test_counter"].value == 5
        
        history = performance_monitor.metrics_collector.get_metrics_history()
        histogram_metrics = [m for m in history if m.name == "test_histogram"]
        assert len(histogram_metrics) == 1
        assert histogram_metrics[0].value == 1.5
    
    def test_get_performance_summary(self, performance_monitor):
        """Test performance summary generation."""
        # Record some operations
        performance_monitor.record_operation("op1", 0.1, True)
        performance_monitor.record_operation("op1", 0.2, False)
        performance_monitor.record_operation("op2", 0.05, True)
        
        summary = performance_monitor.get_performance_summary()
        
        assert 'timestamp' in summary
        assert 'system_metrics' in summary
        assert 'performance_summary' in summary
        assert 'alert_summary' in summary
        
        perf_summary = summary['performance_summary']
        assert perf_summary['total_operations'] == 3
        assert perf_summary['total_errors'] == 1
        assert abs(perf_summary['error_rate_percent'] - 33.33) < 1
        assert perf_summary['operations_tracked'] == 2
    
    def test_get_operation_metrics(self, performance_monitor):
        """Test getting metrics for specific operation."""
        # Record operations
        performance_monitor.record_operation("specific_op", 0.1, True)
        performance_monitor.record_operation("specific_op", 0.3, True)
        performance_monitor.record_operation("specific_op", 0.2, False)
        
        metrics = performance_monitor.get_operation_metrics("specific_op")
        
        assert metrics is not None
        assert metrics['operation_name'] == "specific_op"
        assert metrics['total_calls'] == 3
        assert metrics['successful_calls'] == 2
        assert metrics['failed_calls'] == 1
        assert abs(metrics['success_rate_percent'] - 66.67) < 1
        assert abs(metrics['error_rate_percent'] - 33.33) < 1
        assert abs(metrics['avg_duration_ms'] - 200.0) < 0.01  # (0.1 + 0.3 + 0.2) / 3 * 1000
        
        # Test non-existent operation
        assert performance_monitor.get_operation_metrics("nonexistent") is None
    
    def test_export_metrics(self, performance_monitor):
        """Test metrics export functionality."""
        # Record some data
        performance_monitor.record_operation("test_op", 0.1, True)
        performance_monitor.set_metric("test_gauge", 50.0)
        
        # Export as JSON
        json_export = performance_monitor.export_metrics("json")
        
        # Verify it's valid JSON
        data = json.loads(json_export)
        assert 'timestamp' in data
        assert 'system_metrics' in data
        assert 'performance_summary' in data
        
        # Test unsupported format
        with pytest.raises(ValueError):
            performance_monitor.export_metrics("xml")
    
    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, performance_monitor):
        """Test monitoring start/stop lifecycle."""
        # Initially not monitoring
        assert not performance_monitor.monitoring_active
        
        # Start monitoring
        performance_monitor.start_monitoring()
        assert performance_monitor.monitoring_active
        assert performance_monitor.monitoring_task is not None
        
        # Wait a bit for monitoring to run
        await asyncio.sleep(0.2)
        
        # Stop monitoring
        await performance_monitor.stop_monitoring()
        assert not performance_monitor.monitoring_active


class TestTimerContextManager:
    """Test cases for the timer context manager."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a PerformanceMonitor instance for testing."""
        return PerformanceMonitor()
    
    def test_timer_context_manager_success(self, performance_monitor):
        """Test timer context manager with successful operation."""
        with timer(performance_monitor, "test_operation") as t:
            time.sleep(0.01)  # Simulate work
            assert t.success  # Should be True by default
        
        # Check that timing was recorded
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "test_operation" in perf_metrics
        assert perf_metrics["test_operation"].total_calls == 1
        assert perf_metrics["test_operation"].successful_calls == 1
        assert perf_metrics["test_operation"].failed_calls == 0
    
    def test_timer_context_manager_exception(self, performance_monitor):
        """Test timer context manager with exception."""
        with pytest.raises(ValueError):
            with timer(performance_monitor, "test_operation") as t:
                time.sleep(0.01)
                raise ValueError("Test exception")
        
        # Check that timing was recorded as failure
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "test_operation" in perf_metrics
        assert perf_metrics["test_operation"].total_calls == 1
        assert perf_metrics["test_operation"].successful_calls == 0
        assert perf_metrics["test_operation"].failed_calls == 1


class TestMonitorPerformanceDecorator:
    """Test cases for the monitor_performance decorator."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a PerformanceMonitor instance for testing."""
        return PerformanceMonitor()
    
    def test_sync_function_decorator(self, performance_monitor):
        """Test decorator with synchronous function."""
        @monitor_performance(performance_monitor, "sync_test")
        def test_function(x, y):
            time.sleep(0.01)
            return x + y
        
        result = test_function(2, 3)
        assert result == 5
        
        # Check that timing was recorded
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "sync_test" in perf_metrics
        assert perf_metrics["sync_test"].total_calls == 1
        assert perf_metrics["sync_test"].successful_calls == 1
    
    @pytest.mark.asyncio
    async def test_async_function_decorator(self, performance_monitor):
        """Test decorator with asynchronous function."""
        @monitor_performance(performance_monitor, "async_test")
        async def test_async_function(x, y):
            await asyncio.sleep(0.01)
            return x * y
        
        result = await test_async_function(3, 4)
        assert result == 12
        
        # Check that timing was recorded
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "async_test" in perf_metrics
        assert perf_metrics["async_test"].total_calls == 1
        assert perf_metrics["async_test"].successful_calls == 1
    
    def test_decorator_with_exception(self, performance_monitor):
        """Test decorator behavior with exceptions."""
        @monitor_performance(performance_monitor, "exception_test")
        def failing_function():
            time.sleep(0.01)
            raise RuntimeError("Test error")
        
        with pytest.raises(RuntimeError):
            failing_function()
        
        # Check that timing was recorded as failure
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "exception_test" in perf_metrics
        assert perf_metrics["exception_test"].total_calls == 1
        assert perf_metrics["exception_test"].successful_calls == 0
        assert perf_metrics["exception_test"].failed_calls == 1
    
    def test_decorator_auto_naming(self, performance_monitor):
        """Test decorator with automatic operation naming."""
        @monitor_performance(performance_monitor)  # No operation name provided
        def auto_named_function():
            time.sleep(0.01)
            return "success"
        
        result = auto_named_function()
        assert result == "success"
        
        # Check that timing was recorded with auto-generated name
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        
        # Should have an entry with the module and function name
        operation_names = list(perf_metrics.keys())
        assert len(operation_names) == 1
        assert "auto_named_function" in operation_names[0]


class TestNotificationHandlers:
    """Test cases for notification handlers."""
    
    def test_log_alert_handler(self, caplog):
        """Test the log alert handler."""
        alert = Alert(
            alert_id="test_alert",
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test alert",
            component="test_component"
        )
        
        with caplog.at_level(logging.WARNING):
            log_alert_handler(alert)
        
        assert "ALERT [WARNING] Test Alert: This is a test alert" in caplog.text
    
    def test_file_alert_handler(self):
        """Test the file alert handler."""
        alert = Alert(
            alert_id="test_alert",
            severity=AlertSeverity.ERROR,
            title="Test Alert",
            message="This is a test alert",
            component="test_component",
            metric_name="test_metric",
            metric_value=75.0,
            threshold=70.0
        )
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
            alert_file = f.name
        
        try:
            file_alert_handler(alert, alert_file)
            
            # Read the alert from file
            with open(alert_file, 'r') as f:
                alert_data = json.loads(f.read().strip())
            
            assert alert_data['alert_id'] == "test_alert"
            assert alert_data['severity'] == "error"
            assert alert_data['title'] == "Test Alert"
            assert alert_data['message'] == "This is a test alert"
            assert alert_data['component'] == "test_component"
            assert alert_data['metric_name'] == "test_metric"
            assert alert_data['metric_value'] == 75.0
            assert alert_data['threshold'] == 70.0
            
        finally:
            import os
            os.unlink(alert_file)


if __name__ == "__main__":
    pytest.main([__file__])


class TestEnhancedPerformanceMonitor:
    """Test cases for the EnhancedPerformanceMonitor class."""
    
    @pytest.fixture
    def enhanced_monitor(self):
        """Create an EnhancedPerformanceMonitor instance for testing."""
        from .monitoring import EnhancedPerformanceMonitor
        
        config = {
            'monitoring_interval': 0.1,
            'health_check_interval': 0.2,
            'error_rate_threshold': 10.0,
            'response_time_threshold': 1.0,
            'alerting': {
                'thresholds': {
                    'test_operation_avg_response_time': {
                        'slow': {'value': 0.5, 'operator': 'gt', 'severity': 'warning'}
                    }
                }
            }
        }
        return EnhancedPerformanceMonitor(config)
    
    def test_enhanced_monitor_initialization(self, enhanced_monitor):
        """Test EnhancedPerformanceMonitor initialization."""
        assert enhanced_monitor._error_handler is None
        assert enhanced_monitor._health_check_interval == 0.2
        assert enhanced_monitor._error_rate_threshold == 10.0
        assert enhanced_monitor._response_time_threshold == 1.0
    
    def test_get_operation_metrics(self, enhanced_monitor):
        """Test getting metrics for specific operation."""
        # Record operations
        enhanced_monitor.record_operation("specific_op", 0.1, True)
        enhanced_monitor.record_operation("specific_op", 0.3, True)
        enhanced_monitor.record_operation("specific_op", 0.2, False)
        
        metrics = enhanced_monitor.get_operation_metrics("specific_op")
        
        assert metrics is not None
        assert metrics['operation_name'] == "specific_op"
        assert metrics['total_calls'] == 3
        assert metrics['successful_calls'] == 2
        assert metrics['failed_calls'] == 1
        assert abs(metrics['success_rate_percent'] - 66.67) < 1
        assert abs(metrics['error_rate_percent'] - 33.33) < 1
        assert abs(metrics['avg_duration_ms'] - 200.0) < 0.01  # (0.1 + 0.3 + 0.2) / 3 * 1000
        
        # Test non-existent operation
        assert enhanced_monitor.get_operation_metrics("nonexistent") is None
    
    def test_export_metrics(self, enhanced_monitor):
        """Test metrics export functionality."""
        # Record some data
        enhanced_monitor.record_operation("test_op", 0.1, True)
        enhanced_monitor.set_metric("test_gauge", 50.0)
        
        # Export as JSON
        json_export = enhanced_monitor.export_metrics("json")
        
        # Verify it's valid JSON
        data = json.loads(json_export)
        assert 'timestamp' in data
        assert 'system_metrics' in data
        assert 'performance_summary' in data
        
        # Test unsupported format
        with pytest.raises(ValueError):
            enhanced_monitor.export_metrics("xml")
    
    def test_get_health_status(self, enhanced_monitor):
        """Test health status reporting."""
        # Create some alerts to test health status
        enhanced_monitor.alert_manager.create_alert(
            severity=AlertSeverity.WARNING,
            title="Test Warning",
            message="Test warning message",
            component="test"
        )
        
        health_status = enhanced_monitor.get_health_status()
        
        assert health_status['status'] == 'warning'  # Should be warning due to active alert
        assert 'timestamp' in health_status
        assert 'system_metrics' in health_status
        assert 'alerts' in health_status
        assert health_status['alerts']['total'] == 1
        assert health_status['alerts']['warning'] == 1
        assert 'monitoring_active' in health_status
    
    def test_check_operation_health(self, enhanced_monitor):
        """Test operation health checking."""
        # Test unknown operation
        health = enhanced_monitor.check_operation_health("unknown_op")
        assert health['status'] == 'unknown'
        assert 'No metrics available' in health['message']
        
        # Record operations with high error rate
        for i in range(10):
            success = i < 5  # 50% error rate
            enhanced_monitor.record_operation("error_prone_op", 0.1, success)
        
        health = enhanced_monitor.check_operation_health("error_prone_op")
        assert health['status'] == 'degraded'
        assert health['error_rate_percent'] == 50.0
        assert 'High error rate' in health['message']
        
        # Record operations with slow response time
        for i in range(5):
            enhanced_monitor.record_operation("slow_op", 2.0, True)  # 2 seconds
        
        health = enhanced_monitor.check_operation_health("slow_op")
        assert health['status'] == 'degraded'
        assert health['avg_duration_ms'] == 2000.0
        assert 'Slow response time' in health['message']
    
    def test_get_system_diagnostics(self, enhanced_monitor):
        """Test system diagnostics generation."""
        # Record some operations
        enhanced_monitor.record_operation("fast_op", 0.1, True)
        enhanced_monitor.record_operation("slow_op", 2.5, True)  # Slow operation
        enhanced_monitor.record_operation("error_op", 0.1, False)  # Error-prone
        enhanced_monitor.record_operation("error_op", 0.1, False)
        enhanced_monitor.record_operation("error_op", 0.1, True)  # 67% error rate
        
        diagnostics = enhanced_monitor.get_system_diagnostics()
        
        assert 'timestamp' in diagnostics
        assert 'system_health' in diagnostics
        assert 'performance_analysis' in diagnostics
        assert 'alerts' in diagnostics
        assert 'error_handler_integration' in diagnostics
        
        # Check performance analysis
        perf_analysis = diagnostics['performance_analysis']
        assert perf_analysis['total_operations'] == 3
        
        # Should detect slow operation
        slow_ops = perf_analysis['slow_operations']
        assert len(slow_ops) > 0
        assert any(op['operation'] == 'slow_op' for op in slow_ops)
        
        # Should detect error-prone operation
        error_ops = perf_analysis['error_prone_operations']
        assert len(error_ops) > 0
        assert any(op['operation'] == 'error_op' for op in error_ops)
    
    def test_integrate_with_error_handler(self, enhanced_monitor):
        """Test integration with error handler."""
        # Mock error handler
        mock_error_handler = Mock()
        mock_error_handler.get_error_statistics.return_value = {
            'total_errors': 25,
            'recent_errors_1h': 5,
            'recent_errors_24h': 15,
            'errors_by_category': {'pdf_processing': 10, 'query_processing': 15},
            'errors_by_severity': {'medium': 20, 'high': 5},
            'circuit_breakers_open': 1
        }
        
        # Integrate with error handler
        enhanced_monitor.integrate_with_error_handler(mock_error_handler)
        
        assert enhanced_monitor._error_handler is not None
        assert hasattr(enhanced_monitor, '_error_metrics_tracker')
        
        # Test error metrics tracking
        enhanced_monitor._error_metrics_tracker()
        
        # Check that error metrics were recorded
        current_metrics = enhanced_monitor.metrics_collector.get_current_metrics()
        assert current_metrics['errors_total'].value == 25
        assert current_metrics['errors_recent_1h'].value == 5
        assert current_metrics['errors_by_category_pdf_processing'].value == 10
        
        # Check that alerts were created for circuit breakers
        active_alerts = enhanced_monitor.alert_manager.get_active_alerts()
        circuit_breaker_alerts = [a for a in active_alerts if 'Circuit Breakers' in a.title]
        assert len(circuit_breaker_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_run_health_check(self, enhanced_monitor):
        """Test comprehensive health check."""
        # Record some operations
        enhanced_monitor.record_operation("healthy_op", 0.1, True)
        enhanced_monitor.record_operation("unhealthy_op", 2.0, False)  # Slow and failed
        
        health_report = await enhanced_monitor.run_health_check()
        
        assert 'overall_status' in health_report
        assert 'timestamp' in health_report
        assert 'system_diagnostics' in health_report
        assert 'operation_health' in health_report
        assert 'recommendations' in health_report
        
        # Check operation health
        op_health = health_report['operation_health']
        assert 'healthy_op' in op_health
        assert 'unhealthy_op' in op_health
        
        # Check recommendations
        recommendations = health_report['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_generate_health_recommendations(self, enhanced_monitor):
        """Test health recommendation generation."""
        # Create mock diagnostics with various issues
        diagnostics = {
            'system_health': {
                'cpu_usage_percent': 85.0,  # High CPU
                'memory_usage_percent': 90.0,  # High memory
                'disk_usage_percent': 95.0,  # High disk
                'disk_free_gb': 0.5  # Low disk space
            },
            'performance_analysis': {
                'slow_operations': [{'operation': 'slow_op1'}, {'operation': 'slow_op2'}],
                'error_prone_operations': [{'operation': 'error_op1'}]
            },
            'alerts': {
                'critical_count': 1,
                'error_count': 2
            },
            'error_handler_integration': {
                'integrated': False
            }
        }
        
        operation_health = {
            'degraded_op': {'status': 'degraded'}
        }
        
        recommendations = enhanced_monitor._generate_health_recommendations(diagnostics, operation_health)
        
        assert len(recommendations) > 0
        
        # Check for specific recommendations
        rec_text = ' '.join(recommendations)
        assert 'CPU' in rec_text or 'cpu' in rec_text
        assert 'memory' in rec_text
        assert 'disk' in rec_text
        assert 'slow_op1' in rec_text or 'slow_op2' in rec_text
        assert 'error_op1' in rec_text
        assert 'critical' in rec_text
        assert 'error handler' in rec_text


class TestTimerContextManager:
    """Test cases for the timer context manager."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a PerformanceMonitor instance for testing."""
        return PerformanceMonitor()
    
    def test_timer_context_manager_success(self, performance_monitor):
        """Test timer context manager with successful operation."""
        from .monitoring import timer
        
        with timer(performance_monitor, "test_operation") as t:
            time.sleep(0.01)  # Simulate work
            assert t.success  # Should be True by default
        
        # Check that timing was recorded
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "test_operation" in perf_metrics
        assert perf_metrics["test_operation"].total_calls == 1
        assert perf_metrics["test_operation"].successful_calls == 1
        assert perf_metrics["test_operation"].failed_calls == 0
    
    def test_timer_context_manager_exception(self, performance_monitor):
        """Test timer context manager with exception."""
        from .monitoring import timer
        
        with pytest.raises(ValueError):
            with timer(performance_monitor, "test_operation") as t:
                time.sleep(0.01)
                raise ValueError("Test exception")
        
        # Check that timing was recorded as failure
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "test_operation" in perf_metrics
        assert perf_metrics["test_operation"].total_calls == 1
        assert perf_metrics["test_operation"].successful_calls == 0
        assert perf_metrics["test_operation"].failed_calls == 1


class TestMonitorPerformanceDecorator:
    """Test cases for the monitor_performance decorator."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create a PerformanceMonitor instance for testing."""
        return PerformanceMonitor()
    
    def test_sync_function_decorator(self, performance_monitor):
        """Test decorator with synchronous function."""
        from .monitoring import monitor_performance
        
        @monitor_performance(performance_monitor, "sync_test")
        def test_function(x, y):
            time.sleep(0.01)
            return x + y
        
        result = test_function(2, 3)
        assert result == 5
        
        # Check that timing was recorded
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "sync_test" in perf_metrics
        assert perf_metrics["sync_test"].total_calls == 1
        assert perf_metrics["sync_test"].successful_calls == 1
    
    @pytest.mark.asyncio
    async def test_async_function_decorator(self, performance_monitor):
        """Test decorator with asynchronous function."""
        from .monitoring import monitor_performance
        
        @monitor_performance(performance_monitor, "async_test")
        async def test_async_function(x, y):
            await asyncio.sleep(0.01)
            return x * y
        
        result = await test_async_function(3, 4)
        assert result == 12
        
        # Check that timing was recorded
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "async_test" in perf_metrics
        assert perf_metrics["async_test"].total_calls == 1
        assert perf_metrics["async_test"].successful_calls == 1
    
    def test_decorator_with_exception(self, performance_monitor):
        """Test decorator behavior with exceptions."""
        from .monitoring import monitor_performance
        
        @monitor_performance(performance_monitor, "exception_test")
        def failing_function():
            time.sleep(0.01)
            raise RuntimeError("Test error")
        
        with pytest.raises(RuntimeError):
            failing_function()
        
        # Check that timing was recorded as failure
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        assert "exception_test" in perf_metrics
        assert perf_metrics["exception_test"].total_calls == 1
        assert perf_metrics["exception_test"].successful_calls == 0
        assert perf_metrics["exception_test"].failed_calls == 1
    
    def test_decorator_auto_naming(self, performance_monitor):
        """Test decorator with automatic operation naming."""
        from .monitoring import monitor_performance
        
        @monitor_performance(performance_monitor)  # No operation name provided
        def auto_named_function():
            time.sleep(0.01)
            return "success"
        
        result = auto_named_function()
        assert result == "success"
        
        # Check that timing was recorded with auto-generated name
        perf_metrics = performance_monitor.metrics_collector.get_performance_metrics()
        
        # Should have an entry with the module and function name
        operation_names = list(perf_metrics.keys())
        assert len(operation_names) == 1
        assert "auto_named_function" in operation_names[0]