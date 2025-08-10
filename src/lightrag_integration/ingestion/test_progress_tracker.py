"""
Tests for Progress Tracker

This module contains comprehensive tests for the progress tracking
and status reporting functionality.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from .progress_tracker import (
    ProgressTracker, OperationProgress, StatusReport,
    OperationStatus, OperationType
)

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio


class TestOperationProgress:
    """Test cases for OperationProgress class."""
    
    def test_operation_progress_creation(self):
        """Test creating OperationProgress instance."""
        start_time = datetime.now()
        
        progress = OperationProgress(
            operation_id="test_op",
            operation_type=OperationType.FILE_PROCESSING,
            status=OperationStatus.RUNNING,
            start_time=start_time,
            total_items=10
        )
        
        assert progress.operation_id == "test_op"
        assert progress.operation_type == OperationType.FILE_PROCESSING
        assert progress.status == OperationStatus.RUNNING
        assert progress.start_time == start_time
        assert progress.total_items == 10
        assert progress.completed_items == 0
        assert progress.failed_items == 0
    
    def test_duration_property(self):
        """Test duration property calculation."""
        start_time = datetime.now() - timedelta(seconds=10)
        end_time = datetime.now()
        
        # Active operation (no end time)
        progress = OperationProgress(
            operation_id="test_op",
            operation_type=OperationType.FILE_PROCESSING,
            status=OperationStatus.RUNNING,
            start_time=start_time
        )
        
        duration = progress.duration
        assert duration.total_seconds() >= 10
        
        # Completed operation (with end time)
        progress.end_time = end_time
        progress.status = OperationStatus.COMPLETED
        
        duration = progress.duration
        assert duration.total_seconds() >= 10
    
    def test_is_active_property(self):
        """Test is_active property."""
        progress = OperationProgress(
            operation_id="test_op",
            operation_type=OperationType.FILE_PROCESSING,
            status=OperationStatus.RUNNING,
            start_time=datetime.now()
        )
        
        # Running operation should be active
        assert progress.is_active is True
        
        # Pending operation should be active
        progress.status = OperationStatus.PENDING
        assert progress.is_active is True
        
        # Completed operation should not be active
        progress.status = OperationStatus.COMPLETED
        assert progress.is_active is False
        
        # Failed operation should not be active
        progress.status = OperationStatus.FAILED
        assert progress.is_active is False
    
    def test_success_rate_property(self):
        """Test success rate calculation."""
        progress = OperationProgress(
            operation_id="test_op",
            operation_type=OperationType.BATCH_PROCESSING,
            status=OperationStatus.RUNNING,
            start_time=datetime.now(),
            completed_items=8,
            failed_items=2
        )
        
        # 8 successful out of 10 total = 80%
        assert progress.success_rate == 80.0
        
        # No items processed yet
        progress.completed_items = 0
        progress.failed_items = 0
        assert progress.success_rate == 0.0
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        start_time = datetime.now()
        
        progress = OperationProgress(
            operation_id="test_op",
            operation_type=OperationType.FILE_PROCESSING,
            status=OperationStatus.RUNNING,
            start_time=start_time,
            total_items=5,
            completed_items=3,
            failed_items=1,
            current_step="Processing file 4",
            metadata={"source": "test"}
        )
        
        result = progress.to_dict()
        
        assert result['operation_id'] == "test_op"
        assert result['operation_type'] == "file_processing"
        assert result['status'] == "running"
        assert result['start_time'] == start_time.isoformat()
        assert result['total_items'] == 5
        assert result['completed_items'] == 3
        assert result['failed_items'] == 1
        assert result['current_step'] == "Processing file 4"
        assert result['success_rate'] == 75.0  # 3/(3+1) * 100
        assert result['metadata'] == {"source": "test"}
        assert result['is_active'] is True


class TestStatusReport:
    """Test cases for StatusReport class."""
    
    def test_status_report_creation(self):
        """Test creating StatusReport instance."""
        timestamp = datetime.now()
        
        active_op = OperationProgress(
            operation_id="active_op",
            operation_type=OperationType.FILE_PROCESSING,
            status=OperationStatus.RUNNING,
            start_time=timestamp
        )
        
        completed_op = OperationProgress(
            operation_id="completed_op",
            operation_type=OperationType.BATCH_PROCESSING,
            status=OperationStatus.COMPLETED,
            start_time=timestamp - timedelta(minutes=5),
            end_time=timestamp
        )
        
        failed_op = OperationProgress(
            operation_id="failed_op",
            operation_type=OperationType.ENTITY_EXTRACTION,
            status=OperationStatus.FAILED,
            start_time=timestamp - timedelta(minutes=10),
            end_time=timestamp - timedelta(minutes=5)
        )
        
        report = StatusReport(
            timestamp=timestamp,
            active_operations=[active_op],
            completed_operations=[completed_op],
            failed_operations=[failed_op],
            system_stats={"uptime": 3600}
        )
        
        assert report.timestamp == timestamp
        assert len(report.active_operations) == 1
        assert len(report.completed_operations) == 1
        assert len(report.failed_operations) == 1
        assert report.system_stats == {"uptime": 3600}
    
    def test_total_operations_property(self):
        """Test total_operations property."""
        active_ops = [MagicMock() for _ in range(2)]
        completed_ops = [MagicMock() for _ in range(3)]
        failed_ops = [MagicMock() for _ in range(1)]
        
        report = StatusReport(
            timestamp=datetime.now(),
            active_operations=active_ops,
            completed_operations=completed_ops,
            failed_operations=failed_ops
        )
        
        assert report.total_operations == 6
    
    def test_overall_success_rate_property(self):
        """Test overall_success_rate property."""
        completed_ops = [MagicMock() for _ in range(8)]
        failed_ops = [MagicMock() for _ in range(2)]
        
        report = StatusReport(
            timestamp=datetime.now(),
            active_operations=[],
            completed_operations=completed_ops,
            failed_operations=failed_ops
        )
        
        # 8 successful out of 10 total = 80%
        assert report.overall_success_rate == 80.0
        
        # No completed operations
        report.completed_operations = []
        report.failed_operations = []
        assert report.overall_success_rate == 0.0
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        timestamp = datetime.now()
        
        active_op = OperationProgress(
            operation_id="active_op",
            operation_type=OperationType.FILE_PROCESSING,
            status=OperationStatus.RUNNING,
            start_time=timestamp
        )
        
        report = StatusReport(
            timestamp=timestamp,
            active_operations=[active_op],
            completed_operations=[],
            failed_operations=[],
            system_stats={"uptime": 3600}
        )
        
        result = report.to_dict()
        
        assert result['timestamp'] == timestamp.isoformat()
        assert result['total_operations'] == 1
        assert result['active_operations_count'] == 1
        assert result['completed_operations_count'] == 0
        assert result['failed_operations_count'] == 0
        assert result['overall_success_rate'] == 0.0
        assert len(result['active_operations']) == 1
        assert result['system_stats'] == {"uptime": 3600}


class TestProgressTracker:
    """Test cases for ProgressTracker class."""
    
    @pytest.fixture
    def tracker_config(self):
        """Create tracker configuration for testing."""
        return {
            'max_history': 100,
            'cleanup_interval': 10,
            'persist_to_file': False
        }
    
    @pytest.fixture
    def progress_tracker(self, tracker_config):
        """Create a ProgressTracker instance for testing."""
        tracker = ProgressTracker(tracker_config)
        return tracker
    
    async def test_tracker_initialization(self, tracker_config):
        """Test tracker initialization."""
        tracker = ProgressTracker(tracker_config)
        
        assert tracker.max_history == 100
        assert tracker.cleanup_interval == 10
        assert tracker.persist_to_file is False
    
    async def test_start_stop_tracker(self, tracker_config):
        """Test starting and stopping tracker."""
        tracker = ProgressTracker(tracker_config)
        
        # Start tracker
        await tracker.start()
        
        # Stop tracker
        await tracker.stop()
    
    async def test_start_operation(self, progress_tracker):
        """Test starting operation tracking."""
        await progress_tracker.start()
        
        try:
            operation = progress_tracker.start_operation(
                operation_id="test_op",
                operation_type=OperationType.FILE_PROCESSING,
                total_items=5,
                metadata={"source": "test"}
            )
            
            assert operation.operation_id == "test_op"
            assert operation.operation_type == OperationType.FILE_PROCESSING
            assert operation.status == OperationStatus.RUNNING
            assert operation.total_items == 5
            assert operation.metadata == {"source": "test"}
            
            # Check that operation is tracked
            tracked_op = progress_tracker.get_operation("test_op")
            assert tracked_op is not None
            assert tracked_op.operation_id == "test_op"
        finally:
            await progress_tracker.stop()
    
    async def test_update_progress(self, progress_tracker):
        """Test updating operation progress."""
        # Start operation
        progress_tracker.start_operation(
            operation_id="test_op",
            operation_type=OperationType.BATCH_PROCESSING,
            total_items=10
        )
        
        # Update progress
        updated_op = progress_tracker.update_progress(
            operation_id="test_op",
            progress_percent=50.0,
            current_step="Processing item 5",
            completed_items=4,
            failed_items=1,
            metadata={"current_file": "test.pdf"}
        )
        
        assert updated_op is not None
        assert updated_op.progress_percent == 50.0
        assert updated_op.current_step == "Processing item 5"
        assert updated_op.completed_items == 4
        assert updated_op.failed_items == 1
        assert updated_op.metadata["current_file"] == "test.pdf"
    
    async def test_auto_progress_calculation(self, progress_tracker):
        """Test automatic progress calculation based on items."""
        # Start operation with total items
        progress_tracker.start_operation(
            operation_id="test_op",
            operation_type=OperationType.BATCH_PROCESSING,
            total_items=10
        )
        
        # Update completed items without explicit progress
        updated_op = progress_tracker.update_progress(
            operation_id="test_op",
            completed_items=3,
            failed_items=2
        )
        
        # Progress should be auto-calculated: (3+2)/10 * 100 = 50%
        assert updated_op.progress_percent == 50.0
    
    async def test_complete_operation_success(self, progress_tracker):
        """Test completing operation successfully."""
        # Start operation
        progress_tracker.start_operation(
            operation_id="test_op",
            operation_type=OperationType.FILE_PROCESSING
        )
        
        # Complete operation
        completed_op = progress_tracker.complete_operation(
            operation_id="test_op",
            success=True,
            metadata={"result": "success"}
        )
        
        assert completed_op is not None
        assert completed_op.status == OperationStatus.COMPLETED
        assert completed_op.progress_percent == 100.0
        assert completed_op.end_time is not None
        assert completed_op.metadata["result"] == "success"
        
        # Operation should be moved to history
        assert progress_tracker.get_operation("test_op") is None
        history = progress_tracker.get_operation_history()
        assert len(history) == 1
        assert history[0].operation_id == "test_op"
    
    async def test_complete_operation_failure(self, progress_tracker):
        """Test completing operation with failure."""
        # Start operation
        progress_tracker.start_operation(
            operation_id="test_op",
            operation_type=OperationType.FILE_PROCESSING
        )
        
        # Complete operation with failure
        completed_op = progress_tracker.complete_operation(
            operation_id="test_op",
            success=False,
            error_message="Processing failed"
        )
        
        assert completed_op is not None
        assert completed_op.status == OperationStatus.FAILED
        assert completed_op.error_message == "Processing failed"
        
        # Operation should be moved to history
        history = progress_tracker.get_operation_history()
        assert len(history) == 1
        assert history[0].status == OperationStatus.FAILED
    
    async def test_cancel_operation(self, progress_tracker):
        """Test cancelling operation."""
        # Start operation
        progress_tracker.start_operation(
            operation_id="test_op",
            operation_type=OperationType.FILE_PROCESSING
        )
        
        # Cancel operation
        cancelled_op = progress_tracker.cancel_operation("test_op")
        
        assert cancelled_op is not None
        assert cancelled_op.status == OperationStatus.CANCELLED
        assert cancelled_op.end_time is not None
        
        # Operation should be moved to history
        assert progress_tracker.get_operation("test_op") is None
        history = progress_tracker.get_operation_history()
        assert len(history) == 1
        assert history[0].status == OperationStatus.CANCELLED
    
    async def test_get_active_operations(self, progress_tracker):
        """Test getting active operations."""
        # Start multiple operations
        progress_tracker.start_operation("op1", OperationType.FILE_PROCESSING)
        progress_tracker.start_operation("op2", OperationType.BATCH_PROCESSING)
        progress_tracker.start_operation("op3", OperationType.ENTITY_EXTRACTION)
        
        active_ops = progress_tracker.get_active_operations()
        assert len(active_ops) == 3
        
        operation_ids = [op.operation_id for op in active_ops]
        assert "op1" in operation_ids
        assert "op2" in operation_ids
        assert "op3" in operation_ids
    
    async def test_get_operation_history(self, progress_tracker):
        """Test getting operation history."""
        # Start and complete operations
        progress_tracker.start_operation("op1", OperationType.FILE_PROCESSING)
        progress_tracker.complete_operation("op1", success=True)
        
        progress_tracker.start_operation("op2", OperationType.BATCH_PROCESSING)
        progress_tracker.complete_operation("op2", success=False)
        
        # Get all history
        history = progress_tracker.get_operation_history()
        assert len(history) == 2
        
        # Get filtered history
        file_history = progress_tracker.get_operation_history(
            operation_type=OperationType.FILE_PROCESSING
        )
        assert len(file_history) == 1
        assert file_history[0].operation_id == "op1"
        
        # Get limited history
        limited_history = progress_tracker.get_operation_history(limit=1)
        assert len(limited_history) == 1
    
    async def test_get_status_report(self, progress_tracker):
        """Test getting comprehensive status report."""
        # Create mix of operations
        progress_tracker.start_operation("active1", OperationType.FILE_PROCESSING)
        progress_tracker.start_operation("active2", OperationType.BATCH_PROCESSING)
        
        progress_tracker.start_operation("completed1", OperationType.ENTITY_EXTRACTION)
        progress_tracker.complete_operation("completed1", success=True)
        
        progress_tracker.start_operation("failed1", OperationType.DIRECTORY_SCAN)
        progress_tracker.complete_operation("failed1", success=False)
        
        # Get status report
        report = progress_tracker.get_status_report()
        
        assert isinstance(report, StatusReport)
        assert len(report.active_operations) == 2
        assert len(report.completed_operations) == 1
        assert len(report.failed_operations) == 1
        assert report.total_operations == 4
        assert report.overall_success_rate == 50.0  # 1 success out of 2 completed
    
    async def test_callback_registration(self, progress_tracker):
        """Test callback registration and notification."""
        callback_calls = []
        
        async def test_callback(operation):
            callback_calls.append(operation.operation_id)
        
        # Register callback
        progress_tracker.register_callback('operation_started', test_callback)
        
        # Start operation (should trigger callback)
        progress_tracker.start_operation("test_op", OperationType.FILE_PROCESSING)
        
        # Wait for callback to be called
        await asyncio.sleep(0.1)
        
        assert "test_op" in callback_calls
        
        # Unregister callback
        progress_tracker.unregister_callback('operation_started', test_callback)
    
    async def test_history_cleanup(self, progress_tracker):
        """Test automatic history cleanup."""
        # Set small history limit
        progress_tracker.max_history = 3
        
        # Create more operations than the limit
        for i in range(5):
            progress_tracker.start_operation(f"op{i}", OperationType.FILE_PROCESSING)
            progress_tracker.complete_operation(f"op{i}", success=True)
        
        # Trigger cleanup
        await progress_tracker._cleanup_old_operations()
        
        # Should only keep the most recent operations
        history = progress_tracker.get_operation_history()
        assert len(history) <= 3
    
    async def test_persistence(self):
        """Test status persistence to file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            status_file = Path(temp_dir) / "test_status.json"
            
            config = {
                'max_history': 100,
                'cleanup_interval': 10,
                'persist_to_file': True,
                'status_file': str(status_file)
            }
            
            # Create tracker with persistence
            tracker = ProgressTracker(config)
            await tracker.start()
            
            try:
                # Create some operations
                tracker.start_operation("op1", OperationType.FILE_PROCESSING)
                tracker.complete_operation("op1", success=True)
                
                # Save status
                await tracker._save_status()
                
                # Check that file was created
                assert status_file.exists()
                
                # Check file content
                with open(status_file, 'r') as f:
                    data = json.load(f)
                
                assert 'completed_operations' in data
                assert len(data['completed_operations']) == 1
                
            finally:
                await tracker.stop()


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])