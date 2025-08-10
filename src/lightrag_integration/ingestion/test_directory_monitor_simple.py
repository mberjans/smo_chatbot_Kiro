"""
Simple Tests for Directory Monitor

This module contains basic tests for the directory monitoring functionality.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from .directory_monitor import DirectoryMonitor, BatchProcessor, MonitorStatus
from .progress_tracker import ProgressTracker, OperationType


@pytest.mark.asyncio
class TestDirectoryMonitorBasic:
    """Basic test cases for DirectoryMonitor class."""
    
    async def test_monitor_creation(self):
        """Test creating a DirectoryMonitor instance."""
        with tempfile.TemporaryDirectory() as temp_dir:
            papers_dir = Path(temp_dir) / "papers"
            papers_dir.mkdir()
            
            callback = AsyncMock()
            config = {'scan_interval': 1, 'batch_size': 2}
            
            monitor = DirectoryMonitor(
                papers_directory=str(papers_dir),
                ingestion_callback=callback,
                config=config
            )
            
            assert monitor.papers_directory == papers_dir
            assert monitor.status == MonitorStatus.STOPPED
            assert monitor.scan_interval == 1
            assert monitor.batch_size == 2
            
            await monitor.cleanup()
    
    async def test_monitor_start_stop(self):
        """Test starting and stopping the monitor."""
        with tempfile.TemporaryDirectory() as temp_dir:
            papers_dir = Path(temp_dir) / "papers"
            papers_dir.mkdir()
            
            callback = AsyncMock()
            config = {'scan_interval': 1, 'batch_size': 2}
            
            monitor = DirectoryMonitor(
                papers_directory=str(papers_dir),
                ingestion_callback=callback,
                config=config
            )
            
            try:
                # Test start
                await monitor.start()
                assert monitor.status == MonitorStatus.RUNNING
                assert monitor.stats.start_time is not None
                
                # Test stop
                await monitor.stop()
                assert monitor.status == MonitorStatus.STOPPED
                
            finally:
                await monitor.cleanup()
    
    async def test_force_scan_empty_directory(self):
        """Test forced scan on empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            papers_dir = Path(temp_dir) / "papers"
            papers_dir.mkdir()
            
            callback = AsyncMock()
            config = {'scan_interval': 1, 'batch_size': 2}
            
            monitor = DirectoryMonitor(
                papers_directory=str(papers_dir),
                ingestion_callback=callback,
                config=config
            )
            
            try:
                result = await monitor.force_scan()
                
                assert 'new_files_found' in result
                assert result['new_files_found'] == 0
                assert result['new_files'] == []
                
            finally:
                await monitor.cleanup()
    
    async def test_force_scan_with_pdf_files(self):
        """Test forced scan with PDF files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            papers_dir = Path(temp_dir) / "papers"
            papers_dir.mkdir()
            
            # Create test PDF files
            (papers_dir / "test1.pdf").touch()
            (papers_dir / "test2.pdf").touch()
            (papers_dir / "not_pdf.txt").touch()  # Should be ignored
            
            callback = AsyncMock()
            config = {'scan_interval': 1, 'batch_size': 2}
            
            monitor = DirectoryMonitor(
                papers_directory=str(papers_dir),
                ingestion_callback=callback,
                config=config
            )
            
            try:
                result = await monitor.force_scan()
                
                assert 'new_files_found' in result
                assert result['new_files_found'] == 2
                assert len(result['new_files']) == 2
                
                # Check that only PDF files are included
                pdf_files = [f for f in result['new_files'] if f.endswith('.pdf')]
                assert len(pdf_files) == 2
                
            finally:
                await monitor.cleanup()
    
    async def test_get_status(self):
        """Test getting monitor status."""
        with tempfile.TemporaryDirectory() as temp_dir:
            papers_dir = Path(temp_dir) / "papers"
            papers_dir.mkdir()
            
            callback = AsyncMock()
            config = {'scan_interval': 1, 'batch_size': 2}
            
            monitor = DirectoryMonitor(
                papers_directory=str(papers_dir),
                ingestion_callback=callback,
                config=config
            )
            
            try:
                status = await monitor.get_status()
                
                assert 'status' in status
                assert 'papers_directory' in status
                assert 'stats' in status
                assert status['status'] == MonitorStatus.STOPPED.value
                assert status['papers_directory'] == str(papers_dir)
                
            finally:
                await monitor.cleanup()


@pytest.mark.asyncio
class TestBatchProcessorBasic:
    """Basic test cases for BatchProcessor class."""
    
    async def test_process_empty_list(self):
        """Test processing empty file list."""
        mock_pipeline = MagicMock()
        config = {'batch_size': 2, 'max_concurrent': 2}
        
        processor = BatchProcessor(mock_pipeline, config)
        result = await processor.process_files([])
        
        assert result['total_files'] == 0
        assert result['successful'] == 0
        assert result['failed'] == 0
        assert result['results'] == []
    
    async def test_process_single_file_success(self):
        """Test processing single file successfully."""
        mock_pipeline = MagicMock()
        mock_pipeline.process_file = AsyncMock()
        mock_pipeline.process_file.return_value = {
            'success': True,
            'file_path': 'test.pdf',
            'processing_time': 1.0,
            'timestamp': datetime.now().isoformat()
        }
        
        config = {'batch_size': 2, 'max_concurrent': 2}
        processor = BatchProcessor(mock_pipeline, config)
        
        result = await processor.process_files(['test.pdf'])
        
        assert result['total_files'] == 1
        assert result['successful'] == 1
        assert result['failed'] == 0
        assert len(result['results']) == 1
        
        # Check that pipeline was called
        mock_pipeline.process_file.assert_called_once_with('test.pdf')
    
    async def test_process_multiple_files(self):
        """Test processing multiple files."""
        mock_pipeline = MagicMock()
        mock_pipeline.process_file = AsyncMock()
        mock_pipeline.process_file.return_value = {
            'success': True,
            'file_path': 'test.pdf',
            'processing_time': 1.0,
            'timestamp': datetime.now().isoformat()
        }
        
        config = {'batch_size': 2, 'max_concurrent': 2}
        processor = BatchProcessor(mock_pipeline, config)
        
        files = ['test1.pdf', 'test2.pdf', 'test3.pdf']
        result = await processor.process_files(files)
        
        assert result['total_files'] == 3
        assert result['successful'] == 3
        assert result['failed'] == 0
        assert len(result['results']) == 3
        
        # Check that pipeline was called for each file
        assert mock_pipeline.process_file.call_count == 3
    
    async def test_process_with_failures(self):
        """Test processing with some failures."""
        mock_pipeline = MagicMock()
        
        def mock_process_file(file_path):
            if 'fail' in file_path:
                raise Exception(f"Processing failed for {file_path}")
            return {
                'success': True,
                'file_path': file_path,
                'processing_time': 1.0,
                'timestamp': datetime.now().isoformat()
            }
        
        mock_pipeline.process_file = AsyncMock(side_effect=mock_process_file)
        
        config = {'batch_size': 10, 'max_concurrent': 2}
        processor = BatchProcessor(mock_pipeline, config)
        
        files = ['success1.pdf', 'fail1.pdf', 'success2.pdf', 'fail2.pdf']
        result = await processor.process_files(files)
        
        assert result['total_files'] == 4
        assert result['successful'] == 2
        assert result['failed'] == 2
        assert len(result['errors']) == 2


@pytest.mark.asyncio
class TestProgressTrackerBasic:
    """Basic test cases for ProgressTracker class."""
    
    async def test_tracker_creation(self):
        """Test creating a ProgressTracker instance."""
        config = {'max_history': 100, 'persist_to_file': False}
        tracker = ProgressTracker(config)
        
        assert tracker.max_history == 100
        assert tracker.persist_to_file is False
        
        await tracker.start()
        await tracker.stop()
    
    async def test_start_operation(self):
        """Test starting operation tracking."""
        config = {'max_history': 100, 'persist_to_file': False}
        tracker = ProgressTracker(config)
        
        await tracker.start()
        
        try:
            operation = tracker.start_operation(
                operation_id="test_op",
                operation_type=OperationType.FILE_PROCESSING,
                total_items=5
            )
            
            assert operation.operation_id == "test_op"
            assert operation.operation_type == OperationType.FILE_PROCESSING
            assert operation.total_items == 5
            
            # Check that operation is tracked
            tracked_op = tracker.get_operation("test_op")
            assert tracked_op is not None
            assert tracked_op.operation_id == "test_op"
            
        finally:
            await tracker.stop()
    
    async def test_complete_operation(self):
        """Test completing an operation."""
        config = {'max_history': 100, 'persist_to_file': False}
        tracker = ProgressTracker(config)
        
        await tracker.start()
        
        try:
            # Start operation
            tracker.start_operation(
                operation_id="test_op",
                operation_type=OperationType.FILE_PROCESSING
            )
            
            # Complete operation
            completed_op = tracker.complete_operation(
                operation_id="test_op",
                success=True
            )
            
            assert completed_op is not None
            assert completed_op.progress_percent == 100.0
            assert completed_op.end_time is not None
            
            # Operation should be moved to history
            assert tracker.get_operation("test_op") is None
            history = tracker.get_operation_history()
            assert len(history) == 1
            assert history[0].operation_id == "test_op"
            
        finally:
            await tracker.stop()
    
    async def test_get_status_report(self):
        """Test getting status report."""
        config = {'max_history': 100, 'persist_to_file': False}
        tracker = ProgressTracker(config)
        
        await tracker.start()
        
        try:
            # Create some operations
            tracker.start_operation("active1", OperationType.FILE_PROCESSING)
            tracker.start_operation("completed1", OperationType.BATCH_PROCESSING)
            tracker.complete_operation("completed1", success=True)
            
            # Get status report
            report = tracker.get_status_report()
            
            assert len(report.active_operations) == 1
            assert len(report.completed_operations) == 1
            assert report.total_operations == 2
            
        finally:
            await tracker.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])