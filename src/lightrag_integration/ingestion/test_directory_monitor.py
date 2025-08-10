"""
Tests for Directory Monitor

This module contains comprehensive tests for the directory monitoring
and batch processing functionality.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import time

from .directory_monitor import DirectoryMonitor, BatchProcessor, MonitorStatus, FileEvent
from .progress_tracker import ProgressTracker, OperationType
from ..config.settings import LightRAGConfig

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio


class TestDirectoryMonitor:
    """Test cases for DirectoryMonitor class."""
    
    @pytest.fixture
    def temp_papers_dir(self):
        """Create a temporary papers directory for testing."""
        temp_dir = tempfile.mkdtemp()
        papers_dir = Path(temp_dir) / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)
        
        yield str(papers_dir)
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def mock_ingestion_callback(self):
        """Create a mock ingestion callback."""
        callback = AsyncMock()
        callback.return_value = {
            'successful': 1,
            'failed': 0,
            'errors': []
        }
        return callback
    
    @pytest.fixture
    def monitor_config(self):
        """Create monitor configuration for testing."""
        return {
            'scan_interval': 1,  # Fast scanning for tests
            'batch_size': 2,
            'max_file_age_hours': 24,
            'enable_recursive': False,
            'file_extensions': ['.pdf']
        }
    
    @pytest.fixture
    def directory_monitor(self, temp_papers_dir, mock_ingestion_callback, monitor_config):
        """Create a DirectoryMonitor instance for testing."""
        monitor = DirectoryMonitor(
            papers_directory=temp_papers_dir,
            ingestion_callback=mock_ingestion_callback,
            config=monitor_config
        )
        
        yield monitor
        
        # Cleanup will be handled in individual tests
    
    async def test_monitor_initialization(self, directory_monitor, temp_papers_dir):
        """Test monitor initialization."""
        try:
            assert directory_monitor.papers_directory == Path(temp_papers_dir)
            assert directory_monitor.status == MonitorStatus.STOPPED
            assert directory_monitor.scan_interval == 1
            assert directory_monitor.batch_size == 2
        finally:
            await directory_monitor.cleanup()
    
    async def test_start_stop_monitor(self, directory_monitor):
        """Test starting and stopping the monitor."""
        try:
            # Test start
            await directory_monitor.start()
            assert directory_monitor.status == MonitorStatus.RUNNING
            assert directory_monitor.stats.start_time is not None
            
            # Test stop
            await directory_monitor.stop()
            assert directory_monitor.status == MonitorStatus.STOPPED
        finally:
            await directory_monitor.cleanup()
    
    async def test_initial_scan_existing_files(self, temp_papers_dir, mock_ingestion_callback, monitor_config):
        """Test initial scan with existing PDF files."""
        # Create some existing PDF files
        papers_dir = Path(temp_papers_dir)
        (papers_dir / "existing1.pdf").touch()
        (papers_dir / "existing2.pdf").touch()
        (papers_dir / "not_pdf.txt").touch()
        
        monitor = DirectoryMonitor(temp_papers_dir, mock_ingestion_callback, monitor_config)
        
        # Start monitor (triggers initial scan)
        await monitor.start()
        
        # Check that existing files are tracked but not processed
        assert len(monitor._known_files) == 2  # Only PDF files
        assert str(papers_dir / "existing1.pdf") in monitor._known_files
        assert str(papers_dir / "existing2.pdf") in monitor._known_files
        
        # Ingestion callback should not be called for existing files
        mock_ingestion_callback.assert_not_called()
        
        await monitor.stop()
    
    async def test_detect_new_files(self, directory_monitor, temp_papers_dir, mock_ingestion_callback):
        """Test detection of new PDF files."""
        await directory_monitor.start()
        
        # Wait a moment for initial scan
        await asyncio.sleep(0.1)
        
        # Create new PDF files
        papers_dir = Path(temp_papers_dir)
        new_file1 = papers_dir / "new1.pdf"
        new_file2 = papers_dir / "new2.pdf"
        
        new_file1.touch()
        new_file2.touch()
        
        # Wait for scan to detect new files
        await asyncio.sleep(2)
        
        # Check that ingestion callback was called
        assert mock_ingestion_callback.called
        
        # Check that new files are in known files
        assert str(new_file1) in directory_monitor._known_files
        assert str(new_file2) in directory_monitor._known_files
        
        await directory_monitor.stop()
    
    async def test_batch_processing(self, directory_monitor, temp_papers_dir, mock_ingestion_callback):
        """Test batch processing of multiple files."""
        # Set batch size to 2
        directory_monitor.batch_size = 2
        
        await directory_monitor.start()
        await asyncio.sleep(0.1)
        
        # Create 3 PDF files (should trigger batch processing)
        papers_dir = Path(temp_papers_dir)
        for i in range(3):
            (papers_dir / f"batch{i}.pdf").touch()
        
        # Wait for processing
        await asyncio.sleep(3)
        
        # Should have been called at least once (batch of 2)
        assert mock_ingestion_callback.called
        
        await directory_monitor.stop()
    
    async def test_force_scan(self, directory_monitor, temp_papers_dir):
        """Test forced directory scan."""
        # Create PDF files
        papers_dir = Path(temp_papers_dir)
        (papers_dir / "force1.pdf").touch()
        (papers_dir / "force2.pdf").touch()
        
        # Force scan without starting monitor
        result = await directory_monitor.force_scan()
        
        assert 'new_files_found' in result
        assert result['new_files_found'] == 2
        assert len(result['new_files']) == 2
    
    async def test_get_status(self, directory_monitor):
        """Test getting monitor status."""
        status = await directory_monitor.get_status()
        
        assert 'status' in status
        assert 'papers_directory' in status
        assert 'stats' in status
        assert status['status'] == MonitorStatus.STOPPED.value
    
    async def test_file_age_filtering(self, temp_papers_dir, mock_ingestion_callback):
        """Test filtering files by age."""
        # Create monitor with short max file age
        config = {
            'scan_interval': 1,
            'batch_size': 10,
            'max_file_age_hours': 0.001,  # Very short age limit
            'enable_recursive': False,
            'file_extensions': ['.pdf']
        }
        
        monitor = DirectoryMonitor(temp_papers_dir, mock_ingestion_callback, config)
        
        # Create an old file
        papers_dir = Path(temp_papers_dir)
        old_file = papers_dir / "old.pdf"
        old_file.touch()
        
        # Wait to make file "old"
        await asyncio.sleep(0.1)
        
        # Start monitor
        await monitor.start()
        await asyncio.sleep(0.1)
        
        # Old file should not be processed
        mock_ingestion_callback.assert_not_called()
        
        await monitor.stop()
    
    async def test_recursive_scanning(self, temp_papers_dir, mock_ingestion_callback):
        """Test recursive directory scanning."""
        # Create monitor with recursive enabled
        config = {
            'scan_interval': 1,
            'batch_size': 10,
            'max_file_age_hours': 24,
            'enable_recursive': True,
            'file_extensions': ['.pdf']
        }
        
        monitor = DirectoryMonitor(temp_papers_dir, mock_ingestion_callback, config)
        
        # Create nested directory structure
        papers_dir = Path(temp_papers_dir)
        subdir = papers_dir / "subdir"
        subdir.mkdir()
        
        (papers_dir / "root.pdf").touch()
        (subdir / "nested.pdf").touch()
        
        # Force scan
        result = await monitor.force_scan()
        
        assert result['new_files_found'] == 2
        
        await monitor.cleanup()
    
    async def test_error_handling(self, temp_papers_dir, monitor_config):
        """Test error handling in monitor."""
        # Create callback that raises exception
        error_callback = AsyncMock()
        error_callback.side_effect = Exception("Test error")
        
        monitor = DirectoryMonitor(temp_papers_dir, error_callback, monitor_config)
        
        # Create PDF file
        papers_dir = Path(temp_papers_dir)
        (papers_dir / "error_test.pdf").touch()
        
        # Force scan should handle error gracefully
        result = await monitor.force_scan()
        
        # Should still return result even with error
        assert 'new_files_found' in result
        
        await monitor.cleanup()


class TestBatchProcessor:
    """Test cases for BatchProcessor class."""
    
    @pytest.fixture
    def mock_ingestion_pipeline(self):
        """Create a mock ingestion pipeline."""
        pipeline = MagicMock()
        pipeline.process_file = AsyncMock()
        pipeline.process_file.return_value = {
            'success': True,
            'file_path': 'test.pdf',
            'processing_time': 1.0,
            'timestamp': datetime.now().isoformat()
        }
        return pipeline
    
    @pytest.fixture
    def batch_config(self):
        """Create batch processor configuration."""
        return {
            'batch_size': 2,
            'max_concurrent': 2,
            'progress_callback': None
        }
    
    @pytest.fixture
    def batch_processor(self, mock_ingestion_pipeline, batch_config):
        """Create a BatchProcessor instance for testing."""
        return BatchProcessor(mock_ingestion_pipeline, batch_config)
    
    async def test_process_empty_list(self, batch_processor):
        """Test processing empty file list."""
        result = await batch_processor.process_files([])
        
        assert result['total_files'] == 0
        assert result['successful'] == 0
        assert result['failed'] == 0
        assert result['results'] == []
    
    async def test_process_single_file(self, batch_processor, mock_ingestion_pipeline):
        """Test processing single file."""
        file_paths = ['test1.pdf']
        
        result = await batch_processor.process_files(file_paths)
        
        assert result['total_files'] == 1
        assert result['successful'] == 1
        assert result['failed'] == 0
        assert len(result['results']) == 1
        
        # Check that pipeline was called
        mock_ingestion_pipeline.process_file.assert_called_once_with('test1.pdf')
    
    async def test_process_multiple_files(self, batch_processor, mock_ingestion_pipeline):
        """Test processing multiple files."""
        file_paths = ['test1.pdf', 'test2.pdf', 'test3.pdf']
        
        result = await batch_processor.process_files(file_paths)
        
        assert result['total_files'] == 3
        assert result['successful'] == 3
        assert result['failed'] == 0
        assert len(result['results']) == 3
        
        # Check that pipeline was called for each file
        assert mock_ingestion_pipeline.process_file.call_count == 3
    
    async def test_batch_processing(self, mock_ingestion_pipeline):
        """Test batch processing with specific batch size."""
        config = {
            'batch_size': 2,
            'max_concurrent': 1,
            'progress_callback': None
        }
        
        processor = BatchProcessor(mock_ingestion_pipeline, config)
        file_paths = ['test1.pdf', 'test2.pdf', 'test3.pdf', 'test4.pdf']
        
        result = await processor.process_files(file_paths)
        
        assert result['total_files'] == 4
        assert result['successful'] == 4
        assert result['failed'] == 0
    
    async def test_concurrent_processing(self, mock_ingestion_pipeline):
        """Test concurrent file processing."""
        # Add delay to simulate processing time
        async def slow_process(file_path):
            await asyncio.sleep(0.1)
            return {
                'success': True,
                'file_path': file_path,
                'processing_time': 0.1,
                'timestamp': datetime.now().isoformat()
            }
        
        mock_ingestion_pipeline.process_file.side_effect = slow_process
        
        config = {
            'batch_size': 10,
            'max_concurrent': 3,
            'progress_callback': None
        }
        
        processor = BatchProcessor(mock_ingestion_pipeline, config)
        file_paths = [f'test{i}.pdf' for i in range(6)]
        
        start_time = time.time()
        result = await processor.process_files(file_paths)
        end_time = time.time()
        
        # With concurrency, should be faster than sequential processing
        assert result['successful'] == 6
        assert end_time - start_time < 0.6  # Should be much faster than 0.6s (6 * 0.1s)
    
    async def test_error_handling(self, mock_ingestion_pipeline, batch_config):
        """Test error handling in batch processing."""
        # Make some files fail
        def process_with_errors(file_path):
            if 'fail' in file_path:
                raise Exception(f"Processing failed for {file_path}")
            return {
                'success': True,
                'file_path': file_path,
                'processing_time': 1.0,
                'timestamp': datetime.now().isoformat()
            }
        
        mock_ingestion_pipeline.process_file.side_effect = process_with_errors
        
        processor = BatchProcessor(mock_ingestion_pipeline, batch_config)
        file_paths = ['success1.pdf', 'fail1.pdf', 'success2.pdf', 'fail2.pdf']
        
        result = await processor.process_files(file_paths)
        
        assert result['total_files'] == 4
        assert result['successful'] == 2
        assert result['failed'] == 2
        assert len(result['errors']) == 2
    
    async def test_progress_callback(self, mock_ingestion_pipeline):
        """Test progress callback functionality."""
        progress_updates = []
        
        async def progress_callback(progress):
            progress_updates.append(progress.copy())
        
        config = {
            'batch_size': 2,
            'max_concurrent': 1,
            'progress_callback': progress_callback
        }
        
        processor = BatchProcessor(mock_ingestion_pipeline, config)
        file_paths = ['test1.pdf', 'test2.pdf', 'test3.pdf']
        
        await processor.process_files(file_paths)
        
        # Should have received progress updates
        assert len(progress_updates) > 0
        
        # Check final progress
        final_progress = progress_updates[-1]
        assert final_progress['completed_files'] == 3
        assert final_progress['total_files'] == 3
        assert final_progress['progress_percent'] == 100.0


class TestIntegration:
    """Integration tests for directory monitoring and batch processing."""
    
    @pytest.fixture
    def temp_papers_dir(self):
        """Create a temporary papers directory for testing."""
        temp_dir = tempfile.mkdtemp()
        papers_dir = Path(temp_dir) / "papers"
        papers_dir.mkdir(parents=True, exist_ok=True)
        
        yield str(papers_dir)
        
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def lightrag_config(self, temp_papers_dir):
        """Create LightRAG configuration for testing."""
        temp_dir = Path(temp_papers_dir).parent
        
        config = LightRAGConfig()
        config.papers_directory = temp_papers_dir
        config.knowledge_graph_path = str(temp_dir / "kg")
        config.vector_store_path = str(temp_dir / "vectors")
        config.cache_directory = str(temp_dir / "cache")
        
        return config
    
    async def test_component_integration(self, lightrag_config):
        """Test integration with LightRAG component."""
        from ..component import LightRAGComponent
        
        component = LightRAGComponent(lightrag_config)
        
        try:
            # Initialize component
            await component.initialize()
            
            # Test directory monitoring methods
            status = await component.get_monitoring_status()
            assert 'status' in status
            
            # Test progress reporting
            progress = await component.get_progress_report()
            assert 'timestamp' in progress
            
            # Test force scan
            scan_result = await component.force_directory_scan()
            assert 'new_files_found' in scan_result
            
        finally:
            await component.cleanup()
    
    async def test_end_to_end_workflow(self, temp_papers_dir, lightrag_config):
        """Test complete end-to-end workflow."""
        from ..component import LightRAGComponent
        
        component = LightRAGComponent(lightrag_config)
        
        try:
            # Initialize and start monitoring
            await component.initialize()
            await component.start_directory_monitoring()
            
            # Create PDF files
            papers_dir = Path(temp_papers_dir)
            test_files = []
            for i in range(3):
                test_file = papers_dir / f"test{i}.pdf"
                test_file.write_text(f"Test PDF content {i}")
                test_files.append(str(test_file))
            
            # Wait for processing
            await asyncio.sleep(2)
            
            # Check status
            status = await component.get_monitoring_status()
            assert status['stats']['files_detected'] >= 0
            
            # Test batch processing directly
            batch_result = await component.batch_process_files(test_files)
            assert batch_result['total_files'] == 3
            
        finally:
            await component.stop_directory_monitoring()
            await component.cleanup()


# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])