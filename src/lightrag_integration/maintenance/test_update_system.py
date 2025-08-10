"""
Tests for the knowledge base update system.

Tests cover:
- Incremental document processing
- Change detection and delta updates
- Version control integration
- Error handling and rollback
"""

import asyncio
import json
import tempfile
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

from .update_system import (
    KnowledgeBaseUpdater, 
    UpdateResult, 
    UpdateStatus,
    DocumentMetadata,
    ProcessResult
)
from .version_control import VersionManager, Version
from ..ingestion.pipeline import PDFIngestionPipeline
from ..error_handling import ErrorHandler


class LightRAGError(Exception):
    """Custom exception for LightRAG operations."""
    pass


class TestKnowledgeBaseUpdater:
    """Test suite for KnowledgeBaseUpdater."""
    
    @pytest.fixture
    async def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    async def mock_ingestion_pipeline(self):
        """Mock ingestion pipeline."""
        pipeline = AsyncMock(spec=PDFIngestionPipeline)
        pipeline.process_file.return_value = {
            'success': True,
            'processing_time': 1.0,
            'error_message': None
        }
        return pipeline
    
    @pytest.fixture
    async def mock_version_manager(self):
        """Mock version manager."""
        manager = AsyncMock(spec=VersionManager)
        manager.initialize.return_value = None
        manager.create_version.return_value = Version(
            version_id="v20240101_120000",
            created_at=datetime.now(),
            description="Test version",
            file_count=1,
            size_bytes=1000,
            metadata={}
        )
        return manager
    
    @pytest.fixture
    async def mock_error_handler(self):
        """Mock error handler."""
        handler = AsyncMock(spec=ErrorHandler)
        return handler
    
    @pytest.fixture
    async def updater(self, temp_dir, mock_ingestion_pipeline, mock_version_manager, mock_error_handler):
        """Create KnowledgeBaseUpdater instance for testing."""
        metadata_file = temp_dir / "metadata.json"
        watch_dirs = [str(temp_dir / "papers")]
        
        updater = KnowledgeBaseUpdater(
            ingestion_pipeline=mock_ingestion_pipeline,
            version_manager=mock_version_manager,
            error_handler=mock_error_handler,
            metadata_file=str(metadata_file),
            watch_directories=watch_dirs
        )
        
        # Create watch directory
        (temp_dir / "papers").mkdir(exist_ok=True)
        
        await updater.initialize()
        return updater
    
    async def test_initialization(self, updater, temp_dir):
        """Test updater initialization."""
        # Check metadata file was created
        metadata_file = temp_dir / "metadata.json"
        assert metadata_file.exists()
        
        # Check version manager was initialized
        updater.version_manager.initialize.assert_called_once()
    
    async def test_scan_for_updates_new_files(self, updater, temp_dir):
        """Test scanning for new files."""
        # Create a test PDF file
        papers_dir = temp_dir / "papers"
        test_file = papers_dir / "test.pdf"
        test_file.write_text("test content")
        
        # Scan for updates
        changes = await updater.scan_for_updates()
        
        # Should detect new file
        assert len(changes['new']) == 1
        assert str(test_file) in changes['new']
        assert len(changes['modified']) == 0
        assert len(changes['deleted']) == 0
    
    async def test_scan_for_updates_modified_files(self, updater, temp_dir):
        """Test scanning for modified files."""
        papers_dir = temp_dir / "papers"
        test_file = papers_dir / "test.pdf"
        test_file.write_text("original content")
        
        # First scan to establish baseline
        await updater.scan_for_updates()
        
        # Simulate processing by updating metadata
        metadata = {
            str(test_file): {
                'file_hash': 'old_hash',
                'file_size': 100,
                'last_modified': '2024-01-01T10:00:00'
            }
        }
        await updater._save_metadata(metadata)
        
        # Modify file
        test_file.write_text("modified content")
        
        # Scan again
        changes = await updater.scan_for_updates()
        
        # Should detect modified file
        assert len(changes['new']) == 0
        assert len(changes['modified']) == 1
        assert str(test_file) in changes['modified']
        assert len(changes['deleted']) == 0
    
    async def test_scan_for_updates_deleted_files(self, updater, temp_dir):
        """Test scanning for deleted files."""
        papers_dir = temp_dir / "papers"
        test_file = papers_dir / "test.pdf"
        
        # Set up metadata for existing file
        metadata = {
            str(test_file): {
                'file_hash': 'some_hash',
                'file_size': 100,
                'last_modified': '2024-01-01T10:00:00'
            }
        }
        await updater._save_metadata(metadata)
        
        # Scan for updates (file doesn't exist)
        changes = await updater.scan_for_updates()
        
        # Should detect deleted file
        assert len(changes['new']) == 0
        assert len(changes['modified']) == 0
        assert len(changes['deleted']) == 1
        assert str(test_file) in changes['deleted']
    
    async def test_perform_incremental_update_success(self, updater, temp_dir):
        """Test successful incremental update."""
        # Create test file
        papers_dir = temp_dir / "papers"
        test_file = papers_dir / "test.pdf"
        test_file.write_text("test content")
        
        # Perform update
        result = await updater.perform_incremental_update("test_update")
        
        # Check result
        assert result.update_id == "test_update"
        assert result.status == UpdateStatus.COMPLETED
        assert result.documents_processed == 1
        assert result.documents_added == 1
        assert result.documents_failed == 0
        assert result.end_time is not None
        
        # Check version was created
        updater.version_manager.create_version.assert_called()
    
    async def test_perform_incremental_update_no_changes(self, updater):
        """Test update with no changes."""
        # Perform update with no files
        result = await updater.perform_incremental_update("no_changes")
        
        # Should complete immediately
        assert result.status == UpdateStatus.COMPLETED
        assert result.documents_processed == 0
        assert result.documents_added == 0
    
    async def test_perform_incremental_update_with_failures(self, updater, temp_dir):
        """Test update with processing failures."""
        # Create test file
        papers_dir = temp_dir / "papers"
        test_file = papers_dir / "test.pdf"
        test_file.write_text("test content")
        
        # Mock processing failure
        updater.ingestion_pipeline.process_file.return_value = {
            'success': False,
            'processing_time': 0.0,
            'error_message': "Processing failed"
        }
        
        # Perform update
        result = await updater.perform_incremental_update("test_failures")
        
        # Should complete but with failures
        assert result.status == UpdateStatus.COMPLETED
        assert result.documents_processed == 0
        assert result.documents_failed == 1
    
    async def test_perform_incremental_update_exception_with_rollback(self, updater, temp_dir):
        """Test update exception handling with rollback."""
        # Create test file
        papers_dir = temp_dir / "papers"
        test_file = papers_dir / "test.pdf"
        test_file.write_text("test content")
        
        # Mock processing exception
        updater.ingestion_pipeline.process_file.side_effect = Exception("Processing error")
        
        # Mock rollback success
        updater.version_manager.restore_version.return_value = True
        
        # Perform update - should raise exception
        with pytest.raises(LightRAGError):
            await updater.perform_incremental_update("test_exception")
        
        # Check rollback was attempted
        updater.version_manager.restore_version.assert_called()
    
    async def test_rollback_update_success(self, updater):
        """Test successful update rollback."""
        # Mock successful rollback
        updater.version_manager.restore_version.return_value = True
        
        # Perform rollback
        success = await updater.rollback_update("v20240101_120000")
        
        assert success is True
        updater.version_manager.restore_version.assert_called_with("v20240101_120000")
    
    async def test_rollback_update_failure(self, updater):
        """Test failed update rollback."""
        # Mock failed rollback
        updater.version_manager.restore_version.return_value = False
        
        # Perform rollback
        success = await updater.rollback_update("v20240101_120000")
        
        assert success is False
    
    async def test_get_update_status(self, updater):
        """Test getting update status."""
        # Create mock update result
        update_result = UpdateResult(
            update_id="test_update",
            status=UpdateStatus.RUNNING,
            start_time=datetime.now()
        )
        updater.active_updates["test_update"] = update_result
        
        # Get status
        status = await updater.get_update_status("test_update")
        
        assert status == update_result
        assert status.update_id == "test_update"
        assert status.status == UpdateStatus.RUNNING
    
    async def test_list_active_updates(self, updater):
        """Test listing active updates."""
        # Create mock update results
        update1 = UpdateResult("update1", UpdateStatus.RUNNING, datetime.now())
        update2 = UpdateResult("update2", UpdateStatus.PENDING, datetime.now())
        
        updater.active_updates["update1"] = update1
        updater.active_updates["update2"] = update2
        
        # List updates
        updates = await updater.list_active_updates()
        
        assert len(updates) == 2
        assert update1 in updates
        assert update2 in updates
    
    async def test_calculate_file_hash(self, updater, temp_dir):
        """Test file hash calculation."""
        # Create test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        # Calculate hash
        hash1 = await updater._calculate_file_hash(test_file)
        hash2 = await updater._calculate_file_hash(test_file)
        
        # Should be consistent
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length
        
        # Different content should produce different hash
        test_file.write_text("different content")
        hash3 = await updater._calculate_file_hash(test_file)
        assert hash3 != hash1
    
    async def test_metadata_operations(self, updater, temp_dir):
        """Test metadata save and load operations."""
        # Test data
        metadata = {
            "file1.pdf": {
                "file_hash": "hash1",
                "file_size": 1000,
                "last_modified": "2024-01-01T10:00:00"
            },
            "file2.pdf": {
                "file_hash": "hash2", 
                "file_size": 2000,
                "last_modified": "2024-01-01T11:00:00"
            }
        }
        
        # Save metadata
        await updater._save_metadata(metadata)
        
        # Load metadata
        loaded_metadata = await updater._load_metadata()
        
        assert loaded_metadata == metadata
    
    @pytest.mark.asyncio
    async def test_concurrent_updates(self, updater, temp_dir):
        """Test handling of concurrent update requests."""
        # Create test files
        papers_dir = temp_dir / "papers"
        for i in range(3):
            test_file = papers_dir / f"test{i}.pdf"
            test_file.write_text(f"test content {i}")
        
        # Start multiple updates concurrently
        tasks = [
            updater.perform_incremental_update(f"update_{i}")
            for i in range(2)
        ]
        
        # Wait for completion
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check that both completed (one might fail due to concurrency)
        completed_count = sum(1 for r in results if isinstance(r, UpdateResult) and r.status == UpdateStatus.COMPLETED)
        assert completed_count >= 1  # At least one should complete


if __name__ == "__main__":
    pytest.main([__file__])