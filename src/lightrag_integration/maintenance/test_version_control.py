"""
Tests for the version control system.

Tests cover:
- Version creation and management
- Backup and restore functionality
- Version history and metadata
- Cleanup and maintenance operations
"""

import asyncio
import json
import tempfile
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from .version_control import VersionManager, Version, VersionInfo
from ..error_handling import LightRAGError


class TestVersionManager:
    """Test suite for VersionManager."""
    
    @pytest.fixture
    async def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    async def version_manager(self, temp_dir):
        """Create VersionManager instance for testing."""
        kb_path = temp_dir / "knowledge_base"
        versions_path = temp_dir / "versions"
        
        # Create knowledge base directory with some test files
        kb_path.mkdir(exist_ok=True)
        (kb_path / "test1.txt").write_text("test content 1")
        (kb_path / "test2.txt").write_text("test content 2")
        
        manager = VersionManager(
            knowledge_base_path=str(kb_path),
            versions_path=str(versions_path),
            max_versions=5,
            compress_backups=False  # Disable compression for easier testing
        )
        
        await manager.initialize()
        return manager
    
    async def test_initialization(self, version_manager, temp_dir):
        """Test version manager initialization."""
        versions_path = temp_dir / "versions"
        
        # Check versions directory was created
        assert versions_path.exists()
        assert versions_path.is_dir()
        
        # Check versions metadata file was created
        versions_file = versions_path / "versions.json"
        assert versions_file.exists()
        
        # Check metadata is valid JSON
        with open(versions_file, 'r') as f:
            metadata = json.load(f)
            assert isinstance(metadata, dict)
    
    async def test_create_version(self, version_manager):
        """Test creating a new version."""
        description = "Test version creation"
        metadata = {"test_key": "test_value"}
        
        # Create version
        version = await version_manager.create_version(description, metadata)
        
        # Check version properties
        assert version.version_id.startswith("v")
        assert version.description == description
        assert version.metadata == metadata
        assert version.file_count > 0
        assert version.size_bytes > 0
        assert version.backup_path is not None
        
        # Check backup was created
        backup_path = Path(version.backup_path)
        assert backup_path.exists()
    
    async def test_list_versions(self, version_manager):
        """Test listing versions."""
        # Create multiple versions
        version1 = await version_manager.create_version("Version 1")
        await asyncio.sleep(0.1)  # Ensure different timestamps
        version2 = await version_manager.create_version("Version 2")
        
        # List versions
        versions = await version_manager.list_versions()
        
        # Should return versions in reverse chronological order
        assert len(versions) == 2
        assert versions[0].version_id == version2.version_id  # Newest first
        assert versions[1].version_id == version1.version_id
    
    async def test_list_versions_with_limit(self, version_manager):
        """Test listing versions with limit."""
        # Create multiple versions
        for i in range(3):
            await version_manager.create_version(f"Version {i}")
            await asyncio.sleep(0.1)
        
        # List with limit
        versions = await version_manager.list_versions(limit=2)
        
        assert len(versions) == 2
    
    async def test_get_version(self, version_manager):
        """Test getting a specific version."""
        # Create version
        created_version = await version_manager.create_version("Test version")
        
        # Get version
        retrieved_version = await version_manager.get_version(created_version.version_id)
        
        assert retrieved_version is not None
        assert retrieved_version.version_id == created_version.version_id
        assert retrieved_version.description == created_version.description
    
    async def test_get_nonexistent_version(self, version_manager):
        """Test getting a non-existent version."""
        version = await version_manager.get_version("nonexistent_version")
        assert version is None
    
    async def test_restore_version(self, version_manager, temp_dir):
        """Test restoring from a version."""
        kb_path = temp_dir / "knowledge_base"
        
        # Create initial version
        version = await version_manager.create_version("Initial version")
        
        # Modify knowledge base
        (kb_path / "new_file.txt").write_text("new content")
        (kb_path / "test1.txt").write_text("modified content")
        
        # Restore to previous version
        success = await version_manager.restore_version(version.version_id)
        
        assert success is True
        
        # Check that knowledge base was restored
        assert not (kb_path / "new_file.txt").exists()
        assert (kb_path / "test1.txt").read_text() == "test content 1"
    
    async def test_restore_nonexistent_version(self, version_manager):
        """Test restoring from a non-existent version."""
        with pytest.raises(LightRAGError):
            await version_manager.restore_version("nonexistent_version")
    
    async def test_delete_version(self, version_manager):
        """Test deleting a version."""
        # Create version
        version = await version_manager.create_version("Version to delete")
        backup_path = Path(version.backup_path)
        
        # Verify backup exists
        assert backup_path.exists()
        
        # Delete version
        success = await version_manager.delete_version(version.version_id)
        
        assert success is True
        
        # Check backup was removed
        assert not backup_path.exists()
        
        # Check version is no longer in list
        versions = await version_manager.list_versions()
        version_ids = [v.version_id for v in versions]
        assert version.version_id not in version_ids
    
    async def test_delete_nonexistent_version(self, version_manager):
        """Test deleting a non-existent version."""
        success = await version_manager.delete_version("nonexistent_version")
        assert success is False
    
    async def test_get_version_diff(self, version_manager, temp_dir):
        """Test getting differences between versions."""
        kb_path = temp_dir / "knowledge_base"
        
        # Create first version
        version1 = await version_manager.create_version("Version 1")
        
        # Modify knowledge base
        (kb_path / "new_file.txt").write_text("new content")
        
        # Create second version
        version2 = await version_manager.create_version("Version 2")
        
        # Get diff
        diff = await version_manager.get_version_diff(version1.version_id, version2.version_id)
        
        # Check diff structure
        assert 'version1' in diff
        assert 'version2' in diff
        assert 'differences' in diff
        
        # Check that version2 has more files
        assert diff['differences']['file_count_delta'] > 0
        assert diff['differences']['size_delta'] > 0
    
    async def test_cleanup_old_versions(self, version_manager):
        """Test cleanup of old versions when limit is exceeded."""
        # Create more versions than the limit (5)
        created_versions = []
        for i in range(7):
            version = await version_manager.create_version(f"Version {i}")
            created_versions.append(version)
            await asyncio.sleep(0.1)  # Ensure different timestamps
        
        # List versions - should only have the maximum allowed
        versions = await version_manager.list_versions()
        assert len(versions) <= version_manager.max_versions
        
        # Check that newest versions are kept
        version_ids = [v.version_id for v in versions]
        # Last 5 versions should be kept
        for version in created_versions[-5:]:
            assert version.version_id in version_ids
    
    async def test_calculate_kb_stats(self, version_manager, temp_dir):
        """Test knowledge base statistics calculation."""
        # Calculate stats
        stats = await version_manager._calculate_kb_stats()
        
        # Should have counted the test files
        assert stats['file_count'] >= 2  # At least test1.txt and test2.txt
        assert stats['size_bytes'] > 0
    
    async def test_calculate_kb_stats_empty_kb(self, temp_dir):
        """Test statistics calculation with empty knowledge base."""
        # Create manager with non-existent KB path
        manager = VersionManager(
            knowledge_base_path=str(temp_dir / "nonexistent"),
            versions_path=str(temp_dir / "versions")
        )
        await manager.initialize()
        
        stats = await manager._calculate_kb_stats()
        
        assert stats['file_count'] == 0
        assert stats['size_bytes'] == 0
    
    async def test_metadata_operations(self, version_manager, temp_dir):
        """Test metadata save and load operations."""
        # Test data
        metadata = {
            "v1": {
                "created_at": "2024-01-01T10:00:00",
                "description": "Version 1",
                "file_count": 10,
                "size_bytes": 1000
            },
            "v2": {
                "created_at": "2024-01-01T11:00:00", 
                "description": "Version 2",
                "file_count": 15,
                "size_bytes": 1500
            }
        }
        
        # Save metadata
        await version_manager._save_versions_metadata(metadata)
        
        # Load metadata
        loaded_metadata = await version_manager._load_versions_metadata()
        
        assert loaded_metadata == metadata
    
    async def test_compressed_backups(self, temp_dir):
        """Test version manager with compressed backups."""
        kb_path = temp_dir / "knowledge_base"
        versions_path = temp_dir / "versions"
        
        # Create knowledge base
        kb_path.mkdir(exist_ok=True)
        (kb_path / "test.txt").write_text("test content")
        
        # Create manager with compression enabled
        manager = VersionManager(
            knowledge_base_path=str(kb_path),
            versions_path=str(versions_path),
            compress_backups=True
        )
        await manager.initialize()
        
        # Create version
        version = await manager.create_version("Compressed version")
        
        # Check that backup is compressed
        backup_path = Path(version.backup_path)
        assert backup_path.exists()
        assert backup_path.suffix == '.gz'
    
    @pytest.mark.asyncio
    async def test_concurrent_version_operations(self, version_manager):
        """Test concurrent version operations."""
        # Start multiple version creation tasks
        tasks = [
            version_manager.create_version(f"Concurrent version {i}")
            for i in range(3)
        ]
        
        # Wait for completion
        versions = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(versions) == 3
        for version in versions:
            assert version.version_id is not None
            assert version.backup_path is not None
        
        # All versions should be unique
        version_ids = [v.version_id for v in versions]
        assert len(set(version_ids)) == 3


if __name__ == "__main__":
    pytest.main([__file__])