"""
Version control system for LightRAG knowledge base.

This module provides:
- Version creation and management
- Snapshot functionality for rollback
- Version metadata and history tracking
- Backup and restore capabilities
"""

import asyncio
import json
import os
import shutil
import tarfile
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

try:
    import aiofiles
    import aiofiles.os
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

class LightRAGError(Exception):
    """Custom exception for LightRAG operations."""
    pass


@dataclass
class Version:
    """Represents a version of the knowledge base."""
    version_id: str
    created_at: datetime
    description: str
    file_count: int
    size_bytes: int
    metadata: Dict[str, Any]
    backup_path: Optional[str] = None


@dataclass 
class VersionInfo:
    """Extended version information with statistics."""
    version: Version
    documents_added: int = 0
    documents_modified: int = 0
    documents_removed: int = 0
    entities_count: int = 0
    relationships_count: int = 0


class VersionManager:
    """
    Manages versions and snapshots of the LightRAG knowledge base.
    
    Provides functionality for:
    - Creating version snapshots
    - Restoring from versions
    - Version history management
    - Backup and cleanup operations
    """
    
    def __init__(
        self,
        knowledge_base_path: str,
        versions_path: str = "data/versions",
        max_versions: int = 50,
        compress_backups: bool = True
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.versions_path = Path(versions_path)
        self.max_versions = max_versions
        self.compress_backups = compress_backups
        self.logger = logging.getLogger(__name__)
        
        # Version metadata file
        self.versions_file = self.versions_path / "versions.json"
        
    async def initialize(self) -> None:
        """Initialize the version control system."""
        try:
            # Create versions directory
            self.versions_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize versions file if it doesn't exist
            if not self.versions_file.exists():
                await self._save_versions_metadata({})
                
            self.logger.info("Version manager initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize version manager: {e}")
            raise LightRAGError(f"Version manager initialization failed: {e}")
    
    async def create_version(
        self,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Version:
        """
        Create a new version snapshot of the knowledge base.
        
        Args:
            description: Description of this version
            metadata: Additional metadata to store
            
        Returns:
            Created Version object
        """
        try:
            version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            created_at = datetime.now()
            
            # Calculate knowledge base statistics
            stats = await self._calculate_kb_stats()
            
            # Create version object
            version = Version(
                version_id=version_id,
                created_at=created_at,
                description=description,
                file_count=stats['file_count'],
                size_bytes=stats['size_bytes'],
                metadata=metadata or {}
            )
            
            # Create backup
            backup_path = await self._create_backup(version_id)
            version.backup_path = str(backup_path)
            
            # Save version metadata
            await self._add_version_to_metadata(version)
            
            # Cleanup old versions if needed
            await self._cleanup_old_versions()
            
            self.logger.info(f"Created version {version_id}: {description}")
            return version
            
        except Exception as e:
            self.logger.error(f"Failed to create version: {e}")
            raise LightRAGError(f"Version creation failed: {e}")
    
    async def restore_version(self, version_id: str) -> bool:
        """
        Restore knowledge base from a specific version.
        
        Args:
            version_id: Version to restore
            
        Returns:
            True if restoration successful
        """
        try:
            versions = await self._load_versions_metadata()
            
            if version_id not in versions:
                raise LightRAGError(f"Version {version_id} not found")
            
            version_data = versions[version_id]
            backup_path = version_data.get('backup_path')
            
            if not backup_path or not Path(backup_path).exists():
                raise LightRAGError(f"Backup file for version {version_id} not found")
            
            # Create backup of current state before restore
            current_backup = await self.create_version(
                f"Pre-restore backup before restoring {version_id}"
            )
            
            # Restore from backup
            success = await self._restore_from_backup(backup_path)
            
            if success:
                self.logger.info(f"Successfully restored version {version_id}")
            else:
                self.logger.error(f"Failed to restore version {version_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to restore version {version_id}: {e}")
            raise LightRAGError(f"Version restoration failed: {e}")
    
    async def list_versions(self, limit: Optional[int] = None) -> List[Version]:
        """
        List available versions, sorted by creation date (newest first).
        
        Args:
            limit: Maximum number of versions to return
            
        Returns:
            List of Version objects
        """
        try:
            versions_data = await self._load_versions_metadata()
            
            versions = []
            for version_id, data in versions_data.items():
                version = Version(
                    version_id=version_id,
                    created_at=datetime.fromisoformat(data['created_at']),
                    description=data['description'],
                    file_count=data['file_count'],
                    size_bytes=data['size_bytes'],
                    metadata=data.get('metadata', {}),
                    backup_path=data.get('backup_path')
                )
                versions.append(version)
            
            # Sort by creation date (newest first)
            versions.sort(key=lambda v: v.created_at, reverse=True)
            
            if limit:
                versions = versions[:limit]
                
            return versions
            
        except Exception as e:
            self.logger.error(f"Failed to list versions: {e}")
            raise LightRAGError(f"Version listing failed: {e}")
    
    async def get_version(self, version_id: str) -> Optional[Version]:
        """Get details of a specific version."""
        try:
            versions_data = await self._load_versions_metadata()
            
            if version_id not in versions_data:
                return None
            
            data = versions_data[version_id]
            return Version(
                version_id=version_id,
                created_at=datetime.fromisoformat(data['created_at']),
                description=data['description'],
                file_count=data['file_count'],
                size_bytes=data['size_bytes'],
                metadata=data.get('metadata', {}),
                backup_path=data.get('backup_path')
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get version {version_id}: {e}")
            return None
    
    async def delete_version(self, version_id: str) -> bool:
        """
        Delete a specific version and its backup.
        
        Args:
            version_id: Version to delete
            
        Returns:
            True if deletion successful
        """
        try:
            versions_data = await self._load_versions_metadata()
            
            if version_id not in versions_data:
                return False
            
            version_data = versions_data[version_id]
            backup_path = version_data.get('backup_path')
            
            # Remove backup file
            if backup_path and Path(backup_path).exists():
                if HAS_AIOFILES:
                    await aiofiles.os.remove(backup_path)
                else:
                    # Fallback to synchronous file removal
                    await asyncio.get_event_loop().run_in_executor(None, os.remove, backup_path)
            
            # Remove from metadata
            del versions_data[version_id]
            await self._save_versions_metadata(versions_data)
            
            self.logger.info(f"Deleted version {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete version {version_id}: {e}")
            return False
    
    async def get_version_diff(
        self, 
        version1_id: str, 
        version2_id: str
    ) -> Dict[str, Any]:
        """
        Get differences between two versions.
        
        Args:
            version1_id: First version ID
            version2_id: Second version ID
            
        Returns:
            Dictionary with difference information
        """
        try:
            version1 = await self.get_version(version1_id)
            version2 = await self.get_version(version2_id)
            
            if not version1 or not version2:
                raise LightRAGError("One or both versions not found")
            
            diff = {
                'version1': {
                    'id': version1.version_id,
                    'created_at': version1.created_at.isoformat(),
                    'file_count': version1.file_count,
                    'size_bytes': version1.size_bytes
                },
                'version2': {
                    'id': version2.version_id,
                    'created_at': version2.created_at.isoformat(),
                    'file_count': version2.file_count,
                    'size_bytes': version2.size_bytes
                },
                'differences': {
                    'file_count_delta': version2.file_count - version1.file_count,
                    'size_delta': version2.size_bytes - version1.size_bytes,
                    'time_delta': (version2.created_at - version1.created_at).total_seconds()
                }
            }
            
            return diff
            
        except Exception as e:
            self.logger.error(f"Failed to get version diff: {e}")
            raise LightRAGError(f"Version diff failed: {e}")
    
    async def _calculate_kb_stats(self) -> Dict[str, int]:
        """Calculate knowledge base statistics."""
        try:
            file_count = 0
            size_bytes = 0
            
            if self.knowledge_base_path.exists():
                for file_path in self.knowledge_base_path.rglob("*"):
                    if file_path.is_file():
                        file_count += 1
                        try:
                            if HAS_AIOFILES:
                                stat = await aiofiles.os.stat(file_path)
                            else:
                                stat = os.stat(file_path)
                            size_bytes += stat.st_size
                        except:
                            pass  # Skip files we can't stat
            
            return {
                'file_count': file_count,
                'size_bytes': size_bytes
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate KB stats: {e}")
            return {'file_count': 0, 'size_bytes': 0}
    
    async def _create_backup(self, version_id: str) -> Path:
        """Create a backup archive of the knowledge base."""
        try:
            if self.compress_backups:
                backup_file = self.versions_path / f"{version_id}.tar.gz"
                
                # Create compressed archive
                def create_archive():
                    with tarfile.open(backup_file, "w:gz") as tar:
                        if self.knowledge_base_path.exists():
                            tar.add(self.knowledge_base_path, arcname="knowledge_base")
                
                # Run in thread pool to avoid blocking
                await asyncio.get_event_loop().run_in_executor(None, create_archive)
                
            else:
                backup_dir = self.versions_path / version_id
                
                # Copy directory structure
                if self.knowledge_base_path.exists():
                    def copy_tree():
                        shutil.copytree(self.knowledge_base_path, backup_dir, dirs_exist_ok=True)
                    
                    await asyncio.get_event_loop().run_in_executor(None, copy_tree)
                    backup_file = backup_dir
                else:
                    backup_dir.mkdir(exist_ok=True)
                    backup_file = backup_dir
            
            return backup_file
            
        except Exception as e:
            self.logger.error(f"Failed to create backup for {version_id}: {e}")
            raise
    
    async def _restore_from_backup(self, backup_path: str) -> bool:
        """Restore knowledge base from backup."""
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                return False
            
            # Remove current knowledge base
            if self.knowledge_base_path.exists():
                def remove_tree():
                    shutil.rmtree(self.knowledge_base_path)
                
                await asyncio.get_event_loop().run_in_executor(None, remove_tree)
            
            # Restore from backup
            if backup_file.suffix == '.gz':
                # Extract compressed archive
                def extract_archive():
                    with tarfile.open(backup_file, "r:gz") as tar:
                        tar.extractall(self.knowledge_base_path.parent)
                
                await asyncio.get_event_loop().run_in_executor(None, extract_archive)
                
            else:
                # Copy directory
                def copy_tree():
                    shutil.copytree(backup_file, self.knowledge_base_path, dirs_exist_ok=True)
                
                await asyncio.get_event_loop().run_in_executor(None, copy_tree)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore from backup {backup_path}: {e}")
            return False
    
    async def _load_versions_metadata(self) -> Dict[str, Any]:
        """Load versions metadata from file."""
        try:
            if self.versions_file.exists():
                if HAS_AIOFILES:
                    async with aiofiles.open(self.versions_file, 'r') as f:
                        content = await f.read()
                        return json.loads(content)
                else:
                    # Fallback to synchronous file reading
                    def read_file():
                        with open(self.versions_file, 'r') as f:
                            return json.load(f)
                    
                    return await asyncio.get_event_loop().run_in_executor(None, read_file)
            return {}
        except Exception as e:
            self.logger.warning(f"Failed to load versions metadata: {e}")
            return {}
    
    async def _save_versions_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save versions metadata to file."""
        try:
            if HAS_AIOFILES:
                async with aiofiles.open(self.versions_file, 'w') as f:
                    await f.write(json.dumps(metadata, indent=2, default=str))
            else:
                # Fallback to synchronous file writing
                def write_file():
                    with open(self.versions_file, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                
                await asyncio.get_event_loop().run_in_executor(None, write_file)
        except Exception as e:
            self.logger.error(f"Failed to save versions metadata: {e}")
            raise
    
    async def _add_version_to_metadata(self, version: Version) -> None:
        """Add a new version to the metadata."""
        try:
            metadata = await self._load_versions_metadata()
            
            metadata[version.version_id] = {
                'created_at': version.created_at.isoformat(),
                'description': version.description,
                'file_count': version.file_count,
                'size_bytes': version.size_bytes,
                'metadata': version.metadata,
                'backup_path': version.backup_path
            }
            
            await self._save_versions_metadata(metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to add version to metadata: {e}")
            raise
    
    async def _cleanup_old_versions(self) -> None:
        """Remove old versions if we exceed the maximum count."""
        try:
            versions = await self.list_versions()
            
            if len(versions) > self.max_versions:
                # Remove oldest versions
                versions_to_remove = versions[self.max_versions:]
                
                for version in versions_to_remove:
                    await self.delete_version(version.version_id)
                    self.logger.info(f"Cleaned up old version {version.version_id}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old versions: {e}")