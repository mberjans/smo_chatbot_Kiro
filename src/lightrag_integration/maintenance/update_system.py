"""
Knowledge base update system for incremental document processing.

This module handles:
- Incremental processing of new documents
- Change detection and delta updates
- Progress tracking and status reporting
- Error handling and recovery
"""

import asyncio
import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple

try:
    import aiofiles
    import aiofiles.os
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

from ..ingestion.pipeline import PDFIngestionPipeline
from ..ingestion.progress_tracker import ProgressTracker
from ..error_handling import ErrorHandler


class LightRAGError(Exception):
    """Custom exception for LightRAG operations."""
    pass
from .version_control import VersionManager, Version


@dataclass
class ProcessResult:
    """Result of document processing operation."""
    success: bool
    file_path: str
    processing_time: float = 0.0
    error: Optional[str] = None


class UpdateStatus(Enum):
    """Status of knowledge base update operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class UpdateResult:
    """Result of a knowledge base update operation."""
    update_id: str
    status: UpdateStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    documents_processed: int = 0
    documents_added: int = 0
    documents_updated: int = 0
    documents_failed: int = 0
    error_message: Optional[str] = None
    version_created: Optional[str] = None
    rollback_version: Optional[str] = None


@dataclass
class DocumentMetadata:
    """Metadata for tracking document changes."""
    file_path: str
    file_hash: str
    file_size: int
    last_modified: datetime
    processed_time: Optional[datetime] = None
    version: Optional[str] = None


class KnowledgeBaseUpdater:
    """
    Manages incremental updates to the LightRAG knowledge base.
    
    Provides functionality for:
    - Detecting new and changed documents
    - Incremental processing with version control
    - Rollback capabilities for failed updates
    - Progress tracking and status reporting
    """
    
    def __init__(
        self,
        ingestion_pipeline: PDFIngestionPipeline,
        version_manager: VersionManager,
        error_handler: ErrorHandler,
        metadata_file: str = "document_metadata.json",
        watch_directories: Optional[List[str]] = None
    ):
        self.ingestion_pipeline = ingestion_pipeline
        self.version_manager = version_manager
        self.error_handler = error_handler
        self.metadata_file = Path(metadata_file)
        self.watch_directories = watch_directories or ["papers/", "custom_papers/"]
        self.logger = logging.getLogger(__name__)
        
        # Track active updates
        self.active_updates: Dict[str, UpdateResult] = {}
        
    async def initialize(self) -> None:
        """Initialize the update system."""
        try:
            # Ensure metadata file exists
            if not self.metadata_file.exists():
                await self._save_metadata({})
                
            # Initialize version manager
            await self.version_manager.initialize()
            
            self.logger.info("Knowledge base updater initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize updater: {e}")
            raise LightRAGError(f"Updater initialization failed: {e}")
    
    async def scan_for_updates(self) -> Dict[str, List[str]]:
        """
        Scan watch directories for new or changed documents.
        
        Returns:
            Dict with 'new', 'modified', and 'deleted' document lists
        """
        try:
            current_metadata = await self._load_metadata()
            current_files = await self._scan_directories()
            
            new_files = []
            modified_files = []
            deleted_files = []
            
            # Check for new and modified files
            for file_path, file_info in current_files.items():
                if file_path not in current_metadata:
                    new_files.append(file_path)
                else:
                    stored_info = current_metadata[file_path]
                    if (file_info['file_hash'] != stored_info.get('file_hash') or
                        file_info['last_modified'] != stored_info.get('last_modified')):
                        modified_files.append(file_path)
            
            # Check for deleted files
            for file_path in current_metadata:
                if file_path not in current_files:
                    deleted_files.append(file_path)
            
            result = {
                'new': new_files,
                'modified': modified_files,
                'deleted': deleted_files
            }
            
            self.logger.info(f"Scan results: {len(new_files)} new, {len(modified_files)} modified, {len(deleted_files)} deleted")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to scan for updates: {e}")
            raise LightRAGError(f"Update scan failed: {e}")
    
    async def perform_incremental_update(
        self,
        update_id: Optional[str] = None,
        create_version: bool = True
    ) -> UpdateResult:
        """
        Perform incremental update of the knowledge base.
        
        Args:
            update_id: Optional custom update ID
            create_version: Whether to create a new version
            
        Returns:
            UpdateResult with operation details
        """
        if update_id is None:
            update_id = f"update_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        update_result = UpdateResult(
            update_id=update_id,
            status=UpdateStatus.PENDING,
            start_time=datetime.now()
        )
        
        self.active_updates[update_id] = update_result
        
        try:
            update_result.status = UpdateStatus.RUNNING
            
            # Create version snapshot if requested
            if create_version:
                version = await self.version_manager.create_version(
                    description=f"Incremental update {update_id}"
                )
                update_result.rollback_version = version.version_id
                self.logger.info(f"Created version {version.version_id} for rollback")
            
            # Scan for changes
            changes = await self.scan_for_updates()
            total_files = len(changes['new']) + len(changes['modified'])
            
            if total_files == 0:
                self.logger.info("No changes detected, update complete")
                update_result.status = UpdateStatus.COMPLETED
                update_result.end_time = datetime.now()
                return update_result
            
            # Process new and modified files
            progress_tracker = ProgressTracker(total_files)
            
            for file_path in changes['new'] + changes['modified']:
                try:
                    await self._process_single_document(file_path, update_result)
                    progress_tracker.update_progress(1)
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    update_result.documents_failed += 1
                    await self.error_handler.handle_processing_error(e, {"file_path": file_path})
            
            # Handle deleted files
            for file_path in changes['deleted']:
                await self._handle_deleted_document(file_path, update_result)
            
            # Update metadata
            await self._update_metadata_after_processing(changes, update_result)
            
            # Create new version with updates
            if create_version:
                new_version = await self.version_manager.create_version(
                    description=f"Update {update_id} completed - {update_result.documents_added} added, {update_result.documents_updated} updated"
                )
                update_result.version_created = new_version.version_id
            
            update_result.status = UpdateStatus.COMPLETED
            update_result.end_time = datetime.now()
            
            self.logger.info(f"Update {update_id} completed successfully")
            return update_result
            
        except Exception as e:
            self.logger.error(f"Update {update_id} failed: {e}")
            update_result.status = UpdateStatus.FAILED
            update_result.error_message = str(e)
            update_result.end_time = datetime.now()
            
            # Attempt rollback if version was created
            if update_result.rollback_version:
                try:
                    await self.rollback_update(update_result.rollback_version)
                    update_result.status = UpdateStatus.ROLLED_BACK
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed: {rollback_error}")
            
            raise LightRAGError(f"Update failed: {e}")
            
        finally:
            if update_id in self.active_updates:
                del self.active_updates[update_id]
    
    async def rollback_update(self, version_id: str) -> bool:
        """
        Rollback knowledge base to a previous version.
        
        Args:
            version_id: Version to rollback to
            
        Returns:
            True if rollback successful
        """
        try:
            success = await self.version_manager.restore_version(version_id)
            
            if success:
                # Reload metadata from the restored version
                await self._reload_metadata_from_version(version_id)
                self.logger.info(f"Successfully rolled back to version {version_id}")
            else:
                self.logger.error(f"Failed to rollback to version {version_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Rollback to {version_id} failed: {e}")
            raise LightRAGError(f"Rollback failed: {e}")
    
    async def get_update_status(self, update_id: str) -> Optional[UpdateResult]:
        """Get status of an active or completed update."""
        return self.active_updates.get(update_id)
    
    async def list_active_updates(self) -> List[UpdateResult]:
        """List all currently active updates."""
        return list(self.active_updates.values())
    
    async def _scan_directories(self) -> Dict[str, Dict[str, Any]]:
        """Scan watch directories for documents."""
        files_info = {}
        
        for directory in self.watch_directories:
            dir_path = Path(directory)
            if not dir_path.exists():
                continue
                
            for file_path in dir_path.rglob("*.pdf"):
                try:
                    if HAS_AIOFILES:
                        stat = await aiofiles.os.stat(file_path)
                    else:
                        stat = os.stat(file_path)
                    
                    # Calculate file hash
                    file_hash = await self._calculate_file_hash(file_path)
                    
                    files_info[str(file_path)] = {
                        'file_hash': file_hash,
                        'file_size': stat.st_size,
                        'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process file {file_path}: {e}")
                    
        return files_info
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        
        if HAS_AIOFILES:
            async with aiofiles.open(file_path, 'rb') as f:
                while chunk := await f.read(8192):
                    hash_sha256.update(chunk)
        else:
            # Fallback to synchronous file reading
            def read_file():
                with open(file_path, 'rb') as f:
                    while chunk := f.read(8192):
                        hash_sha256.update(chunk)
            
            await asyncio.get_event_loop().run_in_executor(None, read_file)
                
        return hash_sha256.hexdigest()
    
    async def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load document metadata from file."""
        try:
            if self.metadata_file.exists():
                if HAS_AIOFILES:
                    async with aiofiles.open(self.metadata_file, 'r') as f:
                        content = await f.read()
                        return json.loads(content)
                else:
                    # Fallback to synchronous file reading
                    def read_file():
                        with open(self.metadata_file, 'r') as f:
                            return json.load(f)
                    
                    return await asyncio.get_event_loop().run_in_executor(None, read_file)
            return {}
        except Exception as e:
            self.logger.warning(f"Failed to load metadata: {e}")
            return {}
    
    async def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        """Save document metadata to file."""
        try:
            if HAS_AIOFILES:
                async with aiofiles.open(self.metadata_file, 'w') as f:
                    await f.write(json.dumps(metadata, indent=2, default=str))
            else:
                # Fallback to synchronous file writing
                def write_file():
                    with open(self.metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                
                await asyncio.get_event_loop().run_in_executor(None, write_file)
        except Exception as e:
            self.logger.error(f"Failed to save metadata: {e}")
            raise
    
    async def _process_single_document(
        self, 
        file_path: str, 
        update_result: UpdateResult
    ) -> None:
        """Process a single document and update counters."""
        try:
            # Check if this is a new or updated document
            metadata = await self._load_metadata()
            is_new = file_path not in metadata
            
            # Process the document
            result_dict = await self.ingestion_pipeline.process_file(file_path)
            
            # Convert to ProcessResult
            result = ProcessResult(
                success=result_dict.get('success', False),
                file_path=file_path,
                processing_time=result_dict.get('processing_time', 0.0),
                error=result_dict.get('error_message')
            )
            
            if result.success:
                if is_new:
                    update_result.documents_added += 1
                else:
                    update_result.documents_updated += 1
                    
                update_result.documents_processed += 1
                self.logger.debug(f"Successfully processed {file_path}")
            else:
                update_result.documents_failed += 1
                self.logger.error(f"Failed to process {file_path}: {result.error}")
                
        except Exception as e:
            update_result.documents_failed += 1
            raise
    
    async def _handle_deleted_document(
        self, 
        file_path: str, 
        update_result: UpdateResult
    ) -> None:
        """Handle removal of deleted documents from knowledge base."""
        try:
            # Remove document from knowledge base
            # This would depend on the specific implementation of the knowledge base
            # For now, we'll just log it
            self.logger.info(f"Document deleted: {file_path}")
            
            # In a real implementation, you would:
            # 1. Remove document nodes from the graph
            # 2. Remove associated embeddings
            # 3. Update indexes
            
        except Exception as e:
            self.logger.error(f"Failed to handle deleted document {file_path}: {e}")
    
    async def _update_metadata_after_processing(
        self, 
        changes: Dict[str, List[str]], 
        update_result: UpdateResult
    ) -> None:
        """Update metadata file after processing changes."""
        try:
            metadata = await self._load_metadata()
            current_files = await self._scan_directories()
            
            # Update metadata for processed files
            for file_path in changes['new'] + changes['modified']:
                if file_path in current_files:
                    file_info = current_files[file_path]
                    metadata[file_path] = {
                        **file_info,
                        'processed_time': datetime.now().isoformat(),
                        'version': update_result.version_created
                    }
            
            # Remove metadata for deleted files
            for file_path in changes['deleted']:
                metadata.pop(file_path, None)
            
            await self._save_metadata(metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to update metadata: {e}")
            raise
    
    async def _reload_metadata_from_version(self, version_id: str) -> None:
        """Reload metadata from a specific version after rollback."""
        try:
            # This would restore metadata from the version
            # For now, we'll rescan the directories
            current_files = await self._scan_directories()
            
            # Create basic metadata
            metadata = {}
            for file_path, file_info in current_files.items():
                metadata[file_path] = {
                    **file_info,
                    'processed_time': datetime.now().isoformat(),
                    'version': version_id
                }
            
            await self._save_metadata(metadata)
            
        except Exception as e:
            self.logger.error(f"Failed to reload metadata from version {version_id}: {e}")
            raise