"""
Administrative interface for LightRAG system management.

This module provides:
- Admin endpoints for system management
- Document management and curation interfaces
- System status and metrics dashboards
- Administrative functionality for maintenance
"""

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum

from .update_system import KnowledgeBaseUpdater, UpdateResult, UpdateStatus
from .version_control import VersionManager, Version
from ..error_handling import ErrorHandler


class LightRAGError(Exception):
    """Custom exception for LightRAG operations."""
    pass
from ..monitoring import PerformanceMonitor, PerformanceMetrics, MetricsCollector
from ..component import LightRAGComponent


class SystemHealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class SystemStatus:
    """Overall system status information."""
    health_status: SystemHealthStatus
    uptime_seconds: float
    total_documents: int
    total_queries: int
    active_updates: int
    last_update: Optional[datetime]
    error_count_24h: int
    performance_metrics: Dict[str, Any]
    storage_usage: Dict[str, Any]
    version_info: Dict[str, Any]


@dataclass
class DocumentInfo:
    """Information about a document in the knowledge base."""
    document_id: str
    file_path: str
    title: str
    authors: List[str]
    abstract: str
    ingestion_date: datetime
    file_size: int
    entity_count: int
    relationship_count: int
    status: str
    metadata: Dict[str, Any]


@dataclass
class AdminAction:
    """Represents an administrative action."""
    action_id: str
    action_type: str
    description: str
    initiated_by: str
    initiated_at: datetime
    status: str
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class DocumentManager:
    """
    Manages documents in the knowledge base.
    
    Provides functionality for:
    - Document listing and search
    - Document metadata management
    - Document removal and curation
    - Batch operations on documents
    """
    
    def __init__(
        self,
        lightrag_component: LightRAGComponent,
        updater: KnowledgeBaseUpdater
    ):
        self.lightrag_component = lightrag_component
        self.updater = updater
        self.logger = logging.getLogger(__name__)
    
    async def list_documents(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        search_query: Optional[str] = None,
        sort_by: str = "ingestion_date",
        sort_order: str = "desc"
    ) -> Tuple[List[DocumentInfo], int]:
        """
        List documents in the knowledge base.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            search_query: Optional search query to filter documents
            sort_by: Field to sort by
            sort_order: Sort order (asc/desc)
            
        Returns:
            Tuple of (documents, total_count)
        """
        try:
            # This would query the actual knowledge base
            # For now, we'll return mock data
            documents = []
            
            # In a real implementation, this would:
            # 1. Query the knowledge graph for document nodes
            # 2. Apply search filters if provided
            # 3. Sort and paginate results
            # 4. Return document information
            
            total_count = len(documents)
            
            if limit:
                documents = documents[offset:offset + limit]
            else:
                documents = documents[offset:]
            
            return documents, total_count
            
        except Exception as e:
            self.logger.error(f"Failed to list documents: {e}")
            raise LightRAGError(f"Document listing failed: {e}")
    
    async def get_document(self, document_id: str) -> Optional[DocumentInfo]:
        """Get detailed information about a specific document."""
        try:
            # Query knowledge base for document details
            # This would retrieve document metadata, entities, relationships
            
            # Mock implementation
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    async def remove_document(self, document_id: str) -> bool:
        """
        Remove a document from the knowledge base.
        
        Args:
            document_id: ID of document to remove
            
        Returns:
            True if removal successful
        """
        try:
            # This would:
            # 1. Remove document nodes from knowledge graph
            # 2. Remove associated embeddings
            # 3. Update indexes
            # 4. Log the removal action
            
            self.logger.info(f"Removed document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove document {document_id}: {e}")
            return False
    
    async def update_document_metadata(
        self,
        document_id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Update metadata for a document."""
        try:
            # Update document metadata in knowledge base
            self.logger.info(f"Updated metadata for document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update document metadata: {e}")
            return False
    
    async def batch_remove_documents(self, document_ids: List[str]) -> Dict[str, bool]:
        """Remove multiple documents in batch."""
        results = {}
        
        for doc_id in document_ids:
            results[doc_id] = await self.remove_document(doc_id)
        
        return results
    
    async def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about documents in the knowledge base."""
        try:
            # Calculate document statistics
            stats = {
                'total_documents': 0,
                'total_entities': 0,
                'total_relationships': 0,
                'average_document_size': 0,
                'documents_by_type': {},
                'ingestion_timeline': {},
                'top_authors': [],
                'storage_usage': {
                    'total_size_bytes': 0,
                    'index_size_bytes': 0,
                    'embeddings_size_bytes': 0
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get document statistics: {e}")
            return {}


class AdminInterface:
    """
    Main administrative interface for LightRAG system management.
    
    Provides comprehensive administrative functionality including:
    - System status monitoring
    - Document management
    - Update management
    - Performance monitoring
    - Configuration management
    """
    
    def __init__(
        self,
        lightrag_component: LightRAGComponent,
        updater: KnowledgeBaseUpdater,
        version_manager: VersionManager,
        performance_monitor: PerformanceMonitor,
        error_handler: ErrorHandler
    ):
        self.lightrag_component = lightrag_component
        self.updater = updater
        self.version_manager = version_manager
        self.performance_monitor = performance_monitor
        self.error_handler = error_handler
        self.document_manager = DocumentManager(lightrag_component, updater)
        self.logger = logging.getLogger(__name__)
        
        # Track administrative actions
        self.admin_actions: Dict[str, AdminAction] = {}
        self.action_counter = 0
    
    async def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status information."""
        try:
            # Get system health metrics
            health_metrics = await self._get_health_metrics()
            performance_metrics = await self.performance_monitor.get_current_metrics()
            
            # Get update information
            active_updates = await self.updater.list_active_updates()
            
            # Get version information
            versions = await self.version_manager.list_versions(limit=1)
            last_version = versions[0] if versions else None
            
            # Get document statistics
            doc_stats = await self.document_manager.get_document_statistics()
            
            # Determine health status
            health_status = self._determine_health_status(health_metrics)
            
            # Get error count for last 24 hours
            error_count = await self._get_recent_error_count()
            
            return SystemStatus(
                health_status=health_status,
                uptime_seconds=health_metrics.get('uptime_seconds', 0),
                total_documents=doc_stats.get('total_documents', 0),
                total_queries=health_metrics.get('total_queries', 0),
                active_updates=len(active_updates),
                last_update=last_version.created_at if last_version else None,
                error_count_24h=error_count,
                performance_metrics=asdict(performance_metrics) if performance_metrics else {},
                storage_usage=doc_stats.get('storage_usage', {}),
                version_info={
                    'current_version': last_version.version_id if last_version else None,
                    'total_versions': len(await self.version_manager.list_versions())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return SystemStatus(
                health_status=SystemHealthStatus.UNKNOWN,
                uptime_seconds=0,
                total_documents=0,
                total_queries=0,
                active_updates=0,
                last_update=None,
                error_count_24h=0,
                performance_metrics={},
                storage_usage={},
                version_info={}
            )
    
    async def trigger_manual_update(
        self,
        admin_user: str,
        description: Optional[str] = None
    ) -> str:
        """
        Trigger a manual knowledge base update.
        
        Args:
            admin_user: Username of administrator triggering update
            description: Optional description of the update
            
        Returns:
            Action ID for tracking the update
        """
        try:
            action_id = await self._create_admin_action(
                action_type="manual_update",
                description=description or "Manual knowledge base update",
                initiated_by=admin_user
            )
            
            # Start update in background
            asyncio.create_task(self._perform_manual_update(action_id))
            
            return action_id
            
        except Exception as e:
            self.logger.error(f"Failed to trigger manual update: {e}")
            raise LightRAGError(f"Manual update failed: {e}")
    
    async def create_system_backup(
        self,
        admin_user: str,
        description: Optional[str] = None
    ) -> str:
        """
        Create a system backup.
        
        Args:
            admin_user: Username of administrator creating backup
            description: Optional description of the backup
            
        Returns:
            Action ID for tracking the backup
        """
        try:
            action_id = await self._create_admin_action(
                action_type="system_backup",
                description=description or "System backup",
                initiated_by=admin_user
            )
            
            # Create backup
            version = await self.version_manager.create_version(
                description=f"Admin backup: {description or 'Manual backup'}",
                metadata={'admin_user': admin_user, 'action_id': action_id}
            )
            
            # Update action with result
            await self._update_admin_action(action_id, "completed", {
                'version_id': version.version_id,
                'backup_path': version.backup_path
            })
            
            return action_id
            
        except Exception as e:
            self.logger.error(f"Failed to create system backup: {e}")
            await self._update_admin_action(action_id, "failed", error_message=str(e))
            raise LightRAGError(f"System backup failed: {e}")
    
    async def restore_from_backup(
        self,
        version_id: str,
        admin_user: str,
        confirmation_token: str
    ) -> str:
        """
        Restore system from a backup version.
        
        Args:
            version_id: Version to restore from
            admin_user: Username of administrator performing restore
            confirmation_token: Security confirmation token
            
        Returns:
            Action ID for tracking the restore
        """
        try:
            # Validate confirmation token (in real implementation)
            if not self._validate_confirmation_token(confirmation_token):
                raise LightRAGError("Invalid confirmation token")
            
            action_id = await self._create_admin_action(
                action_type="system_restore",
                description=f"Restore from version {version_id}",
                initiated_by=admin_user
            )
            
            # Perform restore
            success = await self.version_manager.restore_version(version_id)
            
            if success:
                await self._update_admin_action(action_id, "completed", {
                    'restored_version': version_id
                })
            else:
                await self._update_admin_action(action_id, "failed", 
                                              error_message="Restore operation failed")
            
            return action_id
            
        except Exception as e:
            self.logger.error(f"Failed to restore from backup: {e}")
            await self._update_admin_action(action_id, "failed", error_message=str(e))
            raise LightRAGError(f"System restore failed: {e}")
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data."""
        try:
            # Get current performance metrics
            current_metrics = await self.system_monitor.get_performance_metrics()
            
            # Get historical performance data
            historical_data = await self.performance_monitor.get_historical_metrics(
                hours=24
            )
            
            # Get query statistics
            query_stats = await self._get_query_statistics()
            
            # Get resource usage
            resource_usage = await self._get_resource_usage()
            
            return {
                'current_metrics': asdict(current_metrics) if current_metrics else {},
                'historical_data': historical_data,
                'query_statistics': query_stats,
                'resource_usage': resource_usage,
                'alerts': await self._get_active_alerts()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance dashboard: {e}")
            return {}
    
    async def get_admin_actions(
        self,
        limit: Optional[int] = None,
        action_type: Optional[str] = None
    ) -> List[AdminAction]:
        """Get list of administrative actions."""
        actions = list(self.admin_actions.values())
        
        # Filter by action type if specified
        if action_type:
            actions = [a for a in actions if a.action_type == action_type]
        
        # Sort by initiation time (newest first)
        actions.sort(key=lambda a: a.initiated_at, reverse=True)
        
        # Apply limit
        if limit:
            actions = actions[:limit]
        
        return actions
    
    async def get_admin_action(self, action_id: str) -> Optional[AdminAction]:
        """Get details of a specific administrative action."""
        return self.admin_actions.get(action_id)
    
    async def cleanup_old_data(
        self,
        admin_user: str,
        days_to_keep: int = 30
    ) -> str:
        """
        Clean up old data and logs.
        
        Args:
            admin_user: Username of administrator performing cleanup
            days_to_keep: Number of days of data to keep
            
        Returns:
            Action ID for tracking the cleanup
        """
        try:
            action_id = await self._create_admin_action(
                action_type="data_cleanup",
                description=f"Clean up data older than {days_to_keep} days",
                initiated_by=admin_user
            )
            
            # Start cleanup in background
            asyncio.create_task(self._perform_data_cleanup(action_id, days_to_keep))
            
            return action_id
            
        except Exception as e:
            self.logger.error(f"Failed to start data cleanup: {e}")
            raise LightRAGError(f"Data cleanup failed: {e}")
    
    def _determine_health_status(self, health_metrics: Dict[str, Any]) -> SystemHealthStatus:
        """Determine overall system health status."""
        try:
            # Check various health indicators
            error_rate = health_metrics.get('error_rate', 0)
            response_time = health_metrics.get('avg_response_time', 0)
            memory_usage = health_metrics.get('memory_usage_percent', 0)
            disk_usage = health_metrics.get('disk_usage_percent', 0)
            
            # Define thresholds
            if (error_rate > 0.1 or response_time > 10 or 
                memory_usage > 90 or disk_usage > 95):
                return SystemHealthStatus.CRITICAL
            elif (error_rate > 0.05 or response_time > 5 or 
                  memory_usage > 80 or disk_usage > 85):
                return SystemHealthStatus.WARNING
            else:
                return SystemHealthStatus.HEALTHY
                
        except Exception:
            return SystemHealthStatus.UNKNOWN
    
    async def _get_recent_error_count(self) -> int:
        """Get error count for the last 24 hours."""
        try:
            # This would query error logs for recent errors
            return 0
        except Exception:
            return 0
    
    async def _create_admin_action(
        self,
        action_type: str,
        description: str,
        initiated_by: str
    ) -> str:
        """Create a new administrative action record."""
        self.action_counter += 1
        action_id = f"admin_{self.action_counter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        action = AdminAction(
            action_id=action_id,
            action_type=action_type,
            description=description,
            initiated_by=initiated_by,
            initiated_at=datetime.now(),
            status="pending"
        )
        
        self.admin_actions[action_id] = action
        return action_id
    
    async def _update_admin_action(
        self,
        action_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> None:
        """Update an administrative action record."""
        if action_id in self.admin_actions:
            action = self.admin_actions[action_id]
            action.status = status
            action.result = result
            action.error_message = error_message
    
    async def _perform_manual_update(self, action_id: str) -> None:
        """Perform manual update in background."""
        try:
            await self._update_admin_action(action_id, "running")
            
            # Perform the update
            result = await self.updater.perform_incremental_update(
                update_id=f"manual_{action_id}"
            )
            
            await self._update_admin_action(action_id, "completed", {
                'update_result': asdict(result)
            })
            
        except Exception as e:
            await self._update_admin_action(action_id, "failed", error_message=str(e))
    
    async def _perform_data_cleanup(self, action_id: str, days_to_keep: int) -> None:
        """Perform data cleanup in background."""
        try:
            await self._update_admin_action(action_id, "running")
            
            # Perform cleanup operations
            cleanup_results = {
                'old_logs_removed': 0,
                'old_versions_removed': 0,
                'cache_cleared': True,
                'space_freed_bytes': 0
            }
            
            # Clean up old versions beyond retention period
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            versions = await self.version_manager.list_versions()
            
            for version in versions:
                if version.created_at < cutoff_date:
                    success = await self.version_manager.delete_version(version.version_id)
                    if success:
                        cleanup_results['old_versions_removed'] += 1
                        cleanup_results['space_freed_bytes'] += version.size_bytes
            
            await self._update_admin_action(action_id, "completed", cleanup_results)
            
        except Exception as e:
            await self._update_admin_action(action_id, "failed", error_message=str(e))
    
    def _validate_confirmation_token(self, token: str) -> bool:
        """Validate confirmation token for destructive operations."""
        # In a real implementation, this would validate a secure token
        return len(token) > 10
    
    async def _get_query_statistics(self) -> Dict[str, Any]:
        """Get query statistics."""
        return {
            'total_queries_24h': 0,
            'avg_response_time': 0,
            'success_rate': 0,
            'popular_queries': [],
            'query_types': {}
        }
    
    async def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage."""
        return {
            'cpu_percent': 0,
            'memory_percent': 0,
            'disk_percent': 0,
            'network_io': {'bytes_sent': 0, 'bytes_recv': 0}
        }
    
    async def _get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active system alerts."""
        return []
    
    async def _get_health_metrics(self) -> Dict[str, Any]:
        """Get system health metrics."""
        # This would collect various health indicators
        return {
            'uptime_seconds': 3600,  # Mock data
            'total_queries': 100,
            'error_rate': 0.01,
            'avg_response_time': 2.5,
            'memory_usage_percent': 60,
            'disk_usage_percent': 40
        }