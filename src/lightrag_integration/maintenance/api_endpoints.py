"""
REST API endpoints for LightRAG administrative interface.

This module provides:
- REST API endpoints for system management
- Document management API endpoints
- System status and metrics API endpoints
- Administrative action endpoints
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel, Field

from .admin_interface import AdminInterface, SystemStatus, DocumentInfo, AdminAction
from .update_system import KnowledgeBaseUpdater, UpdateResult
from .version_control import VersionManager, Version
from ..component import LightRAGComponent
from ..monitoring import PerformanceMonitor
from ..error_handling import ErrorHandler


# Pydantic models for API requests/responses
class SystemStatusResponse(BaseModel):
    """System status response model."""
    health_status: str
    uptime_seconds: float
    total_documents: int
    total_queries: int
    active_updates: int
    last_update: Optional[datetime]
    error_count_24h: int
    performance_metrics: Dict[str, Any]
    storage_usage: Dict[str, Any]
    version_info: Dict[str, Any]


class DocumentInfoResponse(BaseModel):
    """Document information response model."""
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


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: List[DocumentInfoResponse]
    total_count: int
    page: int
    page_size: int


class UpdateRequest(BaseModel):
    """Manual update request model."""
    description: Optional[str] = Field(None, description="Description of the update")


class BackupRequest(BaseModel):
    """System backup request model."""
    description: Optional[str] = Field(None, description="Description of the backup")


class RestoreRequest(BaseModel):
    """System restore request model."""
    version_id: str = Field(..., description="Version ID to restore from")
    confirmation_token: str = Field(..., description="Security confirmation token")


class AdminActionResponse(BaseModel):
    """Administrative action response model."""
    action_id: str
    action_type: str
    description: str
    initiated_by: str
    initiated_at: datetime
    status: str
    result: Optional[Dict[str, Any]]
    error_message: Optional[str]


class VersionResponse(BaseModel):
    """Version response model."""
    version_id: str
    created_at: datetime
    description: str
    file_count: int
    size_bytes: int
    metadata: Dict[str, Any]
    backup_path: Optional[str]


class PerformanceDashboardResponse(BaseModel):
    """Performance dashboard response model."""
    current_metrics: Dict[str, Any]
    historical_data: List[Dict[str, Any]]
    query_statistics: Dict[str, Any]
    resource_usage: Dict[str, Any]
    alerts: List[Dict[str, Any]]


class CleanupRequest(BaseModel):
    """Data cleanup request model."""
    days_to_keep: int = Field(30, description="Number of days of data to keep")


# Dependency to get admin interface
async def get_admin_interface() -> AdminInterface:
    """Dependency to get admin interface instance."""
    # This would be injected from the main application
    # For now, we'll raise an error to indicate it needs to be configured
    raise HTTPException(
        status_code=500,
        detail="Admin interface not configured. This endpoint needs to be properly initialized."
    )


# Create router
router = APIRouter(prefix="/admin", tags=["administration"])


@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> SystemStatusResponse:
    """Get comprehensive system status information."""
    try:
        status = await admin_interface.get_system_status()
        
        return SystemStatusResponse(
            health_status=status.health_status.value,
            uptime_seconds=status.uptime_seconds,
            total_documents=status.total_documents,
            total_queries=status.total_queries,
            active_updates=status.active_updates,
            last_update=status.last_update,
            error_count_24h=status.error_count_24h,
            performance_metrics=status.performance_metrics,
            storage_usage=status.storage_usage,
            version_info=status.version_info
        )
        
    except Exception as e:
        logging.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Number of documents per page"),
    search: Optional[str] = Query(None, description="Search query"),
    sort_by: str = Query("ingestion_date", description="Field to sort by"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$", description="Sort order"),
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> DocumentListResponse:
    """List documents with pagination and search."""
    try:
        offset = (page - 1) * page_size
        
        documents, total_count = await admin_interface.document_manager.list_documents(
            limit=page_size,
            offset=offset,
            search_query=search,
            sort_by=sort_by,
            sort_order=sort_order
        )
        
        document_responses = [
            DocumentInfoResponse(
                document_id=doc.document_id,
                file_path=doc.file_path,
                title=doc.title,
                authors=doc.authors,
                abstract=doc.abstract,
                ingestion_date=doc.ingestion_date,
                file_size=doc.file_size,
                entity_count=doc.entity_count,
                relationship_count=doc.relationship_count,
                status=doc.status,
                metadata=doc.metadata
            )
            for doc in documents
        ]
        
        return DocumentListResponse(
            documents=document_responses,
            total_count=total_count,
            page=page,
            page_size=page_size
        )
        
    except Exception as e:
        logging.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}", response_model=DocumentInfoResponse)
async def get_document(
    document_id: str,
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> DocumentInfoResponse:
    """Get detailed information about a specific document."""
    try:
        document = await admin_interface.document_manager.get_document(document_id)
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentInfoResponse(
            document_id=document.document_id,
            file_path=document.file_path,
            title=document.title,
            authors=document.authors,
            abstract=document.abstract,
            ingestion_date=document.ingestion_date,
            file_size=document.file_size,
            entity_count=document.entity_count,
            relationship_count=document.relationship_count,
            status=document.status,
            metadata=document.metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def remove_document(
    document_id: str,
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> Dict[str, str]:
    """Remove a document from the knowledge base."""
    try:
        success = await admin_interface.document_manager.remove_document(document_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or removal failed")
        
        return {"message": f"Document {document_id} removed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to remove document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/{document_id}/metadata")
async def update_document_metadata(
    document_id: str,
    metadata: Dict[str, Any],
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> Dict[str, str]:
    """Update metadata for a document."""
    try:
        success = await admin_interface.document_manager.update_document_metadata(
            document_id, metadata
        )
        
        if not success:
            raise HTTPException(status_code=404, detail="Document not found or update failed")
        
        return {"message": f"Document {document_id} metadata updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to update document metadata: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/update", response_model=AdminActionResponse)
async def trigger_manual_update(
    request: UpdateRequest,
    background_tasks: BackgroundTasks,
    admin_user: str = Query(..., description="Administrator username"),
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> AdminActionResponse:
    """Trigger a manual knowledge base update."""
    try:
        action_id = await admin_interface.trigger_manual_update(
            admin_user=admin_user,
            description=request.description
        )
        
        # Get the action details
        action = await admin_interface.get_admin_action(action_id)
        
        if not action:
            raise HTTPException(status_code=500, detail="Failed to create update action")
        
        return AdminActionResponse(
            action_id=action.action_id,
            action_type=action.action_type,
            description=action.description,
            initiated_by=action.initiated_by,
            initiated_at=action.initiated_at,
            status=action.status,
            result=action.result,
            error_message=action.error_message
        )
        
    except Exception as e:
        logging.error(f"Failed to trigger manual update: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backup", response_model=AdminActionResponse)
async def create_system_backup(
    request: BackupRequest,
    admin_user: str = Query(..., description="Administrator username"),
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> AdminActionResponse:
    """Create a system backup."""
    try:
        action_id = await admin_interface.create_system_backup(
            admin_user=admin_user,
            description=request.description
        )
        
        # Get the action details
        action = await admin_interface.get_admin_action(action_id)
        
        if not action:
            raise HTTPException(status_code=500, detail="Failed to create backup action")
        
        return AdminActionResponse(
            action_id=action.action_id,
            action_type=action.action_type,
            description=action.description,
            initiated_by=action.initiated_by,
            initiated_at=action.initiated_at,
            status=action.status,
            result=action.result,
            error_message=action.error_message
        )
        
    except Exception as e:
        logging.error(f"Failed to create system backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore", response_model=AdminActionResponse)
async def restore_from_backup(
    request: RestoreRequest,
    admin_user: str = Query(..., description="Administrator username"),
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> AdminActionResponse:
    """Restore system from a backup version."""
    try:
        action_id = await admin_interface.restore_from_backup(
            version_id=request.version_id,
            admin_user=admin_user,
            confirmation_token=request.confirmation_token
        )
        
        # Get the action details
        action = await admin_interface.get_admin_action(action_id)
        
        if not action:
            raise HTTPException(status_code=500, detail="Failed to create restore action")
        
        return AdminActionResponse(
            action_id=action.action_id,
            action_type=action.action_type,
            description=action.description,
            initiated_by=action.initiated_by,
            initiated_at=action.initiated_at,
            status=action.status,
            result=action.result,
            error_message=action.error_message
        )
        
    except Exception as e:
        logging.error(f"Failed to restore from backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/versions", response_model=List[VersionResponse])
async def list_versions(
    limit: Optional[int] = Query(None, ge=1, description="Maximum number of versions to return"),
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> List[VersionResponse]:
    """List available versions."""
    try:
        versions = await admin_interface.version_manager.list_versions(limit=limit)
        
        return [
            VersionResponse(
                version_id=version.version_id,
                created_at=version.created_at,
                description=version.description,
                file_count=version.file_count,
                size_bytes=version.size_bytes,
                metadata=version.metadata,
                backup_path=version.backup_path
            )
            for version in versions
        ]
        
    except Exception as e:
        logging.error(f"Failed to list versions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/versions/{version_id}", response_model=VersionResponse)
async def get_version(
    version_id: str,
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> VersionResponse:
    """Get details of a specific version."""
    try:
        version = await admin_interface.version_manager.get_version(version_id)
        
        if not version:
            raise HTTPException(status_code=404, detail="Version not found")
        
        return VersionResponse(
            version_id=version.version_id,
            created_at=version.created_at,
            description=version.description,
            file_count=version.file_count,
            size_bytes=version.size_bytes,
            metadata=version.metadata,
            backup_path=version.backup_path
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get version {version_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/versions/{version_id}")
async def delete_version(
    version_id: str,
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> Dict[str, str]:
    """Delete a specific version."""
    try:
        success = await admin_interface.version_manager.delete_version(version_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Version not found or deletion failed")
        
        return {"message": f"Version {version_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to delete version {version_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance", response_model=PerformanceDashboardResponse)
async def get_performance_dashboard(
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> PerformanceDashboardResponse:
    """Get performance dashboard data."""
    try:
        dashboard_data = await admin_interface.get_performance_dashboard()
        
        return PerformanceDashboardResponse(
            current_metrics=dashboard_data.get('current_metrics', {}),
            historical_data=dashboard_data.get('historical_data', []),
            query_statistics=dashboard_data.get('query_statistics', {}),
            resource_usage=dashboard_data.get('resource_usage', {}),
            alerts=dashboard_data.get('alerts', [])
        )
        
    except Exception as e:
        logging.error(f"Failed to get performance dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/actions", response_model=List[AdminActionResponse])
async def list_admin_actions(
    limit: Optional[int] = Query(None, ge=1, description="Maximum number of actions to return"),
    action_type: Optional[str] = Query(None, description="Filter by action type"),
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> List[AdminActionResponse]:
    """Get list of administrative actions."""
    try:
        actions = await admin_interface.get_admin_actions(
            limit=limit,
            action_type=action_type
        )
        
        return [
            AdminActionResponse(
                action_id=action.action_id,
                action_type=action.action_type,
                description=action.description,
                initiated_by=action.initiated_by,
                initiated_at=action.initiated_at,
                status=action.status,
                result=action.result,
                error_message=action.error_message
            )
            for action in actions
        ]
        
    except Exception as e:
        logging.error(f"Failed to list admin actions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/actions/{action_id}", response_model=AdminActionResponse)
async def get_admin_action(
    action_id: str,
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> AdminActionResponse:
    """Get details of a specific administrative action."""
    try:
        action = await admin_interface.get_admin_action(action_id)
        
        if not action:
            raise HTTPException(status_code=404, detail="Action not found")
        
        return AdminActionResponse(
            action_id=action.action_id,
            action_type=action.action_type,
            description=action.description,
            initiated_by=action.initiated_by,
            initiated_at=action.initiated_at,
            status=action.status,
            result=action.result,
            error_message=action.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to get admin action {action_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cleanup", response_model=AdminActionResponse)
async def cleanup_old_data(
    request: CleanupRequest,
    admin_user: str = Query(..., description="Administrator username"),
    admin_interface: AdminInterface = Depends(get_admin_interface)
) -> AdminActionResponse:
    """Clean up old data and logs."""
    try:
        action_id = await admin_interface.cleanup_old_data(
            admin_user=admin_user,
            days_to_keep=request.days_to_keep
        )
        
        # Get the action details
        action = await admin_interface.get_admin_action(action_id)
        
        if not action:
            raise HTTPException(status_code=500, detail="Failed to create cleanup action")
        
        return AdminActionResponse(
            action_id=action.action_id,
            action_type=action.action_type,
            description=action.description,
            initiated_by=action.initiated_by,
            initiated_at=action.initiated_at,
            status=action.status,
            result=action.result,
            error_message=action.error_message
        )
        
    except Exception as e:
        logging.error(f"Failed to cleanup old data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}