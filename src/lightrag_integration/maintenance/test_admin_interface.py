"""
Tests for the administrative interface.

Tests cover:
- System status monitoring
- Document management operations
- Administrative actions
- Performance dashboard functionality
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from .admin_interface import (
    AdminInterface,
    DocumentManager,
    SystemStatus,
    SystemHealthStatus,
    DocumentInfo,
    AdminAction
)
from .update_system import KnowledgeBaseUpdater, UpdateResult, UpdateStatus
from .version_control import VersionManager, Version
from ..component import LightRAGComponent
from ..monitoring import PerformanceMonitor, PerformanceMetrics
from ..error_handling import ErrorHandler


class TestDocumentManager:
    """Test suite for DocumentManager."""
    
    @pytest.fixture
    def mock_lightrag_component(self):
        """Mock LightRAG component."""
        return AsyncMock(spec=LightRAGComponent)
    
    @pytest.fixture
    def mock_updater(self):
        """Mock knowledge base updater."""
        return AsyncMock(spec=KnowledgeBaseUpdater)
    
    @pytest.fixture
    def document_manager(self, mock_lightrag_component, mock_updater):
        """Create DocumentManager instance for testing."""
        return DocumentManager(mock_lightrag_component, mock_updater)
    
    @pytest.mark.asyncio
    async def test_list_documents_basic(self, document_manager):
        """Test basic document listing."""
        documents, total_count = await document_manager.list_documents()
        
        assert isinstance(documents, list)
        assert isinstance(total_count, int)
        assert total_count >= 0
    
    @pytest.mark.asyncio
    async def test_list_documents_with_pagination(self, document_manager):
        """Test document listing with pagination."""
        documents, total_count = await document_manager.list_documents(
            limit=10, offset=5
        )
        
        assert isinstance(documents, list)
        assert len(documents) <= 10
    
    @pytest.mark.asyncio
    async def test_get_document_nonexistent(self, document_manager):
        """Test getting a non-existent document."""
        document = await document_manager.get_document("nonexistent_id")
        assert document is None
    
    async def test_remove_document_success(self, document_manager):
        """Test successful document removal."""
        success = await document_manager.remove_document("test_doc_id")
        assert success is True
    
    async def test_update_document_metadata(self, document_manager):
        """Test updating document metadata."""
        metadata = {"key": "value", "updated": True}
        success = await document_manager.update_document_metadata("test_doc", metadata)
        assert success is True
    
    async def test_batch_remove_documents(self, document_manager):
        """Test batch document removal."""
        doc_ids = ["doc1", "doc2", "doc3"]
        results = await document_manager.batch_remove_documents(doc_ids)
        
        assert isinstance(results, dict)
        assert len(results) == 3
        for doc_id in doc_ids:
            assert doc_id in results
            assert isinstance(results[doc_id], bool)
    
    async def test_get_document_statistics(self, document_manager):
        """Test getting document statistics."""
        stats = await document_manager.get_document_statistics()
        
        assert isinstance(stats, dict)
        expected_keys = [
            'total_documents', 'total_entities', 'total_relationships',
            'storage_usage'
        ]
        for key in expected_keys:
            assert key in stats


class TestAdminInterface:
    """Test suite for AdminInterface."""
    
    @pytest.fixture
    def mock_lightrag_component(self):
        """Mock LightRAG component."""
        return AsyncMock(spec=LightRAGComponent)
    
    @pytest.fixture
    def mock_updater(self):
        """Mock knowledge base updater."""
        updater = AsyncMock(spec=KnowledgeBaseUpdater)
        updater.list_active_updates.return_value = []
        updater.perform_incremental_update.return_value = UpdateResult(
            update_id="test_update",
            status=UpdateStatus.COMPLETED,
            start_time=datetime.now(),
            end_time=datetime.now(),
            documents_processed=5
        )
        return updater
    
    @pytest.fixture
    def mock_version_manager(self):
        """Mock version manager."""
        manager = AsyncMock(spec=VersionManager)
        manager.list_versions.return_value = [
            Version(
                version_id="v20240101_120000",
                created_at=datetime.now(),
                description="Test version",
                file_count=10,
                size_bytes=1000,
                metadata={}
            )
        ]
        manager.create_version.return_value = Version(
            version_id="v20240101_130000",
            created_at=datetime.now(),
            description="New version",
            file_count=12,
            size_bytes=1200,
            metadata={},
            backup_path="/path/to/backup"
        )
        manager.restore_version.return_value = True
        return manager
    
    @pytest.fixture
    def mock_performance_monitor(self):
        """Mock performance monitor."""
        monitor = AsyncMock(spec=PerformanceMonitor)
        monitor.get_current_metrics.return_value = PerformanceMetrics(
            operation_name="system",
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration_seconds=2.5,
            success=True,
            memory_usage_mb=512,
            cpu_usage_percent=25.0,
            custom_metrics={}
        )
        monitor.get_historical_metrics.return_value = []
        return monitor
    
    @pytest.fixture
    def mock_error_handler(self):
        """Mock error handler."""
        return AsyncMock(spec=ErrorHandler)
    
    @pytest.fixture
    def admin_interface(
        self,
        mock_lightrag_component,
        mock_updater,
        mock_version_manager,
        mock_performance_monitor,
        mock_error_handler
    ):
        """Create AdminInterface instance for testing."""
        return AdminInterface(
            lightrag_component=mock_lightrag_component,
            updater=mock_updater,
            version_manager=mock_version_manager,
            performance_monitor=mock_performance_monitor,
            error_handler=mock_error_handler
        )
    
    async def test_get_system_status(self, admin_interface):
        """Test getting system status."""
        status = await admin_interface.get_system_status()
        
        assert isinstance(status, SystemStatus)
        assert status.health_status in [
            SystemHealthStatus.HEALTHY,
            SystemHealthStatus.WARNING,
            SystemHealthStatus.CRITICAL,
            SystemHealthStatus.UNKNOWN
        ]
        assert status.uptime_seconds >= 0
        assert status.total_documents >= 0
        assert status.active_updates >= 0
    
    async def test_get_system_status_healthy(self, admin_interface):
        """Test system status determination - healthy case."""
        status = await admin_interface.get_system_status()
        
        # With the mock metrics, system should be healthy
        assert status.health_status == SystemHealthStatus.HEALTHY
        assert status.uptime_seconds == 3600
        assert status.total_queries == 100
    
    async def test_trigger_manual_update(self, admin_interface):
        """Test triggering a manual update."""
        action_id = await admin_interface.trigger_manual_update(
            admin_user="test_admin",
            description="Test manual update"
        )
        
        assert isinstance(action_id, str)
        assert action_id.startswith("admin_")
        
        # Check that action was recorded
        action = await admin_interface.get_admin_action(action_id)
        assert action is not None
        assert action.action_type == "manual_update"
        assert action.initiated_by == "test_admin"
    
    async def test_create_system_backup(self, admin_interface):
        """Test creating a system backup."""
        action_id = await admin_interface.create_system_backup(
            admin_user="test_admin",
            description="Test backup"
        )
        
        assert isinstance(action_id, str)
        
        # Check that version manager was called
        admin_interface.version_manager.create_version.assert_called_once()
        
        # Check action was recorded
        action = await admin_interface.get_admin_action(action_id)
        assert action is not None
        assert action.action_type == "system_backup"
    
    async def test_restore_from_backup_success(self, admin_interface):
        """Test successful backup restoration."""
        action_id = await admin_interface.restore_from_backup(
            version_id="v20240101_120000",
            admin_user="test_admin",
            confirmation_token="valid_token_12345"
        )
        
        assert isinstance(action_id, str)
        
        # Check that version manager was called
        admin_interface.version_manager.restore_version.assert_called_with("v20240101_120000")
        
        # Check action was recorded
        action = await admin_interface.get_admin_action(action_id)
        assert action is not None
        assert action.action_type == "system_restore"
    
    async def test_restore_from_backup_invalid_token(self, admin_interface):
        """Test backup restoration with invalid token."""
        with pytest.raises(Exception):  # Should raise LightRAGError
            await admin_interface.restore_from_backup(
                version_id="v20240101_120000",
                admin_user="test_admin",
                confirmation_token="short"  # Invalid token
            )
    
    async def test_get_performance_dashboard(self, admin_interface):
        """Test getting performance dashboard data."""
        dashboard = await admin_interface.get_performance_dashboard()
        
        assert isinstance(dashboard, dict)
        expected_keys = [
            'current_metrics', 'historical_data', 'query_statistics',
            'resource_usage', 'alerts'
        ]
        for key in expected_keys:
            assert key in dashboard
    
    async def test_get_admin_actions(self, admin_interface):
        """Test getting list of admin actions."""
        # Create some test actions
        await admin_interface.trigger_manual_update("admin1", "Update 1")
        await admin_interface.create_system_backup("admin2", "Backup 1")
        
        # Get all actions
        actions = await admin_interface.get_admin_actions()
        
        assert isinstance(actions, list)
        assert len(actions) >= 2
        
        # Check actions are sorted by time (newest first)
        if len(actions) > 1:
            assert actions[0].initiated_at >= actions[1].initiated_at
    
    async def test_get_admin_actions_filtered(self, admin_interface):
        """Test getting filtered admin actions."""
        # Create different types of actions
        await admin_interface.trigger_manual_update("admin1", "Update 1")
        await admin_interface.create_system_backup("admin2", "Backup 1")
        
        # Get only backup actions
        backup_actions = await admin_interface.get_admin_actions(
            action_type="system_backup"
        )
        
        assert isinstance(backup_actions, list)
        for action in backup_actions:
            assert action.action_type == "system_backup"
    
    async def test_get_admin_actions_limited(self, admin_interface):
        """Test getting limited number of admin actions."""
        # Create multiple actions
        for i in range(5):
            await admin_interface.trigger_manual_update(f"admin{i}", f"Update {i}")
        
        # Get limited results
        actions = await admin_interface.get_admin_actions(limit=3)
        
        assert len(actions) <= 3
    
    async def test_cleanup_old_data(self, admin_interface):
        """Test cleaning up old data."""
        action_id = await admin_interface.cleanup_old_data(
            admin_user="test_admin",
            days_to_keep=30
        )
        
        assert isinstance(action_id, str)
        
        # Check action was recorded
        action = await admin_interface.get_admin_action(action_id)
        assert action is not None
        assert action.action_type == "data_cleanup"
    
    async def test_determine_health_status_healthy(self, admin_interface):
        """Test health status determination - healthy case."""
        health_metrics = {
            'error_rate': 0.01,
            'avg_response_time': 2.0,
            'memory_usage_percent': 50,
            'disk_usage_percent': 30
        }
        
        status = admin_interface._determine_health_status(health_metrics)
        assert status == SystemHealthStatus.HEALTHY
    
    async def test_determine_health_status_warning(self, admin_interface):
        """Test health status determination - warning case."""
        health_metrics = {
            'error_rate': 0.07,
            'avg_response_time': 7.0,
            'memory_usage_percent': 85,
            'disk_usage_percent': 88
        }
        
        status = admin_interface._determine_health_status(health_metrics)
        assert status == SystemHealthStatus.WARNING
    
    async def test_determine_health_status_critical(self, admin_interface):
        """Test health status determination - critical case."""
        health_metrics = {
            'error_rate': 0.15,
            'avg_response_time': 15.0,
            'memory_usage_percent': 95,
            'disk_usage_percent': 98
        }
        
        status = admin_interface._determine_health_status(health_metrics)
        assert status == SystemHealthStatus.CRITICAL
    
    async def test_determine_health_status_unknown(self, admin_interface):
        """Test health status determination - unknown case."""
        # Empty metrics should result in unknown status
        health_metrics = {}
        
        status = admin_interface._determine_health_status(health_metrics)
        assert status == SystemHealthStatus.UNKNOWN
    
    async def test_validate_confirmation_token(self, admin_interface):
        """Test confirmation token validation."""
        # Valid token
        assert admin_interface._validate_confirmation_token("valid_token_12345") is True
        
        # Invalid token (too short)
        assert admin_interface._validate_confirmation_token("short") is False
    
    @pytest.mark.asyncio
    async def test_concurrent_admin_actions(self, admin_interface):
        """Test concurrent administrative actions."""
        # Start multiple actions concurrently
        tasks = [
            admin_interface.trigger_manual_update(f"admin{i}", f"Update {i}")
            for i in range(3)
        ]
        
        # Wait for completion
        action_ids = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(action_ids) == 3
        for action_id in action_ids:
            assert isinstance(action_id, str)
            action = await admin_interface.get_admin_action(action_id)
            assert action is not None
    
    async def test_admin_action_lifecycle(self, admin_interface):
        """Test complete admin action lifecycle."""
        # Create action
        action_id = await admin_interface._create_admin_action(
            action_type="test_action",
            description="Test action",
            initiated_by="test_user"
        )
        
        # Check initial state
        action = await admin_interface.get_admin_action(action_id)
        assert action.status == "pending"
        assert action.result is None
        assert action.error_message is None
        
        # Update to running
        await admin_interface._update_admin_action(action_id, "running")
        action = await admin_interface.get_admin_action(action_id)
        assert action.status == "running"
        
        # Update to completed with result
        result = {"success": True, "items_processed": 10}
        await admin_interface._update_admin_action(action_id, "completed", result)
        action = await admin_interface.get_admin_action(action_id)
        assert action.status == "completed"
        assert action.result == result


if __name__ == "__main__":
    pytest.main([__file__])