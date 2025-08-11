"""
Tests for the maintenance system integration.

Tests cover:
- MaintenanceSystem initialization and setup
- Component integration
- Health checks
- API router configuration
"""

import pytest
import pytest_asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from .integration import (
    MaintenanceSystem,
    initialize_maintenance_system,
    get_maintenance_system,
    initialize_maintenance_system_async,
    get_admin_interface_dependency,
    configure_api_router
)
from ..component import LightRAGComponent


class TestMaintenanceSystem:
    """Test suite for MaintenanceSystem."""
    
    @pytest_asyncio.fixture
    async def temp_dir(self):
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest_asyncio.fixture
    async def mock_lightrag_component(self):
        """Mock LightRAG component."""
        return AsyncMock(spec=LightRAGComponent)
    
    @pytest_asyncio.fixture
    async def maintenance_system(self, mock_lightrag_component, temp_dir):
        """Create MaintenanceSystem instance for testing."""
        system = MaintenanceSystem(
            lightrag_component=mock_lightrag_component,
            knowledge_base_path=str(temp_dir / "kb"),
            versions_path=str(temp_dir / "versions"),
            metadata_file=str(temp_dir / "metadata.json"),
            watch_directories=[str(temp_dir / "papers")],
            max_versions=10,
            compress_backups=False
        )
        
        # Create required directories
        (temp_dir / "papers").mkdir(exist_ok=True)
        
        return system
    
    @pytest.mark.asyncio
    async def test_maintenance_system_initialization(self, maintenance_system):
        """Test MaintenanceSystem initialization."""
        # Check that components are initialized
        assert maintenance_system.error_handler is not None
        assert maintenance_system.performance_monitor is not None
        assert maintenance_system.version_manager is not None
        assert maintenance_system.ingestion_pipeline is not None
        assert maintenance_system.updater is not None
        assert maintenance_system.admin_interface is not None
    
    @pytest.mark.asyncio
    async def test_maintenance_system_async_initialization(self, maintenance_system):
        """Test async initialization of MaintenanceSystem."""
        # Mock the initialize methods to avoid actual initialization
        with patch.object(maintenance_system.version_manager, 'initialize', new_callable=AsyncMock) as mock_vm_init, \
             patch.object(maintenance_system.updater, 'initialize', new_callable=AsyncMock) as mock_updater_init, \
             patch.object(maintenance_system.performance_monitor, 'initialize', new_callable=AsyncMock) as mock_pm_init:
            
            await maintenance_system.initialize()
            
            # Check that all components were initialized
            mock_vm_init.assert_called_once()
            mock_updater_init.assert_called_once()
            mock_pm_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_admin_interface(self, maintenance_system):
        """Test getting admin interface."""
        admin_interface = maintenance_system.get_admin_interface()
        assert admin_interface is not None
        assert admin_interface == maintenance_system.admin_interface
    
    @pytest.mark.asyncio
    async def test_get_api_router(self, maintenance_system):
        """Test getting API router with configured dependencies."""
        router = maintenance_system.get_api_router()
        assert router is not None
        
        # Check that dependency override was set
        from .api_endpoints import get_admin_interface
        assert get_admin_interface in router.dependency_overrides
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self, maintenance_system):
        """Test health check with healthy components."""
        # Mock component methods to return healthy status
        with patch.object(maintenance_system.version_manager, 'list_versions', new_callable=AsyncMock) as mock_list_versions, \
             patch.object(maintenance_system.updater, 'list_active_updates', new_callable=AsyncMock) as mock_list_updates, \
             patch.object(maintenance_system.admin_interface, 'get_system_status', new_callable=AsyncMock) as mock_get_status:
            
            # Configure mocks
            mock_list_versions.return_value = []
            mock_list_updates.return_value = []
            
            from .admin_interface import SystemStatus, SystemHealthStatus
            mock_get_status.return_value = SystemStatus(
                health_status=SystemHealthStatus.HEALTHY,
                uptime_seconds=3600,
                total_documents=100,
                total_queries=500,
                active_updates=0,
                last_update=None,
                error_count_24h=0,
                performance_metrics={},
                storage_usage={},
                version_info={}
            )
            
            health_status = await maintenance_system.perform_health_check()
            
            assert health_status["maintenance_system"] == "healthy"
            assert "components" in health_status
            assert "timestamp" in health_status
            
            # Check individual components
            assert health_status["components"]["version_manager"]["status"] == "healthy"
            assert health_status["components"]["updater"]["status"] == "healthy"
            assert health_status["components"]["admin_interface"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_with_unhealthy_component(self, maintenance_system):
        """Test health check with one unhealthy component."""
        # Mock one component to fail
        with patch.object(maintenance_system.version_manager, 'list_versions', new_callable=AsyncMock) as mock_list_versions, \
             patch.object(maintenance_system.updater, 'list_active_updates', new_callable=AsyncMock) as mock_list_updates, \
             patch.object(maintenance_system.admin_interface, 'get_system_status', new_callable=AsyncMock) as mock_get_status:
            
            # Configure mocks - one fails
            mock_list_versions.side_effect = Exception("Version manager error")
            mock_list_updates.return_value = []
            
            from .admin_interface import SystemStatus, SystemHealthStatus
            mock_get_status.return_value = SystemStatus(
                health_status=SystemHealthStatus.HEALTHY,
                uptime_seconds=3600,
                total_documents=100,
                total_queries=500,
                active_updates=0,
                last_update=None,
                error_count_24h=0,
                performance_metrics={},
                storage_usage={},
                version_info={}
            )
            
            health_status = await maintenance_system.perform_health_check()
            
            assert health_status["maintenance_system"] == "degraded"
            assert "unhealthy_components" in health_status
            assert "version_manager" in health_status["unhealthy_components"]
            
            # Check that the unhealthy component has error info
            assert health_status["components"]["version_manager"]["status"] == "unhealthy"
            assert "error" in health_status["components"]["version_manager"]
    
    @pytest.mark.asyncio
    async def test_shutdown(self, maintenance_system):
        """Test maintenance system shutdown."""
        # Mock components
        with patch.object(maintenance_system.updater, 'list_active_updates', new_callable=AsyncMock) as mock_list_updates, \
             patch.object(maintenance_system.performance_monitor, 'shutdown', new_callable=AsyncMock) as mock_pm_shutdown:
            
            mock_list_updates.return_value = []
            
            # Should not raise any exceptions
            await maintenance_system.shutdown()
            
            mock_list_updates.assert_called_once()
            mock_pm_shutdown.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_shutdown_with_active_updates(self, maintenance_system):
        """Test shutdown with active updates."""
        from .update_system import UpdateResult, UpdateStatus
        from datetime import datetime
        
        # Mock active updates
        with patch.object(maintenance_system.updater, 'list_active_updates', new_callable=AsyncMock) as mock_list_updates, \
             patch.object(maintenance_system.performance_monitor, 'shutdown', new_callable=AsyncMock) as mock_pm_shutdown:
            
            mock_list_updates.return_value = [
                UpdateResult("update1", UpdateStatus.RUNNING, datetime.now())
            ]
            
            # Should not raise any exceptions even with active updates
            await maintenance_system.shutdown()
            
            mock_list_updates.assert_called_once()
            mock_pm_shutdown.assert_called_once()


class TestGlobalFunctions:
    """Test suite for global maintenance system functions."""
    
    @pytest_asyncio.fixture
    async def mock_lightrag_component(self):
        """Mock LightRAG component."""
        return AsyncMock(spec=LightRAGComponent)
    
    def test_initialize_maintenance_system(self, mock_lightrag_component):
        """Test global maintenance system initialization."""
        # Clear any existing global instance
        import src.lightrag_integration.maintenance.integration as integration_module
        integration_module._maintenance_system = None
        
        system = initialize_maintenance_system(mock_lightrag_component)
        
        assert system is not None
        assert isinstance(system, MaintenanceSystem)
        
        # Test that subsequent calls return the same instance
        system2 = initialize_maintenance_system(mock_lightrag_component)
        assert system is system2
    
    def test_get_maintenance_system(self, mock_lightrag_component):
        """Test getting global maintenance system."""
        # Clear any existing global instance
        import src.lightrag_integration.maintenance.integration as integration_module
        integration_module._maintenance_system = None
        
        # Should return None when not initialized
        system = get_maintenance_system()
        assert system is None
        
        # Initialize and test
        initialize_maintenance_system(mock_lightrag_component)
        system = get_maintenance_system()
        assert system is not None
        assert isinstance(system, MaintenanceSystem)
    
    @pytest.mark.asyncio
    async def test_initialize_maintenance_system_async(self, mock_lightrag_component):
        """Test async initialization of global maintenance system."""
        # Clear any existing global instance
        import src.lightrag_integration.maintenance.integration as integration_module
        integration_module._maintenance_system = None
        
        with patch('src.lightrag_integration.maintenance.integration.MaintenanceSystem.initialize', new_callable=AsyncMock) as mock_init:
            system = await initialize_maintenance_system_async(mock_lightrag_component)
            
            assert system is not None
            assert isinstance(system, MaintenanceSystem)
            mock_init.assert_called_once()
    
    def test_get_admin_interface_dependency_not_initialized(self):
        """Test admin interface dependency when system not initialized."""
        # Clear any existing global instance
        import src.lightrag_integration.maintenance.integration as integration_module
        integration_module._maintenance_system = None
        
        with pytest.raises(RuntimeError, match="Maintenance system not initialized"):
            get_admin_interface_dependency()
    
    def test_get_admin_interface_dependency_initialized(self, mock_lightrag_component):
        """Test admin interface dependency when system is initialized."""
        # Clear any existing global instance
        import src.lightrag_integration.maintenance.integration as integration_module
        integration_module._maintenance_system = None
        
        # Initialize system
        initialize_maintenance_system(mock_lightrag_component)
        
        # Should return admin interface
        admin_interface = get_admin_interface_dependency()
        assert admin_interface is not None
    
    def test_configure_api_router(self, mock_lightrag_component):
        """Test API router configuration."""
        # Clear any existing global instance
        import src.lightrag_integration.maintenance.integration as integration_module
        integration_module._maintenance_system = None
        
        # Initialize system
        initialize_maintenance_system(mock_lightrag_component)
        
        # Configure router
        router = configure_api_router()
        assert router is not None
        
        # Check that dependency override was set
        from .api_endpoints import get_admin_interface
        assert get_admin_interface in router.dependency_overrides


if __name__ == "__main__":
    pytest.main([__file__])