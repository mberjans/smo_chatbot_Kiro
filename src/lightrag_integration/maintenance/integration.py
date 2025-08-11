"""
Integration module for LightRAG maintenance system.

This module provides:
- Easy setup and configuration of maintenance components
- Integration with the main LightRAG system
- Dependency injection for API endpoints
- Configuration management for maintenance features
"""

import logging
from typing import Optional
from pathlib import Path

from .admin_interface import AdminInterface
from .update_system import KnowledgeBaseUpdater
from .version_control import VersionManager
from .api_endpoints import router, get_admin_interface
from ..component import LightRAGComponent
from ..monitoring import PerformanceMonitor
from ..error_handling import ErrorHandler
from ..ingestion.pipeline import PDFIngestionPipeline


class MaintenanceSystem:
    """
    Main maintenance system that coordinates all maintenance components.
    
    Provides a unified interface for:
    - Knowledge base updates
    - Version control
    - Administrative operations
    - API endpoint integration
    """
    
    def __init__(
        self,
        lightrag_component: LightRAGComponent,
        knowledge_base_path: str = "data/lightrag_kg",
        versions_path: str = "data/versions",
        metadata_file: str = "data/document_metadata.json",
        watch_directories: Optional[list] = None,
        max_versions: int = 50,
        compress_backups: bool = True
    ):
        """
        Initialize the maintenance system.
        
        Args:
            lightrag_component: Main LightRAG component
            knowledge_base_path: Path to knowledge base storage
            versions_path: Path to version storage
            metadata_file: Path to document metadata file
            watch_directories: Directories to watch for new documents
            max_versions: Maximum number of versions to keep
            compress_backups: Whether to compress backup files
        """
        self.lightrag_component = lightrag_component
        self.knowledge_base_path = Path(knowledge_base_path)
        self.versions_path = Path(versions_path)
        self.metadata_file = Path(metadata_file)
        self.watch_directories = watch_directories or ["papers/", "custom_papers/"]
        self.max_versions = max_versions
        self.compress_backups = compress_backups
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all maintenance components."""
        try:
            # Create error handler
            self.error_handler = ErrorHandler()
            
            # Create performance monitor
            self.performance_monitor = PerformanceMonitor()
            
            # Create version manager
            self.version_manager = VersionManager(
                knowledge_base_path=str(self.knowledge_base_path),
                versions_path=str(self.versions_path),
                max_versions=self.max_versions,
                compress_backups=self.compress_backups
            )
            
            # Create ingestion pipeline (this would be injected in real implementation)
            # For now, create a basic config
            pipeline_config = {
                'pdf_extractor': {
                    'batch_size': 10,
                    'max_concurrent': 5
                },
                'knowledge_graph': {
                    'entity_extraction': True,
                    'relationship_extraction': True
                }
            }
            self.ingestion_pipeline = PDFIngestionPipeline(pipeline_config)
            
            # Create knowledge base updater
            self.updater = KnowledgeBaseUpdater(
                ingestion_pipeline=self.ingestion_pipeline,
                version_manager=self.version_manager,
                error_handler=self.error_handler,
                metadata_file=str(self.metadata_file),
                watch_directories=self.watch_directories
            )
            
            # Create admin interface
            self.admin_interface = AdminInterface(
                lightrag_component=self.lightrag_component,
                updater=self.updater,
                version_manager=self.version_manager,
                performance_monitor=self.performance_monitor,
                error_handler=self.error_handler
            )
            
            self.logger.info("Maintenance system components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize maintenance components: {e}")
            raise
    
    async def initialize(self):
        """Initialize all components asynchronously."""
        try:
            # Initialize version manager
            await self.version_manager.initialize()
            
            # Initialize updater
            await self.updater.initialize()
            
            # Initialize performance monitor
            await self.performance_monitor.initialize()
            
            self.logger.info("Maintenance system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize maintenance system: {e}")
            raise
    
    def get_admin_interface(self) -> AdminInterface:
        """Get the admin interface instance."""
        return self.admin_interface
    
    def get_api_router(self):
        """Get the FastAPI router with properly configured dependencies."""
        # Override the dependency to return our admin interface
        router.dependency_overrides[get_admin_interface] = lambda: self.admin_interface
        return router
    
    async def perform_health_check(self) -> dict:
        """Perform a comprehensive health check of the maintenance system."""
        try:
            health_status = {
                "maintenance_system": "healthy",
                "components": {},
                "timestamp": None
            }
            
            # Check version manager
            try:
                versions = await self.version_manager.list_versions(limit=1)
                health_status["components"]["version_manager"] = {
                    "status": "healthy",
                    "versions_available": len(versions)
                }
            except Exception as e:
                health_status["components"]["version_manager"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check updater
            try:
                active_updates = await self.updater.list_active_updates()
                health_status["components"]["updater"] = {
                    "status": "healthy",
                    "active_updates": len(active_updates)
                }
            except Exception as e:
                health_status["components"]["updater"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check admin interface
            try:
                system_status = await self.admin_interface.get_system_status()
                health_status["components"]["admin_interface"] = {
                    "status": "healthy",
                    "system_health": system_status.health_status.value
                }
            except Exception as e:
                health_status["components"]["admin_interface"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Determine overall health
            unhealthy_components = [
                name for name, status in health_status["components"].items()
                if status["status"] == "unhealthy"
            ]
            
            if unhealthy_components:
                health_status["maintenance_system"] = "degraded"
                health_status["unhealthy_components"] = unhealthy_components
            
            health_status["timestamp"] = self._get_current_timestamp()
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                "maintenance_system": "unhealthy",
                "error": str(e),
                "timestamp": self._get_current_timestamp()
            }
    
    async def shutdown(self):
        """Shutdown the maintenance system gracefully."""
        try:
            self.logger.info("Shutting down maintenance system...")
            
            # Stop any active updates
            active_updates = await self.updater.list_active_updates()
            if active_updates:
                self.logger.warning(f"Shutting down with {len(active_updates)} active updates")
            
            # Cleanup resources
            if hasattr(self.performance_monitor, 'shutdown'):
                await self.performance_monitor.shutdown()
            
            self.logger.info("Maintenance system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during maintenance system shutdown: {e}")
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()


# Global maintenance system instance
_maintenance_system: Optional[MaintenanceSystem] = None


def initialize_maintenance_system(
    lightrag_component: LightRAGComponent,
    **kwargs
) -> MaintenanceSystem:
    """
    Initialize the global maintenance system instance.
    
    Args:
        lightrag_component: Main LightRAG component
        **kwargs: Additional configuration parameters
        
    Returns:
        MaintenanceSystem instance
    """
    global _maintenance_system
    
    if _maintenance_system is not None:
        logging.warning("Maintenance system already initialized, returning existing instance")
        return _maintenance_system
    
    _maintenance_system = MaintenanceSystem(lightrag_component, **kwargs)
    return _maintenance_system


def get_maintenance_system() -> Optional[MaintenanceSystem]:
    """Get the global maintenance system instance."""
    return _maintenance_system


async def initialize_maintenance_system_async(
    lightrag_component: LightRAGComponent,
    **kwargs
) -> MaintenanceSystem:
    """
    Initialize and setup the maintenance system asynchronously.
    
    Args:
        lightrag_component: Main LightRAG component
        **kwargs: Additional configuration parameters
        
    Returns:
        Initialized MaintenanceSystem instance
    """
    maintenance_system = initialize_maintenance_system(lightrag_component, **kwargs)
    await maintenance_system.initialize()
    return maintenance_system


def get_admin_interface_dependency():
    """
    Dependency function for FastAPI to get admin interface.
    
    This function can be used to override the default dependency
    in the API endpoints.
    """
    maintenance_system = get_maintenance_system()
    if maintenance_system is None:
        raise RuntimeError("Maintenance system not initialized")
    
    return maintenance_system.get_admin_interface()


# Configure the API router with the proper dependency
def configure_api_router():
    """Configure the API router with proper dependencies."""
    router.dependency_overrides[get_admin_interface] = get_admin_interface_dependency
    return router