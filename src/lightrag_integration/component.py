"""
Main LightRAG Component

This module provides the primary interface for LightRAG integration with the
Clinical Metabolomics Oracle system.
"""

import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

from .config.settings import LightRAGConfig
from .utils.logging import setup_logger
from .utils.health import HealthStatus, ComponentHealth, SystemHealth
from .ingestion.directory_monitor import DirectoryMonitor, BatchProcessor
from .ingestion.progress_tracker import ProgressTracker, OperationType
from .caching import CacheManager
from .performance import PerformanceOptimizer, managed_resources
from .concurrency import ConcurrencyManager, RequestPriority


class LightRAGComponent:
    """
    Main LightRAG integration component.
    
    This class encapsulates all LightRAG functionality and provides a clean
    interface for integration with the existing Clinical Metabolomics Oracle system.
    
    The component follows async/await patterns for non-blocking operations and
    includes comprehensive error handling, health monitoring, and logging.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None):
        """
        Initialize the LightRAG component.
        
        Args:
            config: Optional configuration object. If None, loads from environment.
            
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If initialization fails
        """
        try:
            self.config = config or LightRAGConfig.from_env()
            self.config.validate()
        except Exception as e:
            raise ValueError(f"Invalid configuration: {str(e)}") from e
        
        # Set up logging with error handling
        try:
            self.logger = setup_logger(
                name="lightrag_component",
                log_file=f"{self.config.cache_directory}/lightrag.log"
            )
        except Exception as e:
            # Fallback to basic logging if file logging fails
            self.logger = logging.getLogger("lightrag_component")
            self.logger.warning(f"Failed to set up file logging: {str(e)}")
        
        # Component state
        self._initialized = False
        self._initializing = False
        self._lightrag_instance = None
        self._initialization_error = None
        self._last_health_check = None
        self._health_cache_ttl = 60  # Cache health status for 60 seconds
        
        # Directory monitoring and progress tracking
        self._directory_monitor: Optional[DirectoryMonitor] = None
        self._progress_tracker: Optional[ProgressTracker] = None
        self._batch_processor: Optional[BatchProcessor] = None
        
        # Performance and scalability components
        self._cache_manager: Optional[CacheManager] = None
        self._performance_optimizer: Optional[PerformanceOptimizer] = None
        self._concurrency_manager: Optional[ConcurrencyManager] = None
        
        # Statistics tracking
        self._stats = {
            "queries_processed": 0,
            "documents_ingested": 0,
            "errors_encountered": 0,
            "initialization_time": None,
            "last_query_time": None,
            "last_ingestion_time": None
        }
        
        self.logger.info("LightRAG component created with config: %s", self.config.to_dict())
    
    async def initialize(self) -> None:
        """
        Initialize the LightRAG component and its dependencies.
        
        This method is idempotent and can be called multiple times safely.
        If initialization is already in progress, it will wait for completion.
        
        Raises:
            RuntimeError: If initialization fails
        """
        if self._initialized:
            return
        
        # If initialization is in progress, wait for it
        if self._initializing:
            while self._initializing and not self._initialized:
                await asyncio.sleep(0.1)
            if self._initialization_error:
                raise self._initialization_error
            return
        
        self._initializing = True
        start_time = datetime.now()
        
        try:
            self.logger.info("Initializing LightRAG component...")
            
            # Create necessary directories
            await self._create_directories()
            
            # Validate configuration paths
            await self._validate_paths()
            
            # Initialize progress tracker
            await self._initialize_progress_tracker()
            
            # Initialize batch processor
            await self._initialize_batch_processor()
            
            # Initialize directory monitor
            await self._initialize_directory_monitor()
            
            # Initialize performance and scalability components
            await self._initialize_cache_manager()
            await self._initialize_performance_optimizer()
            await self._initialize_concurrency_manager()
            
            # TODO: Initialize LightRAG instance in next task
            # This will be implemented when the actual LightRAG library is integrated
            
            self._initialized = True
            self._initialization_error = None
            initialization_duration = (datetime.now() - start_time).total_seconds()
            self._stats["initialization_time"] = datetime.now().isoformat()
            
            self.logger.info(
                "LightRAG component initialization completed in %.2f seconds",
                initialization_duration
            )
            
        except Exception as e:
            self._initialization_error = RuntimeError(f"Failed to initialize LightRAG component: {str(e)}")
            self.logger.error(
                "LightRAG component initialization failed: %s\n%s",
                str(e),
                traceback.format_exc()
            )
            self._stats["errors_encountered"] += 1
            raise self._initialization_error
        finally:
            self._initializing = False
    
    async def ingest_documents(self, pdf_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ingest PDF documents into the knowledge graph.
        
        Args:
            pdf_paths: Optional list of PDF file paths. If None, processes all PDFs in papers directory.
        
        Returns:
            Dictionary containing ingestion results and statistics.
            
        Raises:
            RuntimeError: If component is not initialized or ingestion fails
        """
        try:
            if not self._initialized:
                await self.initialize()
            
            start_time = datetime.now()
            
            if pdf_paths is None:
                papers_dir = Path(self.config.papers_directory)
                if not papers_dir.exists():
                    self.logger.warning("Papers directory does not exist: %s", papers_dir)
                    pdf_paths = []
                else:
                    pdf_paths = [str(p) for p in papers_dir.glob("*.pdf")]
            
            self.logger.info("Starting document ingestion for %d files", len(pdf_paths))
            
            # Validate PDF paths
            valid_paths = []
            invalid_paths = []
            
            for path in pdf_paths:
                pdf_path = Path(path)
                if pdf_path.exists() and pdf_path.suffix.lower() == '.pdf':
                    valid_paths.append(path)
                else:
                    invalid_paths.append(path)
                    self.logger.warning("Invalid or missing PDF file: %s", path)
            
            # TODO: Implement actual ingestion logic in task 6
            # This is a placeholder for the MVP implementation
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "processed_files": len(valid_paths),
                "successful": len(valid_paths),  # Placeholder - will be actual count
                "failed": len(invalid_paths),
                "errors": [f"Invalid file: {path}" for path in invalid_paths],
                "processing_time": processing_time,
                "valid_files": valid_paths,
                "invalid_files": invalid_paths
            }
            
            self._stats["documents_ingested"] += result["successful"]
            self._stats["last_ingestion_time"] = datetime.now()
            
            self.logger.info("Document ingestion completed: %s", result)
            return result
            
        except Exception as e:
            self._stats["errors_encountered"] += 1
            self.logger.error(
                "Document ingestion failed: %s\n%s",
                str(e),
                traceback.format_exc()
            )
            raise RuntimeError(f"Document ingestion failed: {str(e)}") from e
    
    async def query(self, question: str, context: Optional[Dict[str, Any]] = None, user_id: str = "anonymous") -> Dict[str, Any]:
        """
        Query the LightRAG knowledge graph with caching and concurrency management.
        
        Args:
            question: The question to ask
            context: Optional context information
            user_id: User identifier for rate limiting and tracking
        
        Returns:
            Dictionary containing the response and metadata.
            
        Raises:
            ValueError: If question is empty or invalid
            RuntimeError: If component is not initialized or query fails
        """
        try:
            if not question or not question.strip():
                raise ValueError("Question cannot be empty")
            
            if not self._initialized:
                await self.initialize()
            
            # Check cache first if available
            if self._cache_manager:
                cached_result = await self._cache_manager.get_query_result(question, context)
                if cached_result:
                    self.logger.debug("Returning cached result for query: %s", question[:50])
                    self._stats["queries_processed"] += 1
                    return cached_result
            
            # Use concurrency manager if available
            if self._concurrency_manager:
                async def query_callback():
                    return await self._execute_query_internal(question, context)
                
                # Handle request through concurrency manager
                request_result = await self._concurrency_manager.handle_request(
                    user_id=user_id,
                    request_type="query",
                    callback=query_callback,
                    priority=RequestPriority.NORMAL,
                    context=context
                )
                
                if not request_result['success']:
                    # Return error response for rate limiting or queue issues
                    return {
                        "answer": f"Request could not be processed: {request_result.get('error', 'unknown error')}",
                        "confidence_score": 0.0,
                        "source_documents": [],
                        "entities_used": [],
                        "relationships_used": [],
                        "processing_time": 0.0,
                        "metadata": {
                            "error": request_result.get('error'),
                            "rate_limited": True,
                            "request_info": request_result
                        },
                        "formatted_response": "Request rate limited or queued",
                        "confidence_breakdown": {}
                    }
                
                # Request was queued successfully - this is a simplified implementation
                # In a real system, you'd need to wait for the actual result
                self.logger.info("Request queued successfully: %s", request_result['request_id'])
                return {
                    "answer": "Your request has been queued and will be processed shortly.",
                    "confidence_score": 0.0,
                    "source_documents": [],
                    "entities_used": [],
                    "relationships_used": [],
                    "processing_time": 0.0,
                    "metadata": {
                        "queued": True,
                        "request_id": request_result['request_id'],
                        "estimated_wait_time": request_result.get('estimated_wait_time', 0)
                    },
                    "formatted_response": "Request queued for processing",
                    "confidence_breakdown": {}
                }
            else:
                # Direct execution without concurrency management
                return await self._execute_query_internal(question, context)
            
        except ValueError as e:
            self._stats["errors_encountered"] += 1
            self.logger.error("Query validation failed: %s", str(e))
            raise  # Re-raise validation errors
        except Exception as e:
            self._stats["errors_encountered"] += 1
            self.logger.error(
                "Query processing failed: %s\n%s",
                str(e),
                traceback.format_exc()
            )
            raise RuntimeError(f"Query processing failed: {str(e)}") from e
    
    async def _execute_query_internal(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal query execution with performance monitoring."""
        start_time = datetime.now()
        
        # Use performance optimizer if available
        if self._performance_optimizer:
            async with managed_resources(self._performance_optimizer, "query_processing"):
                result = await self._process_query_with_engine(question, context, start_time)
        else:
            result = await self._process_query_with_engine(question, context, start_time)
        
        # Cache result if cache manager is available
        if self._cache_manager and result.get("confidence_score", 0) > 0.5:
            await self._cache_manager.cache_query_result(question, result, context)
        
        return result
    
    async def _process_query_with_engine(self, question: str, context: Optional[Dict[str, Any]], start_time: datetime) -> Dict[str, Any]:
        """Process query using the query engine."""
        self.logger.info("Processing query: %s", question[:100] + "..." if len(question) > 100 else question)
        
        # Import and use the query engine with error handling
        try:
            from .query.engine import LightRAGQueryEngine
            
            query_engine = LightRAGQueryEngine(self.config)
            result = await query_engine.process_query(question, context)
            
        except ImportError as e:
            self.logger.error("Failed to import query engine: %s", str(e))
            # Fallback response when query engine is not available
            result = self._create_fallback_response(question, start_time)
        
        # Convert QueryResult to dictionary format for backward compatibility
        response = {
            "answer": result.answer if hasattr(result, 'answer') else result.get('answer', ''),
            "confidence_score": result.confidence_score if hasattr(result, 'confidence_score') else result.get('confidence_score', 0.0),
            "source_documents": result.source_documents if hasattr(result, 'source_documents') else result.get('source_documents', []),
            "entities_used": result.entities_used if hasattr(result, 'entities_used') else result.get('entities_used', []),
            "relationships_used": result.relationships_used if hasattr(result, 'relationships_used') else result.get('relationships_used', []),
            "processing_time": result.processing_time if hasattr(result, 'processing_time') else result.get('processing_time', 0.0),
            "metadata": result.metadata if hasattr(result, 'metadata') else result.get('metadata', {}),
            "formatted_response": result.formatted_response if hasattr(result, 'formatted_response') else result.get('formatted_response', ''),
            "confidence_breakdown": result.confidence_breakdown if hasattr(result, 'confidence_breakdown') else result.get('confidence_breakdown', {})
        }
        
        self._stats["queries_processed"] += 1
        self._stats["last_query_time"] = datetime.now()
        
        self.logger.info(
            "Query processing completed with confidence %.2f in %.2f seconds",
            response["confidence_score"],
            response["processing_time"]
        )
        return response
    
    async def get_health_status(self, force_refresh: bool = False) -> SystemHealth:
        """
        Get the health status of the LightRAG component.
        
        Args:
            force_refresh: If True, bypass cache and perform fresh health check
        
        Returns:
            SystemHealth object containing component status information.
        """
        timestamp = datetime.now()
        
        # Use cached health status if available and not expired
        if (not force_refresh and 
            self._last_health_check and 
            (timestamp - self._last_health_check.timestamp).total_seconds() < self._health_cache_ttl):
            return self._last_health_check
        
        try:
            # Check component health
            components = {}
            
            # Initialization health
            init_health = await self._check_initialization_health(timestamp)
            components["initialization"] = init_health
            
            # Configuration health
            config_health = await self._check_configuration_health(timestamp)
            components["configuration"] = config_health
            
            # Storage health
            storage_health = await self._check_storage_health(timestamp)
            components["storage"] = storage_health
            
            # Papers directory health
            papers_health = await self._check_papers_directory_health(timestamp)
            components["papers_directory"] = papers_health
            
            # Query engine health
            query_health = await self._check_query_engine_health(timestamp)
            components["query_engine"] = query_health
            
            # Statistics health
            stats_health = await self._check_statistics_health(timestamp)
            components["statistics"] = stats_health
            
            # Determine overall status
            statuses = [comp.status for comp in components.values()]
            if any(status == HealthStatus.UNHEALTHY for status in statuses):
                overall_status = HealthStatus.UNHEALTHY
            elif any(status == HealthStatus.DEGRADED for status in statuses):
                overall_status = HealthStatus.DEGRADED
            else:
                overall_status = HealthStatus.HEALTHY
            
            health_status = SystemHealth(
                overall_status=overall_status,
                components=components,
                timestamp=timestamp
            )
            
            # Cache the result
            self._last_health_check = health_status
            
            self.logger.debug("Health check completed with status: %s", overall_status.value)
            return health_status
            
        except Exception as e:
            self.logger.error("Health check failed: %s", str(e))
            # Return degraded status if health check itself fails
            return SystemHealth(
                overall_status=HealthStatus.DEGRADED,
                components={
                    "health_check": ComponentHealth(
                        status=HealthStatus.UNHEALTHY,
                        message=f"Health check failed: {str(e)}",
                        last_check=timestamp,
                        metrics={}
                    )
                },
                timestamp=timestamp
            )
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported document formats.
        
        Returns:
            List of supported file extensions.
        """
        return [".pdf"]
    
    async def start_directory_monitoring(self) -> None:
        """Start directory monitoring for automatic PDF ingestion."""
        if not self._initialized:
            await self.initialize()
        
        if self._directory_monitor:
            await self._directory_monitor.start()
            self.logger.info("Directory monitoring started")
        else:
            self.logger.error("Directory monitor not initialized")
    
    async def stop_directory_monitoring(self) -> None:
        """Stop directory monitoring."""
        if self._directory_monitor:
            await self._directory_monitor.stop()
            self.logger.info("Directory monitoring stopped")
    
    async def force_directory_scan(self) -> Dict[str, Any]:
        """Force an immediate scan of the papers directory."""
        if not self._directory_monitor:
            raise RuntimeError("Directory monitor not initialized")
        
        return await self._directory_monitor.force_scan()
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get directory monitoring status."""
        if not self._directory_monitor:
            return {"error": "Directory monitor not initialized"}
        
        return await self._directory_monitor.get_status()
    
    async def get_progress_report(self) -> Dict[str, Any]:
        """Get comprehensive progress report."""
        if not self._progress_tracker:
            return {"error": "Progress tracker not initialized"}
        
        status_report = self._progress_tracker.get_status_report()
        return status_report.to_dict()
    
    async def batch_process_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process multiple files in batches with progress tracking."""
        if not self._batch_processor:
            raise RuntimeError("Batch processor not initialized")
        
        return await self._batch_processor.process_files(file_paths)

    async def cleanup(self) -> None:
        """
        Clean up resources and close connections.
        
        This method should be called when the component is no longer needed
        to ensure proper resource cleanup.
        """
        try:
            self.logger.info("Starting LightRAG component cleanup...")
            
            # Stop directory monitoring
            if self._directory_monitor:
                await self._directory_monitor.cleanup()
                self._directory_monitor = None
            
            # Stop progress tracking
            if self._progress_tracker:
                await self._progress_tracker.stop()
                self._progress_tracker = None
            
            # Clean up batch processor
            self._batch_processor = None
            
            # Clean up performance and scalability components
            if self._cache_manager:
                await self._cache_manager.cleanup()
                self._cache_manager = None
            
            if self._performance_optimizer:
                await self._performance_optimizer.cleanup()
                self._performance_optimizer = None
            
            if self._concurrency_manager:
                await self._concurrency_manager.stop()
                self._concurrency_manager = None
            
            # Clean up LightRAG instance if it exists
            if self._lightrag_instance:
                # TODO: Implement cleanup logic when LightRAG instance is created
                # This might include closing database connections, clearing caches, etc.
                pass
            
            # Reset component state
            self._initialized = False
            self._initializing = False
            self._lightrag_instance = None
            self._initialization_error = None
            self._last_health_check = None
            
            # Log final statistics
            self.logger.info("Component statistics at cleanup: %s", self._stats)
            
            self.logger.info("LightRAG component cleanup completed successfully")
            
        except Exception as e:
            self.logger.error("Error during cleanup: %s", str(e))
            raise RuntimeError(f"Cleanup failed: {str(e)}") from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get component usage statistics.
        
        Returns:
            Dictionary containing usage statistics and metrics.
        """
        base_stats = {
            **self._stats,
            "is_initialized": self._initialized,
            "is_initializing": self._initializing,
            "has_initialization_error": self._initialization_error is not None,
            "uptime_seconds": (
                (datetime.now() - datetime.fromisoformat(self._stats["initialization_time"]))
                .total_seconds() if self._stats["initialization_time"] and isinstance(self._stats["initialization_time"], str) else 0
            )
        }
        
        # Add performance and scalability statistics
        if self._cache_manager:
            base_stats["cache_stats"] = self._cache_manager.get_comprehensive_stats()
        
        if self._performance_optimizer:
            base_stats["performance_stats"] = self._performance_optimizer.get_performance_summary()
        
        if self._concurrency_manager:
            base_stats["concurrency_stats"] = self._concurrency_manager.get_comprehensive_stats()
        
        return base_stats
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        if self._cache_manager:
            return self._cache_manager.get_comprehensive_stats()
        return {"error": "Cache manager not initialized"}
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        if self._performance_optimizer:
            return self._performance_optimizer.get_performance_summary()
        return {"error": "Performance optimizer not initialized"}
    
    async def get_concurrency_stats(self) -> Dict[str, Any]:
        """Get detailed concurrency statistics."""
        if self._concurrency_manager:
            return self._concurrency_manager.get_comprehensive_stats()
        return {"error": "Concurrency manager not initialized"}
    
    async def optimize_performance(self, force: bool = False) -> Dict[str, Any]:
        """Manually trigger performance optimization."""
        if self._performance_optimizer:
            return await self._performance_optimizer.optimize_performance(force)
        return {"error": "Performance optimizer not initialized"}
    
    async def clear_caches(self) -> Dict[str, Any]:
        """Clear all caches."""
        if self._cache_manager:
            await self._cache_manager.clear_all_caches()
            return {"success": True, "message": "All caches cleared"}
        return {"error": "Cache manager not initialized"}
    
    async def _create_directories(self) -> None:
        """Create necessary directories for the component."""
        directories = [
            self.config.knowledge_graph_path,
            self.config.vector_store_path,
            self.config.cache_directory,
            self.config.papers_directory
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                self.logger.debug("Created/verified directory: %s", directory)
            except Exception as e:
                self.logger.error("Failed to create directory %s: %s", directory, str(e))
                raise
    
    async def _validate_paths(self) -> None:
        """Validate that all configured paths are accessible."""
        paths_to_check = {
            "knowledge_graph_path": self.config.knowledge_graph_path,
            "vector_store_path": self.config.vector_store_path,
            "cache_directory": self.config.cache_directory,
            "papers_directory": self.config.papers_directory
        }
        
        for name, path in paths_to_check.items():
            path_obj = Path(path)
            if not path_obj.exists():
                self.logger.warning("Path does not exist: %s = %s", name, path)
            elif not path_obj.is_dir():
                raise RuntimeError(f"Path is not a directory: {name} = {path}")
            else:
                self.logger.debug("Validated path: %s = %s", name, path)
    
    async def _initialize_cache_manager(self) -> None:
        """Initialize cache manager."""
        try:
            self._cache_manager = CacheManager(self.config)
            await self._cache_manager.initialize()
            self.logger.info("Cache manager initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize cache manager: %s", str(e))
            # Cache manager is optional, so we don't fail initialization
            self._cache_manager = None
    
    async def _initialize_performance_optimizer(self) -> None:
        """Initialize performance optimizer."""
        try:
            self._performance_optimizer = PerformanceOptimizer(self.config)
            await self._performance_optimizer.start_monitoring()
            self.logger.info("Performance optimizer initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize performance optimizer: %s", str(e))
            # Performance optimizer is optional, so we don't fail initialization
            self._performance_optimizer = None
    
    async def _initialize_concurrency_manager(self) -> None:
        """Initialize concurrency manager."""
        try:
            self._concurrency_manager = ConcurrencyManager(self.config)
            await self._concurrency_manager.start()
            self.logger.info("Concurrency manager initialized successfully")
        except Exception as e:
            self.logger.error("Failed to initialize concurrency manager: %s", str(e))
            # Concurrency manager is optional, so we don't fail initialization
            self._concurrency_manager = None
    
    def _create_fallback_response(self, question: str, start_time: datetime) -> Dict[str, Any]:
        """Create a fallback response when the query engine is not available."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "answer": "I apologize, but the LightRAG query engine is currently not available. Please try again later.",
            "confidence_score": 0.0,
            "source_documents": [],
            "entities_used": [],
            "relationships_used": [],
            "processing_time": processing_time,
            "metadata": {
                "fallback_response": True,
                "reason": "Query engine not available"
            },
            "formatted_response": "Query engine unavailable",
            "confidence_breakdown": {}
        }
    
    async def _check_initialization_health(self, timestamp: datetime) -> ComponentHealth:
        """Check the initialization health of the component."""
        try:
            if self._initialization_error:
                return ComponentHealth(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Initialization failed: {str(self._initialization_error)}",
                    last_check=timestamp,
                    metrics={"initialized": False, "initializing": self._initializing}
                )
            elif self._initializing:
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    message="Component is currently initializing",
                    last_check=timestamp,
                    metrics={"initialized": False, "initializing": True}
                )
            elif self._initialized:
                return ComponentHealth(
                    status=HealthStatus.HEALTHY,
                    message="Component is initialized and ready",
                    last_check=timestamp,
                    metrics={
                        "initialized": True,
                        "initializing": False,
                        "initialization_time": self._stats.get("initialization_time", 0)
                    }
                )
            else:
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    message="Component is not initialized",
                    last_check=timestamp,
                    metrics={"initialized": False, "initializing": False}
                )
        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Initialization health check failed: {str(e)}",
                last_check=timestamp,
                metrics={}
            )
    
    async def _check_configuration_health(self, timestamp: datetime) -> ComponentHealth:
        """Check the configuration health of the component."""
        try:
            self.config.validate()
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                message="Configuration is valid",
                last_check=timestamp,
                metrics={
                    "config_items": len(self.config.to_dict()),
                    "embedding_model": self.config.embedding_model,
                    "llm_model": self.config.llm_model
                }
            )
        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Configuration error: {str(e)}",
                last_check=timestamp,
                metrics={}
            )
    
    async def _check_storage_health(self, timestamp: datetime) -> ComponentHealth:
        """Check the storage health of the component."""
        try:
            kg_path = Path(self.config.knowledge_graph_path)
            vector_path = Path(self.config.vector_store_path)
            cache_path = Path(self.config.cache_directory)
            
            # Check if paths exist and are writable
            paths_status = {}
            for name, path in [("kg", kg_path), ("vector", vector_path), ("cache", cache_path)]:
                paths_status[f"{name}_exists"] = path.exists()
                paths_status[f"{name}_writable"] = path.exists() and path.is_dir()
                
                # Try to create a test file to verify write access
                if path.exists():
                    try:
                        test_file = path / ".health_check"
                        test_file.touch()
                        test_file.unlink()
                        paths_status[f"{name}_write_test"] = True
                    except Exception:
                        paths_status[f"{name}_write_test"] = False
                else:
                    paths_status[f"{name}_write_test"] = False
            
            # Determine status based on path accessibility
            if all(paths_status[key] for key in paths_status if key.endswith("_exists")):
                if all(paths_status[key] for key in paths_status if key.endswith("_write_test")):
                    status = HealthStatus.HEALTHY
                    message = "All storage paths are accessible and writable"
                else:
                    status = HealthStatus.DEGRADED
                    message = "Some storage paths have write permission issues"
            else:
                status = HealthStatus.DEGRADED
                message = "Some storage paths do not exist"
            
            return ComponentHealth(
                status=status,
                message=message,
                last_check=timestamp,
                metrics=paths_status
            )
            
        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Storage health check failed: {str(e)}",
                last_check=timestamp,
                metrics={}
            )
    
    async def _check_papers_directory_health(self, timestamp: datetime) -> ComponentHealth:
        """Check the papers directory health."""
        try:
            papers_path = Path(self.config.papers_directory)
            
            if not papers_path.exists():
                return ComponentHealth(
                    status=HealthStatus.DEGRADED,
                    message="Papers directory does not exist",
                    last_check=timestamp,
                    metrics={"papers_directory_exists": False, "pdf_count": 0}
                )
            
            # Count PDF files
            pdf_files = list(papers_path.glob("*.pdf"))
            pdf_count = len(pdf_files)
            
            # Check for recent files (within last 30 days)
            recent_files = 0
            if pdf_files:
                thirty_days_ago = datetime.now().timestamp() - (30 * 24 * 60 * 60)
                recent_files = sum(1 for f in pdf_files if f.stat().st_mtime > thirty_days_ago)
            
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                message=f"Papers directory contains {pdf_count} PDF files",
                last_check=timestamp,
                metrics={
                    "papers_directory_exists": True,
                    "pdf_count": pdf_count,
                    "recent_files_count": recent_files,
                    "directory_size_mb": sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Papers directory health check failed: {str(e)}",
                last_check=timestamp,
                metrics={}
            )
    
    async def _check_query_engine_health(self, timestamp: datetime) -> ComponentHealth:
        """Check the query engine health."""
        try:
            # Try to import the query engine
            from .query.engine import LightRAGQueryEngine
            
            # Basic instantiation test
            query_engine = LightRAGQueryEngine(self.config)
            
            return ComponentHealth(
                status=HealthStatus.HEALTHY,
                message="Query engine is available and can be instantiated",
                last_check=timestamp,
                metrics={
                    "query_engine_available": True,
                    "last_query_time": self._stats.get("last_query_time").isoformat() if self._stats.get("last_query_time") else None
                }
            )
            
        except ImportError as e:
            return ComponentHealth(
                status=HealthStatus.DEGRADED,
                message=f"Query engine import failed: {str(e)}",
                last_check=timestamp,
                metrics={"query_engine_available": False}
            )
        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Query engine health check failed: {str(e)}",
                last_check=timestamp,
                metrics={"query_engine_available": False}
            )
    
    async def _check_statistics_health(self, timestamp: datetime) -> ComponentHealth:
        """Check the component statistics and performance metrics."""
        try:
            stats = self._stats.copy()
            
            # Calculate some derived metrics
            error_rate = (
                stats["errors_encountered"] / max(stats["queries_processed"], 1)
                if stats["queries_processed"] > 0 else 0
            )
            
            # Determine status based on error rate
            if error_rate > 0.1:  # More than 10% error rate
                status = HealthStatus.DEGRADED
                message = f"High error rate: {error_rate:.2%}"
            elif error_rate > 0.05:  # More than 5% error rate
                status = HealthStatus.DEGRADED
                message = f"Moderate error rate: {error_rate:.2%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"Low error rate: {error_rate:.2%}"
            
            return ComponentHealth(
                status=status,
                message=message,
                last_check=timestamp,
                metrics={
                    **stats,
                    "error_rate": error_rate,
                    "uptime_seconds": (
                        (timestamp - datetime.fromisoformat(stats["initialization_time"]))
                        .total_seconds() if stats.get("initialization_time") and isinstance(stats.get("initialization_time"), str) else 0
                    )
                }
            )
            
        except Exception as e:
            return ComponentHealth(
                status=HealthStatus.UNHEALTHY,
                message=f"Statistics health check failed: {str(e)}",
                last_check=timestamp,
                metrics={}
            )
    
    async def _initialize_progress_tracker(self) -> None:
        """Initialize the progress tracker."""
        try:
            progress_config = {
                'max_history': 1000,
                'cleanup_interval': 3600,
                'persist_to_file': True,
                'status_file': f"{self.config.cache_directory}/progress_status.json"
            }
            
            self._progress_tracker = ProgressTracker(progress_config)
            await self._progress_tracker.start()
            
            self.logger.info("Progress tracker initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize progress tracker: {str(e)}")
            raise
    
    async def _initialize_batch_processor(self) -> None:
        """Initialize the batch processor."""
        try:
            # Import the ingestion pipeline
            from .ingestion.pipeline import PDFIngestionPipeline
            
            # Create ingestion pipeline instance
            ingestion_pipeline = PDFIngestionPipeline(self.config)
            
            # Configure batch processor
            batch_config = {
                'batch_size': getattr(self.config, 'batch_size', 5),
                'max_concurrent': getattr(self.config, 'max_concurrent_requests', 3),
                'progress_callback': self._progress_callback if self._progress_tracker else None
            }
            
            self._batch_processor = BatchProcessor(ingestion_pipeline, batch_config)
            
            self.logger.info("Batch processor initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize batch processor: {str(e)}")
            raise
    
    async def _initialize_directory_monitor(self) -> None:
        """Initialize the directory monitor."""
        try:
            # Configure directory monitor
            monitor_config = {
                'scan_interval': 30,  # seconds
                'batch_size': getattr(self.config, 'batch_size', 10),
                'max_file_age_hours': 24,
                'enable_recursive': False,
                'file_extensions': ['.pdf']
            }
            
            self._directory_monitor = DirectoryMonitor(
                papers_directory=self.config.papers_directory,
                ingestion_callback=self._directory_ingestion_callback,
                config=monitor_config
            )
            
            self.logger.info("Directory monitor initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize directory monitor: {str(e)}")
            raise
    
    async def _directory_ingestion_callback(self, file_paths: List[str]) -> Dict[str, Any]:
        """Callback for directory monitor to process detected files."""
        try:
            self.logger.info(f"Directory monitor triggered ingestion for {len(file_paths)} files")
            
            # Start progress tracking for this batch
            operation_id = f"directory_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            if self._progress_tracker:
                self._progress_tracker.start_operation(
                    operation_id=operation_id,
                    operation_type=OperationType.BATCH_PROCESSING,
                    total_items=len(file_paths),
                    metadata={'source': 'directory_monitor', 'file_paths': file_paths}
                )
            
            # Process files using batch processor
            if self._batch_processor:
                result = await self._batch_processor.process_files(file_paths)
                
                # Complete progress tracking
                if self._progress_tracker:
                    success = result.get('failed', 0) == 0
                    error_msg = None if success else f"Failed to process {result.get('failed', 0)} files"
                    
                    self._progress_tracker.complete_operation(
                        operation_id=operation_id,
                        success=success,
                        error_message=error_msg,
                        metadata={'result': result}
                    )
                
                return result
            else:
                # Fallback to basic ingestion
                result = await self.ingest_documents(file_paths)
                
                if self._progress_tracker:
                    success = result.get('failed', 0) == 0
                    error_msg = None if success else f"Failed to process {result.get('failed', 0)} files"
                    
                    self._progress_tracker.complete_operation(
                        operation_id=operation_id,
                        success=success,
                        error_message=error_msg,
                        metadata={'result': result}
                    )
                
                return result
                
        except Exception as e:
            error_msg = f"Error in directory ingestion callback: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            if self._progress_tracker:
                self._progress_tracker.complete_operation(
                    operation_id=operation_id,
                    success=False,
                    error_message=error_msg
                )
            
            return {
                'successful': 0,
                'failed': len(file_paths),
                'errors': [error_msg]
            }
    
    async def _progress_callback(self, progress: Dict[str, Any]) -> None:
        """Progress callback for batch processing."""
        try:
            # This can be used to update UI or send notifications
            self.logger.debug(f"Batch processing progress: {progress}")
            
            # You could emit events here for real-time UI updates
            # For now, just log the progress
            
        except Exception as e:
            self.logger.error(f"Error in progress callback: {str(e)}")