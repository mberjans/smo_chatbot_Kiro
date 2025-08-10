"""
Integration tests for the LightRAG component interface.

These tests verify the modular component interface meets the requirements
for async methods, health monitoring, error handling, and cleanup procedures.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime

from .config.settings import LightRAGConfig
from .component import LightRAGComponent
from .utils.health import HealthStatus


@pytest.fixture
def test_config():
    """Create a test configuration with temporary directories."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LightRAGConfig(
            knowledge_graph_path=f"{temp_dir}/kg",
            vector_store_path=f"{temp_dir}/vectors",
            cache_directory=f"{temp_dir}/cache",
            papers_directory=f"{temp_dir}/papers"
        )
        yield config


@pytest.fixture
def component(test_config):
    """Create a LightRAG component instance for testing."""
    return LightRAGComponent(test_config)


class TestComponentInitialization:
    """Test component initialization and configuration."""
    
    def test_component_creation(self, test_config):
        """Test that component can be created with valid configuration."""
        component = LightRAGComponent(test_config)
        assert component.config == test_config
        assert not component._initialized
        assert not component._initializing
        assert component._lightrag_instance is None
    
    def test_component_creation_with_invalid_config(self):
        """Test that component creation fails with invalid configuration."""
        with pytest.raises(ValueError, match="Invalid configuration"):
            # Create config with invalid empty path
            invalid_config = LightRAGConfig(
                knowledge_graph_path="",  # Invalid empty path
                vector_store_path="valid_path",
                cache_directory="valid_path", 
                papers_directory="valid_path"
            )
            LightRAGComponent(invalid_config)
    
    @pytest.mark.asyncio
    async def test_component_initialization(self, component):
        """Test component initialization process."""
        assert not component._initialized
        
        await component.initialize()
        
        assert component._initialized
        assert not component._initializing
        assert component._initialization_error is None
        assert component._stats["initialization_time"] is not None
    
    @pytest.mark.asyncio
    async def test_component_initialization_idempotent(self, component):
        """Test that initialization is idempotent."""
        await component.initialize()
        first_init_time = component._stats["initialization_time"]
        
        await component.initialize()  # Should not reinitialize
        second_init_time = component._stats["initialization_time"]
        
        assert first_init_time == second_init_time
    
    @pytest.mark.asyncio
    async def test_component_initialization_concurrent(self, component):
        """Test that concurrent initialization calls work correctly."""
        # Start multiple initialization tasks concurrently
        tasks = [component.initialize() for _ in range(5)]
        await asyncio.gather(*tasks)
        
        assert component._initialized
        assert not component._initializing
    
    @pytest.mark.asyncio
    async def test_component_initialization_creates_directories(self, component):
        """Test that initialization creates necessary directories."""
        await component.initialize()
        
        # Check that all directories were created
        assert Path(component.config.knowledge_graph_path).exists()
        assert Path(component.config.vector_store_path).exists()
        assert Path(component.config.cache_directory).exists()
        assert Path(component.config.papers_directory).exists()


class TestComponentHealthMonitoring:
    """Test component health monitoring and status reporting."""
    
    @pytest.mark.asyncio
    async def test_health_status_before_initialization(self, component):
        """Test health status before component is initialized."""
        health = await component.get_health_status()
        
        assert health.overall_status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert "initialization" in health.components
        assert health.components["initialization"].status == HealthStatus.DEGRADED
    
    @pytest.mark.asyncio
    async def test_health_status_after_initialization(self, component):
        """Test health status after successful initialization."""
        await component.initialize()
        health = await component.get_health_status()
        
        # Should be healthy or degraded (not unhealthy)
        assert health.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert "initialization" in health.components
        assert health.components["initialization"].status == HealthStatus.HEALTHY
        assert "configuration" in health.components
        assert "storage" in health.components
        assert "papers_directory" in health.components
    
    @pytest.mark.asyncio
    async def test_health_status_caching(self, component):
        """Test that health status is cached appropriately."""
        await component.initialize()
        
        # First call
        health1 = await component.get_health_status()
        timestamp1 = health1.timestamp
        
        # Second call should return cached result
        health2 = await component.get_health_status()
        timestamp2 = health2.timestamp
        
        assert timestamp1 == timestamp2  # Same timestamp indicates cached result
        
        # Force refresh should return new result
        health3 = await component.get_health_status(force_refresh=True)
        timestamp3 = health3.timestamp
        
        assert timestamp3 > timestamp1  # New timestamp indicates fresh check
    
    @pytest.mark.asyncio
    async def test_health_status_with_missing_directories(self, test_config):
        """Test health status when directories are missing."""
        # Create component but don't initialize (so directories won't be created)
        component = LightRAGComponent(test_config)
        
        health = await component.get_health_status()
        
        # Should be degraded due to missing directories
        assert health.overall_status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY]
        assert "storage" in health.components
    
    @pytest.mark.asyncio
    async def test_health_status_components(self, component):
        """Test that all expected health components are present."""
        await component.initialize()
        health = await component.get_health_status()
        
        expected_components = [
            "initialization",
            "configuration", 
            "storage",
            "papers_directory",
            "query_engine",
            "statistics"
        ]
        
        for component_name in expected_components:
            assert component_name in health.components
            assert hasattr(health.components[component_name], 'status')
            assert hasattr(health.components[component_name], 'message')
            assert hasattr(health.components[component_name], 'metrics')


class TestComponentErrorHandling:
    """Test comprehensive error handling and logging."""
    
    @pytest.mark.asyncio
    async def test_query_with_empty_question(self, component):
        """Test that empty questions are handled properly."""
        await component.initialize()
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await component.query("")
        
        with pytest.raises(ValueError, match="Question cannot be empty"):
            await component.query("   ")  # Whitespace only
    
    @pytest.mark.asyncio
    async def test_query_before_initialization(self, component):
        """Test that queries work even before explicit initialization."""
        # Should auto-initialize
        response = await component.query("What is clinical metabolomics?")
        
        assert component._initialized
        assert "answer" in response
        assert "confidence_score" in response
    
    @pytest.mark.asyncio
    async def test_query_with_missing_query_engine(self, component):
        """Test fallback behavior when query engine is not available."""
        await component.initialize()
        
        # Mock the import to fail by patching sys.modules
        import sys
        original_module = sys.modules.get('lightrag_integration.query.engine')
        sys.modules['lightrag_integration.query.engine'] = None
        
        try:
            response = await component.query("What is clinical metabolomics?")
            
            assert response["confidence_score"] == 0.0
            assert response["metadata"]["fallback_response"] is True
            assert "not available" in response["answer"]
        finally:
            # Restore the original module
            if original_module:
                sys.modules['lightrag_integration.query.engine'] = original_module
            else:
                sys.modules.pop('lightrag_integration.query.engine', None)
    
    @pytest.mark.asyncio
    async def test_ingest_documents_with_invalid_paths(self, component):
        """Test document ingestion with invalid file paths."""
        await component.initialize()
        
        invalid_paths = [
            "/nonexistent/file.pdf",
            "not_a_pdf.txt",
            "/another/missing/file.pdf"
        ]
        
        result = await component.ingest_documents(invalid_paths)
        
        assert result["processed_files"] == 0  # No valid files
        assert result["failed"] == len(invalid_paths)
        assert len(result["errors"]) == len(invalid_paths)
        assert result["invalid_files"] == invalid_paths
    
    @pytest.mark.asyncio
    async def test_ingest_documents_with_missing_papers_directory(self, component):
        """Test document ingestion when papers directory doesn't exist."""
        await component.initialize()
        
        # Remove the papers directory
        papers_dir = Path(component.config.papers_directory)
        if papers_dir.exists():
            papers_dir.rmdir()
        
        result = await component.ingest_documents()  # No paths specified
        
        assert result["processed_files"] == 0
        assert result["successful"] == 0
    
    @pytest.mark.asyncio
    async def test_error_statistics_tracking(self, component):
        """Test that errors are properly tracked in statistics."""
        await component.initialize()
        initial_errors = component._stats["errors_encountered"]
        
        # Cause an error
        try:
            await component.query("")  # Empty query should raise ValueError
        except ValueError:
            pass
        
        # Error count should increase
        assert component._stats["errors_encountered"] == initial_errors + 1
    
    @pytest.mark.asyncio
    async def test_health_check_error_handling(self, component):
        """Test that health check handles errors gracefully."""
        # Mock a method to raise an exception
        with patch.object(component, '_check_configuration_health', side_effect=Exception("Test error")):
            health = await component.get_health_status()
            
            # Should still return a health status, possibly degraded
            assert health is not None
            assert hasattr(health, 'overall_status')


class TestComponentCleanup:
    """Test component cleanup procedures."""
    
    @pytest.mark.asyncio
    async def test_cleanup_after_initialization(self, component):
        """Test cleanup after component has been initialized."""
        await component.initialize()
        assert component._initialized
        
        await component.cleanup()
        
        assert not component._initialized
        assert not component._initializing
        assert component._lightrag_instance is None
        assert component._initialization_error is None
    
    @pytest.mark.asyncio
    async def test_cleanup_before_initialization(self, component):
        """Test cleanup before component has been initialized."""
        assert not component._initialized
        
        # Should not raise an error
        await component.cleanup()
        
        assert not component._initialized
    
    @pytest.mark.asyncio
    async def test_cleanup_logs_statistics(self, component):
        """Test that cleanup logs final statistics."""
        await component.initialize()
        
        # Perform some operations to generate statistics
        await component.query("test query")
        
        # Cleanup should log statistics
        with patch.object(component.logger, 'info') as mock_log:
            await component.cleanup()
            
            # Check that statistics were logged
            log_calls = [call.args[0] for call in mock_log.call_args_list]
            assert any("statistics" in call.lower() for call in log_calls)


class TestComponentStatistics:
    """Test component statistics and metrics collection."""
    
    def test_get_statistics_initial_state(self, component):
        """Test statistics in initial state."""
        stats = component.get_statistics()
        
        assert stats["queries_processed"] == 0
        assert stats["documents_ingested"] == 0
        assert stats["errors_encountered"] == 0
        assert stats["is_initialized"] is False
        assert stats["is_initializing"] is False
        assert stats["has_initialization_error"] is False
    
    @pytest.mark.asyncio
    async def test_get_statistics_after_operations(self, component):
        """Test statistics after performing operations."""
        await component.initialize()
        
        # Perform some operations
        await component.query("test query 1")
        await component.query("test query 2")
        
        stats = component.get_statistics()
        
        assert stats["queries_processed"] == 2
        assert stats["is_initialized"] is True
        assert stats["initialization_time"] is not None
        assert stats["last_query_time"] is not None
    
    @pytest.mark.asyncio
    async def test_statistics_error_tracking(self, component):
        """Test that statistics properly track errors."""
        await component.initialize()
        initial_errors = component.get_statistics()["errors_encountered"]
        
        # Cause an error
        try:
            await component.query("")
        except ValueError:
            pass
        
        stats = component.get_statistics()
        assert stats["errors_encountered"] == initial_errors + 1


class TestComponentIntegration:
    """Test overall component integration and workflow."""
    
    @pytest.mark.asyncio
    async def test_full_component_lifecycle(self, component):
        """Test the complete component lifecycle."""
        # 1. Initial state
        assert not component._initialized
        stats = component.get_statistics()
        assert stats["queries_processed"] == 0
        
        # 2. Initialization
        await component.initialize()
        assert component._initialized
        
        # 3. Health check
        health = await component.get_health_status()
        assert health.overall_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
        # 4. Query processing
        response = await component.query("What is clinical metabolomics?")
        assert "answer" in response
        assert response["confidence_score"] >= 0
        
        # 5. Statistics check
        stats = component.get_statistics()
        assert stats["queries_processed"] == 1
        assert stats["is_initialized"] is True
        
        # 6. Document ingestion (with empty directory)
        result = await component.ingest_documents()
        assert "processed_files" in result
        
        # 7. Final cleanup
        await component.cleanup()
        assert not component._initialized
    
    @pytest.mark.asyncio
    async def test_component_with_pdf_files(self, component):
        """Test component behavior with actual PDF files in papers directory."""
        await component.initialize()
        
        # Create some dummy PDF files
        papers_dir = Path(component.config.papers_directory)
        test_pdf1 = papers_dir / "test1.pdf"
        test_pdf2 = papers_dir / "test2.pdf"
        
        # Create minimal PDF content (just for testing file detection)
        test_pdf1.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF")
        test_pdf2.write_bytes(b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n>>\nendobj\nxref\n0 1\n0000000000 65535 f \ntrailer\n<<\n/Size 1\n/Root 1 0 R\n>>\nstartxref\n9\n%%EOF")
        
        # Test health status includes PDF count
        health = await component.get_health_status(force_refresh=True)
        papers_health = health.components["papers_directory"]
        assert papers_health.metrics["pdf_count"] == 2
        
        # Test document ingestion finds the files
        result = await component.ingest_documents()
        assert result["processed_files"] == 2
        assert len(result["valid_files"]) == 2
    
    @pytest.mark.asyncio
    async def test_component_supported_formats(self, component):
        """Test that component reports supported formats correctly."""
        formats = component.get_supported_formats()
        assert ".pdf" in formats
        assert isinstance(formats, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])