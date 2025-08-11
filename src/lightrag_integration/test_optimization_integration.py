"""
Simple Integration Tests for LightRAG Scalability Optimizations

This module provides basic tests to validate that the optimization components
are properly integrated into the main LightRAG component.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch

from .component import LightRAGComponent
from .config.settings import LightRAGConfig


class TestOptimizationIntegration:
    """Test basic optimization integration."""
    
    def test_config_includes_optimization_settings(self):
        """Test that configuration includes optimization settings."""
        config = LightRAGConfig()
        
        # Check caching settings
        assert hasattr(config, 'query_cache_size')
        assert hasattr(config, 'embedding_cache_size')
        assert hasattr(config, 'enable_cache_warming')
        
        # Check performance settings
        assert hasattr(config, 'max_memory_mb')
        assert hasattr(config, 'max_cpu_percent')
        assert hasattr(config, 'performance_monitoring_enabled')
        
        # Check concurrency settings
        assert hasattr(config, 'max_concurrent_users')
        assert hasattr(config, 'default_requests_per_minute')
        assert hasattr(config, 'max_queue_size')

    @pytest.mark.asyncio
    async def test_component_creates_optimization_components(self):
        """Test that component creates optimization components during initialization."""
        config = LightRAGConfig(
            # Minimal settings for testing
            query_cache_size=10,
            max_memory_mb=256,
            max_concurrent_users=5,
            performance_monitoring_enabled=False,  # Disable to avoid background tasks
            enable_cache_warming=False
        )
        
        component = LightRAGComponent(config)
        
        try:
            await component.initialize()
            
            # Check that optimization components are created
            assert component._cache_manager is not None
            assert component._performance_optimizer is not None
            assert component._concurrency_manager is not None
            
            # Check that component is initialized
            assert component._initialized is True
            
        finally:
            await component.cleanup()

    @pytest.mark.asyncio
    async def test_component_statistics_include_optimization_data(self):
        """Test that component statistics include optimization data."""
        config = LightRAGConfig(
            query_cache_size=10,
            max_memory_mb=256,
            max_concurrent_users=5,
            performance_monitoring_enabled=False,
            enable_cache_warming=False
        )
        
        component = LightRAGComponent(config)
        
        try:
            await component.initialize()
            
            # Get statistics
            stats = component.get_statistics()
            
            # Should include optimization statistics
            assert "cache_stats" in stats
            assert "performance_stats" in stats
            assert "concurrency_stats" in stats
            
        finally:
            await component.cleanup()

    @pytest.mark.asyncio
    async def test_component_optimization_methods(self):
        """Test that component exposes optimization methods."""
        config = LightRAGConfig(
            query_cache_size=10,
            max_memory_mb=256,
            max_concurrent_users=5,
            performance_monitoring_enabled=False,
            enable_cache_warming=False
        )
        
        component = LightRAGComponent(config)
        
        try:
            await component.initialize()
            
            # Test cache stats method
            cache_stats = await component.get_cache_stats()
            assert isinstance(cache_stats, dict)
            
            # Test performance stats method
            perf_stats = await component.get_performance_stats()
            assert isinstance(perf_stats, dict)
            
            # Test concurrency stats method
            concurrency_stats = await component.get_concurrency_stats()
            assert isinstance(concurrency_stats, dict)
            
            # Test optimization trigger
            opt_result = await component.optimize_performance(force=True)
            assert isinstance(opt_result, dict)
            
            # Test cache clearing
            cache_result = await component.clear_caches()
            assert isinstance(cache_result, dict)
            assert cache_result.get("success") is True
            
        finally:
            await component.cleanup()

    @pytest.mark.asyncio
    async def test_query_method_with_user_id(self):
        """Test that query method accepts user_id parameter."""
        config = LightRAGConfig(
            query_cache_size=10,
            max_memory_mb=256,
            max_concurrent_users=5,
            performance_monitoring_enabled=False,
            enable_cache_warming=False
        )
        
        component = LightRAGComponent(config)
        
        try:
            await component.initialize()
            
            # Test query with user_id
            result = await component.query("Test query", user_id="test_user")
            assert result is not None
            assert isinstance(result, dict)
            
            # Should have basic response structure
            assert "answer" in result
            assert "confidence_score" in result
            assert "processing_time" in result
            
        finally:
            await component.cleanup()

    @pytest.mark.asyncio
    async def test_component_cleanup_handles_optimization_components(self):
        """Test that cleanup properly handles optimization components."""
        config = LightRAGConfig(
            query_cache_size=10,
            max_memory_mb=256,
            max_concurrent_users=5,
            performance_monitoring_enabled=False,
            enable_cache_warming=False
        )
        
        component = LightRAGComponent(config)
        
        # Initialize component
        await component.initialize()
        
        # Verify components are created
        assert component._cache_manager is not None
        assert component._performance_optimizer is not None
        assert component._concurrency_manager is not None
        
        # Cleanup should not raise exceptions
        await component.cleanup()
        
        # Components should be cleaned up
        assert component._cache_manager is None
        assert component._performance_optimizer is None
        assert component._concurrency_manager is None
        assert component._initialized is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])