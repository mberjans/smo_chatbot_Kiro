"""
Simple integration tests for Chainlit-LightRAG interaction.

Tests the basic integration functionality without complex fixtures.
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from lightrag_integration.component import LightRAGComponent
from lightrag_integration.config.settings import LightRAGConfig


class TestSimpleIntegration:
    """Simple integration tests."""
    
    def test_lightrag_config_creation(self):
        """Test that LightRAG configuration can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LightRAGConfig(
                knowledge_graph_path=os.path.join(temp_dir, "kg"),
                vector_store_path=os.path.join(temp_dir, "vectors"),
                cache_directory=os.path.join(temp_dir, "cache"),
                papers_directory=os.path.join(temp_dir, "papers")
            )
            
            assert config.knowledge_graph_path.endswith("kg")
            assert config.vector_store_path.endswith("vectors")
            assert config.cache_directory.endswith("cache")
            assert config.papers_directory.endswith("papers")
    
    def test_lightrag_component_creation(self):
        """Test that LightRAG component can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LightRAGConfig(
                knowledge_graph_path=os.path.join(temp_dir, "kg"),
                vector_store_path=os.path.join(temp_dir, "vectors"),
                cache_directory=os.path.join(temp_dir, "cache"),
                papers_directory=os.path.join(temp_dir, "papers")
            )
            
            component = LightRAGComponent(config)
            assert component is not None
            assert not component._initialized
    
    @pytest.mark.asyncio
    async def test_lightrag_component_initialization(self):
        """Test that LightRAG component can be initialized."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LightRAGConfig(
                knowledge_graph_path=os.path.join(temp_dir, "kg"),
                vector_store_path=os.path.join(temp_dir, "vectors"),
                cache_directory=os.path.join(temp_dir, "cache"),
                papers_directory=os.path.join(temp_dir, "papers")
            )
            
            component = LightRAGComponent(config)
            await component.initialize()
            
            assert component._initialized
            
            await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_lightrag_query_with_fallback(self):
        """Test LightRAG query with fallback response."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LightRAGConfig(
                knowledge_graph_path=os.path.join(temp_dir, "kg"),
                vector_store_path=os.path.join(temp_dir, "vectors"),
                cache_directory=os.path.join(temp_dir, "cache"),
                papers_directory=os.path.join(temp_dir, "papers")
            )
            
            component = LightRAGComponent(config)
            await component.initialize()
            
            # Query should return fallback response when query engine is not available
            result = await component.query("What is clinical metabolomics?")
            
            assert "answer" in result
            assert "confidence_score" in result
            assert result["confidence_score"] == 0.0  # Fallback response has 0 confidence
            assert result["metadata"]["fallback_response"] is True
            
            await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_lightrag_health_status(self):
        """Test LightRAG health status reporting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LightRAGConfig(
                knowledge_graph_path=os.path.join(temp_dir, "kg"),
                vector_store_path=os.path.join(temp_dir, "vectors"),
                cache_directory=os.path.join(temp_dir, "cache"),
                papers_directory=os.path.join(temp_dir, "papers")
            )
            
            component = LightRAGComponent(config)
            await component.initialize()
            
            health_status = await component.get_health_status()
            
            assert health_status.overall_status is not None
            assert "initialization" in health_status.components
            assert "configuration" in health_status.components
            assert "storage" in health_status.components
            
            await component.cleanup()
    
    def test_supported_formats(self):
        """Test that component reports supported formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LightRAGConfig(
                knowledge_graph_path=os.path.join(temp_dir, "kg"),
                vector_store_path=os.path.join(temp_dir, "vectors"),
                cache_directory=o