"""
Test the main component query functionality with the new query engine.
"""

import pytest
import tempfile
from pathlib import Path

from .config.settings import LightRAGConfig
from .component import LightRAGComponent
from .ingestion.knowledge_graph import KnowledgeGraphBuilder
from .ingestion.entity_extractor import Entity, Relationship


@pytest.fixture
def test_config():
    """Create a test configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LightRAGConfig(
            knowledge_graph_path=f"{temp_dir}/kg",
            vector_store_path=f"{temp_dir}/vectors",
            cache_directory=f"{temp_dir}/cache",
            papers_directory=f"{temp_dir}/papers"
        )
        yield config


@pytest.mark.asyncio
async def test_component_query_integration(test_config):
    """Test the main component query functionality."""
    # Create a knowledge graph first
    kg_builder = KnowledgeGraphBuilder(test_config)
    
    entities = [
        Entity(
            entity_id="e1",
            text="clinical metabolomics",
            entity_type="field",
            confidence_score=0.9,
            start_pos=0,
            end_pos=20,
            context="Clinical metabolomics is the study of metabolites",
            metadata={"original_entity_id": "e1"}
        )
    ]
    
    relationships = []
    
    # Build the knowledge graph
    result = await kg_builder.construct_graph_from_entities_and_relationships(
        entities, relationships, "test_doc"
    )
    
    assert result.success
    
    # Now test the component
    component = LightRAGComponent(test_config)
    await component.initialize()
    
    response = await component.query("What is clinical metabolomics?")
    
    # Verify response structure
    assert "answer" in response
    assert "confidence_score" in response
    assert "source_documents" in response
    assert "entities_used" in response
    assert "relationships_used" in response
    assert "processing_time" in response
    assert "metadata" in response
    assert "formatted_response" in response
    assert "confidence_breakdown" in response
    
    # Verify response content
    assert response["confidence_score"] > 0
    assert "clinical metabolomics" in response["answer"].lower()
    assert response["formatted_response"] is not None
    assert response["confidence_breakdown"] is not None


if __name__ == "__main__":
    pytest.main([__file__])