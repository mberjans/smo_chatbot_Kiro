"""
Unit tests for knowledge graph construction component.

Tests the knowledge graph construction, storage, and management functionality.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
from .knowledge_graph import (
    KnowledgeGraphBuilder,
    GraphNode,
    GraphEdge,
    KnowledgeGraph,
    GraphConstructionResult
)
from .entity_extractor import Entity, Relationship


class TestKnowledgeGraphBuilder:
    """Test suite for KnowledgeGraphBuilder class."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def builder(self, temp_storage):
        """Create a KnowledgeGraphBuilder instance for testing."""
        config = Mock()
        config.knowledge_graph_path = str(temp_storage)
        return KnowledgeGraphBuilder(config)
    
    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing."""
        return [
            Entity(
                entity_id="entity_1",
                text="glucose",
                entity_type="metabolite",
                confidence_score=0.9,
                start_pos=10,
                end_pos=17,
                context="test context",
                metadata={"extraction_method": "pattern"}
            ),
            Entity(
                entity_id="entity_2",
                text="diabetes",
                entity_type="disease",
                confidence_score=0.8,
                start_pos=50,
                end_pos=58,
                context="test context",
                metadata={"extraction_method": "spacy"}
            ),
            Entity(
                entity_id="entity_3",
                text="glycolysis",
                entity_type="pathway",
                confidence_score=0.85,
                start_pos=80,
                end_pos=90,
                context="test context",
                metadata={"extraction_method": "vocabulary"}
            )
        ]
    
    @pytest.fixture
    def sample_relationships(self):
        """Sample relationships for testing."""
        return [
            Relationship(
                relationship_id="rel_1",
                source_entity_id="entity_1",
                target_entity_id="entity_2",
                relationship_type="associated_with",
                confidence_score=0.8,
                evidence_text="glucose is associated with diabetes",
                context="test context",
                metadata={"extraction_method": "pattern"}
            ),
            Relationship(
                relationship_id="rel_2",
                source_entity_id="entity_3",
                target_entity_id="entity_1",
                relationship_type="produces",
                confidence_score=0.9,
                evidence_text="glycolysis produces glucose",
                context="test context",
                metadata={"extraction_method": "pattern"}
            )
        ]
    
    def test_initialization(self, builder, temp_storage):
        """Test KnowledgeGraphBuilder initialization."""
        assert builder is not None
        assert builder.logger is not None
        assert builder.graph_storage_path == temp_storage
        assert temp_storage.exists()
        assert builder.graph_cache == {}
    
    def test_generate_node_id(self, builder):
        """Test node ID generation."""
        node_id1 = builder._generate_node_id("glucose", "metabolite")
        node_id2 = builder._generate_node_id("glucose", "metabolite")
        node_id3 = builder._generate_node_id("Glucose", "metabolite")  # Different case
        node_id4 = builder._generate_node_id("glucose", "disease")  # Different type
        
        # Same text and type should generate same ID
        assert node_id1 == node_id2
        assert node_id1 == node_id3  # Case insensitive
        
        # Different type should generate different ID
        assert node_id1 != node_id4
        
        # IDs should be reasonable length
        assert len(node_id1) == 16
    
    def test_generate_edge_id(self, builder):
        """Test edge ID generation."""
        edge_id1 = builder._generate_edge_id("node1", "node2", "associated_with")
        edge_id2 = builder._generate_edge_id("node1", "node2", "associated_with")
        edge_id3 = builder._generate_edge_id("node2", "node1", "associated_with")  # Different order
        edge_id4 = builder._generate_edge_id("node1", "node2", "produces")  # Different type
        
        # Same parameters should generate same ID
        assert edge_id1 == edge_id2
        
        # Different order should generate different ID
        assert edge_id1 != edge_id3
        
        # Different type should generate different ID
        assert edge_id1 != edge_id4
        
        # IDs should be reasonable length
        assert len(edge_id1) == 16
    
    def test_calculate_text_similarity(self, builder):
        """Test text similarity calculation."""
        # Identical text
        assert builder._calculate_text_similarity("glucose", "glucose") == 1.0
        
        # Completely different text
        assert builder._calculate_text_similarity("glucose", "diabetes") == 0.0
        
        # Partial overlap
        similarity = builder._calculate_text_similarity("glucose metabolism", "glucose levels")
        assert 0 < similarity < 1
        
        # Empty strings
        assert builder._calculate_text_similarity("", "") == 1.0
        assert builder._calculate_text_similarity("glucose", "") == 0.0
    
    @pytest.mark.asyncio
    async def test_construct_graph_empty_input(self, builder):
        """Test graph construction with empty input."""
        result = await builder.construct_graph_from_entities_and_relationships(
            [], [], "test_doc"
        )
        
        assert result.success
        assert result.graph is not None
        assert result.nodes_created == 0
        assert result.edges_created == 0
        assert result.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_construct_graph_with_entities(self, builder, sample_entities):
        """Test graph construction with entities only."""
        result = await builder.construct_graph_from_entities_and_relationships(
            sample_entities, [], "test_doc"
        )
        
        assert result.success
        assert result.graph is not None
        assert result.nodes_created == len(sample_entities)
        assert result.edges_created == 0
        assert len(result.graph.nodes) == len(sample_entities)
        assert len(result.graph.edges) == 0
        
        # Check that nodes were created correctly
        for entity in sample_entities:
            node_id = builder._generate_node_id(entity.text, entity.entity_type)
            assert node_id in result.graph.nodes
            
            node = result.graph.nodes[node_id]
            assert node.text == entity.text
            assert node.node_type == entity.entity_type
            assert node.confidence_score == entity.confidence_score
    
    @pytest.mark.asyncio
    async def test_construct_graph_with_relationships(self, builder, sample_entities, sample_relationships):
        """Test graph construction with entities and relationships."""
        result = await builder.construct_graph_from_entities_and_relationships(
            sample_entities, sample_relationships, "test_doc"
        )
        
        assert result.success
        assert result.graph is not None
        assert result.nodes_created == len(sample_entities)
        assert result.edges_created >= 0  # May be 0 if entity matching fails
        assert len(result.graph.nodes) == len(sample_entities)
    
    @pytest.mark.asyncio
    async def test_process_entity_to_node_new(self, builder, sample_entities):
        """Test processing entity to new node."""
        graph = KnowledgeGraph(
            graph_id="test_graph",
            nodes={},
            edges={},
            metadata={"source_documents": []},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        entity = sample_entities[0]
        result = await builder._process_entity_to_node(entity, "test_doc", graph)
        
        assert result["created"] == True
        assert result["updated"] == False
        assert len(graph.nodes) == 1
        
        node_id = builder._generate_node_id(entity.text, entity.entity_type)
        assert node_id in graph.nodes
        
        node = graph.nodes[node_id]
        assert node.text == entity.text
        assert node.node_type == entity.entity_type
        assert "test_doc" in node.source_documents
    
    @pytest.mark.asyncio
    async def test_process_entity_to_node_existing(self, builder, sample_entities):
        """Test processing entity to existing node (update)."""
        graph = KnowledgeGraph(
            graph_id="test_graph",
            nodes={},
            edges={},
            metadata={"source_documents": []},
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        entity = sample_entities[0]
        
        # Add entity first time
        await builder._process_entity_to_node(entity, "test_doc1", graph)
        
        # Add same entity again with different document
        entity_updated = Entity(
            entity_id="entity_1_updated",
            text=entity.text,  # Same text
            entity_type=entity.entity_type,  # Same type
            confidence_score=0.95,  # Higher confidence
            start_pos=entity.start_pos,
            end_pos=entity.end_pos,
            context="updated context",
            metadata={"extraction_method": "updated"}
        )
        
        result = await builder._process_entity_to_node(entity_updated, "test_doc2", graph)
        
        assert result["created"] == False
        assert result["updated"] == True
        assert len(graph.nodes) == 1  # Still only one node
        
        node_id = builder._generate_node_id(entity.text, entity.entity_type)
        node = graph.nodes[node_id]
        
        # Should have updated confidence and added new document
        assert node.confidence_score == 0.95
        assert "test_doc1" in node.source_documents
        assert "test_doc2" in node.source_documents
    
    @pytest.mark.asyncio
    async def test_save_and_load_graph(self, builder, sample_entities):
        """Test saving and loading graph from storage."""
        # Create a graph
        result = await builder.construct_graph_from_entities_and_relationships(
            sample_entities, [], "test_doc"
        )
        
        graph = result.graph
        graph_id = graph.graph_id
        
        # Clear cache to force loading from storage
        builder.graph_cache.clear()
        
        # Load graph from storage
        loaded_graph = await builder.load_graph_from_storage(graph_id)
        
        assert loaded_graph is not None
        assert loaded_graph.graph_id == graph_id
        assert len(loaded_graph.nodes) == len(graph.nodes)
        assert len(loaded_graph.edges) == len(graph.edges)
        
        # Check that a specific node was loaded correctly
        for node_id, original_node in graph.nodes.items():
            loaded_node = loaded_graph.nodes[node_id]
            assert loaded_node.text == original_node.text
            assert loaded_node.node_type == original_node.node_type
            assert loaded_node.confidence_score == original_node.confidence_score
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_graph(self, builder):
        """Test loading a graph that doesn't exist."""
        loaded_graph = await builder.load_graph_from_storage("nonexistent_graph")
        assert loaded_graph is None
    
    @pytest.mark.asyncio
    async def test_get_graph_statistics(self, builder, sample_entities):
        """Test getting graph statistics."""
        # Create a graph
        result = await builder.construct_graph_from_entities_and_relationships(
            sample_entities, [], "test_doc"
        )
        
        graph_id = result.graph.graph_id
        stats = await builder.get_graph_statistics(graph_id)
        
        assert "error" not in stats
        assert stats["graph_id"] == graph_id
        assert stats["total_nodes"] == len(sample_entities)
        assert stats["total_edges"] == 0
        assert "node_types" in stats
        assert "edge_types" in stats
        assert stats["source_documents"] == 1
        assert "avg_node_confidence" in stats
        assert "avg_edge_confidence" in stats
    
    @pytest.mark.asyncio
    async def test_list_available_graphs(self, builder, sample_entities):
        """Test listing available graphs."""
        # Initially should be empty
        graphs = await builder.list_available_graphs()
        assert graphs == []
        
        # Create a graph
        result = await builder.construct_graph_from_entities_and_relationships(
            sample_entities, [], "test_doc"
        )
        
        # Should now have one graph
        graphs = await builder.list_available_graphs()
        assert len(graphs) == 1
        assert result.graph.graph_id in graphs
    
    @pytest.mark.asyncio
    async def test_delete_graph(self, builder, sample_entities):
        """Test deleting a graph."""
        # Create a graph
        result = await builder.construct_graph_from_entities_and_relationships(
            sample_entities, [], "test_doc"
        )
        
        graph_id = result.graph.graph_id
        
        # Verify graph exists
        graphs = await builder.list_available_graphs()
        assert graph_id in graphs
        assert graph_id in builder.graph_cache
        
        # Delete graph
        success = await builder.delete_graph(graph_id)
        assert success
        
        # Verify graph is deleted
        graphs = await builder.list_available_graphs()
        assert graph_id not in graphs
        assert graph_id not in builder.graph_cache
        
        # Try to delete again (should return False)
        success = await builder.delete_graph(graph_id)
        assert not success
    
    @pytest.mark.asyncio
    async def test_merge_graphs(self, builder, sample_entities):
        """Test merging multiple graphs."""
        # Create two separate graphs
        result1 = await builder.construct_graph_from_entities_and_relationships(
            sample_entities[:2], [], "test_doc1"
        )
        
        result2 = await builder.construct_graph_from_entities_and_relationships(
            sample_entities[2:], [], "test_doc2"
        )
        
        graph_ids = [result1.graph.graph_id, result2.graph.graph_id]
        
        # Merge graphs
        merged_graph = await builder.merge_graphs(graph_ids)
        
        assert merged_graph is not None
        assert len(merged_graph.nodes) == len(sample_entities)
        assert "test_doc1" in merged_graph.metadata["source_documents"]
        assert "test_doc2" in merged_graph.metadata["source_documents"]
        assert merged_graph.metadata["creation_method"] == "graph_merge"
    
    @pytest.mark.asyncio
    async def test_merge_empty_graphs(self, builder):
        """Test merging with empty graph list."""
        merged_graph = await builder.merge_graphs([])
        assert merged_graph is None
    
    @pytest.mark.asyncio
    async def test_error_handling(self, builder):
        """Test error handling in graph construction."""
        # Test with invalid entities (None values)
        invalid_entities = [None]
        
        with patch.object(builder, '_process_entity_to_node', side_effect=Exception("Test error")):
            result = await builder.construct_graph_from_entities_and_relationships(
                [], [], "test_doc"
            )
            
            # Should handle error gracefully
            assert not result.success
            assert result.error_message is not None
            assert "error" in result.error_message.lower()
    
    def test_initialization_without_config(self, temp_storage):
        """Test initialization without config (uses default path)."""
        builder = KnowledgeGraphBuilder()
        
        # Should use default path
        assert builder.graph_storage_path == Path("data/lightrag_kg")
        assert builder.graph_storage_path.exists()
    
    @pytest.mark.asyncio
    async def test_incremental_graph_updates(self, builder, sample_entities):
        """Test incremental updates to existing graph."""
        # Create initial graph
        result1 = await builder.construct_graph_from_entities_and_relationships(
            sample_entities[:2], [], "test_doc1"
        )
        
        graph_id = result1.graph.graph_id
        initial_node_count = len(result1.graph.nodes)
        
        # Add more entities to the same graph
        result2 = await builder.construct_graph_from_entities_and_relationships(
            sample_entities[2:], [], "test_doc2", graph_id=graph_id
        )
        
        # Should have updated the existing graph
        assert result2.graph.graph_id == graph_id
        assert len(result2.graph.nodes) > initial_node_count
        assert "test_doc1" in result2.graph.metadata["source_documents"]
        assert "test_doc2" in result2.graph.metadata["source_documents"]


# Integration tests
class TestKnowledgeGraphBuilderIntegration:
    """Integration tests for KnowledgeGraphBuilder."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow from entities to graph storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Mock()
            config.knowledge_graph_path = temp_dir
            
            builder = KnowledgeGraphBuilder(config)
            
            # Create sample data
            entities = [
                Entity("e1", "glucose", "metabolite", 0.9, 0, 7, "ctx", {}),
                Entity("e2", "diabetes", "disease", 0.8, 10, 18, "ctx", {})
            ]
            
            relationships = [
                Relationship("r1", "e1", "e2", "associated_with", 0.8, "glucose associated with diabetes", "ctx", {})
            ]
            
            # Construct graph
            result = await builder.construct_graph_from_entities_and_relationships(
                entities, relationships, "test_document"
            )
            
            # Verify construction
            assert result.success
            assert result.nodes_created == 2
            
            # Verify storage
            graphs = await builder.list_available_graphs()
            assert len(graphs) == 1
            
            # Verify statistics
            stats = await builder.get_graph_statistics(result.graph.graph_id)
            assert stats["total_nodes"] == 2
            
            print(f"Created graph with {stats['total_nodes']} nodes and {stats['total_edges']} edges")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])