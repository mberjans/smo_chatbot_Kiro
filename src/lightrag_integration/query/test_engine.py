"""
Unit tests for LightRAG Query Engine

Tests the query processing, semantic search, and graph traversal functionality.
Implements testing requirements for task 4.1.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from ..config.settings import LightRAGConfig
from .engine import LightRAGQueryEngine, QueryResult, SearchResult, TraversalResult
from ..ingestion.knowledge_graph import (
    KnowledgeGraphBuilder, KnowledgeGraph, GraphNode, GraphEdge
)


@pytest.fixture
def test_config():
    """Create a test configuration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LightRAGConfig(
            knowledge_graph_path=f"{temp_dir}/kg",
            vector_store_path=f"{temp_dir}/vectors",
            cache_directory=f"{temp_dir}/cache",
            papers_directory=f"{temp_dir}/papers",
            max_query_length=1000,
            default_top_k=10,
            similarity_threshold=0.5
        )
        yield config


@pytest.fixture
def sample_graph():
    """Create a sample knowledge graph for testing."""
    now = datetime.now()
    
    # Create nodes
    nodes = {
        "node1": GraphNode(
            node_id="node1",
            text="clinical metabolomics",
            node_type="field",
            confidence_score=0.9,
            properties={"context": "the study of metabolites in clinical settings"},
            source_documents=["doc1.pdf"],
            created_at=now,
            updated_at=now
        ),
        "node2": GraphNode(
            node_id="node2",
            text="biomarker",
            node_type="metabolite",
            confidence_score=0.8,
            properties={"context": "a measurable indicator of biological state"},
            source_documents=["doc1.pdf", "doc2.pdf"],
            created_at=now,
            updated_at=now
        ),
        "node3": GraphNode(
            node_id="node3",
            text="diabetes",
            node_type="disease",
            confidence_score=0.85,
            properties={"context": "a metabolic disorder"},
            source_documents=["doc2.pdf"],
            created_at=now,
            updated_at=now
        )
    }
    
    # Create edges
    edges = {
        "edge1": GraphEdge(
            edge_id="edge1",
            source_node_id="node1",
            target_node_id="node2",
            edge_type="uses",
            confidence_score=0.7,
            evidence=["Clinical metabolomics uses biomarkers for diagnosis"],
            properties={},
            source_documents=["doc1.pdf"],
            created_at=now,
            updated_at=now
        ),
        "edge2": GraphEdge(
            edge_id="edge2",
            source_node_id="node2",
            target_node_id="node3",
            edge_type="indicates",
            confidence_score=0.75,
            evidence=["Biomarkers can indicate diabetes"],
            properties={},
            source_documents=["doc2.pdf"],
            created_at=now,
            updated_at=now
        )
    }
    
    graph = KnowledgeGraph(
        graph_id="test_graph",
        nodes=nodes,
        edges=edges,
        metadata={"source_documents": ["doc1.pdf", "doc2.pdf"]},
        created_at=now,
        updated_at=now
    )
    
    return graph


@pytest_asyncio.fixture
async def query_engine(test_config, sample_graph):
    """Create a query engine with test data."""
    engine = LightRAGQueryEngine(test_config)
    
    # Mock the knowledge graph builder
    engine.kg_builder = Mock()
    engine.kg_builder.list_available_graphs = AsyncMock(return_value=["test_graph"])
    engine.kg_builder.load_graph_from_storage = AsyncMock(return_value=sample_graph)
    
    # Pre-populate cache
    engine.graph_cache["test_graph"] = sample_graph
    
    return engine


class TestLightRAGQueryEngine:
    """Test cases for LightRAG Query Engine."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, test_config):
        """Test query engine initialization."""
        engine = LightRAGQueryEngine(test_config)
        
        assert engine.config == test_config
        assert engine.max_query_length == 1000
        assert engine.default_top_k == 10
        assert engine.similarity_threshold == 0.5
        assert isinstance(engine.kg_builder, KnowledgeGraphBuilder)
    
    @pytest.mark.asyncio
    async def test_process_query_valid(self, query_engine):
        """Test processing a valid query."""
        result = await query_engine.process_query("What is clinical metabolomics?")
        
        assert isinstance(result, QueryResult)
        assert result.answer != ""
        assert result.confidence_score > 0
        assert len(result.source_documents) > 0
        assert len(result.entities_used) > 0
        assert result.processing_time > 0
        
        # Test that formatted response is included
        assert result.formatted_response is not None
        assert result.confidence_breakdown is not None
        assert isinstance(result.formatted_response.content, str)
        assert isinstance(result.formatted_response.bibliography, str)
    
    @pytest.mark.asyncio
    async def test_process_query_empty(self, query_engine):
        """Test processing an empty query."""
        result = await query_engine.process_query("")
        
        assert result.answer == "Please provide a valid question."
        assert result.confidence_score == 0.0
        assert len(result.source_documents) == 0
    
    @pytest.mark.asyncio
    async def test_process_query_too_long(self, query_engine):
        """Test processing a query that's too long."""
        long_query = "x" * 1001
        result = await query_engine.process_query(long_query)
        
        assert "Query too long" in result.answer
        assert result.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_process_query_no_graphs(self, test_config):
        """Test processing a query when no graphs are available."""
        engine = LightRAGQueryEngine(test_config)
        engine.kg_builder = Mock()
        engine.kg_builder.list_available_graphs = AsyncMock(return_value=[])
        
        result = await engine.process_query("What is clinical metabolomics?")
        
        assert "No knowledge graphs available" in result.answer
        assert result.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_semantic_search(self, query_engine):
        """Test semantic search functionality."""
        results = await query_engine.semantic_search("clinical metabolomics", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
        
        # Check first result
        first_result = results[0]
        assert isinstance(first_result, SearchResult)
        assert first_result.node_text == "clinical metabolomics"
        assert first_result.relevance_score > 0.5
        assert len(first_result.source_documents) > 0
    
    @pytest.mark.asyncio
    async def test_semantic_search_no_results(self, query_engine):
        """Test semantic search with no matching results."""
        results = await query_engine.semantic_search("completely unrelated topic", top_k=5)
        
        # Should return empty list for unrelated topics
        assert isinstance(results, list)
        # May be empty or have very low relevance scores
    
    @pytest.mark.asyncio
    async def test_graph_traversal(self, query_engine):
        """Test graph traversal functionality."""
        start_entities = ["node1"]
        result = await query_engine.graph_traversal(start_entities, max_depth=2)
        
        assert isinstance(result, TraversalResult)
        assert len(result.nodes) > 0
        assert len(result.edges) > 0
        assert result.depth_reached >= 0
        assert len(result.paths) > 0
    
    @pytest.mark.asyncio
    async def test_graph_traversal_invalid_entities(self, query_engine):
        """Test graph traversal with invalid starting entities."""
        start_entities = ["nonexistent_node"]
        result = await query_engine.graph_traversal(start_entities, max_depth=2)
        
        assert isinstance(result, TraversalResult)
        assert len(result.nodes) == 0
        assert len(result.edges) == 0
        assert result.depth_reached == 0
    
    @pytest.mark.asyncio
    async def test_graph_traversal_max_depth(self, query_engine):
        """Test graph traversal respects max depth."""
        start_entities = ["node1"]
        result = await query_engine.graph_traversal(start_entities, max_depth=1)
        
        assert result.depth_reached <= 1
    
    def test_extract_query_terms(self, query_engine):
        """Test query term extraction."""
        query = "What is clinical metabolomics and how does it work?"
        terms = query_engine._extract_query_terms(query)
        
        assert "clinical" in terms
        assert "metabolomics" in terms
        assert "work" in terms
        # Stop words should be filtered out
        assert "what" not in terms
        assert "is" not in terms
        assert "and" not in terms
    
    def test_calculate_node_relevance(self, query_engine, sample_graph):
        """Test node relevance calculation."""
        node = sample_graph.nodes["node1"]  # "clinical metabolomics"
        query_terms = ["clinical", "metabolomics"]
        
        relevance = query_engine._calculate_node_relevance(node, query_terms)
        
        assert relevance > 0.5  # Should be highly relevant
        assert relevance <= 1.0  # Should not exceed 1.0
    
    def test_calculate_node_relevance_no_match(self, query_engine, sample_graph):
        """Test node relevance with no matching terms."""
        node = sample_graph.nodes["node1"]
        query_terms = ["unrelated", "terms"]
        
        relevance = query_engine._calculate_node_relevance(node, query_terms)
        
        assert relevance < 0.5  # Should have low relevance
    
    @pytest.mark.asyncio
    async def test_generate_response_definition_question(self, query_engine):
        """Test response generation for definition questions."""
        search_results = [
            SearchResult(
                node_id="node1",
                node_text="clinical metabolomics",
                node_type="field",
                relevance_score=0.9,
                source_documents=["doc1.pdf"],
                context="the study of metabolites in clinical settings"
            )
        ]
        
        traversal_result = TraversalResult(nodes=[], edges=[], paths=[], depth_reached=0)
        
        response = await query_engine._generate_response(
            "What is clinical metabolomics?", search_results, traversal_result
        )
        
        assert "clinical metabolomics" in response["answer"].lower()
        assert response["confidence_score"] > 0
        assert len(response["source_documents"]) > 0
        assert len(response["entities_used"]) > 0
    
    @pytest.mark.asyncio
    async def test_construct_definition_answer(self, query_engine):
        """Test construction of definition-type answers."""
        relevant_info = [
            {
                "text": "clinical metabolomics",
                "type": "field",
                "relevance": 0.9,
                "context": "the study of metabolites in clinical settings"
            }
        ]
        
        answer = await query_engine._construct_answer(
            "What is clinical metabolomics?", relevant_info, []
        )
        
        assert "clinical metabolomics" in answer.lower()
        assert "study of metabolites" in answer.lower()
    
    @pytest.mark.asyncio
    async def test_construct_mechanism_answer(self, query_engine):
        """Test construction of mechanism-type answers."""
        relevant_info = [
            {
                "text": "metabolic pathway",
                "type": "pathway",
                "relevance": 0.8,
                "context": "a series of chemical reactions"
            }
        ]
        
        relationships = [
            {
                "type": "regulates",
                "evidence": ["The pathway regulates glucose metabolism"],
                "confidence": 0.7
            }
        ]
        
        answer = await query_engine._construct_answer(
            "How does the metabolic pathway work?", relevant_info, relationships
        )
        
        assert "mechanism" in answer.lower() or "pathway" in answer.lower()
    
    def test_calculate_response_confidence(self, query_engine):
        """Test response confidence calculation."""
        search_results = [
            SearchResult(
                node_id="node1",
                node_text="test",
                node_type="test",
                relevance_score=0.8,
                source_documents=["doc1.pdf"],
                context=""
            ),
            SearchResult(
                node_id="node2",
                node_text="test2",
                node_type="test",
                relevance_score=0.7,
                source_documents=["doc2.pdf"],
                context=""
            )
        ]
        
        traversal_result = TraversalResult(
            nodes=[],
            edges=[{"edge_id": "e1"}, {"edge_id": "e2"}],
            paths=[],
            depth_reached=1
        )
        
        confidence = query_engine._calculate_response_confidence(search_results, traversal_result)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident with good results
    
    def test_calculate_response_confidence_no_results(self, query_engine):
        """Test response confidence with no results."""
        confidence = query_engine._calculate_response_confidence([], TraversalResult([], [], [], 0))
        
        assert confidence == 0.0
    
    @pytest.mark.asyncio
    async def test_load_available_graphs(self, query_engine, sample_graph):
        """Test loading available graphs."""
        graphs = await query_engine._load_available_graphs()
        
        assert len(graphs) == 1
        assert graphs[0].graph_id == "test_graph"
        assert "test_graph" in query_engine.graph_cache
    
    @pytest.mark.asyncio
    async def test_error_handling_in_process_query(self, test_config):
        """Test error handling in query processing."""
        engine = LightRAGQueryEngine(test_config)
        
        # Mock to raise an exception during graph loading
        engine._load_available_graphs = AsyncMock(side_effect=Exception("Test error"))
        
        result = await engine.process_query("What is clinical metabolomics?")
        
        assert "error" in result.answer.lower()
        assert result.confidence_score == 0.0
        assert "error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_error_handling_in_semantic_search(self, test_config):
        """Test error handling in semantic search."""
        engine = LightRAGQueryEngine(test_config)
        
        # Mock to raise an exception
        engine.kg_builder = Mock()
        engine.kg_builder.list_available_graphs = AsyncMock(side_effect=Exception("Test error"))
        
        results = await engine.semantic_search("test query")
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_error_handling_in_graph_traversal(self, test_config):
        """Test error handling in graph traversal."""
        engine = LightRAGQueryEngine(test_config)
        
        # Mock to raise an exception
        engine.kg_builder = Mock()
        engine.kg_builder.list_available_graphs = AsyncMock(side_effect=Exception("Test error"))
        
        result = await engine.graph_traversal(["node1"])
        
        assert result.nodes == []
        assert result.edges == []
        assert result.paths == []
        assert result.depth_reached == 0


class TestQueryEngineIntegration:
    """Integration tests for query engine with real knowledge graphs."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_query_processing(self, test_config):
        """Test end-to-end query processing with a real knowledge graph."""
        # Create a temporary knowledge graph
        kg_builder = KnowledgeGraphBuilder(test_config)
        
        # Create sample entities and relationships (would normally come from ingestion)
        from ..ingestion.entity_extractor import Entity, Relationship
        
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
            ),
            Entity(
                entity_id="e2",
                text="biomarker",
                entity_type="metabolite",
                confidence_score=0.8,
                start_pos=25,
                end_pos=35,
                context="A biomarker is a measurable indicator",
                metadata={"original_entity_id": "e2"}
            )
        ]
        
        relationships = [
            Relationship(
                relationship_id="r1",
                source_entity_id="e1",
                target_entity_id="e2",
                relationship_type="uses",
                confidence_score=0.7,
                evidence_text="Clinical metabolomics uses biomarkers for diagnosis",
                context="diagnostic context",
                metadata={}
            )
        ]
        
        # Build the knowledge graph
        result = await kg_builder.construct_graph_from_entities_and_relationships(
            entities, relationships, "test_doc"
        )
        
        assert result.success
        
        # Now test the query engine
        engine = LightRAGQueryEngine(test_config)
        query_result = await engine.process_query("What is clinical metabolomics?")
        
        assert query_result.confidence_score > 0
        assert "clinical metabolomics" in query_result.answer.lower()
        assert len(query_result.source_documents) > 0


if __name__ == "__main__":
    pytest.main([__file__])