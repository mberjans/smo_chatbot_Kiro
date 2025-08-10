"""
Unit tests for the query classifier.

This module tests the LLM-based query classification functionality
to ensure accurate routing decisions.
"""

import asyncio
import json
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from llama_index.core.llms import CompletionResponse
from llama_index.core.llms.mock import MockLLM

from .classifier import QueryClassifier, QueryClassification, QueryType
from ..config.settings import LightRAGConfig


class TestQueryClassifier:
    """Test cases for QueryClassifier."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=LightRAGConfig)
        config.cache_directory = "/tmp/test_cache"
        config.max_concurrent_requests = 5
        return config
    
    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        return MockLLM(max_tokens=1000)
    
    @pytest.fixture
    def classifier(self, mock_llm, mock_config):
        """Create a QueryClassifier instance."""
        with patch('src.lightrag_integration.routing.classifier.setup_logger'):
            return QueryClassifier(mock_llm, mock_config)
    
    def create_mock_llm_response(self, query_type: str, confidence: float, reasoning: str):
        """Create a mock LLM response."""
        response_data = {
            "query_type": query_type,
            "confidence_score": confidence,
            "reasoning": reasoning,
            "suggested_sources": ["research_papers", "textbooks"],
            "keywords": ["metabolomics", "clinical"],
            "temporal_indicators": [],
            "domain_specificity": 0.9
        }
        return json.dumps(response_data)
    
    @pytest.mark.asyncio
    async def test_classify_knowledge_base_query(self, classifier):
        """Test classification of knowledge base queries."""
        # Mock LLM response
        mock_response = self.create_mock_llm_response(
            "KNOWLEDGE_BASE", 0.9, "Query asks for established scientific definition"
        )
        
        # Create a mock LLM that returns our desired response
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(return_value=CompletionResponse(text=mock_response))
        classifier.llm = mock_llm
        
        query = "What is clinical metabolomics?"
        result = await classifier.classify_query(query)
        
        assert isinstance(result, QueryClassification)
        assert result.query_type == QueryType.KNOWLEDGE_BASE
        assert result.confidence_score == 0.9
        assert "established scientific definition" in result.reasoning
        assert result.processing_time > 0
        assert "metabolomics" in result.metadata["keywords"]
    
    @pytest.mark.asyncio
    async def test_classify_real_time_query(self, classifier):
        """Test classification of real-time queries."""
        mock_response = self.create_mock_llm_response(
            "REAL_TIME", 0.85, "Query contains temporal indicators for recent information"
        )
        
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(return_value=CompletionResponse(text=mock_response))
        classifier.llm = mock_llm
        
        query = "What are the latest developments in metabolomics research in 2024?"
        result = await classifier.classify_query(query)
        
        assert result.query_type == QueryType.REAL_TIME
        assert result.confidence_score == 0.85
        assert "temporal indicators" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_classify_hybrid_query(self, classifier):
        """Test classification of hybrid queries."""
        mock_response = self.create_mock_llm_response(
            "HYBRID", 0.75, "Query requires both established knowledge and recent information"
        )
        
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(return_value=CompletionResponse(text=mock_response))
        classifier.llm = mock_llm
        
        query = "How has clinical metabolomics evolved in recent years?"
        result = await classifier.classify_query(query)
        
        assert result.query_type == QueryType.HYBRID
        assert result.confidence_score == 0.75
        assert "established knowledge and recent information" in result.reasoning
    
    @pytest.mark.asyncio
    async def test_classify_with_context(self, classifier):
        """Test classification with context information."""
        mock_response = self.create_mock_llm_response(
            "KNOWLEDGE_BASE", 0.8, "Context indicates focus on established concepts"
        )
        
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(return_value=CompletionResponse(text=mock_response))
        classifier.llm = mock_llm
        
        query = "Explain biomarker discovery"
        context = {"user_type": "researcher", "previous_queries": ["metabolomics basics"]}
        
        result = await classifier.classify_query(query, context)
        
        assert result.query_type == QueryType.KNOWLEDGE_BASE
        assert result.metadata["context_provided"] is True
    
    @pytest.mark.asyncio
    async def test_empty_query_raises_error(self, classifier):
        """Test that empty queries raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await classifier.classify_query("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await classifier.classify_query("   ")
    
    @pytest.mark.asyncio
    async def test_invalid_llm_response_fallback(self, classifier):
        """Test fallback behavior when LLM returns invalid response."""
        # Mock invalid JSON response
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(return_value=CompletionResponse(text="Invalid JSON"))
        classifier.llm = mock_llm
        
        query = "What is metabolomics?"
        result = await classifier.classify_query(query)
        
        # Should return fallback classification
        assert isinstance(result, QueryClassification)
        assert result.confidence_score == 0.3  # Low confidence for fallback
        assert result.metadata["fallback_classification"] is True
        assert "error_reason" in result.metadata
    
    @pytest.mark.asyncio
    async def test_llm_error_fallback(self, classifier):
        """Test fallback behavior when LLM raises exception."""
        # Mock LLM exception
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(side_effect=Exception("LLM service unavailable"))
        classifier.llm = mock_llm
        
        query = "Recent metabolomics trends"
        result = await classifier.classify_query(query)
        
        # Should return fallback classification with temporal detection
        assert result.query_type == QueryType.REAL_TIME  # Due to "Recent" keyword
        assert result.confidence_score == 0.3
        assert result.metadata["fallback_classification"] is True
        assert result.metadata["has_temporal_indicators"] is True
    
    @pytest.mark.asyncio
    async def test_batch_classify(self, classifier):
        """Test batch classification of multiple queries."""
        # Mock responses for different query types
        responses = [
            self.create_mock_llm_response("KNOWLEDGE_BASE", 0.9, "Definition query"),
            self.create_mock_llm_response("REAL_TIME", 0.8, "Recent information query"),
            self.create_mock_llm_response("HYBRID", 0.7, "Complex query")
        ]
        
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(side_effect=[
            CompletionResponse(text=resp) for resp in responses
        ])
        classifier.llm = mock_llm
        
        queries = [
            "What is metabolomics?",
            "Latest metabolomics news",
            "How has the field evolved?"
        ]
        
        results = await classifier.batch_classify(queries)
        
        assert len(results) == 3
        assert results[0].query_type == QueryType.KNOWLEDGE_BASE
        assert results[1].query_type == QueryType.REAL_TIME
        assert results[2].query_type == QueryType.HYBRID
    
    @pytest.mark.asyncio
    async def test_batch_classify_with_contexts(self, classifier):
        """Test batch classification with context information."""
        mock_response = self.create_mock_llm_response("KNOWLEDGE_BASE", 0.8, "Test")
        
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(return_value=CompletionResponse(text=mock_response))
        classifier.llm = mock_llm
        
        queries = ["Query 1", "Query 2"]
        contexts = [{"type": "basic"}, {"type": "advanced"}]
        
        results = await classifier.batch_classify(queries, contexts)
        
        assert len(results) == 2
        assert all(result.metadata["context_provided"] for result in results)
    
    @pytest.mark.asyncio
    async def test_batch_classify_mismatched_contexts(self, classifier):
        """Test batch classification with mismatched contexts length."""
        queries = ["Query 1", "Query 2"]
        contexts = [{"type": "basic"}]  # Only one context for two queries
        
        with pytest.raises(ValueError, match="Contexts list must match queries list length"):
            await classifier.batch_classify(queries, contexts)
    
    @pytest.mark.asyncio
    async def test_batch_classify_with_errors(self, classifier):
        """Test batch classification handling individual query errors."""
        # First query succeeds, second fails
        mock_response = self.create_mock_llm_response("KNOWLEDGE_BASE", 0.8, "Success")
        
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(side_effect=[
            CompletionResponse(text=mock_response),
            Exception("LLM error")
        ])
        classifier.llm = mock_llm
        
        queries = ["Good query", "Bad query"]
        results = await classifier.batch_classify(queries)
        
        assert len(results) == 2
        assert results[0].query_type == QueryType.KNOWLEDGE_BASE
        assert results[1].metadata["fallback_classification"] is True
    
    def test_get_classification_stats(self, classifier):
        """Test getting classification statistics."""
        stats = classifier.get_classification_stats()
        
        assert "classifications_performed" in stats
        assert "knowledge_base_queries" in stats
        assert "real_time_queries" in stats
        assert "hybrid_queries" in stats
        assert "confidence_thresholds" in stats
        assert "query_type_distribution" in stats
    
    def test_update_confidence_thresholds(self, classifier):
        """Test updating confidence thresholds."""
        new_thresholds = {
            "high_confidence": 0.85,
            "medium_confidence": 0.65
        }
        
        classifier.update_confidence_thresholds(new_thresholds)
        
        assert classifier.confidence_thresholds["high_confidence"] == 0.85
        assert classifier.confidence_thresholds["medium_confidence"] == 0.65
        # Should preserve existing thresholds not updated
        assert "low_confidence" in classifier.confidence_thresholds
    
    def test_parse_llm_response_valid_json(self, classifier):
        """Test parsing valid LLM response."""
        response_data = {
            "query_type": "KNOWLEDGE_BASE",
            "confidence_score": 0.9,
            "reasoning": "Test reasoning",
            "suggested_sources": ["papers"],
            "keywords": ["test"]
        }
        response_text = json.dumps(response_data)
        
        parsed = classifier._parse_llm_response(response_text)
        
        assert parsed["query_type"] == "KNOWLEDGE_BASE"
        assert parsed["confidence_score"] == 0.9
        assert parsed["reasoning"] == "Test reasoning"
    
    def test_parse_llm_response_embedded_json(self, classifier):
        """Test parsing LLM response with embedded JSON."""
        response_text = """
        Here is my analysis:
        {
            "query_type": "REAL_TIME",
            "confidence_score": 0.8,
            "reasoning": "Contains temporal indicators"
        }
        That's my classification.
        """
        
        parsed = classifier._parse_llm_response(response_text)
        
        assert parsed["query_type"] == "REAL_TIME"
        assert parsed["confidence_score"] == 0.8
    
    def test_parse_llm_response_invalid_json(self, classifier):
        """Test parsing invalid LLM response."""
        with pytest.raises(ValueError, match="Invalid LLM response format"):
            classifier._parse_llm_response("Not valid JSON")
    
    def test_parse_llm_response_missing_fields(self, classifier):
        """Test parsing LLM response with missing required fields."""
        response_data = {
            "query_type": "KNOWLEDGE_BASE",
            # Missing confidence_score and reasoning
        }
        response_text = json.dumps(response_data)
        
        with pytest.raises(ValueError, match="Missing required field"):
            classifier._parse_llm_response(response_text)
    
    def test_parse_llm_response_invalid_query_type(self, classifier):
        """Test parsing LLM response with invalid query type."""
        response_data = {
            "query_type": "INVALID_TYPE",
            "confidence_score": 0.8,
            "reasoning": "Test"
        }
        response_text = json.dumps(response_data)
        
        with pytest.raises(ValueError, match="Invalid query_type"):
            classifier._parse_llm_response(response_text)
    
    def test_parse_llm_response_invalid_confidence(self, classifier):
        """Test parsing LLM response with invalid confidence score."""
        response_data = {
            "query_type": "KNOWLEDGE_BASE",
            "confidence_score": 1.5,  # Invalid: > 1.0
            "reasoning": "Test"
        }
        response_text = json.dumps(response_data)
        
        with pytest.raises(ValueError, match="Invalid confidence_score"):
            classifier._parse_llm_response(response_text)
    
    def test_create_fallback_classification_temporal(self, classifier):
        """Test fallback classification with temporal indicators."""
        query = "What are the latest trends in metabolomics?"
        fallback = classifier._create_fallback_classification(query, 0.1, "Test error")
        
        assert fallback.query_type == QueryType.REAL_TIME
        assert fallback.confidence_score == 0.3
        assert fallback.metadata["fallback_classification"] is True
        assert fallback.metadata["has_temporal_indicators"] is True
        assert "Test error" in fallback.reasoning
    
    def test_create_fallback_classification_no_temporal(self, classifier):
        """Test fallback classification without temporal indicators."""
        query = "What is metabolomics?"
        fallback = classifier._create_fallback_classification(query, 0.1, "Test error")
        
        assert fallback.query_type == QueryType.KNOWLEDGE_BASE
        assert fallback.confidence_score == 0.3
        assert fallback.metadata["fallback_classification"] is True
        assert fallback.metadata["has_temporal_indicators"] is False
    
    @pytest.mark.asyncio
    async def test_stats_update_after_classification(self, classifier):
        """Test that statistics are updated after classification."""
        mock_response = self.create_mock_llm_response("KNOWLEDGE_BASE", 0.9, "Test")
        
        mock_llm = AsyncMock()
        mock_llm.acomplete = AsyncMock(return_value=CompletionResponse(text=mock_response))
        classifier.llm = mock_llm
        
        initial_stats = classifier.get_classification_stats()
        initial_count = initial_stats["classifications_performed"]
        
        await classifier.classify_query("Test query")
        
        updated_stats = classifier.get_classification_stats()
        assert updated_stats["classifications_performed"] == initial_count + 1
        assert updated_stats["knowledge_base_queries"] > initial_stats["knowledge_base_queries"]
        assert updated_stats["high_confidence_classifications"] > initial_stats["high_confidence_classifications"]


class TestQueryClassification:
    """Test cases for QueryClassification dataclass."""
    
    def test_query_classification_creation(self):
        """Test creating QueryClassification object."""
        classification = QueryClassification(
            query_type=QueryType.KNOWLEDGE_BASE,
            confidence_score=0.9,
            reasoning="Test reasoning",
            suggested_sources=["papers"],
            metadata={"test": "value"},
            processing_time=0.1
        )
        
        assert classification.query_type == QueryType.KNOWLEDGE_BASE
        assert classification.confidence_score == 0.9
        assert classification.reasoning == "Test reasoning"
        assert classification.suggested_sources == ["papers"]
        assert classification.metadata["test"] == "value"
        assert classification.processing_time == 0.1


class TestQueryType:
    """Test cases for QueryType enum."""
    
    def test_query_type_values(self):
        """Test QueryType enum values."""
        assert QueryType.KNOWLEDGE_BASE.value == "knowledge_base"
        assert QueryType.REAL_TIME.value == "real_time"
        assert QueryType.HYBRID.value == "hybrid"
    
    def test_query_type_from_string(self):
        """Test creating QueryType from string."""
        assert QueryType("knowledge_base") == QueryType.KNOWLEDGE_BASE
        assert QueryType("real_time") == QueryType.REAL_TIME
        assert QueryType("hybrid") == QueryType.HYBRID


if __name__ == "__main__":
    pytest.main([__file__])