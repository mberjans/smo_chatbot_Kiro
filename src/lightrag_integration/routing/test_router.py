"""
Unit tests for the query router.

This module tests the QueryRouter class functionality including routing decisions,
fallback mechanisms, response combination, and metrics collection.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from .router import (
    QueryRouter, RoutingStrategy, RoutingDecision, RoutedResponse
)
from .classifier import QueryClassifier, QueryClassification, QueryType
from ..config.settings import LightRAGConfig


class TestQueryRouter:
    """Test cases for QueryRouter."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        config = Mock(spec=LightRAGConfig)
        config.cache_directory = "/tmp/test_cache"
        config.max_concurrent_requests = 5
        return config
    
    @pytest.fixture
    def mock_classifier(self):
        """Create a mock query classifier."""
        classifier = Mock(spec=QueryClassifier)
        return classifier
    
    @pytest.fixture
    def mock_lightrag_func(self):
        """Create a mock LightRAG query function."""
        async def mock_func(query: str):
            return {
                "content": f"LightRAG response to: {query}",
                "bibliography": "**Sources:**\n[1]: Test paper",
                "confidence_score": 0.9,
                "processing_time": 0.5,
                "metadata": {"source_count": 1}
            }
        return mock_func
    
    @pytest.fixture
    def mock_perplexity_func(self):
        """Create a mock Perplexity query function."""
        async def mock_func(query: str):
            return {
                "content": f"Perplexity response to: {query}",
                "bibliography": "**References:**\n[1]: Web source",
                "confidence_score": 0.8,
                "processing_time": 0.3,
                "metadata": {"citations_count": 1}
            }
        return mock_func
    
    @pytest.fixture
    def router(self, mock_classifier, mock_lightrag_func, mock_perplexity_func, mock_config):
        """Create a QueryRouter instance."""
        with patch('src.lightrag_integration.routing.router.setup_logger'):
            return QueryRouter(
                classifier=mock_classifier,
                lightrag_query_func=mock_lightrag_func,
                perplexity_query_func=mock_perplexity_func,
                config=mock_config
            )
    
    def create_mock_classification(
        self, 
        query_type: QueryType, 
        confidence: float = 0.9,
        reasoning: str = "Test classification"
    ) -> QueryClassification:
        """Create a mock query classification."""
        return QueryClassification(
            query_type=query_type,
            confidence_score=confidence,
            reasoning=reasoning,
            suggested_sources=["test"],
            metadata={"test": True},
            processing_time=0.1
        )
    
    @pytest.mark.asyncio
    async def test_route_knowledge_base_query_high_confidence(self, router, mock_classifier):
        """Test routing of high-confidence knowledge base query."""
        # Mock classification
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.9)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        query = "What is clinical metabolomics?"
        result = await router.route_query(query)
        
        assert isinstance(result, RoutedResponse)
        assert result.source == "LightRAG"
        assert result.sources_used == ["lightrag"]
        assert "LightRAG response to:" in result.content
        assert result.routing_decision.strategy == RoutingStrategy.LIGHTRAG_ONLY
        assert result.routing_decision.primary_source == "lightrag"
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_route_real_time_query_high_confidence(self, router, mock_classifier):
        """Test routing of high-confidence real-time query."""
        classification = self.create_mock_classification(QueryType.REAL_TIME, 0.9)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        query = "Latest metabolomics research in 2024"
        result = await router.route_query(query)
        
        assert result.source == "Perplexity"
        assert result.sources_used == ["perplexity"]
        assert "Perplexity response to:" in result.content
        assert result.routing_decision.strategy == RoutingStrategy.PERPLEXITY_ONLY
        assert result.routing_decision.primary_source == "perplexity"
    
    @pytest.mark.asyncio
    async def test_route_hybrid_query_parallel(self, router, mock_classifier):
        """Test routing of hybrid query with parallel execution."""
        classification = self.create_mock_classification(QueryType.HYBRID, 0.8)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        query = "How has metabolomics evolved recently?"
        result = await router.route_query(query)
        
        assert result.source == "Combined"
        assert set(result.sources_used) == {"lightrag", "perplexity"}
        assert "LightRAG Response:" in result.content
        assert "Perplexity Response:" in result.content
        assert result.routing_decision.strategy == RoutingStrategy.PARALLEL
        assert result.metadata["combined_response"] is True
    
    @pytest.mark.asyncio
    async def test_route_low_confidence_fallback(self, router, mock_classifier):
        """Test routing with low confidence triggers fallback strategy."""
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.5)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        query = "Unclear metabolomics question"
        result = await router.route_query(query)
        
        # Should use LIGHTRAG_FIRST strategy for low confidence
        assert result.routing_decision.strategy == RoutingStrategy.LIGHTRAG_FIRST
        assert result.source == "LightRAG"  # Primary should succeed
    
    @pytest.mark.asyncio
    async def test_lightrag_failure_fallback_to_perplexity(self, router, mock_classifier):
        """Test fallback to Perplexity when LightRAG fails."""
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.7)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        # Mock LightRAG to fail
        async def failing_lightrag(query: str):
            raise Exception("LightRAG service unavailable")
        
        router.lightrag_query_func = failing_lightrag
        
        query = "Test query"
        result = await router.route_query(query)
        
        assert result.source == "Perplexity"
        assert result.metadata["fallback_used"] is True
        assert "primary_failure" in result.metadata
        assert len(result.errors) > 0
    
    @pytest.mark.asyncio
    async def test_both_services_fail_error_response(self, router, mock_classifier):
        """Test error response when both services fail."""
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.7)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        # Mock both services to fail
        async def failing_service(query: str):
            raise Exception("Service unavailable")
        
        router.lightrag_query_func = failing_service
        router.perplexity_query_func = failing_service
        
        query = "Test query"
        result = await router.route_query(query)
        
        assert result.source == "Error"
        assert result.confidence_score == 0.0
        assert len(result.errors) > 0
        assert "unable to process your query" in result.content
        assert result.metadata["error_response"] is True
    
    @pytest.mark.asyncio
    async def test_forced_routing_strategy(self, router, mock_classifier):
        """Test forcing a specific routing strategy."""
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.9)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        query = "Test query"
        result = await router.route_query(
            query, 
            force_strategy=RoutingStrategy.PERPLEXITY_ONLY
        )
        
        assert result.source == "Perplexity"
        assert result.routing_decision.strategy == RoutingStrategy.PERPLEXITY_ONLY
        assert result.routing_decision.metadata["forced"] is True
        assert result.routing_decision.confidence_score == 1.0
    
    @pytest.mark.asyncio
    async def test_route_with_context(self, router, mock_classifier):
        """Test routing with context information."""
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.8)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        query = "Test query"
        context = {"user_type": "researcher", "domain": "metabolomics"}
        
        result = await router.route_query(query, context=context)
        
        # Verify classifier was called with context
        mock_classifier.classify_query.assert_called_once_with(query, context)
        assert isinstance(result, RoutedResponse)
    
    @pytest.mark.asyncio
    async def test_empty_query_raises_error(self, router):
        """Test that empty queries raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await router.route_query("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await router.route_query("   ")
    
    @pytest.mark.asyncio
    async def test_classification_failure_with_retry(self, router, mock_classifier):
        """Test classification failure handling with retry logic."""
        # Mock classifier to fail multiple times then succeed
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.8)
        mock_classifier.classify_query = AsyncMock(side_effect=[
            Exception("Classification failed"),
            Exception("Classification failed again"),
            classification
        ])
        
        query = "Test query"
        result = await router.route_query(query)
        
        # Should eventually succeed after retries
        assert result.source == "LightRAG"
        assert mock_classifier.classify_query.call_count == 3
    
    @pytest.mark.asyncio
    async def test_classification_failure_exhausted_retries(self, router, mock_classifier):
        """Test classification failure after exhausting retries."""
        # Mock classifier to always fail
        mock_classifier.classify_query = AsyncMock(side_effect=Exception("Persistent failure"))
        
        query = "Test query"
        result = await router.route_query(query)
        
        # Should return error response
        assert result.source == "Error"
        assert len(result.errors) > 0
        assert mock_classifier.classify_query.call_count == 3  # Max retries
    
    @pytest.mark.asyncio
    async def test_batch_route_queries(self, router, mock_classifier):
        """Test batch routing of multiple queries."""
        # Mock different classifications for different queries
        classifications = [
            self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.9),
            self.create_mock_classification(QueryType.REAL_TIME, 0.8),
            self.create_mock_classification(QueryType.HYBRID, 0.7)
        ]
        mock_classifier.classify_query = AsyncMock(side_effect=classifications)
        
        queries = [
            "What is metabolomics?",
            "Latest metabolomics news",
            "How has the field evolved?"
        ]
        
        results = await router.batch_route_queries(queries)
        
        assert len(results) == 3
        assert results[0].source == "LightRAG"
        assert results[1].source == "Perplexity"
        assert results[2].source == "Combined"
    
    @pytest.mark.asyncio
    async def test_batch_route_with_contexts(self, router, mock_classifier):
        """Test batch routing with context information."""
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.8)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        queries = ["Query 1", "Query 2"]
        contexts = [{"type": "basic"}, {"type": "advanced"}]
        
        results = await router.batch_route_queries(queries, contexts)
        
        assert len(results) == 2
        assert mock_classifier.classify_query.call_count == 2
    
    @pytest.mark.asyncio
    async def test_batch_route_mismatched_contexts(self, router):
        """Test batch routing with mismatched contexts length."""
        queries = ["Query 1", "Query 2"]
        contexts = [{"type": "basic"}]  # Only one context for two queries
        
        with pytest.raises(ValueError, match="Contexts list must match queries list length"):
            await router.batch_route_queries(queries, contexts)
    
    @pytest.mark.asyncio
    async def test_batch_route_with_strategies(self, router, mock_classifier):
        """Test batch routing with forced strategies."""
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.8)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        queries = ["Query 1", "Query 2"]
        strategies = [RoutingStrategy.LIGHTRAG_ONLY, RoutingStrategy.PERPLEXITY_ONLY]
        
        results = await router.batch_route_queries(queries, strategies=strategies)
        
        assert len(results) == 2
        assert results[0].source == "LightRAG"
        assert results[1].source == "Perplexity"
        assert results[0].routing_decision.metadata["forced"] is True
        assert results[1].routing_decision.metadata["forced"] is True
    
    @pytest.mark.asyncio
    async def test_batch_route_with_errors(self, router, mock_classifier):
        """Test batch routing handling individual query errors."""
        # First query succeeds, second fails
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.8)
        mock_classifier.classify_query = AsyncMock(side_effect=[
            classification,
            Exception("Classification error")
        ])
        
        queries = ["Good query", "Bad query"]
        results = await router.batch_route_queries(queries)
        
        assert len(results) == 2
        assert results[0].source == "LightRAG"
        assert results[1].source == "Error"
        assert len(results[1].errors) > 0
    
    @pytest.mark.asyncio
    async def test_parallel_query_timeout(self, router, mock_classifier):
        """Test parallel query timeout handling."""
        classification = self.create_mock_classification(QueryType.HYBRID, 0.8)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        # Mock slow functions
        async def slow_lightrag(query: str):
            await asyncio.sleep(1.0)  # Longer than timeout
            return {"content": "Slow response", "confidence_score": 0.8}
        
        async def slow_perplexity(query: str):
            await asyncio.sleep(1.0)  # Longer than timeout
            return {"content": "Slow response", "confidence_score": 0.8}
        
        router.lightrag_query_func = slow_lightrag
        router.perplexity_query_func = slow_perplexity
        
        # Set short timeout
        router.routing_config["timeout_seconds"]["parallel"] = 0.1
        
        query = "Test query"
        result = await router.route_query(query)
        
        assert result.source == "Error"
        assert any("timeout" in error.lower() for error in result.errors)
    
    @pytest.mark.asyncio
    async def test_parallel_query_partial_success(self, router, mock_classifier):
        """Test parallel query with one service failing."""
        classification = self.create_mock_classification(QueryType.HYBRID, 0.8)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        # Mock one service to fail
        async def failing_lightrag(query: str):
            raise Exception("LightRAG failed")
        
        router.lightrag_query_func = failing_lightrag
        
        query = "Test query"
        result = await router.route_query(query)
        
        assert result.source == "Perplexity"
        assert result.sources_used == ["perplexity"]
        assert result.metadata["partial_parallel"] is True
        assert len(result.errors) > 0  # Should contain LightRAG error
    
    def test_get_routing_metrics(self, router):
        """Test getting routing metrics."""
        metrics = router.get_routing_metrics()
        
        assert "total_queries" in metrics
        assert "successful_routes" in metrics
        assert "failed_routes" in metrics
        assert "lightrag_queries" in metrics
        assert "perplexity_queries" in metrics
        assert "hybrid_queries" in metrics
        assert "fallback_activations" in metrics
        assert "average_response_time" in metrics
        assert "routing_accuracy" in metrics
        assert "error_rates" in metrics
        assert "routing_config" in metrics
        assert "classifier_stats" in metrics
        assert "success_rate" in metrics
        assert "strategy_distribution" in metrics
    
    def test_update_routing_config(self, router):
        """Test updating routing configuration."""
        new_config = {
            "confidence_thresholds": {
                "high_confidence": 0.85
            },
            "timeout_seconds": {
                "lightrag": 25.0
            }
        }
        
        router.update_routing_config(new_config)
        
        assert router.routing_config["confidence_thresholds"]["high_confidence"] == 0.85
        assert router.routing_config["timeout_seconds"]["lightrag"] == 25.0
        # Should preserve other values
        assert "medium_confidence" in router.routing_config["confidence_thresholds"]
        assert "perplexity" in router.routing_config["timeout_seconds"]
    
    def test_get_primary_source(self, router):
        """Test getting primary source for routing strategies."""
        assert router._get_primary_source(RoutingStrategy.LIGHTRAG_ONLY) == "lightrag"
        assert router._get_primary_source(RoutingStrategy.PERPLEXITY_ONLY) == "perplexity"
        assert router._get_primary_source(RoutingStrategy.LIGHTRAG_FIRST) == "lightrag"
        assert router._get_primary_source(RoutingStrategy.PERPLEXITY_FIRST) == "perplexity"
        assert router._get_primary_source(RoutingStrategy.PARALLEL) == "both"
        assert router._get_primary_source(RoutingStrategy.HYBRID) == "both"
    
    def test_get_fallback_sources(self, router):
        """Test getting fallback sources for routing strategies."""
        assert router._get_fallback_sources(RoutingStrategy.LIGHTRAG_ONLY) == ["perplexity"]
        assert router._get_fallback_sources(RoutingStrategy.PERPLEXITY_ONLY) == ["lightrag"]
        assert router._get_fallback_sources(RoutingStrategy.LIGHTRAG_FIRST) == ["perplexity"]
        assert router._get_fallback_sources(RoutingStrategy.PERPLEXITY_FIRST) == ["lightrag"]
        assert router._get_fallback_sources(RoutingStrategy.PARALLEL) == []
        assert router._get_fallback_sources(RoutingStrategy.HYBRID) == []
    
    def test_combine_responses(self, router):
        """Test combining multiple responses."""
        results = [
            ("LightRAG", {
                "content": "LightRAG content",
                "bibliography": "LightRAG sources",
                "confidence_score": 0.9,
                "processing_time": 0.5,
                "metadata": {"source_count": 2}
            }),
            ("Perplexity", {
                "content": "Perplexity content",
                "bibliography": "Perplexity sources",
                "confidence_score": 0.8,
                "processing_time": 0.3,
                "metadata": {"citations_count": 3}
            })
        ]
        
        routing_decision = RoutingDecision(
            strategy=RoutingStrategy.PARALLEL,
            primary_source="both",
            fallback_sources=[],
            confidence_score=0.8,
            reasoning="Test",
            classification=None,
            metadata={}
        )
        
        combined = router._combine_responses(results, routing_decision, [])
        
        assert combined.source == "Combined"
        assert combined.sources_used == ["lightrag", "perplexity"]
        assert "LightRAG Response:" in combined.content
        assert "Perplexity Response:" in combined.content
        assert "LightRAG Sources:" in combined.bibliography
        assert "Perplexity Sources:" in combined.bibliography
        assert combined.confidence_score == 0.85  # Average of 0.9 and 0.8
        assert combined.processing_time == 0.8  # Sum of 0.5 and 0.3
        assert combined.metadata["combined_response"] is True
        assert combined.metadata["sources_combined"] == 2
    
    def test_create_error_response(self, router):
        """Test creating error response."""
        query = "Test query"
        errors = ["Error 1", "Error 2"]
        processing_time = 1.5
        
        error_response = router._create_error_response(query, errors, processing_time)
        
        assert error_response.source == "Error"
        assert error_response.confidence_score == 0.0
        assert error_response.processing_time == processing_time
        assert error_response.sources_used == []
        assert error_response.metadata["error_response"] is True
        assert error_response.metadata["query_length"] == len(query)
        assert error_response.metadata["error_count"] == 2
        assert error_response.errors == errors
        assert "unable to process your query" in error_response.content
    
    @pytest.mark.asyncio
    async def test_router_without_query_functions(self, mock_classifier, mock_config):
        """Test router behavior without query functions configured."""
        with patch('src.lightrag_integration.routing.router.setup_logger'):
            router = QueryRouter(
                classifier=mock_classifier,
                lightrag_query_func=None,
                perplexity_query_func=None,
                config=mock_config
            )
        
        classification = self.create_mock_classification(QueryType.KNOWLEDGE_BASE, 0.9)
        mock_classifier.classify_query = AsyncMock(return_value=classification)
        
        query = "Test query"
        result = await router.route_query(query)
        
        assert result.source == "Error"
        assert any("not available" in error for error in result.errors)


class TestRoutingDecision:
    """Test cases for RoutingDecision dataclass."""
    
    def test_routing_decision_creation(self):
        """Test creating RoutingDecision object."""
        classification = QueryClassification(
            query_type=QueryType.KNOWLEDGE_BASE,
            confidence_score=0.9,
            reasoning="Test",
            suggested_sources=[],
            metadata={},
            processing_time=0.1
        )
        
        decision = RoutingDecision(
            strategy=RoutingStrategy.LIGHTRAG_ONLY,
            primary_source="lightrag",
            fallback_sources=["perplexity"],
            confidence_score=0.9,
            reasoning="High confidence knowledge base query",
            classification=classification,
            metadata={"test": True}
        )
        
        assert decision.strategy == RoutingStrategy.LIGHTRAG_ONLY
        assert decision.primary_source == "lightrag"
        assert decision.fallback_sources == ["perplexity"]
        assert decision.confidence_score == 0.9
        assert decision.reasoning == "High confidence knowledge base query"
        assert decision.classification == classification
        assert decision.metadata["test"] is True


class TestRoutedResponse:
    """Test cases for RoutedResponse dataclass."""
    
    def test_routed_response_creation(self):
        """Test creating RoutedResponse object."""
        routing_decision = RoutingDecision(
            strategy=RoutingStrategy.LIGHTRAG_ONLY,
            primary_source="lightrag",
            fallback_sources=[],
            confidence_score=0.9,
            reasoning="Test",
            classification=None,
            metadata={}
        )
        
        response = RoutedResponse(
            content="Test response",
            bibliography="Test sources",
            confidence_score=0.9,
            processing_time=0.5,
            source="LightRAG",
            sources_used=["lightrag"],
            routing_decision=routing_decision,
            metadata={"test": True},
            errors=[]
        )
        
        assert response.content == "Test response"
        assert response.bibliography == "Test sources"
        assert response.confidence_score == 0.9
        assert response.processing_time == 0.5
        assert response.source == "LightRAG"
        assert response.sources_used == ["lightrag"]
        assert response.routing_decision == routing_decision
        assert response.metadata["test"] is True
        assert response.errors == []


class TestRoutingStrategy:
    """Test cases for RoutingStrategy enum."""
    
    def test_routing_strategy_values(self):
        """Test RoutingStrategy enum values."""
        assert RoutingStrategy.LIGHTRAG_ONLY.value == "lightrag_only"
        assert RoutingStrategy.PERPLEXITY_ONLY.value == "perplexity_only"
        assert RoutingStrategy.LIGHTRAG_FIRST.value == "lightrag_first"
        assert RoutingStrategy.PERPLEXITY_FIRST.value == "perplexity_first"
        assert RoutingStrategy.PARALLEL.value == "parallel"
        assert RoutingStrategy.HYBRID.value == "hybrid"


if __name__ == "__main__":
    pytest.main([__file__])