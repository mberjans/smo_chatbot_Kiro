"""
Integration tests for the query router.

This module provides integration tests that demonstrate the router functionality
without requiring all external dependencies.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

# Mock the external dependencies
class MockBaseLLM:
    """Mock LLM for testing."""
    async def acomplete(self, prompt: str):
        response = Mock()
        response.text = '''
        {
            "query_type": "KNOWLEDGE_BASE",
            "confidence_score": 0.9,
            "reasoning": "Query asks for established scientific definition",
            "suggested_sources": ["research_papers"],
            "keywords": ["metabolomics"],
            "temporal_indicators": [],
            "domain_specificity": 0.9
        }
        '''
        return response

class MockLightRAGConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.cache_directory = "/tmp/test_cache"
        self.max_concurrent_requests = 5
    
    @classmethod
    def from_env(cls):
        return cls()

# Mock the imports
import sys
from unittest.mock import MagicMock

# Mock llama_index modules
sys.modules['llama_index'] = MagicMock()
sys.modules['llama_index.core'] = MagicMock()
sys.modules['llama_index.core.base'] = MagicMock()
sys.modules['llama_index.core.base.llms'] = MagicMock()
sys.modules['llama_index.core.base.llms.base'] = MagicMock()
sys.modules['llama_index.core.prompts'] = MagicMock()

# Set up the mock classes
sys.modules['llama_index.core.base.llms.base'].BaseLLM = MockBaseLLM
sys.modules['llama_index.core.prompts'].PromptTemplate = Mock

# Now import our modules
from .classifier import QueryClassifier, QueryClassification, QueryType
from .router import QueryRouter, RoutingStrategy, RoutingDecision, RoutedResponse


class TestRouterIntegration:
    """Integration tests for the query router."""
    
    @pytest.fixture
    def mock_lightrag_func(self):
        """Create a mock LightRAG query function."""
        async def mock_func(query: str) -> Dict[str, Any]:
            return {
                "content": f"LightRAG response to: {query}",
                "bibliography": "**Sources:**\n[1]: Test paper (Confidence: 0.90)",
                "confidence_score": 0.9,
                "processing_time": 0.5,
                "metadata": {"source_count": 1}
            }
        return mock_func
    
    @pytest.fixture
    def mock_perplexity_func(self):
        """Create a mock Perplexity query function."""
        async def mock_func(query: str) -> Dict[str, Any]:
            return {
                "content": f"Perplexity response to: {query}",
                "bibliography": "**References:**\n[1]: Web source",
                "confidence_score": 0.8,
                "processing_time": 0.3,
                "metadata": {"citations_count": 1}
            }
        return mock_func
    
    @pytest.fixture
    def classifier(self):
        """Create a query classifier."""
        llm = MockBaseLLM()
        config = MockLightRAGConfig()
        
        # Mock the setup_logger function
        with pytest.MonkeyPatch().context() as m:
            m.setattr('src.lightrag_integration.routing.classifier.setup_logger', Mock())
            classifier = QueryClassifier(llm, config)
        
        return classifier
    
    @pytest.fixture
    def router(self, classifier, mock_lightrag_func, mock_perplexity_func):
        """Create a query router."""
        config = MockLightRAGConfig()
        
        # Mock the setup_logger function
        with pytest.MonkeyPatch().context() as m:
            m.setattr('src.lightrag_integration.routing.router.setup_logger', Mock())
            router = QueryRouter(
                classifier=classifier,
                lightrag_query_func=mock_lightrag_func,
                perplexity_query_func=mock_perplexity_func,
                config=config
            )
        
        return router
    
    @pytest.mark.asyncio
    async def test_end_to_end_knowledge_base_routing(self, router):
        """Test end-to-end routing for knowledge base query."""
        query = "What is clinical metabolomics?"
        
        result = await router.route_query(query)
        
        assert isinstance(result, RoutedResponse)
        assert result.source == "LightRAG"
        assert result.sources_used == ["lightrag"]
        assert "LightRAG response to:" in result.content
        assert result.confidence_score > 0.0
        assert result.processing_time > 0.0
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, router):
        """Test fallback mechanism when primary service fails."""
        # Mock LightRAG to fail
        async def failing_lightrag(query: str):
            raise Exception("LightRAG service unavailable")
        
        router.lightrag_query_func = failing_lightrag
        
        query = "What is metabolomics?"
        result = await router.route_query(query)
        
        # Should fall back to Perplexity
        assert result.source == "Perplexity"
        assert result.metadata.get("fallback_used") is True
        assert len(result.errors) > 0  # Should contain the LightRAG error
    
    @pytest.mark.asyncio
    async def test_forced_routing_strategy(self, router):
        """Test forcing a specific routing strategy."""
        query = "Test query"
        
        result = await router.route_query(
            query, 
            force_strategy=RoutingStrategy.PERPLEXITY_ONLY
        )
        
        assert result.source == "Perplexity"
        assert result.routing_decision.strategy == RoutingStrategy.PERPLEXITY_ONLY
        assert result.routing_decision.metadata["forced"] is True
    
    @pytest.mark.asyncio
    async def test_parallel_routing_strategy(self, router):
        """Test parallel routing strategy."""
        # Mock classifier to return HYBRID classification
        async def mock_classify(query, context=None):
            return QueryClassification(
                query_type=QueryType.HYBRID,
                confidence_score=0.8,
                reasoning="Hybrid query requiring both sources",
                suggested_sources=["both"],
                metadata={},
                processing_time=0.1
            )
        
        router.classifier.classify_query = mock_classify
        
        query = "How has metabolomics evolved recently?"
        result = await router.route_query(query)
        
        assert result.source == "Combined"
        assert set(result.sources_used) == {"lightrag", "perplexity"}
        assert "LightRAG Response:" in result.content
        assert "Perplexity Response:" in result.content
        assert result.metadata["combined_response"] is True
    
    @pytest.mark.asyncio
    async def test_batch_routing(self, router):
        """Test batch routing of multiple queries."""
        queries = [
            "What is metabolomics?",
            "Define biomarkers",
            "Explain mass spectrometry"
        ]
        
        results = await router.batch_route_queries(queries)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, RoutedResponse)
            assert result.source in ["LightRAG", "Perplexity", "Combined"]
            assert len(result.content) > 0
    
    def test_routing_metrics_collection(self, router):
        """Test that routing metrics are collected properly."""
        metrics = router.get_routing_metrics()
        
        # Check that all expected metrics are present
        expected_metrics = [
            "total_queries", "successful_routes", "failed_routes",
            "lightrag_queries", "perplexity_queries", "hybrid_queries",
            "fallback_activations", "average_response_time", "routing_accuracy",
            "error_rates", "routing_config", "success_rate", "strategy_distribution"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
    
    def test_routing_config_updates(self, router):
        """Test updating routing configuration."""
        original_threshold = router.routing_config["confidence_thresholds"]["high_confidence"]
        
        new_config = {
            "confidence_thresholds": {
                "high_confidence": 0.85
            }
        }
        
        router.update_routing_config(new_config)
        
        assert router.routing_config["confidence_thresholds"]["high_confidence"] == 0.85
        assert router.routing_config["confidence_thresholds"]["high_confidence"] != original_threshold
    
    @pytest.mark.asyncio
    async def test_error_handling_both_services_fail(self, router):
        """Test error handling when both services fail."""
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
    async def test_empty_query_validation(self, router):
        """Test validation of empty queries."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await router.route_query("")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            await router.route_query("   ")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])