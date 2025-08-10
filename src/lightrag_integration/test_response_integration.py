"""
Unit tests for Response Integration System

Tests the ResponseIntegrator class and related functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any

from .response_integration import (
    ResponseIntegrator, 
    ProcessedResponse, 
    CombinedResponse,
    ResponseSource,
    ResponseQuality
)


class TestResponseIntegrator:
    """Test cases for ResponseIntegrator class."""
    
    @pytest.fixture
    def integrator(self):
        """Create a ResponseIntegrator instance for testing."""
        return ResponseIntegrator()
    
    @pytest.fixture
    def sample_lightrag_response(self):
        """Sample LightRAG response for testing."""
        return {
            "answer": "Clinical metabolomics is the application of metabolomics to clinical research and practice.",
            "content": "Clinical metabolomics is the application of metabolomics to clinical research and practice.",
            "confidence_score": 0.85,
            "source_documents": ["paper1.pdf", "paper2.pdf"],
            "entities_used": [
                {
                    "id": "entity1",
                    "text": "clinical metabolomics",
                    "type": "field",
                    "relevance_score": 0.9,
                    "source_documents": ["paper1.pdf"]
                }
            ],
            "relationships_used": [
                {
                    "id": "rel1",
                    "type": "application_of",
                    "source": "clinical_metabolomics",
                    "target": "metabolomics",
                    "confidence": 0.8
                }
            ],
            "processing_time": 0.5,
            "metadata": {"query_type": "definition"}
        }
    
    @pytest.fixture
    def sample_perplexity_response(self):
        """Sample Perplexity response for testing."""
        return {
            "content": "Clinical metabolomics is a rapidly growing field that applies metabolomic technologies to clinical research.",
            "bibliography": "**References:**\n[1]: Recent Clinical Study\n      (Confidence: 0.8)\n[2]: Review Article\n      (Confidence: 0.7)",
            "confidence_score": 0.8,
            "processing_time": 0.3,
            "metadata": {"citations_count": 2}
        }
    
    @pytest.mark.asyncio
    async def test_process_lightrag_response(self, integrator, sample_lightrag_response):
        """Test processing of LightRAG response."""
        processed = await integrator.process_lightrag_response(sample_lightrag_response)
        
        assert isinstance(processed, ProcessedResponse)
        assert processed.source == ResponseSource.LIGHTRAG
        assert processed.confidence_score == 0.85
        assert len(processed.citations) > 0
        assert processed.quality_score > 0
        assert "clinical metabolomics" in processed.content.lower()
    
    @pytest.mark.asyncio
    async def test_process_perplexity_response(self, integrator, sample_perplexity_response):
        """Test processing of Perplexity response."""
        processed = await integrator.process_perplexity_response(sample_perplexity_response)
        
        assert isinstance(processed, ProcessedResponse)
        assert processed.source == ResponseSource.PERPLEXITY
        assert processed.confidence_score == 0.8
        assert len(processed.citations) > 0
        assert processed.quality_score > 0
        assert "clinical metabolomics" in processed.content.lower()
    
    @pytest.mark.asyncio
    async def test_combine_responses_both_available(self, integrator, sample_lightrag_response, sample_perplexity_response):
        """Test combining responses when both are available."""
        lightrag_processed = await integrator.process_lightrag_response(sample_lightrag_response)
        perplexity_processed = await integrator.process_perplexity_response(sample_perplexity_response)
        
        combined = await integrator.combine_responses(lightrag_processed, perplexity_processed)
        
        assert isinstance(combined, CombinedResponse)
        assert len(combined.sources_used) == 2
        assert ResponseSource.LIGHTRAG in combined.sources_used
        assert ResponseSource.PERPLEXITY in combined.sources_used
        assert combined.confidence_score > 0
        assert len(combined.citations) > 0
    
    @pytest.mark.asyncio
    async def test_combine_responses_lightrag_only(self, integrator, sample_lightrag_response):
        """Test combining responses when only LightRAG is available."""
        lightrag_processed = await integrator.process_lightrag_response(sample_lightrag_response)
        
        combined = await integrator.combine_responses(lightrag_processed, None)
        
        assert isinstance(combined, CombinedResponse)
        assert len(combined.sources_used) == 1
        assert combined.sources_used[0] == ResponseSource.LIGHTRAG
        assert combined.combination_strategy == "lightrag_only"
    
    @pytest.mark.asyncio
    async def test_combine_responses_perplexity_only(self, integrator, sample_perplexity_response):
        """Test combining responses when only Perplexity is available."""
        perplexity_processed = await integrator.process_perplexity_response(sample_perplexity_response)
        
        combined = await integrator.combine_responses(None, perplexity_processed)
        
        assert isinstance(combined, CombinedResponse)
        assert len(combined.sources_used) == 1
        assert combined.sources_used[0] == ResponseSource.PERPLEXITY
        assert combined.combination_strategy == "perplexity_only"
    
    @pytest.mark.asyncio
    async def test_combine_responses_none_available(self, integrator):
        """Test combining responses when neither is available."""
        combined = await integrator.combine_responses(None, None)
        
        assert isinstance(combined, CombinedResponse)
        assert len(combined.sources_used) == 0
        assert combined.combination_strategy == "error"
        assert combined.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_select_best_response(self, integrator, sample_lightrag_response, sample_perplexity_response):
        """Test selecting the best response from multiple options."""
        lightrag_processed = await integrator.process_lightrag_response(sample_lightrag_response)
        perplexity_processed = await integrator.process_perplexity_response(sample_perplexity_response)
        
        # Modify one to have higher quality
        lightrag_processed.quality_score = 0.9
        perplexity_processed.quality_score = 0.7
        
        best = await integrator.select_best_response([lightrag_processed, perplexity_processed])
        
        assert best == lightrag_processed
        assert best.source == ResponseSource.LIGHTRAG
    
    def test_extract_lightrag_citations(self, integrator, sample_lightrag_response):
        """Test extraction of citations from LightRAG response."""
        citations = integrator._extract_lightrag_citations(sample_lightrag_response)
        
        assert len(citations) > 0
        assert all("id" in citation for citation in citations)
        assert all("source" in citation for citation in citations)
        assert all("confidence" in citation for citation in citations)
    
    def test_extract_perplexity_citations(self, integrator, sample_perplexity_response):
        """Test extraction of citations from Perplexity response."""
        citations = integrator._extract_perplexity_citations(sample_perplexity_response)
        
        assert len(citations) > 0
        assert all("id" in citation for citation in citations)
        assert all("source" in citation for citation in citations)
        assert all("type" in citation for citation in citations)
    
    def test_calculate_response_quality(self, integrator):
        """Test response quality calculation."""
        # Test high quality response
        quality = integrator._calculate_response_quality(
            content="This is a well-structured response with good length and clear information about clinical metabolomics.",
            confidence_score=0.9,
            citations=[{"id": "1", "source": "paper1.pdf"}],
            source=ResponseSource.LIGHTRAG
        )
        assert quality > 0.5
        
        # Test low quality response
        quality = integrator._calculate_response_quality(
            content="Short",
            confidence_score=0.3,
            citations=[],
            source=ResponseSource.PERPLEXITY
        )
        assert quality < 0.5
    
    def test_clean_response_content(self, integrator):
        """Test response content cleaning."""
        dirty_content = "This is a response   with   extra   spaces (confidence score: 0.8) and [1] citations."
        clean_content = integrator._clean_response_content(dirty_content)
        
        assert "confidence score" not in clean_content
        assert "   " not in clean_content
        assert clean_content.strip() == clean_content
    
    def test_determine_combination_strategy(self, integrator):
        """Test combination strategy determination."""
        # Create mock responses with different quality scores
        high_quality_lightrag = Mock()
        high_quality_lightrag.quality_score = 0.9
        
        low_quality_perplexity = Mock()
        low_quality_perplexity.quality_score = 0.4
        
        strategy = integrator._determine_combination_strategy(high_quality_lightrag, low_quality_perplexity)
        assert strategy == "lightrag_primary"
        
        # Test equal quality
        equal_quality_lightrag = Mock()
        equal_quality_lightrag.quality_score = 0.8
        
        equal_quality_perplexity = Mock()
        equal_quality_perplexity.quality_score = 0.8
        
        strategy = integrator._determine_combination_strategy(equal_quality_lightrag, equal_quality_perplexity)
        assert strategy == "parallel_sections"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_processing(self, integrator):
        """Test error handling during response processing."""
        # Test with response that will cause an exception during processing
        # Mock the _extract_lightrag_citations method to raise an exception
        original_method = integrator._extract_lightrag_citations
        
        def mock_extract_citations(response):
            raise ValueError("Test exception")
        
        integrator._extract_lightrag_citations = mock_extract_citations
        
        try:
            malformed_response = {"answer": "test", "confidence_score": 0.5}
            processed = await integrator.process_lightrag_response(malformed_response)
            
            assert isinstance(processed, ProcessedResponse)
            assert processed.confidence_score == 0.0
            assert "error" in processed.metadata
            assert "Test exception" in processed.metadata["error"]
        finally:
            # Restore original method
            integrator._extract_lightrag_citations = original_method
    
    def test_format_lightrag_bibliography(self, integrator):
        """Test LightRAG bibliography formatting."""
        citations = [
            {"id": "1", "source": "paper1.pdf", "confidence": 0.9},
            {"id": "2", "source": "paper2.pdf", "confidence": 0.6},
            {"id": "3", "source": "paper3.pdf", "confidence": 0.3}
        ]
        
        bibliography = integrator._format_lightrag_bibliography("", citations)
        
        assert "**Knowledge Base Sources:**" in bibliography
        assert "**Primary References:**" in bibliography
        assert "paper1.pdf" in bibliography
        assert "Confidence: 0.9" in bibliography
    
    def test_format_perplexity_bibliography(self, integrator):
        """Test Perplexity bibliography formatting."""
        citations = [
            {"id": "1", "source": "https://example.com/article1", "confidence": 0.8},
            {"id": "2", "source": "Research Paper Title", "confidence": 0.7}
        ]
        
        bibliography = integrator._format_perplexity_bibliography("", citations)
        
        assert "**Web Sources:**" in bibliography
        assert "https://example.com/article1" in bibliography
        assert "Confidence: 0.8" in bibliography
    
    def test_processed_response_to_dict(self, integrator):
        """Test ProcessedResponse to_dict method."""
        response = ProcessedResponse(
            content="Test content",
            bibliography="Test bibliography",
            confidence_score=0.8,
            source=ResponseSource.LIGHTRAG,
            processing_time=0.5,
            metadata={"test": "data"},
            quality_score=0.7,
            citations=[{"id": "1", "source": "test.pdf"}]
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["content"] == "Test content"
        assert response_dict["source"] == "lightrag"
        assert response_dict["confidence_score"] == 0.8
        assert response_dict["quality_score"] == 0.7
    
    def test_combined_response_to_dict(self, integrator):
        """Test CombinedResponse to_dict method."""
        response = CombinedResponse(
            content="Combined content",
            bibliography="Combined bibliography",
            confidence_score=0.8,
            processing_time=1.0,
            sources_used=[ResponseSource.LIGHTRAG, ResponseSource.PERPLEXITY],
            combination_strategy="parallel_sections",
            metadata={"test": "data"},
            quality_assessment={"combined_quality": 0.8},
            citations=[{"id": "1", "source": "test.pdf"}]
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["content"] == "Combined content"
        assert response_dict["sources_used"] == ["lightrag", "perplexity"]
        assert response_dict["combination_strategy"] == "parallel_sections"
        assert response_dict["quality_assessment"]["combined_quality"] == 0.8


class TestResponseIntegrationEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def integrator(self):
        """Create a ResponseIntegrator instance for testing."""
        return ResponseIntegrator()
    
    @pytest.mark.asyncio
    async def test_empty_response_processing(self, integrator):
        """Test processing of empty responses."""
        empty_response = {}
        
        processed = await integrator.process_lightrag_response(empty_response)
        
        assert isinstance(processed, ProcessedResponse)
        assert processed.content == ""  # Empty response should result in empty content
        assert processed.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_select_best_response_empty_list(self, integrator):
        """Test selecting best response from empty list."""
        with pytest.raises(ValueError):
            await integrator.select_best_response([])
    
    @pytest.mark.asyncio
    async def test_select_best_response_single_item(self, integrator):
        """Test selecting best response from single item list."""
        response = ProcessedResponse(
            content="Test",
            bibliography="",
            confidence_score=0.5,
            source=ResponseSource.LIGHTRAG,
            processing_time=0.1,
            metadata={},
            quality_score=0.6,
            citations=[]
        )
        
        best = await integrator.select_best_response([response])
        assert best == response
    
    def test_citation_extraction_no_citations(self, integrator):
        """Test citation extraction when no citations are present."""
        response_no_citations = {
            "answer": "Test answer",
            "source_documents": [],
            "entities_used": []
        }
        
        citations = integrator._extract_lightrag_citations(response_no_citations)
        assert len(citations) == 0
    
    def test_quality_calculation_edge_cases(self, integrator):
        """Test quality calculation with edge cases."""
        # Very short content
        quality = integrator._calculate_response_quality(
            content="Hi",
            confidence_score=1.0,
            citations=[],
            source=ResponseSource.LIGHTRAG
        )
        assert quality < 0.5
        
        # Very long content
        long_content = "A" * 3000
        quality = integrator._calculate_response_quality(
            content=long_content,
            confidence_score=1.0,
            citations=[],
            source=ResponseSource.LIGHTRAG
        )
        assert quality < 1.0  # Should be penalized for being too long
    
    def test_clean_content_edge_cases(self, integrator):
        """Test content cleaning with edge cases."""
        # Empty content
        assert integrator._clean_response_content("") == ""
        
        # Only whitespace
        assert integrator._clean_response_content("   \n\t   ") == ""
        
        # Multiple confidence scores
        content_with_multiple_scores = "Text (confidence score: 0.8) more text (confidence score: 0.9) end"
        cleaned = integrator._clean_response_content(content_with_multiple_scores)
        assert "confidence score" not in cleaned


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])