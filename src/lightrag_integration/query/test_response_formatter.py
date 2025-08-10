"""
Unit tests for LightRAG Response Formatter

Tests the response formatting and confidence scoring functionality.
Implements testing requirements for task 4.2.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from ..config.settings import LightRAGConfig
from .response_formatter import LightRAGResponseFormatter, FormattedResponse, CitationInfo


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


@pytest.fixture
def response_formatter(test_config):
    """Create a response formatter for testing."""
    return LightRAGResponseFormatter(test_config)


@pytest.fixture
def sample_response_data():
    """Create sample response data for testing."""
    return {
        "answer": "Clinical metabolomics is the study of metabolites in clinical settings. It involves analyzing small molecules to understand disease mechanisms and identify biomarkers.",
        "source_documents": ["paper1.pdf", "paper2.pdf", "paper3.pdf"],
        "entities_used": [
            {
                "id": "entity1",
                "text": "clinical metabolomics",
                "type": "field",
                "relevance_score": 0.9,
                "source_documents": ["paper1.pdf", "paper2.pdf"]
            },
            {
                "id": "entity2",
                "text": "biomarkers",
                "type": "metabolite",
                "relevance_score": 0.8,
                "source_documents": ["paper2.pdf", "paper3.pdf"]
            }
        ],
        "relationships_used": [
            {
                "id": "rel1",
                "type": "involves",
                "source": "entity1",
                "target": "entity2",
                "confidence": 0.75,
                "source_documents": ["paper1.pdf", "paper3.pdf"]
            }
        ],
        "confidence_score": 0.8,
        "processing_time": 1.5
    }


class TestLightRAGResponseFormatter:
    """Test cases for LightRAG Response Formatter."""
    
    def test_initialization(self, test_config):
        """Test response formatter initialization."""
        formatter = LightRAGResponseFormatter(test_config)
        
        assert formatter.config == test_config
        assert formatter.high_confidence_threshold == 0.8
        assert formatter.medium_confidence_threshold == 0.6
        assert formatter.low_confidence_threshold == 0.4
    
    def test_format_response_basic(self, response_formatter, sample_response_data):
        """Test basic response formatting."""
        result = response_formatter.format_response(
            answer=sample_response_data["answer"],
            source_documents=sample_response_data["source_documents"],
            entities_used=sample_response_data["entities_used"],
            relationships_used=sample_response_data["relationships_used"],
            confidence_score=sample_response_data["confidence_score"],
            processing_time=sample_response_data["processing_time"]
        )
        
        assert isinstance(result, FormattedResponse)
        assert result.content != ""
        assert result.bibliography != ""
        assert len(result.confidence_scores) > 0
        assert result.processing_time == sample_response_data["processing_time"]
        assert "seconds" in result.content  # Processing time should be included
    
    def test_format_response_with_citations(self, response_formatter, sample_response_data):
        """Test response formatting includes citation markers."""
        result = response_formatter.format_response(
            answer=sample_response_data["answer"],
            source_documents=sample_response_data["source_documents"],
            entities_used=sample_response_data["entities_used"],
            relationships_used=sample_response_data["relationships_used"],
            confidence_score=sample_response_data["confidence_score"],
            processing_time=sample_response_data["processing_time"]
        )
        
        # Should contain citation markers
        assert "[1]" in result.content or "[2]" in result.content or "[3]" in result.content
        assert "References:" in result.bibliography or "Further Reading:" in result.bibliography
    
    def test_generate_citations(self, response_formatter, sample_response_data):
        """Test citation generation."""
        citations = response_formatter._generate_citations(
            sample_response_data["source_documents"],
            sample_response_data["entities_used"],
            sample_response_data["relationships_used"]
        )
        
        assert len(citations) == 3  # Should have 3 citations for 3 documents
        assert all(isinstance(citation, CitationInfo) for citation in citations)
        assert all(citation.confidence_score > 0 for citation in citations)
        assert all(citation.source_document in sample_response_data["source_documents"] for citation in citations)
    
    def test_generate_bibliography_high_confidence(self, response_formatter):
        """Test bibliography generation with high confidence citations."""
        citations = [
            CitationInfo("1", "paper1.pdf", 0.9),
            CitationInfo("2", "paper2.pdf", 0.85),
            CitationInfo("3", "paper3.pdf", 0.4)
        ]
        
        bibliography = response_formatter._generate_bibliography(citations, 0.8)
        
        assert "Response Confidence: High" in bibliography
        assert "References:" in bibliography
        assert "Further Reading:" in bibliography
        assert "paper1.pdf" in bibliography
        assert "paper2.pdf" in bibliography
        assert "paper3.pdf" in bibliography
    
    def test_generate_bibliography_low_confidence(self, response_formatter):
        """Test bibliography generation with low confidence citations."""
        citations = [
            CitationInfo("1", "paper1.pdf", 0.3),
            CitationInfo("2", "paper2.pdf", 0.2)
        ]
        
        bibliography = response_formatter._generate_bibliography(citations, 0.3)
        
        assert "Response Confidence: Very Low" in bibliography
        assert "Further Reading:" in bibliography
        assert "References:" not in bibliography or bibliography.count("References:") == 0
    
    def test_add_confidence_indicators_high(self, response_formatter):
        """Test adding confidence indicators for high confidence."""
        content = "Clinical metabolomics is important. It helps in diagnosis."
        result = response_formatter._add_confidence_indicators(content, 0.9)
        
        # High confidence should not add indicators
        assert result == content
    
    def test_add_confidence_indicators_medium(self, response_formatter):
        """Test adding confidence indicators for medium confidence."""
        content = "Clinical metabolomics is important. It helps in diagnosis."
        result = response_formatter._add_confidence_indicators(content, 0.7)
        
        assert "(moderate confidence)" in result
    
    def test_add_confidence_indicators_low(self, response_formatter):
        """Test adding confidence indicators for low confidence."""
        content = "Clinical metabolomics is important. It helps in diagnosis."
        result = response_formatter._add_confidence_indicators(content, 0.3)
        
        assert "(low confidence - please verify)" in result
    
    def test_get_confidence_level(self, response_formatter):
        """Test confidence level categorization."""
        assert response_formatter._get_confidence_level(0.9) == "High"
        assert response_formatter._get_confidence_level(0.7) == "Medium"
        assert response_formatter._get_confidence_level(0.5) == "Low"
        assert response_formatter._get_confidence_level(0.2) == "Very Low"
    
    def test_calculate_enhanced_confidence_score(self, response_formatter, sample_response_data):
        """Test enhanced confidence score calculation."""
        enhanced_confidence, breakdown = response_formatter.calculate_enhanced_confidence_score(
            base_confidence=0.6,
            entities_used=sample_response_data["entities_used"],
            relationships_used=sample_response_data["relationships_used"],
            source_documents=sample_response_data["source_documents"]
        )
        
        assert isinstance(enhanced_confidence, float)
        assert 0.0 <= enhanced_confidence <= 1.0
        assert enhanced_confidence >= 0.6  # Should be at least the base confidence
        
        assert isinstance(breakdown, dict)
        assert "base_confidence" in breakdown
        assert "enhancement" in breakdown
        assert "enhanced_confidence" in breakdown
        assert "evidence_factors" in breakdown
    
    def test_calculate_enhanced_confidence_with_good_evidence(self, response_formatter):
        """Test enhanced confidence calculation with strong evidence."""
        entities_used = [
            {
                "id": "entity1",
                "text": "clinical metabolomics",
                "type": "field",
                "relevance_score": 0.95,
                "source_documents": ["paper1.pdf", "paper2.pdf"]
            },
            {
                "id": "entity2",
                "text": "biomarkers",
                "type": "metabolite",
                "relevance_score": 0.9,
                "source_documents": ["paper1.pdf", "paper2.pdf"]
            }
        ]
        
        relationships_used = [
            {
                "id": "rel1",
                "type": "involves",
                "source": "entity1",
                "target": "entity2",
                "confidence": 0.9,
                "source_documents": ["paper1.pdf", "paper2.pdf"]
            }
        ]
        
        enhanced_confidence, breakdown = response_formatter.calculate_enhanced_confidence_score(
            base_confidence=0.7,
            entities_used=entities_used,
            relationships_used=relationships_used,
            source_documents=["paper1.pdf", "paper2.pdf", "paper3.pdf"]
        )
        
        # Should be enhanced due to high entity/relationship confidence and source diversity
        assert enhanced_confidence > 0.7
        assert breakdown["evidence_factors"]["entity_confidence"] > 0.8
        assert breakdown["evidence_factors"]["relationship_confidence"] > 0.8
        assert breakdown["evidence_factors"]["source_diversity"] > 0
    
    def test_calculate_enhanced_confidence_with_poor_evidence(self, response_formatter):
        """Test enhanced confidence calculation with weak evidence."""
        entities_used = [
            {
                "id": "entity1",
                "text": "clinical metabolomics",
                "type": "field",
                "relevance_score": 0.3,
                "source_documents": ["paper1.pdf"]
            }
        ]
        
        relationships_used = [
            {
                "id": "rel1",
                "type": "involves",
                "source": "entity1",
                "target": "entity2",
                "confidence": 0.2,
                "source_documents": ["paper1.pdf"]
            }
        ]
        
        enhanced_confidence, breakdown = response_formatter.calculate_enhanced_confidence_score(
            base_confidence=0.5,
            entities_used=entities_used,
            relationships_used=relationships_used,
            source_documents=["paper1.pdf"]
        )
        
        # Enhancement should be minimal due to poor evidence
        assert enhanced_confidence <= 0.6  # Should not be enhanced much
        assert breakdown["evidence_factors"]["entity_confidence"] < 0.5
        assert breakdown["evidence_factors"]["relationship_confidence"] < 0.5
    
    def test_format_confidence_explanation(self, response_formatter):
        """Test confidence explanation formatting."""
        confidence_breakdown = {
            "base_confidence": 0.7,
            "enhancement": 0.1,
            "enhanced_confidence": 0.8,
            "evidence_factors": {
                "entity_confidence": 0.85,
                "relationship_confidence": 0.75,
                "source_diversity": 0.2,
                "evidence_consistency": 0.6,
                "graph_connectivity": 0.1
            },
            "entity_count": 2,
            "relationship_count": 1,
            "source_count": 3
        }
        
        explanation = response_formatter.format_confidence_explanation(confidence_breakdown)
        
        assert "Base confidence: 0.70" in explanation
        assert "Enhancement: +0.10" in explanation
        assert "Final confidence: 0.80" in explanation
        assert "Entity Confidence: 0.85" in explanation
        assert "Entities used: 2" in explanation
        assert "Relationships used: 1" in explanation
        assert "Source documents: 3" in explanation
    
    def test_format_response_error_handling(self, response_formatter):
        """Test response formatting error handling."""
        # Test with invalid data that might cause errors
        result = response_formatter.format_response(
            answer="Test answer",
            source_documents=[],  # Empty list
            entities_used=[],     # Empty list
            relationships_used=[], # Empty list
            confidence_score=0.5,
            processing_time=1.0
        )
        
        assert isinstance(result, FormattedResponse)
        assert result.content != ""
        assert "Test answer" in result.content
        assert "1.00 seconds" in result.content
    
    def test_insert_citation_markers_empty_citations(self, response_formatter):
        """Test citation marker insertion with no citations."""
        answer = "This is a test answer."
        result = response_formatter._insert_citation_markers(answer, [])
        
        assert result == answer  # Should be unchanged
    
    def test_insert_citation_markers_with_citations(self, response_formatter):
        """Test citation marker insertion with citations."""
        answer = "This is a test answer."
        citations = [
            CitationInfo("1", "paper1.pdf", 0.8),
            CitationInfo("2", "paper2.pdf", 0.7)
        ]
        
        result = response_formatter._insert_citation_markers(answer, citations)
        
        assert "[1]" in result
        assert "[2]" in result
        assert "This is a test answer" in result


class TestResponseFormatterIntegration:
    """Integration tests for response formatter."""
    
    def test_full_response_formatting_workflow(self, response_formatter, sample_response_data):
        """Test the complete response formatting workflow."""
        # Test the full workflow
        result = response_formatter.format_response(
            answer=sample_response_data["answer"],
            source_documents=sample_response_data["source_documents"],
            entities_used=sample_response_data["entities_used"],
            relationships_used=sample_response_data["relationships_used"],
            confidence_score=sample_response_data["confidence_score"],
            processing_time=sample_response_data["processing_time"],
            metadata={"test": "metadata"}
        )
        
        # Verify all components are present
        assert "Clinical metabolomics" in result.content
        assert len(result.confidence_scores) == 3  # 3 source documents
        assert "Response Confidence:" in result.bibliography
        assert "References:" in result.bibliography
        assert result.metadata["test"] == "metadata"
        assert result.metadata["overall_confidence"] == sample_response_data["confidence_score"]
        assert result.metadata["citation_count"] == 3
        assert result.metadata["entity_count"] == 2
        assert result.metadata["relationship_count"] == 1
    
    def test_confidence_enhancement_workflow(self, response_formatter, sample_response_data):
        """Test the confidence enhancement workflow."""
        base_confidence = 0.6
        
        enhanced_confidence, breakdown = response_formatter.calculate_enhanced_confidence_score(
            base_confidence=base_confidence,
            entities_used=sample_response_data["entities_used"],
            relationships_used=sample_response_data["relationships_used"],
            source_documents=sample_response_data["source_documents"]
        )
        
        # Verify enhancement occurred
        assert enhanced_confidence >= base_confidence
        assert breakdown["base_confidence"] == base_confidence
        assert breakdown["enhanced_confidence"] == enhanced_confidence
        
        # Verify explanation can be generated
        explanation = response_formatter.format_confidence_explanation(breakdown)
        assert len(explanation) > 0
        assert str(base_confidence) in explanation


if __name__ == "__main__":
    pytest.main([__file__])