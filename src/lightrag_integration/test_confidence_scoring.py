"""
Unit tests for LightRAG Confidence Scoring

Tests the enhanced confidence scoring functionality that works with
graph-based evidence and source document reliability.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from .confidence_scoring import (
    LightRAGConfidenceScorer,
    SourceReliabilityScore,
    CitationConfidenceScore,
    GraphEvidenceMetrics
)


class TestLightRAGConfidenceScorer:
    """Test cases for LightRAG confidence scorer."""
    
    @pytest.fixture
    def scorer(self):
        """Create confidence scorer instance."""
        return LightRAGConfidenceScorer()
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_entities(self, temp_dir):
        """Create sample entities for testing."""
        pdf_path = str(Path(temp_dir) / "test.pdf")
        Path(pdf_path).write_text("Test PDF content")
        
        return [
            {
                "id": "entity_1",
                "text": "clinical metabolomics",
                "type": "field",
                "relevance_score": 0.9,
                "source_documents": [pdf_path]
            },
            {
                "id": "entity_2",
                "text": "biomarker",
                "type": "concept",
                "relevance_score": 0.8,
                "source_documents": [pdf_path]
            }
        ]
    
    @pytest.fixture
    def sample_relationships(self, temp_dir):
        """Create sample relationships for testing."""
        pdf_path = str(Path(temp_dir) / "test.pdf")
        
        return [
            {
                "id": "rel_1",
                "type": "involves",
                "source": "entity_1",
                "target": "entity_2",
                "confidence": 0.85,
                "source_documents": [pdf_path]
            }
        ]
    
    @pytest.fixture
    def sample_citation_map(self, temp_dir):
        """Create sample citation map."""
        from .citation_formatter import PDFCitation
        
        pdf_path = str(Path(temp_dir) / "test.pdf")
        Path(pdf_path).write_text("Test PDF content")
        
        return {
            "1": PDFCitation("1", pdf_path, confidence_score=0.8)
        }
    
    def test_initialization(self):
        """Test scorer initialization."""
        scorer = LightRAGConfidenceScorer()
        
        assert scorer.min_confidence == 0.1
        assert scorer.max_confidence == 1.0
        assert scorer.default_reliability == 0.6
        assert "graph_evidence" in scorer.weights
        assert "source_reliability" in scorer.weights
    
    def test_calculate_enhanced_confidence(self, scorer, sample_entities, sample_relationships, temp_dir):
        """Test enhanced confidence calculation."""
        pdf_path = str(Path(temp_dir) / "test.pdf")
        Path(pdf_path).write_text("Test PDF content")
        
        enhanced_confidence, breakdown = scorer.calculate_enhanced_confidence(
            base_confidence=0.7,
            entities_used=sample_entities,
            relationships_used=sample_relationships,
            source_documents=[pdf_path]
        )
        
        assert 0.1 <= enhanced_confidence <= 1.0
        assert isinstance(breakdown, dict)
        assert "base_confidence" in breakdown
        assert "enhanced_confidence" in breakdown
        assert "graph_evidence" in breakdown
        assert "source_reliability" in breakdown
    
    def test_calculate_graph_evidence_metrics(self, scorer, sample_entities, sample_relationships, temp_dir):
        """Test graph evidence metrics calculation."""
        pdf_path = str(Path(temp_dir) / "test.pdf")
        
        metrics = scorer._calculate_graph_evidence_metrics(
            sample_entities, sample_relationships, [pdf_path]
        )
        
        assert isinstance(metrics, GraphEvidenceMetrics)
        assert 0.0 <= metrics.entity_support <= 1.0
        assert 0.0 <= metrics.relationship_support <= 1.0
        assert 0.0 <= metrics.connectivity_score <= 1.0
        assert 0.0 <= metrics.consistency_score <= 1.0
        assert 0.0 <= metrics.diversity_score <= 1.0
    
    def test_calculate_connectivity_score(self, scorer):
        """Test connectivity score calculation."""
        entities = [
            {"id": "e1", "text": "entity1"},
            {"id": "e2", "text": "entity2"},
            {"id": "e3", "text": "entity3"}
        ]
        
        relationships = [
            {"source": "e1", "target": "e2"},
            {"source": "e2", "target": "e3"}
        ]
        
        connectivity = scorer._calculate_connectivity_score(entities, relationships)
        
        assert 0.0 <= connectivity <= 1.0
        assert connectivity > 0.5  # Should be high due to good connectivity
    
    def test_calculate_consistency_score(self, scorer, temp_dir):
        """Test consistency score calculation."""
        pdf_path = str(Path(temp_dir) / "test.pdf")
        
        entities = [{"source_documents": [pdf_path]}]
        relationships = [{"source_documents": [pdf_path]}]
        
        consistency = scorer._calculate_consistency_score(
            entities, relationships, [pdf_path]
        )
        
        assert 0.0 <= consistency <= 1.0
        assert consistency > 0.5  # Should be high due to shared sources
    
    def test_calculate_diversity_score(self, scorer, temp_dir):
        """Test diversity score calculation."""
        # Single source
        single_diversity = scorer._calculate_diversity_score([], [], ["source1.pdf"])
        assert single_diversity == 0.4
        
        # Multiple sources
        multi_diversity = scorer._calculate_diversity_score(
            [], [], ["source1.pdf", "source2.pdf", "source3.pdf"]
        )
        assert multi_diversity > single_diversity
    
    def test_calculate_source_reliability_scores(self, scorer, sample_entities, sample_relationships, temp_dir):
        """Test source reliability score calculation."""
        pdf_path = str(Path(temp_dir) / "test.pdf")
        Path(pdf_path).write_text("Test PDF content")
        
        reliability_scores = scorer._calculate_source_reliability_scores(
            [pdf_path], sample_entities, sample_relationships
        )
        
        assert len(reliability_scores) == 1
        assert isinstance(reliability_scores[0], SourceReliabilityScore)
        assert reliability_scores[0].document_path == pdf_path
        assert 0.1 <= reliability_scores[0].reliability_score <= 1.0
        assert "file_accessibility" in reliability_scores[0].factors
    
    def test_check_file_accessibility(self, scorer, temp_dir):
        """Test file accessibility checking."""
        # Existing file
        existing_file = Path(temp_dir) / "existing.pdf"
        existing_file.write_text("Test content")
        
        accessibility = scorer._check_file_accessibility(str(existing_file))
        assert accessibility == 1.0
        
        # Non-existing file
        non_existing = str(Path(temp_dir) / "nonexistent.pdf")
        accessibility = scorer._check_file_accessibility(non_existing)
        assert accessibility == 0.0
    
    def test_assess_metadata_completeness(self, scorer, temp_dir):
        """Test metadata completeness assessment."""
        # Well-named file with year
        good_file = Path(temp_dir) / "clinical_metabolomics_2023.pdf"
        good_file.write_text("Test content" * 100)  # Reasonable size
        
        completeness = scorer._assess_metadata_completeness(str(good_file))
        assert completeness > 0.5
        
        # Poorly named file
        poor_file = Path(temp_dir) / "x.pdf"
        poor_file.write_text("x")
        
        completeness = scorer._assess_metadata_completeness(str(poor_file))
        assert completeness < 0.8
    
    def test_calculate_citation_frequency(self, scorer, temp_dir):
        """Test citation frequency calculation."""
        pdf_path = str(Path(temp_dir) / "test.pdf")
        
        entities = [
            {"source_documents": [pdf_path]},
            {"source_documents": ["other.pdf"]}
        ]
        
        relationships = [
            {"source_documents": [pdf_path]}
        ]
        
        frequency = scorer._calculate_citation_frequency(pdf_path, entities, relationships)
        
        assert 0.0 <= frequency <= 1.0
        # Should be 2/3 = 0.67 (2 citations out of 3 total)
        assert frequency > 0.5
    
    def test_assess_content_quality(self, scorer, temp_dir):
        """Test content quality assessment."""
        pdf_path = str(Path(temp_dir) / "test.pdf")
        
        # High quality entities and relationships
        high_quality_entities = [
            {"source_documents": [pdf_path], "relevance_score": 0.9}
        ]
        
        high_quality_relationships = [
            {"source_documents": [pdf_path], "confidence": 0.85}
        ]
        
        quality = scorer._assess_content_quality(
            pdf_path, high_quality_entities, high_quality_relationships
        )
        
        assert quality > 0.8
    
    def test_assess_document_recency(self, scorer, temp_dir):
        """Test document recency assessment."""
        # File with current year in name
        current_year_file = Path(temp_dir) / "document_2024.pdf"
        current_year_file.write_text("Test")
        
        recency = scorer._assess_document_recency(str(current_year_file))
        assert recency >= 0.9  # Should be high for current year
        
        # File with old year in name
        old_year_file = Path(temp_dir) / "document_2010.pdf"
        old_year_file.write_text("Test")
        
        recency = scorer._assess_document_recency(str(old_year_file))
        assert recency <= 0.5  # Should be lower for old year
    
    def test_assess_source_diversity(self, scorer):
        """Test source diversity assessment."""
        # Similar sources
        similar_sources = ["metabolomics_study_2023.pdf", "metabolomics_research_2023.pdf"]
        diversity = scorer._assess_source_diversity(similar_sources[0], similar_sources)
        
        # Diverse sources
        diverse_sources = ["metabolomics_2023.pdf", "cardiology_2022.pdf"]
        diversity2 = scorer._assess_source_diversity(diverse_sources[0], diverse_sources)
        
        assert diversity2 > diversity  # More diverse sources should score higher
    
    def test_calculate_citation_confidence_scores(self, scorer, sample_citation_map, temp_dir):
        """Test citation confidence score calculation."""
        pdf_path = str(Path(temp_dir) / "test.pdf")
        Path(pdf_path).write_text("Test content")
        
        reliability_scores = [
            SourceReliabilityScore(pdf_path, 0.8, {}, {})
        ]
        
        graph_metrics = GraphEvidenceMetrics(0.8, 0.7, 0.6, 0.7, 0.5)
        
        citation_confidences = scorer._calculate_citation_confidence_scores(
            sample_citation_map, reliability_scores, graph_metrics
        )
        
        assert len(citation_confidences) == 1
        assert isinstance(citation_confidences[0], CitationConfidenceScore)
        assert 0.1 <= citation_confidences[0].confidence_score <= 1.0
    
    def test_combine_confidence_factors(self, scorer):
        """Test confidence factor combination."""
        graph_metrics = GraphEvidenceMetrics(0.8, 0.7, 0.6, 0.7, 0.5)
        reliability_scores = [SourceReliabilityScore("test.pdf", 0.8, {}, {})]
        citation_confidences = [CitationConfidenceScore("1", 0.8, 0.8, 0.7, {})]
        
        combined = scorer._combine_confidence_factors(
            0.7, graph_metrics, reliability_scores, citation_confidences
        )
        
        assert 0.1 <= combined <= 1.0
    
    def test_format_confidence_for_ui(self, scorer):
        """Test confidence formatting for UI."""
        citation_confidences = [
            CitationConfidenceScore("1", 0.85, 0.8, 0.7, {})
        ]
        
        breakdown = {
            "graph_evidence": {"entity_support": 0.8},
            "source_reliability": {"average_reliability": 0.8},
            "improvement": 0.1
        }
        
        ui_format = scorer.format_confidence_for_ui(0.85, citation_confidences, breakdown)
        
        assert "overall" in ui_format
        assert "citations" in ui_format
        assert "breakdown_summary" in ui_format
        assert "recommendations" in ui_format
        
        assert ui_format["overall"]["level"] == "High"
        assert ui_format["overall"]["color"] == "green"
        assert ui_format["overall"]["score"] == 0.85
    
    def test_get_confidence_level(self, scorer):
        """Test confidence level categorization."""
        assert scorer._get_confidence_level(0.9) == "High"
        assert scorer._get_confidence_level(0.7) == "Medium"
        assert scorer._get_confidence_level(0.5) == "Low"
        assert scorer._get_confidence_level(0.2) == "Very Low"
    
    def test_get_confidence_color(self, scorer):
        """Test confidence color assignment."""
        assert scorer._get_confidence_color(0.9) == "green"
        assert scorer._get_confidence_color(0.7) == "orange"
        assert scorer._get_confidence_color(0.5) == "red"
        assert scorer._get_confidence_color(0.2) == "darkred"
    
    def test_generate_confidence_recommendations(self, scorer):
        """Test confidence recommendation generation."""
        # Low confidence scenario
        low_breakdown = {
            "source_reliability": {"average_reliability": 0.4},
            "graph_evidence": {"entity_support": 0.3, "connectivity_score": 0.3},
            "citation_confidence": {"low_confidence_citations": 2}
        }
        
        recommendations = scorer._generate_confidence_recommendations(0.4, low_breakdown)
        
        assert len(recommendations) > 0
        assert any("additional sources" in rec.lower() for rec in recommendations)
        
        # High confidence scenario
        high_breakdown = {
            "source_reliability": {"average_reliability": 0.9},
            "graph_evidence": {"entity_support": 0.9, "connectivity_score": 0.8},
            "citation_confidence": {"low_confidence_citations": 0}
        }
        
        recommendations = scorer._generate_confidence_recommendations(0.9, high_breakdown)
        
        assert len(recommendations) > 0
        assert any("acceptable" in rec.lower() for rec in recommendations)
    
    def test_error_handling(self, scorer):
        """Test error handling in confidence calculation."""
        # Test with invalid data
        enhanced_confidence, breakdown = scorer.calculate_enhanced_confidence(
            base_confidence=0.7,
            entities_used=[{"invalid": "data"}],
            relationships_used=[{"invalid": "data"}],
            source_documents=["nonexistent.pdf"]
        )
        
        # Should return base confidence on error
        assert enhanced_confidence >= 0.1
        assert "error" in breakdown or enhanced_confidence == 0.7


class TestDataClasses:
    """Test the data classes used in confidence scoring."""
    
    def test_source_reliability_score(self):
        """Test SourceReliabilityScore dataclass."""
        score = SourceReliabilityScore(
            document_path="test.pdf",
            reliability_score=0.8,
            factors={"accessibility": 1.0},
            metadata={"size": 1000}
        )
        
        assert score.document_path == "test.pdf"
        assert score.reliability_score == 0.8
        assert score.factors["accessibility"] == 1.0
        assert score.metadata["size"] == 1000
    
    def test_citation_confidence_score(self):
        """Test CitationConfidenceScore dataclass."""
        score = CitationConfidenceScore(
            citation_id="1",
            confidence_score=0.85,
            reliability_score=0.8,
            evidence_strength=0.7,
            factors={"base": 0.8}
        )
        
        assert score.citation_id == "1"
        assert score.confidence_score == 0.85
        assert score.reliability_score == 0.8
        assert score.evidence_strength == 0.7
        assert score.factors["base"] == 0.8
    
    def test_graph_evidence_metrics(self):
        """Test GraphEvidenceMetrics dataclass."""
        metrics = GraphEvidenceMetrics(
            entity_support=0.8,
            relationship_support=0.7,
            connectivity_score=0.6,
            consistency_score=0.7,
            diversity_score=0.5
        )
        
        assert metrics.entity_support == 0.8
        assert metrics.relationship_support == 0.7
        assert metrics.connectivity_score == 0.6
        assert metrics.consistency_score == 0.7
        assert metrics.diversity_score == 0.5


# Integration tests
class TestConfidenceScoringIntegration:
    """Integration tests for confidence scoring system."""
    
    def test_end_to_end_confidence_scoring(self, tmp_path):
        """Test complete confidence scoring workflow."""
        # Create test files
        pdf1 = tmp_path / "clinical_metabolomics_2023.pdf"
        pdf2 = tmp_path / "biomarker_discovery_2022.pdf"
        pdf1.write_text("Clinical metabolomics content")
        pdf2.write_text("Biomarker discovery content")
        
        # Create test data
        entities = [
            {
                "id": "e1",
                "text": "clinical metabolomics",
                "type": "field",
                "relevance_score": 0.9,
                "source_documents": [str(pdf1)]
            },
            {
                "id": "e2",
                "text": "biomarker",
                "type": "concept",
                "relevance_score": 0.8,
                "source_documents": [str(pdf1), str(pdf2)]
            }
        ]
        
        relationships = [
            {
                "id": "r1",
                "type": "involves",
                "source": "e1",
                "target": "e2",
                "confidence": 0.85,
                "source_documents": [str(pdf1)]
            }
        ]
        
        # Create citation map
        from .citation_formatter import PDFCitation
        citation_map = {
            "1": PDFCitation("1", str(pdf1), confidence_score=0.9),
            "2": PDFCitation("2", str(pdf2), confidence_score=0.8)
        }
        
        # Run confidence scoring
        scorer = LightRAGConfidenceScorer()
        enhanced_confidence, breakdown = scorer.calculate_enhanced_confidence(
            base_confidence=0.7,
            entities_used=entities,
            relationships_used=relationships,
            source_documents=[str(pdf1), str(pdf2)],
            citation_map=citation_map
        )
        
        # Verify results
        assert 0.1 <= enhanced_confidence <= 1.0
        assert enhanced_confidence >= 0.7  # Should be at least as good as base
        
        # Verify breakdown structure
        assert "base_confidence" in breakdown
        assert "enhanced_confidence" in breakdown
        assert "graph_evidence" in breakdown
        assert "source_reliability" in breakdown
        assert "citation_confidence" in breakdown
        
        # Test UI formatting
        citation_confidences = [
            CitationConfidenceScore("1", 0.9, 0.85, 0.8, {}),
            CitationConfidenceScore("2", 0.8, 0.75, 0.7, {})
        ]
        
        ui_format = scorer.format_confidence_for_ui(
            enhanced_confidence, citation_confidences, breakdown
        )
        
        assert "overall" in ui_format
        assert "citations" in ui_format
        assert len(ui_format["citations"]) == 2
        assert "recommendations" in ui_format


if __name__ == "__main__":
    pytest.main([__file__])