"""
Simple test for LightRAG Confidence Scoring
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from lightrag_integration.confidence_scoring import (
    LightRAGConfidenceScorer,
    SourceReliabilityScore,
    CitationConfidenceScore,
    GraphEvidenceMetrics
)
from lightrag_integration.citation_formatter import PDFCitation


def test_confidence_scoring():
    """Test confidence scoring functionality."""
    print("Testing LightRAG Confidence Scoring...")
    
    # Create scorer
    scorer = LightRAGConfidenceScorer()
    print("✓ Scorer created successfully")
    
    # Test confidence levels
    assert scorer._get_confidence_level(0.9) == "High"
    assert scorer._get_confidence_level(0.7) == "Medium"
    assert scorer._get_confidence_level(0.5) == "Low"
    print("✓ Confidence levels work")
    
    # Test confidence colors
    assert scorer._get_confidence_color(0.9) == "green"
    assert scorer._get_confidence_color(0.7) == "orange"
    assert scorer._get_confidence_color(0.5) == "red"
    print("✓ Confidence colors work")
    
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test PDF files
        pdf1 = Path(temp_dir) / "clinical_metabolomics_2024.pdf"
        pdf2 = Path(temp_dir) / "biomarker_study_2022.pdf"
        pdf1.write_text("Clinical metabolomics research content")
        pdf2.write_text("Biomarker discovery study content")
        
        # Test file accessibility
        accessibility = scorer._check_file_accessibility(str(pdf1))
        assert accessibility == 1.0
        print("✓ File accessibility check works")
        
        # Test metadata completeness
        completeness = scorer._assess_metadata_completeness(str(pdf1))
        assert completeness > 0.5  # Should be decent due to informative filename
        print("✓ Metadata completeness assessment works")
        
        # Test document recency
        recency = scorer._assess_document_recency(str(pdf1))
        assert recency > 0.8  # Should be high due to 2024 in filename
        print("✓ Document recency assessment works")
        
        # Create sample entities and relationships
        entities = [
            {
                "id": "entity_1",
                "text": "clinical metabolomics",
                "type": "field",
                "relevance_score": 0.9,
                "source_documents": [str(pdf1)]
            },
            {
                "id": "entity_2",
                "text": "biomarker",
                "type": "concept",
                "relevance_score": 0.8,
                "source_documents": [str(pdf1), str(pdf2)]
            }
        ]
        
        relationships = [
            {
                "id": "rel_1",
                "type": "involves",
                "source": "entity_1",
                "target": "entity_2",
                "confidence": 0.85,
                "source_documents": [str(pdf1)]
            }
        ]
        
        # Test graph evidence metrics
        metrics = scorer._calculate_graph_evidence_metrics(
            entities, relationships, [str(pdf1), str(pdf2)]
        )
        
        assert isinstance(metrics, GraphEvidenceMetrics)
        assert 0.0 <= metrics.entity_support <= 1.0
        assert 0.0 <= metrics.relationship_support <= 1.0
        print("✓ Graph evidence metrics calculation works")
        
        # Test source reliability scores
        reliability_scores = scorer._calculate_source_reliability_scores(
            [str(pdf1), str(pdf2)], entities, relationships
        )
        
        assert len(reliability_scores) == 2
        assert all(isinstance(score, SourceReliabilityScore) for score in reliability_scores)
        assert all(0.1 <= score.reliability_score <= 1.0 for score in reliability_scores)
        print("✓ Source reliability scoring works")
        
        # Test citation confidence scores
        citation_map = {
            "1": PDFCitation("1", str(pdf1), confidence_score=0.9),
            "2": PDFCitation("2", str(pdf2), confidence_score=0.8)
        }
        
        citation_confidences = scorer._calculate_citation_confidence_scores(
            citation_map, reliability_scores, metrics
        )
        
        assert len(citation_confidences) == 2
        assert all(isinstance(score, CitationConfidenceScore) for score in citation_confidences)
        print("✓ Citation confidence scoring works")
        
        # Test enhanced confidence calculation
        enhanced_confidence, breakdown = scorer.calculate_enhanced_confidence(
            base_confidence=0.7,
            entities_used=entities,
            relationships_used=relationships,
            source_documents=[str(pdf1), str(pdf2)],
            citation_map=citation_map
        )
        
        assert 0.1 <= enhanced_confidence <= 1.0
        assert isinstance(breakdown, dict)
        assert "base_confidence" in breakdown
        assert "enhanced_confidence" in breakdown
        print("✓ Enhanced confidence calculation works")
        
        # Test UI formatting
        ui_format = scorer.format_confidence_for_ui(
            enhanced_confidence, citation_confidences, breakdown
        )
        
        assert "overall" in ui_format
        assert "citations" in ui_format
        assert "breakdown_summary" in ui_format
        assert "recommendations" in ui_format
        print("✓ UI formatting works")
        
        # Test recommendations
        recommendations = scorer._generate_confidence_recommendations(
            enhanced_confidence, breakdown
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        print("✓ Recommendation generation works")
        
        print(f"\n✅ All tests passed!")
        print(f"Enhanced confidence: {enhanced_confidence:.3f} (base: 0.7)")
        print(f"UI confidence level: {ui_format['overall']['level']}")
        print(f"Number of recommendations: {len(recommendations)}")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_confidence_scoring()