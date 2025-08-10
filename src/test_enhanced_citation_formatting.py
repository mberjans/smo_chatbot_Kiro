"""
Test enhanced citation formatting with confidence scoring
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from lightrag_integration.citation_formatter import LightRAGCitationFormatter


def test_enhanced_citation_formatting():
    """Test enhanced citation formatting with confidence scoring."""
    print("Testing Enhanced Citation Formatting...")
    
    # Create formatter
    formatter = LightRAGCitationFormatter()
    print("✓ Enhanced formatter created successfully")
    
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test PDF files
        pdf1 = Path(temp_dir) / "clinical_metabolomics_2024.pdf"
        pdf2 = Path(temp_dir) / "biomarker_study_2023.pdf"
        pdf1.write_text("Clinical metabolomics research content")
        pdf2.write_text("Biomarker discovery study content")
        
        # Create sample data
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
        
        content = "Clinical metabolomics is important for biomarker discovery and analysis."
        
        # Test enhanced citation formatting
        citation_result, confidence_ui = formatter.format_lightrag_citations_with_confidence(
            content=content,
            source_documents=[str(pdf1), str(pdf2)],
            entities_used=entities,
            relationships_used=relationships,
            base_confidence_score=0.7
        )
        
        # Verify citation result
        assert citation_result.formatted_content
        assert citation_result.bibliography
        assert citation_result.citation_map
        assert citation_result.confidence_scores
        print("✓ Enhanced citation result generated")
        
        # Verify confidence UI format
        assert "overall" in confidence_ui
        assert "score" in confidence_ui["overall"]
        assert "level" in confidence_ui["overall"]
        assert "color" in confidence_ui["overall"]
        print("✓ Confidence UI format generated")
        
        # Check that enhanced confidence is available
        assert "enhanced_overall" in citation_result.confidence_scores
        enhanced_confidence = citation_result.confidence_scores["enhanced_overall"]
        assert 0.1 <= enhanced_confidence <= 1.0
        print(f"✓ Enhanced confidence calculated: {enhanced_confidence:.3f}")
        
        # Check bibliography contains enhanced information
        assert "Enhanced Confidence" in citation_result.bibliography
        assert "Confidence Analysis" in citation_result.bibliography or "reliability" in citation_result.bibliography.lower()
        print("✓ Enhanced bibliography generated")
        
        # Check UI confidence level
        ui_level = confidence_ui["overall"]["level"]
        ui_score = confidence_ui["overall"]["score"]
        print(f"✓ UI confidence: {ui_level} ({ui_score:.3f})")
        
        # Test that citations are properly formatted
        assert "[1]" in citation_result.formatted_content or "[2]" in citation_result.formatted_content
        print("✓ Citation markers inserted")
        
        print(f"\n✅ All enhanced citation formatting tests passed!")
        print(f"Base confidence: 0.7")
        print(f"Enhanced confidence: {enhanced_confidence:.3f}")
        print(f"UI level: {ui_level}")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_enhanced_citation_formatting()