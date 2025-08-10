"""
Simple test for LightRAG Citation Formatter
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from lightrag_integration.citation_formatter import LightRAGCitationFormatter, PDFCitation


def test_citation_formatter_basic():
    """Test basic citation formatter functionality."""
    print("Testing LightRAG Citation Formatter...")
    
    # Create formatter
    formatter = LightRAGCitationFormatter()
    print("✓ Formatter created successfully")
    
    # Test PDF detection
    assert formatter._is_pdf_document("test.pdf") == True
    assert formatter._is_pdf_document("test.txt") == False
    print("✓ PDF detection works")
    
    # Test confidence levels
    assert formatter._get_confidence_level(0.9) == "High"
    assert formatter._get_confidence_level(0.7) == "Medium"
    assert formatter._get_confidence_level(0.5) == "Low"
    print("✓ Confidence levels work")
    
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test PDF files
        pdf_files = []
        for filename in ["test1.pdf", "test2.pdf"]:
            pdf_path = Path(temp_dir) / filename
            pdf_path.write_text(f"Mock PDF content for {filename}")
            pdf_files.append(str(pdf_path))
        
        # Test metadata extraction
        metadata = formatter._extract_pdf_metadata(pdf_files[0])
        assert "filename" in metadata
        assert "title" in metadata
        print("✓ Metadata extraction works")
        
        # Test citation generation
        sample_entities = [
            {
                "id": "entity_1",
                "text": "test entity",
                "type": "concept",
                "relevance_score": 0.8,
                "source_documents": [pdf_files[0]]
            }
        ]
        
        sample_relationships = [
            {
                "id": "rel_1",
                "type": "relates_to",
                "confidence": 0.7,
                "source_documents": [pdf_files[0]]
            }
        ]
        
        citations = formatter._generate_pdf_citations(
            pdf_files, sample_entities, sample_relationships
        )
        
        assert len(citations) == 2
        assert all(isinstance(c, PDFCitation) for c in citations)
        print("✓ Citation generation works")
        
        # Test full citation formatting
        result = formatter.format_lightrag_citations(
            content="This is test content about concepts.",
            source_documents=pdf_files,
            entities_used=sample_entities,
            relationships_used=sample_relationships,
            confidence_score=0.8
        )
        
        assert result.formatted_content
        assert result.bibliography
        assert len(result.citation_map) == 2
        print("✓ Full citation formatting works")
        
        print("\n✅ All tests passed!")
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_citation_formatter_basic()