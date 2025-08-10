"""
Simple test for citation.py LightRAG integration functions
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from citation import (
    process_lightrag_citations,
    merge_lightrag_with_existing_citations,
    format_lightrag_citation_confidence,
    validate_lightrag_citations
)


def test_citation_integration():
    """Test citation integration functions."""
    print("Testing citation integration functions...")
    
    # Test confidence formatting
    confidence_info = format_lightrag_citation_confidence(0.85)
    assert "High" in confidence_info
    assert "0.85" in confidence_info
    print("✓ Confidence formatting works")
    
    # Test merging citations
    merged_content, merged_bibliography = merge_lightrag_with_existing_citations(
        "LightRAG content [1]",
        "### LightRAG Sources\n[1] LightRAG source",
        "Existing content [2]",
        "### Existing Sources\n[2] Existing source"
    )
    
    assert merged_content == "LightRAG content [1]"
    assert "LightRAG source" in merged_bibliography
    assert "Existing source" in merged_bibliography
    print("✓ Citation merging works")
    
    # Test citation validation with mock citations
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test file
        test_file = Path(temp_dir) / "test.pdf"
        test_file.write_text("Test content")
        
        citation_map = {
            "1": Mock(file_path=str(test_file)),
            "2": Mock(file_path="/nonexistent/file.pdf")
        }
        
        validation_results = validate_lightrag_citations(citation_map)
        assert validation_results["total_citations"] == 2
        assert validation_results["valid_citations"] == 1
        assert validation_results["invalid_citations"] == 1
        print("✓ Citation validation works")
        
    finally:
        shutil.rmtree(temp_dir)
    
    # Test LightRAG citation processing with mock
    sample_response = {
        "answer": "Test answer",
        "confidence_score": 0.8,
        "source_documents": ["test.pdf"],
        "entities_used": [],
        "relationships_used": []
    }
    
    # Mock the LightRAG formatter since it might not be available in all environments
    with patch('citation.LightRAGCitationFormatter') as mock_formatter_class:
        mock_result = Mock()
        mock_result.formatted_content = "Test answer [1]"
        mock_result.bibliography = "### Sources\n[1] test.pdf"
        
        mock_formatter = Mock()
        mock_formatter_class.return_value = mock_formatter
        mock_formatter.format_lightrag_citations.return_value = mock_result
        
        content, bibliography = process_lightrag_citations(
            "Test answer",
            sample_response
        )
        
        assert content == "Test answer [1]"
        assert "test.pdf" in bibliography
        print("✓ LightRAG citation processing works")
    
    print("\n✅ All integration tests passed!")


if __name__ == "__main__":
    test_citation_integration()