"""
Test just the new citation functions without importing the full citation module
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def format_lightrag_citation_confidence(confidence_score: float, confidence_breakdown: dict = None) -> str:
    """
    Format confidence score information for LightRAG citations.
    
    Args:
        confidence_score: Overall confidence score
        confidence_breakdown: Detailed confidence breakdown
    
    Returns:
        Formatted confidence information
    """
    try:
        confidence_level = "High" if confidence_score >= 0.8 else \
                          "Medium" if confidence_score >= 0.6 else \
                          "Low" if confidence_score >= 0.4 else "Very Low"
        
        confidence_info = f"**Response Confidence: {confidence_level} ({confidence_score:.2f})**"
        
        if confidence_breakdown:
            # Add detailed breakdown if available
            factors = confidence_breakdown.get("evidence_factors", {})
            if factors:
                confidence_info += "\n\n*Confidence factors:*"
                for factor, value in factors.items():
                    factor_name = factor.replace("_", " ").title()
                    confidence_info += f"\n- {factor_name}: {value:.2f}"
        
        return confidence_info
        
    except Exception as e:
        return f"Confidence: {confidence_score:.2f}"


def merge_lightrag_with_existing_citations(
    lightrag_content: str,
    lightrag_bibliography: str,
    existing_content: str,
    existing_bibliography: str
) -> tuple[str, str]:
    """
    Merge LightRAG citations with existing system citations.
    
    Args:
        lightrag_content: Content with LightRAG citations
        lightrag_bibliography: LightRAG bibliography
        existing_content: Content with existing citations
        existing_bibliography: Existing bibliography
    
    Returns:
        Tuple of (merged_content, merged_bibliography)
    """
    try:
        # Combine content - prioritize LightRAG content if available
        merged_content = lightrag_content if lightrag_content.strip() else existing_content
        
        # Combine bibliographies
        merged_bibliography = ""
        
        if lightrag_bibliography and lightrag_bibliography.strip():
            merged_bibliography += lightrag_bibliography
        
        if existing_bibliography and existing_bibliography.strip():
            if merged_bibliography:
                merged_bibliography += "\n\n---\n\n"
            merged_bibliography += existing_bibliography
        
        return merged_content, merged_bibliography
        
    except Exception as e:
        return lightrag_content or existing_content, lightrag_bibliography or existing_bibliography


def validate_lightrag_citations(citation_map: dict) -> dict:
    """
    Validate LightRAG citations and return validation results.
    
    Args:
        citation_map: Dictionary of citation ID to citation info
    
    Returns:
        Dictionary with validation results
    """
    try:
        validation_results = {
            "valid_citations": 0,
            "invalid_citations": 0,
            "missing_files": [],
            "accessible_files": [],
            "total_citations": len(citation_map)
        }
        
        for citation_id, citation in citation_map.items():
            try:
                file_path = citation.file_path if hasattr(citation, 'file_path') else citation.get('file_path', '')
                
                if not file_path:
                    validation_results["invalid_citations"] += 1
                    continue
                
                # Check if file exists
                path = Path(file_path)
                
                if path.exists():
                    validation_results["valid_citations"] += 1
                    validation_results["accessible_files"].append(file_path)
                else:
                    validation_results["invalid_citations"] += 1
                    validation_results["missing_files"].append(file_path)
                    
            except Exception as e:
                validation_results["invalid_citations"] += 1
        
        return validation_results
        
    except Exception as e:
        return {"error": str(e), "total_citations": len(citation_map)}


def test_new_citation_functions():
    """Test the new citation functions."""
    print("Testing new citation functions...")
    
    # Test confidence formatting
    confidence_info = format_lightrag_citation_confidence(0.85)
    assert "High" in confidence_info
    assert "0.85" in confidence_info
    print("✓ Confidence formatting works")
    
    # Test confidence formatting with breakdown
    breakdown = {
        "evidence_factors": {
            "entity_confidence": 0.9,
            "relationship_confidence": 0.8
        }
    }
    detailed_info = format_lightrag_citation_confidence(0.85, breakdown)
    assert "Entity Confidence" in detailed_info
    assert "0.9" in detailed_info
    print("✓ Detailed confidence formatting works")
    
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
    assert "---" in merged_bibliography
    print("✓ Citation merging works")
    
    # Test citation validation
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test file
        test_file = Path(temp_dir) / "test.pdf"
        test_file.write_text("Test content")
        
        citation_map = {
            "1": Mock(file_path=str(test_file)),
            "2": Mock(file_path="/nonexistent/file.pdf"),
            "3": {"file_path": str(test_file)}  # Dict format
        }
        
        validation_results = validate_lightrag_citations(citation_map)
        assert validation_results["total_citations"] == 3
        assert validation_results["valid_citations"] == 2
        assert validation_results["invalid_citations"] == 1
        assert len(validation_results["accessible_files"]) == 2
        assert len(validation_results["missing_files"]) == 1
        print("✓ Citation validation works")
        
    finally:
        shutil.rmtree(temp_dir)
    
    print("\n✅ All new citation function tests passed!")


if __name__ == "__main__":
    test_new_citation_functions()