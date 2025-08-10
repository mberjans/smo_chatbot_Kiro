"""
Unit tests for LightRAG citation integration in citation.py

Tests the integration functions that connect LightRAG citations
with the existing citation system.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from citation import (
    process_lightrag_citations,
    merge_lightrag_with_existing_citations,
    format_lightrag_citation_confidence,
    validate_lightrag_citations
)


class TestLightRAGCitationIntegration:
    """Test cases for LightRAG citation integration functions."""
    
    @pytest.fixture
    def sample_lightrag_response(self):
        """Create sample LightRAG response."""
        return {
            "answer": "Clinical metabolomics is important for biomarker discovery.",
            "confidence_score": 0.85,
            "source_documents": ["clinical_metabolomics_2023.pdf", "biomarker_study_2022.pdf"],
            "entities_used": [
                {
                    "id": "entity_1",
                    "text": "clinical metabolomics",
                    "type": "field",
                    "relevance_score": 0.9,
                    "source_documents": ["clinical_metabolomics_2023.pdf"]
                }
            ],
            "relationships_used": [
                {
                    "id": "rel_1",
                    "type": "involves",
                    "confidence": 0.8,
                    "source_documents": ["clinical_metabolomics_2023.pdf"]
                }
            ],
            "confidence_breakdown": {
                "base_confidence": 0.8,
                "enhancement": 0.05,
                "evidence_factors": {
                    "entity_confidence": 0.9,
                    "relationship_confidence": 0.8,
                    "source_diversity": 0.2
                }
            }
        }
    
    @pytest.fixture
    def mock_citation_result(self):
        """Create mock citation result."""
        from lightrag_integration.citation_formatter import LightRAGCitationResult, PDFCitation
        
        return LightRAGCitationResult(
            formatted_content="Clinical metabolomics is important [1] [2]",
            bibliography="### Sources\n[1] clinical_metabolomics_2023.pdf\n[2] biomarker_study_2022.pdf",
            citation_map={
                "1": PDFCitation("1", "clinical_metabolomics_2023.pdf", confidence_score=0.9),
                "2": PDFCitation("2", "biomarker_study_2022.pdf", confidence_score=0.8)
            },
            confidence_scores={"1": 0.9, "2": 0.8}
        )
    
    def test_process_lightrag_citations_success(self, sample_lightrag_response, mock_citation_result):
        """Test successful processing of LightRAG citations."""
        with patch('citation.LightRAGCitationFormatter') as mock_formatter_class:
            mock_formatter = Mock()
            mock_formatter_class.return_value = mock_formatter
            mock_formatter.format_lightrag_citations.return_value = mock_citation_result
            
            content, bibliography = process_lightrag_citations(
                "Clinical metabolomics is important.",
                sample_lightrag_response
            )
            
            assert content == mock_citation_result.formatted_content
            assert bibliography == mock_citation_result.bibliography
            mock_formatter.format_lightrag_citations.assert_called_once()
    
    def test_process_lightrag_citations_import_error(self, sample_lightrag_response):
        """Test handling of import error for LightRAG citation formatter."""
        with patch('citation.LightRAGCitationFormatter', side_effect=ImportError("Module not found")):
            content, bibliography = process_lightrag_citations(
                "Test content",
                sample_lightrag_response
            )
            
            assert content == "Test content"
            assert bibliography == ""
    
    def test_process_lightrag_citations_processing_error(self, sample_lightrag_response):
        """Test handling of processing error."""
        with patch('citation.LightRAGCitationFormatter') as mock_formatter_class:
            mock_formatter = Mock()
            mock_formatter_class.return_value = mock_formatter
            mock_formatter.format_lightrag_citations.side_effect = Exception("Processing error")
            
            content, bibliography = process_lightrag_citations(
                "Test content",
                sample_lightrag_response
            )
            
            assert content == "Test content"
            assert "Error generating citations" in bibliography
    
    def test_process_lightrag_citations_empty_response(self):
        """Test processing with empty LightRAG response."""
        empty_response = {
            "source_documents": [],
            "entities_used": [],
            "relationships_used": [],
            "confidence_score": 0.0
        }
        
        with patch('citation.LightRAGCitationFormatter') as mock_formatter_class:
            mock_formatter = Mock()
            mock_formatter_class.return_value = mock_formatter
            mock_formatter.format_lightrag_citations.return_value = Mock(
                formatted_content="Test content",
                bibliography=""
            )
            
            content, bibliography = process_lightrag_citations(
                "Test content",
                empty_response
            )
            
            assert content == "Test content"
            mock_formatter.format_lightrag_citations.assert_called_once()
    
    def test_merge_lightrag_with_existing_citations_both_present(self):
        """Test merging when both LightRAG and existing citations are present."""
        lightrag_content = "LightRAG content [1]"
        lightrag_bibliography = "### LightRAG Sources\n[1] LightRAG source"
        existing_content = "Existing content [2]"
        existing_bibliography = "### Existing Sources\n[2] Existing source"
        
        merged_content, merged_bibliography = merge_lightrag_with_existing_citations(
            lightrag_content, lightrag_bibliography, existing_content, existing_bibliography
        )
        
        assert merged_content == lightrag_content  # LightRAG content takes priority
        assert "LightRAG source" in merged_bibliography
        assert "Existing source" in merged_bibliography
        assert "---" in merged_bibliography  # Separator
    
    def test_merge_lightrag_with_existing_citations_only_lightrag(self):
        """Test merging when only LightRAG citations are present."""
        lightrag_content = "LightRAG content [1]"
        lightrag_bibliography = "### LightRAG Sources\n[1] LightRAG source"
        
        merged_content, merged_bibliography = merge_lightrag_with_existing_citations(
            lightrag_content, lightrag_bibliography, "", ""
        )
        
        assert merged_content == lightrag_content
        assert merged_bibliography == lightrag_bibliography
    
    def test_merge_lightrag_with_existing_citations_only_existing(self):
        """Test merging when only existing citations are present."""
        existing_content = "Existing content [2]"
        existing_bibliography = "### Existing Sources\n[2] Existing source"
        
        merged_content, merged_bibliography = merge_lightrag_with_existing_citations(
            "", "", existing_content, existing_bibliography
        )
        
        assert merged_content == existing_content
        assert merged_bibliography == existing_bibliography
    
    def test_merge_lightrag_with_existing_citations_empty(self):
        """Test merging when both are empty."""
        merged_content, merged_bibliography = merge_lightrag_with_existing_citations(
            "", "", "", ""
        )
        
        assert merged_content == ""
        assert merged_bibliography == ""
    
    def test_merge_lightrag_with_existing_citations_error_handling(self):
        """Test error handling in merge function."""
        # Simulate an error by passing None values
        with patch('citation.logger') as mock_logger:
            merged_content, merged_bibliography = merge_lightrag_with_existing_citations(
                None, None, "fallback content", "fallback bibliography"
            )
            
            # Should fall back to provided values
            assert merged_content == "fallback content"
            assert merged_bibliography == "fallback bibliography"
            mock_logger.error.assert_called_once()
    
    def test_format_lightrag_citation_confidence_basic(self):
        """Test basic confidence formatting."""
        confidence_info = format_lightrag_citation_confidence(0.85)
        
        assert "High" in confidence_info
        assert "0.85" in confidence_info
        assert "Response Confidence" in confidence_info
    
    def test_format_lightrag_citation_confidence_with_breakdown(self):
        """Test confidence formatting with detailed breakdown."""
        confidence_breakdown = {
            "evidence_factors": {
                "entity_confidence": 0.9,
                "relationship_confidence": 0.8,
                "source_diversity": 0.2
            }
        }
        
        confidence_info = format_lightrag_citation_confidence(0.85, confidence_breakdown)
        
        assert "High" in confidence_info
        assert "Entity Confidence" in confidence_info
        assert "Relationship Confidence" in confidence_info
        assert "Source Diversity" in confidence_info
        assert "0.9" in confidence_info
    
    def test_format_lightrag_citation_confidence_levels(self):
        """Test different confidence levels."""
        # High confidence
        high_conf = format_lightrag_citation_confidence(0.9)
        assert "High" in high_conf
        
        # Medium confidence
        medium_conf = format_lightrag_citation_confidence(0.7)
        assert "Medium" in medium_conf
        
        # Low confidence
        low_conf = format_lightrag_citation_confidence(0.5)
        assert "Low" in low_conf
        
        # Very low confidence
        very_low_conf = format_lightrag_citation_confidence(0.2)
        assert "Very Low" in very_low_conf
    
    def test_format_lightrag_citation_confidence_error_handling(self):
        """Test error handling in confidence formatting."""
        with patch('citation.logger') as mock_logger:
            # Pass invalid confidence breakdown to trigger error
            confidence_info = format_lightrag_citation_confidence(0.8, {"invalid": "data"})
            
            # Should still return basic confidence info
            assert "0.8" in confidence_info
    
    def test_validate_lightrag_citations_valid_files(self, tmp_path):
        """Test citation validation with valid files."""
        # Create test PDF files
        pdf1 = tmp_path / "test1.pdf"
        pdf2 = tmp_path / "test2.pdf"
        pdf1.write_text("PDF content 1")
        pdf2.write_text("PDF content 2")
        
        # Create mock citation map
        citation_map = {
            "1": Mock(file_path=str(pdf1)),
            "2": Mock(file_path=str(pdf2))
        }
        
        validation_results = validate_lightrag_citations(citation_map)
        
        assert validation_results["total_citations"] == 2
        assert validation_results["valid_citations"] == 2
        assert validation_results["invalid_citations"] == 0
        assert len(validation_results["accessible_files"]) == 2
        assert len(validation_results["missing_files"]) == 0
    
    def test_validate_lightrag_citations_missing_files(self):
        """Test citation validation with missing files."""
        citation_map = {
            "1": Mock(file_path="/nonexistent/file1.pdf"),
            "2": Mock(file_path="/nonexistent/file2.pdf")
        }
        
        validation_results = validate_lightrag_citations(citation_map)
        
        assert validation_results["total_citations"] == 2
        assert validation_results["valid_citations"] == 0
        assert validation_results["invalid_citations"] == 2
        assert len(validation_results["missing_files"]) == 2
        assert len(validation_results["accessible_files"]) == 0
    
    def test_validate_lightrag_citations_mixed_files(self, tmp_path):
        """Test citation validation with mix of valid and invalid files."""
        # Create one valid file
        valid_pdf = tmp_path / "valid.pdf"
        valid_pdf.write_text("Valid PDF content")
        
        citation_map = {
            "1": Mock(file_path=str(valid_pdf)),
            "2": Mock(file_path="/nonexistent/invalid.pdf"),
            "3": Mock(file_path="")  # Empty path
        }
        
        validation_results = validate_lightrag_citations(citation_map)
        
        assert validation_results["total_citations"] == 3
        assert validation_results["valid_citations"] == 1
        assert validation_results["invalid_citations"] == 2
        assert len(validation_results["accessible_files"]) == 1
        assert len(validation_results["missing_files"]) == 1
    
    def test_validate_lightrag_citations_dict_format(self):
        """Test citation validation with dictionary format citations."""
        citation_map = {
            "1": {"file_path": "/nonexistent/file1.pdf"},
            "2": {"file_path": ""}
        }
        
        validation_results = validate_lightrag_citations(citation_map)
        
        assert validation_results["total_citations"] == 2
        assert validation_results["invalid_citations"] == 2
    
    def test_validate_lightrag_citations_error_handling(self):
        """Test error handling in citation validation."""
        # Pass invalid citation map to trigger error
        invalid_citation_map = {"1": "invalid_citation_object"}
        
        with patch('citation.logger') as mock_logger:
            validation_results = validate_lightrag_citations(invalid_citation_map)
            
            assert validation_results["total_citations"] == 1
            assert validation_results["invalid_citations"] == 1
            mock_logger.error.assert_called()
    
    def test_validate_lightrag_citations_empty_map(self):
        """Test citation validation with empty citation map."""
        validation_results = validate_lightrag_citations({})
        
        assert validation_results["total_citations"] == 0
        assert validation_results["valid_citations"] == 0
        assert validation_results["invalid_citations"] == 0
        assert len(validation_results["accessible_files"]) == 0
        assert len(validation_results["missing_files"]) == 0


class TestCitationIntegrationEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_citation_processing_workflow(self, tmp_path):
        """Test the complete citation processing workflow."""
        # Create test PDF files
        pdf1 = tmp_path / "clinical_metabolomics_2023.pdf"
        pdf2 = tmp_path / "biomarker_study_2022.pdf"
        pdf1.write_text("Clinical metabolomics content")
        pdf2.write_text("Biomarker study content")
        
        # Create LightRAG response
        lightrag_response = {
            "answer": "Clinical metabolomics is important for biomarker discovery.",
            "confidence_score": 0.85,
            "source_documents": [str(pdf1), str(pdf2)],
            "entities_used": [
                {
                    "id": "entity_1",
                    "text": "clinical metabolomics",
                    "type": "field",
                    "relevance_score": 0.9,
                    "source_documents": [str(pdf1)]
                }
            ],
            "relationships_used": [
                {
                    "id": "rel_1",
                    "type": "involves",
                    "confidence": 0.8,
                    "source_documents": [str(pdf1)]
                }
            ]
        }
        
        # Mock the citation formatter to avoid import issues in tests
        with patch('citation.LightRAGCitationFormatter') as mock_formatter_class:
            from lightrag_integration.citation_formatter import LightRAGCitationResult, PDFCitation
            
            mock_result = LightRAGCitationResult(
                formatted_content="Clinical metabolomics is important [1] [2]",
                bibliography="### Sources\n[1] clinical_metabolomics_2023.pdf\n[2] biomarker_study_2022.pdf",
                citation_map={
                    "1": PDFCitation("1", str(pdf1), confidence_score=0.9),
                    "2": PDFCitation("2", str(pdf2), confidence_score=0.8)
                },
                confidence_scores={"1": 0.9, "2": 0.8}
            )
            
            mock_formatter = Mock()
            mock_formatter_class.return_value = mock_formatter
            mock_formatter.format_lightrag_citations.return_value = mock_result
            
            # Process citations
            content, bibliography = process_lightrag_citations(
                lightrag_response["answer"],
                lightrag_response
            )
            
            # Validate results
            assert "[1]" in content
            assert "[2]" in content
            assert "clinical_metabolomics_2023.pdf" in bibliography
            assert "biomarker_study_2022.pdf" in bibliography
            
            # Validate citation map
            validation_results = validate_lightrag_citations(mock_result.citation_map)
            assert validation_results["valid_citations"] == 2
            assert validation_results["invalid_citations"] == 0
            
            # Format confidence information
            confidence_info = format_lightrag_citation_confidence(
                lightrag_response["confidence_score"]
            )
            assert "High" in confidence_info
            assert "0.85" in confidence_info


if __name__ == "__main__":
    pytest.main([__file__])