"""
Unit tests for LightRAG Citation Formatter

Tests the citation formatting functionality for PDF document sources
from LightRAG responses.
"""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lightrag_integration.citation_formatter import (
    LightRAGCitationFormatter, 
    PDFCitation, 
    LightRAGCitationResult
)


class TestLightRAGCitationFormatter(unittest.TestCase):
    """Test cases for LightRAG Citation Formatter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = LightRAGCitationFormatter()
        
        # Sample test data
        self.sample_content = "Clinical metabolomics is a field that studies metabolites in biological systems."
        
        self.sample_source_documents = [
            "papers/clinical_metabolomics_review.pdf",
            "papers/biomarker_discovery_2023.pdf",
            "papers/metabolomics_methods.pdf"
        ]
        
        self.sample_entities_used = [
            {
                "id": "entity_1",
                "text": "clinical metabolomics",
                "type": "field",
                "relevance_score": 0.9,
                "source_documents": ["papers/clinical_metabolomics_review.pdf"],
                "properties": {"page": 1, "section": "Introduction"}
            },
            {
                "id": "entity_2", 
                "text": "metabolites",
                "type": "compound",
                "relevance_score": 0.8,
                "source_documents": ["papers/clinical_metabolomics_review.pdf", "papers/biomarker_discovery_2023.pdf"],
                "properties": {"page": 3}
            }
        ]
        
        self.sample_relationships_used = [
            {
                "id": "rel_1",
                "type": "studies",
                "source": "entity_1",
                "target": "entity_2",
                "confidence": 0.85,
                "evidence": ["Clinical metabolomics studies metabolites in biological systems"],
                "source_documents": ["papers/clinical_metabolomics_review.pdf"]
            }
        ]
    
    def test_initialization(self):
        """Test formatter initialization."""
        formatter = LightRAGCitationFormatter()
        
        self.assertEqual(formatter.citation_style, "apa")
        self.assertEqual(formatter.max_authors_display, 3)
        self.assertEqual(formatter.high_confidence_threshold, 0.8)
        self.assertEqual(formatter.medium_confidence_threshold, 0.6)
        self.assertEqual(formatter.low_confidence_threshold, 0.4)
    
    def test_format_lightrag_citations_basic(self):
        """Test basic citation formatting."""
        result = self.formatter.format_lightrag_citations(
            self.sample_content,
            self.sample_source_documents,
            self.sample_entities_used,
            self.sample_relationships_used,
            0.85
        )
        
        self.assertIsInstance(result, LightRAGCitationResult)
        # Check that some citation marker is present (could be [1], [2], etc.)
        self.assertTrue(any(f"[{i}]" in result.formatted_content for i in range(1, 10)))
        self.assertIn("### Sources", result.bibliography)
        self.assertGreater(result.source_count, 0)
        self.assertGreater(len(result.citation_map), 0)
    
    def test_format_lightrag_citations_empty_sources(self):
        """Test citation formatting with empty source documents."""
        result = self.formatter.format_lightrag_citations(
            self.sample_content,
            [],
            [],
            [],
            0.5
        )
        
        self.assertEqual(result.formatted_content, self.sample_content)
        self.assertEqual(result.bibliography, "")
        self.assertEqual(result.source_count, 0)
        self.assertEqual(len(result.citation_map), 0)
    
    def test_extract_pdf_citations(self):
        """Test PDF citation extraction."""
        citations = self.formatter._extract_pdf_citations(
            self.sample_source_documents,
            self.sample_entities_used,
            self.sample_relationships_used
        )
        
        self.assertIsInstance(citations, list)
        self.assertGreater(len(citations), 0)
        
        for citation in citations:
            self.assertIsInstance(citation, PDFCitation)
            self.assertIsInstance(citation.citation_id, str)
            self.assertIsInstance(citation.file_path, str)
            self.assertIsInstance(citation.title, str)
            self.assertIsInstance(citation.confidence_score, float)
            self.assertGreaterEqual(citation.confidence_score, 0.0)
            self.assertLessEqual(citation.confidence_score, 1.0)
    
    def test_extract_title_from_filename(self):
        """Test title extraction from filename."""
        test_cases = [
            ("clinical_metabolomics_review.pdf", "Clinical Metabolomics Review"),
            ("biomarker-discovery-2023.pdf", "Biomarker Discovery 2023"),
            ("metabolomics_methods_and_applications.pdf", "Metabolomics Methods And Applications"),
        ]
        
        for filename, expected_title in test_cases:
            with self.subTest(filename=filename):
                result = self.formatter._extract_title_from_filename(filename)
                self.assertEqual(result, expected_title)
        
        # Test truncation separately
        long_filename = "very_long_filename_that_should_be_truncated_because_it_exceeds_the_maximum_length_limit.pdf"
        result = self.formatter._extract_title_from_filename(long_filename)
        # Check if result is longer than max length or ends with ...
        self.assertTrue(len(result) <= self.formatter.max_title_length or result.endswith("..."))
    
    def test_calculate_document_confidence(self):
        """Test document confidence calculation."""
        doc_path = "papers/clinical_metabolomics_review.pdf"
        
        confidence = self.formatter._calculate_document_confidence(
            doc_path,
            self.sample_entities_used,
            self.sample_relationships_used
        )
        
        self.assertIsInstance(confidence, float)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Document with references should have higher confidence than default
        self.assertGreater(confidence, 0.3)
    
    def test_calculate_document_confidence_no_references(self):
        """Test document confidence calculation with no references."""
        doc_path = "papers/unreferenced_document.pdf"
        
        confidence = self.formatter._calculate_document_confidence(
            doc_path,
            [],
            []
        )
        
        self.assertEqual(confidence, 0.3)  # Default low confidence
    
    def test_extract_location_info(self):
        """Test extraction of page numbers and sections."""
        doc_path = "papers/clinical_metabolomics_review.pdf"
        
        page_numbers, sections = self.formatter._extract_location_info(
            doc_path,
            self.sample_entities_used,
            self.sample_relationships_used
        )
        
        self.assertIsInstance(page_numbers, list)
        self.assertIsInstance(sections, list)
        
        # Should extract page numbers from entity properties
        self.assertIn(1, page_numbers)
        self.assertIn(3, page_numbers)
        
        # Should extract sections from entity properties
        self.assertIn("Introduction", sections)
    
    def test_format_single_citation(self):
        """Test formatting of a single citation."""
        citation = PDFCitation(
            citation_id="1",
            file_path="papers/clinical_metabolomics_review.pdf",
            title="Clinical Metabolomics Review",
            authors=["Smith, J.", "Johnson, A."],
            year="2023",
            confidence_score=0.85,
            page_numbers=[1, 2, 3],
            sections=["Introduction"],
            doi="10.1234/example",
            journal="Journal of Metabolomics"
        )
        
        formatted = self.formatter._format_single_citation(citation)
        
        self.assertIn("Smith, J., Johnson, A.", formatted)
        self.assertIn("(2023)", formatted)
        self.assertIn("Clinical Metabolomics Review", formatted)
        self.assertIn("Journal of Metabolomics", formatted)
        self.assertIn("DOI: 10.1234/example", formatted)
        self.assertIn("pp. 1-3", formatted)
        self.assertIn("clinical_metabolomics_review.pdf", formatted)
        self.assertIn("[Confidence: High]", formatted)
    
    def test_format_single_citation_minimal(self):
        """Test formatting of a citation with minimal information."""
        citation = PDFCitation(
            citation_id="1",
            file_path="papers/unknown_paper.pdf",
            title="Unknown Paper",
            authors=[],
            year=None,
            confidence_score=0.4,
            page_numbers=[],
            sections=[],
            doi=None,
            journal=None
        )
        
        formatted = self.formatter._format_single_citation(citation)
        
        self.assertIn("Unknown Paper", formatted)
        self.assertIn("unknown_paper.pdf", formatted)
        self.assertIn("[Confidence: Low]", formatted)
    
    def test_get_confidence_level(self):
        """Test confidence level descriptions."""
        test_cases = [
            (0.9, "High"),
            (0.8, "High"),
            (0.7, "Medium"),
            (0.6, "Medium"),
            (0.5, "Low"),
            (0.4, "Low"),
            (0.3, "Very Low"),
            (0.1, "Very Low")
        ]
        
        for confidence, expected_level in test_cases:
            with self.subTest(confidence=confidence):
                result = self.formatter._get_confidence_level(confidence)
                self.assertEqual(result, expected_level)
    
    def test_generate_pdf_bibliography(self):
        """Test PDF bibliography generation."""
        citations = [
            PDFCitation(
                citation_id="1",
                file_path="papers/high_confidence.pdf",
                title="High Confidence Paper",
                authors=["Author, A."],
                year="2023",
                confidence_score=0.9,
                page_numbers=[1],
                sections=[]
            ),
            PDFCitation(
                citation_id="2",
                file_path="papers/medium_confidence.pdf",
                title="Medium Confidence Paper",
                authors=["Author, B."],
                year="2022",
                confidence_score=0.7,
                page_numbers=[],
                sections=[]
            ),
            PDFCitation(
                citation_id="3",
                file_path="papers/low_confidence.pdf",
                title="Low Confidence Paper",
                authors=[],
                year=None,
                confidence_score=0.3,
                page_numbers=[],
                sections=[]
            )
        ]
        
        bibliography = self.formatter._generate_pdf_bibliography(citations, 0.8)
        
        self.assertIn("### Sources", bibliography)
        self.assertIn("Response Confidence: High", bibliography)
        self.assertIn("Primary Sources:", bibliography)
        self.assertIn("Supporting Sources:", bibliography)
        self.assertIn("Additional References:", bibliography)
        self.assertIn("[1]", bibliography)
        self.assertIn("[2]", bibliography)
        self.assertIn("[3]", bibliography)
    
    def test_insert_pdf_citation_markers(self):
        """Test insertion of citation markers into content."""
        citations = [
            PDFCitation(
                citation_id="1",
                file_path="papers/test.pdf",
                title="Test Paper",
                authors=[],
                year=None,
                confidence_score=0.9,
                page_numbers=[],
                sections=[]
            ),
            PDFCitation(
                citation_id="2",
                file_path="papers/test2.pdf",
                title="Test Paper 2",
                authors=[],
                year=None,
                confidence_score=0.7,
                page_numbers=[],
                sections=[]
            )
        ]
        
        content = "This is a test sentence. This is another sentence."
        result = self.formatter._insert_pdf_citation_markers(content, citations)
        
        self.assertIn("[1]", result)
        self.assertIn("[2]", result)
    
    def test_get_citation_statistics(self):
        """Test citation statistics calculation."""
        citations = [
            PDFCitation("1", "test1.pdf", "Title 1", ["Author 1"], "2023", 0.9, [1], [], "10.1234/test1"),
            PDFCitation("2", "test2.pdf", "Title 2", [], None, 0.7, [], [], None),
            PDFCitation("3", "test3.pdf", "Title 3", ["Author 2", "Author 3"], "2022", 0.3, [1, 2], [], None)
        ]
        
        stats = self.formatter.get_citation_statistics(citations)
        
        self.assertEqual(stats["total_citations"], 3)
        self.assertEqual(stats["high_confidence_count"], 1)
        self.assertEqual(stats["medium_confidence_count"], 1)
        self.assertEqual(stats["low_confidence_count"], 1)
        self.assertAlmostEqual(stats["average_confidence"], 0.63, places=1)
        self.assertEqual(stats["max_confidence"], 0.9)
        self.assertEqual(stats["min_confidence"], 0.3)
        self.assertEqual(stats["citations_with_pages"], 2)
        self.assertEqual(stats["citations_with_authors"], 2)
        self.assertEqual(stats["citations_with_doi"], 1)
    
    def test_merge_with_existing_citations(self):
        """Test merging with existing citation system."""
        lightrag_result = LightRAGCitationResult(
            formatted_content="LightRAG content [1]",
            bibliography="### Sources\n[1] LightRAG Source",
            citation_map={"1": Mock()},
            confidence_scores={"1": 0.8},
            source_count=1
        )
        
        existing_content = "Existing content [1]"
        existing_bibliography = "### References\n[1] Existing Source"
        
        merged_content, merged_bibliography = self.formatter.merge_with_existing_citations(
            lightrag_result, existing_content, existing_bibliography
        )
        
        self.assertIn("Existing content", merged_content)
        self.assertIn("LightRAG content", merged_content)
        self.assertIn("Existing Source", merged_bibliography)
        self.assertIn("LightRAG Source", merged_bibliography)
    
    @patch('pathlib.Path.exists')
    def test_extract_pdf_metadata_file_not_found(self, mock_exists):
        """Test PDF metadata extraction when file doesn't exist."""
        mock_exists.return_value = False
        
        metadata = self.formatter._extract_pdf_metadata("nonexistent.pdf")
        
        self.assertIsInstance(metadata, dict)
        # Should return empty metadata for non-existent files
    
    def test_error_handling_in_format_lightrag_citations(self):
        """Test error handling in main formatting function."""
        # Test with invalid data that should trigger error handling
        with patch.object(self.formatter, '_extract_pdf_citations', side_effect=Exception("Test error")):
            result = self.formatter.format_lightrag_citations(
                self.sample_content,
                self.sample_source_documents,
                self.sample_entities_used,
                self.sample_relationships_used,
                0.8
            )
            
            # Should return original content on error
            self.assertEqual(result.formatted_content, self.sample_content)
            self.assertIn("Error generating bibliography", result.bibliography)
            self.assertEqual(result.source_count, 0)


class TestCitationIntegration(unittest.TestCase):
    """Test integration with existing citation system."""
    
    def test_citation_formatter_import(self):
        """Test that citation formatter can be imported."""
        try:
            from lightrag_integration.citation_formatter import LightRAGCitationFormatter
            formatter = LightRAGCitationFormatter()
            self.assertIsNotNone(formatter)
        except ImportError:
            self.fail("Could not import LightRAGCitationFormatter")


if __name__ == '__main__':
    unittest.main()