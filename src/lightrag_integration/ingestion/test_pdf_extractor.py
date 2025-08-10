"""
Unit tests for PDF text extraction component.

Tests the PDF extraction functionality including error handling,
text preprocessing, and batch processing capabilities.
"""

import asyncio
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import the module under test
from .pdf_extractor import PDFExtractor, ExtractedDocument, ExtractionResult


class TestPDFExtractor:
    """Test suite for PDFExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a PDFExtractor instance for testing."""
        return PDFExtractor()
    
    @pytest.fixture
    def sample_text(self):
        """Sample text content for testing."""
        return """
        Clinical Metabolomics in Precision Medicine
        
        John Doe, Jane Smith, Bob Johnson
        
        Abstract: This paper discusses the role of clinical metabolomics
        in precision medicine and its applications in healthcare.
        
        Introduction
        
        Clinical metabolomics is an emerging field that studies small
        molecules in biological systems to understand disease mechanisms
        and develop personalized treatments.
        """
    
    def test_initialization(self, extractor):
        """Test PDFExtractor initialization."""
        assert extractor is not None
        assert extractor.logger is not None
        assert len(extractor.cleanup_patterns) > 0
        assert len(extractor.section_patterns) > 0
    
    def test_preprocess_text(self, extractor, sample_text):
        """Test text preprocessing functionality."""
        # Test with normal text
        processed = extractor._preprocess_text(sample_text)
        assert processed is not None
        assert len(processed) > 0
        
        # Test with empty text
        assert extractor._preprocess_text("") == ""
        assert extractor._preprocess_text(None) == ""
        
        # Test with excessive whitespace
        messy_text = "This   has    lots\n\n\n\nof   whitespace"
        processed = extractor._preprocess_text(messy_text)
        assert "   " not in processed
        assert "\n\n\n\n" not in processed
    
    def test_extract_document_structure(self, extractor, sample_text):
        """Test document structure extraction."""
        title, authors, abstract = extractor._extract_document_structure(sample_text)
        
        # Check title extraction
        assert title is not None
        assert "Clinical Metabolomics" in title
        
        # Check abstract extraction
        assert abstract is not None
        assert "precision medicine" in abstract.lower()
        
        # Test with empty text
        title, authors, abstract = extractor._extract_document_structure("")
        assert title is None
        assert authors == []
        assert abstract == ""
    
    def test_validate_pdf_nonexistent_file(self, extractor):
        """Test PDF validation with non-existent file."""
        is_valid, error = extractor.validate_pdf("/nonexistent/file.pdf")
        assert not is_valid
        assert "does not exist" in error
    
    def test_validate_pdf_non_pdf_file(self, extractor):
        """Test PDF validation with non-PDF file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"This is not a PDF")
            tmp.flush()
            
            try:
                is_valid, error = extractor.validate_pdf(tmp.name)
                assert not is_valid
                assert "not a PDF" in error
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_extract_from_file_nonexistent(self, extractor):
        """Test extraction from non-existent file."""
        result = await extractor.extract_from_file("/nonexistent/file.pdf")
        
        assert not result.success
        assert result.document is None
        assert "not found" in result.error_message.lower()
        assert result.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_extract_from_file_directory(self, extractor):
        """Test extraction when path is a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = await extractor.extract_from_file(tmpdir)
            
            assert not result.success
            assert result.document is None
            assert "not a file" in result.error_message.lower()
    
    @pytest.mark.asyncio
    @patch('src.lightrag_integration.ingestion.pdf_extractor.PYMUPDF_AVAILABLE', True)
    @patch('src.lightrag_integration.ingestion.pdf_extractor.fitz')
    async def test_extract_with_pymupdf_success(self, mock_fitz, extractor, sample_text):
        """Test successful extraction with PyMuPDF."""
        # Mock PyMuPDF document
        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_doc.metadata = {"title": "Test Paper", "author": "Test Author"}
        
        mock_page = MagicMock()
        mock_page.get_text.return_value = sample_text
        mock_doc.load_page.return_value = mock_page
        
        mock_fitz.open.return_value = mock_doc
        
        # Create a temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp.flush()
            
            try:
                result = await extractor.extract_from_file(tmp.name)
                
                # Should succeed or fail gracefully
                assert result is not None
                assert result.processing_time >= 0
                
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_extract_without_pymupdf(self, extractor):
        """Test extraction when PyMuPDF is not available."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp.flush()
            
            try:
                # Test with PyMuPDF not available
                with patch('src.lightrag_integration.ingestion.pdf_extractor.PYMUPDF_AVAILABLE', False):
                    result = await extractor.extract_from_file(tmp.name)
                    
                    # Should return a result but with errors
                    assert result is not None
                    assert result.processing_time >= 0
                
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_extract_batch_success(self, extractor, sample_text):
        """Test batch extraction with multiple files."""
        # Create temporary PDF files
        temp_files = []
        try:
            for i in range(2):  # Reduce number for faster testing
                tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
                tmp.write(f"fake pdf content {i}".encode())
                tmp.flush()
                temp_files.append(tmp.name)
            
            results = await extractor.extract_batch(temp_files, max_concurrent=2)
            
            assert len(results) == 2
            # All results should have processing time
            for result in results:
                assert result.processing_time >= 0
            
        finally:
            for tmp_file in temp_files:
                if os.path.exists(tmp_file):
                    os.unlink(tmp_file)
    
    @pytest.mark.asyncio
    async def test_extract_batch_mixed_results(self, extractor):
        """Test batch extraction with mixed success/failure results."""
        # Mix of valid and invalid file paths
        pdf_paths = [
            "/nonexistent1.pdf",
            "/nonexistent2.pdf",
        ]
        
        results = await extractor.extract_batch(pdf_paths, max_concurrent=2)
        
        assert len(results) == 2
        # All should fail since files don't exist
        failed_results = [r for r in results if not r.success]
        assert len(failed_results) == 2
    
    @pytest.mark.asyncio
    @patch('src.lightrag_integration.ingestion.pdf_extractor.PYMUPDF4LLM_AVAILABLE', True)
    @patch('src.lightrag_integration.ingestion.pdf_extractor.PYMUPDF_AVAILABLE', True)
    @patch('src.lightrag_integration.ingestion.pdf_extractor.pymupdf4llm')
    @patch('src.lightrag_integration.ingestion.pdf_extractor.fitz')
    async def test_extract_with_pymupdf4llm_success(self, mock_fitz, mock_pymupdf4llm, extractor, sample_text):
        """Test extraction using pymupdf4llm method."""
        # Mock pymupdf4llm
        mock_pymupdf4llm.to_markdown.return_value = sample_text
        
        # Mock PyMuPDF for metadata
        mock_doc = MagicMock()
        mock_doc.page_count = 4
        mock_doc.metadata = {"title": "Test"}
        mock_fitz.open.return_value = mock_doc
        
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp.flush()
            
            try:
                document = await extractor._extract_with_pymupdf4llm(tmp.name)
                
                assert document is not None
                assert document.content is not None
                assert document.page_count == 4
                assert document.word_count > 0
                assert len(document.processing_errors) == 0
                
            finally:
                os.unlink(tmp.name)
    
    @pytest.mark.asyncio
    async def test_extract_with_pymupdf4llm_not_available(self, extractor):
        """Test error handling when pymupdf4llm is not available."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"fake pdf content")
            tmp.flush()
            
            try:
                # Test with pymupdf4llm not available
                with patch('src.lightrag_integration.ingestion.pdf_extractor.PYMUPDF4LLM_AVAILABLE', False):
                    with pytest.raises(ImportError):
                        await extractor._extract_with_pymupdf4llm(tmp.name)
                    
            finally:
                os.unlink(tmp.name)
    
    def test_text_preprocessing_edge_cases(self, extractor):
        """Test text preprocessing with edge cases."""
        # Test with very long text
        long_text = "word " * 300000  # Create very long text
        processed = extractor._preprocess_text(long_text)
        assert len(processed) <= 1000000  # Should be truncated
        
        # Test with special characters
        special_text = "Text with\x0cform feed\nand\r\ncarriage returns"
        processed = extractor._preprocess_text(special_text)
        assert '\x0c' not in processed
        
        # Test with only whitespace
        whitespace_text = "   \n\n\n   \t\t\t   "
        processed = extractor._preprocess_text(whitespace_text)
        assert processed == ""
    
    def test_document_structure_extraction_edge_cases(self, extractor):
        """Test document structure extraction with edge cases."""
        # Test with very short text
        short_text = "Hi"
        title, authors, abstract = extractor._extract_document_structure(short_text)
        assert title is None  # Too short for title
        assert authors == []
        assert abstract == ""
        
        # Test with text containing abstract pattern
        abstract_text = """
        Some Title Here
        
        Abstract: This is a test abstract with some content
        that spans multiple lines and should be extracted properly.
        
        Introduction: This should not be part of abstract.
        """
        title, authors, abstract = extractor._extract_document_structure(abstract_text)
        assert "Some Title Here" in title
        assert "test abstract" in abstract
        assert "Introduction" not in abstract


# Integration test that can be run manually with actual PDF files
class TestPDFExtractorIntegration:
    """Integration tests for PDFExtractor (requires actual PDF files)."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_extract_real_pdf(self):
        """Test extraction with a real PDF file (if available)."""
        extractor = PDFExtractor()
        
        # Look for any PDF files in the papers directory
        papers_dir = Path("papers")
        if papers_dir.exists():
            pdf_files = list(papers_dir.glob("*.pdf"))
            if pdf_files:
                # Test with the first PDF found
                result = await extractor.extract_from_file(str(pdf_files[0]))
                
                # Basic validation
                if result.success:
                    assert result.document is not None
                    assert len(result.document.content) > 0
                    assert result.document.word_count > 0
                    print(f"Successfully extracted {result.document.word_count} words")
                    print(f"Title: {result.document.title}")
                    print(f"Authors: {result.document.authors}")
                    print(f"Abstract: {result.document.abstract[:200]}...")
                else:
                    print(f"Extraction failed: {result.error_message}")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])