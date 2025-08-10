"""
Unit tests for LightRAG Citation Formatter

Tests the citation formatting functionality for PDF documents
from LightRAG responses.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from .citation_formatter import (
    LightRAGCitationFormatter,
    PDFCitation,
    LightRAGCitationResult
)
from .config.settings import LightRAGConfig


class TestLightRAGCitationFormatter:
    """Test cases for LightRAG citation formatter."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            knowledge_graph_path="test_kg",
            vector_store_path="test_vectors",
            cache_directory="test_cache",
            papers_directory="test_papers"
        )
    
    @pytest.fixture
    def formatter(self, config):
        """Create citation formatter instance."""
        return LightRAGCitationFormatter(config)
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_pdf_files(self, temp_dir):
        """Create sample PDF files for testing."""
        pdf_files = []
        
        # Create test PDF files
        for i, filename in enumerate([
            "clinical_metabolomics_2023.pdf",
            "biomarker_discovery_2022.pdf",
            "pathway_analysis_2021.pdf"
        ]):
            pdf_path = Path(temp_dir) / filename
            pdf_path.write_text(f"Mock PDF content {i}")
            pdf_files.append(str(pdf_path))
        
        return pdf_files
    
    @pytest.fixture
    def sample_entities(self):
        """Create sample entities for testing."""
        return [
            {
                "id": "entity_1",
                "text": "clinical metabolomics",
                "type": "field",
                "relevance_score": 0.9,
                "source_documents": ["clinical_metabolomics_2023.pdf"]
            },
            {
                "id": "entity_2", 
                "text": "biomarker",
                "type": "concept",
                "relevance_score": 0.8,
                "source_documents": ["biomarker_discovery_2022.pdf", "clinical_metabolomics_2023.pdf"]
            }
        ]
    
    @pytest.fixture
    def sample_relationships(self):
        """Create sample relationships for testing."""
        return [
            {
                "id": "rel_1",
                "type": "involves",
                "source": "entity_1",
                "target": "entity_2",
                "confidence": 0.85,
                "evidence": ["Clinical metabolomics involves biomarker discovery"],
                "source_documents": ["clinical_metabolomics_2023.pdf"]
            }
        ]
    
    def test_initialization(self, config):
        """Test formatter initialization."""
        formatter = LightRAGCitationFormatter(config)
        
        assert formatter.config == config
        assert formatter.citation_style == "apa"
        assert formatter.max_authors_display == 3
        assert formatter.pdf_extensions == {'.pdf', '.PDF'}
    
    def test_is_pdf_document(self, formatter):
        """Test PDF document detection."""
        assert formatter._is_pdf_document("test.pdf") == True
        assert formatter._is_pdf_document("test.PDF") == True
        assert formatter._is_pdf_document("test.txt") == False
        assert formatter._is_pdf_document("test.docx") == False
        assert formatter._is_pdf_document("") == False
    
    def test_extract_pdf_metadata(self, formatter, sample_pdf_files):
        """Test PDF metadata extraction."""
        pdf_path = sample_pdf_files[0]
        metadata = formatter._extract_pdf_metadata(pdf_path)
        
        assert "filename" in metadata
        assert "title" in metadata
        assert "file_size" in metadata
        assert metadata["filename"] == "clinical_metabolomics_2023.pdf"
        assert "clinical metabolomics 2023" in metadata["title"].lower()
        assert metadata["year"] == "2023"
    
    def test_calculate_citation_confidence(self, formatter, sample_entities, sample_relationships):
        """Test citation confidence calculation."""
        doc_path = "clinical_metabolomics_2023.pdf"
        
        confidence = formatter._calculate_citation_confidence(
            doc_path, sample_entities, sample_relationships
        )
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be high due to multiple references
    
    def test_generate_pdf_citations(self, formatter, sample_pdf_files, sample_entities, sample_relationships):
        """Test PDF citation generation."""
        citations = formatter._generate_pdf_citations(
            sample_pdf_files, sample_entities, sample_relationships
        )
        
        assert len(citations) == len(sample_pdf_files)
        assert all(isinstance(citation, PDFCitation) for citation in citations)
        assert all(citation.citation_id for citation in citations)
        assert all(citation.file_path for citation in citations)
        
        # Check that citations are sorted by confidence
        confidences = [c.confidence_score for c in citations]
        assert confidences == sorted(confidences, reverse=True)
    
    def test_insert_citation_markers(self, formatter):
        """Test citation marker insertion."""
        content = "Clinical metabolomics is important. It helps in disease diagnosis."
        citations = [
            PDFCitation("1", "test1.pdf", confidence_score=0.9),
            PDFCitation("2", "test2.pdf", confidence_score=0.8)
        ]
        
        formatted_content = formatter._insert_citation_markers(content, citations)
        
        assert "[1]" in formatted_content
        assert "[2]" in formatted_content
        assert len(formatted_content) > len(content)
    
    def test_format_single_citation(self, formatter):
        """Test single citation formatting."""
        citation = PDFCitation(
            citation_id="1",
            file_path="/path/to/test.pdf",
            title="Test Document",
            authors=["Author One", "Author Two"],
            year="2023",
            confidence_score=0.85
        )
        
        formatted = formatter._format_single_citation(citation)
        
        assert "Test Document" in formatted
        assert "Author One" in formatted
        assert "2023" in formatted
        assert "test.pdf" in formatted
        assert "High confidence" in formatted or "Medium confidence" in formatted
    
    def test_generate_pdf_bibliography(self, formatter):
        """Test PDF bibliography generation."""
        citations = [
            PDFCitation("1", "high_conf.pdf", title="High Confidence Doc", confidence_score=0.9),
            PDFCitation("2", "medium_conf.pdf", title="Medium Confidence Doc", confidence_score=0.7),
            PDFCitation("3", "low_conf.pdf", title="Low Confidence Doc", confidence_score=0.4)
        ]
        
        bibliography = formatter._generate_pdf_bibliography(citations, 0.8)
        
        assert "Primary Sources" in bibliography
        assert "Supporting Sources" in bibliography or "Additional References" in bibliography
        assert "[1]" in bibliography
        assert "[2]" in bibliography
        assert "[3]" in bibliography
    
    def test_format_lightrag_citations_success(self, formatter, sample_pdf_files, sample_entities, sample_relationships):
        """Test successful citation formatting."""
        content = "Clinical metabolomics is a field that studies biomarkers."
        
        result = formatter.format_lightrag_citations(
            content=content,
            source_documents=sample_pdf_files,
            entities_used=sample_entities,
            relationships_used=sample_relationships,
            confidence_score=0.8
        )
        
        assert isinstance(result, LightRAGCitationResult)
        assert result.formatted_content
        assert result.bibliography
        assert result.citation_map
        assert result.confidence_scores
        assert len(result.citation_map) == len(sample_pdf_files)
    
    def test_format_lightrag_citations_empty_sources(self, formatter):
        """Test citation formatting with empty sources."""
        content = "Test content"
        
        result = formatter.format_lightrag_citations(
            content=content,
            source_documents=[],
            entities_used=[],
            relationships_used=[],
            confidence_score=0.5
        )
        
        assert result.formatted_content == content
        assert result.bibliography == ""
        assert len(result.citation_map) == 0
        assert len(result.confidence_scores) == 0
    
    def test_format_lightrag_citations_non_pdf_sources(self, formatter):
        """Test citation formatting with non-PDF sources."""
        content = "Test content"
        non_pdf_sources = ["document.txt", "presentation.pptx", "data.csv"]
        
        result = formatter.format_lightrag_citations(
            content=content,
            source_documents=non_pdf_sources,
            entities_used=[],
            relationships_used=[],
            confidence_score=0.5
        )
        
        # Should filter out non-PDF documents
        assert len(result.citation_map) == 0
    
    def test_merge_with_existing_citations(self, formatter):
        """Test merging with existing citations."""
        lightrag_result = LightRAGCitationResult(
            formatted_content="LightRAG content [1]",
            bibliography="### Sources\n[1] LightRAG source",
            citation_map={"1": PDFCitation("1", "test.pdf")},
            confidence_scores={"1": 0.8}
        )
        
        existing_content = "Existing content [2]"
        existing_bibliography = "### References\n[2] Existing source"
        
        merged_content, merged_bibliography = formatter.merge_with_existing_citations(
            lightrag_result, existing_content, existing_bibliography
        )
        
        assert merged_content == lightrag_result.formatted_content
        assert "LightRAG source" in merged_bibliography
        assert "Existing source" in merged_bibliography
    
    def test_extract_citation_metadata(self, formatter):
        """Test citation metadata extraction."""
        citation_map = {
            "1": PDFCitation("1", "test1.pdf", confidence_score=0.9, authors=["Author A"]),
            "2": PDFCitation("2", "test2.pdf", confidence_score=0.7, authors=["Author B", "Author C"]),
            "3": PDFCitation("3", "test3.pdf", confidence_score=0.4, year="2023")
        }
        
        metadata = formatter.extract_citation_metadata(citation_map)
        
        assert metadata["total_citations"] == 3
        assert metadata["high_confidence_count"] == 1
        assert metadata["medium_confidence_count"] == 1
        assert metadata["low_confidence_count"] == 1
        assert len(metadata["unique_authors"]) == 3
        assert "2023" in metadata["publication_years"]
    
    def test_get_confidence_level(self, formatter):
        """Test confidence level categorization."""
        assert formatter._get_confidence_level(0.9) == "High"
        assert formatter._get_confidence_level(0.7) == "Medium"
        assert formatter._get_confidence_level(0.5) == "Low"
        assert formatter._get_confidence_level(0.2) == "Very Low"
    
    def test_create_file_link(self, formatter, sample_pdf_files):
        """Test file link creation."""
        pdf_path = sample_pdf_files[0]
        link = formatter._create_file_link(pdf_path)
        
        assert link is not None
        assert "file://" in link or "./" in link
    
    def test_create_file_link_nonexistent(self, formatter):
        """Test file link creation for nonexistent file."""
        link = formatter._create_file_link("/nonexistent/file.pdf")
        
        assert link is not None
        assert "file.pdf" in link


class TestPDFCitation:
    """Test cases for PDFCitation dataclass."""
    
    def test_pdf_citation_creation(self):
        """Test PDFCitation creation."""
        citation = PDFCitation(
            citation_id="1",
            file_path="/path/to/test.pdf",
            title="Test Document",
            authors=["Author One"],
            year="2023",
            confidence_score=0.8
        )
        
        assert citation.citation_id == "1"
        assert citation.file_path == "/path/to/test.pdf"
        assert citation.title == "Test Document"
        assert citation.authors == ["Author One"]
        assert citation.year == "2023"
        assert citation.confidence_score == 0.8
    
    def test_pdf_citation_defaults(self):
        """Test PDFCitation with default values."""
        citation = PDFCitation("1", "/path/to/test.pdf")
        
        assert citation.citation_id == "1"
        assert citation.file_path == "/path/to/test.pdf"
        assert citation.title is None
        assert citation.authors is None
        assert citation.year is None
        assert citation.confidence_score == 0.5
        assert citation.metadata is None


class TestLightRAGCitationResult:
    """Test cases for LightRAGCitationResult dataclass."""
    
    def test_citation_result_creation(self):
        """Test LightRAGCitationResult creation."""
        citation_map = {"1": PDFCitation("1", "test.pdf")}
        confidence_scores = {"1": 0.8}
        
        result = LightRAGCitationResult(
            formatted_content="Test content [1]",
            bibliography="### Sources\n[1] Test source",
            citation_map=citation_map,
            confidence_scores=confidence_scores
        )
        
        assert result.formatted_content == "Test content [1]"
        assert result.bibliography == "### Sources\n[1] Test source"
        assert result.citation_map == citation_map
        assert result.confidence_scores == confidence_scores


# Integration tests
class TestCitationIntegration:
    """Integration tests for citation system."""
    
    @pytest.fixture
    def mock_lightrag_response(self):
        """Create mock LightRAG response."""
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
            ]
        }
    
    def test_end_to_end_citation_processing(self, mock_lightrag_response, temp_dir):
        """Test end-to-end citation processing."""
        # Create test PDF files
        for filename in mock_lightrag_response["source_documents"]:
            pdf_path = Path(temp_dir) / filename
            pdf_path.write_text("Mock PDF content")
        
        # Update source documents to use full paths
        mock_lightrag_response["source_documents"] = [
            str(Path(temp_dir) / filename) 
            for filename in mock_lightrag_response["source_documents"]
        ]
        
        # Update entity and relationship source documents
        for entity in mock_lightrag_response["entities_used"]:
            entity["source_documents"] = [
                str(Path(temp_dir) / doc) for doc in entity["source_documents"]
            ]
        
        for rel in mock_lightrag_response["relationships_used"]:
            rel["source_documents"] = [
                str(Path(temp_dir) / doc) for doc in rel["source_documents"]
            ]
        
        # Process citations
        formatter = LightRAGCitationFormatter()
        result = formatter.format_lightrag_citations(
            content=mock_lightrag_response["answer"],
            source_documents=mock_lightrag_response["source_documents"],
            entities_used=mock_lightrag_response["entities_used"],
            relationships_used=mock_lightrag_response["relationships_used"],
            confidence_score=mock_lightrag_response["confidence_score"]
        )
        
        # Verify results
        assert result.formatted_content
        assert result.bibliography
        assert len(result.citation_map) == 2
        assert all(score > 0 for score in result.confidence_scores.values())
        
        # Verify citation markers are present
        assert "[1]" in result.formatted_content or "[2]" in result.formatted_content
        
        # Verify bibliography contains source information
        assert "clinical_metabolomics_2023.pdf" in result.bibliography
        assert "biomarker_study_2022.pdf" in result.bibliography


if __name__ == "__main__":
    pytest.main([__file__])