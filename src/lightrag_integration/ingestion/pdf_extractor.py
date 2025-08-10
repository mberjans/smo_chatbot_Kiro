"""
PDF Text Extraction Component

This module handles PDF text extraction with error handling and preprocessing.
Implements requirements 1.1, 1.5, and 7.1 from the LightRAG integration spec.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import re

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False
    pymupdf4llm = None

from ..utils.logging import setup_logger


@dataclass
class ExtractedDocument:
    """Container for extracted document content and metadata."""
    file_path: str
    title: str
    authors: List[str]
    abstract: str
    content: str
    page_count: int
    word_count: int
    extraction_timestamp: datetime
    metadata: Dict[str, Any]
    processing_errors: List[str]


@dataclass
class ExtractionResult:
    """Result of PDF extraction operation."""
    success: bool
    document: Optional[ExtractedDocument]
    error_message: Optional[str]
    processing_time: float


class PDFExtractor:
    """
    PDF text extraction component with error handling and preprocessing.
    
    This class handles PDF parsing using PyMuPDF, includes comprehensive error
    handling for corrupted files, and provides text preprocessing capabilities.
    """
    
    def __init__(self, config=None):
        """
        Initialize the PDF extractor.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        self.logger = setup_logger("pdf_extractor")
        
        # Text preprocessing patterns
        self.cleanup_patterns = [
            (r'\s+', ' '),  # Multiple whitespace to single space
            (r'\n\s*\n\s*\n+', '\n\n'),  # Multiple newlines to double newline
            (r'[^\x00-\x7F]+', ''),  # Remove non-ASCII characters (optional)
            (r'\x0c', ''),  # Remove form feed characters
        ]
        
        # Common academic paper section patterns
        self.section_patterns = {
            'title': r'(?i)^(.{1,200}?)(?:\n|$)',
            'abstract': r'(?i)abstract\s*:?\s*(.*?)(?=\n\s*(?:keywords|introduction|1\.|background))',
            'authors': r'(?i)(?:authors?|by)\s*:?\s*(.*?)(?=\n\s*(?:abstract|affiliation))',
            'keywords': r'(?i)keywords?\s*:?\s*(.*?)(?=\n\s*(?:introduction|1\.))',
        }
    
    async def extract_from_file(self, pdf_path: str) -> ExtractionResult:
        """
        Extract text content from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            ExtractionResult containing extracted content or error information
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting PDF extraction for: {pdf_path}")
            
            # Validate file exists and is readable
            path_obj = Path(pdf_path)
            if not path_obj.exists():
                return ExtractionResult(
                    success=False,
                    document=None,
                    error_message=f"File not found: {pdf_path}",
                    processing_time=0.0
                )
            
            if not path_obj.is_file():
                return ExtractionResult(
                    success=False,
                    document=None,
                    error_message=f"Path is not a file: {pdf_path}",
                    processing_time=0.0
                )
            
            # Extract text using multiple methods for robustness
            document = await self._extract_with_fallback(pdf_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if document:
                self.logger.info(f"Successfully extracted {document.word_count} words from {pdf_path}")
                return ExtractionResult(
                    success=True,
                    document=document,
                    error_message=None,
                    processing_time=processing_time
                )
            else:
                return ExtractionResult(
                    success=False,
                    document=None,
                    error_message="Failed to extract content with all methods",
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Unexpected error extracting {pdf_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return ExtractionResult(
                success=False,
                document=None,
                error_message=error_msg,
                processing_time=processing_time
            )
    
    async def _extract_with_fallback(self, pdf_path: str) -> Optional[ExtractedDocument]:
        """
        Extract text using multiple methods with fallback.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            ExtractedDocument if successful, None otherwise
        """
        errors = []
        
        # Check if PyMuPDF is available
        if not PYMUPDF_AVAILABLE:
            errors.append("PyMuPDF not available - install with: pip install PyMuPDF")
            self.logger.error("PyMuPDF not available. Please install with: pip install PyMuPDF")
            return await self._create_placeholder_document(pdf_path, errors)
        
        # Method 1: Try pymupdf4llm (optimized for LLM processing)
        if PYMUPDF4LLM_AVAILABLE:
            try:
                document = await self._extract_with_pymupdf4llm(pdf_path)
                if document and document.content.strip():
                    return document
            except Exception as e:
                errors.append(f"pymupdf4llm method failed: {str(e)}")
                self.logger.warning(f"pymupdf4llm extraction failed for {pdf_path}: {str(e)}")
        else:
            errors.append("pymupdf4llm not available")
        
        # Method 2: Try standard PyMuPDF
        try:
            document = await self._extract_with_pymupdf(pdf_path)
            if document and document.content.strip():
                document.processing_errors.extend(errors)
                return document
        except Exception as e:
            errors.append(f"PyMuPDF method failed: {str(e)}")
            self.logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {str(e)}")
        
        # Method 3: Try basic text extraction
        try:
            document = await self._extract_basic_text(pdf_path)
            if document and document.content.strip():
                document.processing_errors.extend(errors)
                return document
        except Exception as e:
            errors.append(f"Basic extraction method failed: {str(e)}")
            self.logger.error(f"All extraction methods failed for {pdf_path}")
        
        return None
    
    async def _extract_with_pymupdf4llm(self, pdf_path: str) -> ExtractedDocument:
        """Extract using pymupdf4llm (optimized for LLM processing)."""
        if not PYMUPDF4LLM_AVAILABLE:
            raise ImportError("pymupdf4llm not available")
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def _extract():
            # Use pymupdf4llm for better text extraction
            md_text = pymupdf4llm.to_markdown(pdf_path)
            return md_text
        
        content = await loop.run_in_executor(None, _extract)
        
        # Also get metadata using standard PyMuPDF
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        page_count = doc.page_count
        doc.close()
        
        # Process the markdown content
        processed_content = self._preprocess_text(content)
        
        # Extract structured information
        title, authors, abstract = self._extract_document_structure(processed_content)
        
        return ExtractedDocument(
            file_path=pdf_path,
            title=title or Path(pdf_path).stem,
            authors=authors,
            abstract=abstract,
            content=processed_content,
            page_count=page_count,
            word_count=len(processed_content.split()),
            extraction_timestamp=datetime.now(),
            metadata=metadata or {},
            processing_errors=[]
        )
    
    async def _extract_with_pymupdf(self, pdf_path: str) -> ExtractedDocument:
        """Extract using standard PyMuPDF."""
        loop = asyncio.get_event_loop()
        
        def _extract():
            doc = fitz.open(pdf_path)
            
            # Extract text from all pages
            text_content = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text = page.get_text()
                if text.strip():
                    text_content.append(text)
            
            content = '\n\n'.join(text_content)
            metadata = doc.metadata
            page_count = doc.page_count
            
            doc.close()
            return content, metadata, page_count
        
        content, metadata, page_count = await loop.run_in_executor(None, _extract)
        
        # Process the content
        processed_content = self._preprocess_text(content)
        
        # Extract structured information
        title, authors, abstract = self._extract_document_structure(processed_content)
        
        return ExtractedDocument(
            file_path=pdf_path,
            title=title or Path(pdf_path).stem,
            authors=authors,
            abstract=abstract,
            content=processed_content,
            page_count=page_count,
            word_count=len(processed_content.split()),
            extraction_timestamp=datetime.now(),
            metadata=metadata or {},
            processing_errors=[]
        )
    
    async def _extract_basic_text(self, pdf_path: str) -> ExtractedDocument:
        """Basic text extraction as last resort."""
        loop = asyncio.get_event_loop()
        
        def _extract():
            doc = fitz.open(pdf_path)
            
            # Simple text extraction
            text_parts = []
            for page_num in range(min(doc.page_count, 10)):  # Limit to first 10 pages
                page = doc.load_page(page_num)
                text = page.get_text("text")  # Plain text extraction
                if text.strip():
                    text_parts.append(text)
            
            content = '\n'.join(text_parts)
            page_count = doc.page_count
            
            doc.close()
            return content, page_count
        
        content, page_count = await loop.run_in_executor(None, _extract)
        
        # Minimal processing
        processed_content = self._preprocess_text(content)
        
        return ExtractedDocument(
            file_path=pdf_path,
            title=Path(pdf_path).stem,
            authors=[],
            abstract="",
            content=processed_content,
            page_count=page_count,
            word_count=len(processed_content.split()),
            extraction_timestamp=datetime.now(),
            metadata={},
            processing_errors=["Used basic extraction method"]
        )
    
    def _preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess extracted text.
        
        Args:
            text: Raw extracted text
        
        Returns:
            Cleaned and preprocessed text
        """
        if not text:
            return ""
        
        # Apply cleanup patterns
        cleaned_text = text
        for pattern, replacement in self.cleanup_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text)
        
        # Remove excessive whitespace
        cleaned_text = cleaned_text.strip()
        
        # Ensure reasonable length
        if len(cleaned_text) > 1000000:  # 1MB text limit
            self.logger.warning("Text content exceeds 1MB, truncating")
            cleaned_text = cleaned_text[:1000000] + "... [TRUNCATED]"
        
        return cleaned_text
    
    def _extract_document_structure(self, text: str) -> Tuple[Optional[str], List[str], str]:
        """
        Extract structured information from document text.
        
        Args:
            text: Processed document text
        
        Returns:
            Tuple of (title, authors, abstract)
        """
        if not text:
            return None, [], ""
        
        # Extract title (first meaningful line)
        title = None
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if len(line) > 10 and len(line) < 200:  # Reasonable title length
                title = line
                break
        
        # Extract abstract
        abstract = ""
        abstract_match = re.search(self.section_patterns['abstract'], text, re.DOTALL | re.IGNORECASE)
        if abstract_match:
            abstract = abstract_match.group(1).strip()[:1000]  # Limit abstract length
        
        # Extract authors (basic pattern matching)
        authors = []
        authors_match = re.search(self.section_patterns['authors'], text, re.DOTALL | re.IGNORECASE)
        if authors_match:
            authors_text = authors_match.group(1).strip()
            # Simple author extraction (split by common separators)
            author_candidates = re.split(r'[,;]|\sand\s', authors_text)
            authors = [author.strip() for author in author_candidates if len(author.strip()) > 2][:10]  # Limit to 10 authors
        
        return title, authors, abstract
    
    async def extract_batch(self, pdf_paths: List[str], max_concurrent: int = 5) -> List[ExtractionResult]:
        """
        Extract text from multiple PDF files concurrently.
        
        Args:
            pdf_paths: List of PDF file paths
            max_concurrent: Maximum number of concurrent extractions
        
        Returns:
            List of ExtractionResult objects
        """
        self.logger.info(f"Starting batch extraction for {len(pdf_paths)} files")
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def _extract_with_semaphore(pdf_path: str) -> ExtractionResult:
            async with semaphore:
                return await self.extract_from_file(pdf_path)
        
        # Process all files concurrently
        tasks = [_extract_with_semaphore(path) for path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ExtractionResult(
                    success=False,
                    document=None,
                    error_message=f"Exception during extraction: {str(result)}",
                    processing_time=0.0
                ))
            else:
                final_results.append(result)
        
        successful = sum(1 for r in final_results if r.success)
        self.logger.info(f"Batch extraction completed: {successful}/{len(pdf_paths)} successful")
        
        return final_results
    
    async def _create_placeholder_document(self, pdf_path: str, errors: List[str]) -> ExtractedDocument:
        """Create a placeholder document when PDF extraction fails."""
        return ExtractedDocument(
            file_path=pdf_path,
            title=f"[EXTRACTION FAILED] {Path(pdf_path).stem}",
            authors=[],
            abstract="PDF extraction failed - PyMuPDF not available",
            content="PDF extraction failed due to missing dependencies. Please install PyMuPDF.",
            page_count=0,
            word_count=0,
            extraction_timestamp=datetime.now(),
            metadata={},
            processing_errors=errors
        )

    def validate_pdf(self, pdf_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a PDF file is readable.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            path_obj = Path(pdf_path)
            
            if not path_obj.exists():
                return False, "File does not exist"
            
            if not path_obj.is_file():
                return False, "Path is not a file"
            
            if path_obj.suffix.lower() != '.pdf':
                return False, "File is not a PDF"
            
            if not PYMUPDF_AVAILABLE:
                return False, "PyMuPDF not available - cannot validate PDF"
            
            # Try to open with PyMuPDF
            doc = fitz.open(pdf_path)
            page_count = doc.page_count
            doc.close()
            
            if page_count == 0:
                return False, "PDF has no pages"
            
            return True, None
            
        except Exception as e:
            return False, f"PDF validation error: {str(e)}"