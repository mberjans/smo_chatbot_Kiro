"""
LightRAG Citation Formatter

This module extends the existing citation system to handle PDF document citations
from LightRAG responses. It provides citation linking back to source documents
and bibliography generation for LightRAG sources.

Implements requirements 4.2 and 4.5.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

try:
    from lightrag_integration.utils.logging import setup_logger
    from lightrag_integration.confidence_scoring import LightRAGConfidenceScorer, ConfidenceBreakdown
    CONFIDENCE_SCORING_AVAILABLE = True
except ImportError:
    # Fallback to basic logging if utils not available
    import logging
    def setup_logger(name, log_file=None):
        return logging.getLogger(name)
    CONFIDENCE_SCORING_AVAILABLE = False


@dataclass
class PDFCitation:
    """Citation information for a PDF document."""
    citation_id: str
    file_path: str
    title: str
    authors: List[str]
    year: Optional[str]
    confidence_score: float
    page_numbers: List[int]
    sections: List[str]
    doi: Optional[str] = None
    journal: Optional[str] = None
    abstract: Optional[str] = None


@dataclass
class LightRAGCitationResult:
    """Result of citation processing for LightRAG responses."""
    formatted_content: str
    bibliography: str
    citation_map: Dict[str, PDFCitation]
    confidence_scores: Dict[str, float]
    source_count: int
    confidence_breakdown: Optional[Any] = None  # ConfidenceBreakdown object
    confidence_display_info: Optional[Dict[str, Any]] = None


class CitationFormatter:
    """Alias for LightRAGCitationFormatter for backward compatibility."""
    
    def __init__(self, config=None):
        self._formatter = LightRAGCitationFormatter(config)
    
    def __getattr__(self, name):
        return getattr(self._formatter, name)

class LightRAGCitationFormatter:
    """
    Citation formatter for LightRAG PDF document sources.
    
    This class extends the existing citation system to handle PDF documents
    from the LightRAG knowledge graph, providing proper citation formatting
    and bibliography generation.
    """
    
    def __init__(self, config=None):
        """Initialize the LightRAG citation formatter."""
        self.config = config
        self.logger = setup_logger("lightrag_citation_formatter")
        
        # Citation formatting settings
        self.citation_style = "apa"  # Default to APA style
        self.max_authors_display = 3
        self.max_title_length = 100
        
        # Confidence thresholds for citation display
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        self.low_confidence_threshold = 0.4
        
        # Initialize confidence scorer if available
        if CONFIDENCE_SCORING_AVAILABLE:
            self.confidence_scorer = LightRAGConfidenceScorer(config)
        else:
            self.confidence_scorer = None
            self.logger.warning("Confidence scoring not available")
        
        self.logger.info("LightRAG Citation Formatter initialized")
    
    def format_lightrag_citations(
        self,
        content: str,
        source_documents: List[str],
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        confidence_score: float
    ) -> LightRAGCitationResult:
        """
        Format citations for LightRAG response content.
        
        Args:
            content: The response content to add citations to
            source_documents: List of source document file paths
            entities_used: List of entities used in the response
            relationships_used: List of relationships used in the response
            confidence_score: Overall response confidence score
        
        Returns:
            LightRAGCitationResult with formatted content and bibliography
        """
        try:
            self.logger.info(f"Formatting citations for {len(source_documents)} source documents")
            
            # Extract PDF metadata for citations
            pdf_citations = self._extract_pdf_citations(
                source_documents, entities_used, relationships_used
            )
            
            if not pdf_citations:
                self.logger.warning("No valid PDF citations found")
                return LightRAGCitationResult(
                    formatted_content=content,
                    bibliography="",
                    citation_map={},
                    confidence_scores={},
                    source_count=0,
                    confidence_breakdown=None,
                    confidence_display_info=None
                )
            
            # Insert citation markers into content
            formatted_content = self._insert_pdf_citation_markers(content, pdf_citations)
            
            # Calculate enhanced confidence breakdown if scorer is available
            confidence_breakdown = None
            confidence_display_info = None
            
            if self.confidence_scorer:
                try:
                    confidence_breakdown = self.confidence_scorer.calculate_response_confidence(
                        confidence_score, source_documents, entities_used, relationships_used
                    )
                    confidence_display_info = self.confidence_scorer.get_confidence_display_info(
                        confidence_breakdown
                    )
                    # Use enhanced confidence for bibliography generation
                    final_confidence = confidence_breakdown.overall_confidence
                except Exception as e:
                    self.logger.error(f"Error calculating enhanced confidence: {str(e)}")
                    final_confidence = confidence_score
            else:
                final_confidence = confidence_score
            
            # Generate bibliography with enhanced confidence
            bibliography = self._generate_pdf_bibliography(pdf_citations, final_confidence)
            
            # Create citation map and confidence scores
            citation_map = {citation.citation_id: citation for citation in pdf_citations}
            confidence_scores = {
                citation.citation_id: citation.confidence_score 
                for citation in pdf_citations
            }
            
            result = LightRAGCitationResult(
                formatted_content=formatted_content,
                bibliography=bibliography,
                citation_map=citation_map,
                confidence_scores=confidence_scores,
                source_count=len(pdf_citations),
                confidence_breakdown=confidence_breakdown,
                confidence_display_info=confidence_display_info
            )
            
            self.logger.info(f"Successfully formatted {len(pdf_citations)} PDF citations")
            return result
            
        except Exception as e:
            self.logger.error(f"Error formatting LightRAG citations: {str(e)}", exc_info=True)
            
            # Return original content on error
            return LightRAGCitationResult(
                formatted_content=content,
                bibliography=f"Error generating bibliography: {str(e)}",
                citation_map={},
                confidence_scores={},
                source_count=0,
                confidence_breakdown=None,
                confidence_display_info=None
            )
    
    def _extract_pdf_citations(
        self,
        source_documents: List[str],
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> List[PDFCitation]:
        """Extract PDF citation information from source documents."""
        citations = []
        citation_counter = 1
        
        # Process each unique source document
        unique_documents = list(set(source_documents))
        
        for doc_path in unique_documents:
            if not doc_path or not doc_path.strip():
                continue
            
            try:
                # Extract metadata from PDF file path and content
                pdf_metadata = self._extract_pdf_metadata(doc_path)
                
                # Calculate confidence score based on entity/relationship references
                confidence = self._calculate_document_confidence(
                    doc_path, entities_used, relationships_used
                )
                
                # Extract page numbers and sections from entities/relationships
                page_numbers, sections = self._extract_location_info(
                    doc_path, entities_used, relationships_used
                )
                
                citation = PDFCitation(
                    citation_id=str(citation_counter),
                    file_path=doc_path,
                    title=pdf_metadata.get("title", self._extract_title_from_filename(doc_path)),
                    authors=pdf_metadata.get("authors", []),
                    year=pdf_metadata.get("year"),
                    confidence_score=confidence,
                    page_numbers=page_numbers,
                    sections=sections,
                    doi=pdf_metadata.get("doi"),
                    journal=pdf_metadata.get("journal"),
                    abstract=pdf_metadata.get("abstract")
                )
                
                citations.append(citation)
                citation_counter += 1
                
            except Exception as e:
                self.logger.warning(f"Error processing document {doc_path}: {str(e)}")
                continue
        
        # Sort citations by confidence score (highest first)
        citations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return citations
    
    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF file.
        
        This is a placeholder implementation. In a full implementation,
        we would use libraries like PyMuPDF or pdfplumber to extract
        actual PDF metadata.
        """
        metadata = {}
        
        try:
            # Check if file exists
            pdf_path = Path(file_path)
            if not pdf_path.exists():
                # Try relative to papers directory if configured
                if self.config and hasattr(self.config, 'papers_directory'):
                    pdf_path = Path(self.config.papers_directory) / Path(file_path).name
            
            if pdf_path.exists():
                # Extract basic information from filename
                filename = pdf_path.stem
                
                # Try to extract year from filename
                year_match = re.search(r'(19|20)\d{2}', filename)
                if year_match:
                    metadata["year"] = year_match.group()
                
                # Try to extract title (remove common patterns)
                title = filename
                title = re.sub(r'_+', ' ', title)  # Replace underscores with spaces
                title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
                title = title.strip()
                
                # Remove year and common suffixes
                title = re.sub(r'\b(19|20)\d{2}\b', '', title)
                title = re.sub(r'\b(pdf|PDF)\b', '', title)
                title = title.strip()
                
                if title:
                    metadata["title"] = title
                
                # TODO: In a full implementation, extract actual PDF metadata
                # using libraries like PyMuPDF:
                # import fitz
                # doc = fitz.open(str(pdf_path))
                # metadata.update(doc.metadata)
                # doc.close()
                
            else:
                self.logger.warning(f"PDF file not found: {file_path}")
                
        except Exception as e:
            self.logger.error(f"Error extracting PDF metadata from {file_path}: {str(e)}")
        
        return metadata
    
    def _extract_title_from_filename(self, file_path: str) -> str:
        """Extract a readable title from the filename."""
        filename = Path(file_path).stem
        
        # Clean up filename to make it more readable
        title = filename.replace('_', ' ').replace('-', ' ')
        title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
        title = title.strip()
        
        # Capitalize first letter of each word
        title = ' '.join(word.capitalize() for word in title.split())
        
        # Truncate if too long
        if len(title) > self.max_title_length:
            title = title[:self.max_title_length] + "..."
        
        return title or "Unknown Document"
    
    def _calculate_document_confidence(
        self,
        doc_path: str,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for a document based on entity/relationship references."""
        try:
            # Count references to this document
            entity_refs = 0
            entity_confidence_sum = 0.0
            
            for entity in entities_used:
                entity_sources = entity.get("source_documents", [])
                if doc_path in entity_sources:
                    entity_refs += 1
                    entity_confidence_sum += entity.get("relevance_score", 0.5)
            
            rel_refs = 0
            rel_confidence_sum = 0.0
            
            for rel in relationships_used:
                rel_sources = rel.get("source_documents", [])
                if doc_path in rel_sources:
                    rel_refs += 1
                    rel_confidence_sum += rel.get("confidence", 0.5)
            
            total_refs = entity_refs + rel_refs
            
            if total_refs == 0:
                return 0.3  # Low confidence for unreferenced documents
            
            # Calculate average confidence from references
            avg_entity_confidence = entity_confidence_sum / max(entity_refs, 1)
            avg_rel_confidence = rel_confidence_sum / max(rel_refs, 1)
            
            # Weighted average with boost for multiple references
            base_confidence = (avg_entity_confidence + avg_rel_confidence) / 2
            
            # Boost confidence based on number of references
            reference_boost = min((total_refs - 1) * 0.05, 0.2)
            
            final_confidence = min(base_confidence + reference_boost, 1.0)
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating document confidence: {str(e)}")
            return 0.5  # Default confidence on error
    
    def _extract_location_info(
        self,
        doc_path: str,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> Tuple[List[int], List[str]]:
        """Extract page numbers and sections from entities and relationships."""
        page_numbers = set()
        sections = set()
        
        try:
            # Extract from entities
            for entity in entities_used:
                if doc_path in entity.get("source_documents", []):
                    # Look for page information in entity properties
                    properties = entity.get("properties", {})
                    
                    # Extract page numbers
                    if "page" in properties:
                        try:
                            page_numbers.add(int(properties["page"]))
                        except (ValueError, TypeError):
                            pass
                    
                    # Extract sections
                    if "section" in properties:
                        sections.add(str(properties["section"]))
            
            # Extract from relationships
            for rel in relationships_used:
                if doc_path in rel.get("source_documents", []):
                    # Look for page information in relationship evidence
                    evidence = rel.get("evidence", [])
                    for evidence_text in evidence:
                        # Try to extract page numbers from evidence text
                        page_matches = re.findall(r'page\s+(\d+)', evidence_text.lower())
                        for page_match in page_matches:
                            try:
                                page_numbers.add(int(page_match))
                            except ValueError:
                                pass
            
        except Exception as e:
            self.logger.error(f"Error extracting location info: {str(e)}")
        
        return sorted(list(page_numbers)), sorted(list(sections))
    
    def _insert_pdf_citation_markers(self, content: str, citations: List[PDFCitation]) -> str:
        """Insert citation markers into the content."""
        if not citations:
            return content
        
        # For now, use a simple approach: add citations at the end of sentences
        # In a more sophisticated implementation, we would use NLP to match
        # content segments to specific sources
        
        # Create citation markers
        high_confidence_citations = [
            c for c in citations 
            if c.confidence_score >= self.high_confidence_threshold
        ]
        
        medium_confidence_citations = [
            c for c in citations 
            if self.medium_confidence_threshold <= c.confidence_score < self.high_confidence_threshold
        ]
        
        # Insert high confidence citations first
        if high_confidence_citations:
            citation_markers = " ".join([f"[{c.citation_id}]" for c in high_confidence_citations])
            
            # Insert after first sentence or at the end
            sentences = content.split('. ')
            if len(sentences) > 1:
                sentences[0] += f" {citation_markers}"
                content = '. '.join(sentences)
            else:
                content += f" {citation_markers}"
        
        # Insert medium confidence citations at the end
        if medium_confidence_citations:
            citation_markers = " ".join([f"[{c.citation_id}]" for c in medium_confidence_citations])
            content += f" {citation_markers}"
        
        return content
    
    def _generate_pdf_bibliography(self, citations: List[PDFCitation], overall_confidence: float) -> str:
        """Generate bibliography for PDF citations."""
        if not citations:
            return ""
        
        bibliography_parts = []
        
        # Add overall confidence indicator with enhanced information
        confidence_level = self._get_confidence_level(overall_confidence)
        bibliography_parts.append(f"\n\n### Sources")
        
        # Add confidence display with icon if available
        if hasattr(self, 'confidence_scorer') and self.confidence_scorer:
            # Get display info for enhanced confidence
            try:
                # Create a minimal confidence breakdown for display
                from lightrag_integration.confidence_scoring import ConfidenceBreakdown, ConfidenceFactors
                temp_breakdown = ConfidenceBreakdown(
                    overall_confidence=overall_confidence,
                    base_confidence=overall_confidence,
                    enhancement_factor=0.0,
                    confidence_factors=ConfidenceFactors(),
                    source_scores={},
                    entity_scores={},
                    relationship_scores={},
                    explanation=""
                )
                display_info = self.confidence_scorer.get_confidence_display_info(temp_breakdown)
                icon = display_info.get("icon", "")
                color_indicator = f" {icon}" if icon else ""
                bibliography_parts.append(f"**Response Confidence: {confidence_level} ({overall_confidence:.2f}){color_indicator}**\n")
            except Exception:
                bibliography_parts.append(f"**Response Confidence: {confidence_level} ({overall_confidence:.2f})**\n")
        else:
            bibliography_parts.append(f"**Response Confidence: {confidence_level} ({overall_confidence:.2f})**\n")
        
        # Group citations by confidence level
        high_confidence = [c for c in citations if c.confidence_score >= self.high_confidence_threshold]
        medium_confidence = [c for c in citations if self.medium_confidence_threshold <= c.confidence_score < self.high_confidence_threshold]
        low_confidence = [c for c in citations if c.confidence_score < self.medium_confidence_threshold]
        
        # Primary sources (high confidence)
        if high_confidence:
            bibliography_parts.append("**Primary Sources:**")
            for citation in high_confidence:
                formatted_citation = self._format_single_citation(citation)
                bibliography_parts.append(f"[{citation.citation_id}] {formatted_citation}")
        
        # Supporting sources (medium confidence)
        if medium_confidence:
            bibliography_parts.append("\n**Supporting Sources:**")
            for citation in medium_confidence:
                formatted_citation = self._format_single_citation(citation)
                bibliography_parts.append(f"[{citation.citation_id}] {formatted_citation}")
        
        # Additional references (low confidence)
        if low_confidence:
            bibliography_parts.append("\n**Additional References:**")
            for citation in low_confidence:
                formatted_citation = self._format_single_citation(citation)
                bibliography_parts.append(f"[{citation.citation_id}] {formatted_citation}")
        
        return "\n".join(bibliography_parts)
    
    def _format_single_citation(self, citation: PDFCitation) -> str:
        """Format a single PDF citation in APA style."""
        try:
            citation_parts = []
            
            # Authors
            if citation.authors:
                if len(citation.authors) <= self.max_authors_display:
                    authors_str = ", ".join(citation.authors)
                else:
                    authors_str = f"{', '.join(citation.authors[:self.max_authors_display])}, et al."
                citation_parts.append(authors_str)
            
            # Year
            if citation.year:
                citation_parts.append(f"({citation.year})")
            
            # Title
            title = citation.title
            if len(title) > self.max_title_length:
                title = title[:self.max_title_length] + "..."
            citation_parts.append(f"*{title}*")
            
            # Journal (if available)
            if citation.journal:
                citation_parts.append(f"*{citation.journal}*")
            
            # DOI (if available)
            if citation.doi:
                citation_parts.append(f"DOI: {citation.doi}")
            
            # Page numbers (if available)
            if citation.page_numbers:
                if len(citation.page_numbers) == 1:
                    citation_parts.append(f"p. {citation.page_numbers[0]}")
                else:
                    citation_parts.append(f"pp. {citation.page_numbers[0]}-{citation.page_numbers[-1]}")
            
            # File path (for local access)
            filename = Path(citation.file_path).name
            citation_parts.append(f"({filename})")
            
            # Confidence indicator
            confidence_level = self._get_confidence_level(citation.confidence_score)
            citation_parts.append(f"[Confidence: {confidence_level}]")
            
            return ". ".join(citation_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting citation: {str(e)}")
            return f"{citation.title} ({Path(citation.file_path).name})"
    
    def _get_confidence_level(self, confidence_score: float) -> str:
        """Get confidence level description."""
        if confidence_score >= self.high_confidence_threshold:
            return "High"
        elif confidence_score >= self.medium_confidence_threshold:
            return "Medium"
        elif confidence_score >= self.low_confidence_threshold:
            return "Low"
        else:
            return "Very Low"
    
    def merge_with_existing_citations(
        self,
        lightrag_result: LightRAGCitationResult,
        existing_content: str,
        existing_bibliography: str
    ) -> Tuple[str, str]:
        """
        Merge LightRAG citations with existing system citations.
        
        This method allows for hybrid responses that combine LightRAG
        citations with citations from other sources (e.g., Perplexity API).
        """
        try:
            # Combine content
            combined_content = f"{existing_content}\n\n{lightrag_result.formatted_content}"
            
            # Combine bibliographies
            combined_bibliography = existing_bibliography
            if lightrag_result.bibliography:
                if existing_bibliography:
                    combined_bibliography += f"\n\n{lightrag_result.bibliography}"
                else:
                    combined_bibliography = lightrag_result.bibliography
            
            return combined_content, combined_bibliography
            
        except Exception as e:
            self.logger.error(f"Error merging citations: {str(e)}")
            return lightrag_result.formatted_content, lightrag_result.bibliography
    
    def get_citation_statistics(self, citations: List[PDFCitation]) -> Dict[str, Any]:
        """Get statistics about the citations."""
        if not citations:
            return {}
        
        confidence_scores = [c.confidence_score for c in citations]
        
        return {
            "total_citations": len(citations),
            "high_confidence_count": len([c for c in citations if c.confidence_score >= self.high_confidence_threshold]),
            "medium_confidence_count": len([c for c in citations if self.medium_confidence_threshold <= c.confidence_score < self.high_confidence_threshold]),
            "low_confidence_count": len([c for c in citations if c.confidence_score < self.medium_confidence_threshold]),
            "average_confidence": sum(confidence_scores) / len(confidence_scores),
            "max_confidence": max(confidence_scores),
            "min_confidence": min(confidence_scores),
            "citations_with_pages": len([c for c in citations if c.page_numbers]),
            "citations_with_authors": len([c for c in citations if c.authors]),
            "citations_with_doi": len([c for c in citations if c.doi])
        }