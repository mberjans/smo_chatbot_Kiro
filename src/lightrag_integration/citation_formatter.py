"""
LightRAG Citation Formatter

This module extends the existing citation system to handle PDF document citations
from LightRAG responses. It provides citation linking back to source documents
and bibliography generation for LightRAG sources.

Implements requirements 4.2 and 4.5.
"""

import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from urllib.parse import quote

try:
    from .utils.logging import setup_logger
    from .confidence_scoring import LightRAGConfidenceScorer
except ImportError:
    # Fallback for testing
    import logging
    def setup_logger(name, log_file=None):
        return logging.getLogger(name)
    
    # Mock confidence scorer for testing
    class LightRAGConfidenceScorer:
        def calculate_enhanced_confidence(self, *args, **kwargs):
            return 0.5, {"error": "Confidence scorer not available"}
        
        def format_confidence_for_ui(self, *args, **kwargs):
            return {"overall": {"score": 0.5, "level": "Medium", "color": "orange"}}


@dataclass
class PDFCitation:
    """Citation information for a PDF document."""
    citation_id: str
    file_path: str
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    year: Optional[str] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    confidence_score: float = 0.5
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class LightRAGCitationResult:
    """Result of citation processing for LightRAG responses."""
    formatted_content: str
    bibliography: str
    citation_map: Dict[str, PDFCitation]
    confidence_scores: Dict[str, float]


class LightRAGCitationFormatter:
    """
    Citation formatter for LightRAG PDF document sources.
    
    This class extends the existing citation system to handle PDF documents
    from the LightRAG knowledge graph, providing proper citation formatting
    and bibliography generation.
    """
    
    def __init__(self, config=None):
        """Initialize the citation formatter."""
        self.config = config
        self.logger = setup_logger("lightrag_citation_formatter")
        
        # Citation formatting settings
        self.citation_style = "apa"  # Default to APA style
        self.max_authors_display = 3
        self.truncate_title_length = 100
        
        # File path patterns for different document types
        self.pdf_extensions = {'.pdf', '.PDF'}
        
        # Initialize confidence scorer
        self.confidence_scorer = LightRAGConfidenceScorer(config)
        
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
            source_documents: List of source document paths
            entities_used: List of entities used in the response
            relationships_used: List of relationships used in the response
            confidence_score: Overall response confidence score
        
        Returns:
            LightRAGCitationResult with formatted content and bibliography
        """
        try:
            self.logger.info(f"Formatting citations for {len(source_documents)} source documents")
            
            # Generate PDF citations from source documents
            pdf_citations = self._generate_pdf_citations(
                source_documents, entities_used, relationships_used
            )
            
            if not pdf_citations:
                self.logger.warning("No valid PDF citations generated")
                return LightRAGCitationResult(
                    formatted_content=content,
                    bibliography="",
                    citation_map={},
                    confidence_scores={}
                )
            
            # Insert citation markers into content
            formatted_content = self._insert_citation_markers(content, pdf_citations)
            
            # Generate bibliography
            bibliography = self._generate_pdf_bibliography(pdf_citations, confidence_score)
            
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
                confidence_scores=confidence_scores
            )
            
            self.logger.info(f"Successfully formatted {len(pdf_citations)} citations")
            return result
            
        except Exception as e:
            self.logger.error(f"Error formatting LightRAG citations: {str(e)}", exc_info=True)
            
            # Return original content on error
            return LightRAGCitationResult(
                formatted_content=content,
                bibliography=f"Error generating bibliography: {str(e)}",
                citation_map={},
                confidence_scores={}
            )
    
    def _generate_pdf_citations(
        self,
        source_documents: List[str],
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> List[PDFCitation]:
        """Generate PDF citations from source documents."""
        citations = []
        citation_counter = 1
        
        # Process unique source documents
        unique_documents = list(set(doc for doc in source_documents if doc and doc.strip()))
        
        for doc_path in unique_documents:
            try:
                # Validate that this is a PDF document
                if not self._is_pdf_document(doc_path):
                    self.logger.warning(f"Skipping non-PDF document: {doc_path}")
                    continue
                
                # Extract document metadata
                metadata = self._extract_pdf_metadata(doc_path)
                
                # Calculate citation confidence based on usage
                confidence = self._calculate_citation_confidence(
                    doc_path, entities_used, relationships_used
                )
                
                # Create PDF citation
                citation = PDFCitation(
                    citation_id=str(citation_counter),
                    file_path=doc_path,
                    title=metadata.get("title"),
                    authors=metadata.get("authors"),
                    year=metadata.get("year"),
                    confidence_score=confidence,
                    metadata=metadata
                )
                
                citations.append(citation)
                citation_counter += 1
                
            except Exception as e:
                self.logger.error(f"Error processing document {doc_path}: {str(e)}")
                continue
        
        # Sort citations by confidence score (highest first)
        citations.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return citations
    
    def _is_pdf_document(self, doc_path: str) -> bool:
        """Check if the document is a PDF file."""
        try:
            path = Path(doc_path)
            return path.suffix in self.pdf_extensions
        except Exception:
            return False
    
    def _extract_pdf_metadata(self, doc_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF document."""
        metadata = {}
        
        try:
            path = Path(doc_path)
            
            # Extract basic information from file path
            metadata["filename"] = path.name
            metadata["file_size"] = path.stat().st_size if path.exists() else 0
            metadata["last_modified"] = datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat() if path.exists() else None
            
            # Try to extract title from filename (remove extension and clean up)
            title_from_filename = path.stem
            title_from_filename = re.sub(r'[_-]', ' ', title_from_filename)
            title_from_filename = re.sub(r'\s+', ' ', title_from_filename).strip()
            
            # Use filename as title if no better title is available
            metadata["title"] = title_from_filename
            
            # Try to extract year from filename
            year_match = re.search(r'\b(19|20)\d{2}\b', title_from_filename)
            if year_match:
                metadata["year"] = year_match.group()
            
            # TODO: In a more sophisticated implementation, we would use
            # libraries like PyMuPDF or pdfplumber to extract actual PDF metadata
            # For now, we use filename-based extraction
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {doc_path}: {str(e)}")
            metadata = {"filename": Path(doc_path).name, "title": Path(doc_path).stem}
        
        return metadata
    
    def _calculate_citation_confidence(
        self,
        doc_path: str,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score for a citation based on usage."""
        try:
            # Count references to this document
            entity_refs = 0
            entity_confidence_sum = 0.0
            
            for entity in entities_used:
                entity_sources = entity.get("source_documents", [])
                if doc_path in entity_sources:
                    entity_refs += 1
                    entity_confidence_sum += entity.get("relevance_score", 0.5)
            
            relationship_refs = 0
            relationship_confidence_sum = 0.0
            
            for rel in relationships_used:
                rel_sources = rel.get("source_documents", [])
                if doc_path in rel_sources:
                    relationship_refs += 1
                    relationship_confidence_sum += rel.get("confidence", 0.5)
            
            total_refs = entity_refs + relationship_refs
            
            if total_refs == 0:
                return 0.3  # Low confidence for unreferenced documents
            
            # Calculate average confidence from references
            avg_entity_confidence = (
                entity_confidence_sum / entity_refs if entity_refs > 0 else 0.5
            )
            avg_rel_confidence = (
                relationship_confidence_sum / relationship_refs if relationship_refs > 0 else 0.5
            )
            
            # Weighted average with boost for multiple references
            base_confidence = (avg_entity_confidence + avg_rel_confidence) / 2
            
            # Boost confidence based on number of references
            reference_boost = min(total_refs * 0.05, 0.2)
            
            # Final confidence score
            confidence = min(base_confidence + reference_boost, 1.0)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating citation confidence: {str(e)}")
            return 0.5  # Default confidence on error
    
    def _insert_citation_markers(self, content: str, citations: List[PDFCitation]) -> str:
        """Insert citation markers into the content."""
        if not citations:
            return content
        
        # For now, use a simple approach: add citations at the end of sentences
        # In a more sophisticated implementation, we would use NLP to match
        # content segments to specific sources
        
        # Create citation markers
        citation_markers = []
        for citation in citations:
            citation_markers.append(f"[{citation.citation_id}]")
        
        # Insert citations strategically
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        if len(sentences) > 1:
            # Add citations to the first substantial sentence
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) > 20:  # Substantial sentence
                    sentences[i] = sentence + " " + " ".join(citation_markers[:2])
                    break
            
            # Add remaining citations to the last sentence if there are many
            if len(citation_markers) > 2:
                sentences[-1] = sentences[-1] + " " + " ".join(citation_markers[2:])
        else:
            # Single sentence or short content
            content = content + " " + " ".join(citation_markers)
            return content
        
        return " ".join(sentences)
    
    def _generate_pdf_bibliography(
        self, 
        citations: List[PDFCitation], 
        overall_confidence: float
    ) -> str:
        """Generate bibliography for PDF citations."""
        if not citations:
            return ""
        
        bibliography_parts = []
        
        # Add confidence indicator
        confidence_level = self._get_confidence_level(overall_confidence)
        bibliography_parts.append(
            f"\n\n### Sources (Confidence: {confidence_level} - {overall_confidence:.2f})\n"
        )
        
        # Group citations by confidence level
        high_conf_citations = [c for c in citations if c.confidence_score >= 0.8]
        medium_conf_citations = [c for c in citations if 0.6 <= c.confidence_score < 0.8]
        low_conf_citations = [c for c in citations if c.confidence_score < 0.6]
        
        # Primary sources (high confidence)
        if high_conf_citations:
            bibliography_parts.append("**Primary Sources:**")
            for citation in high_conf_citations:
                formatted_citation = self._format_single_citation(citation)
                bibliography_parts.append(f"[{citation.citation_id}] {formatted_citation}")
        
        # Supporting sources (medium confidence)
        if medium_conf_citations:
            if high_conf_citations:
                bibliography_parts.append("\n**Supporting Sources:**")
            else:
                bibliography_parts.append("**Sources:**")
            
            for citation in medium_conf_citations:
                formatted_citation = self._format_single_citation(citation)
                bibliography_parts.append(f"[{citation.citation_id}] {formatted_citation}")
        
        # Additional references (low confidence)
        if low_conf_citations:
            bibliography_parts.append("\n**Additional References:**")
            for citation in low_conf_citations:
                formatted_citation = self._format_single_citation(citation)
                bibliography_parts.append(f"[{citation.citation_id}] {formatted_citation}")
        
        return "\n".join(bibliography_parts)
    
    def _format_single_citation(self, citation: PDFCitation) -> str:
        """Format a single PDF citation."""
        try:
            parts = []
            
            # Add title
            if citation.title:
                title = citation.title
                if len(title) > self.truncate_title_length:
                    title = title[:self.truncate_title_length] + "..."
                parts.append(f"*{title}*")
            
            # Add authors
            if citation.authors:
                if len(citation.authors) <= self.max_authors_display:
                    authors_str = ", ".join(citation.authors)
                else:
                    authors_str = ", ".join(citation.authors[:self.max_authors_display]) + " et al."
                parts.append(authors_str)
            
            # Add year
            if citation.year:
                parts.append(f"({citation.year})")
            
            # Add file information
            filename = Path(citation.file_path).name
            parts.append(f"PDF: {filename}")
            
            # Add confidence indicator
            confidence_level = self._get_confidence_level(citation.confidence_score)
            parts.append(f"[{confidence_level} confidence: {citation.confidence_score:.2f}]")
            
            # Create file link if possible
            file_link = self._create_file_link(citation.file_path)
            if file_link:
                parts.append(f"[Open File]({file_link})")
            
            return " - ".join(parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting citation: {str(e)}")
            return f"PDF: {Path(citation.file_path).name}"
    
    def _create_file_link(self, file_path: str) -> Optional[str]:
        """Create a link to the PDF file."""
        try:
            path = Path(file_path)
            
            if path.exists():
                # Create a file:// URL for local files
                # Note: This may not work in all environments/browsers
                return f"file://{path.absolute()}"
            else:
                # If file doesn't exist, try to create a relative link
                return f"./{path.name}"
                
        except Exception as e:
            self.logger.error(f"Error creating file link: {str(e)}")
            return None
    
    def _get_confidence_level(self, confidence_score: float) -> str:
        """Get confidence level description."""
        if confidence_score >= 0.8:
            return "High"
        elif confidence_score >= 0.6:
            return "Medium"
        elif confidence_score >= 0.4:
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
        
        Args:
            lightrag_result: LightRAG citation result
            existing_content: Content with existing citations
            existing_bibliography: Existing bibliography
        
        Returns:
            Tuple of (merged_content, merged_bibliography)
        """
        try:
            # Combine content
            merged_content = lightrag_result.formatted_content
            
            # Combine bibliographies
            merged_bibliography = ""
            
            if lightrag_result.bibliography:
                merged_bibliography += lightrag_result.bibliography
            
            if existing_bibliography:
                if merged_bibliography:
                    merged_bibliography += "\n\n"
                merged_bibliography += existing_bibliography
            
            return merged_content, merged_bibliography
            
        except Exception as e:
            self.logger.error(f"Error merging citations: {str(e)}")
            return lightrag_result.formatted_content, lightrag_result.bibliography
    
    def extract_citation_metadata(self, citation_map: Dict[str, PDFCitation]) -> Dict[str, Any]:
        """Extract metadata from citation map for analysis."""
        try:
            metadata = {
                "total_citations": len(citation_map),
                "high_confidence_count": 0,
                "medium_confidence_count": 0,
                "low_confidence_count": 0,
                "average_confidence": 0.0,
                "unique_authors": set(),
                "publication_years": [],
                "file_types": {},
                "total_file_size": 0
            }
            
            confidence_sum = 0.0
            
            for citation in citation_map.values():
                # Count confidence levels
                if citation.confidence_score >= 0.8:
                    metadata["high_confidence_count"] += 1
                elif citation.confidence_score >= 0.6:
                    metadata["medium_confidence_count"] += 1
                else:
                    metadata["low_confidence_count"] += 1
                
                confidence_sum += citation.confidence_score
                
                # Collect authors
                if citation.authors:
                    metadata["unique_authors"].update(citation.authors)
                
                # Collect years
                if citation.year:
                    metadata["publication_years"].append(citation.year)
                
                # File type analysis
                file_ext = Path(citation.file_path).suffix.lower()
                metadata["file_types"][file_ext] = metadata["file_types"].get(file_ext, 0) + 1
                
                # File size
                if citation.metadata and "file_size" in citation.metadata:
                    metadata["total_file_size"] += citation.metadata["file_size"]
            
            # Calculate averages
            if len(citation_map) > 0:
                metadata["average_confidence"] = confidence_sum / len(citation_map)
            
            # Convert sets to lists for JSON serialization
            metadata["unique_authors"] = list(metadata["unique_authors"])
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting citation metadata: {str(e)}")
            return {"error": str(e)}
    
    def format_lightrag_citations_with_confidence(
        self,
        content: str,
        source_documents: List[str],
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        base_confidence_score: float
    ) -> Tuple[LightRAGCitationResult, Dict[str, Any]]:
        """
        Format citations with enhanced confidence scoring.
        
        Args:
            content: The response content to add citations to
            source_documents: List of source document paths
            entities_used: List of entities used in the response
            relationships_used: List of relationships used in the response
            base_confidence_score: Base confidence score from query processing
        
        Returns:
            Tuple of (LightRAGCitationResult, confidence_ui_format)
        """
        try:
            self.logger.info("Formatting citations with enhanced confidence scoring")
            
            # First, format citations normally
            citation_result = self.format_lightrag_citations(
                content, source_documents, entities_used, relationships_used, base_confidence_score
            )
            
            # Calculate enhanced confidence
            enhanced_confidence, confidence_breakdown = self.confidence_scorer.calculate_enhanced_confidence(
                base_confidence=base_confidence_score,
                entities_used=entities_used,
                relationships_used=relationships_used,
                source_documents=source_documents,
                citation_map=citation_result.citation_map
            )
            
            # Get citation confidence scores from the breakdown
            citation_confidences = []
            if "citation_confidence" in confidence_breakdown:
                # Extract citation confidences from the scorer
                try:
                    # Re-run citation confidence calculation to get individual scores
                    from .confidence_scoring import SourceReliabilityScore, GraphEvidenceMetrics
                    
                    # Calculate reliability scores
                    reliability_scores = self.confidence_scorer._calculate_source_reliability_scores(
                        source_documents, entities_used, relationships_used
                    )
                    
                    # Calculate graph metrics
                    graph_metrics = self.confidence_scorer._calculate_graph_evidence_metrics(
                        entities_used, relationships_used, source_documents
                    )
                    
                    # Calculate citation confidences
                    citation_confidences = self.confidence_scorer._calculate_citation_confidence_scores(
                        citation_result.citation_map, reliability_scores, graph_metrics
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error calculating individual citation confidences: {str(e)}")
                    citation_confidences = []
            
            # Format confidence for UI
            confidence_ui_format = self.confidence_scorer.format_confidence_for_ui(
                enhanced_confidence, citation_confidences, confidence_breakdown
            )
            
            # Update citation result with enhanced confidence
            enhanced_citation_result = LightRAGCitationResult(
                formatted_content=citation_result.formatted_content,
                bibliography=self._generate_enhanced_bibliography(
                    citation_result.citation_map, enhanced_confidence, confidence_breakdown
                ),
                citation_map=citation_result.citation_map,
                confidence_scores={
                    **citation_result.confidence_scores,
                    "enhanced_overall": enhanced_confidence
                }
            )
            
            self.logger.info(f"Enhanced confidence: {enhanced_confidence:.3f} (base: {base_confidence_score:.3f})")
            return enhanced_citation_result, confidence_ui_format
            
        except Exception as e:
            self.logger.error(f"Error in enhanced citation formatting: {str(e)}", exc_info=True)
            
            # Fallback to regular citation formatting
            citation_result = self.format_lightrag_citations(
                content, source_documents, entities_used, relationships_used, base_confidence_score
            )
            
            confidence_ui_format = {
                "overall": {
                    "score": base_confidence_score,
                    "level": self._get_confidence_level(base_confidence_score),
                    "color": self._get_confidence_color(base_confidence_score),
                    "display_text": f"Confidence: {base_confidence_score:.2f}"
                },
                "error": str(e)
            }
            
            return citation_result, confidence_ui_format
    
    def _generate_enhanced_bibliography(
        self,
        citation_map: Dict[str, PDFCitation],
        enhanced_confidence: float,
        confidence_breakdown: Dict[str, Any]
    ) -> str:
        """Generate enhanced bibliography with confidence information."""
        try:
            bibliography_parts = []
            
            # Add enhanced confidence header
            confidence_level = self._get_confidence_level(enhanced_confidence)
            bibliography_parts.append(
                f"\n\n### Sources (Enhanced Confidence: {confidence_level} - {enhanced_confidence:.2f})\n"
            )
            
            # Add confidence improvement information
            improvement = confidence_breakdown.get("improvement", 0.0)
            if improvement > 0.05:
                bibliography_parts.append(
                    f"*Confidence improved by {improvement:.2f} through graph-based analysis*\n"
                )
            
            # Group citations by confidence level
            citations = list(citation_map.values())
            high_conf_citations = [c for c in citations if c.confidence_score >= 0.8]
            medium_conf_citations = [c for c in citations if 0.6 <= c.confidence_score < 0.8]
            low_conf_citations = [c for c in citations if c.confidence_score < 0.6]
            
            # Primary sources (high confidence)
            if high_conf_citations:
                bibliography_parts.append("**Primary Sources (High Reliability):**")
                for citation in high_conf_citations:
                    formatted_citation = self._format_single_citation_enhanced(citation, confidence_breakdown)
                    bibliography_parts.append(f"[{citation.citation_id}] {formatted_citation}")
            
            # Supporting sources (medium confidence)
            if medium_conf_citations:
                if high_conf_citations:
                    bibliography_parts.append("\n**Supporting Sources (Medium Reliability):**")
                else:
                    bibliography_parts.append("**Sources (Medium Reliability):**")
                
                for citation in medium_conf_citations:
                    formatted_citation = self._format_single_citation_enhanced(citation, confidence_breakdown)
                    bibliography_parts.append(f"[{citation.citation_id}] {formatted_citation}")
            
            # Additional references (low confidence)
            if low_conf_citations:
                bibliography_parts.append("\n**Additional References (Lower Reliability):**")
                for citation in low_conf_citations:
                    formatted_citation = self._format_single_citation_enhanced(citation, confidence_breakdown)
                    bibliography_parts.append(f"[{citation.citation_id}] {formatted_citation}")
            
            # Add confidence breakdown summary
            if confidence_breakdown:
                bibliography_parts.append(self._format_confidence_summary(confidence_breakdown))
            
            return "\n".join(bibliography_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced bibliography: {str(e)}")
            # Fallback to regular bibliography
            return self._generate_pdf_bibliography(list(citation_map.values()), enhanced_confidence)
    
    def _format_single_citation_enhanced(
        self,
        citation: PDFCitation,
        confidence_breakdown: Dict[str, Any]
    ) -> str:
        """Format a single citation with enhanced confidence information."""
        try:
            # Start with basic citation
            basic_citation = self._format_single_citation(citation)
            
            # Add enhanced confidence indicators
            source_reliability = confidence_breakdown.get("source_reliability", {})
            avg_reliability = source_reliability.get("average_reliability", 0.5)
            
            # Add reliability indicator
            if avg_reliability >= 0.8:
                reliability_indicator = "✓ High reliability"
            elif avg_reliability >= 0.6:
                reliability_indicator = "~ Medium reliability"
            else:
                reliability_indicator = "! Lower reliability"
            
            return f"{basic_citation} [{reliability_indicator}]"
            
        except Exception as e:
            self.logger.error(f"Error formatting enhanced citation: {str(e)}")
            return self._format_single_citation(citation)
    
    def _format_confidence_summary(self, confidence_breakdown: Dict[str, Any]) -> str:
        """Format confidence breakdown summary for bibliography."""
        try:
            summary_parts = ["\n**Confidence Analysis:**"]
            
            # Graph evidence
            graph_evidence = confidence_breakdown.get("graph_evidence", {})
            entity_support = graph_evidence.get("entity_support", 0.5)
            relationship_support = graph_evidence.get("relationship_support", 0.5)
            
            summary_parts.append(f"- Entity support: {entity_support:.2f}")
            summary_parts.append(f"- Relationship support: {relationship_support:.2f}")
            
            # Source reliability
            source_reliability = confidence_breakdown.get("source_reliability", {})
            avg_reliability = source_reliability.get("average_reliability", 0.5)
            source_count = source_reliability.get("source_count", 0)
            
            summary_parts.append(f"- Average source reliability: {avg_reliability:.2f}")
            summary_parts.append(f"- Sources analyzed: {source_count}")
            
            # Improvement
            improvement = confidence_breakdown.get("improvement", 0.0)
            if improvement > 0:
                summary_parts.append(f"- Confidence improvement: +{improvement:.2f}")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error formatting confidence summary: {str(e)}")
            return ""
    
    def _get_confidence_color(self, score: float) -> str:
        """Get confidence color for UI display."""
        if score >= 0.8:
            return "green"
        elif score >= 0.6:
            return "orange"
        elif score >= 0.4:
            return "red"
        else:
            return "darkred"