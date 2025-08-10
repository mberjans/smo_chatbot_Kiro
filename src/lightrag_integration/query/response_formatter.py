"""
Response Formatter for LightRAG Query Engine

This module handles response formatting consistent with the existing system
and implements confidence scoring based on graph evidence strength.
Implements requirements 4.6 and 8.2.
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from ..utils.logging import setup_logger


@dataclass
class FormattedResponse:
    """Formatted response with consistent structure."""
    content: str
    bibliography: str
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]
    processing_time: float


@dataclass
class CitationInfo:
    """Information about a citation."""
    citation_id: str
    source_document: str
    confidence_score: float
    page_number: Optional[int] = None
    section: Optional[str] = None


class LightRAGResponseFormatter:
    """
    Response formatter for LightRAG query results.
    
    This class formats LightRAG responses to be consistent with the existing
    Clinical Metabolomics Oracle system format, including confidence scores
    and bibliography generation.
    """
    
    def __init__(self, config=None):
        """Initialize the response formatter."""
        self.config = config
        self.logger = setup_logger("lightrag_response_formatter")
        
        # Confidence score thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        self.low_confidence_threshold = 0.4
        
        self.logger.info("LightRAG Response Formatter initialized")
    
    def format_response(
        self, 
        answer: str, 
        source_documents: List[str],
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        confidence_score: float,
        processing_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FormattedResponse:
        """
        Format a LightRAG response to match the existing system format.
        
        Args:
            answer: The generated answer text
            source_documents: List of source document names
            entities_used: List of entities used in the response
            relationships_used: List of relationships used in the response
            confidence_score: Overall confidence score
            processing_time: Time taken to process the query
            metadata: Optional metadata
        
        Returns:
            FormattedResponse with formatted content and bibliography
        """
        try:
            start_time = time.time()
            
            # Generate citations from source documents
            citations = self._generate_citations(
                source_documents, entities_used, relationships_used
            )
            
            # Insert citation markers into the answer
            formatted_content = self._insert_citation_markers(answer, citations)
            
            # Generate bibliography
            bibliography = self._generate_bibliography(citations, confidence_score)
            
            # Add confidence indicators to the content
            formatted_content = self._add_confidence_indicators(
                formatted_content, confidence_score
            )
            
            # Add processing time
            formatted_content += f"\n\n*{processing_time:.2f} seconds*"
            
            # Calculate confidence scores for each citation
            citation_confidence_scores = {
                citation.citation_id: citation.confidence_score
                for citation in citations
            }
            
            format_time = time.time() - start_time
            
            result = FormattedResponse(
                content=formatted_content,
                bibliography=bibliography,
                confidence_scores=citation_confidence_scores,
                metadata={
                    "overall_confidence": confidence_score,
                    "citation_count": len(citations),
                    "entity_count": len(entities_used),
                    "relationship_count": len(relationships_used),
                    "format_time": format_time,
                    **(metadata or {})
                },
                processing_time=processing_time
            )
            
            self.logger.info(f"Response formatted with {len(citations)} citations")
            return result
            
        except Exception as e:
            self.logger.error(f"Error formatting response: {str(e)}", exc_info=True)
            
            # Return a basic formatted response on error
            return FormattedResponse(
                content=answer + f"\n\n*{processing_time:.2f} seconds*",
                bibliography="",
                confidence_scores={},
                metadata={"error": str(e), **(metadata or {})},
                processing_time=processing_time
            )
    
    def _generate_citations(
        self, 
        source_documents: List[str],
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> List[CitationInfo]:
        """Generate citation information from source documents and evidence."""
        citations = []
        citation_counter = 1
        
        # Create citations from source documents
        document_citations = {}
        for doc in set(source_documents):
            if doc and doc.strip():
                # Calculate confidence based on how many entities/relationships reference this document
                entity_refs = sum(1 for entity in entities_used if doc in entity.get("source_documents", []))
                rel_refs = sum(1 for rel in relationships_used if doc in rel.get("source_documents", []))
                
                # Base confidence on number of references and entity/relationship confidence
                total_refs = entity_refs + rel_refs
                if total_refs > 0:
                    avg_entity_confidence = sum(
                        entity.get("relevance_score", 0.5) 
                        for entity in entities_used 
                        if doc in entity.get("source_documents", [])
                    ) / max(entity_refs, 1)
                    
                    avg_rel_confidence = sum(
                        rel.get("confidence", 0.5) 
                        for rel in relationships_used 
                        if doc in rel.get("source_documents", [])
                    ) / max(rel_refs, 1)
                    
                    # Weighted average with boost for multiple references
                    confidence = (avg_entity_confidence + avg_rel_confidence) / 2
                    confidence = min(confidence + (total_refs - 1) * 0.05, 1.0)
                else:
                    confidence = 0.5
                
                citation = CitationInfo(
                    citation_id=str(citation_counter),
                    source_document=doc,
                    confidence_score=confidence
                )
                citations.append(citation)
                document_citations[doc] = citation
                citation_counter += 1
        
        return citations
    
    def _insert_citation_markers(self, answer: str, citations: List[CitationInfo]) -> str:
        """Insert citation markers into the answer text."""
        if not citations:
            return answer
        
        # Create a mapping of document names to citation IDs
        doc_to_citation = {citation.source_document: citation.citation_id for citation in citations}
        
        # For now, add citations at the end of sentences that might reference documents
        # In a more sophisticated implementation, we would use NLP to identify
        # which parts of the text correspond to which sources
        
        # Simple approach: add all citations at the end of the answer
        citation_markers = " ".join([f"[{citation.citation_id}]" for citation in citations])
        
        # Insert citations after the first sentence or at the end
        sentences = answer.split('. ')
        if len(sentences) > 1:
            sentences[0] += f" {citation_markers}"
            return '. '.join(sentences)
        else:
            return f"{answer} {citation_markers}"
    
    def _generate_bibliography(self, citations: List[CitationInfo], overall_confidence: float) -> str:
        """Generate bibliography in the format consistent with the existing system."""
        if not citations:
            return ""
        
        bibliography_parts = []
        
        # Add overall confidence indicator
        confidence_level = self._get_confidence_level(overall_confidence)
        bibliography_parts.append(f"\n\n**Response Confidence: {confidence_level} ({overall_confidence:.2f})**\n")
        
        # Separate citations by confidence level
        high_confidence_citations = [c for c in citations if c.confidence_score >= self.high_confidence_threshold]
        medium_confidence_citations = [c for c in citations if self.medium_confidence_threshold <= c.confidence_score < self.high_confidence_threshold]
        low_confidence_citations = [c for c in citations if c.confidence_score < self.medium_confidence_threshold]
        
        # References section (high and medium confidence)
        references_citations = high_confidence_citations + medium_confidence_citations
        if references_citations:
            bibliography_parts.append("**References:**")
            for citation in references_citations:
                confidence_level = self._get_confidence_level(citation.confidence_score)
                bibliography_parts.append(
                    f"[{citation.citation_id}]: {citation.source_document}\n"
                    f"      (Confidence: {citation.confidence_score:.2f} - {confidence_level})"
                )
        
        # Further Reading section (low confidence)
        if low_confidence_citations:
            bibliography_parts.append("\n**Further Reading:**")
            for citation in low_confidence_citations:
                bibliography_parts.append(f"[{citation.citation_id}]: {citation.source_document}")
        
        return "\n".join(bibliography_parts)
    
    def _add_confidence_indicators(self, content: str, confidence_score: float) -> str:
        """Add confidence indicators to the content."""
        confidence_level = self._get_confidence_level(confidence_score)
        
        # Add a subtle confidence indicator
        if confidence_score >= self.high_confidence_threshold:
            indicator = ""  # No indicator for high confidence
        elif confidence_score >= self.medium_confidence_threshold:
            indicator = " (moderate confidence)"
        else:
            indicator = " (low confidence - please verify)"
        
        # Add indicator to the end of the first sentence
        sentences = content.split('. ')
        if sentences:
            sentences[0] += indicator
            return '. '.join(sentences)
        
        return content + indicator
    
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
    
    def calculate_enhanced_confidence_score(
        self,
        base_confidence: float,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        source_documents: List[str]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate enhanced confidence score based on graph evidence strength.
        
        Args:
            base_confidence: Base confidence from query processing
            entities_used: List of entities used in the response
            relationships_used: List of relationships used in the response
            source_documents: List of source documents
        
        Returns:
            Tuple of (enhanced_confidence_score, confidence_breakdown)
        """
        try:
            # Start with base confidence
            confidence = base_confidence
            
            # Evidence strength factors
            evidence_factors = {
                "entity_confidence": 0.0,
                "relationship_confidence": 0.0,
                "source_diversity": 0.0,
                "evidence_consistency": 0.0,
                "graph_connectivity": 0.0
            }
            
            # Entity confidence factor
            if entities_used:
                entity_confidences = [
                    entity.get("relevance_score", 0.5) 
                    for entity in entities_used
                ]
                evidence_factors["entity_confidence"] = sum(entity_confidences) / len(entity_confidences)
            
            # Relationship confidence factor
            if relationships_used:
                rel_confidences = [
                    rel.get("confidence", 0.5) 
                    for rel in relationships_used
                ]
                evidence_factors["relationship_confidence"] = sum(rel_confidences) / len(rel_confidences)
            
            # Source diversity factor (more diverse sources = higher confidence)
            unique_sources = len(set(source_documents))
            if unique_sources > 0:
                evidence_factors["source_diversity"] = min(unique_sources * 0.1, 0.3)
            
            # Evidence consistency factor (entities and relationships from same sources)
            if entities_used and relationships_used:
                entity_sources = set()
                for entity in entities_used:
                    entity_sources.update(entity.get("source_documents", []))
                
                rel_sources = set()
                for rel in relationships_used:
                    rel_sources.update(rel.get("source_documents", []))
                
                overlap = len(entity_sources.intersection(rel_sources))
                total_sources = len(entity_sources.union(rel_sources))
                
                if total_sources > 0:
                    evidence_factors["evidence_consistency"] = overlap / total_sources
            
            # Graph connectivity factor (more connected entities = higher confidence)
            if len(entities_used) > 1 and relationships_used:
                connected_entities = set()
                for rel in relationships_used:
                    connected_entities.add(rel.get("source", ""))
                    connected_entities.add(rel.get("target", ""))
                
                connectivity_ratio = len(connected_entities) / len(entities_used)
                evidence_factors["graph_connectivity"] = min(connectivity_ratio, 0.2)
            
            # Calculate weighted enhancement
            enhancement = (
                evidence_factors["entity_confidence"] * 0.25 +
                evidence_factors["relationship_confidence"] * 0.25 +
                evidence_factors["source_diversity"] * 0.2 +
                evidence_factors["evidence_consistency"] * 0.15 +
                evidence_factors["graph_connectivity"] * 0.15
            )
            
            # Apply enhancement (cap at 1.0)
            enhanced_confidence = min(confidence + enhancement * 0.3, 1.0)
            
            confidence_breakdown = {
                "base_confidence": base_confidence,
                "enhancement": enhancement,
                "enhanced_confidence": enhanced_confidence,
                "evidence_factors": evidence_factors,
                "entity_count": len(entities_used),
                "relationship_count": len(relationships_used),
                "source_count": len(set(source_documents))
            }
            
            return enhanced_confidence, confidence_breakdown
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced confidence: {str(e)}", exc_info=True)
            return base_confidence, {"error": str(e)}
    
    def format_confidence_explanation(self, confidence_breakdown: Dict[str, Any]) -> str:
        """Format confidence score explanation for debugging/transparency."""
        if "error" in confidence_breakdown:
            return f"Confidence calculation error: {confidence_breakdown['error']}"
        
        explanation_parts = [
            f"Base confidence: {confidence_breakdown['base_confidence']:.2f}",
            f"Enhancement: +{confidence_breakdown['enhancement']:.2f}",
            f"Final confidence: {confidence_breakdown['enhanced_confidence']:.2f}",
            "",
            "Evidence factors:",
        ]
        
        factors = confidence_breakdown.get("evidence_factors", {})
        for factor_name, factor_value in factors.items():
            explanation_parts.append(f"  {factor_name.replace('_', ' ').title()}: {factor_value:.2f}")
        
        explanation_parts.extend([
            "",
            f"Entities used: {confidence_breakdown.get('entity_count', 0)}",
            f"Relationships used: {confidence_breakdown.get('relationship_count', 0)}",
            f"Source documents: {confidence_breakdown.get('source_count', 0)}"
        ])
        
        return "\n".join(explanation_parts)