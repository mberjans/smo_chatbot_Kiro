"""
Confidence Scoring for LightRAG Citations

This module implements confidence scoring that works with graph-based evidence,
source document reliability scoring, and citation confidence display.

Implements requirements 4.3 and 4.6.
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter

try:
    from lightrag_integration.utils.logging import setup_logger
except ImportError:
    # Fallback to basic logging if utils not available
    import logging
    def setup_logger(name, log_file=None):
        return logging.getLogger(name)


@dataclass
class ConfidenceFactors:
    """Individual confidence factors for scoring."""
    entity_confidence: float = 0.0
    relationship_confidence: float = 0.0
    source_reliability: float = 0.0
    graph_connectivity: float = 0.0
    evidence_consistency: float = 0.0
    citation_quality: float = 0.0
    temporal_relevance: float = 0.0


@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of confidence scoring."""
    overall_confidence: float
    base_confidence: float
    enhancement_factor: float
    confidence_factors: ConfidenceFactors
    source_scores: Dict[str, float]
    entity_scores: Dict[str, float]
    relationship_scores: Dict[str, float]
    explanation: str


@dataclass
class SourceReliability:
    """Reliability assessment for a source document."""
    document_path: str
    reliability_score: float
    factors: Dict[str, float]
    metadata_quality: float
    citation_frequency: int
    content_quality: float


class LightRAGConfidenceScorer:
    """
    Confidence scoring system for LightRAG responses.
    
    This class implements sophisticated confidence scoring that considers
    graph-based evidence strength, source document reliability, and
    citation quality factors.
    """
    
    def __init__(self, config=None):
        """Initialize the confidence scorer."""
        self.config = config
        self.logger = setup_logger("lightrag_confidence_scorer")
        
        # Confidence scoring weights
        self.weights = {
            "entity_confidence": 0.20,
            "relationship_confidence": 0.20,
            "source_reliability": 0.25,
            "graph_connectivity": 0.15,
            "evidence_consistency": 0.10,
            "citation_quality": 0.05,
            "temporal_relevance": 0.05
        }
        
        # Confidence thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        self.low_confidence_threshold = 0.4
        
        # Source reliability cache
        self._source_reliability_cache: Dict[str, SourceReliability] = {}
        
        self.logger.info("LightRAG Confidence Scorer initialized")
    
    def calculate_response_confidence(
        self,
        base_confidence: float,
        source_documents: List[str],
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        query_context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceBreakdown:
        """
        Calculate comprehensive confidence score for a LightRAG response.
        
        Args:
            base_confidence: Base confidence from query processing
            source_documents: List of source document paths
            entities_used: List of entities used in the response
            relationships_used: List of relationships used in the response
            query_context: Optional context about the query
        
        Returns:
            ConfidenceBreakdown with detailed scoring information
        """
        try:
            self.logger.info(f"Calculating confidence for response with {len(source_documents)} sources")
            
            # Calculate individual confidence factors
            factors = self._calculate_confidence_factors(
                source_documents, entities_used, relationships_used, query_context
            )
            
            # Calculate source reliability scores
            source_scores = self._calculate_source_reliability_scores(
                source_documents, entities_used, relationships_used
            )
            
            # Calculate entity confidence scores
            entity_scores = self._calculate_entity_confidence_scores(entities_used)
            
            # Calculate relationship confidence scores
            relationship_scores = self._calculate_relationship_confidence_scores(relationships_used)
            
            # Calculate weighted enhancement factor
            enhancement_factor = self._calculate_enhancement_factor(factors)
            
            # Apply enhancement to base confidence
            enhanced_confidence = self._apply_confidence_enhancement(
                base_confidence, enhancement_factor
            )
            
            # Generate explanation
            explanation = self._generate_confidence_explanation(
                base_confidence, enhanced_confidence, factors, source_scores
            )
            
            breakdown = ConfidenceBreakdown(
                overall_confidence=enhanced_confidence,
                base_confidence=base_confidence,
                enhancement_factor=enhancement_factor,
                confidence_factors=factors,
                source_scores=source_scores,
                entity_scores=entity_scores,
                relationship_scores=relationship_scores,
                explanation=explanation
            )
            
            self.logger.info(f"Confidence calculated: {enhanced_confidence:.3f} (base: {base_confidence:.3f})")
            return breakdown
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}", exc_info=True)
            
            # Return fallback confidence breakdown
            return ConfidenceBreakdown(
                overall_confidence=base_confidence,
                base_confidence=base_confidence,
                enhancement_factor=0.0,
                confidence_factors=ConfidenceFactors(),
                source_scores={},
                entity_scores={},
                relationship_scores={},
                explanation=f"Error calculating confidence: {str(e)}"
            )
    
    def _calculate_confidence_factors(
        self,
        source_documents: List[str],
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        query_context: Optional[Dict[str, Any]]
    ) -> ConfidenceFactors:
        """Calculate individual confidence factors."""
        factors = ConfidenceFactors()
        
        try:
            # Entity confidence factor
            if entities_used:
                entity_confidences = [
                    entity.get("relevance_score", 0.5) 
                    for entity in entities_used
                ]
                factors.entity_confidence = sum(entity_confidences) / len(entity_confidences)
            
            # Relationship confidence factor
            if relationships_used:
                rel_confidences = [
                    rel.get("confidence", 0.5) 
                    for rel in relationships_used
                ]
                factors.relationship_confidence = sum(rel_confidences) / len(rel_confidences)
            
            # Source reliability factor
            if source_documents:
                source_reliabilities = []
                for doc in source_documents:
                    reliability = self._assess_source_reliability(
                        doc, entities_used, relationships_used
                    )
                    source_reliabilities.append(reliability.reliability_score)
                factors.source_reliability = sum(source_reliabilities) / len(source_reliabilities)
            
            # Graph connectivity factor
            factors.graph_connectivity = self._calculate_graph_connectivity(
                entities_used, relationships_used
            )
            
            # Evidence consistency factor
            factors.evidence_consistency = self._calculate_evidence_consistency(
                entities_used, relationships_used, source_documents
            )
            
            # Citation quality factor
            factors.citation_quality = self._calculate_citation_quality(
                source_documents, entities_used
            )
            
            # Temporal relevance factor
            factors.temporal_relevance = self._calculate_temporal_relevance(
                source_documents, query_context
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence factors: {str(e)}")
        
        return factors
    
    def _assess_source_reliability(
        self,
        document_path: str,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> SourceReliability:
        """Assess the reliability of a source document."""
        # Check cache first
        if document_path in self._source_reliability_cache:
            return self._source_reliability_cache[document_path]
        
        try:
            factors = {}
            
            # Metadata quality assessment
            metadata_quality = self._assess_metadata_quality(document_path)
            factors["metadata_quality"] = metadata_quality
            
            # Citation frequency (how often this document is referenced)
            citation_frequency = self._calculate_citation_frequency(
                document_path, entities_used, relationships_used
            )
            factors["citation_frequency"] = min(citation_frequency / 10.0, 1.0)  # Normalize
            
            # Content quality assessment
            content_quality = self._assess_content_quality(
                document_path, entities_used, relationships_used
            )
            factors["content_quality"] = content_quality
            
            # Calculate overall reliability score
            reliability_score = (
                factors["metadata_quality"] * 0.3 +
                factors["citation_frequency"] * 0.4 +
                factors["content_quality"] * 0.3
            )
            
            reliability = SourceReliability(
                document_path=document_path,
                reliability_score=reliability_score,
                factors=factors,
                metadata_quality=metadata_quality,
                citation_frequency=citation_frequency,
                content_quality=content_quality
            )
            
            # Cache the result
            self._source_reliability_cache[document_path] = reliability
            
            return reliability
            
        except Exception as e:
            self.logger.error(f"Error assessing source reliability for {document_path}: {str(e)}")
            
            # Return default reliability
            return SourceReliability(
                document_path=document_path,
                reliability_score=0.5,
                factors={"error": 1.0},
                metadata_quality=0.5,
                citation_frequency=1,
                content_quality=0.5
            )
    
    def _assess_metadata_quality(self, document_path: str) -> float:
        """Assess the quality of document metadata."""
        try:
            from pathlib import Path
            
            # Basic assessment based on filename and path structure
            path = Path(document_path)
            
            quality_score = 0.5  # Base score
            
            # Bonus for structured filename
            filename = path.stem.lower()
            if any(keyword in filename for keyword in ["clinical", "metabolomics", "biomarker", "research"]):
                quality_score += 0.1
            
            # Bonus for year in filename
            import re
            if re.search(r'(19|20)\d{2}', filename):
                quality_score += 0.1
            
            # Bonus for author names pattern
            if re.search(r'[a-z]+_[a-z]+', filename):
                quality_score += 0.1
            
            # Bonus for journal or conference patterns
            if any(keyword in filename for keyword in ["journal", "nature", "science", "cell", "plos"]):
                quality_score += 0.2
            
            return min(quality_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error assessing metadata quality: {str(e)}")
            return 0.5
    
    def _calculate_citation_frequency(
        self,
        document_path: str,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> int:
        """Calculate how frequently a document is cited."""
        frequency = 0
        
        # Count entity references
        for entity in entities_used:
            if document_path in entity.get("source_documents", []):
                frequency += 1
        
        # Count relationship references
        for rel in relationships_used:
            if document_path in rel.get("source_documents", []):
                frequency += 1
        
        return frequency
    
    def _assess_content_quality(
        self,
        document_path: str,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> float:
        """Assess the quality of content extracted from the document."""
        try:
            # Calculate based on entity and relationship quality from this document
            entity_quality_scores = []
            for entity in entities_used:
                if document_path in entity.get("source_documents", []):
                    entity_quality_scores.append(entity.get("relevance_score", 0.5))
            
            relationship_quality_scores = []
            for rel in relationships_used:
                if document_path in rel.get("source_documents", []):
                    relationship_quality_scores.append(rel.get("confidence", 0.5))
            
            all_scores = entity_quality_scores + relationship_quality_scores
            
            if all_scores:
                return sum(all_scores) / len(all_scores)
            else:
                return 0.5  # Default quality
                
        except Exception as e:
            self.logger.error(f"Error assessing content quality: {str(e)}")
            return 0.5
    
    def _calculate_graph_connectivity(
        self,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well-connected the entities are in the graph."""
        try:
            if not entities_used or not relationships_used:
                return 0.0
            
            # Create entity ID set
            entity_ids = {entity.get("id", "") for entity in entities_used}
            
            # Count how many entities are connected by relationships
            connected_entities = set()
            for rel in relationships_used:
                source_id = rel.get("source", "")
                target_id = rel.get("target", "")
                
                if source_id in entity_ids:
                    connected_entities.add(source_id)
                if target_id in entity_ids:
                    connected_entities.add(target_id)
            
            # Calculate connectivity ratio
            if len(entity_ids) > 0:
                connectivity = len(connected_entities) / len(entity_ids)
                return min(connectivity, 1.0)
            
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating graph connectivity: {str(e)}")
            return 0.0
    
    def _calculate_evidence_consistency(
        self,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        source_documents: List[str]
    ) -> float:
        """Calculate consistency of evidence across sources."""
        try:
            if not source_documents:
                return 0.0
            
            # Count how many sources support each entity/relationship
            entity_source_counts = defaultdict(set)
            for entity in entities_used:
                entity_id = entity.get("id", "")
                for doc in entity.get("source_documents", []):
                    entity_source_counts[entity_id].add(doc)
            
            relationship_source_counts = defaultdict(set)
            for rel in relationships_used:
                rel_id = rel.get("id", "")
                for doc in rel.get("source_documents", []):
                    relationship_source_counts[rel_id].add(doc)
            
            # Calculate consistency score based on multi-source support
            total_items = len(entity_source_counts) + len(relationship_source_counts)
            if total_items == 0:
                return 0.0
            
            multi_source_items = 0
            for sources in entity_source_counts.values():
                if len(sources) > 1:
                    multi_source_items += 1
            
            for sources in relationship_source_counts.values():
                if len(sources) > 1:
                    multi_source_items += 1
            
            consistency = multi_source_items / total_items
            return min(consistency, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating evidence consistency: {str(e)}")
            return 0.0
    
    def _calculate_citation_quality(
        self,
        source_documents: List[str],
        entities_used: List[Dict[str, Any]]
    ) -> float:
        """Calculate the quality of citations."""
        try:
            if not source_documents:
                return 0.0
            
            quality_scores = []
            
            for doc in source_documents:
                # Basic quality assessment
                quality = 0.5  # Base quality
                
                # Bonus for PDF files
                if doc.lower().endswith('.pdf'):
                    quality += 0.2
                
                # Bonus for academic-looking filenames
                filename = doc.lower()
                if any(keyword in filename for keyword in [
                    'journal', 'research', 'study', 'clinical', 'metabolomics'
                ]):
                    quality += 0.2
                
                # Bonus for having page numbers or sections
                doc_entities = [e for e in entities_used if doc in e.get("source_documents", [])]
                has_location_info = any(
                    "page" in e.get("properties", {}) or "section" in e.get("properties", {})
                    for e in doc_entities
                )
                if has_location_info:
                    quality += 0.1
                
                quality_scores.append(min(quality, 1.0))
            
            return sum(quality_scores) / len(quality_scores) if quality_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating citation quality: {str(e)}")
            return 0.0
    
    def _calculate_temporal_relevance(
        self,
        source_documents: List[str],
        query_context: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate temporal relevance of sources."""
        try:
            if not source_documents:
                return 0.0
            
            import re
            from datetime import datetime
            
            current_year = datetime.now().year
            relevance_scores = []
            
            for doc in source_documents:
                # Try to extract year from filename
                year_match = re.search(r'(19|20)(\d{2})', doc)
                if year_match:
                    doc_year = int(year_match.group())
                    
                    # Calculate relevance based on recency
                    years_old = current_year - doc_year
                    
                    if years_old <= 2:
                        relevance = 1.0  # Very recent
                    elif years_old <= 5:
                        relevance = 0.8  # Recent
                    elif years_old <= 10:
                        relevance = 0.6  # Moderately recent
                    else:
                        relevance = 0.4  # Older
                    
                    relevance_scores.append(relevance)
                else:
                    # No year found, assume moderate relevance
                    relevance_scores.append(0.6)
            
            return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.6
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal relevance: {str(e)}")
            return 0.6
    
    def _calculate_source_reliability_scores(
        self,
        source_documents: List[str],
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate reliability scores for all source documents."""
        scores = {}
        
        for doc in source_documents:
            reliability = self._assess_source_reliability(doc, entities_used, relationships_used)
            scores[doc] = reliability.reliability_score
        
        return scores
    
    def _calculate_entity_confidence_scores(
        self,
        entities_used: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate confidence scores for entities."""
        scores = {}
        
        for entity in entities_used:
            entity_id = entity.get("id", "")
            confidence = entity.get("relevance_score", 0.5)
            scores[entity_id] = confidence
        
        return scores
    
    def _calculate_relationship_confidence_scores(
        self,
        relationships_used: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate confidence scores for relationships."""
        scores = {}
        
        for rel in relationships_used:
            rel_id = rel.get("id", "")
            confidence = rel.get("confidence", 0.5)
            scores[rel_id] = confidence
        
        return scores
    
    def _calculate_enhancement_factor(self, factors: ConfidenceFactors) -> float:
        """Calculate the overall enhancement factor from individual factors."""
        try:
            enhancement = (
                factors.entity_confidence * self.weights["entity_confidence"] +
                factors.relationship_confidence * self.weights["relationship_confidence"] +
                factors.source_reliability * self.weights["source_reliability"] +
                factors.graph_connectivity * self.weights["graph_connectivity"] +
                factors.evidence_consistency * self.weights["evidence_consistency"] +
                factors.citation_quality * self.weights["citation_quality"] +
                factors.temporal_relevance * self.weights["temporal_relevance"]
            )
            
            return min(enhancement, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating enhancement factor: {str(e)}")
            return 0.0
    
    def _apply_confidence_enhancement(
        self,
        base_confidence: float,
        enhancement_factor: float
    ) -> float:
        """Apply enhancement factor to base confidence."""
        try:
            # Use a logarithmic enhancement to prevent over-confidence
            enhancement_multiplier = 0.3  # Maximum 30% enhancement
            enhancement = enhancement_factor * enhancement_multiplier
            
            # Apply enhancement with diminishing returns
            enhanced = base_confidence + (enhancement * (1 - base_confidence))
            
            return min(enhanced, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error applying confidence enhancement: {str(e)}")
            return base_confidence
    
    def _generate_confidence_explanation(
        self,
        base_confidence: float,
        enhanced_confidence: float,
        factors: ConfidenceFactors,
        source_scores: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation of confidence scoring."""
        try:
            explanation_parts = []
            
            # Overall confidence
            confidence_level = self._get_confidence_level(enhanced_confidence)
            explanation_parts.append(f"Overall confidence: {confidence_level} ({enhanced_confidence:.2f})")
            
            if enhanced_confidence != base_confidence:
                enhancement = enhanced_confidence - base_confidence
                explanation_parts.append(f"Enhanced from base confidence of {base_confidence:.2f} (+{enhancement:.2f})")
            
            # Key factors
            explanation_parts.append("\nKey factors:")
            
            if factors.entity_confidence > 0:
                explanation_parts.append(f"• Entity relevance: {factors.entity_confidence:.2f}")
            
            if factors.relationship_confidence > 0:
                explanation_parts.append(f"• Relationship strength: {factors.relationship_confidence:.2f}")
            
            if factors.source_reliability > 0:
                explanation_parts.append(f"• Source reliability: {factors.source_reliability:.2f}")
            
            if factors.graph_connectivity > 0:
                explanation_parts.append(f"• Graph connectivity: {factors.graph_connectivity:.2f}")
            
            # Source quality summary
            if source_scores:
                high_quality_sources = sum(1 for score in source_scores.values() if score >= 0.8)
                total_sources = len(source_scores)
                explanation_parts.append(f"\nSources: {high_quality_sources}/{total_sources} high-quality sources")
            
            return "\n".join(explanation_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating confidence explanation: {str(e)}")
            return f"Confidence: {enhanced_confidence:.2f}"
    
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
    
    def get_confidence_display_info(
        self,
        confidence_breakdown: ConfidenceBreakdown
    ) -> Dict[str, Any]:
        """
        Get information for displaying confidence in the UI.
        
        Returns:
            Dictionary with display information including colors, icons, and text
        """
        try:
            confidence = confidence_breakdown.overall_confidence
            level = self._get_confidence_level(confidence)
            
            # Color coding
            if confidence >= self.high_confidence_threshold:
                color = "green"
                icon = "✓"
            elif confidence >= self.medium_confidence_threshold:
                color = "orange"
                icon = "⚠"
            elif confidence >= self.low_confidence_threshold:
                color = "red"
                icon = "⚠"
            else:
                color = "darkred"
                icon = "✗"
            
            # Display text
            display_text = f"{level} Confidence ({confidence:.1%})"
            
            # Tooltip with detailed breakdown
            tooltip = confidence_breakdown.explanation
            
            return {
                "confidence_score": confidence,
                "confidence_level": level,
                "color": color,
                "icon": icon,
                "display_text": display_text,
                "tooltip": tooltip,
                "show_warning": confidence < self.medium_confidence_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error getting confidence display info: {str(e)}")
            return {
                "confidence_score": 0.0,
                "confidence_level": "Unknown",
                "color": "gray",
                "icon": "?",
                "display_text": "Unknown Confidence",
                "tooltip": "Error calculating confidence",
                "show_warning": True
            }


# Alias for backward compatibility
class ConfidenceScorer:
    """Alias for LightRAGConfidenceScorer for backward compatibility."""
    
    def __init__(self, config=None):
        self._scorer = LightRAGConfidenceScorer(config)
    
    def __getattr__(self, name):
        return getattr(self._scorer, name)