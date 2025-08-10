"""
Enhanced Confidence Scoring for LightRAG Citations

This module implements confidence scoring that works with graph-based evidence,
source document reliability scoring, and citation confidence display.

Implements requirements 4.3 and 4.6.
"""

import math
import statistics
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from .utils.logging import setup_logger
except ImportError:
    import logging
    def setup_logger(name, log_file=None):
        return logging.getLogger(name)


@dataclass
class SourceReliabilityScore:
    """Reliability score for a source document."""
    document_path: str
    reliability_score: float
    factors: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class CitationConfidenceScore:
    """Confidence score for a citation."""
    citation_id: str
    confidence_score: float
    reliability_score: float
    evidence_strength: float
    factors: Dict[str, float]


@dataclass
class GraphEvidenceMetrics:
    """Metrics for graph-based evidence strength."""
    entity_support: float
    relationship_support: float
    connectivity_score: float
    consistency_score: float
    diversity_score: float


class LightRAGConfidenceScorer:
    """
    Enhanced confidence scoring system for LightRAG responses.
    
    This class implements confidence scoring that considers:
    - Graph-based evidence strength
    - Source document reliability
    - Citation confidence for UI display
    - Multi-factor confidence analysis
    """
    
    def __init__(self, config=None):
        """Initialize the confidence scorer."""
        self.config = config
        self.logger = setup_logger("lightrag_confidence_scorer")
        
        # Confidence scoring parameters
        self.min_confidence = 0.1
        self.max_confidence = 1.0
        self.default_reliability = 0.6
        
        # Weight factors for different confidence components
        self.weights = {
            "graph_evidence": 0.35,
            "source_reliability": 0.25,
            "entity_confidence": 0.20,
            "relationship_confidence": 0.15,
            "citation_consistency": 0.05
        }
        
        # Reliability scoring factors
        self.reliability_factors = {
            "file_accessibility": 0.2,
            "metadata_completeness": 0.15,
            "citation_frequency": 0.25,
            "content_quality": 0.2,
            "recency": 0.1,
            "source_diversity": 0.1
        }
        
        self.logger.info("LightRAG Confidence Scorer initialized")
    
    def calculate_enhanced_confidence(
        self,
        base_confidence: float,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        source_documents: List[str],
        citation_map: Optional[Dict[str, Any]] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate enhanced confidence score based on graph evidence and source reliability.
        
        Args:
            base_confidence: Base confidence from query processing
            entities_used: List of entities used in the response
            relationships_used: List of relationships used in the response
            source_documents: List of source document paths
            citation_map: Optional citation map for additional context
        
        Returns:
            Tuple of (enhanced_confidence, detailed_breakdown)
        """
        try:
            self.logger.info(f"Calculating enhanced confidence for {len(source_documents)} sources")
            
            # Calculate graph evidence metrics
            graph_metrics = self._calculate_graph_evidence_metrics(
                entities_used, relationships_used, source_documents
            )
            
            # Calculate source reliability scores
            reliability_scores = self._calculate_source_reliability_scores(
                source_documents, entities_used, relationships_used
            )
            
            # Calculate citation confidence scores
            citation_confidences = self._calculate_citation_confidence_scores(
                citation_map or {}, reliability_scores, graph_metrics
            )
            
            # Combine all factors for enhanced confidence
            enhanced_confidence = self._combine_confidence_factors(
                base_confidence, graph_metrics, reliability_scores, citation_confidences
            )
            
            # Create detailed breakdown
            breakdown = self._create_confidence_breakdown(
                base_confidence, enhanced_confidence, graph_metrics,
                reliability_scores, citation_confidences
            )
            
            self.logger.info(f"Enhanced confidence: {enhanced_confidence:.3f} (base: {base_confidence:.3f})")
            return enhanced_confidence, breakdown
            
        except Exception as e:
            self.logger.error(f"Error calculating enhanced confidence: {str(e)}", exc_info=True)
            return base_confidence, {"error": str(e), "base_confidence": base_confidence}
    
    def _calculate_graph_evidence_metrics(
        self,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        source_documents: List[str]
    ) -> GraphEvidenceMetrics:
        """Calculate metrics for graph-based evidence strength."""
        try:
            # Entity support score
            entity_support = 0.0
            if entities_used:
                entity_confidences = [
                    entity.get("relevance_score", 0.5) for entity in entities_used
                ]
                entity_support = statistics.mean(entity_confidences)
                
                # Boost for multiple high-confidence entities
                high_conf_entities = sum(1 for conf in entity_confidences if conf >= 0.8)
                entity_support += min(high_conf_entities * 0.05, 0.2)
            
            # Relationship support score
            relationship_support = 0.0
            if relationships_used:
                rel_confidences = [
                    rel.get("confidence", 0.5) for rel in relationships_used
                ]
                relationship_support = statistics.mean(rel_confidences)
                
                # Boost for multiple relationships
                relationship_support += min(len(relationships_used) * 0.03, 0.15)
            
            # Connectivity score (how well entities are connected)
            connectivity_score = self._calculate_connectivity_score(
                entities_used, relationships_used
            )
            
            # Consistency score (entities and relationships from same sources)
            consistency_score = self._calculate_consistency_score(
                entities_used, relationships_used, source_documents
            )
            
            # Diversity score (evidence from multiple sources)
            diversity_score = self._calculate_diversity_score(
                entities_used, relationships_used, source_documents
            )
            
            return GraphEvidenceMetrics(
                entity_support=min(entity_support, 1.0),
                relationship_support=min(relationship_support, 1.0),
                connectivity_score=connectivity_score,
                consistency_score=consistency_score,
                diversity_score=diversity_score
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating graph evidence metrics: {str(e)}")
            return GraphEvidenceMetrics(0.5, 0.5, 0.5, 0.5, 0.5)
    
    def _calculate_connectivity_score(
        self,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> float:
        """Calculate how well entities are connected through relationships."""
        if not entities_used or not relationships_used:
            return 0.3  # Low connectivity for isolated entities
        
        # Create entity ID set
        entity_ids = {entity.get("id", "") for entity in entities_used}
        
        # Count how many entities are connected through relationships
        connected_entities = set()
        for rel in relationships_used:
            source_id = rel.get("source", "")
            target_id = rel.get("target", "")
            
            if source_id in entity_ids:
                connected_entities.add(source_id)
            if target_id in entity_ids:
                connected_entities.add(target_id)
        
        # Calculate connectivity ratio
        if len(entity_ids) == 0:
            return 0.3
        
        connectivity_ratio = len(connected_entities) / len(entity_ids)
        
        # Boost for high connectivity
        if connectivity_ratio >= 0.8:
            return min(connectivity_ratio + 0.1, 1.0)
        
        return connectivity_ratio
    
    def _calculate_consistency_score(
        self,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        source_documents: List[str]
    ) -> float:
        """Calculate consistency of evidence across sources."""
        if not source_documents:
            return 0.5
        
        # Count source overlap between entities and relationships
        entity_sources = set()
        for entity in entities_used:
            entity_sources.update(entity.get("source_documents", []))
        
        rel_sources = set()
        for rel in relationships_used:
            rel_sources.update(rel.get("source_documents", []))
        
        all_sources = entity_sources.union(rel_sources)
        overlapping_sources = entity_sources.intersection(rel_sources)
        
        if len(all_sources) == 0:
            return 0.5
        
        # Higher consistency when entities and relationships share sources
        consistency = len(overlapping_sources) / len(all_sources)
        
        # Boost for multiple overlapping sources
        if len(overlapping_sources) >= 2:
            consistency += 0.1
        
        return min(consistency, 1.0)
    
    def _calculate_diversity_score(
        self,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]],
        source_documents: List[str]
    ) -> float:
        """Calculate diversity of evidence sources."""
        unique_sources = set(source_documents)
        
        if len(unique_sources) == 0:
            return 0.0
        elif len(unique_sources) == 1:
            return 0.4  # Single source
        elif len(unique_sources) == 2:
            return 0.7  # Two sources
        else:
            return min(0.8 + (len(unique_sources) - 3) * 0.05, 1.0)  # Multiple sources
    
    def _calculate_source_reliability_scores(
        self,
        source_documents: List[str],
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> List[SourceReliabilityScore]:
        """Calculate reliability scores for source documents."""
        reliability_scores = []
        
        for doc_path in set(source_documents):
            try:
                factors = {}
                
                # File accessibility factor
                factors["file_accessibility"] = self._check_file_accessibility(doc_path)
                
                # Metadata completeness factor
                factors["metadata_completeness"] = self._assess_metadata_completeness(doc_path)
                
                # Citation frequency factor
                factors["citation_frequency"] = self._calculate_citation_frequency(
                    doc_path, entities_used, relationships_used
                )
                
                # Content quality factor (based on entity/relationship confidence)
                factors["content_quality"] = self._assess_content_quality(
                    doc_path, entities_used, relationships_used
                )
                
                # Recency factor
                factors["recency"] = self._assess_document_recency(doc_path)
                
                # Source diversity factor (how unique this source is)
                factors["source_diversity"] = self._assess_source_diversity(
                    doc_path, source_documents
                )
                
                # Calculate weighted reliability score
                reliability_score = sum(
                    factors[factor] * self.reliability_factors[factor]
                    for factor in factors
                )
                
                reliability_scores.append(SourceReliabilityScore(
                    document_path=doc_path,
                    reliability_score=min(max(reliability_score, self.min_confidence), self.max_confidence),
                    factors=factors,
                    metadata={"path": doc_path}
                ))
                
            except Exception as e:
                self.logger.error(f"Error calculating reliability for {doc_path}: {str(e)}")
                # Fallback reliability score
                reliability_scores.append(SourceReliabilityScore(
                    document_path=doc_path,
                    reliability_score=self.default_reliability,
                    factors={"error": 1.0},
                    metadata={"error": str(e)}
                ))
        
        return reliability_scores
    
    def _check_file_accessibility(self, doc_path: str) -> float:
        """Check if the document file is accessible."""
        try:
            path = Path(doc_path)
            if path.exists() and path.is_file():
                # Check if file is readable
                try:
                    with open(path, 'rb') as f:
                        f.read(1)  # Try to read first byte
                    return 1.0  # Fully accessible
                except Exception:
                    return 0.5  # Exists but not readable
            else:
                return 0.0  # File doesn't exist
        except Exception:
            return 0.0
    
    def _assess_metadata_completeness(self, doc_path: str) -> float:
        """Assess completeness of document metadata."""
        try:
            path = Path(doc_path)
            
            # Basic metadata checks
            score = 0.0
            
            # File extension check
            if path.suffix.lower() in ['.pdf', '.txt', '.doc', '.docx']:
                score += 0.3
            
            # Filename informativeness (contains year, meaningful words)
            filename = path.stem.lower()
            if any(year in filename for year in ['2020', '2021', '2022', '2023', '2024']):
                score += 0.2
            
            if len(filename.split('_')) >= 2 or len(filename.split(' ')) >= 2:
                score += 0.2
            
            # File size reasonableness (not empty, not too large)
            if path.exists():
                size = path.stat().st_size
                if 1000 < size < 50_000_000:  # Between 1KB and 50MB
                    score += 0.3
                elif size > 0:
                    score += 0.1
            
            return min(score, 1.0)
            
        except Exception:
            return 0.3  # Default moderate score
    
    def _calculate_citation_frequency(
        self,
        doc_path: str,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> float:
        """Calculate how frequently this document is cited."""
        citation_count = 0
        
        # Count entity citations
        for entity in entities_used:
            if doc_path in entity.get("source_documents", []):
                citation_count += 1
        
        # Count relationship citations
        for rel in relationships_used:
            if doc_path in rel.get("source_documents", []):
                citation_count += 1
        
        # Normalize citation frequency
        total_citations = len(entities_used) + len(relationships_used)
        if total_citations == 0:
            return 0.5
        
        frequency = citation_count / total_citations
        
        # Boost for high citation frequency
        if frequency >= 0.5:
            return min(frequency + 0.2, 1.0)
        
        return frequency
    
    def _assess_content_quality(
        self,
        doc_path: str,
        entities_used: List[Dict[str, Any]],
        relationships_used: List[Dict[str, Any]]
    ) -> float:
        """Assess content quality based on entity/relationship confidence."""
        confidences = []
        
        # Collect confidences from entities citing this document
        for entity in entities_used:
            if doc_path in entity.get("source_documents", []):
                confidences.append(entity.get("relevance_score", 0.5))
        
        # Collect confidences from relationships citing this document
        for rel in relationships_used:
            if doc_path in rel.get("source_documents", []):
                confidences.append(rel.get("confidence", 0.5))
        
        if not confidences:
            return 0.5  # Default quality
        
        # Use mean confidence as quality indicator
        quality = statistics.mean(confidences)
        
        # Boost for consistently high quality
        if all(conf >= 0.8 for conf in confidences):
            quality += 0.1
        
        return min(quality, 1.0)
    
    def _assess_document_recency(self, doc_path: str) -> float:
        """Assess document recency based on file modification time and filename."""
        try:
            path = Path(doc_path)
            
            # Try to extract year from filename
            filename = path.stem.lower()
            current_year = datetime.now().year
            
            for year_str in ['2024', '2023', '2022', '2021', '2020']:
                if year_str in filename:
                    year = int(year_str)
                    age = current_year - year
                    
                    if age == 0:
                        return 1.0  # Current year
                    elif age == 1:
                        return 0.9  # Last year
                    elif age <= 3:
                        return 0.7  # Recent (within 3 years)
                    elif age <= 5:
                        return 0.5  # Moderately recent
                    else:
                        return 0.3  # Older
            
            # Fallback to file modification time
            if path.exists():
                mod_time = datetime.fromtimestamp(path.stat().st_mtime)
                age_days = (datetime.now() - mod_time).days
                
                if age_days < 365:
                    return 0.8  # Modified within a year
                elif age_days < 365 * 2:
                    return 0.6  # Modified within 2 years
                else:
                    return 0.4  # Older modification
            
            return 0.5  # Default moderate recency
            
        except Exception:
            return 0.5
    
    def _assess_source_diversity(self, doc_path: str, all_sources: List[str]) -> float:
        """Assess how unique/diverse this source is compared to others."""
        if len(all_sources) <= 1:
            return 0.5  # Single source
        
        # Simple diversity based on filename uniqueness
        path = Path(doc_path)
        filename_words = set(path.stem.lower().split('_') + path.stem.lower().split(' '))
        
        # Compare with other sources
        similarity_scores = []
        for other_source in all_sources:
            if other_source != doc_path:
                other_path = Path(other_source)
                other_words = set(other_path.stem.lower().split('_') + other_path.stem.lower().split(' '))
                
                # Calculate Jaccard similarity
                intersection = len(filename_words.intersection(other_words))
                union = len(filename_words.union(other_words))
                
                if union > 0:
                    similarity = intersection / union
                    similarity_scores.append(similarity)
        
        if not similarity_scores:
            return 0.8  # Unique source
        
        # Lower similarity means higher diversity
        avg_similarity = statistics.mean(similarity_scores)
        diversity = 1.0 - avg_similarity
        
        return max(diversity, 0.2)  # Minimum diversity score
    
    def _calculate_citation_confidence_scores(
        self,
        citation_map: Dict[str, Any],
        reliability_scores: List[SourceReliabilityScore],
        graph_metrics: GraphEvidenceMetrics
    ) -> List[CitationConfidenceScore]:
        """Calculate confidence scores for individual citations."""
        citation_confidences = []
        
        # Create reliability lookup
        reliability_lookup = {
            score.document_path: score.reliability_score 
            for score in reliability_scores
        }
        
        for citation_id, citation in citation_map.items():
            try:
                # Get document path
                if hasattr(citation, 'file_path'):
                    doc_path = citation.file_path
                elif isinstance(citation, dict):
                    doc_path = citation.get('file_path', '')
                else:
                    doc_path = ''
                
                # Get base citation confidence
                if hasattr(citation, 'confidence_score'):
                    base_confidence = citation.confidence_score
                elif isinstance(citation, dict):
                    base_confidence = citation.get('confidence_score', 0.5)
                else:
                    base_confidence = 0.5
                
                # Get reliability score
                reliability = reliability_lookup.get(doc_path, self.default_reliability)
                
                # Calculate evidence strength from graph metrics
                evidence_strength = (
                    graph_metrics.entity_support * 0.4 +
                    graph_metrics.relationship_support * 0.3 +
                    graph_metrics.connectivity_score * 0.2 +
                    graph_metrics.consistency_score * 0.1
                )
                
                # Combine factors for final citation confidence
                citation_confidence = (
                    base_confidence * 0.4 +
                    reliability * 0.35 +
                    evidence_strength * 0.25
                )
                
                citation_confidences.append(CitationConfidenceScore(
                    citation_id=citation_id,
                    confidence_score=min(max(citation_confidence, self.min_confidence), self.max_confidence),
                    reliability_score=reliability,
                    evidence_strength=evidence_strength,
                    factors={
                        "base_confidence": base_confidence,
                        "reliability": reliability,
                        "evidence_strength": evidence_strength
                    }
                ))
                
            except Exception as e:
                self.logger.error(f"Error calculating confidence for citation {citation_id}: {str(e)}")
                # Fallback confidence
                citation_confidences.append(CitationConfidenceScore(
                    citation_id=citation_id,
                    confidence_score=0.5,
                    reliability_score=0.5,
                    evidence_strength=0.5,
                    factors={"error": str(e)}
                ))
        
        return citation_confidences
    
    def _combine_confidence_factors(
        self,
        base_confidence: float,
        graph_metrics: GraphEvidenceMetrics,
        reliability_scores: List[SourceReliabilityScore],
        citation_confidences: List[CitationConfidenceScore]
    ) -> float:
        """Combine all confidence factors into final enhanced confidence."""
        try:
            # Graph evidence component
            graph_evidence_score = (
                graph_metrics.entity_support * 0.3 +
                graph_metrics.relationship_support * 0.25 +
                graph_metrics.connectivity_score * 0.2 +
                graph_metrics.consistency_score * 0.15 +
                graph_metrics.diversity_score * 0.1
            )
            
            # Source reliability component
            if reliability_scores:
                avg_reliability = statistics.mean([score.reliability_score for score in reliability_scores])
            else:
                avg_reliability = self.default_reliability
            
            # Entity confidence component
            entity_confidence = graph_metrics.entity_support
            
            # Relationship confidence component
            relationship_confidence = graph_metrics.relationship_support
            
            # Citation consistency component
            if citation_confidences:
                citation_consistency = statistics.mean([
                    score.confidence_score for score in citation_confidences
                ])
            else:
                citation_consistency = base_confidence
            
            # Weighted combination
            enhanced_confidence = (
                graph_evidence_score * self.weights["graph_evidence"] +
                avg_reliability * self.weights["source_reliability"] +
                entity_confidence * self.weights["entity_confidence"] +
                relationship_confidence * self.weights["relationship_confidence"] +
                citation_consistency * self.weights["citation_consistency"]
            )
            
            # Apply base confidence influence
            final_confidence = (enhanced_confidence * 0.7) + (base_confidence * 0.3)
            
            return min(max(final_confidence, self.min_confidence), self.max_confidence)
            
        except Exception as e:
            self.logger.error(f"Error combining confidence factors: {str(e)}")
            return base_confidence
    
    def _create_confidence_breakdown(
        self,
        base_confidence: float,
        enhanced_confidence: float,
        graph_metrics: GraphEvidenceMetrics,
        reliability_scores: List[SourceReliabilityScore],
        citation_confidences: List[CitationConfidenceScore]
    ) -> Dict[str, Any]:
        """Create detailed confidence breakdown for analysis."""
        try:
            breakdown = {
                "base_confidence": base_confidence,
                "enhanced_confidence": enhanced_confidence,
                "improvement": enhanced_confidence - base_confidence,
                
                "graph_evidence": {
                    "entity_support": graph_metrics.entity_support,
                    "relationship_support": graph_metrics.relationship_support,
                    "connectivity_score": graph_metrics.connectivity_score,
                    "consistency_score": graph_metrics.consistency_score,
                    "diversity_score": graph_metrics.diversity_score
                },
                
                "source_reliability": {
                    "average_reliability": statistics.mean([s.reliability_score for s in reliability_scores]) if reliability_scores else 0.5,
                    "source_count": len(reliability_scores),
                    "high_reliability_sources": sum(1 for s in reliability_scores if s.reliability_score >= 0.8),
                    "low_reliability_sources": sum(1 for s in reliability_scores if s.reliability_score < 0.5)
                },
                
                "citation_confidence": {
                    "average_citation_confidence": statistics.mean([c.confidence_score for c in citation_confidences]) if citation_confidences else 0.5,
                    "citation_count": len(citation_confidences),
                    "high_confidence_citations": sum(1 for c in citation_confidences if c.confidence_score >= 0.8),
                    "low_confidence_citations": sum(1 for c in citation_confidences if c.confidence_score < 0.5)
                },
                
                "weights_used": self.weights,
                "reliability_factors_used": self.reliability_factors
            }
            
            return breakdown
            
        except Exception as e:
            self.logger.error(f"Error creating confidence breakdown: {str(e)}")
            return {
                "base_confidence": base_confidence,
                "enhanced_confidence": enhanced_confidence,
                "error": str(e)
            }
    
    def format_confidence_for_ui(
        self,
        confidence_score: float,
        citation_confidences: List[CitationConfidenceScore],
        breakdown: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Format confidence information for UI display."""
        try:
            # Determine confidence level and color
            if confidence_score >= 0.8:
                level = "High"
                color = "green"
                icon = "✓"
            elif confidence_score >= 0.6:
                level = "Medium"
                color = "orange"
                icon = "~"
            elif confidence_score >= 0.4:
                level = "Low"
                color = "red"
                icon = "!"
            else:
                level = "Very Low"
                color = "darkred"
                icon = "⚠"
            
            # Create UI-friendly format
            ui_format = {
                "overall": {
                    "score": confidence_score,
                    "level": level,
                    "color": color,
                    "icon": icon,
                    "display_text": f"{level} Confidence ({confidence_score:.2f})"
                },
                
                "citations": [
                    {
                        "citation_id": cite.citation_id,
                        "confidence": cite.confidence_score,
                        "reliability": cite.reliability_score,
                        "level": self._get_confidence_level(cite.confidence_score),
                        "color": self._get_confidence_color(cite.confidence_score)
                    }
                    for cite in citation_confidences
                ],
                
                "breakdown_summary": {
                    "graph_evidence": breakdown.get("graph_evidence", {}).get("entity_support", 0.5),
                    "source_reliability": breakdown.get("source_reliability", {}).get("average_reliability", 0.5),
                    "improvement_from_base": breakdown.get("improvement", 0.0)
                },
                
                "recommendations": self._generate_confidence_recommendations(
                    confidence_score, breakdown
                )
            }
            
            return ui_format
            
        except Exception as e:
            self.logger.error(f"Error formatting confidence for UI: {str(e)}")
            return {
                "overall": {
                    "score": confidence_score,
                    "level": "Unknown",
                    "color": "gray",
                    "icon": "?",
                    "display_text": f"Confidence: {confidence_score:.2f}"
                },
                "error": str(e)
            }
    
    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level string."""
        if score >= 0.8:
            return "High"
        elif score >= 0.6:
            return "Medium"
        elif score >= 0.4:
            return "Low"
        else:
            return "Very Low"
    
    def _get_confidence_color(self, score: float) -> str:
        """Get confidence color for UI."""
        if score >= 0.8:
            return "green"
        elif score >= 0.6:
            return "orange"
        elif score >= 0.4:
            return "red"
        else:
            return "darkred"
    
    def _generate_confidence_recommendations(
        self,
        confidence_score: float,
        breakdown: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving confidence."""
        recommendations = []
        
        try:
            # Low overall confidence
            if confidence_score < 0.6:
                recommendations.append("Consider verifying information with additional sources")
            
            # Low source reliability
            source_reliability = breakdown.get("source_reliability", {})
            if source_reliability.get("average_reliability", 0.5) < 0.6:
                recommendations.append("Some sources may have reliability issues - verify file accessibility")
            
            # Low graph evidence
            graph_evidence = breakdown.get("graph_evidence", {})
            if graph_evidence.get("entity_support", 0.5) < 0.6:
                recommendations.append("Limited entity support - consider additional context")
            
            if graph_evidence.get("connectivity_score", 0.5) < 0.5:
                recommendations.append("Low connectivity between concepts - information may be fragmented")
            
            # Citation issues
            citation_confidence = breakdown.get("citation_confidence", {})
            if citation_confidence.get("low_confidence_citations", 0) > 0:
                recommendations.append("Some citations have low confidence - verify source quality")
            
            if not recommendations:
                recommendations.append("Confidence level is acceptable for this response")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            recommendations = ["Unable to generate specific recommendations"]
        
        return recommendations