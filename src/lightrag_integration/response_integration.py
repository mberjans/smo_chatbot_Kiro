"""
Response Integration System for LightRAG

This module handles the integration and processing of responses from different sources
(LightRAG and Perplexity) to provide unified, high-quality responses to users.
Implements requirements 3.7 and 4.1.
"""

import asyncio
import time
import re
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .utils.logging import setup_logger


class ResponseSource(Enum):
    """Enumeration of response sources."""
    LIGHTRAG = "lightrag"
    PERPLEXITY = "perplexity"
    COMBINED = "combined"
    FALLBACK = "fallback"


class ResponseQuality(Enum):
    """Enumeration of response quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    POOR = "poor"


@dataclass
class ProcessedResponse:
    """Processed response with standardized format."""
    content: str
    bibliography: str
    confidence_score: float
    source: ResponseSource
    processing_time: float
    metadata: Dict[str, Any]
    quality_score: float
    citations: List[Dict[str, Any]]
    language: str = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "bibliography": self.bibliography,
            "confidence_score": self.confidence_score,
            "source": self.source.value,
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "citations": self.citations,
            "language": self.language
        }


@dataclass
class CombinedResponse:
    """Combined response from multiple sources."""
    content: str
    bibliography: str
    confidence_score: float
    processing_time: float
    sources_used: List[ResponseSource]
    combination_strategy: str
    metadata: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    citations: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "bibliography": self.bibliography,
            "confidence_score": self.confidence_score,
            "processing_time": self.processing_time,
            "sources_used": [source.value for source in self.sources_used],
            "combination_strategy": self.combination_strategy,
            "metadata": self.metadata,
            "quality_assessment": self.quality_assessment,
            "citations": self.citations
        }


class ResponseIntegrator:
    """
    Integrates and processes responses from different sources.
    
    This class handles the combination of LightRAG and Perplexity responses,
    quality assessment, and response selection to provide the best possible
    answer to user queries.
    """
    
    def __init__(self, config=None):
        """Initialize the response integrator."""
        self.config = config or {}
        self.logger = setup_logger("response_integrator")
        
        # Quality assessment thresholds
        self.high_quality_threshold = 0.8
        self.medium_quality_threshold = 0.6
        self.low_quality_threshold = 0.4
        
        # Confidence score thresholds
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        
        # Response length thresholds
        self.min_response_length = 50
        self.max_response_length = 2000
        
        self.logger.info("Response Integrator initialized")
    
    async def process_lightrag_response(self, response: Dict[str, Any]) -> ProcessedResponse:
        """
        Process LightRAG response for integration.
        
        Args:
            response: Raw LightRAG response dictionary
            
        Returns:
            ProcessedResponse object with standardized format
        """
        try:
            start_time = time.time()
            
            # Extract basic information
            content = response.get("content", response.get("answer", ""))
            bibliography = response.get("bibliography", "")
            confidence_score = response.get("confidence_score", 0.0)
            processing_time = response.get("processing_time", 0.0)
            metadata = response.get("metadata", {})
            
            # Extract citations from LightRAG response
            citations = self._extract_lightrag_citations(response)
            
            # Calculate quality score
            quality_score = self._calculate_response_quality(
                content, confidence_score, citations, ResponseSource.LIGHTRAG
            )
            
            # Clean and format content
            content = self._clean_response_content(content)
            
            # Format bibliography
            bibliography = self._format_lightrag_bibliography(bibliography, citations)
            
            processing_time_total = time.time() - start_time + processing_time
            
            processed_response = ProcessedResponse(
                content=content,
                bibliography=bibliography,
                confidence_score=confidence_score,
                source=ResponseSource.LIGHTRAG,
                processing_time=processing_time_total,
                metadata={
                    **metadata,
                    "original_source": "lightrag",
                    "citations_count": len(citations),
                    "processing_stage": "processed"
                },
                quality_score=quality_score,
                citations=citations
            )
            
            self.logger.info(f"LightRAG response processed: quality={quality_score:.2f}, confidence={confidence_score:.2f}")
            return processed_response
            
        except Exception as e:
            self.logger.error(f"Error processing LightRAG response: {str(e)}", exc_info=True)
            
            # Return a minimal processed response on error
            error_content = response.get("content", response.get("answer", ""))
            if not error_content:
                error_content = "Error processing response"
            
            return ProcessedResponse(
                content=error_content,
                bibliography="",
                confidence_score=0.0,
                source=ResponseSource.LIGHTRAG,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)},
                quality_score=0.0,
                citations=[]
            )
    
    async def process_perplexity_response(self, response: Dict[str, Any]) -> ProcessedResponse:
        """
        Process Perplexity response for integration.
        
        Args:
            response: Raw Perplexity response dictionary
            
        Returns:
            ProcessedResponse object with standardized format
        """
        try:
            start_time = time.time()
            
            # Extract basic information
            content = response.get("content", "")
            bibliography = response.get("bibliography", "")
            confidence_score = response.get("confidence_score", 0.8)  # Default for Perplexity
            processing_time = response.get("processing_time", 0.0)
            metadata = response.get("metadata", {})
            
            # Extract citations from Perplexity response
            citations = self._extract_perplexity_citations(response)
            
            # Calculate quality score
            quality_score = self._calculate_response_quality(
                content, confidence_score, citations, ResponseSource.PERPLEXITY
            )
            
            # Clean and format content
            content = self._clean_response_content(content)
            
            # Format bibliography
            bibliography = self._format_perplexity_bibliography(bibliography, citations)
            
            processing_time_total = time.time() - start_time + processing_time
            
            processed_response = ProcessedResponse(
                content=content,
                bibliography=bibliography,
                confidence_score=confidence_score,
                source=ResponseSource.PERPLEXITY,
                processing_time=processing_time_total,
                metadata={
                    **metadata,
                    "original_source": "perplexity",
                    "citations_count": len(citations),
                    "processing_stage": "processed"
                },
                quality_score=quality_score,
                citations=citations
            )
            
            self.logger.info(f"Perplexity response processed: quality={quality_score:.2f}, confidence={confidence_score:.2f}")
            return processed_response
            
        except Exception as e:
            self.logger.error(f"Error processing Perplexity response: {str(e)}", exc_info=True)
            
            # Return a minimal processed response on error
            error_content = response.get("content", "")
            if not error_content:
                error_content = "Error processing response"
            
            return ProcessedResponse(
                content=error_content,
                bibliography="",
                confidence_score=0.0,
                source=ResponseSource.PERPLEXITY,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)},
                quality_score=0.0,
                citations=[]
            )
    
    async def combine_responses(
        self, 
        lightrag_resp: Optional[ProcessedResponse], 
        perplexity_resp: Optional[ProcessedResponse],
        combination_strategy: str = "auto"
    ) -> CombinedResponse:
        """
        Combine responses from LightRAG and Perplexity intelligently.
        
        Args:
            lightrag_resp: Processed LightRAG response (optional)
            perplexity_resp: Processed Perplexity response (optional)
            combination_strategy: Strategy for combining responses
            
        Returns:
            CombinedResponse with integrated content
        """
        try:
            start_time = time.time()
            
            # Handle cases where one or both responses are missing
            if not lightrag_resp and not perplexity_resp:
                return self._create_error_response("No responses available to combine")
            
            if not lightrag_resp:
                return self._convert_to_combined_response(perplexity_resp, "perplexity_only")
            
            if not perplexity_resp:
                return self._convert_to_combined_response(lightrag_resp, "lightrag_only")
            
            # Determine combination strategy
            if combination_strategy == "auto":
                combination_strategy = self._determine_combination_strategy(lightrag_resp, perplexity_resp)
            
            # Combine responses based on strategy
            if combination_strategy == "lightrag_primary":
                combined = await self._combine_lightrag_primary(lightrag_resp, perplexity_resp)
            elif combination_strategy == "perplexity_primary":
                combined = await self._combine_perplexity_primary(lightrag_resp, perplexity_resp)
            elif combination_strategy == "parallel_sections":
                combined = await self._combine_parallel_sections(lightrag_resp, perplexity_resp)
            elif combination_strategy == "quality_selection":
                combined = await self._combine_quality_selection(lightrag_resp, perplexity_resp)
            else:
                # Default to quality selection
                combined = await self._combine_quality_selection(lightrag_resp, perplexity_resp)
            
            processing_time = time.time() - start_time
            combined.processing_time += processing_time
            
            self.logger.info(f"Responses combined using strategy: {combination_strategy}")
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining responses: {str(e)}", exc_info=True)
            return self._create_error_response(f"Error combining responses: {str(e)}")
    
    async def select_best_response(
        self, 
        responses: List[ProcessedResponse]
    ) -> ProcessedResponse:
        """
        Select the best response from a list of processed responses.
        
        Args:
            responses: List of processed responses
            
        Returns:
            Best ProcessedResponse based on quality assessment
        """
        if not responses:
            raise ValueError("No responses provided for selection")
        
        if len(responses) == 1:
            return responses[0]
        
        # Score each response
        scored_responses = []
        for response in responses:
            score = self._calculate_selection_score(response)
            scored_responses.append((score, response))
        
        # Sort by score (highest first)
        scored_responses.sort(key=lambda x: x[0], reverse=True)
        
        best_response = scored_responses[0][1]
        self.logger.info(f"Selected best response from {len(responses)} options: {best_response.source.value}")
        
        return best_response
    
    def _extract_lightrag_citations(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations from LightRAG response."""
        citations = []
        
        # Extract from source documents
        source_documents = response.get("source_documents", [])
        for i, doc in enumerate(source_documents, 1):
            citations.append({
                "id": str(i),
                "source": doc,
                "type": "document",
                "confidence": response.get("confidence_score", 0.0),
                "url": None
            })
        
        # Extract from entities used
        entities_used = response.get("entities_used", [])
        for entity in entities_used:
            if entity.get("source_documents"):
                for doc in entity["source_documents"]:
                    if not any(c["source"] == doc for c in citations):
                        citations.append({
                            "id": str(len(citations) + 1),
                            "source": doc,
                            "type": "entity_source",
                            "confidence": entity.get("relevance_score", 0.0),
                            "url": None
                        })
        
        return citations
    
    def _extract_perplexity_citations(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations from Perplexity response."""
        citations = []
        
        # Extract from bibliography
        bibliography = response.get("bibliography", "")
        citation_pattern = r'\[(\d+)\]:\s*([^\n]+?)(?:\s+\(Confidence:\s*([0-9.]+)\))?'
        matches = re.findall(citation_pattern, bibliography)
        
        for match in matches:
            citation_id, source, confidence = match
            confidence_score = float(confidence) if confidence else 0.8
            
            # Check if source is a URL
            url = source if source.startswith(('http://', 'https://')) else None
            
            citations.append({
                "id": citation_id,
                "source": source,
                "type": "web_source",
                "confidence": confidence_score,
                "url": url
            })
        
        return citations
    
    def _calculate_response_quality(
        self, 
        content: str, 
        confidence_score: float, 
        citations: List[Dict[str, Any]], 
        source: ResponseSource
    ) -> float:
        """Calculate quality score for a response."""
        quality_factors = {
            "content_length": 0.0,
            "confidence": 0.0,
            "citations": 0.0,
            "source_bonus": 0.0,
            "coherence": 0.0
        }
        
        # Content length factor (optimal range: 100-800 characters)
        content_length = len(content.strip())
        if content_length < self.min_response_length:
            quality_factors["content_length"] = content_length / self.min_response_length * 0.5
        elif content_length > self.max_response_length:
            quality_factors["content_length"] = 0.8  # Penalize very long responses
        else:
            # Optimal range
            optimal_length = 400
            distance_from_optimal = abs(content_length - optimal_length) / optimal_length
            quality_factors["content_length"] = max(0.5, 1.0 - distance_from_optimal)
        
        # Confidence factor
        quality_factors["confidence"] = confidence_score
        
        # Citations factor
        citation_count = len(citations)
        if citation_count == 0:
            quality_factors["citations"] = 0.0
        elif citation_count <= 3:
            quality_factors["citations"] = citation_count / 3.0 * 0.8
        else:
            quality_factors["citations"] = 0.8  # Cap at 0.8 for many citations
        
        # Source-specific bonus
        if source == ResponseSource.LIGHTRAG:
            quality_factors["source_bonus"] = 0.1  # Slight bonus for knowledge base
        elif source == ResponseSource.PERPLEXITY:
            quality_factors["source_bonus"] = 0.05  # Smaller bonus for real-time
        
        # Coherence factor (simple heuristic based on sentence structure)
        sentences = content.split('.')
        if len(sentences) > 1:
            avg_sentence_length = sum(len(s.strip().split()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
            if 10 <= avg_sentence_length <= 25:  # Optimal sentence length
                quality_factors["coherence"] = 0.2
            else:
                quality_factors["coherence"] = 0.1
        
        # Calculate weighted quality score
        weights = {
            "content_length": 0.25,
            "confidence": 0.35,
            "citations": 0.25,
            "source_bonus": 0.05,
            "coherence": 0.10
        }
        
        quality_score = sum(quality_factors[factor] * weights[factor] for factor in quality_factors)
        return min(quality_score, 1.0)  # Cap at 1.0
    
    def _clean_response_content(self, content: str) -> str:
        """Clean and format response content."""
        if not content:
            return ""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Remove confidence score annotations that might be left over
        content = re.sub(r'\(\s*confidence\s*score:\s*[0-9.]+\s*\)', '', content, flags=re.IGNORECASE)
        
        # Clean up citation formatting
        content = re.sub(r'\s+\[', ' [', content)  # Ensure space before citations
        content = re.sub(r'\]\s*\[', '] [', content)  # Ensure space between citations
        
        return content.strip()
    
    def _format_lightrag_bibliography(self, bibliography: str, citations: List[Dict[str, Any]]) -> str:
        """Format bibliography for LightRAG responses."""
        if not citations:
            return bibliography
        
        # If bibliography is already formatted, return as is
        if bibliography and "**" in bibliography:
            return bibliography
        
        # Create formatted bibliography
        formatted_bib = "\n\n**Knowledge Base Sources:**\n"
        
        high_confidence_citations = [c for c in citations if c["confidence"] >= self.high_confidence_threshold]
        medium_confidence_citations = [c for c in citations if self.medium_confidence_threshold <= c["confidence"] < self.high_confidence_threshold]
        low_confidence_citations = [c for c in citations if c["confidence"] < self.medium_confidence_threshold]
        
        if high_confidence_citations:
            formatted_bib += "**Primary References:**\n"
            for citation in high_confidence_citations:
                formatted_bib += f"[{citation['id']}]: {citation['source']}\n"
                formatted_bib += f"      (Confidence: {citation['confidence']:.2f})\n"
        
        if medium_confidence_citations:
            formatted_bib += "\n**Supporting References:**\n"
            for citation in medium_confidence_citations:
                formatted_bib += f"[{citation['id']}]: {citation['source']}\n"
                formatted_bib += f"      (Confidence: {citation['confidence']:.2f})\n"
        
        if low_confidence_citations:
            formatted_bib += "\n**Additional Reading:**\n"
            for citation in low_confidence_citations:
                formatted_bib += f"[{citation['id']}]: {citation['source']}\n"
        
        return formatted_bib
    
    def _format_perplexity_bibliography(self, bibliography: str, citations: List[Dict[str, Any]]) -> str:
        """Format bibliography for Perplexity responses."""
        if not citations:
            return bibliography
        
        # If bibliography is already well-formatted, return as is
        if bibliography and "**" in bibliography:
            return bibliography
        
        # Create formatted bibliography
        formatted_bib = "\n\n**Web Sources:**\n"
        
        for citation in citations:
            formatted_bib += f"[{citation['id']}]: {citation['source']}\n"
            if citation.get("confidence"):
                formatted_bib += f"      (Confidence: {citation['confidence']:.2f})\n"
        
        return formatted_bib
    
    def _determine_combination_strategy(
        self, 
        lightrag_resp: ProcessedResponse, 
        perplexity_resp: ProcessedResponse
    ) -> str:
        """Determine the best strategy for combining responses."""
        lightrag_quality = lightrag_resp.quality_score
        perplexity_quality = perplexity_resp.quality_score
        
        quality_diff = abs(lightrag_quality - perplexity_quality)
        
        # If one response is significantly better, use it as primary
        if quality_diff > 0.3:
            if lightrag_quality > perplexity_quality:
                return "lightrag_primary"
            else:
                return "perplexity_primary"
        
        # If both are high quality, use parallel sections
        if lightrag_quality > 0.7 and perplexity_quality > 0.7:
            return "parallel_sections"
        
        # Default to quality selection
        return "quality_selection"
    
    async def _combine_lightrag_primary(
        self, 
        lightrag_resp: ProcessedResponse, 
        perplexity_resp: ProcessedResponse
    ) -> CombinedResponse:
        """Combine responses with LightRAG as primary source."""
        content = lightrag_resp.content
        
        # Add supplementary information from Perplexity if it adds value
        if perplexity_resp.quality_score > 0.5:
            content += f"\n\n**Additional Context:**\n{perplexity_resp.content}"
        
        # Combine bibliographies
        bibliography = lightrag_resp.bibliography
        if perplexity_resp.bibliography:
            bibliography += f"\n{perplexity_resp.bibliography}"
        
        # Combine citations
        all_citations = lightrag_resp.citations + perplexity_resp.citations
        
        # Calculate combined confidence
        combined_confidence = (lightrag_resp.confidence_score * 0.7 + 
                             perplexity_resp.confidence_score * 0.3)
        
        return CombinedResponse(
            content=content,
            bibliography=bibliography,
            confidence_score=combined_confidence,
            processing_time=lightrag_resp.processing_time + perplexity_resp.processing_time,
            sources_used=[ResponseSource.LIGHTRAG, ResponseSource.PERPLEXITY],
            combination_strategy="lightrag_primary",
            metadata={
                "primary_source": "lightrag",
                "lightrag_quality": lightrag_resp.quality_score,
                "perplexity_quality": perplexity_resp.quality_score
            },
            quality_assessment={
                "combined_quality": (lightrag_resp.quality_score * 0.7 + 
                                   perplexity_resp.quality_score * 0.3),
                "strategy_reason": "LightRAG had higher quality score"
            },
            citations=all_citations
        )
    
    async def _combine_perplexity_primary(
        self, 
        lightrag_resp: ProcessedResponse, 
        perplexity_resp: ProcessedResponse
    ) -> CombinedResponse:
        """Combine responses with Perplexity as primary source."""
        content = perplexity_resp.content
        
        # Add supplementary information from LightRAG if it adds value
        if lightrag_resp.quality_score > 0.5:
            content += f"\n\n**Knowledge Base Context:**\n{lightrag_resp.content}"
        
        # Combine bibliographies
        bibliography = perplexity_resp.bibliography
        if lightrag_resp.bibliography:
            bibliography += f"\n{lightrag_resp.bibliography}"
        
        # Combine citations
        all_citations = perplexity_resp.citations + lightrag_resp.citations
        
        # Calculate combined confidence
        combined_confidence = (perplexity_resp.confidence_score * 0.7 + 
                             lightrag_resp.confidence_score * 0.3)
        
        return CombinedResponse(
            content=content,
            bibliography=bibliography,
            confidence_score=combined_confidence,
            processing_time=lightrag_resp.processing_time + perplexity_resp.processing_time,
            sources_used=[ResponseSource.PERPLEXITY, ResponseSource.LIGHTRAG],
            combination_strategy="perplexity_primary",
            metadata={
                "primary_source": "perplexity",
                "lightrag_quality": lightrag_resp.quality_score,
                "perplexity_quality": perplexity_resp.quality_score
            },
            quality_assessment={
                "combined_quality": (perplexity_resp.quality_score * 0.7 + 
                                   lightrag_resp.quality_score * 0.3),
                "strategy_reason": "Perplexity had higher quality score"
            },
            citations=all_citations
        )
    
    async def _combine_parallel_sections(
        self, 
        lightrag_resp: ProcessedResponse, 
        perplexity_resp: ProcessedResponse
    ) -> CombinedResponse:
        """Combine responses in parallel sections."""
        content = (
            f"**Knowledge Base Response:**\n{lightrag_resp.content}\n\n"
            f"**Current Information:**\n{perplexity_resp.content}"
        )
        
        # Combine bibliographies
        bibliography = lightrag_resp.bibliography
        if perplexity_resp.bibliography:
            bibliography += f"\n{perplexity_resp.bibliography}"
        
        # Combine citations
        all_citations = lightrag_resp.citations + perplexity_resp.citations
        
        # Calculate combined confidence (average)
        combined_confidence = (lightrag_resp.confidence_score + perplexity_resp.confidence_score) / 2
        
        return CombinedResponse(
            content=content,
            bibliography=bibliography,
            confidence_score=combined_confidence,
            processing_time=lightrag_resp.processing_time + perplexity_resp.processing_time,
            sources_used=[ResponseSource.LIGHTRAG, ResponseSource.PERPLEXITY],
            combination_strategy="parallel_sections",
            metadata={
                "lightrag_quality": lightrag_resp.quality_score,
                "perplexity_quality": perplexity_resp.quality_score
            },
            quality_assessment={
                "combined_quality": (lightrag_resp.quality_score + perplexity_resp.quality_score) / 2,
                "strategy_reason": "Both responses had high quality"
            },
            citations=all_citations
        )
    
    async def _combine_quality_selection(
        self, 
        lightrag_resp: ProcessedResponse, 
        perplexity_resp: ProcessedResponse
    ) -> CombinedResponse:
        """Select the higher quality response and add context from the other."""
        if lightrag_resp.quality_score >= perplexity_resp.quality_score:
            primary_resp = lightrag_resp
            secondary_resp = perplexity_resp
            primary_source = "lightrag"
        else:
            primary_resp = perplexity_resp
            secondary_resp = lightrag_resp
            primary_source = "perplexity"
        
        content = primary_resp.content
        
        # Add context from secondary response if it's decent quality
        if secondary_resp.quality_score > 0.4:
            content += f"\n\n**Additional Context:**\n{secondary_resp.content[:200]}..."
        
        # Combine bibliographies
        bibliography = primary_resp.bibliography
        if secondary_resp.bibliography:
            bibliography += f"\n{secondary_resp.bibliography}"
        
        # Combine citations
        all_citations = primary_resp.citations + secondary_resp.citations
        
        return CombinedResponse(
            content=content,
            bibliography=bibliography,
            confidence_score=primary_resp.confidence_score,
            processing_time=lightrag_resp.processing_time + perplexity_resp.processing_time,
            sources_used=[primary_resp.source, secondary_resp.source],
            combination_strategy="quality_selection",
            metadata={
                "primary_source": primary_source,
                "lightrag_quality": lightrag_resp.quality_score,
                "perplexity_quality": perplexity_resp.quality_score
            },
            quality_assessment={
                "combined_quality": primary_resp.quality_score,
                "strategy_reason": f"{primary_source} had higher quality score"
            },
            citations=all_citations
        )
    
    def _convert_to_combined_response(
        self, 
        response: ProcessedResponse, 
        strategy: str
    ) -> CombinedResponse:
        """Convert a single ProcessedResponse to CombinedResponse."""
        return CombinedResponse(
            content=response.content,
            bibliography=response.bibliography,
            confidence_score=response.confidence_score,
            processing_time=response.processing_time,
            sources_used=[response.source],
            combination_strategy=strategy,
            metadata=response.metadata,
            quality_assessment={
                "combined_quality": response.quality_score,
                "strategy_reason": f"Only {response.source.value} response available"
            },
            citations=response.citations
        )
    
    def _create_error_response(self, error_message: str) -> CombinedResponse:
        """Create an error response."""
        return CombinedResponse(
            content=f"I apologize, but I encountered an issue: {error_message}",
            bibliography="",
            confidence_score=0.0,
            processing_time=0.0,
            sources_used=[],
            combination_strategy="error",
            metadata={"error": error_message},
            quality_assessment={
                "combined_quality": 0.0,
                "strategy_reason": "Error occurred during processing"
            },
            citations=[]
        )
    
    def _calculate_selection_score(self, response: ProcessedResponse) -> float:
        """Calculate selection score for response ranking."""
        # Weighted combination of quality and confidence
        return (response.quality_score * 0.6 + response.confidence_score * 0.4)