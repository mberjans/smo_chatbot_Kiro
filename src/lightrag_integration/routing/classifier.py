"""
LLM-based query classifier for intelligent routing.

This module implements query classification logic to determine whether
queries should be routed to LightRAG (knowledge base) or Perplexity API
(real-time information).
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.prompts import PromptTemplate

from ..config.settings import LightRAGConfig
from ..utils.logging import setup_logger


class QueryType(Enum):
    """Types of queries for routing decisions."""
    KNOWLEDGE_BASE = "knowledge_base"
    REAL_TIME = "real_time"
    HYBRID = "hybrid"


@dataclass
class QueryClassification:
    """Result of query classification."""
    query_type: QueryType
    confidence_score: float
    reasoning: str
    suggested_sources: List[str]
    metadata: Dict[str, Any]
    processing_time: float


class QueryClassifier:
    """
    LLM-based query classifier for routing decisions.
    
    This classifier analyzes incoming queries and determines whether they
    should be routed to LightRAG (for established knowledge) or Perplexity API
    (for real-time/recent information).
    """
    
    # Classification prompt template
    CLASSIFICATION_PROMPT = PromptTemplate(
        """You are an intelligent query classifier for a Clinical Metabolomics Oracle system.

Your task is to analyze user queries and classify them into one of three categories:

1. KNOWLEDGE_BASE: Queries about established scientific knowledge, definitions, mechanisms, 
   or concepts that would be found in research papers and textbooks. These include:
   - "What is clinical metabolomics?"
   - "How does mass spectrometry work?"
   - "What are the main metabolic pathways?"
   - "Define biomarker discovery"
   - "Explain metabolite identification methods"

2. REAL_TIME: Queries requiring current, recent, or time-sensitive information that 
   changes frequently. These include:
   - "Latest research on metabolomics in 2024"
   - "Recent clinical trials for metabolomics"
   - "Current market trends in metabolomics"
   - "News about metabolomics companies"
   - "What happened at the recent metabolomics conference?"

3. HYBRID: Complex queries that require both established knowledge AND recent information:
   - "How has metabolomics evolved in recent years?"
   - "Compare traditional methods with recent advances"
   - "What are the current challenges in clinical metabolomics?"

Analyze the following query and provide your classification:

Query: "{query}"

Context (if provided): {context}

Respond in the following JSON format:
{{
    "query_type": "KNOWLEDGE_BASE|REAL_TIME|HYBRID",
    "confidence_score": 0.0-1.0,
    "reasoning": "Brief explanation of your classification decision",
    "suggested_sources": ["list", "of", "relevant", "source", "types"],
    "keywords": ["key", "terms", "from", "query"],
    "temporal_indicators": ["any", "time-related", "terms"],
    "domain_specificity": 0.0-1.0
}}

Be precise and consider:
- Temporal indicators (recent, latest, current, new, 2024, etc.)
- Knowledge type (definitions vs. current events)
- Query complexity and scope
- Domain specificity to clinical metabolomics"""
    )
    
    def __init__(self, llm: BaseLLM, config: Optional[LightRAGConfig] = None):
        """
        Initialize the query classifier.
        
        Args:
            llm: Language model for classification
            config: Optional configuration object
        """
        self.llm = llm
        self.config = config or LightRAGConfig.from_env()
        
        # Set up logging
        self.logger = setup_logger(
            name="query_classifier",
            log_file=f"{self.config.cache_directory}/classifier.log"
        )
        
        # Classification thresholds
        self.confidence_thresholds = {
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
            "low_confidence": 0.4
        }
        
        # Statistics tracking
        self._stats = {
            "classifications_performed": 0,
            "knowledge_base_queries": 0,
            "real_time_queries": 0,
            "hybrid_queries": 0,
            "high_confidence_classifications": 0,
            "low_confidence_classifications": 0,
            "errors_encountered": 0,
            "average_processing_time": 0.0
        }
        
        self.logger.info("Query classifier initialized with LLM: %s", type(llm).__name__)
    
    async def classify_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> QueryClassification:
        """
        Classify a query to determine routing strategy.
        
        Args:
            query: The user query to classify
            context: Optional context information
            
        Returns:
            QueryClassification object with routing decision
            
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If classification fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = datetime.now()
        
        try:
            self.logger.debug("Classifying query: %s", query[:100] + "..." if len(query) > 100 else query)
            
            # Prepare context string
            context_str = ""
            if context:
                context_items = []
                for key, value in context.items():
                    context_items.append(f"{key}: {value}")
                context_str = "; ".join(context_items)
            
            # Format the prompt
            formatted_prompt = self.CLASSIFICATION_PROMPT.format(
                query=query,
                context=context_str if context_str else "None provided"
            )
            
            # Get LLM response
            response = await self.llm.acomplete(formatted_prompt)
            response_text = response.text.strip()
            
            # Parse the JSON response
            classification_data = self._parse_llm_response(response_text)
            
            # Create classification object
            processing_time = (datetime.now() - start_time).total_seconds()
            
            classification = QueryClassification(
                query_type=QueryType(classification_data["query_type"].lower()),
                confidence_score=float(classification_data["confidence_score"]),
                reasoning=classification_data["reasoning"],
                suggested_sources=classification_data.get("suggested_sources", []),
                metadata={
                    "keywords": classification_data.get("keywords", []),
                    "temporal_indicators": classification_data.get("temporal_indicators", []),
                    "domain_specificity": classification_data.get("domain_specificity", 0.0),
                    "context_provided": context is not None,
                    "query_length": len(query)
                },
                processing_time=processing_time
            )
            
            # Update statistics
            self._update_stats(classification)
            
            self.logger.info(
                "Query classified as %s with confidence %.2f in %.3f seconds",
                classification.query_type.value,
                classification.confidence_score,
                processing_time
            )
            
            return classification
            
        except Exception as e:
            self._stats["errors_encountered"] += 1
            self.logger.error("Query classification failed: %s", str(e))
            
            # Return fallback classification
            processing_time = (datetime.now() - start_time).total_seconds()
            return self._create_fallback_classification(query, processing_time, str(e))
    
    async def batch_classify(
        self, 
        queries: List[str], 
        contexts: Optional[List[Dict[str, Any]]] = None
    ) -> List[QueryClassification]:
        """
        Classify multiple queries in batch.
        
        Args:
            queries: List of queries to classify
            contexts: Optional list of context dictionaries
            
        Returns:
            List of QueryClassification objects
        """
        if not queries:
            return []
        
        # Ensure contexts list matches queries length
        if contexts is None:
            contexts = [None] * len(queries)
        elif len(contexts) != len(queries):
            raise ValueError("Contexts list must match queries list length")
        
        self.logger.info("Starting batch classification of %d queries", len(queries))
        
        # Process queries concurrently with semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def classify_with_semaphore(query: str, context: Optional[Dict[str, Any]]):
            async with semaphore:
                return await self.classify_query(query, context)
        
        tasks = [
            classify_with_semaphore(query, context)
            for query, context in zip(queries, contexts)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions in results
        classifications = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error("Batch classification failed for query %d: %s", i, str(result))
                # Create fallback classification
                fallback = self._create_fallback_classification(
                    queries[i], 0.0, f"Batch processing error: {str(result)}"
                )
                classifications.append(fallback)
            else:
                classifications.append(result)
        
        self.logger.info("Batch classification completed: %d successful, %d failed", 
                        len([r for r in results if not isinstance(r, Exception)]),
                        len([r for r in results if isinstance(r, Exception)]))
        
        return classifications
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics."""
        return {
            **self._stats,
            "confidence_thresholds": self.confidence_thresholds,
            "query_type_distribution": {
                "knowledge_base": self._stats["knowledge_base_queries"],
                "real_time": self._stats["real_time_queries"], 
                "hybrid": self._stats["hybrid_queries"]
            }
        }
    
    def update_confidence_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Update confidence thresholds for classification decisions.
        
        Args:
            thresholds: Dictionary with threshold values
        """
        self.confidence_thresholds.update(thresholds)
        self.logger.info("Updated confidence thresholds: %s", self.confidence_thresholds)
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response JSON.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed classification data
            
        Raises:
            ValueError: If response cannot be parsed
        """
        import json
        import re
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
            else:
                # Fallback: try to parse entire response as JSON
                data = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["query_type", "confidence_score", "reasoning"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate query_type
            valid_types = ["knowledge_base", "real_time", "hybrid"]
            if data["query_type"].lower() not in valid_types:
                raise ValueError(f"Invalid query_type: {data['query_type']}")
            
            # Validate confidence_score
            confidence = float(data["confidence_score"])
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Invalid confidence_score: {confidence}")
            
            return data
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error("Failed to parse LLM response: %s\nResponse: %s", str(e), response_text)
            raise ValueError(f"Invalid LLM response format: {str(e)}")
    
    def _create_fallback_classification(
        self, 
        query: str, 
        processing_time: float, 
        error_reason: str
    ) -> QueryClassification:
        """
        Create a fallback classification when normal classification fails.
        
        Args:
            query: Original query
            processing_time: Time spent processing
            error_reason: Reason for fallback
            
        Returns:
            Fallback QueryClassification
        """
        # Simple heuristic-based fallback classification
        query_lower = query.lower()
        
        # Check for temporal indicators
        temporal_keywords = [
            "recent", "latest", "current", "new", "today", "now", 
            "2024", "2023", "this year", "last year", "trending"
        ]
        
        has_temporal = any(keyword in query_lower for keyword in temporal_keywords)
        
        if has_temporal:
            query_type = QueryType.REAL_TIME
            reasoning = f"Fallback classification: detected temporal indicators. Error: {error_reason}"
        else:
            query_type = QueryType.KNOWLEDGE_BASE
            reasoning = f"Fallback classification: no temporal indicators detected. Error: {error_reason}"
        
        return QueryClassification(
            query_type=query_type,
            confidence_score=0.3,  # Low confidence for fallback
            reasoning=reasoning,
            suggested_sources=["fallback"],
            metadata={
                "fallback_classification": True,
                "error_reason": error_reason,
                "query_length": len(query),
                "has_temporal_indicators": has_temporal
            },
            processing_time=processing_time
        )
    
    def _update_stats(self, classification: QueryClassification) -> None:
        """Update classification statistics."""
        self._stats["classifications_performed"] += 1
        
        # Update query type counts
        if classification.query_type == QueryType.KNOWLEDGE_BASE:
            self._stats["knowledge_base_queries"] += 1
        elif classification.query_type == QueryType.REAL_TIME:
            self._stats["real_time_queries"] += 1
        elif classification.query_type == QueryType.HYBRID:
            self._stats["hybrid_queries"] += 1
        
        # Update confidence stats
        if classification.confidence_score >= self.confidence_thresholds["high_confidence"]:
            self._stats["high_confidence_classifications"] += 1
        elif classification.confidence_score < self.confidence_thresholds["low_confidence"]:
            self._stats["low_confidence_classifications"] += 1
        
        # Update average processing time
        total_time = (self._stats["average_processing_time"] * 
                     (self._stats["classifications_performed"] - 1) + 
                     classification.processing_time)
        self._stats["average_processing_time"] = total_time / self._stats["classifications_performed"]