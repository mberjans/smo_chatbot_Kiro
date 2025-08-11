"""
Query router for intelligent routing between LightRAG and Perplexity API.

This module implements the QueryRouter class that uses query classification
to intelligently route queries to the most appropriate response system,
with comprehensive fallback mechanisms and response combination capabilities.
"""

import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union, Tuple
from dataclasses import dataclass

from .classifier import QueryClassifier, QueryClassification, QueryType
from ..config.settings import LightRAGConfig
from ..utils.logging import setup_logger


class RoutingStrategy(Enum):
    """Routing strategies for query processing."""
    LIGHTRAG_ONLY = "lightrag_only"
    PERPLEXITY_ONLY = "perplexity_only"
    LIGHTRAG_FIRST = "lightrag_first"
    PERPLEXITY_FIRST = "perplexity_first"
    PARALLEL = "parallel"
    HYBRID = "hybrid"


@dataclass
class RoutingDecision:
    """Decision made by the router for query processing."""
    strategy: RoutingStrategy
    primary_source: str
    fallback_sources: List[str]
    confidence_score: float
    reasoning: str
    classification: Optional[QueryClassification]
    metadata: Dict[str, Any]


@dataclass
class RoutedResponse:
    """Response from the routing system."""
    content: str
    bibliography: str
    confidence_score: float
    processing_time: float
    source: str
    sources_used: List[str]
    routing_decision: RoutingDecision
    metadata: Dict[str, Any]
    errors: List[str]


class QueryRouter:
    """
    Intelligent query router for LightRAG integration.
    
    This router analyzes queries using the QueryClassifier and routes them
    to the most appropriate system (LightRAG or Perplexity API) based on
    the classification results and configured strategies.
    """
    
    def __init__(
        self,
        classifier: QueryClassifier,
        lightrag_query_func: Optional[Callable[[str], Awaitable[Dict[str, Any]]]] = None,
        perplexity_query_func: Optional[Callable[[str], Awaitable[Dict[str, Any]]]] = None,
        config: Optional[LightRAGConfig] = None
    ):
        """
        Initialize the query router.
        
        Args:
            classifier: Query classifier for routing decisions
            lightrag_query_func: Function to query LightRAG system
            perplexity_query_func: Function to query Perplexity API
            config: Optional configuration object
        """
        self.classifier = classifier
        self.lightrag_query_func = lightrag_query_func
        self.perplexity_query_func = perplexity_query_func
        self.config = config or LightRAGConfig.from_env()
        
        # Set up logging
        self.logger = setup_logger(
            name="query_router",
            log_file=f"{self.config.cache_directory}/router.log"
        )
        
        # Default routing configuration
        self.routing_config = {
            "confidence_thresholds": {
                "high_confidence": 0.8,
                "medium_confidence": 0.6,
                "low_confidence": 0.4
            },
            "timeout_seconds": {
                "lightrag": 30.0,
                "perplexity": 15.0,
                "parallel": 45.0
            },
            "retry_attempts": {
                "lightrag": 2,
                "perplexity": 3,
                "classification": 3
            },
            "enable_parallel_queries": True,
            "enable_response_combination": True
        }
        
        # Statistics tracking
        self._stats = {
            "total_queries": 0,
            "successful_routes": 0,
            "failed_routes": 0,
            "lightrag_queries": 0,
            "perplexity_queries": 0,
            "hybrid_queries": 0,
            "fallback_activations": 0,
            "classification_failures": 0,
            "average_response_time": 0.0,
            "strategy_counts": {strategy.value: 0 for strategy in RoutingStrategy},
            "error_counts": {
                "lightrag_errors": 0,
                "perplexity_errors": 0,
                "classification_errors": 0,
                "timeout_errors": 0
            }
        }
        
        self.logger.info("Query router initialized with classifier: %s", type(classifier).__name__)
    
    async def route_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        force_strategy: Optional[RoutingStrategy] = None
    ) -> RoutedResponse:
        """
        Route a query to the appropriate system(s).
        
        Args:
            query: The user query to route
            context: Optional context information
            force_strategy: Optional strategy to force (bypasses classification)
            
        Returns:
            RoutedResponse with the result and routing information
            
        Raises:
            ValueError: If query is empty
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        start_time = datetime.now()
        errors = []
        
        try:
            self._stats["total_queries"] += 1
            self.logger.debug("Routing query: %s", query[:100] + "..." if len(query) > 100 else query)
            
            # Get routing decision
            if force_strategy:
                routing_decision = self._create_forced_routing_decision(force_strategy, query)
            else:
                routing_decision = await self._make_routing_decision(query, context)
            
            # Execute routing strategy
            response = await self._execute_routing_strategy(query, routing_decision)
            
            # Update statistics
            self._update_routing_stats(routing_decision, response, start_time)
            
            return response
            
        except Exception as e:
            self._stats["failed_routes"] += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error("Query routing failed: %s", str(e))
            
            return self._create_error_response(
                query, 
                [f"Routing failed: {str(e)}"], 
                processing_time
            )
    
    async def batch_route_queries(
        self,
        queries: List[str],
        contexts: Optional[List[Dict[str, Any]]] = None,
        strategies: Optional[List[RoutingStrategy]] = None
    ) -> List[RoutedResponse]:
        """
        Route multiple queries in batch.
        
        Args:
            queries: List of queries to route
            contexts: Optional list of context dictionaries
            strategies: Optional list of forced strategies
            
        Returns:
            List of RoutedResponse objects
            
        Raises:
            ValueError: If input lists have mismatched lengths
        """
        if not queries:
            return []
        
        # Validate input lengths
        if contexts is not None and len(contexts) != len(queries):
            raise ValueError("Contexts list must match queries list length")
        if strategies is not None and len(strategies) != len(queries):
            raise ValueError("Strategies list must match queries list length")
        
        # Prepare arguments
        if contexts is None:
            contexts = [None] * len(queries)
        if strategies is None:
            strategies = [None] * len(queries)
        
        self.logger.info("Starting batch routing of %d queries", len(queries))
        
        # Process queries concurrently with semaphore
        semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        
        async def route_with_semaphore(query: str, context: Optional[Dict[str, Any]], strategy: Optional[RoutingStrategy]):
            async with semaphore:
                return await self.route_query(query, context, strategy)
        
        tasks = [
            route_with_semaphore(query, context, strategy)
            for query, context, strategy in zip(queries, contexts, strategies)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error("Batch routing failed for query %d: %s", i, str(result))
                error_response = self._create_error_response(
                    queries[i],
                    [f"Batch processing error: {str(result)}"],
                    0.0
                )
                responses.append(error_response)
            else:
                responses.append(result)
        
        successful_count = len([r for r in results if not isinstance(r, Exception)])
        failed_count = len(results) - successful_count
        
        self.logger.info("Batch routing completed: %d successful, %d failed", successful_count, failed_count)
        
        return responses
    
    async def _make_routing_decision(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> RoutingDecision:
        """
        Make a routing decision based on query classification.
        
        Args:
            query: The query to classify
            context: Optional context information
            
        Returns:
            RoutingDecision object
        """
        # Classify query with retry logic
        classification = None
        classification_error = None
        
        for attempt in range(self.routing_config["retry_attempts"]["classification"]):
            try:
                classification = await self.classifier.classify_query(query, context)
                break
            except Exception as e:
                classification_error = str(e)
                self.logger.warning("Classification attempt %d failed: %s", attempt + 1, str(e))
                if attempt < self.routing_config["retry_attempts"]["classification"] - 1:
                    await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
        
        if classification is None:
            self._stats["classification_failures"] += 1
            # Raise exception to trigger error response
            raise RuntimeError(f"Classification failed after all retries: {classification_error}")
        
        # Determine routing strategy based on classification
        strategy = self._determine_routing_strategy(classification)
        primary_source = self._get_primary_source(strategy)
        fallback_sources = self._get_fallback_sources(strategy)
        
        return RoutingDecision(
            strategy=strategy,
            primary_source=primary_source,
            fallback_sources=fallback_sources,
            confidence_score=classification.confidence_score,
            reasoning=f"Classified as {classification.query_type.value} with {classification.confidence_score:.2f} confidence",
            classification=classification,
            metadata={
                "query_type": classification.query_type.value,
                "suggested_sources": classification.suggested_sources,
                "keywords": classification.metadata.get("keywords", [])
            }
        )
    
    def _create_forced_routing_decision(
        self, 
        strategy: RoutingStrategy, 
        query: str
    ) -> RoutingDecision:
        """
        Create a routing decision for a forced strategy.
        
        Args:
            strategy: The forced routing strategy
            query: The original query
            
        Returns:
            RoutingDecision object
        """
        return RoutingDecision(
            strategy=strategy,
            primary_source=self._get_primary_source(strategy),
            fallback_sources=self._get_fallback_sources(strategy),
            confidence_score=1.0,  # Forced strategies have full confidence
            reasoning=f"Forced routing strategy: {strategy.value}",
            classification=None,
            metadata={"forced": True, "query_length": len(query)}
        )
    
    def _determine_routing_strategy(self, classification: QueryClassification) -> RoutingStrategy:
        """
        Determine routing strategy based on classification.
        
        Args:
            classification: Query classification result
            
        Returns:
            Appropriate routing strategy
        """
        confidence = classification.confidence_score
        query_type = classification.query_type
        
        high_confidence = confidence >= self.routing_config["confidence_thresholds"]["high_confidence"]
        
        if query_type == QueryType.KNOWLEDGE_BASE:
            return RoutingStrategy.LIGHTRAG_ONLY if high_confidence else RoutingStrategy.LIGHTRAG_FIRST
        elif query_type == QueryType.REAL_TIME:
            return RoutingStrategy.PERPLEXITY_ONLY if high_confidence else RoutingStrategy.PERPLEXITY_FIRST
        elif query_type == QueryType.HYBRID:
            if self.routing_config["enable_parallel_queries"]:
                return RoutingStrategy.PARALLEL
            else:
                return RoutingStrategy.LIGHTRAG_FIRST
        else:
            # Fallback for unknown query types
            return RoutingStrategy.LIGHTRAG_FIRST
    
    def _get_primary_source(self, strategy: RoutingStrategy) -> str:
        """Get the primary source for a routing strategy."""
        strategy_mapping = {
            RoutingStrategy.LIGHTRAG_ONLY: "lightrag",
            RoutingStrategy.PERPLEXITY_ONLY: "perplexity",
            RoutingStrategy.LIGHTRAG_FIRST: "lightrag",
            RoutingStrategy.PERPLEXITY_FIRST: "perplexity",
            RoutingStrategy.PARALLEL: "both",
            RoutingStrategy.HYBRID: "both"
        }
        return strategy_mapping.get(strategy, "lightrag")
    
    def _get_fallback_sources(self, strategy: RoutingStrategy) -> List[str]:
        """Get the fallback sources for a routing strategy."""
        fallback_mapping = {
            RoutingStrategy.LIGHTRAG_ONLY: ["perplexity"],
            RoutingStrategy.PERPLEXITY_ONLY: ["lightrag"],
            RoutingStrategy.LIGHTRAG_FIRST: ["perplexity"],
            RoutingStrategy.PERPLEXITY_FIRST: ["lightrag"],
            RoutingStrategy.PARALLEL: [],
            RoutingStrategy.HYBRID: []
        }
        return fallback_mapping.get(strategy, [])
    
    async def _execute_routing_strategy(
        self, 
        query: str, 
        routing_decision: RoutingDecision
    ) -> RoutedResponse:
        """
        Execute the routing strategy.
        
        Args:
            query: The query to process
            routing_decision: The routing decision
            
        Returns:
            RoutedResponse object
        """
        strategy = routing_decision.strategy
        errors = []
        
        try:
            if strategy in [RoutingStrategy.PARALLEL, RoutingStrategy.HYBRID]:
                return await self._execute_parallel_strategy(query, routing_decision, errors)
            else:
                return await self._execute_sequential_strategy(query, routing_decision, errors)
                
        except Exception as e:
            errors.append(f"Strategy execution failed: {str(e)}")
            self.logger.error("Strategy execution failed: %s", str(e))
            return self._create_error_response(query, errors, 0.0)
    
    async def _execute_parallel_strategy(
        self, 
        query: str, 
        routing_decision: RoutingDecision, 
        errors: List[str]
    ) -> RoutedResponse:
        """Execute parallel routing strategy."""
        timeout = self.routing_config["timeout_seconds"]["parallel"]
        
        # Create tasks for both systems
        tasks = []
        task_names = []
        
        if self.lightrag_query_func:
            tasks.append(self._query_with_timeout("lightrag", self.lightrag_query_func, query, timeout))
            task_names.append("LightRAG")
        
        if self.perplexity_query_func:
            tasks.append(self._query_with_timeout("perplexity", self.perplexity_query_func, query, timeout))
            task_names.append("Perplexity")
        
        if not tasks:
            errors.append("No query functions available for parallel execution")
            return self._create_error_response(query, errors, 0.0)
        
        # Execute tasks in parallel
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_msg = f"{task_names[i]} failed: {str(result)}"
                    errors.append(error_msg)
                    self.logger.warning(error_msg)
                else:
                    successful_results.append((task_names[i], result))
            
            if successful_results:
                if len(successful_results) == 1:
                    # Only one system succeeded
                    source_name, result_data = successful_results[0]
                    return self._create_single_response(source_name, result_data, routing_decision, errors, partial_parallel=True)
                else:
                    # Multiple systems succeeded, combine responses
                    return self._combine_responses(successful_results, routing_decision, errors)
            else:
                # All systems failed
                errors.append("All parallel queries failed")
                return self._create_error_response(query, errors, 0.0)
                
        except asyncio.TimeoutError:
            errors.append(f"Parallel query timeout after {timeout} seconds")
            self._stats["error_counts"]["timeout_errors"] += 1
            return self._create_error_response(query, errors, timeout)
    
    async def _execute_sequential_strategy(
        self, 
        query: str, 
        routing_decision: RoutingDecision, 
        errors: List[str]
    ) -> RoutedResponse:
        """Execute sequential routing strategy with fallbacks."""
        strategy = routing_decision.strategy
        primary_source = routing_decision.primary_source
        fallback_sources = routing_decision.fallback_sources
        
        # Try primary source first
        primary_result = await self._try_query_source(primary_source, query, errors)
        if primary_result:
            source_name, result_data = primary_result
            return self._create_single_response(source_name, result_data, routing_decision, errors)
        
        # Try fallback sources
        for fallback_source in fallback_sources:
            self._stats["fallback_activations"] += 1
            self.logger.info("Trying fallback source: %s", fallback_source)
            
            fallback_result = await self._try_query_source(fallback_source, query, errors)
            if fallback_result:
                source_name, result_data = fallback_result
                return self._create_single_response(
                    source_name, 
                    result_data, 
                    routing_decision, 
                    errors, 
                    fallback_used=True,
                    primary_failure=f"{primary_source} failed"
                )
        
        # All sources failed
        return self._create_error_response(query, errors, 0.0)
    
    async def _try_query_source(
        self, 
        source: str, 
        query: str, 
        errors: List[str]
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Try to query a specific source.
        
        Args:
            source: Source name ("lightrag" or "perplexity")
            query: The query to execute
            errors: List to append errors to
            
        Returns:
            Tuple of (source_name, result_data) if successful, None if failed
        """
        if source == "lightrag":
            if not self.lightrag_query_func:
                errors.append("LightRAG query function not available")
                return None
            
            timeout = self.routing_config["timeout_seconds"]["lightrag"]
            retry_attempts = self.routing_config["retry_attempts"]["lightrag"]
            
            for attempt in range(retry_attempts):
                try:
                    result = await self._query_with_timeout("lightrag", self.lightrag_query_func, query, timeout)
                    return ("LightRAG", result)
                except Exception as e:
                    error_msg = f"LightRAG attempt {attempt + 1} failed: {str(e)}"
                    self.logger.warning(error_msg)
                    if attempt == retry_attempts - 1:  # Last attempt
                        errors.append(error_msg)
                        self._stats["error_counts"]["lightrag_errors"] += 1
                    else:
                        await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
        
        elif source == "perplexity":
            if not self.perplexity_query_func:
                errors.append("Perplexity query function not available")
                return None
            
            timeout = self.routing_config["timeout_seconds"]["perplexity"]
            retry_attempts = self.routing_config["retry_attempts"]["perplexity"]
            
            for attempt in range(retry_attempts):
                try:
                    result = await self._query_with_timeout("perplexity", self.perplexity_query_func, query, timeout)
                    return ("Perplexity", result)
                except Exception as e:
                    error_msg = f"Perplexity attempt {attempt + 1} failed: {str(e)}"
                    self.logger.warning(error_msg)
                    if attempt == retry_attempts - 1:  # Last attempt
                        errors.append(error_msg)
                        self._stats["error_counts"]["perplexity_errors"] += 1
                    else:
                        await asyncio.sleep(0.5 * (2 ** attempt))  # Exponential backoff
        
        return None
    
    async def _query_with_timeout(
        self, 
        source_name: str, 
        query_func: Callable[[str], Awaitable[Dict[str, Any]]], 
        query: str, 
        timeout: float
    ) -> Dict[str, Any]:
        """
        Execute a query function with timeout.
        
        Args:
            source_name: Name of the source for logging
            query_func: The query function to execute
            query: The query to process
            timeout: Timeout in seconds
            
        Returns:
            Query result dictionary
            
        Raises:
            asyncio.TimeoutError: If query times out
            Exception: If query fails
        """
        try:
            result = await asyncio.wait_for(query_func(query), timeout=timeout)
            self.logger.debug("%s query completed in time", source_name)
            return result
        except asyncio.TimeoutError:
            error_msg = f"{source_name} query timeout after {timeout} seconds"
            self.logger.warning(error_msg)
            self._stats["error_counts"]["timeout_errors"] += 1
            raise asyncio.TimeoutError(error_msg)
        except Exception as e:
            self.logger.error("%s query failed: %s", source_name, str(e))
            raise
    
    def _create_single_response(
        self, 
        source_name: str, 
        result_data: Dict[str, Any], 
        routing_decision: RoutingDecision, 
        errors: List[str],
        fallback_used: bool = False,
        primary_failure: Optional[str] = None,
        partial_parallel: bool = False
    ) -> RoutedResponse:
        """Create a response from a single source."""
        metadata = {
            "single_source": True,
            "fallback_used": fallback_used,
            "partial_parallel": partial_parallel
        }
        
        if primary_failure:
            metadata["primary_failure"] = primary_failure
        
        # Add result metadata
        if "metadata" in result_data:
            metadata.update(result_data["metadata"])
        
        return RoutedResponse(
            content=result_data.get("content", ""),
            bibliography=result_data.get("bibliography", ""),
            confidence_score=result_data.get("confidence_score", 0.0),
            processing_time=result_data.get("processing_time", 0.0),
            source=source_name,
            sources_used=[source_name.lower()],
            routing_decision=routing_decision,
            metadata=metadata,
            errors=errors
        )
    
    def _combine_responses(
        self, 
        results: List[Tuple[str, Dict[str, Any]]], 
        routing_decision: RoutingDecision, 
        errors: List[str]
    ) -> RoutedResponse:
        """
        Combine multiple responses into a single response.
        
        Args:
            results: List of (source_name, result_data) tuples
            routing_decision: The routing decision
            errors: List of errors encountered
            
        Returns:
            Combined RoutedResponse
        """
        if not results:
            return self._create_error_response("", ["No results to combine"], 0.0)
        
        # Combine content
        content_parts = []
        bibliography_parts = []
        confidence_scores = []
        processing_times = []
        sources_used = []
        combined_metadata = {"combined_response": True, "sources_combined": len(results)}
        
        for source_name, result_data in results:
            # Add content section
            content_parts.append(f"## {source_name} Response:\n{result_data.get('content', '')}")
            
            # Add bibliography section
            if result_data.get("bibliography"):
                bibliography_parts.append(f"### {source_name} Sources:\n{result_data['bibliography']}")
            
            # Collect metrics
            confidence_scores.append(result_data.get("confidence_score", 0.0))
            processing_times.append(result_data.get("processing_time", 0.0))
            sources_used.append(source_name.lower())
            
            # Merge metadata
            if "metadata" in result_data:
                for key, value in result_data["metadata"].items():
                    combined_key = f"{source_name.lower()}_{key}"
                    combined_metadata[combined_key] = value
        
        # Calculate combined metrics
        avg_confidence = round(sum(confidence_scores) / len(confidence_scores), 2) if confidence_scores else 0.0
        total_processing_time = sum(processing_times)
        
        # Create combined content
        combined_content = "\n\n".join(content_parts)
        combined_bibliography = "\n\n".join(bibliography_parts) if bibliography_parts else ""
        
        return RoutedResponse(
            content=combined_content,
            bibliography=combined_bibliography,
            confidence_score=avg_confidence,
            processing_time=total_processing_time,
            source="Combined",
            sources_used=sources_used,
            routing_decision=routing_decision,
            metadata=combined_metadata,
            errors=errors
        )
    
    def _create_error_response(
        self, 
        query: str, 
        errors: List[str], 
        processing_time: float
    ) -> RoutedResponse:
        """
        Create an error response.
        
        Args:
            query: The original query
            errors: List of error messages
            processing_time: Time spent processing
            
        Returns:
            Error RoutedResponse
        """
        error_content = (
            "I apologize, but I'm unable to process your query at the moment. "
            "This could be due to temporary service issues or system maintenance. "
            "Please try again in a few moments."
        )
        
        # Create minimal routing decision for error case
        error_routing_decision = RoutingDecision(
            strategy=RoutingStrategy.LIGHTRAG_FIRST,  # Default strategy
            primary_source="none",
            fallback_sources=[],
            confidence_score=0.0,
            reasoning="Error occurred during processing",
            classification=None,
            metadata={"error_response": True}
        )
        
        return RoutedResponse(
            content=error_content,
            bibliography="",
            confidence_score=0.0,
            processing_time=processing_time,
            source="Error",
            sources_used=[],
            routing_decision=error_routing_decision,
            metadata={
                "error_response": True,
                "query_length": len(query),
                "error_count": len(errors)
            },
            errors=errors
        )
    
    def _update_routing_stats(
        self, 
        routing_decision: RoutingDecision, 
        response: RoutedResponse, 
        start_time: datetime
    ) -> None:
        """Update routing statistics."""
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Update success/failure counts
        if response.source != "Error":
            self._stats["successful_routes"] += 1
        else:
            self._stats["failed_routes"] += 1
        
        # Update source counts
        if "lightrag" in response.sources_used:
            self._stats["lightrag_queries"] += 1
        if "perplexity" in response.sources_used:
            self._stats["perplexity_queries"] += 1
        if len(response.sources_used) > 1:
            self._stats["hybrid_queries"] += 1
        
        # Update strategy counts
        strategy_key = routing_decision.strategy.value
        self._stats["strategy_counts"][strategy_key] += 1
        
        # Update average response time
        total_time = (self._stats["average_response_time"] * 
                     (self._stats["total_queries"] - 1) + processing_time)
        self._stats["average_response_time"] = total_time / self._stats["total_queries"]
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive routing metrics.
        
        Returns:
            Dictionary with routing statistics and performance metrics
        """
        total_queries = self._stats["total_queries"]
        successful_routes = self._stats["successful_routes"]
        
        # Calculate derived metrics
        success_rate = successful_routes / total_queries if total_queries > 0 else 0.0
        routing_accuracy = success_rate  # Simplified metric
        
        # Get classifier stats
        classifier_stats = {}
        if hasattr(self.classifier, 'get_classification_stats'):
            classifier_stats = self.classifier.get_classification_stats()
        
        return {
            # Basic counts
            "total_queries": total_queries,
            "successful_routes": successful_routes,
            "failed_routes": self._stats["failed_routes"],
            
            # Source distribution
            "lightrag_queries": self._stats["lightrag_queries"],
            "perplexity_queries": self._stats["perplexity_queries"],
            "hybrid_queries": self._stats["hybrid_queries"],
            
            # Performance metrics
            "average_response_time": self._stats["average_response_time"],
            "success_rate": success_rate,
            "routing_accuracy": routing_accuracy,
            
            # Error metrics
            "fallback_activations": self._stats["fallback_activations"],
            "classification_failures": self._stats["classification_failures"],
            "error_rates": {
                "total_error_rate": self._stats["failed_routes"] / total_queries if total_queries > 0 else 0.0,
                **{k: v / total_queries if total_queries > 0 else 0.0 
                   for k, v in self._stats["error_counts"].items()}
            },
            
            # Strategy distribution
            "strategy_distribution": {
                k: v / total_queries if total_queries > 0 else 0.0 
                for k, v in self._stats["strategy_counts"].items()
            },
            
            # Configuration
            "routing_config": self.routing_config,
            
            # Classifier metrics
            "classifier_stats": classifier_stats
        }
    
    def update_routing_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update routing configuration.
        
        Args:
            config_updates: Dictionary with configuration updates
        """
        def deep_update(base_dict: Dict[str, Any], updates: Dict[str, Any]) -> None:
            """Recursively update nested dictionaries."""
            for key, value in updates.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.routing_config, config_updates)
        self.logger.info("Updated routing configuration: %s", config_updates)
        
        # Update classifier thresholds if provided
        if "confidence_thresholds" in config_updates:
            if hasattr(self.classifier, 'update_confidence_thresholds'):
                self.classifier.update_confidence_thresholds(config_updates["confidence_thresholds"])