"""
Demonstration of the QueryRouter functionality.

This script demonstrates the key features of the QueryRouter without requiring
all external dependencies.
"""

import asyncio
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


# Mock the required classes for demonstration
class QueryType(Enum):
    KNOWLEDGE_BASE = "knowledge_base"
    REAL_TIME = "real_time"
    HYBRID = "hybrid"


@dataclass
class QueryClassification:
    query_type: QueryType
    confidence_score: float
    reasoning: str
    suggested_sources: list
    metadata: dict
    processing_time: float


class MockQueryClassifier:
    """Mock classifier for demonstration."""
    
    async def classify_query(self, query: str, context=None):
        """Mock classification based on simple heuristics."""
        query_lower = query.lower()
        
        # Simple heuristic classification
        if any(word in query_lower for word in ["recent", "latest", "current", "2024", "new"]):
            return QueryClassification(
                query_type=QueryType.REAL_TIME,
                confidence_score=0.85,
                reasoning="Query contains temporal indicators",
                suggested_sources=["web_search"],
                metadata={"temporal_keywords": True},
                processing_time=0.1
            )
        elif any(word in query_lower for word in ["what is", "define", "explain", "how does"]):
            return QueryClassification(
                query_type=QueryType.KNOWLEDGE_BASE,
                confidence_score=0.9,
                reasoning="Query asks for established knowledge",
                suggested_sources=["knowledge_base"],
                metadata={"definition_query": True},
                processing_time=0.1
            )
        else:
            return QueryClassification(
                query_type=QueryType.HYBRID,
                confidence_score=0.7,
                reasoning="Complex query requiring multiple sources",
                suggested_sources=["both"],
                metadata={"complex_query": True},
                processing_time=0.1
            )


class MockConfig:
    """Mock configuration."""
    def __init__(self):
        self.cache_directory = "/tmp/demo"
        self.max_concurrent_requests = 5


# Simplified router implementation for demonstration
class DemoQueryRouter:
    """Simplified router for demonstration purposes."""
    
    def __init__(self, classifier, lightrag_func=None, perplexity_func=None):
        self.classifier = classifier
        self.lightrag_func = lightrag_func
        self.perplexity_func = perplexity_func
        self.metrics = {
            "total_queries": 0,
            "lightrag_queries": 0,
            "perplexity_queries": 0,
            "hybrid_queries": 0,
            "fallback_activations": 0
        }
    
    async def route_query(self, query: str) -> Dict[str, Any]:
        """Route a query and return response."""
        self.metrics["total_queries"] += 1
        
        # Step 1: Classify the query
        classification = await self.classifier.classify_query(query)
        
        # Step 2: Route based on classification
        if classification.query_type == QueryType.KNOWLEDGE_BASE:
            return await self._handle_knowledge_base_query(query, classification)
        elif classification.query_type == QueryType.REAL_TIME:
            return await self._handle_real_time_query(query, classification)
        else:  # HYBRID
            return await self._handle_hybrid_query(query, classification)
    
    async def _handle_knowledge_base_query(self, query: str, classification) -> Dict[str, Any]:
        """Handle knowledge base queries."""
        self.metrics["lightrag_queries"] += 1
        
        if self.lightrag_func:
            try:
                result = await self.lightrag_func(query)
                result["routing_decision"] = {
                    "strategy": "lightrag_only",
                    "classification": classification,
                    "confidence": classification.confidence_score
                }
                return result
            except Exception as e:
                # Fallback to Perplexity
                self.metrics["fallback_activations"] += 1
                if self.perplexity_func:
                    result = await self.perplexity_func(query)
                    result["routing_decision"] = {
                        "strategy": "fallback_to_perplexity",
                        "primary_failure": str(e),
                        "classification": classification
                    }
                    return result
        
        return {
            "content": f"Mock LightRAG response to: {query}",
            "source": "LightRAG",
            "confidence_score": 0.9,
            "routing_decision": {
                "strategy": "lightrag_only",
                "classification": classification
            }
        }
    
    async def _handle_real_time_query(self, query: str, classification) -> Dict[str, Any]:
        """Handle real-time queries."""
        self.metrics["perplexity_queries"] += 1
        
        if self.perplexity_func:
            try:
                result = await self.perplexity_func(query)
                result["routing_decision"] = {
                    "strategy": "perplexity_only",
                    "classification": classification,
                    "confidence": classification.confidence_score
                }
                return result
            except Exception as e:
                # Fallback to LightRAG
                self.metrics["fallback_activations"] += 1
                if self.lightrag_func:
                    result = await self.lightrag_func(query)
                    result["routing_decision"] = {
                        "strategy": "fallback_to_lightrag",
                        "primary_failure": str(e),
                        "classification": classification
                    }
                    return result
        
        return {
            "content": f"Mock Perplexity response to: {query}",
            "source": "Perplexity",
            "confidence_score": 0.8,
            "routing_decision": {
                "strategy": "perplexity_only",
                "classification": classification
            }
        }
    
    async def _handle_hybrid_query(self, query: str, classification) -> Dict[str, Any]:
        """Handle hybrid queries by combining both sources."""
        self.metrics["hybrid_queries"] += 1
        
        # Try to get responses from both sources
        lightrag_result = None
        perplexity_result = None
        errors = []
        
        if self.lightrag_func:
            try:
                lightrag_result = await self.lightrag_func(query)
            except Exception as e:
                errors.append(f"LightRAG failed: {str(e)}")
        
        if self.perplexity_func:
            try:
                perplexity_result = await self.perplexity_func(query)
            except Exception as e:
                errors.append(f"Perplexity failed: {str(e)}")
        
        # Combine results
        if lightrag_result and perplexity_result:
            combined_content = (
                f"**Knowledge Base Response:**\n{lightrag_result['content']}\n\n"
                f"**Real-time Response:**\n{perplexity_result['content']}"
            )
            return {
                "content": combined_content,
                "source": "Combined",
                "confidence_score": (lightrag_result.get("confidence_score", 0.0) + 
                                   perplexity_result.get("confidence_score", 0.0)) / 2,
                "routing_decision": {
                    "strategy": "hybrid_parallel",
                    "classification": classification,
                    "sources_used": ["lightrag", "perplexity"]
                },
                "errors": errors
            }
        elif lightrag_result:
            lightrag_result["routing_decision"] = {
                "strategy": "hybrid_lightrag_only",
                "classification": classification,
                "partial_failure": True
            }
            lightrag_result["errors"] = errors
            return lightrag_result
        elif perplexity_result:
            perplexity_result["routing_decision"] = {
                "strategy": "hybrid_perplexity_only", 
                "classification": classification,
                "partial_failure": True
            }
            perplexity_result["errors"] = errors
            return perplexity_result
        else:
            return {
                "content": "I apologize, but I'm unable to process your query at the moment.",
                "source": "Error",
                "confidence_score": 0.0,
                "routing_decision": {
                    "strategy": "error_fallback",
                    "classification": classification
                },
                "errors": errors
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get routing metrics."""
        return self.metrics.copy()


# Mock query functions
async def mock_lightrag_query(query: str) -> Dict[str, Any]:
    """Mock LightRAG query function."""
    await asyncio.sleep(0.1)  # Simulate processing time
    return {
        "content": f"LightRAG knowledge base response to: {query}",
        "source": "LightRAG",
        "confidence_score": 0.9,
        "bibliography": "**Sources:**\n[1]: Research Paper A\n[2]: Clinical Study B",
        "processing_time": 0.1
    }


async def mock_perplexity_query(query: str) -> Dict[str, Any]:
    """Mock Perplexity query function."""
    await asyncio.sleep(0.05)  # Simulate processing time
    return {
        "content": f"Perplexity real-time response to: {query}",
        "source": "Perplexity", 
        "confidence_score": 0.8,
        "bibliography": "**References:**\n[1]: Recent Web Article\n[2]: News Source",
        "processing_time": 0.05
    }


async def demo_routing_functionality():
    """Demonstrate the routing functionality."""
    print("=== QueryRouter Demonstration ===\n")
    
    # Initialize components
    classifier = MockQueryClassifier()
    router = DemoQueryRouter(
        classifier=classifier,
        lightrag_func=mock_lightrag_query,
        perplexity_func=mock_perplexity_query
    )
    
    # Test queries
    test_queries = [
        "What is clinical metabolomics?",  # Should route to LightRAG
        "Latest metabolomics research in 2024",  # Should route to Perplexity
        "How has metabolomics evolved recently?",  # Should use hybrid approach
        "Define biomarker discovery",  # Should route to LightRAG
        "Current trends in mass spectrometry"  # Should route to Perplexity
    ]
    
    print("Testing different query types and routing decisions:\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"{i}. Query: '{query}'")
        
        try:
            result = await router.route_query(query)
            
            print(f"   → Routed to: {result['source']}")
            print(f"   → Strategy: {result['routing_decision']['strategy']}")
            print(f"   → Classification: {result['routing_decision']['classification'].query_type.value}")
            print(f"   → Confidence: {result['confidence_score']:.2f}")
            print(f"   → Response: {result['content'][:100]}...")
            
            if result.get('errors'):
                print(f"   → Errors: {len(result['errors'])} error(s)")
            
        except Exception as e:
            print(f"   → Error: {str(e)}")
        
        print()
    
    # Show metrics
    print("=== Routing Metrics ===")
    metrics = router.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    print("\n=== Fallback Demonstration ===")
    
    # Demonstrate fallback mechanism
    async def failing_lightrag(query: str):
        raise Exception("LightRAG service unavailable")
    
    router.lightrag_func = failing_lightrag
    
    query = "What is metabolomics?"
    print(f"Query with LightRAG failure: '{query}'")
    
    result = await router.route_query(query)
    print(f"   → Routed to: {result['source']}")
    print(f"   → Strategy: {result['routing_decision']['strategy']}")
    print(f"   → Primary failure: {result['routing_decision'].get('primary_failure', 'N/A')}")
    
    print(f"\nFinal metrics:")
    final_metrics = router.get_metrics()
    for key, value in final_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(demo_routing_functionality())