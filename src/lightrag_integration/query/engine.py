"""
LightRAG Query Engine

This module handles query processing and response generation using the
constructed knowledge graphs. Implements requirements 1.7 and 8.1.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import re
import math

from ..utils.logging import setup_logger
from ..ingestion.knowledge_graph import KnowledgeGraphBuilder, GraphNode, GraphEdge, KnowledgeGraph
from .response_formatter import LightRAGResponseFormatter, FormattedResponse


@dataclass
class QueryResult:
    """Result of a query operation."""
    answer: str
    confidence_score: float
    source_documents: List[str]
    entities_used: List[Dict[str, Any]]
    relationships_used: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]
    formatted_response: Optional[FormattedResponse] = None
    confidence_breakdown: Optional[Dict[str, Any]] = None


@dataclass
class SearchResult:
    """Result of a semantic search operation."""
    node_id: str
    node_text: str
    node_type: str
    relevance_score: float
    source_documents: List[str]
    context: str


@dataclass
class TraversalResult:
    """Result of a graph traversal operation."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    paths: List[List[str]]
    depth_reached: int


class LightRAGQueryEngine:
    """
    Query engine for processing questions against the LightRAG knowledge graph.
    
    This class implements semantic search, graph traversal, and response generation
    capabilities for the LightRAG system.
    """
    
    def __init__(self, config):
        """Initialize the query engine."""
        self.config = config
        self.logger = setup_logger("lightrag_query")
        
        # Initialize knowledge graph builder for accessing graphs
        self.kg_builder = KnowledgeGraphBuilder(config)
        
        # Initialize response formatter
        self.response_formatter = LightRAGResponseFormatter(config)
        
        # Cache for loaded graphs
        self.graph_cache: Dict[str, KnowledgeGraph] = {}
        
        # Query processing parameters
        self.max_query_length = getattr(config, 'max_query_length', 1000)
        self.default_top_k = getattr(config, 'default_top_k', 10)
        self.similarity_threshold = getattr(config, 'similarity_threshold', 0.5)
        
        self.logger.info("LightRAG Query Engine initialized")
    
    async def process_query(self, question: str, context: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Process a query against the knowledge graph.
        
        Args:
            question: The question to answer
            context: Optional context information
        
        Returns:
            QueryResult with answer and metadata
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing query: {question[:100]}...")
            
            # Validate query
            if not question or len(question.strip()) == 0:
                return QueryResult(
                    answer="Please provide a valid question.",
                    confidence_score=0.0,
                    source_documents=[],
                    entities_used=[],
                    relationships_used=[],
                    processing_time=time.time() - start_time,
                    metadata=context or {}
                )
            
            if len(question) > self.max_query_length:
                return QueryResult(
                    answer=f"Query too long. Maximum length is {self.max_query_length} characters.",
                    confidence_score=0.0,
                    source_documents=[],
                    entities_used=[],
                    relationships_used=[],
                    processing_time=time.time() - start_time,
                    metadata=context or {}
                )
            
            # Load available graphs
            available_graphs = await self._load_available_graphs()
            
            if not available_graphs:
                return QueryResult(
                    answer="No knowledge graphs available. Please ingest some documents first.",
                    confidence_score=0.0,
                    source_documents=[],
                    entities_used=[],
                    relationships_used=[],
                    processing_time=time.time() - start_time,
                    metadata=context or {}
                )
            
            # Perform semantic search to find relevant nodes
            search_results = await self.semantic_search(question, top_k=self.default_top_k)
            
            if not search_results:
                return QueryResult(
                    answer="I couldn't find relevant information to answer your question.",
                    confidence_score=0.0,
                    source_documents=[],
                    entities_used=[],
                    relationships_used=[],
                    processing_time=time.time() - start_time,
                    metadata=context or {}
                )
            
            # Extract relevant entities for graph traversal
            relevant_entities = [result.node_id for result in search_results[:5]]
            
            # Perform graph traversal to find connected information
            traversal_result = await self.graph_traversal(relevant_entities, max_depth=2)
            
            # Generate response based on search and traversal results
            response = await self._generate_response(
                question, search_results, traversal_result, context
            )
            
            # Calculate enhanced confidence score
            enhanced_confidence, confidence_breakdown = self.response_formatter.calculate_enhanced_confidence_score(
                response["confidence_score"],
                response["entities_used"],
                response["relationships_used"],
                response["source_documents"]
            )
            
            # Format response for consistency with existing system
            formatted_response = self.response_formatter.format_response(
                answer=response["answer"],
                source_documents=response["source_documents"],
                entities_used=response["entities_used"],
                relationships_used=response["relationships_used"],
                confidence_score=enhanced_confidence,
                processing_time=time.time() - start_time,
                metadata=context
            )
            
            processing_time = time.time() - start_time
            
            self.logger.info(f"Query processed in {processing_time:.2f}s with confidence {enhanced_confidence:.2f}")
            
            return QueryResult(
                answer=response["answer"],
                confidence_score=enhanced_confidence,
                source_documents=response["source_documents"],
                entities_used=response["entities_used"],
                relationships_used=response["relationships_used"],
                processing_time=processing_time,
                metadata=context or {},
                formatted_response=formatted_response,
                confidence_breakdown=confidence_breakdown
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Error processing query: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return QueryResult(
                answer="I encountered an error while processing your question. Please try again.",
                confidence_score=0.0,
                source_documents=[],
                entities_used=[],
                relationships_used=[],
                processing_time=processing_time,
                metadata={"error": error_msg, **(context or {})}
            )
    
    async def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """
        Perform semantic search in the knowledge graph.
        
        Args:
            query: Search query
            top_k: Number of top results to return
        
        Returns:
            List of SearchResult objects
        """
        try:
            self.logger.info(f"Performing semantic search for: {query[:50]}...")
            
            # Load available graphs
            available_graphs = await self._load_available_graphs()
            
            if not available_graphs:
                return []
            
            # Collect all nodes from all graphs
            all_nodes = []
            for graph in available_graphs:
                for node_id, node in graph.nodes.items():
                    all_nodes.append((node_id, node, graph.graph_id))
            
            # Calculate relevance scores for each node
            scored_results = []
            query_terms = self._extract_query_terms(query)
            
            for node_id, node, graph_id in all_nodes:
                relevance_score = self._calculate_node_relevance(node, query_terms)
                
                if relevance_score >= self.similarity_threshold:
                    search_result = SearchResult(
                        node_id=node_id,
                        node_text=node.text,
                        node_type=node.node_type,
                        relevance_score=relevance_score,
                        source_documents=node.source_documents,
                        context=node.properties.get("context", "")
                    )
                    scored_results.append(search_result)
            
            # Sort by relevance score and return top_k
            scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
            results = scored_results[:top_k]
            
            self.logger.info(f"Found {len(results)} relevant nodes")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
            return []
    
    async def graph_traversal(self, start_entities: List[str], max_depth: int = 3) -> TraversalResult:
        """
        Perform graph traversal from starting entities.
        
        Args:
            start_entities: List of entity IDs to start from
            max_depth: Maximum traversal depth
        
        Returns:
            TraversalResult with traversed nodes and edges
        """
        try:
            self.logger.info(f"Performing graph traversal from {len(start_entities)} entities")
            
            # Load available graphs
            available_graphs = await self._load_available_graphs()
            
            if not available_graphs:
                return TraversalResult(nodes=[], edges=[], paths=[], depth_reached=0)
            
            # Combine all graphs for traversal
            combined_nodes = {}
            combined_edges = {}
            
            for graph in available_graphs:
                combined_nodes.update(graph.nodes)
                combined_edges.update(graph.edges)
            
            # Perform breadth-first traversal
            visited_nodes = set()
            visited_edges = set()
            traversal_queue = deque([(entity_id, 0) for entity_id in start_entities if entity_id in combined_nodes])
            paths = []
            max_depth_reached = 0
            
            while traversal_queue:
                current_node_id, depth = traversal_queue.popleft()
                
                if depth > max_depth or current_node_id in visited_nodes:
                    continue
                
                visited_nodes.add(current_node_id)
                max_depth_reached = max(max_depth_reached, depth)
                
                # Find connected edges
                for edge_id, edge in combined_edges.items():
                    if edge_id in visited_edges:
                        continue
                    
                    next_node_id = None
                    if edge.source_node_id == current_node_id:
                        next_node_id = edge.target_node_id
                    elif edge.target_node_id == current_node_id:
                        next_node_id = edge.source_node_id
                    
                    if next_node_id and next_node_id in combined_nodes and depth < max_depth:
                        visited_edges.add(edge_id)
                        traversal_queue.append((next_node_id, depth + 1))
                        
                        # Record path
                        paths.append([current_node_id, next_node_id])
            
            # Prepare results
            result_nodes = []
            for node_id in visited_nodes:
                if node_id in combined_nodes:
                    node = combined_nodes[node_id]
                    result_nodes.append({
                        "node_id": node_id,
                        "text": node.text,
                        "node_type": node.node_type,
                        "confidence_score": node.confidence_score,
                        "source_documents": node.source_documents,
                        "properties": node.properties
                    })
            
            result_edges = []
            for edge_id in visited_edges:
                if edge_id in combined_edges:
                    edge = combined_edges[edge_id]
                    result_edges.append({
                        "edge_id": edge_id,
                        "source_node_id": edge.source_node_id,
                        "target_node_id": edge.target_node_id,
                        "edge_type": edge.edge_type,
                        "confidence_score": edge.confidence_score,
                        "evidence": edge.evidence,
                        "source_documents": edge.source_documents
                    })
            
            result = TraversalResult(
                nodes=result_nodes,
                edges=result_edges,
                paths=paths,
                depth_reached=max_depth_reached
            )
            
            self.logger.info(f"Traversal completed: {len(result_nodes)} nodes, {len(result_edges)} edges")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in graph traversal: {str(e)}", exc_info=True)
            return TraversalResult(nodes=[], edges=[], paths=[], depth_reached=0)
    
    async def _load_available_graphs(self) -> List[KnowledgeGraph]:
        """Load all available knowledge graphs."""
        try:
            graph_ids = await self.kg_builder.list_available_graphs()
            graphs = []
            
            for graph_id in graph_ids:
                if graph_id in self.graph_cache:
                    graphs.append(self.graph_cache[graph_id])
                else:
                    graph = await self.kg_builder.load_graph_from_storage(graph_id)
                    if graph:
                        self.graph_cache[graph_id] = graph
                        graphs.append(graph)
            
            return graphs
            
        except Exception as e:
            self.logger.error(f"Error loading graphs: {str(e)}", exc_info=True)
            return []
    
    def _extract_query_terms(self, query: str) -> List[str]:
        """Extract meaningful terms from the query."""
        # Simple term extraction - in a more sophisticated implementation,
        # we would use NLP techniques like named entity recognition
        
        # Remove common stop words
        stop_words = {
            'what', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'about', 'how', 'why', 'when', 'where', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'can', 'could', 'should', 'would', 'will', 'shall', 'may', 'might', 'must',
            'do', 'does', 'did', 'have', 'has', 'had', 'be', 'been', 'being', 'am', 'is', 'are', 'was', 'were'
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return terms
    
    def _calculate_node_relevance(self, node: GraphNode, query_terms: List[str]) -> float:
        """Calculate relevance score between a node and query terms."""
        if not query_terms:
            return 0.0
        
        # Combine node text and context for matching
        searchable_text = f"{node.text} {node.properties.get('context', '')}".lower()
        
        # Calculate term frequency
        term_matches = 0
        for term in query_terms:
            if term in searchable_text:
                term_matches += 1
        
        # Base relevance score
        relevance = term_matches / len(query_terms)
        
        # Boost score based on node type (clinical metabolomics specific)
        type_boosts = {
            'metabolite': 1.2,
            'disease': 1.1,
            'pathway': 1.15,
            'biomarker': 1.25,
            'clinical_condition': 1.1,
            'measurement': 1.05
        }
        
        type_boost = type_boosts.get(node.node_type.lower(), 1.0)
        relevance *= type_boost
        
        # Boost based on confidence score
        confidence_boost = 0.8 + (node.confidence_score * 0.2)
        relevance *= confidence_boost
        
        return min(relevance, 1.0)  # Cap at 1.0
    
    async def _generate_response(
        self, 
        question: str, 
        search_results: List[SearchResult], 
        traversal_result: TraversalResult,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate a response based on search and traversal results."""
        try:
            # Extract information from results
            relevant_info = []
            source_documents = set()
            entities_used = []
            relationships_used = []
            
            # Process search results
            for result in search_results[:5]:  # Use top 5 results
                relevant_info.append({
                    "text": result.node_text,
                    "type": result.node_type,
                    "relevance": result.relevance_score,
                    "context": result.context
                })
                source_documents.update(result.source_documents)
                entities_used.append({
                    "id": result.node_id,
                    "text": result.node_text,
                    "type": result.node_type,
                    "relevance_score": result.relevance_score
                })
            
            # Process traversal results for relationships
            for edge in traversal_result.edges[:10]:  # Use top 10 edges
                relationships_used.append({
                    "id": edge["edge_id"],
                    "type": edge["edge_type"],
                    "source": edge["source_node_id"],
                    "target": edge["target_node_id"],
                    "confidence": edge["confidence_score"],
                    "evidence": edge["evidence"][:2]  # First 2 pieces of evidence
                })
                source_documents.update(edge["source_documents"])
            
            # Generate answer based on question type
            answer = await self._construct_answer(question, relevant_info, relationships_used)
            
            # Calculate confidence score
            confidence_score = self._calculate_response_confidence(search_results, traversal_result)
            
            return {
                "answer": answer,
                "confidence_score": confidence_score,
                "source_documents": list(source_documents),
                "entities_used": entities_used,
                "relationships_used": relationships_used
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                "answer": "I encountered an error while generating the response.",
                "confidence_score": 0.0,
                "source_documents": [],
                "entities_used": [],
                "relationships_used": []
            }
    
    async def _construct_answer(
        self, 
        question: str, 
        relevant_info: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]]
    ) -> str:
        """Construct an answer from relevant information."""
        if not relevant_info:
            return "I couldn't find relevant information to answer your question."
        
        # Detect question type
        question_lower = question.lower()
        
        if any(phrase in question_lower for phrase in ["what is", "define", "definition"]):
            # Definition question
            return self._construct_definition_answer(relevant_info, relationships)
        elif any(phrase in question_lower for phrase in ["how does", "how do", "mechanism"]):
            # Mechanism question
            return self._construct_mechanism_answer(relevant_info, relationships)
        elif any(phrase in question_lower for phrase in ["why", "reason", "cause"]):
            # Causal question
            return self._construct_causal_answer(relevant_info, relationships)
        else:
            # General question
            return self._construct_general_answer(relevant_info, relationships)
    
    def _construct_definition_answer(self, relevant_info: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> str:
        """Construct a definition-type answer."""
        # Find the most relevant entity
        primary_entity = relevant_info[0] if relevant_info else None
        
        if not primary_entity:
            return "I couldn't find a clear definition for your query."
        
        answer_parts = []
        
        # Start with the primary definition
        if primary_entity["context"]:
            answer_parts.append(f"{primary_entity['text']} is {primary_entity['context']}")
        else:
            answer_parts.append(f"{primary_entity['text']} is a {primary_entity['type']}")
        
        # Add related information
        related_entities = [info for info in relevant_info[1:3] if info["relevance"] > 0.6]
        if related_entities:
            related_texts = [f"{info['text']} ({info['type']})" for info in related_entities]
            answer_parts.append(f"It is related to {', '.join(related_texts)}")
        
        # Add relationship information
        relevant_relationships = [rel for rel in relationships[:3] if rel["confidence"] > 0.6]
        if relevant_relationships:
            rel_descriptions = []
            for rel in relevant_relationships:
                if rel["evidence"]:
                    rel_descriptions.append(rel["evidence"][0][:100])
            if rel_descriptions:
                answer_parts.append(f"Key relationships include: {'. '.join(rel_descriptions)}")
        
        return ". ".join(answer_parts) + "."
    
    def _construct_mechanism_answer(self, relevant_info: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> str:
        """Construct a mechanism-type answer."""
        answer_parts = []
        
        # Focus on pathway and process entities
        process_entities = [info for info in relevant_info if info["type"] in ["pathway", "process", "mechanism"]]
        
        if process_entities:
            primary_process = process_entities[0]
            answer_parts.append(f"The mechanism involves {primary_process['text']}")
            
            if primary_process["context"]:
                answer_parts.append(primary_process["context"])
        
        # Add relationship-based mechanism information
        mechanism_relationships = [rel for rel in relationships if any(
            keyword in rel["type"].lower() for keyword in ["regulates", "activates", "inhibits", "causes", "leads_to"]
        )]
        
        if mechanism_relationships:
            for rel in mechanism_relationships[:2]:
                if rel["evidence"]:
                    answer_parts.append(rel["evidence"][0][:150])
        
        if not answer_parts:
            return "I found some relevant information but couldn't determine the specific mechanism."
        
        return ". ".join(answer_parts) + "."
    
    def _construct_causal_answer(self, relevant_info: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> str:
        """Construct a causal-type answer."""
        answer_parts = []
        
        # Look for causal relationships
        causal_relationships = [rel for rel in relationships if any(
            keyword in rel["type"].lower() for keyword in ["causes", "leads_to", "results_in", "triggers"]
        )]
        
        if causal_relationships:
            for rel in causal_relationships[:2]:
                if rel["evidence"]:
                    answer_parts.append(rel["evidence"][0][:150])
        
        # Add relevant entity information
        if relevant_info:
            primary_entity = relevant_info[0]
            if primary_entity["context"]:
                answer_parts.append(f"This relates to {primary_entity['text']}: {primary_entity['context']}")
        
        if not answer_parts:
            return "I found relevant information but couldn't identify clear causal relationships."
        
        return ". ".join(answer_parts) + "."
    
    def _construct_general_answer(self, relevant_info: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> str:
        """Construct a general answer."""
        answer_parts = []
        
        # Include top relevant entities
        for info in relevant_info[:3]:
            if info["context"]:
                answer_parts.append(f"{info['text']}: {info['context']}")
            else:
                answer_parts.append(f"{info['text']} is a {info['type']}")
        
        # Add relationship information
        if relationships:
            rel_info = []
            for rel in relationships[:2]:
                if rel["evidence"]:
                    rel_info.append(rel["evidence"][0][:100])
            if rel_info:
                answer_parts.append(f"Related information: {'. '.join(rel_info)}")
        
        if not answer_parts:
            return "I found some relevant information but couldn't formulate a comprehensive answer."
        
        return ". ".join(answer_parts) + "."
    
    def _calculate_response_confidence(self, search_results: List[SearchResult], traversal_result: TraversalResult) -> float:
        """Calculate confidence score for the response."""
        if not search_results:
            return 0.0
        
        # Base confidence from search results
        avg_relevance = sum(result.relevance_score for result in search_results[:5]) / min(len(search_results), 5)
        
        # Boost confidence based on number of supporting relationships
        relationship_boost = min(len(traversal_result.edges) * 0.05, 0.2)
        
        # Boost confidence based on source document diversity
        all_sources = set()
        for result in search_results:
            all_sources.update(result.source_documents)
        source_diversity_boost = min(len(all_sources) * 0.02, 0.1)
        
        # Calculate final confidence
        confidence = avg_relevance + relationship_boost + source_diversity_boost
        
        return min(confidence, 1.0)  # Cap at 1.0