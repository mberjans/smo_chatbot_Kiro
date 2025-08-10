"""
Knowledge Graph Construction Component

This module handles the construction of knowledge graphs from extracted
entities and relationships. Implements requirements 1.6 and 6.2.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict
import hashlib

from ..utils.logging import setup_logger
from .entity_extractor import Entity, Relationship


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    node_id: str
    text: str
    node_type: str
    confidence_score: float
    properties: Dict[str, Any]
    source_documents: List[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    edge_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    confidence_score: float
    evidence: List[str]
    properties: Dict[str, Any]
    source_documents: List[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class KnowledgeGraph:
    """Represents a complete knowledge graph."""
    graph_id: str
    nodes: Dict[str, GraphNode]
    edges: Dict[str, GraphEdge]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


@dataclass
class GraphConstructionResult:
    """Result of knowledge graph construction."""
    success: bool
    graph: Optional[KnowledgeGraph]
    nodes_created: int
    edges_created: int
    nodes_updated: int
    edges_updated: int
    processing_time: float
    error_message: Optional[str]


class KnowledgeGraphBuilder:
    """
    Knowledge graph construction and management system.
    
    This class handles the creation, updating, and storage of knowledge graphs
    from extracted entities and relationships.
    """
    
    def __init__(self, config=None):
        """
        Initialize the knowledge graph builder.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        self.logger = setup_logger("knowledge_graph_builder")
        
        # Storage paths
        self.graph_storage_path = self._get_storage_path()
        self.ensure_storage_directory()
        
        # In-memory graph cache
        self.graph_cache: Dict[str, KnowledgeGraph] = {}
        
        # Node and edge similarity thresholds
        self.node_similarity_threshold = 0.8
        self.edge_similarity_threshold = 0.7
        
        self.logger.info("KnowledgeGraphBuilder initialized")
    
    def _get_storage_path(self) -> Path:
        """Get the storage path for knowledge graphs."""
        if self.config and hasattr(self.config, 'knowledge_graph_path'):
            return Path(self.config.knowledge_graph_path)
        else:
            return Path("data/lightrag_kg")
    
    def ensure_storage_directory(self):
        """Ensure the storage directory exists."""
        self.graph_storage_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Knowledge graph storage: {self.graph_storage_path}")
    
    def _generate_node_id(self, text: str, node_type: str) -> str:
        """Generate a unique node ID based on text and type."""
        content = f"{node_type}:{text.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _generate_edge_id(self, source_id: str, target_id: str, edge_type: str) -> str:
        """Generate a unique edge ID."""
        content = f"{source_id}:{target_id}:{edge_type}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        # Simple Jaccard similarity for now
        # In a more sophisticated implementation, we'd use embeddings
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def construct_graph_from_entities_and_relationships(
        self, 
        entities: List[Entity], 
        relationships: List[Relationship],
        document_id: str,
        graph_id: Optional[str] = None
    ) -> GraphConstructionResult:
        """
        Construct a knowledge graph from entities and relationships.
        
        Args:
            entities: List of extracted entities
            relationships: List of extracted relationships
            document_id: ID of the source document
            graph_id: Optional existing graph ID to update
        
        Returns:
            GraphConstructionResult with construction details
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Constructing graph from {len(entities)} entities and {len(relationships)} relationships")
            
            # Get or create graph
            if graph_id and graph_id in self.graph_cache:
                graph = self.graph_cache[graph_id]
            else:
                graph_id = graph_id or f"graph_{document_id}_{int(datetime.now().timestamp())}"
                graph = KnowledgeGraph(
                    graph_id=graph_id,
                    nodes={},
                    edges={},
                    metadata={
                        "source_documents": [],
                        "creation_method": "entity_relationship_extraction"
                    },
                    created_at=datetime.now(),
                    updated_at=datetime.now()
                )
            
            # Track statistics
            nodes_created = 0
            nodes_updated = 0
            edges_created = 0
            edges_updated = 0
            
            # Process entities into nodes
            for entity in entities:
                node_result = await self._process_entity_to_node(entity, document_id, graph)
                if node_result["created"]:
                    nodes_created += 1
                elif node_result["updated"]:
                    nodes_updated += 1
            
            # Process relationships into edges
            for relationship in relationships:
                edge_result = await self._process_relationship_to_edge(relationship, document_id, graph)
                if edge_result["created"]:
                    edges_created += 1
                elif edge_result["updated"]:
                    edges_updated += 1
            
            # Update graph metadata
            if document_id not in graph.metadata["source_documents"]:
                graph.metadata["source_documents"].append(document_id)
            graph.updated_at = datetime.now()
            
            # Cache the graph
            self.graph_cache[graph_id] = graph
            
            # Save to storage
            await self._save_graph_to_storage(graph)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = GraphConstructionResult(
                success=True,
                graph=graph,
                nodes_created=nodes_created,
                edges_created=edges_created,
                nodes_updated=nodes_updated,
                edges_updated=edges_updated,
                processing_time=processing_time,
                error_message=None
            )
            
            self.logger.info(f"Graph construction completed: {nodes_created} nodes created, {edges_created} edges created")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Error constructing knowledge graph: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return GraphConstructionResult(
                success=False,
                graph=None,
                nodes_created=0,
                edges_created=0,
                nodes_updated=0,
                edges_updated=0,
                processing_time=processing_time,
                error_message=error_msg
            )
    
    async def _process_entity_to_node(self, entity: Entity, document_id: str, graph: KnowledgeGraph) -> Dict[str, bool]:
        """Process an entity into a graph node."""
        node_id = self._generate_node_id(entity.text, entity.entity_type)
        
        # Check if node already exists
        if node_id in graph.nodes:
            # Update existing node
            existing_node = graph.nodes[node_id]
            
            # Update confidence score (take maximum)
            if entity.confidence_score > existing_node.confidence_score:
                existing_node.confidence_score = entity.confidence_score
            
            # Add source document
            if document_id not in existing_node.source_documents:
                existing_node.source_documents.append(document_id)
            
            # Update properties
            existing_node.properties.update(entity.metadata)
            existing_node.updated_at = datetime.now()
            
            return {"created": False, "updated": True}
        
        else:
            # Create new node
            node = GraphNode(
                node_id=node_id,
                text=entity.text,
                node_type=entity.entity_type,
                confidence_score=entity.confidence_score,
                properties={
                    "start_pos": entity.start_pos,
                    "end_pos": entity.end_pos,
                    "context": entity.context,
                    **entity.metadata
                },
                source_documents=[document_id],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            graph.nodes[node_id] = node
            return {"created": True, "updated": False}
    
    async def _process_relationship_to_edge(self, relationship: Relationship, document_id: str, graph: KnowledgeGraph) -> Dict[str, bool]:
        """Process a relationship into a graph edge."""
        # Find corresponding nodes
        source_node_id = None
        target_node_id = None
        
        for node_id, node in graph.nodes.items():
            # Match by entity ID from the relationship
            if hasattr(relationship, 'source_entity_id') and relationship.source_entity_id:
                # Try to find nodes that match the entity IDs
                if relationship.source_entity_id in node.properties.get('original_entity_id', ''):
                    source_node_id = node_id
                if relationship.target_entity_id in node.properties.get('original_entity_id', ''):
                    target_node_id = node_id
        
        # If we couldn't find nodes by entity ID, try to match by text similarity
        if not source_node_id or not target_node_id:
            for node_id, node in graph.nodes.items():
                if not source_node_id:
                    # Look for source entity text in evidence
                    if any(word in relationship.evidence_text.lower() for word in node.text.lower().split()):
                        source_node_id = node_id
                
                if not target_node_id and source_node_id != node_id:
                    # Look for target entity text in evidence
                    if any(word in relationship.evidence_text.lower() for word in node.text.lower().split()):
                        target_node_id = node_id
        
        if not source_node_id or not target_node_id:
            self.logger.warning(f"Could not find nodes for relationship: {relationship.relationship_type}")
            return {"created": False, "updated": False}
        
        edge_id = self._generate_edge_id(source_node_id, target_node_id, relationship.relationship_type)
        
        # Check if edge already exists
        if edge_id in graph.edges:
            # Update existing edge
            existing_edge = graph.edges[edge_id]
            
            # Update confidence score (take maximum)
            if relationship.confidence_score > existing_edge.confidence_score:
                existing_edge.confidence_score = relationship.confidence_score
            
            # Add evidence
            if relationship.evidence_text not in existing_edge.evidence:
                existing_edge.evidence.append(relationship.evidence_text)
            
            # Add source document
            if document_id not in existing_edge.source_documents:
                existing_edge.source_documents.append(document_id)
            
            # Update properties
            existing_edge.properties.update(relationship.metadata)
            existing_edge.updated_at = datetime.now()
            
            return {"created": False, "updated": True}
        
        else:
            # Create new edge
            edge = GraphEdge(
                edge_id=edge_id,
                source_node_id=source_node_id,
                target_node_id=target_node_id,
                edge_type=relationship.relationship_type,
                confidence_score=relationship.confidence_score,
                evidence=[relationship.evidence_text],
                properties={
                    "context": relationship.context,
                    **relationship.metadata
                },
                source_documents=[document_id],
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            graph.edges[edge_id] = edge
            return {"created": True, "updated": False}
    
    async def _save_graph_to_storage(self, graph: KnowledgeGraph):
        """Save graph to persistent storage."""
        try:
            graph_file = self.graph_storage_path / f"{graph.graph_id}.json"
            
            # Convert graph to serializable format
            graph_data = {
                "graph_id": graph.graph_id,
                "nodes": {
                    node_id: {
                        **asdict(node),
                        "created_at": node.created_at.isoformat(),
                        "updated_at": node.updated_at.isoformat()
                    }
                    for node_id, node in graph.nodes.items()
                },
                "edges": {
                    edge_id: {
                        **asdict(edge),
                        "created_at": edge.created_at.isoformat(),
                        "updated_at": edge.updated_at.isoformat()
                    }
                    for edge_id, edge in graph.edges.items()
                },
                "metadata": graph.metadata,
                "created_at": graph.created_at.isoformat(),
                "updated_at": graph.updated_at.isoformat()
            }
            
            with open(graph_file, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Graph saved to {graph_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving graph to storage: {str(e)}", exc_info=True)
    
    async def load_graph_from_storage(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Load graph from persistent storage."""
        try:
            graph_file = self.graph_storage_path / f"{graph_id}.json"
            
            if not graph_file.exists():
                self.logger.warning(f"Graph file not found: {graph_file}")
                return None
            
            with open(graph_file, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Convert back to graph objects
            nodes = {}
            for node_id, node_data in graph_data["nodes"].items():
                node_data["created_at"] = datetime.fromisoformat(node_data["created_at"])
                node_data["updated_at"] = datetime.fromisoformat(node_data["updated_at"])
                nodes[node_id] = GraphNode(**node_data)
            
            edges = {}
            for edge_id, edge_data in graph_data["edges"].items():
                edge_data["created_at"] = datetime.fromisoformat(edge_data["created_at"])
                edge_data["updated_at"] = datetime.fromisoformat(edge_data["updated_at"])
                edges[edge_id] = GraphEdge(**edge_data)
            
            graph = KnowledgeGraph(
                graph_id=graph_data["graph_id"],
                nodes=nodes,
                edges=edges,
                metadata=graph_data["metadata"],
                created_at=datetime.fromisoformat(graph_data["created_at"]),
                updated_at=datetime.fromisoformat(graph_data["updated_at"])
            )
            
            # Cache the loaded graph
            self.graph_cache[graph_id] = graph
            
            self.logger.info(f"Graph loaded from {graph_file}")
            return graph
            
        except Exception as e:
            self.logger.error(f"Error loading graph from storage: {str(e)}", exc_info=True)
            return None
    
    async def merge_graphs(self, graph_ids: List[str]) -> Optional[KnowledgeGraph]:
        """Merge multiple graphs into a single graph."""
        try:
            if not graph_ids:
                return None
            
            # Load all graphs
            graphs = []
            for graph_id in graph_ids:
                graph = await self.load_graph_from_storage(graph_id)
                if graph:
                    graphs.append(graph)
            
            if not graphs:
                return None
            
            # Create merged graph
            merged_graph_id = f"merged_{'_'.join(graph_ids[:3])}_{int(datetime.now().timestamp())}"
            merged_graph = KnowledgeGraph(
                graph_id=merged_graph_id,
                nodes={},
                edges={},
                metadata={
                    "source_graphs": graph_ids,
                    "source_documents": [],
                    "creation_method": "graph_merge"
                },
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            # Merge nodes
            for graph in graphs:
                for node_id, node in graph.nodes.items():
                    if node_id in merged_graph.nodes:
                        # Merge existing node
                        existing_node = merged_graph.nodes[node_id]
                        existing_node.confidence_score = max(existing_node.confidence_score, node.confidence_score)
                        existing_node.source_documents.extend(node.source_documents)
                        existing_node.source_documents = list(set(existing_node.source_documents))
                        existing_node.properties.update(node.properties)
                        existing_node.updated_at = datetime.now()
                    else:
                        # Add new node
                        merged_graph.nodes[node_id] = node
                
                # Merge edges
                for edge_id, edge in graph.edges.items():
                    if edge_id in merged_graph.edges:
                        # Merge existing edge
                        existing_edge = merged_graph.edges[edge_id]
                        existing_edge.confidence_score = max(existing_edge.confidence_score, edge.confidence_score)
                        existing_edge.evidence.extend(edge.evidence)
                        existing_edge.evidence = list(set(existing_edge.evidence))
                        existing_edge.source_documents.extend(edge.source_documents)
                        existing_edge.source_documents = list(set(existing_edge.source_documents))
                        existing_edge.properties.update(edge.properties)
                        existing_edge.updated_at = datetime.now()
                    else:
                        # Add new edge
                        merged_graph.edges[edge_id] = edge
                
                # Merge metadata
                merged_graph.metadata["source_documents"].extend(graph.metadata.get("source_documents", []))
            
            # Remove duplicates from source documents
            merged_graph.metadata["source_documents"] = list(set(merged_graph.metadata["source_documents"]))
            
            # Save merged graph
            await self._save_graph_to_storage(merged_graph)
            
            self.logger.info(f"Merged {len(graphs)} graphs into {merged_graph_id}")
            return merged_graph
            
        except Exception as e:
            self.logger.error(f"Error merging graphs: {str(e)}", exc_info=True)
            return None
    
    async def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Get statistics about a knowledge graph."""
        try:
            graph = self.graph_cache.get(graph_id) or await self.load_graph_from_storage(graph_id)
            
            if not graph:
                return {"error": "Graph not found"}
            
            # Calculate statistics
            node_types = defaultdict(int)
            edge_types = defaultdict(int)
            
            for node in graph.nodes.values():
                node_types[node.node_type] += 1
            
            for edge in graph.edges.values():
                edge_types[edge.edge_type] += 1
            
            stats = {
                "graph_id": graph.graph_id,
                "total_nodes": len(graph.nodes),
                "total_edges": len(graph.edges),
                "node_types": dict(node_types),
                "edge_types": dict(edge_types),
                "source_documents": len(graph.metadata.get("source_documents", [])),
                "created_at": graph.created_at.isoformat(),
                "updated_at": graph.updated_at.isoformat(),
                "avg_node_confidence": sum(node.confidence_score for node in graph.nodes.values()) / len(graph.nodes) if graph.nodes else 0,
                "avg_edge_confidence": sum(edge.confidence_score for edge in graph.edges.values()) / len(graph.edges) if graph.edges else 0
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting graph statistics: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    async def list_available_graphs(self) -> List[str]:
        """List all available graph IDs in storage."""
        try:
            graph_files = list(self.graph_storage_path.glob("*.json"))
            graph_ids = [f.stem for f in graph_files]
            return sorted(graph_ids)
            
        except Exception as e:
            self.logger.error(f"Error listing graphs: {str(e)}", exc_info=True)
            return []
    
    async def delete_graph(self, graph_id: str) -> bool:
        """Delete a graph from storage and cache."""
        try:
            # Remove from cache
            if graph_id in self.graph_cache:
                del self.graph_cache[graph_id]
            
            # Remove from storage
            graph_file = self.graph_storage_path / f"{graph_id}.json"
            if graph_file.exists():
                graph_file.unlink()
                self.logger.info(f"Deleted graph: {graph_id}")
                return True
            else:
                self.logger.warning(f"Graph file not found for deletion: {graph_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error deleting graph {graph_id}: {str(e)}", exc_info=True)
            return False