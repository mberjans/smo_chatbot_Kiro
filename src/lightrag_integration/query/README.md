# LightRAG Query Processing Engine

This module implements the query processing engine for the LightRAG integration, providing semantic search, graph traversal, and response generation capabilities.

## Components

### 1. Query Engine (`engine.py`)

The main query processing engine that handles:

- **Semantic Search**: Finds relevant nodes in the knowledge graph based on query terms
- **Graph Traversal**: Explores connected entities and relationships using breadth-first search
- **Response Generation**: Creates answers based on retrieved information
- **Query Validation**: Ensures queries meet length and format requirements

Key features:
- Async/await support for non-blocking operations
- Configurable similarity thresholds and result limits
- Comprehensive error handling and logging
- Support for different question types (definition, mechanism, causal)

### 2. Response Formatter (`response_formatter.py`)

Formats query results to be consistent with the existing Clinical Metabolomics Oracle system:

- **Citation Generation**: Creates citations from source documents
- **Bibliography Formatting**: Generates references and further reading sections
- **Confidence Indicators**: Adds confidence levels to responses
- **Enhanced Confidence Scoring**: Calculates confidence based on graph evidence strength

Key features:
- Compatible with existing system response format
- Multi-level confidence scoring (High, Medium, Low, Very Low)
- Evidence-based confidence enhancement
- Detailed confidence breakdown for transparency

## Data Models

### QueryResult
```python
@dataclass
class QueryResult:
    answer: str
    confidence_score: float
    source_documents: List[str]
    entities_used: List[Dict[str, Any]]
    relationships_used: List[Dict[str, Any]]
    processing_time: float
    metadata: Dict[str, Any]
    formatted_response: Optional[FormattedResponse]
    confidence_breakdown: Optional[Dict[str, Any]]
```

### SearchResult
```python
@dataclass
class SearchResult:
    node_id: str
    node_text: str
    node_type: str
    relevance_score: float
    source_documents: List[str]
    context: str
```

### TraversalResult
```python
@dataclass
class TraversalResult:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    paths: List[List[str]]
    depth_reached: int
```

### FormattedResponse
```python
@dataclass
class FormattedResponse:
    content: str
    bibliography: str
    confidence_scores: Dict[str, float]
    metadata: Dict[str, Any]
    processing_time: float
```

## Usage

### Basic Query Processing

```python
from lightrag_integration.query.engine import LightRAGQueryEngine
from lightrag_integration.config.settings import LightRAGConfig

# Initialize
config = LightRAGConfig()
engine = LightRAGQueryEngine(config)

# Process query
result = await engine.process_query("What is clinical metabolomics?")

# Access results
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence_score}")
print(f"Sources: {result.source_documents}")

# Access formatted response
if result.formatted_response:
    print(f"Formatted content: {result.formatted_response.content}")
    print(f"Bibliography: {result.formatted_response.bibliography}")
```

### Semantic Search

```python
# Perform semantic search
search_results = await engine.semantic_search("biomarkers", top_k=5)

for result in search_results:
    print(f"Node: {result.node_text} (relevance: {result.relevance_score})")
```

### Graph Traversal

```python
# Traverse from specific entities
traversal_result = await engine.graph_traversal(["entity_id_1"], max_depth=2)

print(f"Found {len(traversal_result.nodes)} connected nodes")
print(f"Found {len(traversal_result.edges)} relationships")
```

## Configuration

The query engine uses the following configuration parameters:

- `max_query_length`: Maximum allowed query length (default: 1000)
- `default_top_k`: Default number of search results (default: 10)
- `similarity_threshold`: Minimum similarity for search results (default: 0.5)

## Confidence Scoring

The system uses a multi-factor confidence scoring approach:

### Base Confidence Factors
- Entity relevance scores
- Relationship confidence scores
- Source document diversity

### Enhancement Factors
- Evidence consistency across sources
- Graph connectivity (how well entities are connected)
- Number of supporting relationships

### Confidence Levels
- **High (≥0.8)**: Strong evidence from multiple reliable sources
- **Medium (0.6-0.8)**: Good evidence with some uncertainty
- **Low (0.4-0.6)**: Limited evidence, verification recommended
- **Very Low (<0.4)**: Insufficient evidence, high uncertainty

## Testing

The module includes comprehensive tests:

- `test_engine.py`: Tests for query processing functionality
- `test_response_formatter.py`: Tests for response formatting and confidence scoring

Run tests with:
```bash
python -m pytest src/lightrag_integration/query/ -v
```

## Error Handling

The system includes robust error handling:

- Query validation (length, format)
- Graph loading failures
- Search and traversal errors
- Response generation errors
- Graceful degradation when components fail

## Performance Considerations

- Caches loaded knowledge graphs for efficiency
- Uses async/await for non-blocking operations
- Configurable result limits to control performance
- Efficient graph traversal algorithms
- Minimal memory footprint for large graphs

## Integration

The query engine integrates with:

- Knowledge Graph Builder for accessing stored graphs
- Entity Extractor for understanding query entities
- Main LightRAG Component for unified interface
- Existing Clinical Metabolomics Oracle response format

## Future Enhancements

Potential improvements for future versions:

- Advanced NLP for better query understanding
- Machine learning-based relevance scoring
- Caching of frequent queries
- Real-time graph updates
- Multi-language query support
- Advanced citation linking