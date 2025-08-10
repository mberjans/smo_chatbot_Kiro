# LightRAG Integration API Documentation

## Overview

The LightRAG Integration provides a comprehensive system for integrating LightRAG (Light Retrieval-Augmented Generation) capabilities into the Clinical Metabolomics Oracle. This documentation covers all public APIs, configuration options, and integration patterns.

## Table of Contents

1. [Core Components](#core-components)
2. [Configuration](#configuration)
3. [Ingestion Pipeline](#ingestion-pipeline)
4. [Query Engine](#query-engine)
5. [Response Integration](#response-integration)
6. [Routing System](#routing-system)
7. [Monitoring and Health](#monitoring-and-health)
8. [Error Handling](#error-handling)
9. [Maintenance](#maintenance)
10. [Testing](#testing)

## Core Components

### LightRAGComponent

The main component class that encapsulates all LightRAG functionality.

```python
from lightrag_integration.component import LightRAGComponent
from lightrag_integration.config.settings import LightRAGConfig

# Initialize component
config = LightRAGConfig()
component = LightRAGComponent(config)

# Initialize the component
await component.initialize()

# Query the system
response = await component.query("What is clinical metabolomics?")

# Get health status
health = await component.get_health_status()

# Cleanup
await component.cleanup()
```

#### Methods

##### `__init__(config: LightRAGConfig)`
Initialize the LightRAG component with configuration.

**Parameters:**
- `config`: LightRAGConfig instance with system configuration

##### `async initialize() -> None`
Initialize the component and all subsystems.

**Raises:**
- `LightRAGInitializationError`: If initialization fails

##### `async query(question: str, context: Optional[Dict] = None) -> LightRAGResponse`
Query the LightRAG system.

**Parameters:**
- `question`: The query string
- `context`: Optional context dictionary

**Returns:**
- `LightRAGResponse`: Response object with answer, confidence, and metadata

**Raises:**
- `LightRAGQueryError`: If query processing fails

##### `async get_health_status() -> HealthStatus`
Get current system health and statistics.

**Returns:**
- `HealthStatus`: Current system health information

##### `async cleanup() -> None`
Clean up resources and connections.

### LightRAGResponse

Response object returned by query operations.

```python
@dataclass
class LightRAGResponse:
    answer: str                          # Generated answer
    confidence_score: float              # Confidence score (0.0-1.0)
    source_documents: List[str]          # Source document paths
    entities_used: List[Entity]          # Entities used in response
    relationships_used: List[Relationship] # Relationships used
    processing_time: float               # Processing time in seconds
    metadata: Dict[str, Any]             # Additional metadata
    citations: List[Citation]            # Citation information
```

## Configuration

### LightRAGConfig

Main configuration class for the LightRAG system.

```python
from lightrag_integration.config.settings import LightRAGConfig

config = LightRAGConfig(
    # Storage paths
    knowledge_graph_path="data/lightrag_kg",
    vector_store_path="data/lightrag_vectors",
    cache_directory="data/lightrag_cache",
    
    # Processing configuration
    chunk_size=1000,
    chunk_overlap=200,
    max_entities_per_chunk=50,
    
    # Model configuration
    embedding_model="intfloat/e5-base-v2",
    llm_model="groq:Llama-3.3-70b-Versatile",
    
    # Performance configuration
    batch_size=32,
    max_concurrent_requests=10,
    cache_ttl_seconds=3600
)
```

#### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `knowledge_graph_path` | str | "data/lightrag_kg" | Path to knowledge graph storage |
| `vector_store_path` | str | "data/lightrag_vectors" | Path to vector embeddings |
| `cache_directory` | str | "data/lightrag_cache" | Cache directory path |
| `chunk_size` | int | 1000 | Text chunk size for processing |
| `chunk_overlap` | int | 200 | Overlap between chunks |
| `max_entities_per_chunk` | int | 50 | Maximum entities per chunk |
| `embedding_model` | str | "intfloat/e5-base-v2" | Embedding model name |
| `llm_model` | str | "groq:Llama-3.3-70b-Versatile" | LLM model for processing |
| `batch_size` | int | 32 | Batch size for processing |
| `max_concurrent_requests` | int | 10 | Maximum concurrent requests |
| `cache_ttl_seconds` | int | 3600 | Cache TTL in seconds |

## Ingestion Pipeline

### PDFIngestionPipeline

Handles PDF document ingestion and processing.

```python
from lightrag_integration.ingestion.pipeline import PDFIngestionPipeline

pipeline = PDFIngestionPipeline(config)

# Process a directory of PDFs
results = await pipeline.process_directory("papers/")

# Process a single PDF
result = await pipeline.process_file("paper.pdf")

# Get processing statistics
stats = pipeline.get_statistics()
```

#### Methods

##### `async process_directory(directory_path: str) -> List[ProcessResult]`
Process all PDF files in a directory.

**Parameters:**
- `directory_path`: Path to directory containing PDFs

**Returns:**
- `List[ProcessResult]`: Results for each processed file

##### `async process_file(pdf_path: str) -> ProcessResult`
Process a single PDF file.

**Parameters:**
- `pdf_path`: Path to PDF file

**Returns:**
- `ProcessResult`: Processing result with metadata

### DirectoryMonitor

Monitors directories for new PDF files and automatically processes them.

```python
from lightrag_integration.ingestion.directory_monitor import DirectoryMonitor

monitor = DirectoryMonitor(
    watch_directory="papers/",
    pipeline=pipeline,
    check_interval=60  # Check every 60 seconds
)

# Start monitoring
await monitor.start()

# Stop monitoring
await monitor.stop()
```

## Query Engine

### QueryEngine

Handles query processing and response generation.

```python
from lightrag_integration.query.engine import QueryEngine

engine = QueryEngine(config)

# Process a query
response = await engine.process_query(
    query="What is clinical metabolomics?",
    context={"user_id": "user123"}
)

# Get query statistics
stats = engine.get_query_statistics()
```

#### Methods

##### `async process_query(query: str, context: Optional[Dict] = None) -> QueryResponse`
Process a query and generate response.

**Parameters:**
- `query`: Query string
- `context`: Optional context information

**Returns:**
- `QueryResponse`: Generated response with metadata

## Response Integration

### ResponseIntegrator

Integrates LightRAG responses with existing system features.

```python
from lightrag_integration.response_integration import ResponseIntegrator

integrator = ResponseIntegrator(config)

# Process LightRAG response for integration
processed = await integrator.process_lightrag_response(lightrag_response)

# Apply translation
translated = await integrator.apply_translation(processed, "es")

# Generate citations
cited = await integrator.generate_citations(processed)
```

## Routing System

### QueryRouter

Routes queries to appropriate systems (LightRAG vs Perplexity).

```python
from lightrag_integration.routing.router import QueryRouter

router = QueryRouter(config)

# Classify a query
classification = await router.classify_query("What is clinical metabolomics?")

# Route a query
response = await router.route_query("What is clinical metabolomics?")

# Get routing metrics
metrics = router.get_routing_metrics()
```

## Monitoring and Health

### HealthMonitor

Monitors system health and performance.

```python
from lightrag_integration.monitoring import HealthMonitor

monitor = HealthMonitor(config)

# Get current health status
health = await monitor.get_health_status()

# Get performance metrics
metrics = await monitor.get_performance_metrics()

# Check component health
component_health = await monitor.check_component_health("query_engine")
```

### HealthStatus

Health status information.

```python
@dataclass
class HealthStatus:
    status: str                    # "healthy", "degraded", "unhealthy"
    components: Dict[str, str]     # Component health status
    metrics: Dict[str, float]      # Performance metrics
    errors: List[str]              # Recent errors
    uptime: float                  # System uptime in seconds
    last_check: datetime           # Last health check time
```

## Error Handling

### Error Types

The system defines several custom exception types:

```python
from lightrag_integration.error_handling import (
    LightRAGError,
    LightRAGInitializationError,
    LightRAGQueryError,
    LightRAGIngestionError,
    LightRAGConfigurationError
)

try:
    response = await component.query("test query")
except LightRAGQueryError as e:
    print(f"Query failed: {e}")
    # Handle query error
except LightRAGError as e:
    print(f"General LightRAG error: {e}")
    # Handle general error
```

### ErrorHandler

Centralized error handling and recovery.

```python
from lightrag_integration.error_handling import ErrorHandler

handler = ErrorHandler(config)

# Handle PDF processing error
result = await handler.handle_pdf_error(error, "document.pdf")

# Handle query error with fallback
response = await handler.handle_query_error(error, "query")
```

## Maintenance

### UpdateSystem

Handles knowledge base updates and maintenance.

```python
from lightrag_integration.maintenance.update_system import UpdateSystem

updater = UpdateSystem(config)

# Update knowledge base with new documents
result = await updater.update_knowledge_base(["new_paper.pdf"])

# Get update history
history = await updater.get_update_history()

# Rollback to previous version
await updater.rollback_to_version("v1.2.3")
```

### AdminInterface

Administrative interface for system management.

```python
from lightrag_integration.maintenance.admin_interface import AdminInterface

admin = AdminInterface(config)

# Get system status
status = await admin.get_system_status()

# Manage documents
await admin.add_document("new_paper.pdf")
await admin.remove_document("old_paper.pdf")

# Get system metrics
metrics = await admin.get_system_metrics()
```

## Testing

### Test Utilities

The system provides comprehensive testing utilities:

```python
from lightrag_integration.testing.test_runner import TestRunner

runner = TestRunner(config)

# Run all tests
results = await runner.run_all_tests()

# Run specific test suite
results = await runner.run_test_suite("ingestion")

# Run performance tests
results = await runner.run_performance_tests()
```

## Environment Variables

The following environment variables can be used to configure the system:

| Variable | Description | Default |
|----------|-------------|---------|
| `LIGHTRAG_KG_PATH` | Knowledge graph storage path | "data/lightrag_kg" |
| `LIGHTRAG_VECTOR_PATH` | Vector store path | "data/lightrag_vectors" |
| `LIGHTRAG_CACHE_PATH` | Cache directory path | "data/lightrag_cache" |
| `LIGHTRAG_EMBEDDING_MODEL` | Embedding model name | "intfloat/e5-base-v2" |
| `LIGHTRAG_LLM_MODEL` | LLM model name | "groq:Llama-3.3-70b-Versatile" |
| `LIGHTRAG_BATCH_SIZE` | Processing batch size | "32" |
| `LIGHTRAG_MAX_CONCURRENT` | Max concurrent requests | "10" |
| `LIGHTRAG_CACHE_TTL` | Cache TTL in seconds | "3600" |

## Error Codes

The system uses standardized error codes:

| Code | Description |
|------|-------------|
| `LIGHTRAG_001` | Initialization error |
| `LIGHTRAG_002` | Configuration error |
| `LIGHTRAG_003` | PDF processing error |
| `LIGHTRAG_004` | Query processing error |
| `LIGHTRAG_005` | Knowledge graph error |
| `LIGHTRAG_006` | Vector store error |
| `LIGHTRAG_007` | Cache error |
| `LIGHTRAG_008` | Translation error |
| `LIGHTRAG_009` | Citation error |
| `LIGHTRAG_010` | Routing error |

## Best Practices

1. **Initialization**: Always call `initialize()` before using the component
2. **Error Handling**: Use try-catch blocks for all async operations
3. **Resource Management**: Call `cleanup()` when done with the component
4. **Configuration**: Use environment variables for production configuration
5. **Monitoring**: Regularly check health status in production
6. **Testing**: Run comprehensive tests before deployment
7. **Updates**: Use the update system for knowledge base maintenance

## Examples

### Basic Usage

```python
import asyncio
from lightrag_integration.component import LightRAGComponent
from lightrag_integration.config.settings import LightRAGConfig

async def main():
    # Initialize
    config = LightRAGConfig()
    component = LightRAGComponent(config)
    
    try:
        await component.initialize()
        
        # Query the system
        response = await component.query("What is clinical metabolomics?")
        print(f"Answer: {response.answer}")
        print(f"Confidence: {response.confidence_score}")
        
    finally:
        await component.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Integration with Chainlit

```python
import chainlit as cl
from lightrag_integration.component import LightRAGComponent

component = None

@cl.on_chat_start
async def start():
    global component
    config = LightRAGConfig()
    component = LightRAGComponent(config)
    await component.initialize()

@cl.on_message
async def main(message: cl.Message):
    response = await component.query(message.content)
    await cl.Message(content=response.answer).send()
```

This API documentation provides comprehensive coverage of all LightRAG integration components and their usage patterns.