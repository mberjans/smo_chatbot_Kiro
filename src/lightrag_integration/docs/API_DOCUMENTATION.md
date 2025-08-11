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
        print(f"Sources: {response.source_documents}")
        
    finally:
        await component.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### Integration with Chainlit

```python
import chainlit as cl
from lightrag_integration.component import LightRAGComponent
from lightrag_integration.config.settings import LightRAGConfig

component = None

@cl.on_chat_start
async def start():
    global component
    config = LightRAGConfig()
    component = LightRAGComponent(config)
    await component.initialize()
    
    await cl.Message(
        content="Welcome! I'm ready to answer questions about clinical metabolomics."
    ).send()

@cl.on_message
async def main(message: cl.Message):
    try:
        response = await component.query(message.content)
        
        # Create response with citations
        content = response.answer
        if response.citations:
            content += "\n\n**Sources:**\n"
            for i, citation in enumerate(response.citations, 1):
                content += f"{i}. {citation.title} - {citation.source}\n"
        
        await cl.Message(
            content=content,
            metadata={
                "confidence": response.confidence_score,
                "processing_time": response.processing_time,
                "entities_used": len(response.entities_used)
            }
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"I encountered an error: {str(e)}. Please try again."
        ).send()

@cl.on_stop
async def stop():
    global component
    if component:
        await component.cleanup()
```

### Advanced Usage with Error Handling

```python
import asyncio
import logging
from lightrag_integration.component import LightRAGComponent
from lightrag_integration.config.settings import LightRAGConfig
from lightrag_integration.error_handling import LightRAGError

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def robust_query_example():
    """Example showing robust error handling and monitoring"""
    config = LightRAGConfig()
    component = LightRAGComponent(config)
    
    try:
        # Initialize with health check
        await component.initialize()
        health = await component.get_health_status()
        
        if health.status != "healthy":
            logger.warning(f"System health: {health.status}")
            for component_name, status in health.components.items():
                logger.info(f"  {component_name}: {status}")
        
        # Process multiple queries with error handling
        queries = [
            "What is clinical metabolomics?",
            "How are metabolites analyzed?",
            "What are biomarkers in metabolomics?"
        ]
        
        for query in queries:
            try:
                logger.info(f"Processing query: {query}")
                response = await component.query(query)
                
                logger.info(f"Response received in {response.processing_time:.2f}s")
                logger.info(f"Confidence: {response.confidence_score:.2f}")
                logger.info(f"Sources: {len(response.source_documents)}")
                
                print(f"\nQ: {query}")
                print(f"A: {response.answer}")
                print(f"Confidence: {response.confidence_score:.2f}")
                
            except LightRAGError as e:
                logger.error(f"LightRAG error for query '{query}': {e}")
            except Exception as e:
                logger.error(f"Unexpected error for query '{query}': {e}")
        
    except Exception as e:
        logger.error(f"Component initialization failed: {e}")
    
    finally:
        await component.cleanup()

if __name__ == "__main__":
    asyncio.run(robust_query_example())
```

### Batch Processing Example

```python
import asyncio
from pathlib import Path
from lightrag_integration.ingestion.pipeline import PDFIngestionPipeline
from lightrag_integration.config.settings import LightRAGConfig

async def batch_ingestion_example():
    """Example of batch PDF processing"""
    config = LightRAGConfig()
    pipeline = PDFIngestionPipeline(config)
    
    # Process all PDFs in papers directory
    papers_dir = Path("papers")
    if papers_dir.exists():
        print(f"Processing PDFs in {papers_dir}")
        results = await pipeline.process_directory(str(papers_dir))
        
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        print(f"Processing complete: {successful} successful, {failed} failed")
        
        # Show statistics
        stats = pipeline.get_statistics()
        print(f"Total documents processed: {stats.get('total_documents', 0)}")
        print(f"Total entities extracted: {stats.get('total_entities', 0)}")
        print(f"Average processing time: {stats.get('avg_processing_time', 0):.2f}s")
    else:
        print(f"Papers directory {papers_dir} not found")

if __name__ == "__main__":
    asyncio.run(batch_ingestion_example())
```

### Monitoring and Health Checks

```python
import asyncio
from lightrag_integration.monitoring import HealthMonitor
from lightrag_integration.config.settings import LightRAGConfig

async def monitoring_example():
    """Example of system monitoring"""
    config = LightRAGConfig()
    monitor = HealthMonitor(config)
    
    # Get comprehensive health status
    health = await monitor.get_health_status()
    print(f"System Status: {health.status}")
    print(f"Uptime: {health.uptime:.2f} seconds")
    
    print("\nComponent Health:")
    for component, status in health.components.items():
        print(f"  {component}: {status}")
    
    print("\nPerformance Metrics:")
    metrics = await monitor.get_performance_metrics()
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # Check specific component
    component_health = await monitor.check_component_health("query_engine")
    print(f"\nQuery Engine Health: {component_health}")

if __name__ == "__main__":
    asyncio.run(monitoring_example())
```

### Configuration Management

```python
from lightrag_integration.config.settings import LightRAGConfig
import os

def configuration_example():
    """Example of configuration management"""
    
    # Load configuration from environment variables
    config = LightRAGConfig()
    print(f"Knowledge graph path: {config.knowledge_graph_path}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Batch size: {config.batch_size}")
    
    # Override specific settings
    custom_config = LightRAGConfig(
        batch_size=16,  # Smaller batch for limited memory
        max_concurrent_requests=5,  # Reduce concurrency
        cache_ttl_seconds=7200  # Longer cache TTL
    )
    
    print(f"\nCustom configuration:")
    print(f"Batch size: {custom_config.batch_size}")
    print(f"Max concurrent: {custom_config.max_concurrent_requests}")
    print(f"Cache TTL: {custom_config.cache_ttl_seconds}")
    
    # Environment-specific configuration
    env = os.getenv("DEPLOYMENT_ENV", "development")
    if env == "production":
        production_config = LightRAGConfig(
            batch_size=32,
            max_concurrent_requests=20,
            cache_ttl_seconds=3600
        )
        print(f"\nProduction configuration loaded")
    elif env == "development":
        dev_config = LightRAGConfig(
            batch_size=8,
            max_concurrent_requests=3,
            cache_ttl_seconds=1800
        )
        print(f"\nDevelopment configuration loaded")

if __name__ == "__main__":
    configuration_example()
```

## SDK Usage Patterns

### Synchronous Wrapper

For applications that need synchronous interfaces:

```python
import asyncio
from lightrag_integration.component import LightRAGComponent
from lightrag_integration.config.settings import LightRAGConfig

class LightRAGSync:
    """Synchronous wrapper for LightRAG component"""
    
    def __init__(self, config: LightRAGConfig = None):
        self.config = config or LightRAGConfig()
        self.component = None
        self.loop = None
    
    def __enter__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.component = LightRAGComponent(self.config)
        self.loop.run_until_complete(self.component.initialize())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.component:
            self.loop.run_until_complete(self.component.cleanup())
        if self.loop:
            self.loop.close()
    
    def query(self, question: str, context=None):
        """Synchronous query method"""
        return self.loop.run_until_complete(
            self.component.query(question, context)
        )
    
    def get_health_status(self):
        """Synchronous health check"""
        return self.loop.run_until_complete(
            self.component.get_health_status()
        )

# Usage example
def sync_usage_example():
    with LightRAGSync() as lightrag:
        response = lightrag.query("What is clinical metabolomics?")
        print(f"Answer: {response.answer}")
        
        health = lightrag.get_health_status()
        print(f"System health: {health.status}")

if __name__ == "__main__":
    sync_usage_example()
```

This API documentation provides comprehensive coverage of all LightRAG integration components and their usage patterns, including advanced examples for production use cases.