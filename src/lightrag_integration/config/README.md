# LightRAG Configuration System

This module provides a comprehensive configuration system for the LightRAG integration component. It supports loading configuration from environment variables, JSON files, and YAML files, with environment variables taking precedence.

## Features

- **Environment Variable Support**: All configuration options can be set via environment variables
- **Config File Support**: Load configuration from JSON or YAML files
- **Validation**: Comprehensive validation of all configuration parameters
- **Default Values**: Sensible defaults for all configuration options
- **Security**: Automatic redaction of sensitive information (API keys) in logs and string representations
- **Path Management**: Automatic creation of required directories
- **Type Safety**: Strong typing with dataclass implementation

## Configuration Options

### Storage Configuration
- `knowledge_graph_path`: Path to store the knowledge graph data (default: `./data/lightrag_kg`)
- `vector_store_path`: Path to store vector embeddings (default: `./data/lightrag_vectors`)
- `cache_directory`: Path for caching data (default: `./data/lightrag_cache`)
- `papers_directory`: Directory containing PDF papers to ingest (default: `./papers`)

### Processing Configuration
- `chunk_size`: Size of text chunks for processing (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `max_entities_per_chunk`: Maximum entities to extract per chunk (default: 50)

### Model Configuration
- `embedding_model`: Model for text embeddings (default: `intfloat/e5-base-v2`)
- `llm_model`: Language model for processing (default: `groq:Llama-3.3-70b-Versatile`)

### Performance Configuration
- `batch_size`: Batch size for processing (default: 32)
- `max_concurrent_requests`: Maximum concurrent requests (default: 10)
- `cache_ttl_seconds`: Cache time-to-live in seconds (default: 3600)

### API Keys and Credentials
- `groq_api_key`: API key for Groq models (from `GROQ_API_KEY` env var)
- `openai_api_key`: API key for OpenAI models (from `OPENAI_API_KEY` env var)

### Logging Configuration
- `log_level`: Logging level (default: `INFO`)
- `log_file`: Path to log file (auto-generated if not specified)

### Feature Flags
- `enable_entity_extraction`: Enable entity extraction (default: `true`)
- `enable_relationship_extraction`: Enable relationship extraction (default: `true`)
- `enable_caching`: Enable caching (default: `true`)

### Advanced Processing Options
- `min_entity_confidence`: Minimum confidence for entity extraction (default: 0.7)
- `min_relationship_confidence`: Minimum confidence for relationship extraction (default: 0.6)
- `max_document_size_mb`: Maximum document size in MB (default: 50)

### Query Processing Configuration
- `max_query_length`: Maximum query length (default: 1000)
- `default_top_k`: Default number of results to return (default: 10)
- `similarity_threshold`: Similarity threshold for retrieval (default: 0.5)

## Environment Variables

All configuration options can be set using environment variables with the `LIGHTRAG_` prefix:

```bash
export LIGHTRAG_CHUNK_SIZE=1500
export LIGHTRAG_CHUNK_OVERLAP=300
export LIGHTRAG_EMBEDDING_MODEL="intfloat/e5-large-v2"
export LIGHTRAG_LOG_LEVEL="DEBUG"
export LIGHTRAG_ENABLE_CACHING="false"
export GROQ_API_KEY="your-groq-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

## Usage Examples

### Basic Usage (Environment Variables)

```python
from lightrag_integration.config.settings import LightRAGConfig

# Load configuration from environment variables
config = LightRAGConfig.from_env()
config.validate()
```

### Loading from JSON File

```python
from lightrag_integration.config.settings import LightRAGConfig

# Load from JSON file (environment variables still take precedence)
config = LightRAGConfig.from_file("config.json")
config.validate()
```

### Loading from YAML File

```python
from lightrag_integration.config.settings import LightRAGConfig

# Load from YAML file
config = LightRAGConfig.from_file("config.yaml")
config.validate()
```

### Creating from Dictionary

```python
from lightrag_integration.config.settings import LightRAGConfig

config_dict = {
    "chunk_size": 1500,
    "embedding_model": "custom-model",
    "enable_caching": False
}

config = LightRAGConfig.from_dict(config_dict)
config.validate()
```

### Using with LightRAG Component

```python
from lightrag_integration.config.settings import LightRAGConfig
from lightrag_integration.component import LightRAGComponent

# Load configuration
config = LightRAGConfig.from_file("my_config.yaml")

# Create component with configuration
component = LightRAGComponent(config)
```

## Configuration File Examples

### JSON Configuration

```json
{
  "chunk_size": 1500,
  "chunk_overlap": 300,
  "embedding_model": "intfloat/e5-large-v2",
  "llm_model": "groq:Llama-3.3-70b-Versatile",
  "batch_size": 64,
  "log_level": "DEBUG",
  "enable_caching": true,
  "min_entity_confidence": 0.8
}
```

### YAML Configuration

```yaml
# LightRAG Configuration
chunk_size: 1500
chunk_overlap: 300
embedding_model: "intfloat/e5-large-v2"
llm_model: "groq:Llama-3.3-70b-Versatile"
batch_size: 64
log_level: "DEBUG"
enable_caching: true
min_entity_confidence: 0.8
```

## Validation

The configuration system includes comprehensive validation:

- **Range Validation**: Ensures numeric values are within valid ranges
- **Type Validation**: Ensures values are of the correct type
- **Dependency Validation**: Ensures required API keys are present for specific models
- **Path Validation**: Ensures paths are not empty
- **Confidence Validation**: Ensures confidence thresholds are between 0.0 and 1.0

```python
config = LightRAGConfig()
try:
    config.validate()
    print("Configuration is valid")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Security Features

- **API Key Redaction**: API keys are automatically redacted in logs and string representations
- **Safe Serialization**: The `save_to_file()` method redacts sensitive information
- **Environment Priority**: Environment variables take precedence over config files for security

## Testing

The configuration system includes comprehensive unit tests:

```bash
python -m pytest src/lightrag_integration/config/test_settings.py -v
```

## Advanced Features

### Getting Effective Configuration

```python
config = LightRAGConfig()
effective_config = config.get_effective_config()
# Returns configuration with resolved absolute paths
```

### Saving Configuration

```python
config = LightRAGConfig()
config.save_to_file("output_config.json", format="json")
config.save_to_file("output_config.yaml", format="yaml")
```

### Configuration Inspection

```python
config = LightRAGConfig()
print(config)  # Prints configuration with redacted sensitive data
config_dict = config.to_dict()  # Convert to dictionary
```

## Integration with Existing System

The configuration system is designed to be consistent with the existing Clinical Metabolomics Oracle codebase:

- Uses environment variables like the existing system
- Follows the same logging patterns
- Integrates with the existing directory structure
- Maintains compatibility with existing API key management