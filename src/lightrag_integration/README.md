# LightRAG Integration

This module provides integration between LightRAG and the Clinical Metabolomics Oracle system.

## Setup

### Dependencies

The following dependencies have been added to `requirements.txt`:

- `lightrag-hku==1.4.7` - Core LightRAG library from HKUDS
- `PyMuPDF==1.24.1` - PDF processing
- `pymupdf4llm==0.0.17` - PDF to markdown conversion
- `scikit-learn==1.4.1.post1` - Machine learning utilities
- `spacy==3.7.4` - NLP processing for entity extraction

### Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Directory Structure

```
src/lightrag_integration/
├── __init__.py                 # Main module exports
├── component.py               # Main LightRAG component
├── config/
│   ├── __init__.py
│   └── settings.py           # Configuration management
├── ingestion/
│   ├── __init__.py
│   └── pipeline.py           # PDF ingestion pipeline (placeholder)
├── query/
│   ├── __init__.py
│   └── engine.py             # Query processing engine (placeholder)
└── utils/
    ├── __init__.py
    ├── logging.py            # Logging utilities
    └── health.py             # Health monitoring
```

### Configuration

The system uses environment variables for configuration:

- `LIGHTRAG_KG_PATH` - Knowledge graph storage path (default: `./data/lightrag_kg`)
- `LIGHTRAG_VECTOR_PATH` - Vector store path (default: `./data/lightrag_vectors`)
- `LIGHTRAG_CACHE_DIR` - Cache directory (default: `./data/lightrag_cache`)
- `LIGHTRAG_PAPERS_DIR` - Papers directory (default: `./papers`)
- `LIGHTRAG_EMBEDDING_MODEL` - Embedding model (default: `intfloat/e5-base-v2`)
- `LIGHTRAG_LLM_MODEL` - LLM model (default: `groq:Llama-3.3-70b-Versatile`)

### Testing

Run the setup test to verify everything is working:

```bash
python test_lightrag_setup.py
```

## Usage

```python
from lightrag_integration import LightRAGComponent, LightRAGConfig

# Initialize component
config = LightRAGConfig.from_env()
component = LightRAGComponent(config)
await component.initialize()

# Check health
health = await component.get_health_status()
print(f"System status: {health.overall_status.value}")

# Process documents (placeholder)
result = await component.ingest_documents()

# Query system (placeholder)
response = await component.query("What is clinical metabolomics?")
```

## Implementation Status

- ✅ **Task 1**: Development environment and dependencies setup
- ⏳ **Task 2**: Core LightRAG configuration system (next)
- ⏳ **Task 3**: PDF ingestion pipeline
- ⏳ **Task 4**: Query processing engine
- ⏳ **Task 5+**: Additional features and integration

## Next Steps

1. Implement core LightRAG configuration system (Task 2)
2. Develop PDF ingestion pipeline (Task 3)
3. Create query processing engine (Task 4)
4. Integrate with existing Chainlit interface (Task 8)