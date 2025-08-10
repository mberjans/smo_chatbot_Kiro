# Project Structure

## Root Directory Layout
```
├── src/                    # Main application source code
├── current/                # Duplicate/backup of src (legacy)
├── prisma/                 # Database schema and migrations
├── .kiro/                  # Kiro AI assistant configuration
├── package.json            # Node.js dependencies (Prisma)
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

## Source Code Organization (`src/`)

### Core Application Files
- `main.py` - Primary Chainlit application entry point
- `app.py` - FastAPI application with Chainlit mounting
- `pipelines.py` - RAG pipeline configuration and LLM setup
- `callbacks.py` - Custom LlamaIndex callback handlers

### AI/ML Components
- `embeddings.py` - Custom SentenceTransformer embedding implementations
- `query_engine.py` - Custom citation query engine
- `retrievers.py` - Knowledge graph RAG retrievers
- `graph_stores.py` - Neo4j graph store implementations
- `index.py` - Document indexing utilities

### Language & Translation
- `translation.py` - Multi-language support and translation
- `translators/` - Translation model implementations
  - `llm.py` - LLM-based translation
  - `opusmt.py` - OPUS-MT model integration
- `lingua_iso_codes.py` - Language code mappings

### Data Processing
- `reader.py` - Document readers and parsers
- `textualize.py` - Text processing utilities
- `citation.py` - Citation processing and formatting

### UI & Configuration
- `.chainlit/` - Chainlit framework configuration
  - `config.toml` - Main Chainlit settings
  - `translations/` - UI translations for multiple languages
- `public/` - Static assets (logos, custom JS/CSS)
- `.files/` - User-uploaded files storage

## Database Structure (`prisma/`)
- `schema.prisma` - Database schema definition
- `migrations/` - Database migration files
  - Auto-generated migration folders with SQL files
  - `migration_lock.toml` - Migration lock file

## Key Architectural Patterns

### RAG Pipeline Flow
1. **Query Processing** → Language detection → Translation (if needed)
2. **Retrieval** → Knowledge graph traversal → Entity extraction
3. **Generation** → LLM response → Citation formatting
4. **Post-processing** → Translation back → Response delivery

### Database Models
- `Thread` - Conversation sessions with metadata and tags
- `Step` - Individual conversation steps (user/assistant messages)
- `Element` - File attachments and rich content
- `User` - User authentication and session management
- `Feedback` - User feedback on responses

### Configuration Patterns
- Environment variables for API keys and database connections
- Modular LLM provider switching (Groq/OpenAI/Ollama)
- Configurable embedding models and batch sizes
- Multi-language translation pipeline