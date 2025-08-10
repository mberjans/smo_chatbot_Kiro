# Technology Stack

## Core Framework
- **Chainlit**: Web-based chat interface framework (v1.0.401+)
- **FastAPI**: Backend API framework for mounting Chainlit
- **Python**: Primary programming language

## Database & Data Layer
- **PostgreSQL**: Primary database with Prisma ORM
- **Prisma**: Database ORM with async support and migrations
- **Neo4j**: Graph database for knowledge graph storage and retrieval

## AI/ML Stack
- **LlamaIndex**: Core framework for RAG (Retrieval Augmented Generation)
- **Sentence Transformers**: Text embeddings (default: intfloat/e5-base-v2)
- **HuggingFace**: Alternative embedding models
- **Ollama**: Local LLM inference
- **Groq**: Fast LLM inference (default: Llama-3.3-70b-Versatile)
- **OpenAI/OpenRouter**: Alternative LLM providers

## Translation & Language
- **Lingua**: Language detection
- **Deep Translator**: Google Translate integration
- **OPUS-MT**: Alternative translation models

## External APIs
- **Perplexity AI**: Enhanced search and citation capabilities

## Common Commands

### Database Operations
```bash
# Generate Prisma client
npx prisma generate

# Run database migrations
npx prisma migrate dev

# Reset database
npx prisma migrate reset
```

### Development
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node.js dependencies
npm install

# Run the application
python src/main.py

# Run with FastAPI mounting
python src/app.py
```

### Environment Setup
Required environment variables:
- `DATABASE_URL`: PostgreSQL connection string
- `NEO4J_PASSWORD`: Neo4j database password
- `GROQ_API_KEY`: Groq API key
- `OPENAI_API_KEY`: OpenAI API key (optional)
- `PERPLEXITY_API`: Perplexity AI API key