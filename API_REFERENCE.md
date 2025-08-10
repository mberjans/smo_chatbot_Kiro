# Clinical Metabolomics Oracle - API Reference

## Overview

The Clinical Metabolomics Oracle (CMO) provides both a web interface and programmatic API access for querying clinical metabolomics research information. This document describes the available endpoints, request/response formats, and integration patterns.

## Base URL

- **Development**: `http://localhost:8001`
- **Production**: `https://your-domain.com`

## Authentication

### Web Interface Authentication
The web interface uses simple credential-based authentication:

- **Admin Access**: `admin` / `admin123`
- **Testing Access**: `testing` / `ku9R_3`

### API Authentication
Currently, the API uses the same session-based authentication as the web interface. Future versions will include API key authentication.

## Endpoints

### Health and Status

#### GET /health
Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-09T10:30:00Z",
  "version": "1.1.0"
}
```

#### GET /health/detailed
Comprehensive health check with component status.

**Response:**
```json
{
  "overall_status": "healthy",
  "timestamp": "2025-01-09T10:30:00Z",
  "components": {
    "initialization": {
      "status": "healthy",
      "message": "Component initialized successfully",
      "last_check": "2025-01-09T10:30:00Z",
      "metrics": {
        "initialization_time": "2025-01-09T10:25:00Z",
        "uptime_seconds": 300
      }
    },
    "database": {
      "status": "healthy",
      "message": "Database connection active",
      "last_check": "2025-01-09T10:30:00Z",
      "metrics": {
        "connection_pool_size": 10,
        "active_connections": 2
      }
    },
    "lightrag": {
      "status": "healthy",
      "message": "LightRAG component operational",
      "last_check": "2025-01-09T10:30:00Z",
      "metrics": {
        "documents_indexed": 5,
        "knowledge_graph_nodes": 1250
      }
    },
    "openrouter": {
      "status": "healthy",
      "message": "OpenRouter integration active",
      "last_check": "2025-01-09T10:30:00Z",
      "metrics": {
        "api_key_valid": true,
        "available_models": 4
      }
    }
  }
}
```

#### GET /metrics
Prometheus-compatible metrics endpoint.

**Response:**
```
# HELP queries_total Total number of queries processed
# TYPE queries_total counter
queries_total 1234

# HELP queries_successful Number of successful queries
# TYPE queries_successful counter
queries_successful 1180

# HELP query_processing_duration_seconds Query processing time
# TYPE query_processing_duration_seconds histogram
query_processing_duration_seconds_bucket{le="0.5"} 450
query_processing_duration_seconds_bucket{le="1.0"} 890
query_processing_duration_seconds_bucket{le="2.0"} 1150
query_processing_duration_seconds_bucket{le="5.0"} 1220
query_processing_duration_seconds_bucket{le="+Inf"} 1234

# HELP system_memory_usage_percent System memory usage percentage
# TYPE system_memory_usage_percent gauge
system_memory_usage_percent 45.2

# HELP cache_hit_rate Cache hit rate percentage
# TYPE cache_hit_rate gauge
cache_hit_rate 72.5
```

### Query Processing

#### POST /api/query
Submit a query to the CMO system.

**Request:**
```json
{
  "question": "What are the main applications of metabolomics in clinical research?",
  "context": {
    "user_id": "user123",
    "session_id": "session456"
  },
  "options": {
    "include_citations": true,
    "max_response_length": 2000,
    "confidence_threshold": 0.3
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "answer": "Clinical metabolomics has several key applications including biomarker discovery, disease diagnosis, drug development, and personalized medicine. It provides insights into metabolic pathways and helps identify disease-specific metabolic signatures that can be used for early detection and treatment monitoring.",
    "confidence_score": 0.75,
    "source_documents": [
      "Clinical metabolomics review - Section 3.2",
      "Biomarker discovery in metabolomics - Abstract"
    ],
    "entities_used": [
      "metabolomics",
      "biomarker discovery",
      "clinical diagnosis",
      "personalized medicine"
    ],
    "relationships_used": [
      "metabolomics -> enables -> biomarker discovery",
      "biomarkers -> support -> clinical diagnosis"
    ],
    "processing_time": 1.25,
    "metadata": {
      "source": "LightRAG Knowledge Base",
      "model_used": "lightrag",
      "fallback_used": false,
      "cache_hit": false
    },
    "citations": [
      {
        "source": "Clinical Metabolomics Review 2024",
        "confidence": 0.8,
        "relevance": 0.9
      },
      {
        "source": "Biomarker Discovery Methods",
        "confidence": 0.7,
        "relevance": 0.85
      }
    ]
  },
  "request_id": "req_123456789",
  "timestamp": "2025-01-09T10:30:00Z"
}
```

**Error Response:**
```json
{
  "success": false,
  "error": {
    "code": "QUERY_PROCESSING_FAILED",
    "message": "Unable to process query due to system unavailability",
    "details": {
      "primary_error": "LightRAG component unavailable",
      "fallback_attempted": true,
      "fallback_result": "partial_success"
    }
  },
  "request_id": "req_123456789",
  "timestamp": "2025-01-09T10:30:00Z"
}
```

#### GET /api/query/{request_id}
Get the status or result of a previously submitted query.

**Response:**
```json
{
  "request_id": "req_123456789",
  "status": "completed",
  "submitted_at": "2025-01-09T10:29:30Z",
  "completed_at": "2025-01-09T10:30:00Z",
  "result": {
    // Same structure as POST /api/query response
  }
}
```

### Document Management

#### POST /api/documents/ingest
Ingest new PDF documents into the knowledge base.

**Request (multipart/form-data):**
```
Content-Type: multipart/form-data

file: [PDF file]
metadata: {
  "title": "Clinical Metabolomics Research Paper",
  "authors": ["Dr. Smith", "Dr. Johnson"],
  "publication_date": "2024-12-01",
  "journal": "Journal of Clinical Metabolomics"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "document_id": "doc_123456",
    "filename": "clinical_metabolomics_paper.pdf",
    "processing_status": "completed",
    "extracted_text_length": 45678,
    "entities_extracted": 234,
    "relationships_created": 156,
    "processing_time": 45.2,
    "metadata": {
      "title": "Clinical Metabolomics Research Paper",
      "authors": ["Dr. Smith", "Dr. Johnson"],
      "publication_date": "2024-12-01",
      "journal": "Journal of Clinical Metabolomics"
    }
  },
  "timestamp": "2025-01-09T10:30:00Z"
}
```

#### GET /api/documents
List ingested documents.

**Query Parameters:**
- `limit`: Number of documents to return (default: 50)
- `offset`: Offset for pagination (default: 0)
- `status`: Filter by processing status (all, completed, processing, failed)

**Response:**
```json
{
  "success": true,
  "data": {
    "documents": [
      {
        "document_id": "doc_123456",
        "filename": "clinical_metabolomics_paper.pdf",
        "status": "completed",
        "ingested_at": "2025-01-09T09:00:00Z",
        "metadata": {
          "title": "Clinical Metabolomics Research Paper",
          "authors": ["Dr. Smith", "Dr. Johnson"]
        }
      }
    ],
    "total_count": 5,
    "limit": 50,
    "offset": 0
  }
}
```

#### GET /api/documents/{document_id}
Get details about a specific document.

**Response:**
```json
{
  "success": true,
  "data": {
    "document_id": "doc_123456",
    "filename": "clinical_metabolomics_paper.pdf",
    "status": "completed",
    "ingested_at": "2025-01-09T09:00:00Z",
    "processing_details": {
      "text_extraction_time": 12.5,
      "entity_extraction_time": 18.3,
      "graph_construction_time": 14.4,
      "total_processing_time": 45.2
    },
    "statistics": {
      "extracted_text_length": 45678,
      "entities_extracted": 234,
      "relationships_created": 156,
      "pages_processed": 12
    },
    "metadata": {
      "title": "Clinical Metabolomics Research Paper",
      "authors": ["Dr. Smith", "Dr. Johnson"],
      "publication_date": "2024-12-01",
      "journal": "Journal of Clinical Metabolomics"
    }
  }
}
```

### System Configuration

#### GET /api/config
Get current system configuration (admin only).

**Response:**
```json
{
  "success": true,
  "data": {
    "system_info": {
      "version": "1.1.0",
      "environment": "development",
      "uptime_seconds": 3600
    },
    "ai_backends": {
      "lightrag": {
        "enabled": true,
        "status": "healthy",
        "documents_indexed": 5
      },
      "openrouter": {
        "enabled": true,
        "status": "healthy",
        "api_key_configured": true,
        "available_models": ["perplexity/sonar-pro"]
      },
      "fallback": {
        "enabled": true,
        "status": "healthy"
      }
    },
    "performance": {
      "max_concurrent_requests": 10,
      "cache_ttl_seconds": 3600,
      "query_timeout_seconds": 30
    }
  }
}
```

#### PUT /api/config
Update system configuration (admin only).

**Request:**
```json
{
  "performance": {
    "max_concurrent_requests": 15,
    "cache_ttl_seconds": 7200
  },
  "ai_backends": {
    "openrouter": {
      "default_model": "perplexity/sonar-pro"
    }
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Configuration updated successfully",
  "updated_fields": [
    "performance.max_concurrent_requests",
    "performance.cache_ttl_seconds",
    "ai_backends.openrouter.default_model"
  ]
}
```

## WebSocket API

### /ws/chat
Real-time chat interface via WebSocket.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8001/ws/chat');
```

**Message Format (Client to Server):**
```json
{
  "type": "query",
  "data": {
    "question": "What is metabolomics?",
    "session_id": "session123"
  }
}
```

**Message Format (Server to Client):**
```json
{
  "type": "response",
  "data": {
    "answer": "Metabolomics is the scientific study of chemical processes...",
    "confidence_score": 0.8,
    "processing_time": 1.5,
    "source": "LightRAG Knowledge Base"
  },
  "request_id": "req_123456"
}
```

**Status Messages:**
```json
{
  "type": "status",
  "data": {
    "status": "processing",
    "message": "Searching knowledge base..."
  },
  "request_id": "req_123456"
}
```

## Error Codes

### HTTP Status Codes
- `200 OK`: Request successful
- `400 Bad Request`: Invalid request format or parameters
- `401 Unauthorized`: Authentication required
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: System temporarily unavailable

### Application Error Codes
- `QUERY_PROCESSING_FAILED`: Query processing encountered an error
- `DOCUMENT_INGESTION_FAILED`: Document ingestion failed
- `INVALID_DOCUMENT_FORMAT`: Unsupported document format
- `SYSTEM_UNAVAILABLE`: Core system components unavailable
- `RATE_LIMIT_EXCEEDED`: Too many requests from user
- `CONFIGURATION_ERROR`: System configuration error
- `AUTHENTICATION_FAILED`: Invalid credentials
- `INSUFFICIENT_PERMISSIONS`: User lacks required permissions

## Rate Limiting

### Default Limits
- **Queries**: 100 per hour per user
- **Document Ingestion**: 10 per hour per user
- **Health Checks**: 1000 per hour per IP
- **Configuration Changes**: 10 per hour per admin user

### Rate Limit Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1641811200
X-RateLimit-Retry-After: 3600
```

## SDK Examples

### Python SDK
```python
import requests
import json

class CMOClient:
    def __init__(self, base_url="http://localhost:8001", auth=None):
        self.base_url = base_url
        self.session = requests.Session()
        if auth:
            self.session.auth = auth
    
    def query(self, question, **options):
        """Submit a query to the CMO system."""
        response = self.session.post(
            f"{self.base_url}/api/query",
            json={
                "question": question,
                "options": options
            }
        )
        return response.json()
    
    def health_check(self):
        """Check system health."""
        response = self.session.get(f"{self.base_url}/health")
        return response.json()
    
    def ingest_document(self, file_path, metadata=None):
        """Ingest a PDF document."""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'metadata': json.dumps(metadata or {})}
            response = self.session.post(
                f"{self.base_url}/api/documents/ingest",
                files=files,
                data=data
            )
        return response.json()

# Usage example
client = CMOClient()
result = client.query("What is metabolomics?")
print(result['data']['answer'])
```

### JavaScript SDK
```javascript
class CMOClient {
    constructor(baseUrl = 'http://localhost:8001') {
        this.baseUrl = baseUrl;
    }
    
    async query(question, options = {}) {
        const response = await fetch(`${this.baseUrl}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question,
                options
            })
        });
        return response.json();
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return response.json();
    }
    
    async ingestDocument(file, metadata = {}) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('metadata', JSON.stringify(metadata));
        
        const response = await fetch(`${this.baseUrl}/api/documents/ingest`, {
            method: 'POST',
            body: formData
        });
        return response.json();
    }
}

// Usage example
const client = new CMOClient();
client.query('What is metabolomics?').then(result => {
    console.log(result.data.answer);
});
```

### cURL Examples

#### Submit Query
```bash
curl -X POST http://localhost:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main applications of metabolomics?",
    "options": {
      "include_citations": true,
      "confidence_threshold": 0.5
    }
  }'
```

#### Health Check
```bash
curl http://localhost:8001/health/detailed
```

#### Ingest Document
```bash
curl -X POST http://localhost:8001/api/documents/ingest \
  -F "file=@research_paper.pdf" \
  -F 'metadata={"title":"Research Paper","authors":["Dr. Smith"]}'
```

#### Get Metrics
```bash
curl http://localhost:8001/metrics
```

## Integration Patterns

### Batch Processing
```python
# Process multiple queries in batch
questions = [
    "What is metabolomics?",
    "How is mass spectrometry used?",
    "What are biomarkers?"
]

results = []
for question in questions:
    result = client.query(question)
    results.append(result)
    time.sleep(0.1)  # Rate limiting
```

### Async Processing
```python
import asyncio
import aiohttp

async def async_query(session, question):
    async with session.post(
        'http://localhost:8001/api/query',
        json={'question': question}
    ) as response:
        return await response.json()

async def batch_queries(questions):
    async with aiohttp.ClientSession() as session:
        tasks = [async_query(session, q) for q in questions]
        return await asyncio.gather(*tasks)
```

### Streaming Responses
```javascript
// WebSocket streaming
const ws = new WebSocket('ws://localhost:8001/ws/chat');

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    if (message.type === 'response') {
        console.log('Answer:', message.data.answer);
    } else if (message.type === 'status') {
        console.log('Status:', message.data.message);
    }
};

ws.send(JSON.stringify({
    type: 'query',
    data: { question: 'What is metabolomics?' }
}));
```

## Best Practices

### Query Optimization
- **Specific Questions**: Ask specific, focused questions for better results
- **Context Provision**: Provide relevant context when available
- **Confidence Thresholds**: Set appropriate confidence thresholds
- **Caching**: Leverage caching for repeated queries

### Error Handling
- **Retry Logic**: Implement exponential backoff for retries
- **Fallback Strategies**: Handle system unavailability gracefully
- **Error Logging**: Log errors for debugging and monitoring
- **User Feedback**: Provide meaningful error messages to users

### Performance
- **Connection Pooling**: Use connection pooling for multiple requests
- **Async Operations**: Use async/await for better concurrency
- **Batch Processing**: Group related operations when possible
- **Monitoring**: Monitor response times and error rates

### Security
- **Authentication**: Always authenticate API requests
- **Input Validation**: Validate all input parameters
- **Rate Limiting**: Respect rate limits and implement backoff
- **Secure Transport**: Use HTTPS in production environments

---

*This API reference is maintained by the CMO development team. Last updated: January 9, 2025*