# Query Routing System

The Query Routing System provides intelligent routing capabilities for the LightRAG integration, automatically determining whether queries should be handled by LightRAG (knowledge base) or Perplexity API (real-time information), with support for fallback mechanisms and response combination.

## Overview

The routing system consists of two main components:

1. **QueryClassifier**: Uses LLM-based analysis to classify queries into categories
2. **QueryRouter**: Routes queries to appropriate systems based on classification

## Components

### QueryClassifier

The `QueryClassifier` analyzes incoming queries and classifies them into three categories:

- **KNOWLEDGE_BASE**: Queries about established scientific knowledge, definitions, mechanisms
- **REAL_TIME**: Queries requiring current, recent, or time-sensitive information  
- **HYBRID**: Complex queries requiring both established knowledge and recent information

#### Features

- LLM-based classification with configurable confidence thresholds
- Batch processing support with concurrency control
- Comprehensive error handling with fallback classification
- Statistics tracking and performance metrics
- Retry logic for transient failures

#### Usage

```python
from lightrag_integration.routing import QueryClassifier, QueryType
from llama_index.llms.groq import Groq

# Initialize classifier
llm = Groq(model="llama-3.3-70b-versatile", api_key="your-api-key")
classifier = QueryClassifier(llm, config)

# Classify a query
classification = await classifier.classify_query("What is clinical metabolomics?")
print(f"Query type: {classification.query_type}")
print(f"Confidence: {classification.confidence_score}")
print(f"Reasoning: {classification.reasoning}")
```

### QueryRouter

The `QueryRouter` uses classification results to intelligently route queries to the most appropriate system(s).

#### Routing Strategies

- **LIGHTRAG_ONLY**: Route exclusively to LightRAG
- **PERPLEXITY_ONLY**: Route exclusively to Perplexity API
- **LIGHTRAG_FIRST**: Try LightRAG first, fallback to Perplexity
- **PERPLEXITY_FIRST**: Try Perplexity first, fallback to LightRAG
- **PARALLEL**: Query both systems simultaneously
- **HYBRID**: Intelligent combination of multiple sources

#### Features

- Automatic routing decision based on query classification
- Configurable confidence thresholds and timeouts
- Comprehensive fallback mechanisms
- Response combination for hybrid queries
- Batch processing with concurrency control
- Detailed metrics collection and logging
- Error recovery and graceful degradation

#### Usage

```python
from lightrag_integration.routing import QueryRouter, QueryClassifier

# Initialize components
classifier = QueryClassifier(llm, config)
router = QueryRouter(
    classifier=classifier,
    lightrag_query_func=your_lightrag_function,
    perplexity_query_func=your_perplexity_function,
    config=config
)

# Route a query
response = await router.route_query("What is clinical metabolomics?")
print(f"Source: {response.source}")
print(f"Strategy: {response.routing_decision.strategy}")
print(f"Content: {response.content}")
```

## Configuration

The routing system supports extensive configuration through the `LightRAGConfig` class:

```python
# Confidence thresholds for routing decisions
routing_config = {
    "confidence_thresholds": {
        "high_confidence": 0.8,
        "medium_confidence": 0.6,
        "low_confidence": 0.4
    },
    "timeout_seconds": {
        "lightrag": 30.0,
        "perplexity": 15.0,
        "parallel": 45.0
    },
    "retry_attempts": {
        "lightrag": 2,
        "perplexity": 3
    },
    "enable_parallel_queries": True,
    "enable_response_combination": True
}

# Update configuration
router.update_routing_config(routing_config)
```

## Integration with Existing System

The routing system is designed to integrate seamlessly with the existing Chainlit application. Here's how to modify your `main.py`:

### 1. Add Imports

```python
from lightrag_integration.routing import QueryRouter, QueryClassifier
from llama_index.llms.groq import Groq
```

### 2. Initialize Routing in `on_chat_start`

```python
@cl.on_chat_start
async def on_chat_start(accepted: bool = False):
    # ... existing initialization ...
    
    # Initialize routing system
    try:
        llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        classifier = QueryClassifier(llm, lightrag_config)
        
        query_router = QueryRouter(
            classifier=classifier,
            lightrag_query_func=lambda q: query_lightrag(lightrag_component, q),
            perplexity_query_func=query_perplexity,
            config=lightrag_config
        )
        cl.user_session.set("query_router", query_router)
        
    except Exception as e:
        logging.error(f"Failed to initialize routing: {str(e)}")
        cl.user_session.set("query_router", None)
```

### 3. Use Routing in `on_message`

```python
@cl.on_message
async def on_message(message: cl.Message):
    # ... existing setup ...
    
    query_router = cl.user_session.get("query_router")
    
    if query_router:
        # Use intelligent routing
        routed_response = await query_router.route_query(content)
        response_data = {
            "content": routed_response.content,
            "bibliography": routed_response.bibliography,
            "confidence_score": routed_response.confidence_score,
            "source": routed_response.source,
            "routing_info": {
                "strategy": routed_response.routing_decision.strategy.value,
                "classification": routed_response.routing_decision.classification.query_type.value
            }
        }
    else:
        # Fallback to original logic
        # ... existing fallback code ...
```

## Routing Decision Logic

The router makes routing decisions based on query classification and confidence scores:

### Knowledge Base Queries
- **High confidence (≥0.8)**: Route to LightRAG only
- **Lower confidence**: Route to LightRAG first with Perplexity fallback

### Real-time Queries  
- **High confidence (≥0.8)**: Route to Perplexity only
- **Lower confidence**: Route to Perplexity first with LightRAG fallback

### Hybrid Queries
- **Parallel enabled**: Query both systems simultaneously and combine responses
- **Parallel disabled**: Route to LightRAG first with Perplexity fallback

## Error Handling and Fallbacks

The routing system implements comprehensive error handling:

1. **Classification Errors**: Automatic retry with exponential backoff, fallback to heuristic classification
2. **Query Execution Errors**: Automatic fallback to alternative systems
3. **Timeout Handling**: Configurable timeouts with graceful degradation
4. **Service Unavailability**: Intelligent fallback routing with error logging

## Metrics and Monitoring

The router collects detailed metrics for monitoring and optimization:

```python
metrics = router.get_routing_metrics()
print(f"Total queries: {metrics['total_queries']}")
print(f"Success rate: {metrics['success_rate']:.2%}")
print(f"Average response time: {metrics['average_response_time']:.3f}s")
print(f"Fallback activations: {metrics['fallback_activations']}")
```

Available metrics include:
- Query counts by source and strategy
- Success/failure rates
- Response times and performance metrics
- Error rates by component
- Routing accuracy and confidence distributions

## Testing

The routing system includes comprehensive tests:

```bash
# Run unit tests (requires dependencies)
python -m pytest src/lightrag_integration/routing/test_router.py -v

# Run integration demonstration
python src/lightrag_integration/routing/demo_router.py

# View integration example
python src/lightrag_integration/routing/integration_example.py
```

## Performance Considerations

- **Concurrency Control**: Configurable limits on concurrent requests
- **Timeout Management**: Separate timeouts for different operations
- **Caching**: Classification results can be cached for repeated queries
- **Batch Processing**: Efficient handling of multiple queries
- **Resource Management**: Automatic cleanup and connection pooling

## Best Practices

1. **Configuration**: Tune confidence thresholds based on your use case
2. **Monitoring**: Regularly review routing metrics and adjust strategies
3. **Error Handling**: Implement proper logging and alerting for failures
4. **Testing**: Use the demo and integration examples to validate behavior
5. **Fallbacks**: Always configure fallback mechanisms for reliability

## Troubleshooting

### Common Issues

1. **Classification Failures**: Check LLM API keys and connectivity
2. **Routing Errors**: Verify query functions are properly configured
3. **Timeout Issues**: Adjust timeout settings based on system performance
4. **Memory Usage**: Monitor resource usage with large batch operations

### Debug Logging

Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger("query_router").setLevel(logging.DEBUG)
logging.getLogger("query_classifier").setLevel(logging.DEBUG)
```

## Future Enhancements

Planned improvements include:
- Machine learning-based routing optimization
- Advanced response combination strategies
- Real-time performance adaptation
- Enhanced caching mechanisms
- Integration with additional knowledge sources