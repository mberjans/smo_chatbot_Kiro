# LightRAG Integration Administration Guide

## Overview

This guide provides comprehensive instructions for administering and maintaining the LightRAG integration within the Clinical Metabolomics Oracle system. It covers installation, configuration, monitoring, troubleshooting, and maintenance procedures.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [System Administration](#system-administration)
5. [Monitoring](#monitoring)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Backup and Recovery](#backup-and-recovery)
8. [Performance Tuning](#performance-tuning)
9. [Security](#security)
10. [Troubleshooting](#troubleshooting)

## System Requirements

### Hardware Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.5 GHz
- RAM: 8 GB
- Storage: 50 GB available space
- Network: Stable internet connection

**Recommended Requirements:**
- CPU: 8+ cores, 3.0+ GHz
- RAM: 16+ GB
- Storage: 100+ GB SSD
- Network: High-speed internet connection

### Software Requirements

- Python 3.8+
- Node.js 16+
- PostgreSQL 12+
- Neo4j 4.4+
- Docker (optional, for containerized deployment)

### Dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Node.js dependencies
npm install
```

## Installation

### 1. Environment Setup

Create and configure the environment:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
npm install
```

### 2. Database Setup

Configure PostgreSQL and Neo4j databases:

```bash
# PostgreSQL setup
createdb clinical_metabolomics_oracle

# Run Prisma migrations
npx prisma migrate dev

# Neo4j setup (ensure Neo4j is running)
# Default connection: bolt://localhost:7687
```

### 3. Environment Variables

Create `.env` file with required configuration:

```env
# Database Configuration
DATABASE_URL="postgresql://username:password@localhost:5432/clinical_metabolomics_oracle"
NEO4J_PASSWORD="your_neo4j_password"

# API Keys
GROQ_API_KEY="your_groq_api_key"
OPENAI_API_KEY="your_openai_api_key"
PERPLEXITY_API="your_perplexity_api_key"

# LightRAG Configuration
LIGHTRAG_KG_PATH="data/lightrag_kg"
LIGHTRAG_VECTOR_PATH="data/lightrag_vectors"
LIGHTRAG_CACHE_PATH="data/lightrag_cache"
LIGHTRAG_EMBEDDING_MODEL="intfloat/e5-base-v2"
LIGHTRAG_LLM_MODEL="groq:Llama-3.3-70b-Versatile"
LIGHTRAG_BATCH_SIZE="32"
LIGHTRAG_MAX_CONCURRENT="10"
LIGHTRAG_CACHE_TTL="3600"

# Monitoring Configuration
ENABLE_MONITORING="true"
MONITORING_INTERVAL="60"
LOG_LEVEL="INFO"
```

### 4. Initial Data Setup

Set up initial data directories and permissions:

```bash
# Create data directories
mkdir -p data/lightrag_kg
mkdir -p data/lightrag_vectors
mkdir -p data/lightrag_cache
mkdir -p papers/

# Set appropriate permissions
chmod 755 data/
chmod 755 papers/
```

## Configuration

### LightRAG Configuration

The system uses a hierarchical configuration approach:

1. **Environment Variables** (highest priority)
2. **Configuration Files** (medium priority)
3. **Default Values** (lowest priority)

#### Configuration File

Create `config/lightrag_config.yaml`:

```yaml
# Storage Configuration
storage:
  knowledge_graph_path: "data/lightrag_kg"
  vector_store_path: "data/lightrag_vectors"
  cache_directory: "data/lightrag_cache"

# Processing Configuration
processing:
  chunk_size: 1000
  chunk_overlap: 200
  max_entities_per_chunk: 50
  batch_size: 32

# Model Configuration
models:
  embedding_model: "intfloat/e5-base-v2"
  llm_model: "groq:Llama-3.3-70b-Versatile"

# Performance Configuration
performance:
  max_concurrent_requests: 10
  cache_ttl_seconds: 3600
  query_timeout: 30

# Monitoring Configuration
monitoring:
  enabled: true
  interval: 60
  metrics_retention_days: 30
```

### Logging Configuration

Configure logging in `config/logging.yaml`:

```yaml
version: 1
disable_existing_loggers: false

formatters:
  standard:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  detailed:
    format: '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/lightrag.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

  error_file:
    class: logging.handlers.RotatingFileHandler
    level: ERROR
    formatter: detailed
    filename: logs/lightrag_errors.log
    maxBytes: 10485760  # 10MB
    backupCount: 5

loggers:
  lightrag_integration:
    level: DEBUG
    handlers: [console, file, error_file]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
```

## System Administration

### Starting the System

```bash
# Start the main application
python src/main.py

# Or with FastAPI mounting
python src/app.py

# Start with specific configuration
LIGHTRAG_CONFIG_PATH=config/production.yaml python src/main.py
```

### Stopping the System

```bash
# Graceful shutdown (Ctrl+C or SIGTERM)
# The system will complete current operations and cleanup resources

# Force stop (if needed)
pkill -f "python src/main.py"
```

### Service Management

For production deployment, use systemd service:

Create `/etc/systemd/system/lightrag-oracle.service`:

```ini
[Unit]
Description=Clinical Metabolomics Oracle with LightRAG
After=network.target postgresql.service neo4j.service

[Service]
Type=simple
User=oracle
Group=oracle
WorkingDirectory=/opt/clinical-metabolomics-oracle
Environment=PATH=/opt/clinical-metabolomics-oracle/venv/bin
ExecStart=/opt/clinical-metabolomics-oracle/venv/bin/python src/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl enable lightrag-oracle
sudo systemctl start lightrag-oracle
sudo systemctl status lightrag-oracle
```

### User Management

The system integrates with the existing user management system. Admin users can:

1. **Access Admin Interface**: Navigate to `/admin` endpoint
2. **Manage Documents**: Add/remove documents from knowledge base
3. **Monitor System**: View system health and metrics
4. **Configure Settings**: Modify system configuration

### Document Management

#### Adding Documents

```bash
# Copy PDFs to papers directory
cp new_research_papers/*.pdf papers/

# Or use the admin interface
curl -X POST http://localhost:8000/admin/documents \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -F "file=@new_paper.pdf"
```

#### Removing Documents

```bash
# Use admin interface
curl -X DELETE http://localhost:8000/admin/documents/document_id \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

#### Batch Operations

```bash
# Process multiple documents
python scripts/batch_ingest.py --directory papers/ --batch-size 10

# Update knowledge base
python scripts/update_knowledge_base.py --incremental
```

## Monitoring

### Health Checks

The system provides several health check endpoints:

```bash
# Overall system health
curl http://localhost:8000/health

# Component-specific health
curl http://localhost:8000/health/lightrag
curl http://localhost:8000/health/database
curl http://localhost:8000/health/neo4j

# Detailed health information
curl http://localhost:8000/health/detailed
```

### Metrics Collection

Key metrics to monitor:

1. **Performance Metrics**
   - Query response time
   - Document processing time
   - Memory usage
   - CPU utilization

2. **System Metrics**
   - Active connections
   - Cache hit rate
   - Error rate
   - Uptime

3. **Business Metrics**
   - Number of queries processed
   - Document ingestion rate
   - User satisfaction scores

### Monitoring Tools

#### Built-in Monitoring

```python
from lightrag_integration.monitoring import HealthMonitor

monitor = HealthMonitor()

# Get current metrics
metrics = await monitor.get_performance_metrics()
print(f"Average response time: {metrics['avg_response_time']}")
print(f"Cache hit rate: {metrics['cache_hit_rate']}")
```

#### External Monitoring

Integrate with external monitoring tools:

```bash
# Prometheus metrics endpoint
curl http://localhost:8000/metrics

# Grafana dashboard configuration available in config/grafana/
```

### Alerting

Configure alerts for critical issues:

```yaml
# config/alerts.yaml
alerts:
  - name: "High Response Time"
    condition: "avg_response_time > 10"
    severity: "warning"
    notification: "email"
    
  - name: "System Down"
    condition: "health_status != 'healthy'"
    severity: "critical"
    notification: "email,slack"
    
  - name: "High Error Rate"
    condition: "error_rate > 0.05"
    severity: "warning"
    notification: "email"
```

## Maintenance Procedures

### Regular Maintenance Tasks

#### Daily Tasks

1. **Check System Health**
   ```bash
   python scripts/health_check.py --detailed
   ```

2. **Review Logs**
   ```bash
   tail -f logs/lightrag.log
   grep ERROR logs/lightrag_errors.log
   ```

3. **Monitor Resource Usage**
   ```bash
   python scripts/resource_monitor.py
   ```

#### Weekly Tasks

1. **Update Knowledge Base**
   ```bash
   python scripts/update_knowledge_base.py --incremental
   ```

2. **Clean Cache**
   ```bash
   python scripts/clean_cache.py --older-than 7d
   ```

3. **Review Performance Metrics**
   ```bash
   python scripts/performance_report.py --period week
   ```

#### Monthly Tasks

1. **Full System Backup**
   ```bash
   python scripts/backup_system.py --full
   ```

2. **Performance Optimization**
   ```bash
   python scripts/optimize_performance.py
   ```

3. **Security Audit**
   ```bash
   python scripts/security_audit.py
   ```

### Knowledge Base Updates

#### Incremental Updates

```bash
# Add new documents
python scripts/incremental_update.py --directory papers/new/

# Update existing documents
python scripts/incremental_update.py --update-existing
```

#### Full Rebuild

```bash
# Full knowledge base rebuild (use with caution)
python scripts/rebuild_knowledge_base.py --confirm
```

#### Version Control

```bash
# Create knowledge base snapshot
python scripts/create_snapshot.py --name "v1.2.3"

# List available snapshots
python scripts/list_snapshots.py

# Rollback to previous version
python scripts/rollback.py --version "v1.2.2"
```

## Backup and Recovery

### Backup Strategy

#### What to Backup

1. **Knowledge Graph Data** (`data/lightrag_kg/`)
2. **Vector Embeddings** (`data/lightrag_vectors/`)
3. **Configuration Files** (`config/`)
4. **Database Data** (PostgreSQL and Neo4j)
5. **Source Documents** (`papers/`)

#### Backup Schedule

```bash
# Daily incremental backup
0 2 * * * /opt/scripts/backup_incremental.sh

# Weekly full backup
0 1 * * 0 /opt/scripts/backup_full.sh

# Monthly archive
0 0 1 * * /opt/scripts/backup_archive.sh
```

#### Backup Script

```bash
#!/bin/bash
# backup_full.sh

BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup knowledge graph
tar -czf "$BACKUP_DIR/lightrag_kg.tar.gz" data/lightrag_kg/

# Backup vector store
tar -czf "$BACKUP_DIR/lightrag_vectors.tar.gz" data/lightrag_vectors/

# Backup configuration
tar -czf "$BACKUP_DIR/config.tar.gz" config/

# Backup PostgreSQL
pg_dump clinical_metabolomics_oracle > "$BACKUP_DIR/postgres.sql"

# Backup Neo4j
neo4j-admin dump --database=neo4j --to="$BACKUP_DIR/neo4j.dump"

# Backup source documents
tar -czf "$BACKUP_DIR/papers.tar.gz" papers/

echo "Backup completed: $BACKUP_DIR"
```

### Recovery Procedures

#### Disaster Recovery

```bash
# 1. Stop the system
sudo systemctl stop lightrag-oracle

# 2. Restore data from backup
BACKUP_DIR="/backups/20240109_020000"

# Restore knowledge graph
rm -rf data/lightrag_kg/
tar -xzf "$BACKUP_DIR/lightrag_kg.tar.gz"

# Restore vector store
rm -rf data/lightrag_vectors/
tar -xzf "$BACKUP_DIR/lightrag_vectors.tar.gz"

# Restore configuration
tar -xzf "$BACKUP_DIR/config.tar.gz"

# Restore PostgreSQL
dropdb clinical_metabolomics_oracle
createdb clinical_metabolomics_oracle
psql clinical_metabolomics_oracle < "$BACKUP_DIR/postgres.sql"

# Restore Neo4j
neo4j-admin load --database=neo4j --from="$BACKUP_DIR/neo4j.dump"

# 3. Start the system
sudo systemctl start lightrag-oracle

# 4. Verify recovery
python scripts/verify_recovery.py
```

## Performance Tuning

### System Optimization

#### Memory Optimization

```python
# config/performance.yaml
memory:
  max_heap_size: "4g"
  cache_size: "2g"
  batch_size: 16  # Reduce for lower memory usage
  max_concurrent_requests: 5  # Reduce for lower memory usage
```

#### CPU Optimization

```python
# config/performance.yaml
cpu:
  worker_threads: 8  # Match CPU cores
  async_workers: 4
  processing_timeout: 30
```

#### Storage Optimization

```bash
# Use SSD for better I/O performance
# Optimize database settings
# Regular cleanup of old cache files
python scripts/cleanup_cache.py --older-than 30d
```

### Query Optimization

#### Caching Strategy

```python
# config/caching.yaml
caching:
  query_cache_size: 1000
  embedding_cache_size: 5000
  ttl_seconds: 3600
  cleanup_interval: 300
```

#### Index Optimization

```bash
# Optimize Neo4j indexes
python scripts/optimize_neo4j_indexes.py

# Optimize vector store indexes
python scripts/optimize_vector_indexes.py
```

## Security

### Access Control

1. **API Authentication**: Use JWT tokens for API access
2. **Admin Interface**: Restrict access to authorized users
3. **File Upload**: Validate and sanitize uploaded files
4. **Database Access**: Use least privilege principle

### Security Configuration

```yaml
# config/security.yaml
security:
  jwt_secret: "your-secret-key"
  token_expiry: 3600
  max_file_size: 10485760  # 10MB
  allowed_file_types: [".pdf"]
  rate_limiting:
    requests_per_minute: 60
    burst_size: 10
```

### Security Best Practices

1. **Regular Updates**: Keep dependencies updated
2. **Input Validation**: Validate all user inputs
3. **Secure Communication**: Use HTTPS in production
4. **Audit Logging**: Log all administrative actions
5. **Backup Encryption**: Encrypt backup files

## Troubleshooting

### Common Issues

#### System Won't Start

**Symptoms**: Application fails to start or crashes immediately

**Possible Causes**:
- Missing dependencies
- Database connection issues
- Configuration errors
- Port conflicts

**Solutions**:
```bash
# Check dependencies
pip check

# Test database connections
python scripts/test_connections.py

# Validate configuration
python scripts/validate_config.py

# Check port availability
netstat -tulpn | grep :8000
```

#### Slow Query Performance

**Symptoms**: Queries take longer than expected

**Possible Causes**:
- Large knowledge graph
- Insufficient memory
- Unoptimized indexes
- High concurrent load

**Solutions**:
```bash
# Check system resources
python scripts/resource_monitor.py

# Optimize indexes
python scripts/optimize_indexes.py

# Analyze query patterns
python scripts/query_analysis.py

# Adjust configuration
# Reduce batch_size or max_concurrent_requests
```

#### PDF Processing Failures

**Symptoms**: PDFs fail to process or extract text

**Possible Causes**:
- Corrupted PDF files
- Unsupported PDF format
- Memory issues
- Permission problems

**Solutions**:
```bash
# Test PDF file
python scripts/test_pdf.py --file problematic.pdf

# Check file permissions
ls -la papers/

# Process with debug logging
LIGHTRAG_LOG_LEVEL=DEBUG python scripts/process_pdf.py --file problematic.pdf
```

#### Memory Issues

**Symptoms**: Out of memory errors or high memory usage

**Possible Causes**:
- Large batch sizes
- Memory leaks
- Insufficient system memory
- Large document collections

**Solutions**:
```bash
# Monitor memory usage
python scripts/memory_monitor.py

# Reduce batch sizes
# Edit config/performance.yaml

# Restart system to clear memory
sudo systemctl restart lightrag-oracle

# Check for memory leaks
python scripts/memory_profiler.py
```

### Diagnostic Tools

#### Health Check Script

```bash
#!/bin/bash
# scripts/health_check.py

python -c "
import asyncio
from lightrag_integration.monitoring import HealthMonitor

async def main():
    monitor = HealthMonitor()
    health = await monitor.get_health_status()
    print(f'System Status: {health.status}')
    for component, status in health.components.items():
        print(f'  {component}: {status}')

asyncio.run(main())
"
```

#### Log Analysis

```bash
# Find errors in logs
grep -n "ERROR" logs/lightrag.log | tail -20

# Analyze query patterns
grep "Query:" logs/lightrag.log | awk '{print $NF}' | sort | uniq -c | sort -nr

# Check response times
grep "Response time:" logs/lightrag.log | awk '{print $NF}' | sort -n
```

### Getting Help

1. **Check Documentation**: Review API documentation and guides
2. **Search Logs**: Look for error messages and stack traces
3. **Test Components**: Use diagnostic scripts to isolate issues
4. **Contact Support**: Provide logs and system information

### Support Information

When contacting support, provide:

1. **System Information**:
   ```bash
   python scripts/system_info.py
   ```

2. **Error Logs**:
   ```bash
   tail -100 logs/lightrag_errors.log
   ```

3. **Configuration**:
   ```bash
   python scripts/config_dump.py --sanitized
   ```

4. **Health Status**:
   ```bash
   python scripts/health_check.py --detailed
   ```

This administration guide provides comprehensive coverage of all aspects of managing the LightRAG integration system.