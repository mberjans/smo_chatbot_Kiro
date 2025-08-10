# LightRAG Integration Troubleshooting Guide

## Overview

This guide provides comprehensive troubleshooting procedures for common issues encountered with the LightRAG integration in the Clinical Metabolomics Oracle system.

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [System Health Checks](#system-health-checks)
4. [Performance Issues](#performance-issues)
5. [Database Issues](#database-issues)
6. [API and Integration Issues](#api-and-integration-issues)
7. [Deployment Issues](#deployment-issues)
8. [Log Analysis](#log-analysis)
9. [Recovery Procedures](#recovery-procedures)
10. [Getting Help](#getting-help)

## Quick Diagnostics

### System Status Check

Run this quick diagnostic script to get an overview of system health:

```bash
#!/bin/bash
# Quick system diagnostic

echo "=== LightRAG System Diagnostics ==="
echo "Timestamp: $(date)"
echo

# Check if application is running
echo "1. Application Status:"
if pgrep -f "python.*main.py" > /dev/null; then
    echo "   ✓ Application is running (PID: $(pgrep -f 'python.*main.py'))"
else
    echo "   ✗ Application is not running"
fi

# Check system resources
echo
echo "2. System Resources:"
echo "   Memory: $(free -h | awk 'NR==2{printf "%.1f%% used", $3*100/$2}')"
echo "   Disk: $(df -h . | awk 'NR==2{printf "%s used", $5}')"
echo "   Load: $(uptime | awk -F'load average:' '{print $2}')"

# Check database connections
echo
echo "3. Database Status:"
if pg_isready -q; then
    echo "   ✓ PostgreSQL is accessible"
else
    echo "   ✗ PostgreSQL connection failed"
fi

# Check Neo4j (if available)
if command -v cypher-shell &> /dev/null; then
    if echo "RETURN 1" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD" &> /dev/null; then
        echo "   ✓ Neo4j is accessible"
    else
        echo "   ✗ Neo4j connection failed"
    fi
fi

# Check API endpoints
echo
echo "4. API Health:"
if curl -s -f http://localhost:8000/health > /dev/null; then
    echo "   ✓ Health endpoint is responding"
else
    echo "   ✗ Health endpoint is not responding"
fi

# Check recent errors
echo
echo "5. Recent Errors:"
if [[ -f "logs/lightrag_errors.log" ]]; then
    ERROR_COUNT=$(tail -100 logs/lightrag_errors.log | grep -c "ERROR")
    echo "   Last 100 log entries contain $ERROR_COUNT errors"
    if [[ $ERROR_COUNT -gt 0 ]]; then
        echo "   Recent errors:"
        tail -5 logs/lightrag_errors.log | grep "ERROR" | sed 's/^/     /'
    fi
else
    echo "   No error log found"
fi

echo
echo "=== End Diagnostics ==="
```

### Health Check API

Use the built-in health check endpoints:

```bash
# Overall system health
curl -s http://localhost:8000/health | jq .

# Detailed component health
curl -s http://localhost:8000/health/detailed | jq .

# Specific component health
curl -s http://localhost:8000/health/lightrag | jq .
curl -s http://localhost:8000/health/database | jq .
curl -s http://localhost:8000/health/neo4j | jq .
```

## Common Issues

### Issue 1: Application Won't Start

**Symptoms:**
- Application exits immediately after starting
- Import errors in logs
- Configuration errors

**Diagnostic Steps:**

1. **Check Python environment:**
   ```bash
   source venv/bin/activate
   python --version
   pip check
   ```

2. **Test imports:**
   ```bash
   python -c "
   import sys
   sys.path.append('src')
   from lightrag_integration.component import LightRAGComponent
   print('✓ Imports successful')
   "
   ```

3. **Validate configuration:**
   ```bash
   python -c "
   import sys
   sys.path.append('src')
   from lightrag_integration.config.settings import LightRAGConfig
   config = LightRAGConfig()
   print('✓ Configuration valid')
   "
   ```

**Common Solutions:**

- **Missing dependencies:** `pip install -r requirements.txt`
- **Python path issues:** Ensure `PYTHONPATH` includes `src` directory
- **Configuration errors:** Check `.env` file and environment variables
- **Port conflicts:** Change port in configuration or kill conflicting processes

### Issue 2: Slow Query Performance

**Symptoms:**
- Queries taking longer than 10 seconds
- High CPU or memory usage
- Timeout errors

**Diagnostic Steps:**

1. **Check system resources:**
   ```bash
   top -p $(pgrep -f "python.*main.py")
   free -h
   df -h
   ```

2. **Analyze query patterns:**
   ```bash
   grep "Query:" logs/lightrag.log | tail -20
   grep "Response time:" logs/lightrag.log | awk '{print $NF}' | sort -n | tail -10
   ```

3. **Check cache performance:**
   ```bash
   python -c "
   import sys
   sys.path.append('src')
   from lightrag_integration.monitoring import HealthMonitor
   import asyncio
   
   async def check_cache():
       monitor = HealthMonitor()
       metrics = await monitor.get_performance_metrics()
       print(f'Cache hit rate: {metrics.get(\"cache_hit_rate\", \"N/A\")}')
   
   asyncio.run(check_cache())
   "
   ```

**Common Solutions:**

- **Reduce batch size:** Lower `LIGHTRAG_BATCH_SIZE` in configuration
- **Increase memory:** Add more RAM or adjust memory limits
- **Optimize indexes:** Run index optimization scripts
- **Enable caching:** Ensure caching is properly configured
- **Limit concurrent requests:** Reduce `LIGHTRAG_MAX_CONCURRENT`

### Issue 3: PDF Processing Failures

**Symptoms:**
- PDFs not being processed
- Text extraction errors
- Entity extraction failures

**Diagnostic Steps:**

1. **Test PDF file:**
   ```bash
   python -c "
   import sys
   sys.path.append('src')
   from lightrag_integration.ingestion.pdf_extractor import PDFExtractor
   import asyncio
   
   async def test_pdf():
       extractor = PDFExtractor()
       try:
           text = await extractor.extract_text('problematic.pdf')
           print(f'✓ Extracted {len(text)} characters')
       except Exception as e:
           print(f'✗ Error: {e}')
   
   asyncio.run(test_pdf())
   "
   ```

2. **Check file permissions:**
   ```bash
   ls -la papers/
   file papers/problematic.pdf
   ```

3. **Test with different PDF:**
   ```bash
   # Try with a simple PDF to isolate the issue
   ```

**Common Solutions:**

- **Corrupted PDF:** Try with a different PDF file
- **Permissions:** Fix file permissions with `chmod 644 papers/*.pdf`
- **Memory issues:** Process PDFs in smaller batches
- **Format issues:** Convert PDF to a supported format
- **Dependencies:** Ensure PyMuPDF or alternative PDF library is installed

### Issue 4: Database Connection Issues

**Symptoms:**
- Database connection errors
- Query failures
- Migration issues

**Diagnostic Steps:**

1. **Test PostgreSQL connection:**
   ```bash
   pg_isready -h localhost -p 5432
   psql -h localhost -p 5432 -U postgres -d clinical_metabolomics_oracle -c "SELECT 1;"
   ```

2. **Test Neo4j connection:**
   ```bash
   echo "RETURN 1" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD"
   ```

3. **Check database logs:**
   ```bash
   # PostgreSQL logs (location varies by system)
   tail -f /var/log/postgresql/postgresql-*.log
   
   # Neo4j logs
   tail -f /var/log/neo4j/neo4j.log
   ```

**Common Solutions:**

- **Service not running:** Start database services
- **Wrong credentials:** Check username/password in `.env`
- **Network issues:** Verify host/port configuration
- **Database doesn't exist:** Create database with `createdb`
- **Migrations needed:** Run `npx prisma migrate deploy`

### Issue 5: Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- Out of memory errors after extended operation
- System becoming unresponsive

**Diagnostic Steps:**

1. **Monitor memory usage over time:**
   ```bash
   # Create a monitoring script
   while true; do
       echo "$(date): $(ps -p $(pgrep -f 'python.*main.py') -o pid,vsz,rss,pcpu,pmem --no-headers)"
       sleep 60
   done > memory_monitor.log
   ```

2. **Profile memory usage:**
   ```bash
   python -m memory_profiler src/main.py
   ```

3. **Check for unclosed resources:**
   ```bash
   lsof -p $(pgrep -f "python.*main.py") | wc -l
   ```

**Common Solutions:**

- **Restart application:** Temporary fix to clear memory
- **Reduce batch sizes:** Lower memory usage per operation
- **Fix resource leaks:** Ensure proper cleanup in code
- **Increase swap:** Add swap space as temporary measure
- **Optimize caching:** Implement proper cache eviction

## System Health Checks

### Automated Health Monitoring

Create a comprehensive health check script:

```bash
#!/bin/bash
# comprehensive_health_check.sh

HEALTH_LOG="logs/health_check.log"
ALERT_THRESHOLD_CPU=80
ALERT_THRESHOLD_MEMORY=90
ALERT_THRESHOLD_DISK=90

log_health() {
    echo "[$(date)] $1" >> "$HEALTH_LOG"
}

check_application() {
    echo "Checking application health..."
    
    if pgrep -f "python.*main.py" > /dev/null; then
        log_health "✓ Application is running"
        
        # Check API response
        if curl -s -f http://localhost:8000/health > /dev/null; then
            log_health "✓ API is responding"
        else
            log_health "✗ API is not responding"
            return 1
        fi
    else
        log_health "✗ Application is not running"
        return 1
    fi
}

check_resources() {
    echo "Checking system resources..."
    
    # CPU usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
    if (( $(echo "$CPU_USAGE > $ALERT_THRESHOLD_CPU" | bc -l) )); then
        log_health "⚠ High CPU usage: ${CPU_USAGE}%"
    else
        log_health "✓ CPU usage normal: ${CPU_USAGE}%"
    fi
    
    # Memory usage
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    if (( $(echo "$MEMORY_USAGE > $ALERT_THRESHOLD_MEMORY" | bc -l) )); then
        log_health "⚠ High memory usage: ${MEMORY_USAGE}%"
    else
        log_health "✓ Memory usage normal: ${MEMORY_USAGE}%"
    fi
    
    # Disk usage
    DISK_USAGE=$(df . | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $DISK_USAGE -gt $ALERT_THRESHOLD_DISK ]]; then
        log_health "⚠ High disk usage: ${DISK_USAGE}%"
    else
        log_health "✓ Disk usage normal: ${DISK_USAGE}%"
    fi
}

check_databases() {
    echo "Checking database connections..."
    
    # PostgreSQL
    if pg_isready -q; then
        log_health "✓ PostgreSQL is accessible"
    else
        log_health "✗ PostgreSQL connection failed"
    fi
    
    # Neo4j
    if command -v cypher-shell &> /dev/null; then
        if echo "RETURN 1" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD" &> /dev/null; then
            log_health "✓ Neo4j is accessible"
        else
            log_health "✗ Neo4j connection failed"
        fi
    fi
}

check_logs() {
    echo "Checking for recent errors..."
    
    if [[ -f "logs/lightrag_errors.log" ]]; then
        RECENT_ERRORS=$(tail -100 logs/lightrag_errors.log | grep -c "ERROR")
        if [[ $RECENT_ERRORS -gt 10 ]]; then
            log_health "⚠ High error rate: $RECENT_ERRORS errors in last 100 log entries"
        else
            log_health "✓ Error rate normal: $RECENT_ERRORS errors in last 100 log entries"
        fi
    fi
}

# Run all checks
check_application
check_resources
check_databases
check_logs

echo "Health check completed. See $HEALTH_LOG for details."
```

### Performance Monitoring

Monitor key performance metrics:

```python
# performance_monitor.py
import asyncio
import time
import psutil
import logging
from lightrag_integration.monitoring import HealthMonitor

async def monitor_performance():
    monitor = HealthMonitor()
    
    while True:
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # Get application metrics
            health = await monitor.get_health_status()
            metrics = await monitor.get_performance_metrics()
            
            # Log metrics
            logging.info(f"CPU: {cpu_percent}%, Memory: {memory.percent}%, Disk: {disk.percent}%")
            logging.info(f"Cache hit rate: {metrics.get('cache_hit_rate', 'N/A')}")
            logging.info(f"Avg response time: {metrics.get('avg_response_time', 'N/A')}s")
            
            # Check thresholds and alert if necessary
            if cpu_percent > 80:
                logging.warning(f"High CPU usage: {cpu_percent}%")
            
            if memory.percent > 90:
                logging.warning(f"High memory usage: {memory.percent}%")
            
            if disk.percent > 90:
                logging.warning(f"High disk usage: {disk.percent}%")
            
        except Exception as e:
            logging.error(f"Performance monitoring error: {e}")
        
        await asyncio.sleep(60)  # Monitor every minute

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(monitor_performance())
```

## Performance Issues

### Query Optimization

1. **Analyze slow queries:**
   ```bash
   grep "Response time:" logs/lightrag.log | awk '$NF > 5 {print}' | tail -10
   ```

2. **Check query patterns:**
   ```bash
   grep "Query:" logs/lightrag.log | awk '{print $NF}' | sort | uniq -c | sort -nr | head -10
   ```

3. **Optimize frequently used queries:**
   - Add caching for common queries
   - Pre-compute results for popular questions
   - Optimize knowledge graph indexes

### Memory Optimization

1. **Reduce memory usage:**
   ```yaml
   # config/performance.yaml
   processing:
     batch_size: 16  # Reduce from default 32
     max_entities_per_chunk: 25  # Reduce from default 50
   
   performance:
     max_concurrent_requests: 5  # Reduce from default 10
   ```

2. **Enable garbage collection:**
   ```python
   import gc
   gc.set_threshold(700, 10, 10)  # More aggressive GC
   ```

3. **Monitor memory usage:**
   ```bash
   python -c "
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
   print(f'Memory percent: {process.memory_percent():.1f}%')
   "
   ```

## Database Issues

### PostgreSQL Troubleshooting

1. **Connection issues:**
   ```bash
   # Check if PostgreSQL is running
   sudo systemctl status postgresql
   
   # Check connections
   sudo -u postgres psql -c "SELECT count(*) FROM pg_stat_activity;"
   
   # Check database exists
   sudo -u postgres psql -l | grep clinical_metabolomics_oracle
   ```

2. **Performance issues:**
   ```sql
   -- Check slow queries
   SELECT query, mean_time, calls 
   FROM pg_stat_statements 
   ORDER BY mean_time DESC 
   LIMIT 10;
   
   -- Check database size
   SELECT pg_size_pretty(pg_database_size('clinical_metabolomics_oracle'));
   ```

3. **Migration issues:**
   ```bash
   # Check migration status
   npx prisma migrate status
   
   # Reset migrations (caution: data loss)
   npx prisma migrate reset
   
   # Deploy pending migrations
   npx prisma migrate deploy
   ```

### Neo4j Troubleshooting

1. **Connection issues:**
   ```bash
   # Check Neo4j status
   sudo systemctl status neo4j
   
   # Test connection
   echo "RETURN 1" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD"
   
   # Check logs
   tail -f /var/log/neo4j/neo4j.log
   ```

2. **Performance issues:**
   ```cypher
   // Check database statistics
   CALL db.stats.retrieve('GRAPH COUNTS');
   
   // Check indexes
   SHOW INDEXES;
   
   // Check constraints
   SHOW CONSTRAINTS;
   ```

3. **Memory issues:**
   ```bash
   # Adjust Neo4j memory settings in neo4j.conf
   dbms.memory.heap.initial_size=512m
   dbms.memory.heap.max_size=2g
   dbms.memory.pagecache.size=1g
   ```

## API and Integration Issues

### Chainlit Integration

1. **UI not loading:**
   ```bash
   # Check Chainlit logs
   grep "chainlit" logs/lightrag.log
   
   # Test Chainlit directly
   chainlit run src/main.py --port 8001
   ```

2. **WebSocket issues:**
   ```bash
   # Check WebSocket connections
   netstat -an | grep :8000
   
   # Test WebSocket endpoint
   wscat -c ws://localhost:8000/ws
   ```

### Translation Issues

1. **Translation failures:**
   ```python
   # Test translation service
   python -c "
   import sys
   sys.path.append('src')
   from translation import detect_language, translate_text
   
   text = 'What is clinical metabolomics?'
   lang = detect_language(text)
   print(f'Detected language: {lang}')
   
   translated = translate_text(text, 'es')
   print(f'Translated: {translated}')
   "
   ```

2. **Language detection issues:**
   ```bash
   # Check lingua installation
   python -c "import lingua; print('✓ Lingua available')"
   
   # Test with different text samples
   ```

## Deployment Issues

### Docker Deployment

1. **Container won't start:**
   ```bash
   # Check container logs
   docker logs lightrag-oracle
   
   # Check container status
   docker ps -a
   
   # Inspect container
   docker inspect lightrag-oracle
   ```

2. **Network issues:**
   ```bash
   # Check network connectivity
   docker network ls
   docker network inspect lightrag-network
   
   # Test inter-container communication
   docker exec lightrag-oracle ping postgres
   ```

3. **Volume issues:**
   ```bash
   # Check volume mounts
   docker volume ls
   docker volume inspect lightrag_data
   
   # Check permissions
   docker exec lightrag-oracle ls -la /app/data
   ```

### Systemd Service Issues

1. **Service won't start:**
   ```bash
   # Check service status
   sudo systemctl status lightrag-oracle
   
   # Check service logs
   sudo journalctl -u lightrag-oracle -f
   
   # Check service file
   sudo systemctl cat lightrag-oracle
   ```

2. **Permission issues:**
   ```bash
   # Check file ownership
   ls -la /opt/clinical-metabolomics-oracle
   
   # Fix permissions
   sudo chown -R oracle:oracle /opt/clinical-metabolomics-oracle
   ```

## Log Analysis

### Log File Locations

- **Application logs:** `logs/lightrag.log`
- **Error logs:** `logs/lightrag_errors.log`
- **Access logs:** `logs/lightrag_access.log`
- **Performance logs:** `logs/lightrag_performance.log`
- **Security logs:** `logs/lightrag_security.log`

### Common Log Analysis Commands

```bash
# Find errors in the last hour
grep "$(date -d '1 hour ago' '+%Y-%m-%d %H')" logs/lightrag_errors.log

# Count errors by type
grep "ERROR" logs/lightrag_errors.log | awk '{print $4}' | sort | uniq -c | sort -nr

# Find slow queries
grep "Response time:" logs/lightrag.log | awk '$NF > 5 {print}' | tail -20

# Monitor logs in real-time
tail -f logs/lightrag.log | grep -E "(ERROR|WARNING|Query:|Response time:)"

# Analyze query patterns
grep "Query:" logs/lightrag.log | awk '{print $NF}' | sort | uniq -c | sort -nr | head -20

# Check memory usage patterns
grep "Memory usage:" logs/lightrag.log | tail -20

# Find authentication issues
grep -i "auth" logs/lightrag_security.log

# Check API response codes
grep "HTTP" logs/lightrag_access.log | awk '{print $9}' | sort | uniq -c | sort -nr
```

### Log Analysis Scripts

```bash
#!/bin/bash
# log_analyzer.sh

echo "=== LightRAG Log Analysis ==="
echo "Analysis time: $(date)"
echo

# Error summary
echo "1. Error Summary (last 24 hours):"
ERROR_COUNT=$(grep "$(date -d '1 day ago' '+%Y-%m-%d')" logs/lightrag_errors.log | wc -l)
echo "   Total errors: $ERROR_COUNT"

if [[ $ERROR_COUNT -gt 0 ]]; then
    echo "   Error types:"
    grep "$(date -d '1 day ago' '+%Y-%m-%d')" logs/lightrag_errors.log | \
        awk '{print $4}' | sort | uniq -c | sort -nr | head -5 | \
        sed 's/^/     /'
fi

# Performance summary
echo
echo "2. Performance Summary:"
AVG_RESPONSE_TIME=$(grep "Response time:" logs/lightrag.log | \
    awk '{sum+=$NF; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
echo "   Average response time: ${AVG_RESPONSE_TIME}s"

SLOW_QUERIES=$(grep "Response time:" logs/lightrag.log | awk '$NF > 5 {count++} END {print count+0}')
echo "   Slow queries (>5s): $SLOW_QUERIES"

# Top queries
echo
echo "3. Top Queries:"
grep "Query:" logs/lightrag.log | awk '{print $NF}' | sort | uniq -c | sort -nr | head -5 | \
    sed 's/^/   /'

# System health
echo
echo "4. System Health Indicators:"
MEMORY_WARNINGS=$(grep -c "High memory usage" logs/lightrag.log)
CPU_WARNINGS=$(grep -c "High CPU usage" logs/lightrag.log)
DISK_WARNINGS=$(grep -c "High disk usage" logs/lightrag.log)

echo "   Memory warnings: $MEMORY_WARNINGS"
echo "   CPU warnings: $CPU_WARNINGS"
echo "   Disk warnings: $DISK_WARNINGS"

echo
echo "=== End Analysis ==="
```

## Recovery Procedures

### Application Recovery

1. **Graceful restart:**
   ```bash
   # Send SIGTERM to allow graceful shutdown
   kill -TERM $(pgrep -f "python.*main.py")
   
   # Wait for shutdown
   sleep 10
   
   # Start application
   python src/main.py &
   ```

2. **Force restart:**
   ```bash
   # Kill application
   pkill -f "python.*main.py"
   
   # Clean up resources
   rm -f .app.pid
   
   # Start application
   python src/main.py &
   ```

3. **Service restart:**
   ```bash
   sudo systemctl restart lightrag-oracle
   sudo systemctl status lightrag-oracle
   ```

### Database Recovery

1. **PostgreSQL recovery:**
   ```bash
   # Restart PostgreSQL
   sudo systemctl restart postgresql
   
   # Check database integrity
   sudo -u postgres psql -d clinical_metabolomics_oracle -c "SELECT 1;"
   
   # Rebuild indexes if needed
   sudo -u postgres psql -d clinical_metabolomics_oracle -c "REINDEX DATABASE clinical_metabolomics_oracle;"
   ```

2. **Neo4j recovery:**
   ```bash
   # Restart Neo4j
   sudo systemctl restart neo4j
   
   # Check database consistency
   echo "CALL db.ping()" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD"
   ```

### Data Recovery

1. **Restore from backup:**
   ```bash
   # Stop application
   sudo systemctl stop lightrag-oracle
   
   # Restore data
   BACKUP_PATH="/backups/20240109_020000"
   rm -rf data/lightrag_kg data/lightrag_vectors
   tar -xzf "$BACKUP_PATH/lightrag_kg.tar.gz"
   tar -xzf "$BACKUP_PATH/lightrag_vectors.tar.gz"
   
   # Start application
   sudo systemctl start lightrag-oracle
   ```

2. **Rebuild knowledge base:**
   ```bash
   # Clear existing data
   rm -rf data/lightrag_kg/* data/lightrag_vectors/*
   
   # Rebuild from source documents
   python scripts/rebuild_knowledge_base.py --directory papers/
   ```

## Getting Help

### Information to Collect

When seeking help, collect the following information:

1. **System Information:**
   ```bash
   python scripts/system_info.py > system_info.txt
   ```

2. **Error Logs:**
   ```bash
   tail -100 logs/lightrag_errors.log > recent_errors.txt
   ```

3. **Configuration:**
   ```bash
   python scripts/config_dump.py --sanitized > config_info.txt
   ```

4. **Health Status:**
   ```bash
   curl -s http://localhost:8000/health/detailed > health_status.json
   ```

### Support Channels

1. **Documentation:** Check API documentation and admin guide
2. **Logs:** Search error logs for specific error messages
3. **Community:** Check GitHub issues and discussions
4. **Support:** Contact technical support with collected information

### Emergency Contacts

- **System Administrator:** [admin@example.com]
- **Technical Support:** [support@example.com]
- **On-call Engineer:** [oncall@example.com]

This troubleshooting guide provides comprehensive coverage of common issues and their solutions for the LightRAG integration system.