# LightRAG Integration Operational Procedures

## Overview

This document provides detailed operational procedures for managing the LightRAG integration in the Clinical Metabolomics Oracle system. It covers daily operations, maintenance tasks, incident response, and emergency procedures.

## Table of Contents

1. [Daily Operations](#daily-operations)
2. [Weekly Maintenance](#weekly-maintenance)
3. [Monthly Procedures](#monthly-procedures)
4. [Incident Response](#incident-response)
5. [Emergency Procedures](#emergency-procedures)
6. [Change Management](#change-management)
7. [Monitoring and Alerting](#monitoring-and-alerting)
8. [Backup and Recovery](#backup-and-recovery)
9. [Performance Management](#performance-management)
10. [Security Procedures](#security-procedures)

## Daily Operations

### Morning Checklist

Execute these tasks every morning to ensure system health:

```bash
#!/bin/bash
# daily_morning_checklist.sh

echo "=== Daily Morning Checklist - $(date) ==="

# 1. Check system status
echo "1. Checking system status..."
if systemctl is-active lightrag-oracle &> /dev/null; then
    echo "   ✓ LightRAG Oracle service is running"
else
    echo "   ✗ LightRAG Oracle service is not running"
    echo "   Action: Investigate and restart service"
fi

# 2. Check API health
echo "2. Checking API health..."
if curl -s -f http://localhost:8000/health > /dev/null; then
    echo "   ✓ API is responding"
    
    # Get detailed health info
    HEALTH_STATUS=$(curl -s http://localhost:8000/health | jq -r '.status')
    echo "   Status: $HEALTH_STATUS"
else
    echo "   ✗ API is not responding"
    echo "   Action: Check application logs and restart if needed"
fi

# 3. Check database connections
echo "3. Checking database connections..."
if pg_isready -q; then
    echo "   ✓ PostgreSQL is accessible"
else
    echo "   ✗ PostgreSQL connection failed"
    echo "   Action: Check PostgreSQL service status"
fi

if command -v cypher-shell &> /dev/null; then
    if echo "RETURN 1" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD" &> /dev/null; then
        echo "   ✓ Neo4j is accessible"
    else
        echo "   ✗ Neo4j connection failed"
        echo "   Action: Check Neo4j service status"
    fi
fi

# 4. Check system resources
echo "4. Checking system resources..."
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
DISK_USAGE=$(df . | awk 'NR==2 {print $5}' | sed 's/%//')
LOAD_AVG=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')

echo "   Memory usage: ${MEMORY_USAGE}%"
echo "   Disk usage: ${DISK_USAGE}%"
echo "   Load average: ${LOAD_AVG}"

if (( $(echo "$MEMORY_USAGE > 85" | bc -l) )); then
    echo "   ⚠ High memory usage detected"
fi

if [[ $DISK_USAGE -gt 85 ]]; then
    echo "   ⚠ High disk usage detected"
fi

# 5. Check recent errors
echo "5. Checking recent errors..."
if [[ -f "logs/lightrag_errors.log" ]]; then
    RECENT_ERRORS=$(tail -100 logs/lightrag_errors.log | grep -c "ERROR")
    echo "   Recent errors (last 100 entries): $RECENT_ERRORS"
    
    if [[ $RECENT_ERRORS -gt 10 ]]; then
        echo "   ⚠ High error rate detected"
        echo "   Action: Review error logs for patterns"
    fi
else
    echo "   No error log found"
fi

# 6. Check log file sizes
echo "6. Checking log file sizes..."
if [[ -d "logs" ]]; then
    LOG_SIZE=$(du -sh logs/ | awk '{print $1}')
    echo "   Total log size: $LOG_SIZE"
    
    # Check individual log files
    find logs/ -name "*.log" -size +100M -exec echo "   ⚠ Large log file: {} ($(du -h {} | awk '{print $1}'))" \;
fi

# 7. Check new documents
echo "7. Checking for new documents..."
if [[ -d "papers" ]]; then
    NEW_DOCS=$(find papers/ -name "*.pdf" -mtime -1 | wc -l)
    echo "   New documents (last 24h): $NEW_DOCS"
    
    if [[ $NEW_DOCS -gt 0 ]]; then
        echo "   Action: Process new documents if not auto-processed"
    fi
fi

echo
echo "=== Morning checklist completed ==="
echo "Next steps:"
echo "- Review any warnings or errors above"
echo "- Check monitoring dashboards"
echo "- Process any pending maintenance tasks"
```

### Evening Checklist

Execute these tasks every evening:

```bash
#!/bin/bash
# daily_evening_checklist.sh

echo "=== Daily Evening Checklist - $(date) ==="

# 1. Performance summary
echo "1. Daily performance summary..."
if [[ -f "logs/lightrag.log" ]]; then
    TOTAL_QUERIES=$(grep -c "Query:" logs/lightrag.log)
    AVG_RESPONSE_TIME=$(grep "Response time:" logs/lightrag.log | \
        awk '{sum+=$NF; count++} END {if(count>0) printf "%.2f", sum/count; else print "N/A"}')
    SLOW_QUERIES=$(grep "Response time:" logs/lightrag.log | awk '$NF > 5 {count++} END {print count+0}')
    
    echo "   Total queries today: $TOTAL_QUERIES"
    echo "   Average response time: ${AVG_RESPONSE_TIME}s"
    echo "   Slow queries (>5s): $SLOW_QUERIES"
fi

# 2. Error summary
echo "2. Daily error summary..."
if [[ -f "logs/lightrag_errors.log" ]]; then
    TODAY_ERRORS=$(grep "$(date '+%Y-%m-%d')" logs/lightrag_errors.log | wc -l)
    echo "   Errors today: $TODAY_ERRORS"
    
    if [[ $TODAY_ERRORS -gt 0 ]]; then
        echo "   Error types:"
        grep "$(date '+%Y-%m-%d')" logs/lightrag_errors.log | \
            awk '{print $4}' | sort | uniq -c | sort -nr | head -3 | \
            sed 's/^/     /'
    fi
fi

# 3. Resource usage trends
echo "3. Resource usage trends..."
# This would typically integrate with monitoring system
echo "   Check monitoring dashboard for detailed trends"

# 4. Backup status
echo "4. Checking backup status..."
if [[ -f ".last_backup" ]]; then
    LAST_BACKUP=$(cat .last_backup)
    BACKUP_AGE=$(find "$LAST_BACKUP" -mtime +1 2>/dev/null | wc -l)
    
    if [[ $BACKUP_AGE -gt 0 ]]; then
        echo "   ⚠ Last backup is older than 24 hours"
        echo "   Action: Check backup system"
    else
        echo "   ✓ Recent backup available"
    fi
else
    echo "   ⚠ No backup information found"
fi

# 5. Log rotation check
echo "5. Checking log rotation..."
find logs/ -name "*.log" -size +50M -exec echo "   ⚠ Log file needs rotation: {}" \;

# 6. Cleanup tasks
echo "6. Performing cleanup tasks..."
# Clean temporary files
find /tmp -name "lightrag_*" -mtime +1 -delete 2>/dev/null || true
echo "   ✓ Temporary files cleaned"

# Clean old cache files
find data/lightrag_cache -name "*.cache" -mtime +7 -delete 2>/dev/null || true
echo "   ✓ Old cache files cleaned"

echo
echo "=== Evening checklist completed ==="
```

## Weekly Maintenance

### Weekly Maintenance Script

```bash
#!/bin/bash
# weekly_maintenance.sh

echo "=== Weekly Maintenance - $(date) ==="

# 1. System health report
echo "1. Generating weekly health report..."
python scripts/weekly_health_report.py > reports/weekly_health_$(date +%Y%m%d).txt

# 2. Performance analysis
echo "2. Analyzing weekly performance..."
python scripts/performance_analysis.py --period week > reports/weekly_performance_$(date +%Y%m%d).txt

# 3. Log analysis
echo "3. Analyzing logs..."
python scripts/log_analysis.py --period week > reports/weekly_logs_$(date +%Y%m%d).txt

# 4. Database maintenance
echo "4. Performing database maintenance..."

# PostgreSQL maintenance
echo "   PostgreSQL maintenance..."
sudo -u postgres psql -d clinical_metabolomics_oracle -c "VACUUM ANALYZE;"
sudo -u postgres psql -d clinical_metabolomics_oracle -c "REINDEX DATABASE clinical_metabolomics_oracle;"

# Neo4j maintenance
echo "   Neo4j maintenance..."
# This would typically involve Neo4j-specific maintenance commands

# 5. Knowledge base updates
echo "5. Checking for knowledge base updates..."
if [[ -d "papers" ]]; then
    NEW_PAPERS=$(find papers/ -name "*.pdf" -mtime -7 | wc -l)
    if [[ $NEW_PAPERS -gt 0 ]]; then
        echo "   Processing $NEW_PAPERS new papers..."
        python scripts/incremental_update.py --directory papers/
    else
        echo "   No new papers to process"
    fi
fi

# 6. Cache optimization
echo "6. Optimizing cache..."
python scripts/optimize_cache.py

# 7. Index optimization
echo "7. Optimizing indexes..."
python scripts/optimize_indexes.py

# 8. Security audit
echo "8. Running security audit..."
python scripts/security_audit.py > reports/weekly_security_$(date +%Y%m%d).txt

# 9. Backup verification
echo "9. Verifying backups..."
python scripts/verify_backups.py

# 10. Update system packages (if approved)
echo "10. Checking for system updates..."
# This would typically be handled by system administrators

echo
echo "=== Weekly maintenance completed ==="
echo "Reports generated in reports/ directory"
```

### Weekly Tasks Checklist

- [ ] Review system health reports
- [ ] Analyze performance trends
- [ ] Update knowledge base with new papers
- [ ] Optimize database indexes
- [ ] Clean up old log files
- [ ] Verify backup integrity
- [ ] Review security audit results
- [ ] Update documentation if needed
- [ ] Plan capacity adjustments if needed
- [ ] Review and update monitoring thresholds

## Monthly Procedures

### Monthly Maintenance Tasks

```bash
#!/bin/bash
# monthly_maintenance.sh

echo "=== Monthly Maintenance - $(date) ==="

# 1. Comprehensive system audit
echo "1. Performing comprehensive system audit..."
python scripts/system_audit.py > reports/monthly_audit_$(date +%Y%m).txt

# 2. Capacity planning analysis
echo "2. Analyzing capacity requirements..."
python scripts/capacity_analysis.py > reports/monthly_capacity_$(date +%Y%m).txt

# 3. Full backup
echo "3. Creating full system backup..."
python scripts/full_backup.py --monthly

# 4. Security review
echo "4. Conducting security review..."
python scripts/security_review.py > reports/monthly_security_$(date +%Y%m).txt

# 5. Performance optimization
echo "5. Performing performance optimization..."
python scripts/performance_optimization.py

# 6. Knowledge base statistics
echo "6. Generating knowledge base statistics..."
python scripts/kb_statistics.py > reports/monthly_kb_stats_$(date +%Y%m).txt

# 7. User activity analysis
echo "7. Analyzing user activity..."
python scripts/user_activity_analysis.py > reports/monthly_users_$(date +%Y%m).txt

# 8. System updates planning
echo "8. Planning system updates..."
python scripts/update_planning.py > reports/monthly_updates_$(date +%Y%m).txt

echo
echo "=== Monthly maintenance completed ==="
```

### Monthly Review Meeting Agenda

1. **System Performance Review**
   - Response time trends
   - Throughput analysis
   - Resource utilization
   - Capacity planning

2. **Reliability and Availability**
   - Uptime statistics
   - Incident summary
   - Error rate analysis
   - Recovery procedures effectiveness

3. **Security Review**
   - Security audit results
   - Access control review
   - Vulnerability assessment
   - Compliance status

4. **Knowledge Base Management**
   - Content growth statistics
   - Quality metrics
   - User feedback analysis
   - Content curation needs

5. **Operational Efficiency**
   - Automation opportunities
   - Process improvements
   - Tool effectiveness
   - Training needs

## Incident Response

### Incident Classification

**Severity 1 (Critical)**
- System completely down
- Data corruption or loss
- Security breach
- Response time: 15 minutes

**Severity 2 (High)**
- Significant performance degradation
- Partial system functionality loss
- Database connectivity issues
- Response time: 1 hour

**Severity 3 (Medium)**
- Minor performance issues
- Non-critical feature failures
- Intermittent errors
- Response time: 4 hours

**Severity 4 (Low)**
- Cosmetic issues
- Documentation problems
- Enhancement requests
- Response time: 24 hours

### Incident Response Procedure

```bash
#!/bin/bash
# incident_response.sh

INCIDENT_ID="$1"
SEVERITY="$2"
DESCRIPTION="$3"

if [[ -z "$INCIDENT_ID" || -z "$SEVERITY" || -z "$DESCRIPTION" ]]; then
    echo "Usage: $0 <incident_id> <severity> <description>"
    echo "Severity: 1=Critical, 2=High, 3=Medium, 4=Low"
    exit 1
fi

INCIDENT_DIR="incidents/$INCIDENT_ID"
mkdir -p "$INCIDENT_DIR"

echo "=== Incident Response Started ==="
echo "Incident ID: $INCIDENT_ID"
echo "Severity: $SEVERITY"
echo "Description: $DESCRIPTION"
echo "Start time: $(date)"

# 1. Initial assessment
echo "1. Performing initial assessment..."
python scripts/system_assessment.py > "$INCIDENT_DIR/initial_assessment.txt"

# 2. Collect system information
echo "2. Collecting system information..."
python scripts/collect_diagnostics.py > "$INCIDENT_DIR/diagnostics.txt"

# 3. Collect logs
echo "3. Collecting relevant logs..."
cp logs/lightrag_errors.log "$INCIDENT_DIR/"
tail -1000 logs/lightrag.log > "$INCIDENT_DIR/recent_logs.txt"

# 4. Take system snapshot
echo "4. Taking system snapshot..."
python scripts/system_snapshot.py > "$INCIDENT_DIR/system_snapshot.txt"

# 5. Notify stakeholders based on severity
echo "5. Notifying stakeholders..."
case $SEVERITY in
    1)
        echo "Critical incident - notifying all stakeholders"
        python scripts/notify_stakeholders.py --severity critical --incident "$INCIDENT_ID"
        ;;
    2)
        echo "High severity incident - notifying operations team"
        python scripts/notify_stakeholders.py --severity high --incident "$INCIDENT_ID"
        ;;
    *)
        echo "Lower severity incident - logging for review"
        ;;
esac

# 6. Create incident log
cat > "$INCIDENT_DIR/incident_log.txt" << EOF
Incident ID: $INCIDENT_ID
Severity: $SEVERITY
Description: $DESCRIPTION
Start Time: $(date)
Status: INVESTIGATING

Timeline:
$(date): Incident reported
$(date): Initial assessment completed
$(date): Diagnostics collected
$(date): Stakeholders notified

Next Steps:
- Analyze collected data
- Implement immediate fixes if available
- Monitor system stability
- Update stakeholders on progress
EOF

echo
echo "=== Incident response initiated ==="
echo "Incident directory: $INCIDENT_DIR"
echo "Next: Analyze collected data and implement fixes"
```

### Incident Communication Template

```
Subject: [INCIDENT-{ID}] {SEVERITY} - {BRIEF_DESCRIPTION}

Incident Details:
- Incident ID: {INCIDENT_ID}
- Severity: {SEVERITY}
- Start Time: {START_TIME}
- Status: {STATUS}

Description:
{DETAILED_DESCRIPTION}

Impact:
{IMPACT_DESCRIPTION}

Current Actions:
{CURRENT_ACTIONS}

Next Update:
{NEXT_UPDATE_TIME}

Contact:
{INCIDENT_COMMANDER_CONTACT}
```

## Emergency Procedures

### System Down Emergency

```bash
#!/bin/bash
# emergency_system_down.sh

echo "=== EMERGENCY: System Down Response ==="
echo "Start time: $(date)"

# 1. Immediate assessment
echo "1. Checking system status..."
if ! pgrep -f "python.*main.py" > /dev/null; then
    echo "   Application is not running"
    
    # Check if it's a simple restart issue
    echo "2. Attempting restart..."
    cd /opt/clinical-metabolomics-oracle
    source venv/bin/activate
    
    # Try to start application
    timeout 30 python src/main.py &
    APP_PID=$!
    
    sleep 10
    
    if kill -0 $APP_PID 2>/dev/null; then
        echo "   ✓ Application restarted successfully"
        exit 0
    else
        echo "   ✗ Application failed to start"
    fi
fi

# 2. Check dependencies
echo "3. Checking dependencies..."
if ! pg_isready -q; then
    echo "   PostgreSQL is not accessible"
    echo "   Action: Check PostgreSQL service"
    sudo systemctl status postgresql
fi

if command -v cypher-shell &> /dev/null; then
    if ! echo "RETURN 1" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD" &> /dev/null; then
        echo "   Neo4j is not accessible"
        echo "   Action: Check Neo4j service"
        sudo systemctl status neo4j
    fi
fi

# 3. Check system resources
echo "4. Checking system resources..."
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
DISK_USAGE=$(df . | awk 'NR==2 {print $5}' | sed 's/%//')

echo "   Memory usage: ${MEMORY_USAGE}%"
echo "   Disk usage: ${DISK_USAGE}%"

if (( $(echo "$MEMORY_USAGE > 95" | bc -l) )); then
    echo "   ⚠ Critical memory usage - system may be out of memory"
fi

if [[ $DISK_USAGE -gt 95 ]]; then
    echo "   ⚠ Critical disk usage - system may be out of space"
fi

# 4. Collect emergency diagnostics
echo "5. Collecting emergency diagnostics..."
EMERGENCY_DIR="emergency_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EMERGENCY_DIR"

# System information
uname -a > "$EMERGENCY_DIR/system_info.txt"
free -h > "$EMERGENCY_DIR/memory_info.txt"
df -h > "$EMERGENCY_DIR/disk_info.txt"
ps aux > "$EMERGENCY_DIR/processes.txt"

# Recent logs
if [[ -f "logs/lightrag_errors.log" ]]; then
    tail -100 logs/lightrag_errors.log > "$EMERGENCY_DIR/recent_errors.txt"
fi

# 5. Attempt recovery
echo "6. Attempting emergency recovery..."

# Kill any hung processes
pkill -f "python.*main.py"
sleep 5

# Clear any locks or temporary files
rm -f .app.pid
rm -f /tmp/lightrag_*

# Try service restart if available
if systemctl is-enabled lightrag-oracle &> /dev/null; then
    echo "   Attempting service restart..."
    sudo systemctl restart lightrag-oracle
    sleep 10
    
    if systemctl is-active lightrag-oracle &> /dev/null; then
        echo "   ✓ Service restarted successfully"
        exit 0
    fi
fi

# 7. Escalate if recovery failed
echo "7. Emergency recovery failed - escalating..."
echo "   Emergency diagnostics saved to: $EMERGENCY_DIR"
echo "   Contact: System Administrator immediately"
echo "   Phone: [EMERGENCY_PHONE]"
echo "   Email: [EMERGENCY_EMAIL]"

# Send emergency notification
python scripts/emergency_notification.py \
    --type "system_down" \
    --diagnostics "$EMERGENCY_DIR" \
    --timestamp "$(date)"

echo
echo "=== Emergency response completed ==="
echo "Status: ESCALATED"
```

### Data Corruption Emergency

```bash
#!/bin/bash
# emergency_data_corruption.sh

echo "=== EMERGENCY: Data Corruption Response ==="
echo "Start time: $(date)"

# 1. Immediate isolation
echo "1. Isolating system..."
sudo systemctl stop lightrag-oracle
echo "   System stopped to prevent further corruption"

# 2. Assess corruption extent
echo "2. Assessing corruption extent..."
CORRUPTION_DIR="corruption_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CORRUPTION_DIR"

# Check knowledge graph integrity
if [[ -d "data/lightrag_kg" ]]; then
    echo "   Checking knowledge graph integrity..."
    # Add specific integrity checks here
fi

# Check vector store integrity
if [[ -d "data/lightrag_vectors" ]]; then
    echo "   Checking vector store integrity..."
    # Add specific integrity checks here
fi

# Check database integrity
echo "   Checking database integrity..."
sudo -u postgres pg_dump clinical_metabolomics_oracle > "$CORRUPTION_DIR/db_dump.sql" 2>&1
if [[ $? -ne 0 ]]; then
    echo "   ⚠ Database corruption detected"
fi

# 3. Locate latest good backup
echo "3. Locating latest good backup..."
if [[ -f ".last_backup" ]]; then
    BACKUP_PATH=$(cat .last_backup)
    echo "   Latest backup: $BACKUP_PATH"
    
    if [[ -d "$BACKUP_PATH" ]]; then
        echo "   Backup is accessible"
    else
        echo "   ⚠ Backup directory not found"
    fi
else
    echo "   ⚠ No backup information found"
fi

# 4. Immediate notification
echo "4. Sending emergency notifications..."
python scripts/emergency_notification.py \
    --type "data_corruption" \
    --corruption_dir "$CORRUPTION_DIR" \
    --backup_path "$BACKUP_PATH" \
    --timestamp "$(date)"

echo
echo "=== Data corruption emergency response completed ==="
echo "Status: SYSTEM ISOLATED - AWAITING RECOVERY DECISION"
echo "Corruption assessment: $CORRUPTION_DIR"
echo "Contact: Data Recovery Team immediately"
```

## Change Management

### Change Request Process

1. **Change Request Submission**
   - Complete change request form
   - Include impact assessment
   - Specify rollback plan
   - Get stakeholder approval

2. **Change Review**
   - Technical review by operations team
   - Risk assessment
   - Resource allocation
   - Schedule approval

3. **Change Implementation**
   - Pre-change backup
   - Implementation during maintenance window
   - Post-change verification
   - Documentation update

4. **Change Closure**
   - Success verification
   - Stakeholder notification
   - Lessons learned documentation
   - Process improvement recommendations

### Change Implementation Template

```bash
#!/bin/bash
# change_implementation.sh

CHANGE_ID="$1"
CHANGE_TYPE="$2"
DESCRIPTION="$3"

echo "=== Change Implementation: $CHANGE_ID ==="
echo "Type: $CHANGE_TYPE"
echo "Description: $DESCRIPTION"
echo "Start time: $(date)"

# 1. Pre-change backup
echo "1. Creating pre-change backup..."
python scripts/create_backup.py --change-id "$CHANGE_ID"

# 2. Pre-change verification
echo "2. Pre-change system verification..."
python scripts/pre_change_verification.py > "changes/${CHANGE_ID}_pre_verification.txt"

# 3. Implementation
echo "3. Implementing change..."
case $CHANGE_TYPE in
    "configuration")
        echo "   Implementing configuration change..."
        # Configuration change logic
        ;;
    "code_deployment")
        echo "   Implementing code deployment..."
        # Code deployment logic
        ;;
    "database_migration")
        echo "   Implementing database migration..."
        # Database migration logic
        ;;
    *)
        echo "   Unknown change type: $CHANGE_TYPE"
        exit 1
        ;;
esac

# 4. Post-change verification
echo "4. Post-change system verification..."
python scripts/post_change_verification.py > "changes/${CHANGE_ID}_post_verification.txt"

# 5. Rollback if verification fails
if [[ $? -ne 0 ]]; then
    echo "5. Post-change verification failed - initiating rollback..."
    python scripts/rollback_change.py --change-id "$CHANGE_ID"
    exit 1
fi

# 6. Change completion
echo "5. Change completed successfully"
echo "End time: $(date)"

# 7. Notification
python scripts/change_notification.py \
    --change-id "$CHANGE_ID" \
    --status "completed" \
    --timestamp "$(date)"

echo
echo "=== Change implementation completed ==="
```

## Monitoring and Alerting

### Monitoring Setup

1. **System Metrics**
   - CPU utilization
   - Memory usage
   - Disk space
   - Network I/O

2. **Application Metrics**
   - Response times
   - Error rates
   - Query throughput
   - Cache hit rates

3. **Database Metrics**
   - Connection counts
   - Query performance
   - Lock contention
   - Replication lag

4. **Business Metrics**
   - User activity
   - Query patterns
   - Content usage
   - Satisfaction scores

### Alert Configuration

```yaml
# alerts.yaml
alerts:
  system:
    cpu_high:
      threshold: 80
      duration: 5m
      severity: warning
      notification: email
    
    memory_critical:
      threshold: 95
      duration: 2m
      severity: critical
      notification: email,sms
    
    disk_full:
      threshold: 90
      duration: 1m
      severity: critical
      notification: email,sms
  
  application:
    response_time_high:
      threshold: 10s
      duration: 5m
      severity: warning
      notification: email
    
    error_rate_high:
      threshold: 5%
      duration: 2m
      severity: critical
      notification: email,slack
    
    system_down:
      threshold: 0
      duration: 1m
      severity: critical
      notification: email,sms,slack
  
  database:
    connection_failed:
      threshold: 1
      duration: 1m
      severity: critical
      notification: email,sms
    
    query_slow:
      threshold: 30s
      duration: 5m
      severity: warning
      notification: email
```

This operational procedures document provides comprehensive coverage of all operational aspects of managing the LightRAG integration system.