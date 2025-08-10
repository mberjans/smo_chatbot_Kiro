# Clinical Metabolomics Oracle - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Clinical Metabolomics Oracle (CMO) system in various environments, from local development to production deployment.

## 🚀 Quick Start (Recommended)

### Prerequisites
- Python 3.8+ installed
- Node.js 16+ (for Prisma)
- Git

### One-Command Setup
```bash
# Clone repository
git clone <repository-url>
cd clinical-metabolomics-oracle

# Install dependencies
pip install -r requirements.txt
npm install

# Setup environment
cp .env.example .env
# Edit .env with your configuration (see Configuration section)

# Initialize database
npx prisma generate
npx prisma migrate dev

# Start the application
python start_chatbot_uvicorn.py

# Access at http://localhost:8001/chat
```

## 📋 Detailed Setup Instructions

### 1. System Requirements

#### Minimum Requirements
- **CPU**: 2 cores, 2.0 GHz
- **RAM**: 4 GB
- **Storage**: 10 GB free space
- **Network**: Internet connection for AI services

#### Recommended Requirements
- **CPU**: 4+ cores, 2.5+ GHz
- **RAM**: 8+ GB
- **Storage**: 50+ GB SSD
- **Network**: High-speed internet for optimal AI performance

#### Supported Operating Systems
- **macOS**: 10.15+ (tested on macOS with M1/M2)
- **Linux**: Ubuntu 18.04+, CentOS 7+, Debian 10+
- **Windows**: Windows 10+ (with WSL2 recommended)

### 2. Environment Setup

#### Python Environment
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Verify installation
python -c "import chainlit; print('Chainlit version:', chainlit.__version__)"
```

#### Node.js Dependencies
```bash
# Install Node.js dependencies for Prisma
npm install

# Generate Prisma client
npx prisma generate
```

### 3. Database Configuration

#### PostgreSQL Setup
```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE cmo_db;
CREATE USER cmo_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE cmo_db TO cmo_user;
\q
```

#### Neo4j Setup (Optional)
```bash
# Install Neo4j (for full LightRAG features)
# Ubuntu/Debian
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.4' | sudo tee -a /etc/apt/sources.list.d/neo4j.list
sudo apt update
sudo apt install neo4j

# Start Neo4j
sudo systemctl start neo4j
sudo systemctl enable neo4j

# Set password
sudo neo4j-admin set-initial-password your_neo4j_password
```

#### Database Migration
```bash
# Run database migrations
npx prisma migrate dev

# Verify database setup
npx prisma studio  # Opens database browser at http://localhost:5555
```

### 4. Configuration

#### Environment Variables (.env)
```bash
# Copy example configuration
cp .env.example .env

# Edit .env file with your settings
nano .env
```

#### Required Configuration
```bash
# Database connections
DATABASE_URL="postgresql://cmo_user:your_secure_password@localhost:5432/cmo_db"
NEO4J_PASSWORD="your_neo4j_password"

# Basic API keys (for fallback functionality)
PERPLEXITY_API="placeholder_key"
OPENAI_API_KEY="sk-placeholder_key"
GROQ_API_KEY="placeholder_key"
```

#### Optional Configuration (Enhanced Features)
```bash
# OpenRouter API key for Perplexity AI integration
OPENROUTER_API_KEY="your_openrouter_api_key"

# Performance tuning
MAX_CONCURRENT_REQUESTS=10
CACHE_TTL_SECONDS=3600
QUERY_TIMEOUT_SECONDS=30

# Monitoring
ENABLE_PERFORMANCE_MONITORING=true
LOG_LEVEL=INFO
```

### 5. Application Startup

#### Development Mode
```bash
# Simple development startup
python start_chatbot_uvicorn.py

# With auto-reload for development
uvicorn src.main_simple:app --host 0.0.0.0 --port 8001 --reload
```

#### Production Mode
```bash
# Production startup with Gunicorn
python start_chatbot_gunicorn.py

# Manual Gunicorn startup
gunicorn -c gunicorn.conf.py src.main_simple:app
```

#### Service Mode (Linux)
```bash
# Create systemd service file
sudo nano /etc/systemd/system/cmo.service
```

```ini
[Unit]
Description=Clinical Metabolomics Oracle
After=network.target postgresql.service

[Service]
Type=exec
User=cmo
Group=cmo
WorkingDirectory=/opt/clinical-metabolomics-oracle
Environment=PATH=/opt/clinical-metabolomics-oracle/venv/bin
ExecStart=/opt/clinical-metabolomics-oracle/venv/bin/python start_chatbot_gunicorn.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable cmo
sudo systemctl start cmo
sudo systemctl status cmo
```

## 🐳 Docker Deployment

### Docker Compose (Recommended)
```yaml
# docker-compose.yml
version: '3.8'

services:
  cmo-app:
    build: .
    ports:
      - "8001:8001"
    environment:
      - DATABASE_URL=postgresql://cmo_user:password@postgres:5432/cmo_db
      - NEO4J_PASSWORD=neo4j_password
    depends_on:
      - postgres
      - neo4j
    volumes:
      - ./papers:/app/papers
      - ./data:/app/data

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=cmo_db
      - POSTGRES_USER=cmo_user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  neo4j:
    image: neo4j:5.15
    environment:
      - NEO4J_AUTH=neo4j/neo4j_password
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data

volumes:
  postgres_data:
  neo4j_data:
```

```bash
# Deploy with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f cmo-app

# Stop deployment
docker-compose down
```

### Standalone Docker
```bash
# Build image
docker build -t cmo:latest .

# Run container
docker run -d \
  --name cmo \
  -p 8001:8001 \
  -e DATABASE_URL="your_database_url" \
  -e NEO4J_PASSWORD="your_neo4j_password" \
  -v $(pwd)/papers:/app/papers \
  -v $(pwd)/data:/app/data \
  cmo:latest
```

## ☁️ Cloud Deployment

### AWS Deployment

#### EC2 Instance
```bash
# Launch EC2 instance (t3.medium or larger recommended)
# Install dependencies
sudo yum update -y
sudo yum install -y python3 python3-pip nodejs npm postgresql-server

# Clone and setup application
git clone <repository-url>
cd clinical-metabolomics-oracle
pip3 install -r requirements.txt
npm install

# Configure environment
cp .env.example .env
# Edit .env with RDS and other AWS service endpoints

# Start application
python3 start_chatbot_gunicorn.py
```

#### RDS PostgreSQL
```bash
# Create RDS PostgreSQL instance
aws rds create-db-instance \
  --db-instance-identifier cmo-postgres \
  --db-instance-class db.t3.micro \
  --engine postgres \
  --master-username cmo_user \
  --master-user-password your_secure_password \
  --allocated-storage 20

# Update DATABASE_URL in .env
DATABASE_URL="postgresql://cmo_user:password@cmo-postgres.region.rds.amazonaws.com:5432/cmo_db"
```

### Google Cloud Platform

#### Cloud Run Deployment
```bash
# Build and deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/cmo
gcloud run deploy cmo \
  --image gcr.io/PROJECT_ID/cmo \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8001
```

### Azure Deployment

#### Container Instances
```bash
# Deploy to Azure Container Instances
az container create \
  --resource-group cmo-rg \
  --name cmo-container \
  --image your-registry/cmo:latest \
  --ports 8001 \
  --environment-variables \
    DATABASE_URL="your_database_url" \
    NEO4J_PASSWORD="your_neo4j_password"
```

## 🔧 Configuration Management

### Environment-Specific Configuration

#### Development (.env.development)
```bash
LOG_LEVEL=DEBUG
ENABLE_PERFORMANCE_MONITORING=true
CACHE_TTL_SECONDS=300
MAX_CONCURRENT_REQUESTS=5
```

#### Staging (.env.staging)
```bash
LOG_LEVEL=INFO
ENABLE_PERFORMANCE_MONITORING=true
CACHE_TTL_SECONDS=1800
MAX_CONCURRENT_REQUESTS=8
```

#### Production (.env.production)
```bash
LOG_LEVEL=WARNING
ENABLE_PERFORMANCE_MONITORING=true
CACHE_TTL_SECONDS=3600
MAX_CONCURRENT_REQUESTS=10
```

### Configuration Validation
```bash
# Validate configuration
python -c "
from src.lightrag_integration.config.settings import LightRAGConfig
config = LightRAGConfig.from_env()
config.validate()
print('Configuration valid!')
"
```

## 📊 Monitoring Setup

### Health Checks
```bash
# Basic health check
curl http://localhost:8001/health

# Detailed health check
curl http://localhost:8001/health/detailed

# Metrics endpoint
curl http://localhost:8001/metrics
```

### Log Monitoring
```bash
# View application logs
tail -f src/lightrag_integration/logs/lightrag.log

# View error logs
grep "ERROR" src/lightrag_integration/logs/lightrag.log

# Monitor performance
grep "Response time" src/lightrag_integration/logs/lightrag.log | tail -20
```

### Performance Monitoring
```bash
# System resource monitoring
htop
iostat -x 1
free -h

# Application-specific monitoring
python -c "
from src.lightrag_integration.monitoring import PerformanceMonitor
monitor = PerformanceMonitor({})
print(monitor.get_current_metrics())
"
```

## 🔒 Security Configuration

### SSL/TLS Setup
```bash
# Generate self-signed certificate for development
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Start with HTTPS
uvicorn src.main_simple:app --host 0.0.0.0 --port 8001 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

### Firewall Configuration
```bash
# Ubuntu/Debian firewall setup
sudo ufw allow 8001/tcp
sudo ufw allow ssh
sudo ufw enable

# CentOS/RHEL firewall setup
sudo firewall-cmd --permanent --add-port=8001/tcp
sudo firewall-cmd --reload
```

### API Key Security
```bash
# Secure .env file permissions
chmod 600 .env

# Use environment variables in production
export OPENROUTER_API_KEY="your_secure_key"
export DATABASE_URL="your_secure_connection_string"
```

## 🧪 Testing Deployment

### Automated Testing
```bash
# Run comprehensive tests
python test_chatbot_with_pdf.py
python test_chatbot_perplexity.py
python test_openrouter_setup.py

# Run all tests
python -m pytest tests/ -v
```

### Manual Testing
```bash
# Test web interface
open http://localhost:8001/chat

# Test API endpoints
curl -X POST http://localhost:8001/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is metabolomics?"}'

# Test health endpoints
curl http://localhost:8001/health
curl http://localhost:8001/metrics
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load test
locust -f tests/load_test.py --host http://localhost:8001
```

## 🚨 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port 8001
lsof -i :8001

# Kill process
kill -9 <PID>

# Or use the stop script
python stop_chatbot_server.py
```

#### Database Connection Issues
```bash
# Test PostgreSQL connection
psql -h localhost -U cmo_user -d cmo_db

# Check database status
sudo systemctl status postgresql

# Reset database
npx prisma migrate reset
```

#### Missing Dependencies
```bash
# Reinstall Python dependencies
pip install -r requirements.txt --force-reinstall

# Reinstall Node.js dependencies
rm -rf node_modules package-lock.json
npm install
```

#### Permission Issues
```bash
# Fix file permissions
chmod +x start_chatbot_uvicorn.py
chmod +x start_chatbot_gunicorn.py

# Fix directory permissions
chmod -R 755 papers/
chmod -R 755 data/
```

### Log Analysis
```bash
# Check application logs
tail -f src/lightrag_integration/logs/lightrag.log

# Check system logs
sudo journalctl -u cmo -f

# Check Docker logs
docker-compose logs -f cmo-app
```

### Performance Issues
```bash
# Monitor system resources
top
htop
iotop

# Check disk space
df -h

# Monitor network
netstat -tuln
ss -tuln
```

## 📈 Scaling Considerations

### Horizontal Scaling
```bash
# Load balancer configuration (nginx)
upstream cmo_backend {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 80;
    location / {
        proxy_pass http://cmo_backend;
    }
}
```

### Database Scaling
```bash
# PostgreSQL read replicas
# Configure in postgresql.conf
wal_level = replica
max_wal_senders = 3
wal_keep_segments = 64

# Create read replica
pg_basebackup -h master_host -D /var/lib/postgresql/replica -U replication -v -P -W
```

### Caching Optimization
```bash
# Redis setup for distributed caching
sudo apt install redis-server
redis-cli ping

# Update configuration for Redis
REDIS_URL="redis://localhost:6379"
```

## 🔄 Maintenance

### Regular Maintenance Tasks
```bash
# Update dependencies
pip install -r requirements.txt --upgrade
npm update

# Database maintenance
npx prisma migrate deploy
VACUUM ANALYZE;  # PostgreSQL maintenance

# Log rotation
logrotate /etc/logrotate.d/cmo

# Backup data
pg_dump cmo_db > backup_$(date +%Y%m%d).sql
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/
```

### Monitoring Alerts
```bash
# Setup monitoring alerts
# Configure in src/lightrag_integration/deployment/alert_rules.yml

# Test alerts
python -c "
from src.lightrag_integration.monitoring import PerformanceMonitor
monitor = PerformanceMonitor({})
monitor.trigger_test_alert()
"
```

## 📞 Support

### Getting Help
- **Documentation**: Check `src/lightrag_integration/docs/`
- **Logs**: Review application and system logs
- **Health Checks**: Use `/health` and `/metrics` endpoints
- **Testing**: Run diagnostic scripts in `test_*.py`

### Reporting Issues
1. Check logs for error messages
2. Run diagnostic tests
3. Gather system information
4. Create detailed issue report

---

*This deployment guide is maintained by the CMO development team. Last updated: January 9, 2025*