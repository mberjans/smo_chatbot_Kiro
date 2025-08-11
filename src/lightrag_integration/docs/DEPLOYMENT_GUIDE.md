# LightRAG Integration Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the LightRAG integration in different environments. It covers automated deployment, manual deployment, containerized deployment, and cloud deployment options.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start Deployment](#quick-start-deployment)
3. [Environment-Specific Deployments](#environment-specific-deployments)
4. [Container Deployment](#container-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Configuration Management](#configuration-management)
7. [Post-Deployment Verification](#post-deployment-verification)
8. [Rollback Procedures](#rollback-procedures)
9. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

**Minimum Requirements:**
- OS: Ubuntu 20.04+ / CentOS 8+ / macOS 11+
- CPU: 4 cores, 2.5 GHz
- RAM: 8 GB
- Storage: 50 GB available space
- Network: Stable internet connection

**Recommended Requirements:**
- OS: Ubuntu 22.04 LTS
- CPU: 8+ cores, 3.0+ GHz
- RAM: 16+ GB
- Storage: 100+ GB SSD
- Network: High-speed internet connection

### Software Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip nodejs npm postgresql-client git curl

# CentOS/RHEL
sudo dnf install -y python3.11 python3-pip nodejs npm postgresql git curl

# macOS (using Homebrew)
brew install python@3.11 node postgresql git
```

### Database Requirements

- **PostgreSQL 12+**: For application data storage
- **Neo4j 4.4+**: For knowledge graph storage
- **Redis 6+** (optional): For caching

## Quick Start Deployment

### Automated Deployment

The fastest way to deploy is using the automated deployment script:

```bash
# Clone the repository
git clone <repository-url>
cd clinical-metabolomics-oracle

# Run automated deployment
./src/lightrag_integration/deployment/deploy.sh

# Or with specific environment
DEPLOYMENT_ENV=production ./src/lightrag_integration/deployment/deploy.sh
```

### Manual Quick Start

If you prefer manual control:

```bash
# 1. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
npm install

# 2. Configure environment
cp .env.example .env
# Edit .env with your configuration

# 3. Setup databases
createdb clinical_metabolomics_oracle
npx prisma migrate deploy

# 4. Create directories
mkdir -p data/lightrag_kg data/lightrag_vectors data/lightrag_cache papers logs

# 5. Start application
python src/main.py
```

## Environment-Specific Deployments

### Development Environment

```bash
# Create development configuration
python src/lightrag_integration/deployment/config_manager.py create development

# Deploy development environment
DEPLOYMENT_ENV=development ./src/lightrag_integration/deployment/deploy.sh

# Start in development mode
python src/main.py --debug
```

**Development Features:**
- Debug logging enabled
- Hot reload for code changes
- Reduced batch sizes for faster testing
- Local file storage
- Simplified monitoring

### Staging Environment

```bash
# Create staging configuration
python src/lightrag_integration/deployment/config_manager.py create staging

# Deploy staging environment
DEPLOYMENT_ENV=staging ./src/lightrag_integration/deployment/deploy.sh

# Verify staging deployment
curl -f http://staging-server:8000/health
```

**Staging Features:**
- Production-like configuration
- Full monitoring enabled
- Performance testing capabilities
- Backup and recovery testing
- Load testing environment

### Production Environment

```bash
# Create production configuration
python src/lightrag_integration/deployment/config_manager.py create production

# Validate configuration
python src/lightrag_integration/deployment/config_manager.py validate config/production.yaml

# Deploy production environment
DEPLOYMENT_ENV=production ./src/lightrag_integration/deployment/deploy.sh

# Enable and start service
sudo systemctl enable lightrag-oracle
sudo systemctl start lightrag-oracle
```

**Production Features:**
- Optimized performance settings
- Comprehensive monitoring and alerting
- Automated backups
- Security hardening
- High availability configuration

## Container Deployment

### Docker Compose Deployment

```bash
# Create environment file for Docker
cat > .env.docker << EOF
POSTGRES_PASSWORD=secure_password
NEO4J_PASSWORD=secure_password
GROQ_API_KEY=your_groq_api_key
OPENAI_API_KEY=your_openai_api_key
PERPLEXITY_API=your_perplexity_api_key
GRAFANA_PASSWORD=admin_password
EOF

# Deploy with Docker Compose
docker-compose -f src/lightrag_integration/deployment/docker-compose.yml up -d

# Check deployment status
docker-compose -f src/lightrag_integration/deployment/docker-compose.yml ps

# View logs
docker-compose -f src/lightrag_integration/deployment/docker-compose.yml logs -f lightrag-oracle
```

### Kubernetes Deployment

Create Kubernetes manifests:

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: lightrag-oracle
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: lightrag-config
  namespace: lightrag-oracle
data:
  LIGHTRAG_KG_PATH: "/app/data/lightrag_kg"
  LIGHTRAG_VECTOR_PATH: "/app/data/lightrag_vectors"
  LIGHTRAG_CACHE_PATH: "/app/data/lightrag_cache"
  LIGHTRAG_EMBEDDING_MODEL: "intfloat/e5-base-v2"
  LIGHTRAG_LLM_MODEL: "groq:Llama-3.3-70b-Versatile"
  ENABLE_MONITORING: "true"
  LOG_LEVEL: "INFO"
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: lightrag-secrets
  namespace: lightrag-oracle
type: Opaque
stringData:
  DATABASE_URL: "postgresql://user:password@postgres:5432/clinical_metabolomics_oracle"
  NEO4J_PASSWORD: "your_neo4j_password"
  GROQ_API_KEY: "your_groq_api_key"
  OPENAI_API_KEY: "your_openai_api_key"
  PERPLEXITY_API: "your_perplexity_api_key"
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lightrag-oracle
  namespace: lightrag-oracle
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lightrag-oracle
  template:
    metadata:
      labels:
        app: lightrag-oracle
    spec:
      containers:
      - name: lightrag-oracle
        image: lightrag-oracle:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: lightrag-config
        - secretRef:
            name: lightrag-secrets
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: papers-volume
          mountPath: /app/papers
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: lightrag-data-pvc
      - name: papers-volume
        persistentVolumeClaim:
          claimName: lightrag-papers-pvc
---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: lightrag-oracle-service
  namespace: lightrag-oracle
spec:
  selector:
    app: lightrag-oracle
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy to Kubernetes:

```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment
kubectl get pods -n lightrag-oracle
kubectl get services -n lightrag-oracle

# View logs
kubectl logs -f deployment/lightrag-oracle -n lightrag-oracle
```

## Cloud Deployment

### AWS Deployment

#### Using AWS ECS

```bash
# Create ECS task definition
aws ecs register-task-definition --cli-input-json file://aws/task-definition.json

# Create ECS service
aws ecs create-service \
  --cluster lightrag-cluster \
  --service-name lightrag-oracle \
  --task-definition lightrag-oracle:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

#### Using AWS Lambda (for serverless deployment)

```python
# lambda/handler.py
import json
import asyncio
from lightrag_integration.component import LightRAGComponent
from lightrag_integration.config.settings import LightRAGConfig

component = None

def lambda_handler(event, context):
    global component
    
    if component is None:
        config = LightRAGConfig()
        component = LightRAGComponent(config)
        asyncio.run(component.initialize())
    
    query = event.get('query', '')
    if not query:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'Query parameter is required'})
        }
    
    try:
        response = asyncio.run(component.query(query))
        return {
            'statusCode': 200,
            'body': json.dumps({
                'answer': response.answer,
                'confidence': response.confidence_score,
                'processing_time': response.processing_time
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Platform

#### Using Cloud Run

```yaml
# cloudbuild.yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/lightrag-oracle', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/lightrag-oracle']
- name: 'gcr.io/cloud-builders/gcloud'
  args:
  - 'run'
  - 'deploy'
  - 'lightrag-oracle'
  - '--image'
  - 'gcr.io/$PROJECT_ID/lightrag-oracle'
  - '--region'
  - 'us-central1'
  - '--platform'
  - 'managed'
  - '--allow-unauthenticated'
```

Deploy to Cloud Run:

```bash
# Build and deploy
gcloud builds submit --config cloudbuild.yaml

# Set environment variables
gcloud run services update lightrag-oracle \
  --set-env-vars="DATABASE_URL=postgresql://...,NEO4J_PASSWORD=..." \
  --region=us-central1
```

### Azure Deployment

#### Using Azure Container Instances

```bash
# Create resource group
az group create --name lightrag-rg --location eastus

# Deploy container
az container create \
  --resource-group lightrag-rg \
  --name lightrag-oracle \
  --image lightrag-oracle:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8000 \
  --environment-variables \
    DATABASE_URL="postgresql://..." \
    NEO4J_PASSWORD="..." \
    GROQ_API_KEY="..."
```

## Configuration Management

### Environment Configuration

Use the configuration manager to create and manage environment-specific configurations:

```bash
# Create configuration for each environment
python src/lightrag_integration/deployment/config_manager.py create development
python src/lightrag_integration/deployment/config_manager.py create staging
python src/lightrag_integration/deployment/config_manager.py create production

# Validate configurations
python src/lightrag_integration/deployment/config_manager.py validate config/production.yaml

# Generate environment files
python src/lightrag_integration/deployment/config_manager.py generate-env config/production.yaml --output .env.production

# Compare configurations
python src/lightrag_integration/deployment/config_manager.py compare config/staging.yaml config/production.yaml
```

### Secrets Management

#### Using HashiCorp Vault

```bash
# Store secrets in Vault
vault kv put secret/lightrag-oracle \
  database_url="postgresql://..." \
  neo4j_password="..." \
  groq_api_key="..." \
  openai_api_key="..." \
  perplexity_api="..."

# Retrieve secrets in deployment script
DATABASE_URL=$(vault kv get -field=database_url secret/lightrag-oracle)
```

#### Using Kubernetes Secrets

```bash
# Create secret from command line
kubectl create secret generic lightrag-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=neo4j-password="..." \
  --from-literal=groq-api-key="..." \
  -n lightrag-oracle
```

## Post-Deployment Verification

### Health Checks

```bash
#!/bin/bash
# post_deployment_verification.sh

echo "=== Post-Deployment Verification ==="

# 1. Check application health
echo "1. Checking application health..."
if curl -f http://localhost:8000/health; then
    echo "✓ Application is healthy"
else
    echo "✗ Application health check failed"
    exit 1
fi

# 2. Check database connections
echo "2. Checking database connections..."
if curl -f http://localhost:8000/health/database; then
    echo "✓ Database connections are healthy"
else
    echo "✗ Database connection check failed"
    exit 1
fi

# 3. Test query functionality
echo "3. Testing query functionality..."
RESPONSE=$(curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is clinical metabolomics?"}')

if echo "$RESPONSE" | grep -q "answer"; then
    echo "✓ Query functionality is working"
else
    echo "✗ Query functionality test failed"
    exit 1
fi

# 4. Check monitoring endpoints
echo "4. Checking monitoring endpoints..."
if curl -f http://localhost:8000/metrics; then
    echo "✓ Monitoring endpoints are accessible"
else
    echo "✗ Monitoring endpoints check failed"
fi

# 5. Verify file permissions
echo "5. Verifying file permissions..."
if [[ -w "data/lightrag_kg" && -w "data/lightrag_vectors" && -w "logs" ]]; then
    echo "✓ File permissions are correct"
else
    echo "✗ File permissions check failed"
    exit 1
fi

echo "=== Verification Complete ==="
```

### Performance Testing

```bash
#!/bin/bash
# performance_test.sh

echo "=== Performance Testing ==="

# Load testing with curl
echo "1. Running load test..."
for i in {1..10}; do
    START_TIME=$(date +%s.%N)
    curl -s -X POST http://localhost:8000/query \
      -H "Content-Type: application/json" \
      -d '{"query": "What is clinical metabolomics?"}' > /dev/null
    END_TIME=$(date +%s.%N)
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    echo "Request $i: ${DURATION}s"
done

# Memory usage check
echo "2. Checking memory usage..."
MEMORY_USAGE=$(ps -p $(pgrep -f "python.*main.py") -o %mem --no-headers)
echo "Memory usage: ${MEMORY_USAGE}%"

# Disk usage check
echo "3. Checking disk usage..."
DISK_USAGE=$(df -h . | awk 'NR==2 {print $5}')
echo "Disk usage: $DISK_USAGE"

echo "=== Performance Testing Complete ==="
```

## Rollback Procedures

### Automated Rollback

```bash
# Rollback to previous deployment
./src/lightrag_integration/deployment/deploy.sh rollback

# Or rollback to specific backup
BACKUP_PATH="/backups/20240109_020000" ./src/lightrag_integration/deployment/deploy.sh rollback
```

### Manual Rollback

```bash
#!/bin/bash
# manual_rollback.sh

BACKUP_PATH="$1"

if [[ -z "$BACKUP_PATH" ]]; then
    echo "Usage: $0 <backup_path>"
    exit 1
fi

echo "Rolling back to: $BACKUP_PATH"

# 1. Stop application
sudo systemctl stop lightrag-oracle

# 2. Restore data
rm -rf data/lightrag_kg data/lightrag_vectors
tar -xzf "$BACKUP_PATH/data.tar.gz"

# 3. Restore configuration
cp "$BACKUP_PATH/.env.backup" .env

# 4. Restore database (if needed)
if [[ -f "$BACKUP_PATH/postgres.sql" ]]; then
    psql clinical_metabolomics_oracle < "$BACKUP_PATH/postgres.sql"
fi

# 5. Start application
sudo systemctl start lightrag-oracle

echo "Rollback complete"
```

## Troubleshooting

### Common Deployment Issues

#### Issue: Application fails to start

**Symptoms:**
- Service fails to start
- Import errors in logs
- Configuration errors

**Solutions:**
```bash
# Check logs
sudo journalctl -u lightrag-oracle -f

# Validate configuration
python src/lightrag_integration/deployment/config_manager.py validate config/production.yaml

# Test imports
python -c "from lightrag_integration.component import LightRAGComponent; print('OK')"

# Check permissions
ls -la data/ logs/
```

#### Issue: Database connection failures

**Symptoms:**
- Database connection errors
- Migration failures
- Query timeouts

**Solutions:**
```bash
# Test database connections
pg_isready -h localhost -p 5432
echo "RETURN 1" | cypher-shell -u neo4j -p "$NEO4J_PASSWORD"

# Check database logs
tail -f /var/log/postgresql/postgresql-*.log

# Run migrations manually
npx prisma migrate deploy
```

#### Issue: High memory usage

**Symptoms:**
- Out of memory errors
- Slow performance
- System becomes unresponsive

**Solutions:**
```bash
# Reduce batch sizes in configuration
# Restart application to clear memory
sudo systemctl restart lightrag-oracle

# Monitor memory usage
watch -n 1 'ps -p $(pgrep -f "python.*main.py") -o pid,vsz,rss,pcpu,pmem'
```

### Deployment Validation Checklist

- [ ] System requirements met
- [ ] Dependencies installed
- [ ] Configuration validated
- [ ] Database connections working
- [ ] Application starts successfully
- [ ] Health checks pass
- [ ] Query functionality works
- [ ] Monitoring endpoints accessible
- [ ] File permissions correct
- [ ] Backup system configured
- [ ] Logging working properly
- [ ] Performance within acceptable limits

### Getting Help

1. **Check logs**: Review application and system logs
2. **Run diagnostics**: Use built-in diagnostic tools
3. **Validate configuration**: Ensure configuration is correct
4. **Test components**: Test individual components
5. **Contact support**: Provide logs and system information

---

This deployment guide provides comprehensive instructions for deploying the LightRAG integration in various environments and scenarios.