#!/bin/bash

# LightRAG Integration Deployment Script
# This script handles the deployment of the LightRAG integration system

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
BACKUP_DIR="${BACKUP_DIR:-/backups}"
LOG_FILE="${LOG_FILE:-/var/log/lightrag-deploy.log}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" | tee -a "$LOG_FILE"
    exit 1
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a "$LOG_FILE"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        error "This script should not be run as root for security reasons"
    fi
}

# Check system requirements
check_requirements() {
    log "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        error "Python 3 is required but not installed"
    fi
    
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
        error "Python 3.8+ is required, found $PYTHON_VERSION"
    fi
    
    # Check Node.js version
    if ! command -v node &> /dev/null; then
        error "Node.js is required but not installed"
    fi
    
    NODE_VERSION=$(node -v | sed 's/v//')
    if [[ $(echo "$NODE_VERSION < 16.0.0" | bc -l) -eq 1 ]]; then
        error "Node.js 16+ is required, found $NODE_VERSION"
    fi
    
    # Check PostgreSQL
    if ! command -v psql &> /dev/null; then
        warning "PostgreSQL client not found, database operations may fail"
    fi
    
    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [[ $AVAILABLE_SPACE -lt 10485760 ]]; then  # 10GB in KB
        warning "Less than 10GB available disk space"
    fi
    
    success "System requirements check completed"
}

# Create backup before deployment
create_backup() {
    log "Creating backup before deployment..."
    
    BACKUP_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_PATH="$BACKUP_DIR/lightrag_backup_$BACKUP_TIMESTAMP"
    
    mkdir -p "$BACKUP_PATH"
    
    # Backup current installation
    if [[ -d "$PROJECT_ROOT/data" ]]; then
        log "Backing up data directory..."
        tar -czf "$BACKUP_PATH/data.tar.gz" -C "$PROJECT_ROOT" data/
    fi
    
    # Backup configuration
    if [[ -d "$PROJECT_ROOT/config" ]]; then
        log "Backing up configuration..."
        tar -czf "$BACKUP_PATH/config.tar.gz" -C "$PROJECT_ROOT" config/
    fi
    
    # Backup environment file
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        cp "$PROJECT_ROOT/.env" "$BACKUP_PATH/.env.backup"
    fi
    
    # Create backup manifest
    cat > "$BACKUP_PATH/manifest.txt" << EOF
Backup created: $(date)
Environment: $DEPLOYMENT_ENV
Project root: $PROJECT_ROOT
Python version: $(python3 --version)
Node version: $(node --version)
Git commit: $(cd "$PROJECT_ROOT" && git rev-parse HEAD 2>/dev/null || echo "N/A")
EOF
    
    echo "$BACKUP_PATH" > "$PROJECT_ROOT/.last_backup"
    success "Backup created at $BACKUP_PATH"
}

# Setup virtual environment
setup_venv() {
    log "Setting up Python virtual environment..."
    
    cd "$PROJECT_ROOT"
    
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
    fi
    
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
    else
        error "requirements.txt not found"
    fi
    
    success "Virtual environment setup completed"
}

# Install Node.js dependencies
install_node_deps() {
    log "Installing Node.js dependencies..."
    
    cd "$PROJECT_ROOT"
    
    if [[ -f "package.json" ]]; then
        npm ci --production
    else
        warning "package.json not found, skipping Node.js dependencies"
    fi
    
    success "Node.js dependencies installed"
}

# Setup directories and permissions
setup_directories() {
    log "Setting up directories and permissions..."
    
    cd "$PROJECT_ROOT"
    
    # Create necessary directories
    mkdir -p data/lightrag_kg
    mkdir -p data/lightrag_vectors
    mkdir -p data/lightrag_cache
    mkdir -p papers/
    mkdir -p logs/
    mkdir -p config/
    
    # Set permissions
    chmod 755 data/
    chmod 755 papers/
    chmod 755 logs/
    chmod 755 config/
    
    # Create log files
    touch logs/lightrag.log
    touch logs/lightrag_errors.log
    chmod 644 logs/*.log
    
    success "Directories and permissions setup completed"
}

# Configure environment
configure_environment() {
    log "Configuring environment..."
    
    cd "$PROJECT_ROOT"
    
    # Copy environment template if .env doesn't exist
    if [[ ! -f ".env" && -f ".env.example" ]]; then
        cp .env.example .env
        warning "Created .env from template. Please update with your configuration."
    fi
    
    # Validate required environment variables
    if [[ -f ".env" ]]; then
        source .env
        
        REQUIRED_VARS=(
            "DATABASE_URL"
            "NEO4J_PASSWORD"
            "GROQ_API_KEY"
        )
        
        for var in "${REQUIRED_VARS[@]}"; do
            if [[ -z "${!var}" ]]; then
                warning "Required environment variable $var is not set"
            fi
        done
    else
        warning ".env file not found. Please create one with required configuration."
    fi
    
    success "Environment configuration completed"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Generate Prisma client
    if command -v npx &> /dev/null; then
        npx prisma generate
        
        # Run migrations
        npx prisma migrate deploy
    else
        warning "npx not available, skipping database migrations"
    fi
    
    success "Database migrations completed"
}

# Test deployment
test_deployment() {
    log "Testing deployment..."
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Test Python imports
    python3 -c "
import sys
sys.path.append('src')
try:
    from lightrag_integration.component import LightRAGComponent
    from lightrag_integration.config.settings import LightRAGConfig
    print('✓ LightRAG imports successful')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
"
    
    # Test configuration loading
    python3 -c "
import sys
sys.path.append('src')
try:
    from lightrag_integration.config.settings import LightRAGConfig
    config = LightRAGConfig()
    print('✓ Configuration loading successful')
except Exception as e:
    print(f'✗ Configuration error: {e}')
    sys.exit(1)
"
    
    success "Deployment tests passed"
}

# Setup systemd service
setup_service() {
    log "Setting up systemd service..."
    
    if [[ "$DEPLOYMENT_ENV" != "production" ]]; then
        log "Skipping service setup for non-production environment"
        return
    fi
    
    SERVICE_FILE="/etc/systemd/system/lightrag-oracle.service"
    
    # Check if we can write to systemd directory
    if [[ ! -w "/etc/systemd/system/" ]]; then
        warning "Cannot write to /etc/systemd/system/. Please run with sudo or manually create service file."
        log "Service file template available at: $SCRIPT_DIR/lightrag-oracle.service"
        return
    fi
    
    # Create service file
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Clinical Metabolomics Oracle with LightRAG
After=network.target postgresql.service neo4j.service
Wants=postgresql.service neo4j.service

[Service]
Type=simple
User=$(whoami)
Group=$(id -gn)
WorkingDirectory=$PROJECT_ROOT
Environment=PATH=$PROJECT_ROOT/venv/bin
ExecStart=$PROJECT_ROOT/venv/bin/python src/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$PROJECT_ROOT

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    sudo systemctl daemon-reload
    sudo systemctl enable lightrag-oracle
    
    success "Systemd service setup completed"
}

# Start services
start_services() {
    log "Starting services..."
    
    if [[ "$DEPLOYMENT_ENV" == "production" ]]; then
        if systemctl is-enabled lightrag-oracle &> /dev/null; then
            sudo systemctl start lightrag-oracle
            sleep 5
            
            if systemctl is-active lightrag-oracle &> /dev/null; then
                success "LightRAG Oracle service started successfully"
            else
                error "Failed to start LightRAG Oracle service"
            fi
        else
            warning "Service not enabled, starting manually..."
            cd "$PROJECT_ROOT"
            source venv/bin/activate
            nohup python src/main.py > logs/app.log 2>&1 &
            echo $! > .app.pid
            success "Application started in background (PID: $(cat .app.pid))"
        fi
    else
        log "Development environment detected, not starting services automatically"
        log "To start manually: cd $PROJECT_ROOT && source venv/bin/activate && python src/main.py"
    fi
}

# Cleanup function
cleanup() {
    log "Performing cleanup..."
    
    # Remove temporary files
    find "$PROJECT_ROOT" -name "*.pyc" -delete
    find "$PROJECT_ROOT" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    
    # Clean old log files (keep last 30 days)
    find "$PROJECT_ROOT/logs" -name "*.log.*" -mtime +30 -delete 2>/dev/null || true
    
    success "Cleanup completed"
}

# Rollback function
rollback() {
    log "Rolling back deployment..."
    
    if [[ -f "$PROJECT_ROOT/.last_backup" ]]; then
        BACKUP_PATH=$(cat "$PROJECT_ROOT/.last_backup")
        
        if [[ -d "$BACKUP_PATH" ]]; then
            log "Restoring from backup: $BACKUP_PATH"
            
            # Stop services
            if systemctl is-active lightrag-oracle &> /dev/null; then
                sudo systemctl stop lightrag-oracle
            fi
            
            # Restore data
            if [[ -f "$BACKUP_PATH/data.tar.gz" ]]; then
                rm -rf "$PROJECT_ROOT/data"
                tar -xzf "$BACKUP_PATH/data.tar.gz" -C "$PROJECT_ROOT"
            fi
            
            # Restore configuration
            if [[ -f "$BACKUP_PATH/config.tar.gz" ]]; then
                rm -rf "$PROJECT_ROOT/config"
                tar -xzf "$BACKUP_PATH/config.tar.gz" -C "$PROJECT_ROOT"
            fi
            
            # Restore environment
            if [[ -f "$BACKUP_PATH/.env.backup" ]]; then
                cp "$BACKUP_PATH/.env.backup" "$PROJECT_ROOT/.env"
            fi
            
            # Restart services
            if systemctl is-enabled lightrag-oracle &> /dev/null; then
                sudo systemctl start lightrag-oracle
            fi
            
            success "Rollback completed"
        else
            error "Backup directory not found: $BACKUP_PATH"
        fi
    else
        error "No backup information found"
    fi
}

# Main deployment function
deploy() {
    log "Starting LightRAG Integration deployment..."
    log "Environment: $DEPLOYMENT_ENV"
    log "Project root: $PROJECT_ROOT"
    
    check_root
    check_requirements
    create_backup
    setup_venv
    install_node_deps
    setup_directories
    configure_environment
    run_migrations
    test_deployment
    setup_service
    start_services
    cleanup
    
    success "Deployment completed successfully!"
    log "Application should be available at: http://localhost:8000"
    log "Logs are available at: $PROJECT_ROOT/logs/"
    log "Backup created at: $(cat "$PROJECT_ROOT/.last_backup" 2>/dev/null || echo "N/A")"
}

# Help function
show_help() {
    cat << EOF
LightRAG Integration Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy          Full deployment (default)
    rollback        Rollback to last backup
    test            Test deployment without starting services
    backup          Create backup only
    help            Show this help message

Environment Variables:
    DEPLOYMENT_ENV  Deployment environment (development|staging|production)
    BACKUP_DIR      Backup directory path (default: /backups)
    LOG_FILE        Log file path (default: /var/log/lightrag-deploy.log)

Examples:
    $0 deploy                           # Full deployment
    DEPLOYMENT_ENV=staging $0 deploy    # Deploy to staging
    $0 rollback                         # Rollback deployment
    $0 test                             # Test deployment
    $0 backup                           # Create backup only

EOF
}

# Parse command line arguments
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    rollback)
        rollback
        ;;
    test)
        check_root
        check_requirements
        setup_venv
        install_node_deps
        setup_directories
        configure_environment
        test_deployment
        success "Test deployment completed"
        ;;
    backup)
        create_backup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        error "Unknown command: $1. Use '$0 help' for usage information."
        ;;
esac