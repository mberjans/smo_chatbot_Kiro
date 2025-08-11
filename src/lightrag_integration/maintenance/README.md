# LightRAG Maintenance System

This directory contains the complete maintenance and administrative system for the LightRAG integration. The system provides comprehensive tools for managing the knowledge base, monitoring system health, and performing administrative operations.

## Components

### Core Components

#### 1. Update System (`update_system.py`)
- **Purpose**: Manages incremental updates to the knowledge base
- **Features**:
  - Automatic detection of new, modified, and deleted documents
  - Incremental processing without full rebuilds
  - Progress tracking and status reporting
  - Error handling with retry mechanisms
  - Integration with version control for rollback capabilities

#### 2. Version Control (`version_control.py`)
- **Purpose**: Provides version control and backup capabilities
- **Features**:
  - Automatic version creation with snapshots
  - Backup and restore functionality
  - Version history management
  - Compressed backup support
  - Automatic cleanup of old versions

#### 3. Administrative Interface (`admin_interface.py`)
- **Purpose**: Provides high-level administrative operations
- **Features**:
  - System status monitoring and health checks
  - Document management and curation
  - Administrative action tracking
  - Performance dashboard
  - Batch operations support

### API Layer

#### 4. REST API Endpoints (`api_endpoints.py`)
- **Purpose**: Provides REST API access to administrative functions
- **Features**:
  - Complete CRUD operations for documents
  - System status and health endpoints
  - Version management endpoints
  - Administrative action endpoints
  - Performance monitoring endpoints
  - Proper error handling and validation

#### 5. Integration Module (`integration.py`)
- **Purpose**: Coordinates all maintenance components
- **Features**:
  - Unified maintenance system setup
  - Dependency injection for API endpoints
  - Health check coordination
  - Graceful shutdown handling
  - Global instance management

## API Endpoints

### System Management
- `GET /admin/status` - Get comprehensive system status
- `GET /admin/health` - Simple health check
- `GET /admin/performance` - Performance dashboard data

### Document Management
- `GET /admin/documents` - List documents with pagination and search
- `GET /admin/documents/{id}` - Get specific document details
- `DELETE /admin/documents/{id}` - Remove document from knowledge base
- `POST /admin/documents/{id}/metadata` - Update document metadata

### Version Management
- `GET /admin/versions` - List available versions
- `GET /admin/versions/{id}` - Get specific version details
- `DELETE /admin/versions/{id}` - Delete version and backup

### Administrative Actions
- `POST /admin/update` - Trigger manual knowledge base update
- `POST /admin/backup` - Create system backup
- `POST /admin/restore` - Restore from backup version
- `POST /admin/cleanup` - Clean up old data and logs
- `GET /admin/actions` - List administrative actions
- `GET /admin/actions/{id}` - Get specific action details

## Usage

### Basic Setup

```python
from lightrag_integration.maintenance.integration import initialize_maintenance_system_async
from lightrag_integration.component import LightRAGComponent

# Initialize LightRAG component
lightrag_component = LightRAGComponent(config)

# Initialize maintenance system
maintenance_system = await initialize_maintenance_system_async(
    lightrag_component=lightrag_component,
    knowledge_base_path="data/lightrag_kg",
    versions_path="data/versions",
    watch_directories=["papers/", "custom_papers/"]
)

# Get admin interface
admin_interface = maintenance_system.get_admin_interface()

# Get API router for FastAPI integration
api_router = maintenance_system.get_api_router()
```

### FastAPI Integration

```python
from fastapi import FastAPI
from lightrag_integration.maintenance.integration import configure_api_router

app = FastAPI()

# Configure and include the admin router
admin_router = configure_api_router()
app.include_router(admin_router)
```

### Manual Operations

```python
# Trigger manual update
action_id = await admin_interface.trigger_manual_update(
    admin_user="admin",
    description="Monthly knowledge base update"
)

# Create system backup
backup_action_id = await admin_interface.create_system_backup(
    admin_user="admin",
    description="Pre-maintenance backup"
)

# Check system status
status = await admin_interface.get_system_status()
print(f"System health: {status.health_status}")
print(f"Total documents: {status.total_documents}")
```

## Configuration

### Environment Variables
- `LIGHTRAG_KB_PATH` - Knowledge base storage path
- `LIGHTRAG_VERSIONS_PATH` - Version storage path
- `LIGHTRAG_MAX_VERSIONS` - Maximum versions to keep
- `LIGHTRAG_COMPRESS_BACKUPS` - Enable backup compression

### Watch Directories
Configure directories to monitor for new documents:
```python
watch_directories = [
    "papers/",
    "custom_papers/",
    "research_docs/"
]
```

## Testing

The maintenance system includes comprehensive tests:

```bash
# Run all maintenance tests
python -m pytest src/lightrag_integration/maintenance/test_*.py -v

# Run specific component tests
python -m pytest src/lightrag_integration/maintenance/test_update_system.py -v
python -m pytest src/lightrag_integration/maintenance/test_version_control.py -v
python -m pytest src/lightrag_integration/maintenance/test_admin_interface.py -v
python -m pytest src/lightrag_integration/maintenance/test_api_endpoints.py -v
python -m pytest src/lightrag_integration/maintenance/test_integration.py -v
```

## Monitoring and Alerts

### Health Checks
The system provides multiple levels of health monitoring:
- Component-level health checks
- System-wide health aggregation
- Performance metrics collection
- Error rate monitoring

### Alerts
- High error rates
- Performance degradation
- Storage space issues
- Failed updates or backups

## Security Considerations

### Authentication
- Admin operations require user identification
- Destructive operations require confirmation tokens
- API endpoints should be protected with authentication middleware

### Data Protection
- Backups are stored securely
- Sensitive operations are logged
- Access controls for administrative functions

## Maintenance Procedures

### Regular Maintenance
1. **Daily**: Monitor system health and error rates
2. **Weekly**: Review and clean up old versions
3. **Monthly**: Perform comprehensive system backup
4. **Quarterly**: Review and update watch directories

### Troubleshooting
1. Check system status endpoint for health information
2. Review administrative action logs for failed operations
3. Use version control to rollback problematic updates
4. Monitor performance metrics for degradation

## Requirements Compliance

This maintenance system fulfills the following requirements:

### Requirement 6.1: New Research Papers
- ✅ Automatic monitoring of watch directories
- ✅ Incremental processing of new documents
- ✅ Batch processing capabilities

### Requirement 6.2: Incremental Updates
- ✅ Efficient delta processing
- ✅ Version control integration
- ✅ Rollback capabilities

### Requirement 6.3: Content Management
- ✅ Document removal and curation interfaces
- ✅ Metadata management
- ✅ Administrative oversight

### Requirement 6.4: Version Control
- ✅ Automatic version creation
- ✅ Backup and restore functionality
- ✅ Version history management

### Requirement 6.5: Administrative Interfaces
- ✅ REST API endpoints
- ✅ System management tools
- ✅ Performance monitoring

### Requirement 6.6: System Status
- ✅ Health monitoring
- ✅ Metrics collection
- ✅ Status dashboards

### Requirement 6.7: Update Procedures
- ✅ Non-disruptive updates
- ✅ Progress tracking
- ✅ Error handling and recovery