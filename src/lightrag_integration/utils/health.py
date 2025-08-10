"""
Health monitoring utilities for LightRAG integration.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a component."""
    status: HealthStatus
    message: str
    last_check: datetime
    metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check.isoformat(),
            "metrics": self.metrics
        }


@dataclass
class SystemHealth:
    """Overall system health information."""
    overall_status: HealthStatus
    components: Dict[str, ComponentHealth]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "overall_status": self.overall_status.value,
            "components": {
                name: component.to_dict() 
                for name, component in self.components.items()
            },
            "timestamp": self.timestamp.isoformat()
        }