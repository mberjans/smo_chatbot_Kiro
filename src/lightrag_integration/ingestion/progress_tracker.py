"""
Progress Tracking and Status Reporting

This module provides progress tracking and status reporting capabilities
for document ingestion and processing operations.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from ..utils.logging import setup_logger


class OperationStatus(Enum):
    """Status of an operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OperationType(Enum):
    """Type of operation being tracked."""
    FILE_PROCESSING = "file_processing"
    BATCH_PROCESSING = "batch_processing"
    DIRECTORY_SCAN = "directory_scan"
    KNOWLEDGE_GRAPH_BUILD = "knowledge_graph_build"
    ENTITY_EXTRACTION = "entity_extraction"


@dataclass
class OperationProgress:
    """Progress information for an operation."""
    operation_id: str
    operation_type: OperationType
    status: OperationStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    progress_percent: float = 0.0
    current_step: str = ""
    total_items: int = 0
    completed_items: int = 0
    failed_items: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Get operation duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return datetime.now() - self.start_time
    
    @property
    def is_active(self) -> bool:
        """Check if operation is currently active."""
        return self.status in [OperationStatus.PENDING, OperationStatus.RUNNING]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.completed_items + self.failed_items == 0:
            return 0.0
        return (self.completed_items / (self.completed_items + self.failed_items)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'operation_id': self.operation_id,
            'operation_type': self.operation_type.value,
            'status': self.status.value,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration.total_seconds() if self.duration else None,
            'progress_percent': self.progress_percent,
            'current_step': self.current_step,
            'total_items': self.total_items,
            'completed_items': self.completed_items,
            'failed_items': self.failed_items,
            'success_rate': self.success_rate,
            'error_message': self.error_message,
            'metadata': self.metadata,
            'is_active': self.is_active
        }


@dataclass
class StatusReport:
    """Comprehensive status report."""
    timestamp: datetime
    active_operations: List[OperationProgress]
    completed_operations: List[OperationProgress]
    failed_operations: List[OperationProgress]
    system_stats: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_operations(self) -> int:
        """Total number of operations."""
        return len(self.active_operations) + len(self.completed_operations) + len(self.failed_operations)
    
    @property
    def overall_success_rate(self) -> float:
        """Overall success rate across all operations."""
        total_completed = len(self.completed_operations)
        total_failed = len(self.failed_operations)
        
        if total_completed + total_failed == 0:
            return 0.0
        
        return (total_completed / (total_completed + total_failed)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_operations': self.total_operations,
            'active_operations_count': len(self.active_operations),
            'completed_operations_count': len(self.completed_operations),
            'failed_operations_count': len(self.failed_operations),
            'overall_success_rate': self.overall_success_rate,
            'active_operations': [op.to_dict() for op in self.active_operations],
            'completed_operations': [op.to_dict() for op in self.completed_operations],
            'failed_operations': [op.to_dict() for op in self.failed_operations],
            'system_stats': self.system_stats
        }


class ProgressTracker:
    """
    Tracks progress of various operations and provides status reporting.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the progress tracker.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.logger = setup_logger("progress_tracker")
        
        # Configuration
        self.max_history = self.config.get('max_history', 1000)
        self.cleanup_interval = self.config.get('cleanup_interval', 3600)  # seconds
        self.persist_to_file = self.config.get('persist_to_file', False)
        self.status_file = self.config.get('status_file', 'progress_status.json')
        
        # Operation tracking
        self._operations: Dict[str, OperationProgress] = {}
        self._operation_history: List[OperationProgress] = []
        self._callbacks: Dict[str, List[Callable]] = {}
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._stop_cleanup = asyncio.Event()
        
        self.logger.info("Progress tracker initialized")
    
    async def start(self) -> None:
        """Start the progress tracker."""
        self.logger.info("Starting progress tracker...")
        
        # Load persisted status if enabled
        if self.persist_to_file:
            await self._load_status()
        
        # Start cleanup task
        self._stop_cleanup.clear()
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Progress tracker started")
    
    async def stop(self) -> None:
        """Stop the progress tracker."""
        self.logger.info("Stopping progress tracker...")
        
        # Stop cleanup task
        self._stop_cleanup.set()
        if self._cleanup_task and not self._cleanup_task.done():
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=5.0)
            except asyncio.TimeoutError:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass
        
        # Persist status if enabled
        if self.persist_to_file:
            await self._save_status()
        
        self.logger.info("Progress tracker stopped")
    
    def start_operation(self, 
                       operation_id: str,
                       operation_type: OperationType,
                       total_items: int = 0,
                       metadata: Optional[Dict[str, Any]] = None) -> OperationProgress:
        """
        Start tracking a new operation.
        
        Args:
            operation_id: Unique identifier for the operation
            operation_type: Type of operation
            total_items: Total number of items to process
            metadata: Optional metadata dictionary
        
        Returns:
            OperationProgress instance
        """
        if operation_id in self._operations:
            self.logger.warning(f"Operation {operation_id} already exists, updating...")
        
        operation = OperationProgress(
            operation_id=operation_id,
            operation_type=operation_type,
            status=OperationStatus.RUNNING,
            start_time=datetime.now(),
            total_items=total_items,
            metadata=metadata or {}
        )
        
        self._operations[operation_id] = operation
        
        self.logger.info(f"Started tracking operation: {operation_id} ({operation_type.value})")
        
        # Notify callbacks
        asyncio.create_task(self._notify_callbacks('operation_started', operation))
        
        return operation
    
    def update_progress(self,
                       operation_id: str,
                       progress_percent: Optional[float] = None,
                       current_step: Optional[str] = None,
                       completed_items: Optional[int] = None,
                       failed_items: Optional[int] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Optional[OperationProgress]:
        """
        Update progress for an operation.
        
        Args:
            operation_id: Operation identifier
            progress_percent: Progress percentage (0-100)
            current_step: Current step description
            completed_items: Number of completed items
            failed_items: Number of failed items
            metadata: Additional metadata to merge
        
        Returns:
            Updated OperationProgress or None if operation not found
        """
        if operation_id not in self._operations:
            self.logger.warning(f"Operation {operation_id} not found for progress update")
            return None
        
        operation = self._operations[operation_id]
        
        # Update fields if provided
        if progress_percent is not None:
            operation.progress_percent = max(0, min(100, progress_percent))
        
        if current_step is not None:
            operation.current_step = current_step
        
        if completed_items is not None:
            operation.completed_items = completed_items
        
        if failed_items is not None:
            operation.failed_items = failed_items
        
        if metadata:
            operation.metadata.update(metadata)
        
        # Auto-calculate progress if total_items is set
        if operation.total_items > 0 and progress_percent is None:
            total_processed = operation.completed_items + operation.failed_items
            operation.progress_percent = (total_processed / operation.total_items) * 100
        
        self.logger.debug(f"Updated progress for {operation_id}: {operation.progress_percent:.1f}%")
        
        # Notify callbacks
        asyncio.create_task(self._notify_callbacks('progress_updated', operation))
        
        return operation
    
    def complete_operation(self,
                          operation_id: str,
                          success: bool = True,
                          error_message: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[OperationProgress]:
        """
        Mark an operation as completed.
        
        Args:
            operation_id: Operation identifier
            success: Whether the operation was successful
            error_message: Error message if operation failed
            metadata: Additional metadata to merge
        
        Returns:
            Completed OperationProgress or None if operation not found
        """
        if operation_id not in self._operations:
            self.logger.warning(f"Operation {operation_id} not found for completion")
            return None
        
        operation = self._operations[operation_id]
        operation.end_time = datetime.now()
        operation.status = OperationStatus.COMPLETED if success else OperationStatus.FAILED
        operation.progress_percent = 100.0
        
        if error_message:
            operation.error_message = error_message
        
        if metadata:
            operation.metadata.update(metadata)
        
        # Move to history
        self._operation_history.append(operation)
        del self._operations[operation_id]
        
        # Limit history size
        if len(self._operation_history) > self.max_history:
            self._operation_history = self._operation_history[-self.max_history:]
        
        status_text = "completed" if success else "failed"
        self.logger.info(f"Operation {operation_id} {status_text} in {operation.duration}")
        
        # Notify callbacks
        event_type = 'operation_completed' if success else 'operation_failed'
        asyncio.create_task(self._notify_callbacks(event_type, operation))
        
        return operation
    
    def cancel_operation(self, operation_id: str) -> Optional[OperationProgress]:
        """
        Cancel an active operation.
        
        Args:
            operation_id: Operation identifier
        
        Returns:
            Cancelled OperationProgress or None if operation not found
        """
        if operation_id not in self._operations:
            self.logger.warning(f"Operation {operation_id} not found for cancellation")
            return None
        
        operation = self._operations[operation_id]
        operation.end_time = datetime.now()
        operation.status = OperationStatus.CANCELLED
        
        # Move to history
        self._operation_history.append(operation)
        del self._operations[operation_id]
        
        self.logger.info(f"Operation {operation_id} cancelled")
        
        # Notify callbacks
        asyncio.create_task(self._notify_callbacks('operation_cancelled', operation))
        
        return operation
    
    def get_operation(self, operation_id: str) -> Optional[OperationProgress]:
        """
        Get current operation by ID.
        
        Args:
            operation_id: Operation identifier
        
        Returns:
            OperationProgress or None if not found
        """
        return self._operations.get(operation_id)
    
    def get_active_operations(self) -> List[OperationProgress]:
        """Get all active operations."""
        return list(self._operations.values())
    
    def get_operation_history(self, 
                             operation_type: Optional[OperationType] = None,
                             limit: Optional[int] = None) -> List[OperationProgress]:
        """
        Get operation history.
        
        Args:
            operation_type: Filter by operation type
            limit: Maximum number of operations to return
        
        Returns:
            List of historical operations
        """
        history = self._operation_history
        
        if operation_type:
            history = [op for op in history if op.operation_type == operation_type]
        
        # Sort by start time (most recent first)
        history = sorted(history, key=lambda x: x.start_time, reverse=True)
        
        if limit:
            history = history[:limit]
        
        return history
    
    def get_status_report(self) -> StatusReport:
        """
        Generate a comprehensive status report.
        
        Returns:
            StatusReport instance
        """
        # Categorize historical operations
        completed_ops = [op for op in self._operation_history if op.status == OperationStatus.COMPLETED]
        failed_ops = [op for op in self._operation_history if op.status in [OperationStatus.FAILED, OperationStatus.CANCELLED]]
        
        # Calculate system stats
        system_stats = {
            'uptime_seconds': (datetime.now() - datetime.now()).total_seconds(),  # Will be updated by caller
            'total_operations_tracked': len(self._operations) + len(self._operation_history),
            'operations_per_hour': self._calculate_operations_per_hour(),
            'average_operation_duration': self._calculate_average_duration(),
            'memory_usage_operations': len(self._operations),
            'memory_usage_history': len(self._operation_history)
        }
        
        return StatusReport(
            timestamp=datetime.now(),
            active_operations=list(self._operations.values()),
            completed_operations=completed_ops,
            failed_operations=failed_ops,
            system_stats=system_stats
        )
    
    def register_callback(self, event_type: str, callback: Callable) -> None:
        """
        Register a callback for progress events.
        
        Args:
            event_type: Type of event ('operation_started', 'progress_updated', 'operation_completed', etc.)
            callback: Async callback function
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        
        self._callbacks[event_type].append(callback)
        self.logger.debug(f"Registered callback for event type: {event_type}")
    
    def unregister_callback(self, event_type: str, callback: Callable) -> None:
        """
        Unregister a callback.
        
        Args:
            event_type: Type of event
            callback: Callback function to remove
        """
        if event_type in self._callbacks and callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)
            self.logger.debug(f"Unregistered callback for event type: {event_type}")
    
    async def _notify_callbacks(self, event_type: str, operation: OperationProgress) -> None:
        """Notify registered callbacks about an event."""
        if event_type not in self._callbacks:
            return
        
        for callback in self._callbacks[event_type]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(operation)
                else:
                    callback(operation)
            except Exception as e:
                self.logger.error(f"Error in callback for {event_type}: {str(e)}")
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of old operations."""
        while not self._stop_cleanup.is_set():
            try:
                await self._cleanup_old_operations()
                
                # Wait for next cleanup or stop signal
                try:
                    await asyncio.wait_for(self._stop_cleanup.wait(), timeout=self.cleanup_interval)
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    continue  # Timeout is expected
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _cleanup_old_operations(self) -> None:
        """Clean up old operations from history."""
        if len(self._operation_history) <= self.max_history:
            return
        
        # Keep only the most recent operations
        old_count = len(self._operation_history)
        self._operation_history = sorted(
            self._operation_history, 
            key=lambda x: x.start_time, 
            reverse=True
        )[:self.max_history]
        
        cleaned_count = old_count - len(self._operation_history)
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old operations from history")
    
    async def _save_status(self) -> None:
        """Save current status to file."""
        if not self.persist_to_file:
            return
        
        try:
            status_report = self.get_status_report()
            status_data = status_report.to_dict()
            
            status_path = Path(self.status_file)
            status_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(status_path, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, indent=2, default=str)
            
            self.logger.debug(f"Status saved to {status_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving status: {str(e)}")
    
    async def _load_status(self) -> None:
        """Load status from file."""
        if not self.persist_to_file:
            return
        
        try:
            status_path = Path(self.status_file)
            if not status_path.exists():
                return
            
            with open(status_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
            
            # Reconstruct operation history (active operations are not restored)
            if 'completed_operations' in status_data:
                for op_data in status_data['completed_operations']:
                    operation = self._operation_from_dict(op_data)
                    if operation:
                        self._operation_history.append(operation)
            
            if 'failed_operations' in status_data:
                for op_data in status_data['failed_operations']:
                    operation = self._operation_from_dict(op_data)
                    if operation:
                        self._operation_history.append(operation)
            
            self.logger.info(f"Loaded {len(self._operation_history)} operations from status file")
            
        except Exception as e:
            self.logger.error(f"Error loading status: {str(e)}")
    
    def _operation_from_dict(self, op_data: Dict[str, Any]) -> Optional[OperationProgress]:
        """Reconstruct OperationProgress from dictionary."""
        try:
            return OperationProgress(
                operation_id=op_data['operation_id'],
                operation_type=OperationType(op_data['operation_type']),
                status=OperationStatus(op_data['status']),
                start_time=datetime.fromisoformat(op_data['start_time']),
                end_time=datetime.fromisoformat(op_data['end_time']) if op_data.get('end_time') else None,
                progress_percent=op_data.get('progress_percent', 0.0),
                current_step=op_data.get('current_step', ''),
                total_items=op_data.get('total_items', 0),
                completed_items=op_data.get('completed_items', 0),
                failed_items=op_data.get('failed_items', 0),
                error_message=op_data.get('error_message'),
                metadata=op_data.get('metadata', {})
            )
        except Exception as e:
            self.logger.error(f"Error reconstructing operation from dict: {str(e)}")
            return None
    
    def _calculate_operations_per_hour(self) -> float:
        """Calculate operations per hour based on recent history."""
        if not self._operation_history:
            return 0.0
        
        # Look at operations from the last 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_ops = [op for op in self._operation_history if op.start_time >= cutoff_time]
        
        if not recent_ops:
            return 0.0
        
        # Calculate rate based on actual time span
        earliest_time = min(op.start_time for op in recent_ops)
        time_span_hours = (datetime.now() - earliest_time).total_seconds() / 3600
        
        if time_span_hours == 0:
            return 0.0
        
        return len(recent_ops) / time_span_hours
    
    def _calculate_average_duration(self) -> float:
        """Calculate average operation duration in seconds."""
        completed_ops = [op for op in self._operation_history if op.end_time]
        
        if not completed_ops:
            return 0.0
        
        total_duration = sum(op.duration.total_seconds() for op in completed_ops)
        return total_duration / len(completed_ops)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        await self.stop()
        self.logger.info("Progress tracker cleanup completed")