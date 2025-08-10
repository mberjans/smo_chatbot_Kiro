"""
Error Handling and Recovery Mechanisms

This module provides comprehensive error handling, retry logic, and graceful
degradation capabilities for the LightRAG integration system.

Implements requirements 7.1, 7.2, and 7.3 for robust error recovery.
"""

import asyncio
import logging
import traceback
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import inspect

from .utils.logging import setup_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for different types of failures."""
    PDF_PROCESSING = "pdf_processing"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    QUERY_PROCESSING = "query_processing"
    STORAGE = "storage"
    NETWORK = "network"
    CONFIGURATION = "configuration"
    RESOURCE = "resource"
    VALIDATION = "validation"
    EXTERNAL_API = "external_api"
    UNKNOWN = "unknown"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    component: str
    input_data: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    attempt_number: int = 1
    max_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: Tuple[Type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
        OSError,
        asyncio.TimeoutError,
    )


@dataclass
class ErrorRecord:
    """Record of an error occurrence."""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    exception: Exception
    context: ErrorContext
    traceback_str: str
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution_method: Optional[str] = None


@dataclass
class FallbackResult:
    """Result from a fallback operation."""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    fallback_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorHandler:
    """
    Comprehensive error handler with retry logic and graceful degradation.
    
    This class provides centralized error handling capabilities including:
    - Retry mechanisms with exponential backoff
    - Graceful degradation strategies
    - Error categorization and severity assessment
    - Fallback operations
    - Error tracking and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the error handler."""
        self.config = config or {}
        self.logger = setup_logger("error_handler")
        
        # Error tracking
        self.error_history: List[ErrorRecord] = []
        self.error_counts: Dict[str, int] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Default retry configuration
        self.default_retry_config = RetryConfig(
            max_attempts=self.config.get('max_retry_attempts', 3),
            base_delay=self.config.get('base_retry_delay', 1.0),
            max_delay=self.config.get('max_retry_delay', 60.0),
            exponential_base=self.config.get('exponential_base', 2.0),
            jitter=self.config.get('retry_jitter', True)
        )
        
        # Circuit breaker configuration
        self.circuit_breaker_threshold = self.config.get('circuit_breaker_threshold', 5)
        self.circuit_breaker_timeout = self.config.get('circuit_breaker_timeout', 300)  # 5 minutes
        
        self.logger.info("Error handler initialized with config: %s", self.config)
    
    def with_retry(
        self,
        retry_config: Optional[RetryConfig] = None,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ):
        """
        Decorator for adding retry logic to functions.
        
        Args:
            retry_config: Custom retry configuration
            category: Error category for tracking
            severity: Error severity level
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute_with_retry(
                    func, args, kwargs, retry_config, category, severity
                )
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(self.execute_with_retry(
                    func, args, kwargs, retry_config, category, severity
                ))
            
            # Return appropriate wrapper based on function type
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    async def execute_with_retry(
        self,
        func: Callable,
        args: tuple,
        kwargs: dict,
        retry_config: Optional[RetryConfig] = None,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            retry_config: Retry configuration
            category: Error category
            severity: Error severity
        
        Returns:
            Function result
        
        Raises:
            Exception: If all retry attempts fail
        """
        config = retry_config or self.default_retry_config
        operation_name = f"{func.__module__}.{func.__name__}"
        
        # Check circuit breaker
        if self._is_circuit_breaker_open(operation_name):
            raise RuntimeError(f"Circuit breaker open for operation: {operation_name}")
        
        context = ErrorContext(
            operation=operation_name,
            component=func.__module__.split('.')[-1],
            input_data={"args_count": len(args), "kwargs_keys": list(kwargs.keys())},
            max_attempts=config.max_attempts
        )
        
        last_exception = None
        
        for attempt in range(1, config.max_attempts + 1):
            context.attempt_number = attempt
            
            try:
                self.logger.debug(f"Executing {operation_name}, attempt {attempt}/{config.max_attempts}")
                
                # Execute function
                if inspect.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Reset circuit breaker on success
                self._reset_circuit_breaker(operation_name)
                
                if attempt > 1:
                    self.logger.info(f"Operation {operation_name} succeeded on attempt {attempt}")
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Record error
                error_record = self._create_error_record(e, category, severity, context)
                self._track_error(error_record)
                
                # Check if exception is retryable
                if not self._is_retryable_exception(e, config):
                    self.logger.error(f"Non-retryable exception in {operation_name}: {str(e)}")
                    self._update_circuit_breaker(operation_name)
                    raise
                
                # Check if we should continue retrying
                if attempt >= config.max_attempts:
                    self.logger.error(f"All retry attempts failed for {operation_name}")
                    self._update_circuit_breaker(operation_name)
                    break
                
                # Calculate delay for next attempt
                delay = self._calculate_retry_delay(attempt, config)
                
                self.logger.warning(
                    f"Attempt {attempt} failed for {operation_name}: {str(e)}. "
                    f"Retrying in {delay:.2f}s..."
                )
                
                await asyncio.sleep(delay)
        
        # All attempts failed
        raise last_exception
    
    async def execute_with_fallback(
        self,
        primary_func: Callable,
        fallback_func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> FallbackResult:
        """
        Execute a function with fallback on failure.
        
        Args:
            primary_func: Primary function to execute
            fallback_func: Fallback function if primary fails
            args: Function arguments
            kwargs: Function keyword arguments
            category: Error category
            severity: Error severity
        
        Returns:
            FallbackResult with execution outcome
        """
        kwargs = kwargs or {}
        
        try:
            # Try primary function first
            if inspect.iscoroutinefunction(primary_func):
                result = await primary_func(*args, **kwargs)
            else:
                result = primary_func(*args, **kwargs)
            
            return FallbackResult(
                success=True,
                data=result,
                fallback_method=None
            )
            
        except Exception as e:
            # Record primary function failure
            context = ErrorContext(
                operation=f"{primary_func.__module__}.{primary_func.__name__}",
                component=primary_func.__module__.split('.')[-1],
                input_data={"args_count": len(args), "kwargs_keys": list(kwargs.keys())}
            )
            
            error_record = self._create_error_record(e, category, severity, context)
            self._track_error(error_record)
            
            self.logger.warning(f"Primary function failed: {str(e)}. Trying fallback...")
            
            try:
                # Try fallback function
                if inspect.iscoroutinefunction(fallback_func):
                    fallback_result = await fallback_func(*args, **kwargs)
                else:
                    fallback_result = fallback_func(*args, **kwargs)
                
                self.logger.info("Fallback function succeeded")
                
                return FallbackResult(
                    success=True,
                    data=fallback_result,
                    fallback_method=f"{fallback_func.__module__}.{fallback_func.__name__}",
                    metadata={"primary_error": str(e)}
                )
                
            except Exception as fallback_error:
                # Both primary and fallback failed
                self.logger.error(f"Both primary and fallback functions failed: {str(fallback_error)}")
                
                return FallbackResult(
                    success=False,
                    error_message=f"Primary: {str(e)}, Fallback: {str(fallback_error)}",
                    metadata={"primary_error": str(e), "fallback_error": str(fallback_error)}
                )
    
    def handle_pdf_processing_error(self, error: Exception, pdf_path: str) -> Dict[str, Any]:
        """
        Handle PDF processing errors with specific recovery strategies.
        
        Args:
            error: The exception that occurred
            pdf_path: Path to the PDF file being processed
        
        Returns:
            Dictionary with error handling result
        """
        context = ErrorContext(
            operation="pdf_processing",
            component="pdf_extractor",
            input_data={"pdf_path": pdf_path}
        )
        
        error_record = self._create_error_record(
            error, ErrorCategory.PDF_PROCESSING, ErrorSeverity.MEDIUM, context
        )
        self._track_error(error_record)
        
        # Determine recovery strategy based on error type
        if isinstance(error, FileNotFoundError):
            return {
                "success": False,
                "error_type": "file_not_found",
                "recovery_action": "skip_file",
                "message": f"PDF file not found: {pdf_path}",
                "should_retry": False
            }
        
        elif isinstance(error, PermissionError):
            return {
                "success": False,
                "error_type": "permission_denied",
                "recovery_action": "skip_file",
                "message": f"Permission denied accessing PDF: {pdf_path}",
                "should_retry": False
            }
        
        elif "corrupted" in str(error).lower() or "invalid" in str(error).lower():
            return {
                "success": False,
                "error_type": "corrupted_file",
                "recovery_action": "skip_file",
                "message": f"Corrupted or invalid PDF: {pdf_path}",
                "should_retry": False
            }
        
        elif isinstance(error, MemoryError):
            return {
                "success": False,
                "error_type": "memory_error",
                "recovery_action": "retry_with_smaller_batch",
                "message": f"Memory error processing PDF: {pdf_path}",
                "should_retry": True,
                "retry_delay": 30
            }
        
        else:
            # Generic error - allow retry
            return {
                "success": False,
                "error_type": "generic_error",
                "recovery_action": "retry",
                "message": f"Error processing PDF {pdf_path}: {str(error)}",
                "should_retry": True,
                "retry_delay": 5
            }
    
    def handle_knowledge_graph_error(self, error: Exception, operation: str) -> Dict[str, Any]:
        """
        Handle knowledge graph construction errors.
        
        Args:
            error: The exception that occurred
            operation: The KG operation being performed
        
        Returns:
            Dictionary with error handling result
        """
        context = ErrorContext(
            operation=f"knowledge_graph_{operation}",
            component="knowledge_graph",
            input_data={"operation": operation}
        )
        
        error_record = self._create_error_record(
            error, ErrorCategory.KNOWLEDGE_GRAPH, ErrorSeverity.HIGH, context
        )
        self._track_error(error_record)
        
        # Determine recovery strategy
        if "storage" in str(error).lower() or "disk" in str(error).lower():
            return {
                "success": False,
                "error_type": "storage_error",
                "recovery_action": "retry_with_cleanup",
                "message": f"Storage error in KG operation {operation}: {str(error)}",
                "should_retry": True,
                "retry_delay": 10
            }
        
        elif "memory" in str(error).lower():
            return {
                "success": False,
                "error_type": "memory_error",
                "recovery_action": "process_in_smaller_chunks",
                "message": f"Memory error in KG operation {operation}: {str(error)}",
                "should_retry": True,
                "retry_delay": 30
            }
        
        elif "validation" in str(error).lower():
            return {
                "success": False,
                "error_type": "validation_error",
                "recovery_action": "skip_invalid_data",
                "message": f"Validation error in KG operation {operation}: {str(error)}",
                "should_retry": False
            }
        
        else:
            return {
                "success": False,
                "error_type": "generic_kg_error",
                "recovery_action": "retry",
                "message": f"Error in KG operation {operation}: {str(error)}",
                "should_retry": True,
                "retry_delay": 15
            }
    
    def handle_query_processing_error(self, error: Exception, query: str) -> Dict[str, Any]:
        """
        Handle query processing errors with fallback strategies.
        
        Args:
            error: The exception that occurred
            query: The query being processed
        
        Returns:
            Dictionary with error handling result
        """
        context = ErrorContext(
            operation="query_processing",
            component="query_engine",
            input_data={"query_length": len(query)}
        )
        
        error_record = self._create_error_record(
            error, ErrorCategory.QUERY_PROCESSING, ErrorSeverity.MEDIUM, context
        )
        self._track_error(error_record)
        
        # Determine recovery strategy
        if "timeout" in str(error).lower():
            return {
                "success": False,
                "error_type": "timeout_error",
                "recovery_action": "retry_with_simpler_query",
                "message": f"Query timeout: {str(error)}",
                "should_retry": True,
                "fallback_available": True
            }
        
        elif "memory" in str(error).lower():
            return {
                "success": False,
                "error_type": "memory_error",
                "recovery_action": "retry_with_limited_scope",
                "message": f"Memory error during query processing: {str(error)}",
                "should_retry": True,
                "fallback_available": True
            }
        
        elif "graph" in str(error).lower() and "not found" in str(error).lower():
            return {
                "success": False,
                "error_type": "no_graph_data",
                "recovery_action": "fallback_to_external_api",
                "message": "No knowledge graph data available",
                "should_retry": False,
                "fallback_available": True
            }
        
        else:
            return {
                "success": False,
                "error_type": "generic_query_error",
                "recovery_action": "fallback_to_external_api",
                "message": f"Query processing error: {str(error)}",
                "should_retry": True,
                "fallback_available": True
            }
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and trends."""
        now = datetime.now()
        last_hour = now - timedelta(hours=1)
        last_day = now - timedelta(days=1)
        
        # Count errors by category and time period
        recent_errors = [e for e in self.error_history if e.timestamp >= last_hour]
        daily_errors = [e for e in self.error_history if e.timestamp >= last_day]
        
        category_counts = {}
        severity_counts = {}
        
        for error in daily_errors:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "recent_errors_1h": len(recent_errors),
            "recent_errors_24h": len(daily_errors),
            "errors_by_category": category_counts,
            "errors_by_severity": severity_counts,
            "circuit_breakers_open": len([cb for cb in self.circuit_breakers.values() if cb.get("open", False)]),
            "most_common_errors": self._get_most_common_errors(5)
        }
    
    def reset_error_history(self, older_than_hours: int = 24) -> int:
        """
        Reset error history older than specified hours.
        
        Args:
            older_than_hours: Remove errors older than this many hours
        
        Returns:
            Number of errors removed
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        initial_count = len(self.error_history)
        
        self.error_history = [e for e in self.error_history if e.timestamp >= cutoff_time]
        
        removed_count = initial_count - len(self.error_history)
        self.logger.info(f"Removed {removed_count} error records older than {older_than_hours} hours")
        
        return removed_count
    
    def _create_error_record(
        self,
        exception: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: ErrorContext
    ) -> ErrorRecord:
        """Create an error record from an exception."""
        error_id = f"{category.value}_{int(time.time())}_{context.attempt_number}"
        
        return ErrorRecord(
            error_id=error_id,
            category=category,
            severity=severity,
            exception=exception,
            context=context,
            traceback_str=traceback.format_exc()
        )
    
    def _track_error(self, error_record: ErrorRecord) -> None:
        """Track an error occurrence."""
        self.error_history.append(error_record)
        
        # Update error counts
        error_key = f"{error_record.category.value}_{type(error_record.exception).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log error based on severity
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(
                f"Critical error in {error_record.context.operation}: {str(error_record.exception)}"
            )
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(
                f"High severity error in {error_record.context.operation}: {str(error_record.exception)}"
            )
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(
                f"Medium severity error in {error_record.context.operation}: {str(error_record.exception)}"
            )
        else:
            self.logger.info(
                f"Low severity error in {error_record.context.operation}: {str(error_record.exception)}"
            )
    
    def _is_retryable_exception(self, exception: Exception, config: RetryConfig) -> bool:
        """Check if an exception is retryable."""
        return isinstance(exception, config.retryable_exceptions)
    
    def _calculate_retry_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay before next retry attempt."""
        # Exponential backoff
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        
        # Cap at max delay
        delay = min(delay, config.max_delay)
        
        # Add jitter if enabled
        if config.jitter:
            import random
            jitter = random.uniform(0.1, 0.3) * delay
            delay += jitter
        
        return delay
    
    def _is_circuit_breaker_open(self, operation: str) -> bool:
        """Check if circuit breaker is open for an operation."""
        if operation not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[operation]
        
        if not breaker.get("open", False):
            return False
        
        # Check if timeout has passed
        if datetime.now() >= breaker.get("reset_time", datetime.now()):
            self._reset_circuit_breaker(operation)
            return False
        
        return True
    
    def _update_circuit_breaker(self, operation: str) -> None:
        """Update circuit breaker state after a failure."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = {"failure_count": 0, "open": False}
        
        breaker = self.circuit_breakers[operation]
        breaker["failure_count"] = breaker.get("failure_count", 0) + 1
        
        if breaker["failure_count"] >= self.circuit_breaker_threshold:
            breaker["open"] = True
            breaker["reset_time"] = datetime.now() + timedelta(seconds=self.circuit_breaker_timeout)
            self.logger.warning(f"Circuit breaker opened for operation: {operation}")
    
    def _reset_circuit_breaker(self, operation: str) -> None:
        """Reset circuit breaker after successful operation."""
        if operation in self.circuit_breakers:
            self.circuit_breakers[operation] = {"failure_count": 0, "open": False}
            self.logger.info(f"Circuit breaker reset for operation: {operation}")
    
    def _get_most_common_errors(self, limit: int) -> List[Dict[str, Any]]:
        """Get most common error types."""
        error_type_counts = {}
        
        for error in self.error_history:
            error_type = f"{error.category.value}_{type(error.exception).__name__}"
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        # Sort by count and return top N
        sorted_errors = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {"error_type": error_type, "count": count}
            for error_type, count in sorted_errors[:limit]
        ]


# Convenience functions for common error handling patterns

async def safe_execute(
    func: Callable,
    *args,
    error_handler: Optional[ErrorHandler] = None,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    **kwargs
) -> Tuple[bool, Any, Optional[str]]:
    """
    Safely execute a function with error handling.
    
    Returns:
        Tuple of (success, result, error_message)
    """
    handler = error_handler or ErrorHandler()
    
    try:
        if inspect.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        return True, result, None
        
    except Exception as e:
        context = ErrorContext(
            operation=f"{func.__module__}.{func.__name__}",
            component=func.__module__.split('.')[-1]
        )
        
        error_record = handler._create_error_record(e, category, severity, context)
        handler._track_error(error_record)
        
        return False, None, str(e)


def create_fallback_response(
    error_message: str,
    operation: str,
    fallback_data: Any = None
) -> Dict[str, Any]:
    """Create a standardized fallback response."""
    return {
        "success": False,
        "error": True,
        "error_message": error_message,
        "operation": operation,
        "fallback_data": fallback_data,
        "timestamp": datetime.now().isoformat(),
        "recovery_suggestions": [
            "Check system logs for detailed error information",
            "Verify input data and try again",
            "Contact system administrator if problem persists"
        ]
    }