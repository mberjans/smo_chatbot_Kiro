"""
Tests for Error Handling and Recovery Mechanisms

This module tests the comprehensive error handling, retry logic, and graceful
degradation capabilities implemented in error_handling.py.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import Any, Dict

from .error_handling import (
    ErrorHandler,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    RetryConfig,
    ErrorRecord,
    FallbackResult,
    safe_execute,
    create_fallback_response
)


class TestErrorHandler:
    """Test cases for the ErrorHandler class."""
    
    @pytest.fixture
    def error_handler(self):
        """Create an ErrorHandler instance for testing."""
        config = {
            'max_retry_attempts': 3,
            'base_retry_delay': 0.1,  # Short delay for testing
            'max_retry_delay': 1.0,
            'circuit_breaker_threshold': 3,
            'circuit_breaker_timeout': 5
        }
        return ErrorHandler(config)
    
    @pytest.fixture
    def sample_error_context(self):
        """Create a sample ErrorContext for testing."""
        return ErrorContext(
            operation="test_operation",
            component="test_component",
            input_data={"test": "data"},
            max_attempts=3
        )
    
    def test_error_handler_initialization(self, error_handler):
        """Test ErrorHandler initialization."""
        assert error_handler.config is not None
        assert error_handler.logger is not None
        assert error_handler.error_history == []
        assert error_handler.error_counts == {}
        assert error_handler.circuit_breakers == {}
    
    def test_create_error_record(self, error_handler, sample_error_context):
        """Test error record creation."""
        exception = ValueError("Test error")
        
        error_record = error_handler._create_error_record(
            exception, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, sample_error_context
        )
        
        assert error_record.category == ErrorCategory.VALIDATION
        assert error_record.severity == ErrorSeverity.MEDIUM
        assert error_record.exception == exception
        assert error_record.context == sample_error_context
        assert error_record.traceback_str is not None
        assert not error_record.resolved
    
    def test_track_error(self, error_handler, sample_error_context):
        """Test error tracking functionality."""
        exception = ValueError("Test error")
        
        error_record = error_handler._create_error_record(
            exception, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, sample_error_context
        )
        
        initial_count = len(error_handler.error_history)
        error_handler._track_error(error_record)
        
        assert len(error_handler.error_history) == initial_count + 1
        assert error_handler.error_history[-1] == error_record
        
        # Check error counts
        error_key = f"{ErrorCategory.VALIDATION.value}_ValueError"
        assert error_handler.error_counts[error_key] == 1
    
    def test_is_retryable_exception(self, error_handler):
        """Test retryable exception detection."""
        config = RetryConfig(retryable_exceptions=(ValueError, ConnectionError))
        
        assert error_handler._is_retryable_exception(ValueError("test"), config)
        assert error_handler._is_retryable_exception(ConnectionError("test"), config)
        assert not error_handler._is_retryable_exception(TypeError("test"), config)
    
    def test_calculate_retry_delay(self, error_handler):
        """Test retry delay calculation."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, max_delay=10.0, jitter=False)
        
        # Test exponential backoff
        delay1 = error_handler._calculate_retry_delay(1, config)
        delay2 = error_handler._calculate_retry_delay(2, config)
        delay3 = error_handler._calculate_retry_delay(3, config)
        
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0
        
        # Test max delay cap
        delay_large = error_handler._calculate_retry_delay(10, config)
        assert delay_large == 10.0
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_success(self, error_handler):
        """Test successful execution with retry decorator."""
        call_count = 0
        
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"
        
        result = await error_handler.execute_with_retry(
            test_func, (), {}, None, ErrorCategory.UNKNOWN, ErrorSeverity.LOW
        )
        
        assert result == "success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_failure_then_success(self, error_handler):
        """Test retry logic with initial failure then success."""
        call_count = 0
        
        async def test_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = await error_handler.execute_with_retry(
            test_func, (), {}, None, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        )
        
        assert result == "success"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_all_attempts_fail(self, error_handler):
        """Test retry logic when all attempts fail."""
        call_count = 0
        
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent failure")
        
        with pytest.raises(ConnectionError):
            await error_handler.execute_with_retry(
                test_func, (), {}, None, ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
            )
        
        assert call_count == 3  # Default max attempts
    
    @pytest.mark.asyncio
    async def test_execute_with_retry_non_retryable_exception(self, error_handler):
        """Test that non-retryable exceptions are not retried."""
        call_count = 0
        
        async def test_func():
            nonlocal call_count
            call_count += 1
            raise TypeError("Non-retryable error")
        
        with pytest.raises(TypeError):
            await error_handler.execute_with_retry(
                test_func, (), {}, None, ErrorCategory.VALIDATION, ErrorSeverity.HIGH
            )
        
        assert call_count == 1  # Should not retry
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_primary_success(self, error_handler):
        """Test fallback execution when primary function succeeds."""
        async def primary_func():
            return "primary_result"
        
        async def fallback_func():
            return "fallback_result"
        
        result = await error_handler.execute_with_fallback(
            primary_func, fallback_func, (), {}, ErrorCategory.UNKNOWN, ErrorSeverity.LOW
        )
        
        assert result.success
        assert result.data == "primary_result"
        assert result.fallback_method is None
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_primary_fails_fallback_succeeds(self, error_handler):
        """Test fallback execution when primary fails but fallback succeeds."""
        async def primary_func():
            raise ValueError("Primary failure")
        
        async def fallback_func():
            return "fallback_result"
        
        result = await error_handler.execute_with_fallback(
            primary_func, fallback_func, (), {}, ErrorCategory.UNKNOWN, ErrorSeverity.LOW
        )
        
        assert result.success
        assert result.data == "fallback_result"
        assert result.fallback_method is not None
        assert "primary_error" in result.metadata
    
    @pytest.mark.asyncio
    async def test_execute_with_fallback_both_fail(self, error_handler):
        """Test fallback execution when both primary and fallback fail."""
        async def primary_func():
            raise ValueError("Primary failure")
        
        async def fallback_func():
            raise RuntimeError("Fallback failure")
        
        result = await error_handler.execute_with_fallback(
            primary_func, fallback_func, (), {}, ErrorCategory.UNKNOWN, ErrorSeverity.LOW
        )
        
        assert not result.success
        assert "Primary: Primary failure" in result.error_message
        assert "Fallback: Fallback failure" in result.error_message
        assert "primary_error" in result.metadata
        assert "fallback_error" in result.metadata
    
    def test_handle_pdf_processing_error_file_not_found(self, error_handler):
        """Test PDF processing error handling for file not found."""
        error = FileNotFoundError("File not found")
        pdf_path = "/path/to/missing.pdf"
        
        result = error_handler.handle_pdf_processing_error(error, pdf_path)
        
        assert not result["success"]
        assert result["error_type"] == "file_not_found"
        assert result["recovery_action"] == "skip_file"
        assert not result["should_retry"]
        assert pdf_path in result["message"]
    
    def test_handle_pdf_processing_error_corrupted_file(self, error_handler):
        """Test PDF processing error handling for corrupted file."""
        error = Exception("Corrupted PDF file")
        pdf_path = "/path/to/corrupted.pdf"
        
        result = error_handler.handle_pdf_processing_error(error, pdf_path)
        
        assert not result["success"]
        assert result["error_type"] == "corrupted_file"
        assert result["recovery_action"] == "skip_file"
        assert not result["should_retry"]
    
    def test_handle_pdf_processing_error_memory_error(self, error_handler):
        """Test PDF processing error handling for memory error."""
        error = MemoryError("Out of memory")
        pdf_path = "/path/to/large.pdf"
        
        result = error_handler.handle_pdf_processing_error(error, pdf_path)
        
        assert not result["success"]
        assert result["error_type"] == "memory_error"
        assert result["recovery_action"] == "retry_with_smaller_batch"
        assert result["should_retry"]
        assert "retry_delay" in result
    
    def test_handle_knowledge_graph_error_storage(self, error_handler):
        """Test knowledge graph error handling for storage issues."""
        error = Exception("Storage disk full")
        operation = "construct_graph"
        
        result = error_handler.handle_knowledge_graph_error(error, operation)
        
        assert not result["success"]
        assert result["error_type"] == "storage_error"
        assert result["recovery_action"] == "retry_with_cleanup"
        assert result["should_retry"]
    
    def test_handle_knowledge_graph_error_validation(self, error_handler):
        """Test knowledge graph error handling for validation errors."""
        error = Exception("Validation failed for entity")
        operation = "add_entity"
        
        result = error_handler.handle_knowledge_graph_error(error, operation)
        
        assert not result["success"]
        assert result["error_type"] == "validation_error"
        assert result["recovery_action"] == "skip_invalid_data"
        assert not result["should_retry"]
    
    def test_handle_query_processing_error_timeout(self, error_handler):
        """Test query processing error handling for timeout."""
        error = Exception("Query timeout exceeded")
        query = "What is clinical metabolomics?"
        
        result = error_handler.handle_query_processing_error(error, query)
        
        assert not result["success"]
        assert result["error_type"] == "timeout_error"
        assert result["recovery_action"] == "retry_with_simpler_query"
        assert result["should_retry"]
        assert result["fallback_available"]
    
    def test_handle_query_processing_error_no_graph_data(self, error_handler):
        """Test query processing error handling when no graph data available."""
        error = Exception("Knowledge graph not found")
        query = "What is clinical metabolomics?"
        
        result = error_handler.handle_query_processing_error(error, query)
        
        assert not result["success"]
        assert result["error_type"] == "no_graph_data"
        assert result["recovery_action"] == "fallback_to_external_api"
        assert not result["should_retry"]
        assert result["fallback_available"]
    
    def test_circuit_breaker_functionality(self, error_handler):
        """Test circuit breaker functionality."""
        operation = "test_operation"
        
        # Initially circuit breaker should be closed
        assert not error_handler._is_circuit_breaker_open(operation)
        
        # Trigger failures to open circuit breaker
        for _ in range(3):  # Threshold is 3
            error_handler._update_circuit_breaker(operation)
        
        # Circuit breaker should now be open
        assert error_handler._is_circuit_breaker_open(operation)
        
        # Reset circuit breaker
        error_handler._reset_circuit_breaker(operation)
        assert not error_handler._is_circuit_breaker_open(operation)
    
    def test_get_error_statistics(self, error_handler, sample_error_context):
        """Test error statistics generation."""
        # Add some test errors
        for i in range(5):
            exception = ValueError(f"Test error {i}")
            error_record = error_handler._create_error_record(
                exception, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, sample_error_context
            )
            error_handler._track_error(error_record)
        
        # Add different category error
        exception = ConnectionError("Network error")
        error_record = error_handler._create_error_record(
            exception, ErrorCategory.NETWORK, ErrorSeverity.HIGH, sample_error_context
        )
        error_handler._track_error(error_record)
        
        stats = error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 6
        assert stats["recent_errors_24h"] == 6
        assert "validation" in stats["errors_by_category"]
        assert "network" in stats["errors_by_category"]
        assert "medium" in stats["errors_by_severity"]
        assert "high" in stats["errors_by_severity"]
        assert len(stats["most_common_errors"]) > 0
    
    def test_reset_error_history(self, error_handler, sample_error_context):
        """Test error history reset functionality."""
        # Add some test errors
        for i in range(3):
            exception = ValueError(f"Test error {i}")
            error_record = error_handler._create_error_record(
                exception, ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM, sample_error_context
            )
            error_handler._track_error(error_record)
        
        initial_count = len(error_handler.error_history)
        assert initial_count == 3
        
        # Reset errors older than 0 hours (should remove all)
        removed_count = error_handler.reset_error_history(older_than_hours=0)
        
        assert removed_count == 3
        assert len(error_handler.error_history) == 0


class TestRetryDecorator:
    """Test cases for the retry decorator functionality."""
    
    @pytest.fixture
    def error_handler(self):
        """Create an ErrorHandler instance for testing."""
        config = {
            'max_retry_attempts': 2,
            'base_retry_delay': 0.01,  # Very short delay for testing
            'max_retry_delay': 0.1
        }
        return ErrorHandler(config)
    
    @pytest.mark.asyncio
    async def test_retry_decorator_async_success(self, error_handler):
        """Test retry decorator with async function that succeeds."""
        call_count = 0
        
        @error_handler.with_retry(category=ErrorCategory.UNKNOWN, severity=ErrorSeverity.LOW)
        async def test_async_func():
            nonlocal call_count
            call_count += 1
            return "async_success"
        
        result = await test_async_func()
        assert result == "async_success"
        assert call_count == 1
    
    @pytest.mark.asyncio
    async def test_retry_decorator_async_with_retries(self, error_handler):
        """Test retry decorator with async function that fails then succeeds."""
        call_count = 0
        
        @error_handler.with_retry(category=ErrorCategory.NETWORK, severity=ErrorSeverity.MEDIUM)
        async def test_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Temporary failure")
            return "async_success_after_retry"
        
        result = await test_async_func()
        assert result == "async_success_after_retry"
        assert call_count == 2


class TestUtilityFunctions:
    """Test cases for utility functions."""
    
    @pytest.mark.asyncio
    async def test_safe_execute_success(self):
        """Test safe_execute with successful function."""
        async def test_func(value):
            return value * 2
        
        success, result, error = await safe_execute(test_func, 5)
        
        assert success
        assert result == 10
        assert error is None
    
    @pytest.mark.asyncio
    async def test_safe_execute_failure(self):
        """Test safe_execute with failing function."""
        async def test_func():
            raise ValueError("Test error")
        
        success, result, error = await safe_execute(test_func)
        
        assert not success
        assert result is None
        assert error == "Test error"
    
    def test_create_fallback_response(self):
        """Test fallback response creation."""
        response = create_fallback_response(
            "Test error message",
            "test_operation",
            {"fallback": "data"}
        )
        
        assert not response["success"]
        assert response["error"]
        assert response["error_message"] == "Test error message"
        assert response["operation"] == "test_operation"
        assert response["fallback_data"] == {"fallback": "data"}
        assert "timestamp" in response
        assert "recovery_suggestions" in response
        assert len(response["recovery_suggestions"]) > 0


class TestErrorRecoveryScenarios:
    """Test cases for specific error recovery scenarios."""
    
    @pytest.fixture
    def error_handler(self):
        """Create an ErrorHandler instance for testing."""
        return ErrorHandler()
    
    def test_pdf_processing_recovery_scenarios(self, error_handler):
        """Test various PDF processing error recovery scenarios."""
        scenarios = [
            (FileNotFoundError("File not found"), "file_not_found", False),
            (PermissionError("Permission denied"), "permission_denied", False),
            (Exception("Corrupted PDF"), "corrupted_file", False),
            (MemoryError("Out of memory"), "memory_error", True),
            (Exception("Generic error"), "generic_error", True)
        ]
        
        for error, expected_type, should_retry in scenarios:
            result = error_handler.handle_pdf_processing_error(error, "/test/path.pdf")
            
            assert result["error_type"] == expected_type
            assert result["should_retry"] == should_retry
            assert not result["success"]
            assert "message" in result
    
    def test_knowledge_graph_recovery_scenarios(self, error_handler):
        """Test various knowledge graph error recovery scenarios."""
        scenarios = [
            (Exception("Storage error"), "storage_error", True),
            (Exception("Memory exhausted"), "memory_error", True),
            (Exception("Validation failed"), "validation_error", False),
            (Exception("Generic KG error"), "generic_kg_error", True)
        ]
        
        for error, expected_type, should_retry in scenarios:
            result = error_handler.handle_knowledge_graph_error(error, "test_operation")
            
            assert result["error_type"] == expected_type
            assert result["should_retry"] == should_retry
            assert not result["success"]
            assert "message" in result
    
    def test_query_processing_recovery_scenarios(self, error_handler):
        """Test various query processing error recovery scenarios."""
        scenarios = [
            (Exception("Query timeout"), "timeout_error", True, True),
            (Exception("Memory error"), "memory_error", True, True),
            (Exception("Knowledge graph not found"), "no_graph_data", False, True),
            (Exception("Generic query error"), "generic_query_error", True, True)
        ]
        
        for error, expected_type, should_retry, fallback_available in scenarios:
            result = error_handler.handle_query_processing_error(error, "test query")
            
            assert result["error_type"] == expected_type
            assert result["should_retry"] == should_retry
            assert result["fallback_available"] == fallback_available
            assert not result["success"]
            assert "message" in result


if __name__ == "__main__":
    pytest.main([__file__])