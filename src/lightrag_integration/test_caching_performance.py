"""
Tests for LightRAG Caching and Performance Optimization

This module tests the caching system and performance optimizations
to ensure they meet the requirements for scalability.
"""

import asyncio
import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from .caching import (
    LRUCache, QueryResultCache, EmbeddingCache, ConnectionPool, 
    CacheManager, CacheEntry, CacheStats
)
from .performance import (
    AsyncTaskManager, MemoryManager, PerformanceOptimizer,
    PerformanceMetrics, monitor_performance, managed_resources
)
from .config.settings import LightRAGConfig


class TestLRUCache:
    """Test LRU cache implementation."""
    
    def test_basic_operations(self):
        """Test basic cache operations."""
        cache = LRUCache(max_size=3, max_memory_mb=1)
        
        # Test put and get
        assert cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test cache miss
        assert cache.get("nonexistent") is None
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.total_entries == 1

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        cache = LRUCache(max_size=2, max_memory_mb=10)
        
        # Fill cache to capacity
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        # Access key1 to make it recently used
        cache.get("key1")
        
        # Add new item, should evict key2 (least recently used)
        cache.put("key3", "value3")
        
        assert cache.get("key1") == "value1"  # Still present
        assert cache.get("key2") is None      # Evicted
        assert cache.get("key3") == "value3"  # New item

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        cache = LRUCache(max_size=10, max_memory_mb=10)
        
        # Add item with short TTL
        cache.put("key1", "value1", ttl_seconds=1)
        
        # Should be available immediately
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired
        assert cache.get("key1") is None

    def test_memory_limit_eviction(self):
        """Test memory-based eviction."""
        # Create cache with very small memory limit
        cache = LRUCache(max_size=100, max_memory_mb=1)  # 1MB limit
        
        # Create large values that should trigger memory eviction
        large_value = "x" * (500 * 1024)  # 500KB
        
        cache.put("key1", large_value)
        cache.put("key2", large_value)
        
        # Adding third large value should trigger eviction
        cache.put("key3", large_value)
        
        # First item should be evicted due to memory pressure
        assert cache.get("key1") is None
        assert cache.get("key2") == large_value
        assert cache.get("key3") == large_value

    def test_thread_safety(self):
        """Test thread safety of cache operations."""
        cache = LRUCache(max_size=1000, max_memory_mb=10)
        results = []
        
        def worker(thread_id: int):
            """Worker function for threading test."""
            for i in range(100):
                key = f"thread_{thread_id}_key_{i}"
                value = f"thread_{thread_id}_value_{i}"
                
                cache.put(key, value)
                retrieved = cache.get(key)
                results.append(retrieved == value)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All operations should have succeeded
        assert all(results)


class TestQueryResultCache:
    """Test query result caching."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            query_cache_size=100,
            query_cache_memory_mb=10,
            cache_ttl_seconds=3600
        )
    
    @pytest.fixture
    def cache(self, config):
        """Create query result cache."""
        return QueryResultCache(config)
    
    @pytest.mark.asyncio
    async def test_query_caching(self, cache):
        """Test basic query result caching."""
        query = "What is clinical metabolomics?"
        result = {
            "answer": "Clinical metabolomics is...",
            "confidence_score": 0.9,
            "source_documents": ["doc1.pdf"]
        }
        
        # Cache result
        success = await cache.put(query, result)
        assert success
        
        # Retrieve result
        cached_result = await cache.get(query)
        assert cached_result is not None
        assert cached_result["answer"] == result["answer"]
        assert cached_result["metadata"]["cached"] is True

    @pytest.mark.asyncio
    async def test_query_normalization(self, cache):
        """Test query normalization for cache keys."""
        result = {"answer": "Test answer", "confidence_score": 0.8}
        
        # Cache with original query
        await cache.put("What is Clinical Metabolomics?", result)
        
        # Should find with normalized query
        cached = await cache.get("what is clinical metabolomics?")
        assert cached is not None
        assert cached["answer"] == result["answer"]

    @pytest.mark.asyncio
    async def test_context_sensitive_caching(self, cache):
        """Test caching with different contexts."""
        query = "What is metabolomics?"
        result1 = {"answer": "Answer 1", "confidence_score": 0.8}
        result2 = {"answer": "Answer 2", "confidence_score": 0.9}
        
        context1 = {"language": "en", "domain": "clinical"}
        context2 = {"language": "es", "domain": "clinical"}
        
        # Cache with different contexts
        await cache.put(query, result1, context1)
        await cache.put(query, result2, context2)
        
        # Should retrieve correct results for each context
        cached1 = await cache.get(query, context1)
        cached2 = await cache.get(query, context2)
        
        assert cached1["answer"] == result1["answer"]
        assert cached2["answer"] == result2["answer"]

    @pytest.mark.asyncio
    async def test_cache_statistics(self, cache):
        """Test cache statistics tracking."""
        # Perform some cache operations
        await cache.put("query1", {"answer": "answer1"})
        await cache.put("query2", {"answer": "answer2"})
        
        await cache.get("query1")  # Hit
        await cache.get("query2")  # Hit
        await cache.get("query3")  # Miss
        
        stats = cache.get_stats()
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 2/3
        assert stats["total_entries"] == 2


class TestEmbeddingCache:
    """Test embedding caching."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            embedding_cache_size=1000,
            embedding_cache_memory_mb=50,
            embedding_cache_ttl_seconds=86400
        )
    
    @pytest.fixture
    def cache(self, config):
        """Create embedding cache."""
        return EmbeddingCache(config)
    
    @pytest.mark.asyncio
    async def test_embedding_caching(self, cache):
        """Test basic embedding caching."""
        text = "Clinical metabolomics studies"
        model_name = "intfloat/e5-base-v2"
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Cache embedding
        success = await cache.put_embedding(text, model_name, embedding)
        assert success
        
        # Retrieve embedding
        cached_embedding = await cache.get_embedding(text, model_name)
        assert cached_embedding == embedding

    @pytest.mark.asyncio
    async def test_model_specific_caching(self, cache):
        """Test that embeddings are cached per model."""
        text = "metabolomics"
        embedding1 = [0.1, 0.2, 0.3]
        embedding2 = [0.4, 0.5, 0.6]
        
        # Cache same text with different models
        await cache.put_embedding(text, "model1", embedding1)
        await cache.put_embedding(text, "model2", embedding2)
        
        # Should retrieve correct embeddings for each model
        cached1 = await cache.get_embedding(text, "model1")
        cached2 = await cache.get_embedding(text, "model2")
        
        assert cached1 == embedding1
        assert cached2 == embedding2

    @pytest.mark.asyncio
    async def test_embedding_cache_performance(self, cache):
        """Test embedding cache performance with many entries."""
        # Add many embeddings
        embeddings = []
        for i in range(100):
            text = f"text_{i}"
            embedding = [float(j) for j in range(10)]  # 10-dimensional embedding
            embeddings.append((text, embedding))
            
            await cache.put_embedding(text, "test_model", embedding)
        
        # Measure retrieval performance
        start_time = time.time()
        
        for text, expected_embedding in embeddings:
            cached = await cache.get_embedding(text, "test_model")
            assert cached == expected_embedding
        
        retrieval_time = time.time() - start_time
        
        # Should be fast (less than 1 second for 100 retrievals)
        assert retrieval_time < 1.0
        
        # Check cache statistics
        stats = cache.get_stats()
        assert stats["hits"] == 100
        assert stats["hit_rate"] == 1.0


class TestConnectionPool:
    """Test database connection pooling."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            min_db_connections=2,
            max_db_connections=5,
            db_connection_timeout=10
        )
    
    @pytest.fixture
    def pool(self, config):
        """Create connection pool."""
        return ConnectionPool(config)
    
    @pytest.mark.asyncio
    async def test_connection_pool_initialization(self, pool):
        """Test connection pool initialization."""
        await pool.initialize()
        
        stats = pool.get_stats()
        assert stats["total_connections"] >= pool.min_connections
        assert stats["available_connections"] >= pool.min_connections

    @pytest.mark.asyncio
    async def test_connection_borrowing(self, pool):
        """Test connection borrowing and returning."""
        await pool.initialize()
        
        # Borrow connection
        conn = await pool.get_connection()
        assert conn is not None
        
        stats_after_borrow = pool.get_stats()
        assert stats_after_borrow["connections_borrowed"] == 1
        
        # Return connection
        await pool.return_connection(conn)
        
        stats_after_return = pool.get_stats()
        assert stats_after_return["connections_returned"] == 1

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, pool):
        """Test concurrent connection usage."""
        await pool.initialize()
        
        async def borrow_and_return():
            conn = await pool.get_connection()
            if conn:
                await asyncio.sleep(0.1)  # Simulate work
                await pool.return_connection(conn)
                return True
            return False
        
        # Create multiple concurrent tasks
        tasks = [borrow_and_return() for _ in range(10)]
        results = await asyncio.gather(*tasks)
        
        # All tasks should succeed
        assert all(results)
        
        # Check final stats
        stats = pool.get_stats()
        assert stats["connections_borrowed"] == 10
        assert stats["connections_returned"] == 10

    @pytest.mark.asyncio
    async def test_pool_exhaustion(self, pool):
        """Test behavior when pool is exhausted."""
        await pool.initialize()
        
        # Borrow all available connections
        connections = []
        for _ in range(pool.max_connections):
            conn = await pool.get_connection()
            if conn:
                connections.append(conn)
        
        # Try to borrow one more (should timeout or fail)
        start_time = time.time()
        extra_conn = await pool.get_connection()
        elapsed_time = time.time() - start_time
        
        # Should either return None or take significant time
        assert extra_conn is None or elapsed_time > 1.0
        
        # Return all connections
        for conn in connections:
            await pool.return_connection(conn)


class TestAsyncTaskManager:
    """Test async task management."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            max_concurrent_tasks=5,
            task_timeout_seconds=10,
            max_memory_mb=1024,
            max_cpu_percent=80.0
        )
    
    @pytest.fixture
    def task_manager(self, config):
        """Create task manager."""
        return AsyncTaskManager(config)
    
    @pytest.mark.asyncio
    async def test_task_execution(self, task_manager):
        """Test basic task execution."""
        async def test_task():
            await asyncio.sleep(0.1)
            return "task_result"
        
        result = await task_manager.execute_task(test_task(), "test_task")
        assert result == "task_result"
        
        stats = task_manager.get_task_stats()
        assert stats["completed_tasks"] == 1
        assert stats["failed_tasks"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_task_limit(self, task_manager):
        """Test concurrent task limiting."""
        async def slow_task(task_id: int):
            await asyncio.sleep(0.5)
            return f"result_{task_id}"
        
        # Start more tasks than the limit
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                task_manager.execute_task(slow_task(i), f"task_{i}")
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # All tasks should complete successfully
        assert len(results) == 10
        assert all(f"result_{i}" in results for i in range(10))
        
        stats = task_manager.get_task_stats()
        assert stats["completed_tasks"] == 10

    @pytest.mark.asyncio
    async def test_task_timeout(self, task_manager):
        """Test task timeout handling."""
        async def timeout_task():
            await asyncio.sleep(20)  # Longer than timeout
            return "should_not_complete"
        
        with pytest.raises(asyncio.TimeoutError):
            await task_manager.execute_task(timeout_task(), "timeout_task", timeout=1.0)
        
        stats = task_manager.get_task_stats()
        assert stats["failed_tasks"] == 1

    @pytest.mark.asyncio
    async def test_batch_execution(self, task_manager):
        """Test batch task execution."""
        async def batch_task(value: int):
            await asyncio.sleep(0.1)
            return value * 2
        
        # Create batch of tasks
        tasks = [batch_task(i) for i in range(20)]
        
        # Execute in batches
        results = await task_manager.execute_batch(tasks, batch_size=5)
        
        # Check results
        assert len(results) == 20
        for i, result in enumerate(results):
            assert result == i * 2

    @pytest.mark.asyncio
    async def test_task_cancellation(self, task_manager):
        """Test task cancellation."""
        async def long_task():
            await asyncio.sleep(10)
            return "completed"
        
        # Start a long-running task
        task = asyncio.create_task(
            task_manager.execute_task(long_task(), "long_task")
        )
        
        # Let it start
        await asyncio.sleep(0.1)
        
        # Cancel all tasks
        await task_manager.cancel_all_tasks()
        
        # Task should be cancelled
        with pytest.raises(asyncio.CancelledError):
            await task


class TestMemoryManager:
    """Test memory management."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            max_memory_mb=1024,
            memory_warning_threshold=0.8,
            memory_critical_threshold=0.9,
            auto_gc_enabled=True
        )
    
    @pytest.fixture
    def memory_manager(self, config):
        """Create memory manager."""
        return MemoryManager(config)
    
    def test_memory_usage_tracking(self, memory_manager):
        """Test memory usage tracking."""
        usage = memory_manager.get_memory_usage()
        
        assert "process_memory_mb" in usage
        assert "system_memory_percent" in usage
        assert usage["process_memory_mb"] > 0
        assert 0 <= usage["system_memory_percent"] <= 100

    def test_memory_pressure_detection(self, memory_manager):
        """Test memory pressure detection."""
        pressure_info = memory_manager.check_memory_pressure()
        
        assert "pressure_level" in pressure_info
        assert pressure_info["pressure_level"] in ["normal", "warning", "critical"]
        assert "recommendations" in pressure_info
        assert isinstance(pressure_info["recommendations"], list)

    @pytest.mark.asyncio
    async def test_memory_optimization(self, memory_manager):
        """Test memory optimization."""
        # Force optimization
        result = await memory_manager.optimize_memory(force=True)
        
        assert result["optimization_performed"] is True
        assert "memory_freed_mb" in result
        assert "memory_before" in result
        assert "memory_after" in result

    def test_memory_statistics(self, memory_manager):
        """Test memory statistics."""
        stats = memory_manager.get_memory_stats()
        
        assert "current_usage" in stats
        assert "pressure_info" in stats
        assert "limits" in stats
        assert "gc_stats" in stats


class TestPerformanceOptimizer:
    """Test performance optimizer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            performance_monitoring_enabled=True,
            performance_monitoring_interval=1,
            auto_optimization_enabled=True,
            optimization_interval=5
        )
    
    @pytest.fixture
    def optimizer(self, config):
        """Create performance optimizer."""
        return PerformanceOptimizer(config)
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self, optimizer):
        """Test performance monitoring."""
        # Start monitoring
        await optimizer.start_monitoring()
        
        # Let it collect some metrics
        await asyncio.sleep(2)
        
        # Stop monitoring
        await optimizer.stop_monitoring()
        
        # Check that metrics were collected
        summary = optimizer.get_performance_summary()
        assert "current_metrics" in summary
        assert "averages" in summary
        assert len(optimizer._performance_history) > 0

    @pytest.mark.asyncio
    async def test_performance_optimization(self, optimizer):
        """Test performance optimization."""
        result = await optimizer.optimize_performance(force=True)
        
        assert "optimization_time" in result
        assert "results" in result
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_performance_summary(self, optimizer):
        """Test performance summary generation."""
        summary = optimizer.get_performance_summary()
        
        required_keys = [
            "current_metrics", "averages", "task_stats", 
            "memory_stats", "monitoring", "optimization"
        ]
        
        for key in required_keys:
            assert key in summary

    @pytest.mark.asyncio
    async def test_cleanup(self, optimizer):
        """Test optimizer cleanup."""
        await optimizer.start_monitoring()
        await asyncio.sleep(0.5)
        
        # Cleanup should not raise exceptions
        await optimizer.cleanup()


class TestCacheManager:
    """Test cache manager integration."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            query_cache_size=100,
            embedding_cache_size=500,
            min_db_connections=2,
            max_db_connections=5,
            enable_cache_warming=False  # Disable for testing
        )
    
    @pytest.fixture
    def cache_manager(self, config):
        """Create cache manager."""
        return CacheManager(config)
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self, cache_manager):
        """Test cache manager initialization."""
        await cache_manager.initialize()
        
        # Should have initialized all components
        assert cache_manager.query_cache is not None
        assert cache_manager.embedding_cache is not None
        assert cache_manager.connection_pool is not None

    @pytest.mark.asyncio
    async def test_integrated_caching(self, cache_manager):
        """Test integrated caching operations."""
        await cache_manager.initialize()
        
        # Test query caching
        query = "test query"
        result = {"answer": "test answer"}
        
        success = await cache_manager.cache_query_result(query, result)
        assert success
        
        cached_result = await cache_manager.get_query_result(query)
        assert cached_result["answer"] == result["answer"]
        
        # Test embedding caching
        text = "test text"
        embedding = [0.1, 0.2, 0.3]
        
        success = await cache_manager.cache_embedding(text, "test_model", embedding)
        assert success
        
        cached_embedding = await cache_manager.get_embedding(text, "test_model")
        assert cached_embedding == embedding

    @pytest.mark.asyncio
    async def test_comprehensive_statistics(self, cache_manager):
        """Test comprehensive statistics collection."""
        await cache_manager.initialize()
        
        # Perform some operations
        await cache_manager.cache_query_result("query1", {"answer": "answer1"})
        await cache_manager.cache_embedding("text1", "model1", [0.1, 0.2])
        
        stats = cache_manager.get_comprehensive_stats()
        
        assert "query_cache" in stats
        assert "embedding_cache" in stats
        assert "connection_pool" in stats
        assert "cache_manager" in stats

    @pytest.mark.asyncio
    async def test_cache_cleanup(self, cache_manager):
        """Test cache cleanup."""
        await cache_manager.initialize()
        
        # Add some data
        await cache_manager.cache_query_result("query", {"answer": "answer"})
        
        # Clear caches
        await cache_manager.clear_all_caches()
        
        # Data should be gone
        result = await cache_manager.get_query_result("query")
        assert result is None
        
        # Cleanup should not raise exceptions
        await cache_manager.cleanup()


class TestPerformanceDecorators:
    """Test performance monitoring decorators."""
    
    @pytest.mark.asyncio
    async def test_async_performance_monitoring(self):
        """Test async function performance monitoring."""
        @monitor_performance("test_async_function")
        async def test_function():
            await asyncio.sleep(0.1)
            return "result"
        
        # Should complete without errors
        result = await test_function()
        assert result == "result"

    def test_sync_performance_monitoring(self):
        """Test sync function performance monitoring."""
        @monitor_performance("test_sync_function")
        def test_function():
            time.sleep(0.1)
            return "result"
        
        # Should complete without errors
        result = test_function()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_managed_resources(self):
        """Test managed resources context manager."""
        config = LightRAGConfig()
        optimizer = PerformanceOptimizer(config)
        
        async with managed_resources(optimizer, "test_resource"):
            # Should execute without errors
            await asyncio.sleep(0.1)


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "-k", "test_lru_cache or test_query_caching or test_performance",
        "--tb=short"
    ])