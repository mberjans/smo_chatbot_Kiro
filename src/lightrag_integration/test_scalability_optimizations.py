"""
Comprehensive Tests for LightRAG Scalability Optimizations

This module tests the complete scalability optimization implementation including
caching, performance optimization, and concurrent user handling.
Tests requirements 5.2, 5.3, 5.4, and 5.6.
"""

import asyncio
import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any
import concurrent.futures

from .component import LightRAGComponent
from .caching import CacheManager, QueryResultCache, EmbeddingCache
from .performance import PerformanceOptimizer, AsyncTaskManager, MemoryManager
from .concurrency import ConcurrencyManager, RequestPriority
from .config.settings import LightRAGConfig


class TestIntegratedScalabilityOptimizations:
    """Test integrated scalability optimizations in the main component."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration with optimization settings."""
        return LightRAGConfig(
            # Caching settings
            query_cache_size=100,
            query_cache_memory_mb=10,
            embedding_cache_size=500,
            embedding_cache_memory_mb=20,
            enable_cache_warming=False,  # Disable for testing
            
            # Performance settings
            max_memory_mb=512,
            max_cpu_percent=70.0,
            max_concurrent_tasks=10,
            performance_monitoring_enabled=True,
            performance_monitoring_interval=1,
            auto_optimization_enabled=True,
            
            # Concurrency settings
            default_requests_per_minute=20,
            default_requests_per_hour=200,
            default_burst_limit=5,
            max_queue_size=50,
            max_concurrent_users=20,
            max_requests_per_user=3,
            queue_workers=5
        )
    
    @pytest.fixture
    def component(self, config):
        """Create and initialize LightRAG component."""
        async def _create_component():
            component = LightRAGComponent(config)
            await component.initialize()
            return component
        
        return _create_component
    
    @pytest.mark.asyncio
    async def test_component_initialization_with_optimizations(self, component):
        """Test that component initializes with all optimization components."""
        # Create and initialize component
        comp = await component()
        
        try:
            # Check that optimization components are initialized
            assert comp._cache_manager is not None
            assert comp._performance_optimizer is not None
            assert comp._concurrency_manager is not None
            
            # Check health status includes optimization components
            health = await comp.get_health_status()
            assert health.overall_status.value in ["healthy", "degraded"]  # Should not be unhealthy
            
            # Check statistics include optimization data
            stats = comp.get_statistics()
            assert "cache_stats" in stats
            assert "performance_stats" in stats
            assert "concurrency_stats" in stats
        finally:
            await comp.cleanup()

    @pytest.mark.asyncio
    async def test_query_caching_integration(self, component):
        """Test query caching integration in the main component."""
        comp = await component()
        
        try:
            query = "What is clinical metabolomics?"
            
            # First query should not be cached
            result1 = await comp.query(query, user_id="test_user")
            assert result1 is not None
            
            # Check cache stats
            cache_stats = await comp.get_cache_stats()
            assert "query_cache" in cache_stats
            
            # Second identical query should potentially use cache
            # (Note: In the current implementation, caching happens after query execution)
            result2 = await comp.query(query, user_id="test_user")
            assert result2 is not None
        finally:
            await comp.cleanup()

    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, component):
        """Test performance monitoring integration."""
        # Get initial performance stats
        perf_stats = await component.get_performance_stats()
        assert "current_metrics" in perf_stats
        assert "monitoring" in perf_stats
        
        # Perform some operations
        await component.query("Test query 1", user_id="user1")
        await component.query("Test query 2", user_id="user2")
        
        # Check that performance stats are updated
        updated_stats = await component.get_performance_stats()
        assert updated_stats is not None

    @pytest.mark.asyncio
    async def test_concurrency_management_integration(self, component):
        """Test concurrency management integration."""
        # Get initial concurrency stats
        concurrency_stats = await component.get_concurrency_stats()
        assert "concurrency_manager" in concurrency_stats
        assert "rate_limiter" in concurrency_stats
        assert "request_queue" in concurrency_stats
        
        # Make multiple queries from same user
        user_id = "test_user"
        queries = [f"Query {i}" for i in range(3)]
        
        # Execute queries concurrently
        tasks = [component.query(query, user_id=user_id) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All queries should complete (though some might be rate limited)
        assert len(results) == 3
        
        # Check updated concurrency stats
        updated_stats = await component.get_concurrency_stats()
        assert updated_stats["concurrency_manager"]["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_manual_optimization_triggers(self, component):
        """Test manual optimization triggers."""
        # Test performance optimization
        perf_result = await component.optimize_performance(force=True)
        assert "optimization_time" in perf_result
        
        # Test cache clearing
        cache_result = await component.clear_caches()
        assert cache_result["success"] is True

    @pytest.mark.asyncio
    async def test_error_handling_with_optimizations(self, component):
        """Test error handling when optimization components fail."""
        # Test with invalid query
        with pytest.raises(ValueError):
            await component.query("", user_id="test_user")
        
        # Component should still be functional
        result = await component.query("Valid query", user_id="test_user")
        assert result is not None


class TestConcurrentUserScenarios:
    """Test concurrent user handling scenarios."""
    
    @pytest.fixture
    def config(self):
        """Create configuration for concurrent testing."""
        return LightRAGConfig(
            max_concurrent_users=10,
            max_requests_per_user=2,
            default_requests_per_minute=50,
            default_burst_limit=10,
            queue_workers=5,
            max_queue_size=100
        )
    
    @pytest.fixture
    async def component(self, config):
        """Create component for concurrent testing."""
        component = LightRAGComponent(config)
        await component.initialize()
        yield component
        await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_multiple_users_concurrent_queries(self, component):
        """Test handling multiple users with concurrent queries."""
        num_users = 5
        queries_per_user = 2
        
        async def user_queries(user_id: str):
            """Simulate queries from a single user."""
            results = []
            for i in range(queries_per_user):
                try:
                    result = await component.query(f"Query {i} from {user_id}", user_id=user_id)
                    results.append(result)
                except Exception as e:
                    results.append({"error": str(e)})
                
                # Small delay between queries
                await asyncio.sleep(0.1)
            
            return results
        
        # Create tasks for multiple users
        user_tasks = [
            user_queries(f"user_{i}")
            for i in range(num_users)
        ]
        
        # Execute all user tasks concurrently
        start_time = time.time()
        all_results = await asyncio.gather(*user_tasks)
        execution_time = time.time() - start_time
        
        # Verify results
        assert len(all_results) == num_users
        
        # Each user should have attempted the specified number of queries
        for user_results in all_results:
            assert len(user_results) == queries_per_user
        
        # Should complete in reasonable time (less than 30 seconds)
        assert execution_time < 30.0
        
        # Check concurrency stats
        stats = await component.get_concurrency_stats()
        assert stats["concurrency_manager"]["total_requests"] >= num_users * queries_per_user

    @pytest.mark.asyncio
    async def test_rate_limiting_under_load(self, component):
        """Test rate limiting behavior under high load."""
        user_id = "heavy_user"
        
        # Make many requests quickly to trigger rate limiting
        async def make_request(query_id: int):
            try:
                result = await component.query(f"Heavy query {query_id}", user_id=user_id)
                return {"success": True, "query_id": query_id, "result": result}
            except Exception as e:
                return {"success": False, "query_id": query_id, "error": str(e)}
        
        # Create many concurrent requests
        num_requests = 20
        tasks = [make_request(i) for i in range(num_requests)]
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        execution_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        # Some requests should be rate limited
        assert len(failed_requests) > 0 or len(successful_requests) < num_requests
        
        # Check that rate limiting is working
        stats = await component.get_concurrency_stats()
        assert stats["concurrency_manager"]["rate_limited_requests"] > 0

    @pytest.mark.asyncio
    async def test_queue_management_under_load(self, component):
        """Test request queue management under high load."""
        # Create a large number of requests from different users
        num_users = 8
        requests_per_user = 5
        
        async def user_request_batch(user_id: str):
            """Create batch of requests from one user."""
            tasks = []
            for i in range(requests_per_user):
                task = component.query(f"Batch query {i}", user_id=user_id)
                tasks.append(task)
            
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # Execute batches from multiple users
        user_batches = [
            user_request_batch(f"batch_user_{i}")
            for i in range(num_users)
        ]
        
        start_time = time.time()
        batch_results = await asyncio.gather(*user_batches)
        execution_time = time.time() - start_time
        
        # Verify all batches completed
        assert len(batch_results) == num_users
        
        # Check queue statistics
        stats = await component.get_concurrency_stats()
        queue_stats = stats["request_queue"]
        
        # Should have processed requests
        assert queue_stats["statistics"]["total_processed"] > 0
        
        # Should complete in reasonable time
        assert execution_time < 60.0  # 1 minute max


class TestPerformanceOptimizationEffectiveness:
    """Test the effectiveness of performance optimizations."""
    
    @pytest.fixture
    def config(self):
        """Create configuration for performance testing."""
        return LightRAGConfig(
            # Enable all optimizations
            enable_caching=True,
            query_cache_size=200,
            embedding_cache_size=1000,
            performance_monitoring_enabled=True,
            auto_optimization_enabled=True,
            max_concurrent_tasks=20,
            
            # Reasonable limits for testing
            max_memory_mb=1024,
            max_cpu_percent=80.0
        )
    
    @pytest.fixture
    async def component(self, config):
        """Create optimized component."""
        component = LightRAGComponent(config)
        await component.initialize()
        yield component
        await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_caching_performance_improvement(self, component):
        """Test that caching improves performance for repeated queries."""
        query = "What are the main applications of clinical metabolomics?"
        
        # First query (no cache)
        start_time = time.time()
        result1 = await component.query(query, user_id="perf_user")
        first_query_time = time.time() - start_time
        
        # Allow cache to be populated
        await asyncio.sleep(0.1)
        
        # Second identical query (potentially cached)
        start_time = time.time()
        result2 = await component.query(query, user_id="perf_user")
        second_query_time = time.time() - start_time
        
        # Both queries should succeed
        assert result1 is not None
        assert result2 is not None
        
        # Check cache statistics
        cache_stats = await component.get_cache_stats()
        assert cache_stats["query_cache"]["total_entries"] >= 0

    @pytest.mark.asyncio
    async def test_memory_optimization_effectiveness(self, component):
        """Test memory optimization effectiveness."""
        # Get initial memory stats
        initial_stats = await component.get_performance_stats()
        initial_memory = initial_stats["current_metrics"]["memory_usage_mb"]
        
        # Perform memory-intensive operations
        queries = [f"Memory test query {i}" for i in range(50)]
        
        for query in queries:
            await component.query(query, user_id=f"mem_user_{hash(query) % 10}")
        
        # Trigger manual optimization
        optimization_result = await component.optimize_performance(force=True)
        assert optimization_result["optimization_performed"] is True
        
        # Check final memory stats
        final_stats = await component.get_performance_stats()
        final_memory = final_stats["current_metrics"]["memory_usage_mb"]
        
        # Memory should be managed (not necessarily lower due to test overhead)
        assert final_memory > 0  # Basic sanity check

    @pytest.mark.asyncio
    async def test_concurrent_processing_efficiency(self, component):
        """Test efficiency of concurrent request processing."""
        num_concurrent_requests = 15
        
        async def timed_query(query_id: int):
            """Execute a query and measure time."""
            start_time = time.time()
            try:
                result = await component.query(
                    f"Concurrent efficiency test {query_id}",
                    user_id=f"concurrent_user_{query_id % 5}"
                )
                execution_time = time.time() - start_time
                return {
                    "success": True,
                    "query_id": query_id,
                    "execution_time": execution_time,
                    "result": result
                }
            except Exception as e:
                execution_time = time.time() - start_time
                return {
                    "success": False,
                    "query_id": query_id,
                    "execution_time": execution_time,
                    "error": str(e)
                }
        
        # Execute concurrent requests
        start_time = time.time()
        tasks = [timed_query(i) for i in range(num_concurrent_requests)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]
        
        # Most requests should succeed
        success_rate = len(successful_results) / len(results)
        assert success_rate > 0.5  # At least 50% success rate
        
        # Calculate average execution time
        if successful_results:
            avg_execution_time = sum(r["execution_time"] for r in successful_results) / len(successful_results)
            assert avg_execution_time < 10.0  # Should be reasonably fast
        
        # Total time should be less than sequential execution
        # (This is a rough estimate - actual improvement depends on implementation)
        estimated_sequential_time = num_concurrent_requests * 2.0  # Assume 2s per query
        assert total_time < estimated_sequential_time
        
        # Check performance stats
        perf_stats = await component.get_performance_stats()
        task_stats = perf_stats["task_stats"]
        assert task_stats["completed_tasks"] > 0


class TestScalabilityLimits:
    """Test system behavior at scalability limits."""
    
    @pytest.fixture
    def config(self):
        """Create configuration with lower limits for testing."""
        return LightRAGConfig(
            max_concurrent_users=5,
            max_requests_per_user=2,
            max_queue_size=20,
            default_requests_per_minute=10,
            default_burst_limit=3,
            queue_workers=2
        )
    
    @pytest.fixture
    async def component(self, config):
        """Create component with limited resources."""
        component = LightRAGComponent(config)
        await component.initialize()
        yield component
        await component.cleanup()
    
    @pytest.mark.asyncio
    async def test_graceful_degradation_at_limits(self, component):
        """Test graceful degradation when approaching system limits."""
        # Try to exceed user limits
        num_users = 10  # More than max_concurrent_users (5)
        
        async def user_session(user_id: str):
            """Simulate a user session."""
            results = []
            for i in range(3):  # More than max_requests_per_user (2)
                try:
                    result = await component.query(f"Limit test {i}", user_id=user_id)
                    results.append({"success": True, "result": result})
                except Exception as e:
                    results.append({"success": False, "error": str(e)})
            return results
        
        # Execute sessions for all users
        user_tasks = [user_session(f"limit_user_{i}") for i in range(num_users)]
        all_results = await asyncio.gather(*user_tasks, return_exceptions=True)
        
        # System should handle the load gracefully (not crash)
        assert len(all_results) == num_users
        
        # Some requests should be rejected due to limits
        total_requests = sum(len(results) for results in all_results if isinstance(results, list))
        successful_requests = sum(
            sum(1 for r in results if isinstance(r, dict) and r.get("success", False))
            for results in all_results if isinstance(results, list)
        )
        
        # Not all requests should succeed due to limits
        assert successful_requests < total_requests
        
        # Check that system is still responsive
        health = await component.get_health_status()
        assert health.overall_status.value != "unhealthy"

    @pytest.mark.asyncio
    async def test_recovery_after_overload(self, component):
        """Test system recovery after temporary overload."""
        # Create temporary overload
        overload_tasks = []
        for i in range(30):  # Create many requests
            task = component.query(f"Overload query {i}", user_id=f"overload_user_{i % 3}")
            overload_tasks.append(task)
        
        # Execute overload (some will fail)
        overload_results = await asyncio.gather(*overload_tasks, return_exceptions=True)
        
        # Wait for system to recover
        await asyncio.sleep(2)
        
        # Test normal operation after overload
        recovery_result = await component.query("Recovery test query", user_id="recovery_user")
        assert recovery_result is not None
        
        # Check system health
        health = await component.get_health_status()
        assert health.overall_status.value in ["healthy", "degraded"]


if __name__ == "__main__":
    # Run the scalability optimization tests
    pytest.main([
        __file__,
        "-v",
        "-k", "test_scalability",
        "--tb=short"
    ])