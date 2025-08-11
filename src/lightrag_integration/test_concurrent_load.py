"""
Load Tests for Concurrent User Handling

This module implements comprehensive load tests to validate that the LightRAG
system can handle multiple concurrent users efficiently.
Tests requirements 5.2 and 5.4.
"""

import asyncio
import pytest
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any
import concurrent.futures
import threading

from .component import LightRAGComponent
from .concurrency import ConcurrencyManager, RequestPriority
from .config.settings import LightRAGConfig


class TestConcurrentUserLoad:
    """Test concurrent user load scenarios."""
    
    @pytest.fixture
    def load_config(self):
        """Create configuration optimized for load testing."""
        return LightRAGConfig(
            # Concurrency settings
            max_concurrent_users=50,
            max_requests_per_user=10,
            default_requests_per_minute=100,
            default_requests_per_hour=1000,
            default_burst_limit=20,
            max_queue_size=500,
            queue_workers=20,
            
            # Performance settings
            max_concurrent_tasks=100,
            task_timeout_seconds=30,
            max_memory_mb=1024,
            performance_monitoring_enabled=True,
            performance_monitoring_interval=5,
            
            # Caching settings
            query_cache_size=1000,
            embedding_cache_size=5000,
            enable_cache_warming=False,  # Disable for testing
            
            # Disable background monitoring for cleaner tests
            auto_optimization_enabled=False
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_users_basic_load(self, load_config):
        """Test basic concurrent user load."""
        component = LightRAGComponent(load_config)
        
        try:
            await component.initialize()
            
            num_users = 10
            queries_per_user = 5
            
            async def simulate_user(user_id: str) -> Dict[str, Any]:
                """Simulate a user making multiple queries."""
                user_results = []
                start_time = time.time()
                
                for i in range(queries_per_user):
                    query = f"User {user_id} query {i}: What is metabolomics?"
                    
                    try:
                        result = await component.query(query, user_id=user_id)
                        user_results.append({
                            "query_id": i,
                            "success": True,
                            "response_time": time.time() - start_time,
                            "has_answer": "answer" in result and len(result["answer"]) > 0
                        })
                    except Exception as e:
                        user_results.append({
                            "query_id": i,
                            "success": False,
                            "error": str(e),
                            "response_time": time.time() - start_time
                        })
                    
                    # Small delay between queries
                    await asyncio.sleep(0.1)
                
                total_time = time.time() - start_time
                return {
                    "user_id": user_id,
                    "total_time": total_time,
                    "results": user_results,
                    "success_count": sum(1 for r in user_results if r["success"]),
                    "failure_count": sum(1 for r in user_results if not r["success"])
                }
            
            # Execute concurrent user simulations
            start_time = time.time()
            user_tasks = [simulate_user(f"load_user_{i}") for i in range(num_users)]
            user_results = await asyncio.gather(*user_tasks)
            total_execution_time = time.time() - start_time
            
            # Analyze results
            total_queries = num_users * queries_per_user
            successful_queries = sum(r["success_count"] for r in user_results)
            failed_queries = sum(r["failure_count"] for r in user_results)
            
            # Assertions
            assert len(user_results) == num_users
            assert successful_queries + failed_queries == total_queries
            
            # At least 70% of queries should succeed
            success_rate = successful_queries / total_queries
            assert success_rate >= 0.7, f"Success rate {success_rate:.2%} is below 70%"
            
            # Should complete in reasonable time (less than 60 seconds)
            assert total_execution_time < 60.0, f"Execution took {total_execution_time:.2f}s"
            
            # Check system statistics
            stats = component.get_statistics()
            assert stats["queries_processed"] >= successful_queries
            
            print(f"Load test results: {successful_queries}/{total_queries} queries succeeded "
                  f"({success_rate:.2%}) in {total_execution_time:.2f}s")
            
        finally:
            await component.cleanup()

    @pytest.mark.asyncio
    async def test_high_concurrency_burst_load(self, load_config):
        """Test system behavior under high concurrency burst load."""
        component = LightRAGComponent(load_config)
        
        try:
            await component.initialize()
            
            # Create a burst of concurrent requests
            num_concurrent_requests = 50
            
            async def burst_request(request_id: int) -> Dict[str, Any]:
                """Execute a single request in the burst."""
                start_time = time.time()
                user_id = f"burst_user_{request_id % 10}"  # 10 different users
                
                try:
                    result = await component.query(
                        f"Burst request {request_id}: Clinical metabolomics applications",
                        user_id=user_id
                    )
                    
                    return {
                        "request_id": request_id,
                        "user_id": user_id,
                        "success": True,
                        "response_time": time.time() - start_time,
                        "queued": result.get("metadata", {}).get("queued", False),
                        "rate_limited": result.get("metadata", {}).get("rate_limited", False)
                    }
                    
                except Exception as e:
                    return {
                        "request_id": request_id,
                        "user_id": user_id,
                        "success": False,
                        "error": str(e),
                        "response_time": time.time() - start_time
                    }
            
            # Execute burst load
            start_time = time.time()
            burst_tasks = [burst_request(i) for i in range(num_concurrent_requests)]
            burst_results = await asyncio.gather(*burst_tasks)
            total_time = time.time() - start_time
            
            # Analyze burst results
            successful_requests = [r for r in burst_results if r["success"]]
            failed_requests = [r for r in burst_results if not r["success"]]
            queued_requests = [r for r in successful_requests if r.get("queued", False)]
            rate_limited_requests = [r for r in successful_requests if r.get("rate_limited", False)]
            
            # System should handle the burst gracefully
            assert len(burst_results) == num_concurrent_requests
            
            # Some requests should succeed
            success_rate = len(successful_requests) / num_concurrent_requests
            assert success_rate > 0.3, f"Success rate {success_rate:.2%} is too low for burst load"
            
            # Should complete in reasonable time
            assert total_time < 120.0, f"Burst load took {total_time:.2f}s"
            
            # Check concurrency statistics
            concurrency_stats = await component.get_concurrency_stats()
            assert concurrency_stats["concurrency_manager"]["total_requests"] >= num_concurrent_requests
            
            print(f"Burst load results: {len(successful_requests)}/{num_concurrent_requests} "
                  f"succeeded ({success_rate:.2%}), {len(queued_requests)} queued, "
                  f"{len(rate_limited_requests)} rate limited in {total_time:.2f}s")
            
        finally:
            await component.cleanup()

    @pytest.mark.asyncio
    async def test_sustained_concurrent_load(self, load_config):
        """Test sustained concurrent load over time."""
        # Reduce limits for sustained testing
        load_config.max_concurrent_users = 20
        load_config.default_requests_per_minute = 60
        
        component = LightRAGComponent(load_config)
        
        try:
            await component.initialize()
            
            # Sustained load parameters
            duration_seconds = 30
            users_count = 15
            request_interval = 2.0  # Seconds between requests per user
            
            async def sustained_user_load(user_id: str, duration: float) -> Dict[str, Any]:
                """Generate sustained load from a single user."""
                results = []
                start_time = time.time()
                request_count = 0
                
                while time.time() - start_time < duration:
                    request_start = time.time()
                    
                    try:
                        result = await component.query(
                            f"Sustained query {request_count} from {user_id}",
                            user_id=user_id
                        )
                        
                        results.append({
                            "request_id": request_count,
                            "success": True,
                            "response_time": time.time() - request_start,
                            "timestamp": time.time()
                        })
                        
                    except Exception as e:
                        results.append({
                            "request_id": request_count,
                            "success": False,
                            "error": str(e),
                            "response_time": time.time() - request_start,
                            "timestamp": time.time()
                        })
                    
                    request_count += 1
                    
                    # Wait for next request interval
                    elapsed = time.time() - request_start
                    sleep_time = max(0, request_interval - elapsed)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                
                return {
                    "user_id": user_id,
                    "duration": time.time() - start_time,
                    "request_count": request_count,
                    "results": results,
                    "success_count": sum(1 for r in results if r["success"]),
                    "avg_response_time": sum(r["response_time"] for r in results) / len(results) if results else 0
                }
            
            # Start sustained load from multiple users
            start_time = time.time()
            user_tasks = [
                sustained_user_load(f"sustained_user_{i}", duration_seconds)
                for i in range(users_count)
            ]
            
            user_results = await asyncio.gather(*user_tasks)
            total_duration = time.time() - start_time
            
            # Analyze sustained load results
            total_requests = sum(r["request_count"] for r in user_results)
            total_successful = sum(r["success_count"] for r in user_results)
            avg_response_times = [r["avg_response_time"] for r in user_results if r["avg_response_time"] > 0]
            
            # Assertions for sustained load
            assert len(user_results) == users_count
            assert total_requests > 0
            
            # Success rate should be reasonable for sustained load
            success_rate = total_successful / total_requests if total_requests > 0 else 0
            assert success_rate >= 0.5, f"Sustained load success rate {success_rate:.2%} is too low"
            
            # Average response time should be reasonable
            if avg_response_times:
                overall_avg_response_time = sum(avg_response_times) / len(avg_response_times)
                assert overall_avg_response_time < 10.0, f"Average response time {overall_avg_response_time:.2f}s is too high"
            
            # Check final system state
            final_stats = component.get_statistics()
            concurrency_stats = await component.get_concurrency_stats()
            
            print(f"Sustained load results: {total_successful}/{total_requests} requests "
                  f"succeeded ({success_rate:.2%}) over {total_duration:.1f}s")
            
            if avg_response_times:
                print(f"Average response time: {overall_avg_response_time:.2f}s")
            
        finally:
            await component.cleanup()

    @pytest.mark.asyncio
    async def test_mixed_priority_concurrent_load(self, load_config):
        """Test concurrent load with mixed request priorities."""
        component = LightRAGComponent(load_config)
        
        try:
            await component.initialize()
            
            # Mixed priority load parameters
            high_priority_users = 3
            normal_priority_users = 7
            low_priority_users = 5
            requests_per_user = 3
            
            async def priority_user_load(user_id: str, priority: RequestPriority, num_requests: int) -> Dict[str, Any]:
                """Generate load with specific priority."""
                results = []
                
                for i in range(num_requests):
                    start_time = time.time()
                    
                    try:
                        # Note: Current implementation doesn't expose priority in query method
                        # This is a placeholder for when priority support is added
                        result = await component.query(
                            f"Priority {priority.name} query {i} from {user_id}",
                            user_id=user_id
                        )
                        
                        results.append({
                            "request_id": i,
                            "priority": priority.name,
                            "success": True,
                            "response_time": time.time() - start_time
                        })
                        
                    except Exception as e:
                        results.append({
                            "request_id": i,
                            "priority": priority.name,
                            "success": False,
                            "error": str(e),
                            "response_time": time.time() - start_time
                        })
                    
                    # Small delay between requests
                    await asyncio.sleep(0.2)
                
                return {
                    "user_id": user_id,
                    "priority": priority.name,
                    "results": results,
                    "success_count": sum(1 for r in results if r["success"]),
                    "avg_response_time": sum(r["response_time"] for r in results) / len(results) if results else 0
                }
            
            # Create mixed priority tasks
            all_tasks = []
            
            # High priority users
            for i in range(high_priority_users):
                task = priority_user_load(f"high_user_{i}", RequestPriority.HIGH, requests_per_user)
                all_tasks.append(task)
            
            # Normal priority users
            for i in range(normal_priority_users):
                task = priority_user_load(f"normal_user_{i}", RequestPriority.NORMAL, requests_per_user)
                all_tasks.append(task)
            
            # Low priority users
            for i in range(low_priority_users):
                task = priority_user_load(f"low_user_{i}", RequestPriority.LOW, requests_per_user)
                all_tasks.append(task)
            
            # Execute mixed priority load
            start_time = time.time()
            all_results = await asyncio.gather(*all_tasks)
            total_time = time.time() - start_time
            
            # Analyze mixed priority results
            high_priority_results = [r for r in all_results if r["priority"] == "HIGH"]
            normal_priority_results = [r for r in all_results if r["priority"] == "NORMAL"]
            low_priority_results = [r for r in all_results if r["priority"] == "LOW"]
            
            total_requests = len(all_results) * requests_per_user
            total_successful = sum(r["success_count"] for r in all_results)
            
            # Basic assertions
            assert len(all_results) == high_priority_users + normal_priority_users + low_priority_users
            assert total_successful > 0
            
            # Success rate should be reasonable
            success_rate = total_successful / total_requests
            assert success_rate >= 0.6, f"Mixed priority success rate {success_rate:.2%} is too low"
            
            print(f"Mixed priority load: {total_successful}/{total_requests} succeeded "
                  f"({success_rate:.2%}) in {total_time:.2f}s")
            print(f"High priority: {len(high_priority_results)} users, "
                  f"Normal: {len(normal_priority_results)} users, "
                  f"Low: {len(low_priority_results)} users")
            
        finally:
            await component.cleanup()


class TestResourceManagement:
    """Test resource management under concurrent load."""
    
    @pytest.fixture
    def resource_config(self):
        """Create configuration with limited resources."""
        return LightRAGConfig(
            # Limited resources for testing
            max_memory_mb=512,
            max_cpu_percent=70.0,
            max_concurrent_tasks=20,
            max_concurrent_users=15,
            max_requests_per_user=5,
            
            # Performance monitoring
            performance_monitoring_enabled=True,
            performance_monitoring_interval=2,
            auto_optimization_enabled=True,
            optimization_interval=10,
            
            # Caching with limits
            query_cache_size=100,
            query_cache_memory_mb=50,
            embedding_cache_size=500,
            embedding_cache_memory_mb=100
        )
    
    @pytest.mark.asyncio
    async def test_memory_management_under_load(self, resource_config):
        """Test memory management under concurrent load."""
        component = LightRAGComponent(resource_config)
        
        try:
            await component.initialize()
            
            # Get initial memory stats
            initial_perf_stats = await component.get_performance_stats()
            initial_memory = initial_perf_stats["current_metrics"]["memory_usage_mb"]
            
            # Generate memory-intensive load
            num_users = 10
            large_queries_per_user = 5
            
            async def memory_intensive_user(user_id: str) -> Dict[str, Any]:
                """Generate memory-intensive queries."""
                results = []
                
                for i in range(large_queries_per_user):
                    # Create a longer query to use more memory
                    long_query = f"User {user_id} memory test query {i}: " + \
                                "What are the detailed applications of clinical metabolomics in " * 10 + \
                                "personalized medicine and biomarker discovery?"
                    
                    try:
                        result = await component.query(long_query, user_id=user_id)
                        results.append({"success": True, "query_length": len(long_query)})
                    except Exception as e:
                        results.append({"success": False, "error": str(e)})
                    
                    await asyncio.sleep(0.1)
                
                return {
                    "user_id": user_id,
                    "results": results,
                    "success_count": sum(1 for r in results if r["success"])
                }
            
            # Execute memory-intensive load
            user_tasks = [memory_intensive_user(f"memory_user_{i}") for i in range(num_users)]
            user_results = await asyncio.gather(*user_tasks)
            
            # Check memory management
            final_perf_stats = await component.get_performance_stats()
            final_memory = final_perf_stats["current_metrics"]["memory_usage_mb"]
            
            # Trigger manual optimization
            optimization_result = await component.optimize_performance(force=True)
            
            # Check post-optimization memory
            post_opt_perf_stats = await component.get_performance_stats()
            post_opt_memory = post_opt_perf_stats["current_metrics"]["memory_usage_mb"]
            
            # Assertions
            total_successful = sum(r["success_count"] for r in user_results)
            assert total_successful > 0, "No queries succeeded under memory load"
            
            # Memory should be managed (optimization should have been performed)
            assert optimization_result["optimization_performed"] is True
            
            print(f"Memory management test: {total_successful} queries succeeded")
            print(f"Memory usage: Initial={initial_memory:.1f}MB, "
                  f"Peak={final_memory:.1f}MB, Post-opt={post_opt_memory:.1f}MB")
            
        finally:
            await component.cleanup()

    @pytest.mark.asyncio
    async def test_task_management_under_load(self, resource_config):
        """Test task management under concurrent load."""
        component = LightRAGComponent(resource_config)
        
        try:
            await component.initialize()
            
            # Create load that exceeds task limits
            num_concurrent_tasks = 30  # More than max_concurrent_tasks (20)
            
            async def concurrent_task(task_id: int) -> Dict[str, Any]:
                """Execute a concurrent task."""
                start_time = time.time()
                
                try:
                    result = await component.query(
                        f"Concurrent task {task_id}: metabolomics analysis",
                        user_id=f"task_user_{task_id % 5}"
                    )
                    
                    return {
                        "task_id": task_id,
                        "success": True,
                        "execution_time": time.time() - start_time
                    }
                    
                except Exception as e:
                    return {
                        "task_id": task_id,
                        "success": False,
                        "error": str(e),
                        "execution_time": time.time() - start_time
                    }
            
            # Execute concurrent tasks
            start_time = time.time()
            task_results = await asyncio.gather(*[
                concurrent_task(i) for i in range(num_concurrent_tasks)
            ])
            total_time = time.time() - start_time
            
            # Analyze task management
            successful_tasks = [r for r in task_results if r["success"]]
            failed_tasks = [r for r in task_results if not r["success"]]
            
            # Get performance statistics
            perf_stats = await component.get_performance_stats()
            task_stats = perf_stats["task_stats"]
            
            # Assertions
            assert len(task_results) == num_concurrent_tasks
            
            # Some tasks should succeed (system should handle the load)
            success_rate = len(successful_tasks) / num_concurrent_tasks
            assert success_rate > 0.4, f"Task success rate {success_rate:.2%} is too low"
            
            # Should complete in reasonable time
            assert total_time < 90.0, f"Task execution took {total_time:.2f}s"
            
            print(f"Task management test: {len(successful_tasks)}/{num_concurrent_tasks} "
                  f"tasks succeeded ({success_rate:.2%}) in {total_time:.2f}s")
            print(f"Task stats: {task_stats}")
            
        finally:
            await component.cleanup()


if __name__ == "__main__":
    # Run load tests
    pytest.main([
        __file__,
        "-v",
        "-s",  # Show print statements
        "--tb=short"
    ])