"""
Tests for LightRAG Concurrency and Rate Limiting

This module tests the concurrency management, rate limiting, and request queuing
systems to ensure they handle multiple users efficiently.
"""

import asyncio
import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from .concurrency import (
    RateLimiter, RequestQueue, ConcurrencyManager,
    QueuedRequest, RequestPriority, RateLimitRule, UserSession
)
from .config.settings import LightRAGConfig


class TestRateLimiter:
    """Test rate limiting functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            default_requests_per_minute=10,
            default_requests_per_hour=100,
            default_burst_limit=5
        )
    
    @pytest.fixture
    def rate_limiter(self, config):
        """Create rate limiter."""
        return RateLimiter(config)
    
    @pytest.mark.asyncio
    async def test_rate_limit_basic(self, rate_limiter):
        """Test basic rate limiting."""
        await rate_limiter.start()
        
        try:
            user_id = "test_user"
            
            # Check the actual burst limit from the rate limiter
            rules = rate_limiter._get_user_rules("standard")
            burst_limit = rules.burst_limit
            
            # Make requests up to the burst limit
            allowed_requests = 0
            for i in range(burst_limit + 2):  # Try more than the limit
                result = await rate_limiter.check_rate_limit(user_id)
                if result['allowed']:
                    allowed_requests += 1
                else:
                    # Should eventually be denied
                    assert result['reason'] == 'burst_limit_exceeded'
                    break
            
            # Should have been limited before exceeding burst limit
            assert allowed_requests <= burst_limit
            
        finally:
            await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_rate_limit_minute_window(self, rate_limiter):
        """Test minute-based rate limiting."""
        await rate_limiter.start()
        
        try:
            user_id = "test_user"
            
            # Simulate requests over time to test minute window
            # This is a simplified test - in reality we'd need to wait or mock time
            
            # Make requests up to the minute limit
            allowed_count = 0
            for i in range(15):  # Try more than the limit
                result = await rate_limiter.check_rate_limit(user_id)
                if result['allowed']:
                    allowed_count += 1
                else:
                    break
            
            # Should have been limited before reaching 15
            assert allowed_count <= 10  # Minute limit is 10
            
        finally:
            await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_rate_limit_user_types(self, rate_limiter):
        """Test different rate limits for different user types."""
        await rate_limiter.start()
        
        try:
            # Add premium user rule
            premium_rule = RateLimitRule(
                requests_per_minute=20,
                requests_per_hour=500,
                burst_limit=10
            )
            rate_limiter.add_user_rule("premium", premium_rule)
            
            # Test standard user
            standard_result = await rate_limiter.check_rate_limit("standard_user", "standard")
            assert standard_result['limits']['burst_limit'] == 5
            
            # Test premium user
            premium_result = await rate_limiter.check_rate_limit("premium_user", "premium")
            assert premium_result['limits']['burst_limit'] == 10
            
        finally:
            await rate_limiter.stop()

    @pytest.mark.asyncio
    async def test_rate_limit_stats(self, rate_limiter):
        """Test rate limiting statistics."""
        await rate_limiter.start()
        
        try:
            user_id = "test_user"
            
            # Make some requests
            for _ in range(3):
                await rate_limiter.check_rate_limit(user_id)
            
            # Check stats
            stats = rate_limiter.get_user_stats(user_id)
            assert stats['user_id'] == user_id
            assert 'tokens_remaining' in stats
            assert 'requests_last_minute' in stats
            
        finally:
            await rate_limiter.stop()


class TestRequestQueue:
    """Test request queue functionality."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            max_queue_size=100,
            default_request_timeout=10
        )
    
    @pytest.fixture
    def request_queue(self, config):
        """Create request queue."""
        return RequestQueue(config)
    
    @pytest.mark.asyncio
    async def test_queue_basic_operations(self, request_queue):
        """Test basic queue operations."""
        await request_queue.start(num_workers=2)
        
        try:
            # Create test request
            async def test_callback():
                await asyncio.sleep(0.1)
                return "test_result"
            
            request = QueuedRequest(
                request_id="test_request",
                user_id="test_user",
                request_type="test",
                priority=RequestPriority.NORMAL,
                created_at=datetime.now(),
                timeout_seconds=5.0,
                callback=test_callback
            )
            
            # Queue the request
            success = await request_queue.enqueue_request(request)
            assert success is True
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Check stats
            stats = request_queue.get_queue_stats()
            assert stats['statistics']['total_processed'] >= 1
            
        finally:
            await request_queue.stop()

    @pytest.mark.asyncio
    async def test_queue_priority_ordering(self, request_queue):
        """Test priority-based request ordering."""
        await request_queue.start(num_workers=1)  # Single worker for predictable ordering
        
        try:
            results = []
            
            async def make_callback(result_value):
                async def callback():
                    await asyncio.sleep(0.1)
                    results.append(result_value)
                    return result_value
                return callback
            
            # Create requests with different priorities
            requests = [
                QueuedRequest(
                    request_id=f"request_{i}",
                    user_id="test_user",
                    request_type="test",
                    priority=priority,
                    created_at=datetime.now(),
                    timeout_seconds=5.0,
                    callback=await make_callback(f"result_{priority.name}")
                )
                for i, priority in enumerate([
                    RequestPriority.LOW,
                    RequestPriority.CRITICAL,
                    RequestPriority.NORMAL,
                    RequestPriority.HIGH
                ])
            ]
            
            # Queue all requests
            for request in requests:
                await request_queue.enqueue_request(request)
            
            # Wait for processing
            await asyncio.sleep(1.0)
            
            # Critical should be processed first, then HIGH, NORMAL, LOW
            assert len(results) >= 2  # At least some should be processed
            if len(results) >= 2:
                # First result should be from CRITICAL priority
                assert "CRITICAL" in results[0]
            
        finally:
            await request_queue.stop()

    @pytest.mark.asyncio
    async def test_queue_timeout_handling(self, request_queue):
        """Test request timeout handling."""
        await request_queue.start(num_workers=1)
        
        try:
            async def timeout_callback():
                await asyncio.sleep(2.0)  # Longer than timeout
                return "should_not_complete"
            
            request = QueuedRequest(
                request_id="timeout_request",
                user_id="test_user",
                request_type="test",
                priority=RequestPriority.NORMAL,
                created_at=datetime.now(),
                timeout_seconds=0.5,  # Short timeout
                callback=timeout_callback
            )
            
            # Queue the request
            await request_queue.enqueue_request(request)
            
            # Wait for timeout
            await asyncio.sleep(1.0)
            
            # Check that request was marked as expired
            stats = request_queue.get_queue_stats()
            assert stats['statistics']['total_expired'] >= 1
            
        finally:
            await request_queue.stop()

    @pytest.mark.asyncio
    async def test_queue_user_info(self, request_queue):
        """Test user-specific queue information."""
        await request_queue.start(num_workers=1)
        
        try:
            async def slow_callback():
                await asyncio.sleep(0.5)
                return "result"
            
            # Create requests for specific user
            user_id = "test_user"
            requests = []
            
            for i in range(3):
                request = QueuedRequest(
                    request_id=f"user_request_{i}",
                    user_id=user_id,
                    request_type="test",
                    priority=RequestPriority.NORMAL,
                    created_at=datetime.now(),
                    timeout_seconds=5.0,
                    callback=slow_callback
                )
                requests.append(request)
                await request_queue.enqueue_request(request)
            
            # Get user info
            user_info = request_queue.get_user_queue_info(user_id)
            assert user_info['user_id'] == user_id
            assert user_info['active_requests'] >= 1  # At least one should be active
            
            # Wait for completion
            await asyncio.sleep(2.0)
            
            # Check completion info
            user_info = request_queue.get_user_queue_info(user_id)
            assert user_info['recent_completions'] >= 1
            
        finally:
            await request_queue.stop()


class TestConcurrencyManager:
    """Test concurrency manager integration."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return LightRAGConfig(
            default_requests_per_minute=20,
            default_burst_limit=10,
            max_concurrent_users=50,
            max_requests_per_user=3,
            queue_workers=2
        )
    
    @pytest.fixture
    def concurrency_manager(self, config):
        """Create concurrency manager."""
        return ConcurrencyManager(config)
    
    @pytest.mark.asyncio
    async def test_concurrency_manager_basic(self, concurrency_manager):
        """Test basic concurrency manager functionality."""
        await concurrency_manager.start()
        
        try:
            async def test_callback():
                await asyncio.sleep(0.1)
                return "test_result"
            
            # Handle a request
            result = await concurrency_manager.handle_request(
                user_id="test_user",
                request_type="test",
                callback=test_callback
            )
            
            assert result['success'] is True
            assert 'request_id' in result
            
            # Wait for processing
            await asyncio.sleep(0.5)
            
            # Check stats
            stats = concurrency_manager.get_comprehensive_stats()
            assert stats['concurrency_manager']['total_requests'] >= 1
            
        finally:
            await concurrency_manager.stop()

    @pytest.mark.asyncio
    async def test_rate_limiting_integration(self, concurrency_manager):
        """Test rate limiting integration."""
        await concurrency_manager.start()
        
        try:
            async def quick_callback():
                return "result"
            
            user_id = "rate_test_user"
            
            # Make requests up to burst limit
            successful_requests = 0
            for i in range(15):  # More than burst limit
                result = await concurrency_manager.handle_request(
                    user_id=user_id,
                    request_type="test",
                    callback=quick_callback
                )
                
                if result['success']:
                    successful_requests += 1
                else:
                    # Should eventually hit rate limit
                    assert result['error'] == 'rate_limit_exceeded'
                    break
            
            # Should have been rate limited before 15 requests
            assert successful_requests < 15
            
        finally:
            await concurrency_manager.stop()

    @pytest.mark.asyncio
    async def test_concurrent_user_limits(self, concurrency_manager):
        """Test concurrent user limits."""
        # This test would need to be more complex to actually test the limit
        # For now, just test that the limit checking works
        
        await concurrency_manager.start()
        
        try:
            # Simulate many users by creating sessions
            for i in range(5):
                concurrency_manager._update_user_session(f"user_{i}", "standard")
            
            async def test_callback():
                return "result"
            
            # Should still accept requests from existing users
            result = await concurrency_manager.handle_request(
                user_id="user_1",
                request_type="test",
                callback=test_callback
            )
            
            assert result['success'] is True
            
        finally:
            await concurrency_manager.stop()

    @pytest.mark.asyncio
    async def test_per_user_request_limits(self, concurrency_manager):
        """Test per-user request limits."""
        await concurrency_manager.start()
        
        try:
            async def slow_callback():
                await asyncio.sleep(1.0)  # Slow to keep requests active
                return "result"
            
            user_id = "limit_test_user"
            
            # Make requests up to per-user limit
            successful_requests = 0
            for i in range(5):  # More than per-user limit (3)
                result = await concurrency_manager.handle_request(
                    user_id=user_id,
                    request_type="test",
                    callback=slow_callback
                )
                
                if result['success']:
                    successful_requests += 1
                else:
                    assert result['error'] == 'too_many_user_requests'
                    break
            
            # Should have been limited at 3 requests
            assert successful_requests <= 3
            
        finally:
            await concurrency_manager.stop()

    @pytest.mark.asyncio
    async def test_user_info_tracking(self, concurrency_manager):
        """Test user information tracking."""
        await concurrency_manager.start()
        
        try:
            async def test_callback():
                await asyncio.sleep(0.1)
                return "result"
            
            user_id = "info_test_user"
            
            # Make a request to create user session
            await concurrency_manager.handle_request(
                user_id=user_id,
                request_type="test",
                callback=test_callback,
                user_type="premium"
            )
            
            # Get user info
            user_info = concurrency_manager.get_user_info(user_id)
            assert user_info['user_id'] == user_id
            assert 'session_info' in user_info
            assert 'rate_limit_stats' in user_info
            assert 'queue_info' in user_info
            
            # Check session info
            if user_info['session_info']:
                assert user_info['session_info']['user_type'] == 'premium'
            
        finally:
            await concurrency_manager.stop()

    @pytest.mark.asyncio
    async def test_comprehensive_stats(self, concurrency_manager):
        """Test comprehensive statistics collection."""
        await concurrency_manager.start()
        
        try:
            async def test_callback():
                return "result"
            
            # Make some requests
            for i in range(3):
                await concurrency_manager.handle_request(
                    user_id=f"stats_user_{i}",
                    request_type="test",
                    callback=test_callback
                )
            
            # Get comprehensive stats
            stats = concurrency_manager.get_comprehensive_stats()
            
            required_sections = [
                'concurrency_manager',
                'rate_limiter',
                'request_queue',
                'user_sessions'
            ]
            
            for section in required_sections:
                assert section in stats
            
            # Check that we have some data
            assert stats['concurrency_manager']['total_requests'] >= 3
            assert stats['user_sessions']['active_sessions'] >= 1
            
        finally:
            await concurrency_manager.stop()


class TestConcurrencyLoad:
    """Load testing for concurrency system."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration for load testing."""
        return LightRAGConfig(
            default_requests_per_minute=100,
            default_burst_limit=50,
            max_concurrent_users=20,
            max_requests_per_user=10,
            queue_workers=5,
            max_queue_size=200
        )
    
    @pytest.fixture
    def concurrency_manager(self, config):
        """Create concurrency manager for load testing."""
        return ConcurrencyManager(config)
    
    @pytest.mark.asyncio
    async def test_concurrent_users_load(self, concurrency_manager):
        """Test handling multiple concurrent users."""
        await concurrency_manager.start()
        
        try:
            async def user_simulation(user_id: str, num_requests: int):
                """Simulate a user making multiple requests."""
                results = []
                
                for i in range(num_requests):
                    async def request_callback():
                        await asyncio.sleep(0.1)  # Simulate work
                        return f"result_{user_id}_{i}"
                    
                    result = await concurrency_manager.handle_request(
                        user_id=user_id,
                        request_type="load_test",
                        callback=request_callback
                    )
                    
                    results.append(result['success'])
                    
                    # Small delay between requests
                    await asyncio.sleep(0.05)
                
                return results
            
            # Simulate 10 concurrent users, each making 5 requests
            user_tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    user_simulation(f"load_user_{i}", 5)
                )
                user_tasks.append(task)
            
            # Wait for all users to complete
            user_results = await asyncio.gather(*user_tasks)
            
            # Check results
            total_requests = sum(len(results) for results in user_results)
            successful_requests = sum(
                sum(1 for success in results if success)
                for results in user_results
            )
            
            # Most requests should succeed (allowing for some rate limiting)
            success_rate = successful_requests / total_requests
            assert success_rate > 0.7  # At least 70% success rate
            
            # Check final stats
            stats = concurrency_manager.get_comprehensive_stats()
            assert stats['concurrency_manager']['total_requests'] == total_requests
            
        finally:
            await concurrency_manager.stop()

    @pytest.mark.asyncio
    async def test_queue_performance(self, concurrency_manager):
        """Test queue performance under load."""
        await concurrency_manager.start()
        
        try:
            start_time = time.time()
            
            # Create many quick requests
            tasks = []
            for i in range(50):
                async def quick_callback():
                    return f"result_{i}"
                
                task = asyncio.create_task(
                    concurrency_manager.handle_request(
                        user_id=f"perf_user_{i % 5}",  # 5 users sharing the load
                        request_type="performance_test",
                        callback=quick_callback,
                        priority=RequestPriority.HIGH
                    )
                )
                tasks.append(task)
            
            # Wait for all requests to be queued
            results = await asyncio.gather(*tasks)
            
            # Wait for processing to complete
            await asyncio.sleep(2.0)
            
            processing_time = time.time() - start_time
            
            # Check that most requests were accepted
            successful_queues = sum(1 for result in results if result['success'])
            assert successful_queues >= 40  # At least 80% should be queued
            
            # Processing should be reasonably fast
            assert processing_time < 10.0  # Should complete within 10 seconds
            
            # Check queue stats
            queue_stats = concurrency_manager.request_queue.get_queue_stats()
            assert queue_stats['statistics']['total_processed'] >= 40
            
        finally:
            await concurrency_manager.stop()


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",
        "-k", "test_rate_limit or test_queue or test_concurrency",
        "--tb=short"
    ])