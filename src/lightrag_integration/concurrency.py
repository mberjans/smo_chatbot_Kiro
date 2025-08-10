"""
LightRAG Concurrency and Rate Limiting

This module implements request queuing, rate limiting, and concurrent user handling
optimizations to support multiple users efficiently.
Implements requirements 5.2 and 5.4.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import weakref
from enum import Enum
import uuid
import threading

from .utils.logging import setup_logger
from .performance import monitor_performance


class RequestPriority(Enum):
    """Request priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class QueuedRequest:
    """Represents a queued request."""
    request_id: str
    user_id: str
    request_type: str
    priority: RequestPriority
    created_at: datetime
    timeout_seconds: float
    callback: Callable[[], Awaitable[Any]]
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())

    @property
    def is_expired(self) -> bool:
        """Check if request has expired."""
        return (datetime.now() - self.created_at).total_seconds() > self.timeout_seconds

    @property
    def age_seconds(self) -> float:
        """Get request age in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration."""
    requests_per_minute: int
    requests_per_hour: int
    burst_limit: int
    user_type: Optional[str] = None  # None means applies to all users


@dataclass
class UserSession:
    """Tracks user session information."""
    user_id: str
    session_start: datetime
    request_count: int = 0
    last_request: Optional[datetime] = None
    user_type: str = "standard"
    rate_limit_violations: int = 0
    
    def update_request(self):
        """Update session with new request."""
        self.request_count += 1
        self.last_request = datetime.now()


class RateLimiter:
    """
    Token bucket rate limiter with per-user tracking.
    """
    
    def __init__(self, config):
        """Initialize rate limiter."""
        self.config = config
        self.logger = setup_logger("rate_limiter")
        
        # Rate limiting rules
        self.default_rules = RateLimitRule(
            requests_per_minute=getattr(config, 'default_requests_per_minute', 30),
            requests_per_hour=getattr(config, 'default_requests_per_hour', 500),
            burst_limit=getattr(config, 'default_burst_limit', 10)
        )
        
        # User-specific rules
        self.user_rules: Dict[str, RateLimitRule] = {}
        
        # Token buckets per user
        self._user_buckets: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'tokens': self.default_rules.burst_limit,
            'last_refill': time.time(),
            'minute_requests': deque(maxlen=60),  # Track requests per minute
            'hour_requests': deque(maxlen=3600)   # Track requests per hour
        })
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._cleanup_interval = 3600  # 1 hour

    async def start(self):
        """Start the rate limiter."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self.logger.info("Rate limiter started")

    async def stop(self):
        """Stop the rate limiter."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Rate limiter stopped")

    async def check_rate_limit(self, user_id: str, user_type: str = "standard") -> Dict[str, Any]:
        """
        Check if user is within rate limits.
        
        Args:
            user_id: User identifier
            user_type: Type of user (affects rate limits)
        
        Returns:
            Dictionary with rate limit status
        """
        now = time.time()
        bucket = self._user_buckets[user_id]
        
        # Get applicable rules
        rules = self._get_user_rules(user_type)
        
        # Refill tokens based on time elapsed
        time_elapsed = now - bucket['last_refill']
        tokens_to_add = (time_elapsed / 60.0) * (rules.requests_per_minute / 60.0)
        bucket['tokens'] = min(rules.burst_limit, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now
        
        # Check burst limit (token bucket)
        if bucket['tokens'] < 1:
            return {
                'allowed': False,
                'reason': 'burst_limit_exceeded',
                'retry_after_seconds': 60.0 / rules.requests_per_minute,
                'tokens_remaining': bucket['tokens'],
                'limits': {
                    'requests_per_minute': rules.requests_per_minute,
                    'requests_per_hour': rules.requests_per_hour,
                    'burst_limit': rules.burst_limit
                }
            }
        
        # Check minute limit
        minute_requests = self._count_recent_requests(bucket['minute_requests'], 60)
        if minute_requests >= rules.requests_per_minute:
            return {
                'allowed': False,
                'reason': 'minute_limit_exceeded',
                'retry_after_seconds': 60.0,
                'requests_this_minute': minute_requests,
                'limits': {
                    'requests_per_minute': rules.requests_per_minute,
                    'requests_per_hour': rules.requests_per_hour,
                    'burst_limit': rules.burst_limit
                }
            }
        
        # Check hour limit
        hour_requests = self._count_recent_requests(bucket['hour_requests'], 3600)
        if hour_requests >= rules.requests_per_hour:
            return {
                'allowed': False,
                'reason': 'hour_limit_exceeded',
                'retry_after_seconds': 3600.0,
                'requests_this_hour': hour_requests,
                'limits': {
                    'requests_per_minute': rules.requests_per_minute,
                    'requests_per_hour': rules.requests_per_hour,
                    'burst_limit': rules.burst_limit
                }
            }
        
        # Request is allowed
        bucket['tokens'] -= 1
        bucket['minute_requests'].append(now)
        bucket['hour_requests'].append(now)
        
        return {
            'allowed': True,
            'tokens_remaining': bucket['tokens'],
            'requests_this_minute': minute_requests + 1,
            'requests_this_hour': hour_requests + 1,
            'limits': {
                'requests_per_minute': rules.requests_per_minute,
                'requests_per_hour': rules.requests_per_hour,
                'burst_limit': rules.burst_limit
            }
        }

    def _get_user_rules(self, user_type: str) -> RateLimitRule:
        """Get rate limiting rules for user type."""
        # Check for user-type specific rules
        for rule in self.user_rules.values():
            if rule.user_type == user_type:
                return rule
        
        # Return default rules
        return self.default_rules

    def _count_recent_requests(self, request_times: deque, window_seconds: int) -> int:
        """Count requests within the time window."""
        now = time.time()
        cutoff = now - window_seconds
        
        # Remove old requests
        while request_times and request_times[0] < cutoff:
            request_times.popleft()
        
        return len(request_times)

    async def _cleanup_loop(self):
        """Cleanup old user buckets periodically."""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_old_buckets()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {str(e)}")

    async def _cleanup_old_buckets(self):
        """Remove buckets for inactive users."""
        now = time.time()
        cutoff = now - (2 * self._cleanup_interval)  # 2 hours of inactivity
        
        users_to_remove = []
        for user_id, bucket in self._user_buckets.items():
            if bucket['last_refill'] < cutoff:
                users_to_remove.append(user_id)
        
        for user_id in users_to_remove:
            del self._user_buckets[user_id]
        
        if users_to_remove:
            self.logger.info(f"Cleaned up {len(users_to_remove)} inactive user buckets")

    def add_user_rule(self, user_type: str, rule: RateLimitRule):
        """Add custom rate limiting rule for user type."""
        rule.user_type = user_type
        self.user_rules[user_type] = rule
        self.logger.info(f"Added rate limit rule for user type: {user_type}")

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get rate limiting statistics for a user."""
        if user_id not in self._user_buckets:
            return {"error": "User not found"}
        
        bucket = self._user_buckets[user_id]
        now = time.time()
        
        return {
            'user_id': user_id,
            'tokens_remaining': bucket['tokens'],
            'requests_last_minute': self._count_recent_requests(bucket['minute_requests'], 60),
            'requests_last_hour': self._count_recent_requests(bucket['hour_requests'], 3600),
            'last_request': bucket['last_refill'],
            'time_since_last_request': now - bucket['last_refill']
        }


class RequestQueue:
    """
    Priority-based request queue with timeout handling.
    """
    
    def __init__(self, config):
        """Initialize request queue."""
        self.config = config
        self.logger = setup_logger("request_queue")
        
        # Queue configuration
        self.max_queue_size = getattr(config, 'max_queue_size', 1000)
        self.default_timeout = getattr(config, 'default_request_timeout', 300)
        
        # Priority queues
        self._queues: Dict[RequestPriority, asyncio.Queue] = {
            priority: asyncio.Queue(maxsize=self.max_queue_size // 4)
            for priority in RequestPriority
        }
        
        # Request tracking
        self._active_requests: Dict[str, QueuedRequest] = {}
        self._completed_requests: deque = deque(maxlen=1000)
        
        # Statistics
        self._stats = {
            'total_queued': 0,
            'total_processed': 0,
            'total_expired': 0,
            'total_failed': 0,
            'queue_full_rejections': 0
        }
        
        # Processing control
        self._processing = False
        self._processor_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

    async def start(self, num_workers: int = 5):
        """Start request processing."""
        if self._processing:
            return
        
        self._processing = True
        self._shutdown_event.clear()
        
        # Start worker tasks
        for i in range(num_workers):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self._processor_tasks.append(task)
        
        self.logger.info(f"Request queue started with {num_workers} workers")

    async def stop(self):
        """Stop request processing."""
        if not self._processing:
            return
        
        self._processing = False
        self._shutdown_event.set()
        
        # Cancel all worker tasks
        for task in self._processor_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._processor_tasks:
            await asyncio.gather(*self._processor_tasks, return_exceptions=True)
        
        self._processor_tasks.clear()
        self.logger.info("Request queue stopped")

    async def enqueue_request(self, request: QueuedRequest) -> bool:
        """
        Enqueue a request for processing.
        
        Args:
            request: Request to enqueue
        
        Returns:
            True if successfully queued, False otherwise
        """
        try:
            # Check if request is already expired
            if request.is_expired:
                self.logger.warning(f"Request {request.request_id} expired before queuing")
                self._stats['total_expired'] += 1
                return False
            
            # Try to add to appropriate priority queue
            queue = self._queues[request.priority]
            
            try:
                queue.put_nowait(request)
                self._active_requests[request.request_id] = request
                self._stats['total_queued'] += 1
                
                self.logger.debug(
                    f"Queued request {request.request_id} with priority {request.priority.name}"
                )
                return True
                
            except asyncio.QueueFull:
                self.logger.warning(
                    f"Queue full for priority {request.priority.name}, rejecting request {request.request_id}"
                )
                self._stats['queue_full_rejections'] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Error queuing request {request.request_id}: {str(e)}")
            return False

    async def _worker(self, worker_name: str):
        """Worker task that processes requests from queues."""
        self.logger.info(f"Worker {worker_name} started")
        
        while self._processing and not self._shutdown_event.is_set():
            try:
                # Try to get request from highest priority queue first
                request = await self._get_next_request()
                
                if request is None:
                    # No requests available, wait a bit
                    await asyncio.sleep(0.1)
                    continue
                
                # Process the request
                await self._process_request(request, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in worker {worker_name}: {str(e)}")
                await asyncio.sleep(1)  # Brief pause on error
        
        self.logger.info(f"Worker {worker_name} stopped")

    async def _get_next_request(self) -> Optional[QueuedRequest]:
        """Get next request from queues in priority order."""
        # Check queues in priority order (highest first)
        for priority in sorted(RequestPriority, key=lambda p: p.value, reverse=True):
            queue = self._queues[priority]
            
            if not queue.empty():
                try:
                    request = queue.get_nowait()
                    
                    # Check if request has expired
                    if request.is_expired:
                        self.logger.warning(f"Request {request.request_id} expired in queue")
                        self._stats['total_expired'] += 1
                        self._active_requests.pop(request.request_id, None)
                        continue
                    
                    return request
                    
                except asyncio.QueueEmpty:
                    continue
        
        return None

    @monitor_performance("request_processing")
    async def _process_request(self, request: QueuedRequest, worker_name: str):
        """Process a single request."""
        start_time = time.time()
        
        try:
            self.logger.debug(f"Worker {worker_name} processing request {request.request_id}")
            
            # Execute the request callback
            result = await asyncio.wait_for(
                request.callback(),
                timeout=request.timeout_seconds
            )
            
            processing_time = time.time() - start_time
            
            # Record successful completion
            self._completed_requests.append({
                'request_id': request.request_id,
                'user_id': request.user_id,
                'request_type': request.request_type,
                'priority': request.priority.name,
                'processing_time': processing_time,
                'completed_at': datetime.now(),
                'success': True,
                'worker': worker_name
            })
            
            self._stats['total_processed'] += 1
            
            self.logger.debug(
                f"Request {request.request_id} completed in {processing_time:.2f}s by {worker_name}"
            )
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Request {request.request_id} timed out after {request.timeout_seconds}s")
            self._stats['total_expired'] += 1
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Request {request.request_id} failed: {str(e)}")
            
            # Record failed completion
            self._completed_requests.append({
                'request_id': request.request_id,
                'user_id': request.user_id,
                'request_type': request.request_type,
                'priority': request.priority.name,
                'processing_time': processing_time,
                'completed_at': datetime.now(),
                'success': False,
                'error': str(e),
                'worker': worker_name
            })
            
            self._stats['total_failed'] += 1
            
        finally:
            # Clean up request tracking
            self._active_requests.pop(request.request_id, None)

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        queue_sizes = {
            priority.name: queue.qsize()
            for priority, queue in self._queues.items()
        }
        
        return {
            'queue_sizes': queue_sizes,
            'total_queued_size': sum(queue_sizes.values()),
            'active_requests': len(self._active_requests),
            'processing': self._processing,
            'worker_count': len(self._processor_tasks),
            'statistics': self._stats.copy(),
            'recent_completions': len(self._completed_requests)
        }

    def get_user_queue_info(self, user_id: str) -> Dict[str, Any]:
        """Get queue information for a specific user."""
        user_requests = [
            req for req in self._active_requests.values()
            if req.user_id == user_id
        ]
        
        user_completions = [
            comp for comp in self._completed_requests
            if comp['user_id'] == user_id
        ]
        
        return {
            'user_id': user_id,
            'active_requests': len(user_requests),
            'active_request_details': [
                {
                    'request_id': req.request_id,
                    'request_type': req.request_type,
                    'priority': req.priority.name,
                    'age_seconds': req.age_seconds,
                    'timeout_seconds': req.timeout_seconds
                }
                for req in user_requests
            ],
            'recent_completions': len(user_completions),
            'recent_completion_details': user_completions[-5:]  # Last 5 completions
        }


class ConcurrencyManager:
    """
    Main concurrency manager that coordinates rate limiting and request queuing.
    """
    
    def __init__(self, config):
        """Initialize concurrency manager."""
        self.config = config
        self.logger = setup_logger("concurrency_manager")
        
        # Initialize components
        self.rate_limiter = RateLimiter(config)
        self.request_queue = RequestQueue(config)
        
        # User session tracking
        self.user_sessions: Dict[str, UserSession] = {}
        self._session_cleanup_interval = 3600  # 1 hour
        self._session_cleanup_task: Optional[asyncio.Task] = None
        
        # Concurrency limits
        self.max_concurrent_users = getattr(config, 'max_concurrent_users', 100)
        self.max_requests_per_user = getattr(config, 'max_requests_per_user', 5)
        
        # Statistics
        self._stats = {
            'total_requests': 0,
            'rate_limited_requests': 0,
            'queued_requests': 0,
            'rejected_requests': 0,
            'active_users': 0
        }

    async def start(self):
        """Start the concurrency manager."""
        await self.rate_limiter.start()
        await self.request_queue.start(
            num_workers=getattr(self.config, 'queue_workers', 10)
        )
        
        # Start session cleanup
        self._session_cleanup_task = asyncio.create_task(self._session_cleanup_loop())
        
        self.logger.info("Concurrency manager started")

    async def stop(self):
        """Stop the concurrency manager."""
        await self.rate_limiter.stop()
        await self.request_queue.stop()
        
        if self._session_cleanup_task:
            self._session_cleanup_task.cancel()
            try:
                await self._session_cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Concurrency manager stopped")

    async def handle_request(self, 
                           user_id: str,
                           request_type: str,
                           callback: Callable[[], Awaitable[Any]],
                           priority: RequestPriority = RequestPriority.NORMAL,
                           timeout_seconds: Optional[float] = None,
                           user_type: str = "standard",
                           context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an incoming request with rate limiting and queuing.
        
        Args:
            user_id: User identifier
            request_type: Type of request
            callback: Async function to execute
            priority: Request priority
            timeout_seconds: Request timeout
            user_type: Type of user
            context: Additional context
        
        Returns:
            Dictionary with request handling result
        """
        self._stats['total_requests'] += 1
        
        try:
            # Update user session
            self._update_user_session(user_id, user_type)
            
            # Check rate limits
            rate_limit_result = await self.rate_limiter.check_rate_limit(user_id, user_type)
            
            if not rate_limit_result['allowed']:
                self._stats['rate_limited_requests'] += 1
                return {
                    'success': False,
                    'error': 'rate_limit_exceeded',
                    'rate_limit_info': rate_limit_result
                }
            
            # Check concurrent user limits
            if len(self.user_sessions) > self.max_concurrent_users:
                self._stats['rejected_requests'] += 1
                return {
                    'success': False,
                    'error': 'too_many_concurrent_users',
                    'max_concurrent_users': self.max_concurrent_users
                }
            
            # Check per-user request limits
            user_active_requests = len([
                req for req in self.request_queue._active_requests.values()
                if req.user_id == user_id
            ])
            
            if user_active_requests >= self.max_requests_per_user:
                self._stats['rejected_requests'] += 1
                return {
                    'success': False,
                    'error': 'too_many_user_requests',
                    'max_requests_per_user': self.max_requests_per_user,
                    'active_requests': user_active_requests
                }
            
            # Create and queue request
            request = QueuedRequest(
                request_id=str(uuid.uuid4()),
                user_id=user_id,
                request_type=request_type,
                priority=priority,
                created_at=datetime.now(),
                timeout_seconds=timeout_seconds or getattr(self.config, 'default_request_timeout', 300),
                callback=callback,
                context=context or {}
            )
            
            # Try to queue the request
            queued = await self.request_queue.enqueue_request(request)
            
            if not queued:
                self._stats['rejected_requests'] += 1
                return {
                    'success': False,
                    'error': 'queue_full',
                    'queue_stats': self.request_queue.get_queue_stats()
                }
            
            self._stats['queued_requests'] += 1
            
            return {
                'success': True,
                'request_id': request.request_id,
                'queued_at': request.created_at.isoformat(),
                'estimated_wait_time': self._estimate_wait_time(priority),
                'rate_limit_info': rate_limit_result
            }
            
        except Exception as e:
            self.logger.error(f"Error handling request for user {user_id}: {str(e)}")
            return {
                'success': False,
                'error': 'internal_error',
                'message': str(e)
            }

    def _update_user_session(self, user_id: str, user_type: str):
        """Update user session information."""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = UserSession(
                user_id=user_id,
                session_start=datetime.now(),
                user_type=user_type
            )
        
        self.user_sessions[user_id].update_request()
        self._stats['active_users'] = len(self.user_sessions)

    def _estimate_wait_time(self, priority: RequestPriority) -> float:
        """Estimate wait time for a request based on queue state."""
        queue_stats = self.request_queue.get_queue_stats()
        
        # Simple estimation based on queue sizes and priority
        higher_priority_requests = sum(
            queue_stats['queue_sizes'].get(p.name, 0)
            for p in RequestPriority
            if p.value > priority.value
        )
        
        same_priority_requests = queue_stats['queue_sizes'].get(priority.name, 0)
        
        # Assume average processing time of 2 seconds per request
        avg_processing_time = 2.0
        worker_count = queue_stats['worker_count']
        
        # Estimate based on queue position and worker availability
        queue_position = higher_priority_requests + (same_priority_requests / 2)
        estimated_wait = (queue_position / max(1, worker_count)) * avg_processing_time
        
        return max(0.0, estimated_wait)

    async def _session_cleanup_loop(self):
        """Clean up inactive user sessions."""
        while True:
            try:
                await asyncio.sleep(self._session_cleanup_interval)
                await self._cleanup_inactive_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in session cleanup: {str(e)}")

    async def _cleanup_inactive_sessions(self):
        """Remove inactive user sessions."""
        now = datetime.now()
        cutoff = now - timedelta(hours=2)  # 2 hours of inactivity
        
        inactive_users = [
            user_id for user_id, session in self.user_sessions.items()
            if session.last_request and session.last_request < cutoff
        ]
        
        for user_id in inactive_users:
            del self.user_sessions[user_id]
        
        if inactive_users:
            self.logger.info(f"Cleaned up {len(inactive_users)} inactive user sessions")
        
        self._stats['active_users'] = len(self.user_sessions)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive concurrency statistics."""
        return {
            'concurrency_manager': self._stats.copy(),
            'rate_limiter': {
                'active_users': len(self.rate_limiter._user_buckets),
                'rules_count': len(self.rate_limiter.user_rules)
            },
            'request_queue': self.request_queue.get_queue_stats(),
            'user_sessions': {
                'active_sessions': len(self.user_sessions),
                'session_details': [
                    {
                        'user_id': session.user_id,
                        'user_type': session.user_type,
                        'request_count': session.request_count,
                        'session_duration_minutes': (
                            (datetime.now() - session.session_start).total_seconds() / 60
                        ),
                        'last_request_minutes_ago': (
                            (datetime.now() - session.last_request).total_seconds() / 60
                            if session.last_request else None
                        )
                    }
                    for session in list(self.user_sessions.values())[:10]  # Top 10 sessions
                ]
            }
        }

    def get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive information for a specific user."""
        session_info = {}
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            session_info = {
                'user_type': session.user_type,
                'session_start': session.session_start.isoformat(),
                'request_count': session.request_count,
                'last_request': session.last_request.isoformat() if session.last_request else None,
                'rate_limit_violations': session.rate_limit_violations
            }
        
        return {
            'user_id': user_id,
            'session_info': session_info,
            'rate_limit_stats': self.rate_limiter.get_user_stats(user_id),
            'queue_info': self.request_queue.get_user_queue_info(user_id)
        }