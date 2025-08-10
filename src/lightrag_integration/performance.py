"""
LightRAG Performance Optimization

This module implements performance optimization features including
async processing, resource management, and performance monitoring.
Implements requirements 5.3 and 5.6.
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import weakref
import gc
from contextlib import asynccontextmanager
import functools

from .utils.logging import setup_logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_io_sent_mb: float = 0.0
    network_io_recv_mb: float = 0.0
    active_threads: int = 0
    active_tasks: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_mb': self.memory_usage_mb,
            'memory_usage_percent': self.memory_usage_percent,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb,
            'network_io_sent_mb': self.network_io_sent_mb,
            'network_io_recv_mb': self.network_io_recv_mb,
            'active_threads': self.active_threads,
            'active_tasks': self.active_tasks,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ResourceLimits:
    """Resource usage limits for optimization."""
    max_memory_mb: Optional[int] = None
    max_cpu_percent: Optional[float] = None
    max_concurrent_tasks: Optional[int] = None
    max_disk_io_mb_per_sec: Optional[float] = None
    max_network_io_mb_per_sec: Optional[float] = None


class AsyncTaskManager:
    """
    Manages async tasks with resource limits and performance optimization.
    """
    
    def __init__(self, config):
        """Initialize task manager."""
        self.config = config
        self.logger = setup_logger("task_manager")
        
        # Task management
        self.max_concurrent_tasks = getattr(config, 'max_concurrent_tasks', 50)
        self.task_timeout = getattr(config, 'task_timeout_seconds', 300)
        
        # Active tasks tracking
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        self._task_counter = 0
        self._lock = asyncio.Lock()
        
        # Performance tracking
        self._task_metrics = defaultdict(list)
        self._completed_tasks = 0
        self._failed_tasks = 0
        
        # Resource monitoring
        self.resource_limits = ResourceLimits(
            max_memory_mb=getattr(config, 'max_memory_mb', 2048),
            max_cpu_percent=getattr(config, 'max_cpu_percent', 80.0),
            max_concurrent_tasks=self.max_concurrent_tasks
        )

    async def execute_task(self, 
                          coro: Awaitable[Any], 
                          task_name: Optional[str] = None,
                          priority: int = 0,
                          timeout: Optional[float] = None) -> Any:
        """
        Execute an async task with resource management.
        
        Args:
            coro: Coroutine to execute
            task_name: Optional task name for tracking
            priority: Task priority (higher = more important)
            timeout: Task timeout in seconds
        
        Returns:
            Task result
        """
        # Generate task ID
        async with self._lock:
            self._task_counter += 1
            task_id = f"{task_name or 'task'}_{self._task_counter}"
        
        # Check resource limits before starting
        if not await self._check_resource_limits():
            raise RuntimeError("Resource limits exceeded, cannot start new task")
        
        # Acquire semaphore to limit concurrent tasks
        async with self._task_semaphore:
            start_time = time.time()
            
            try:
                # Create and track task
                task = asyncio.create_task(coro, name=task_id)
                
                async with self._lock:
                    self._active_tasks[task_id] = task
                
                # Execute with timeout
                timeout_value = timeout or self.task_timeout
                result = await asyncio.wait_for(task, timeout=timeout_value)
                
                # Record success metrics
                execution_time = time.time() - start_time
                self._record_task_completion(task_id, execution_time, True)
                
                return result
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Task {task_id} timed out after {timeout_value}s")
                self._record_task_completion(task_id, time.time() - start_time, False)
                raise
                
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {str(e)}")
                self._record_task_completion(task_id, time.time() - start_time, False)
                raise
                
            finally:
                # Clean up task tracking
                async with self._lock:
                    self._active_tasks.pop(task_id, None)

    async def execute_batch(self, 
                           coros: List[Awaitable[Any]], 
                           batch_size: Optional[int] = None,
                           return_exceptions: bool = True) -> List[Any]:
        """
        Execute multiple tasks in batches.
        
        Args:
            coros: List of coroutines to execute
            batch_size: Maximum batch size (defaults to max_concurrent_tasks)
            return_exceptions: Whether to return exceptions instead of raising
        
        Returns:
            List of results
        """
        if not coros:
            return []
        
        batch_size = batch_size or self.max_concurrent_tasks
        results = []
        
        # Process in batches
        for i in range(0, len(coros), batch_size):
            batch = coros[i:i + batch_size]
            
            # Execute batch
            batch_tasks = [
                self.execute_task(coro, f"batch_{i//batch_size}_{j}")
                for j, coro in enumerate(batch)
            ]
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=return_exceptions)
            results.extend(batch_results)
            
            # Brief pause between batches to allow resource recovery
            if i + batch_size < len(coros):
                await asyncio.sleep(0.1)
        
        return results

    async def _check_resource_limits(self) -> bool:
        """Check if resource limits allow starting new tasks."""
        try:
            # Check memory usage
            if self.resource_limits.max_memory_mb:
                memory_usage = psutil.virtual_memory().used / (1024 * 1024)
                if memory_usage > self.resource_limits.max_memory_mb:
                    self.logger.warning(f"Memory usage {memory_usage:.1f}MB exceeds limit {self.resource_limits.max_memory_mb}MB")
                    return False
            
            # Check CPU usage
            if self.resource_limits.max_cpu_percent:
                cpu_usage = psutil.cpu_percent(interval=0.1)
                if cpu_usage > self.resource_limits.max_cpu_percent:
                    self.logger.warning(f"CPU usage {cpu_usage:.1f}% exceeds limit {self.resource_limits.max_cpu_percent}%")
                    return False
            
            # Check concurrent tasks
            if len(self._active_tasks) >= self.max_concurrent_tasks:
                self.logger.warning(f"Active tasks {len(self._active_tasks)} at maximum limit {self.max_concurrent_tasks}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking resource limits: {str(e)}")
            return True  # Allow task to proceed if check fails

    def _record_task_completion(self, task_id: str, execution_time: float, success: bool) -> None:
        """Record task completion metrics."""
        self._task_metrics[task_id].append({
            'execution_time': execution_time,
            'success': success,
            'timestamp': datetime.now()
        })
        
        if success:
            self._completed_tasks += 1
        else:
            self._failed_tasks += 1

    def get_task_stats(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        active_count = len(self._active_tasks)
        
        # Calculate average execution time
        all_times = []
        for metrics in self._task_metrics.values():
            all_times.extend([m['execution_time'] for m in metrics if m['success']])
        
        avg_execution_time = sum(all_times) / len(all_times) if all_times else 0.0
        
        return {
            'active_tasks': active_count,
            'completed_tasks': self._completed_tasks,
            'failed_tasks': self._failed_tasks,
            'total_tasks': self._completed_tasks + self._failed_tasks,
            'success_rate': self._completed_tasks / max(1, self._completed_tasks + self._failed_tasks),
            'average_execution_time': avg_execution_time,
            'max_concurrent_tasks': self.max_concurrent_tasks
        }

    async def cancel_all_tasks(self) -> None:
        """Cancel all active tasks."""
        async with self._lock:
            tasks_to_cancel = list(self._active_tasks.values())
        
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
        
        self.logger.info(f"Cancelled {len(tasks_to_cancel)} active tasks")


class MemoryManager:
    """
    Manages memory usage and performs optimization.
    """
    
    def __init__(self, config):
        """Initialize memory manager."""
        self.config = config
        self.logger = setup_logger("memory_manager")
        
        # Memory limits
        self.max_memory_mb = getattr(config, 'max_memory_mb', 2048)
        self.memory_warning_threshold = getattr(config, 'memory_warning_threshold', 0.8)
        self.memory_critical_threshold = getattr(config, 'memory_critical_threshold', 0.9)
        
        # Garbage collection settings
        self.gc_threshold_mb = getattr(config, 'gc_threshold_mb', 100)
        self.auto_gc_enabled = getattr(config, 'auto_gc_enabled', True)
        
        # Memory tracking
        self._memory_history = deque(maxlen=100)
        self._last_gc_time = datetime.now()
        self._gc_count = 0

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        return {
            'process_memory_mb': memory_info.rss / (1024 * 1024),
            'process_memory_percent': process.memory_percent(),
            'system_memory_mb': system_memory.used / (1024 * 1024),
            'system_memory_percent': system_memory.percent,
            'system_available_mb': system_memory.available / (1024 * 1024)
        }

    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check if system is under memory pressure."""
        memory_usage = self.get_memory_usage()
        
        # Check process memory
        process_pressure = memory_usage['process_memory_mb'] > self.max_memory_mb * self.memory_warning_threshold
        
        # Check system memory
        system_pressure = memory_usage['system_memory_percent'] > (self.memory_warning_threshold * 100)
        
        # Determine pressure level
        pressure_level = "normal"
        if memory_usage['process_memory_mb'] > self.max_memory_mb * self.memory_critical_threshold:
            pressure_level = "critical"
        elif process_pressure or system_pressure:
            pressure_level = "warning"
        
        return {
            'pressure_level': pressure_level,
            'process_pressure': process_pressure,
            'system_pressure': system_pressure,
            'memory_usage': memory_usage,
            'recommendations': self._get_memory_recommendations(pressure_level, memory_usage)
        }

    def _get_memory_recommendations(self, pressure_level: str, memory_usage: Dict[str, float]) -> List[str]:
        """Get memory optimization recommendations."""
        recommendations = []
        
        if pressure_level == "critical":
            recommendations.extend([
                "Immediately run garbage collection",
                "Clear all non-essential caches",
                "Reduce concurrent task limits",
                "Consider restarting the process"
            ])
        elif pressure_level == "warning":
            recommendations.extend([
                "Run garbage collection",
                "Clear old cache entries",
                "Reduce batch sizes for processing"
            ])
        
        return recommendations

    async def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """Perform memory optimization."""
        start_memory = self.get_memory_usage()
        
        # Check if optimization is needed
        pressure_info = self.check_memory_pressure()
        
        if not force and pressure_info['pressure_level'] == "normal":
            return {
                'optimization_performed': False,
                'reason': 'Memory pressure is normal',
                'memory_before': start_memory,
                'memory_after': start_memory
            }
        
        self.logger.info(f"Starting memory optimization (pressure: {pressure_info['pressure_level']})")
        
        # Perform garbage collection
        if self.auto_gc_enabled:
            collected = gc.collect()
            self._gc_count += 1
            self._last_gc_time = datetime.now()
            self.logger.debug(f"Garbage collection freed {collected} objects")
        
        # Brief pause to allow memory to be freed
        await asyncio.sleep(0.1)
        
        end_memory = self.get_memory_usage()
        memory_freed = start_memory['process_memory_mb'] - end_memory['process_memory_mb']
        
        self.logger.info(f"Memory optimization completed, freed {memory_freed:.1f}MB")
        
        return {
            'optimization_performed': True,
            'memory_freed_mb': memory_freed,
            'memory_before': start_memory,
            'memory_after': end_memory,
            'gc_objects_collected': collected if self.auto_gc_enabled else 0,
            'pressure_level_before': pressure_info['pressure_level'],
            'pressure_level_after': self.check_memory_pressure()['pressure_level']
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current_usage = self.get_memory_usage()
        pressure_info = self.check_memory_pressure()
        
        return {
            'current_usage': current_usage,
            'pressure_info': pressure_info,
            'limits': {
                'max_memory_mb': self.max_memory_mb,
                'warning_threshold': self.memory_warning_threshold,
                'critical_threshold': self.memory_critical_threshold
            },
            'gc_stats': {
                'auto_gc_enabled': self.auto_gc_enabled,
                'gc_count': self._gc_count,
                'last_gc_time': self._last_gc_time.isoformat(),
                'gc_threshold_mb': self.gc_threshold_mb
            }
        }


class PerformanceOptimizer:
    """
    Main performance optimization coordinator.
    """
    
    def __init__(self, config):
        """Initialize performance optimizer."""
        self.config = config
        self.logger = setup_logger("performance_optimizer")
        
        # Initialize components
        self.task_manager = AsyncTaskManager(config)
        self.memory_manager = MemoryManager(config)
        
        # Performance monitoring
        self.monitoring_enabled = getattr(config, 'performance_monitoring_enabled', True)
        self.monitoring_interval = getattr(config, 'performance_monitoring_interval', 30)
        self._monitoring_task: Optional[asyncio.Task] = None
        self._performance_history = deque(maxlen=1000)
        
        # Optimization settings
        self.auto_optimization_enabled = getattr(config, 'auto_optimization_enabled', True)
        self.optimization_interval = getattr(config, 'optimization_interval', 300)  # 5 minutes
        self._last_optimization = datetime.now()

    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if self.monitoring_enabled and not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self.logger.info("Performance monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
            self.logger.info("Performance monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while True:
            try:
                # Collect performance metrics
                metrics = await self._collect_performance_metrics()
                self._performance_history.append(metrics)
                
                # Check if optimization is needed
                if self.auto_optimization_enabled:
                    await self._check_and_optimize()
                
                # Wait for next monitoring cycle
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(self.monitoring_interval)

    async def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            # I/O metrics
            disk_io = psutil.disk_io_counters()
            net_io = psutil.net_io_counters()
            
            # Task metrics
            task_stats = self.task_manager.get_task_stats()
            
            return PerformanceMetrics(
                cpu_usage_percent=cpu_percent,
                memory_usage_mb=process_memory.rss / (1024 * 1024),
                memory_usage_percent=memory.percent,
                disk_io_read_mb=(disk_io.read_bytes if disk_io else 0) / (1024 * 1024),
                disk_io_write_mb=(disk_io.write_bytes if disk_io else 0) / (1024 * 1024),
                network_io_sent_mb=(net_io.bytes_sent if net_io else 0) / (1024 * 1024),
                network_io_recv_mb=(net_io.bytes_recv if net_io else 0) / (1024 * 1024),
                active_threads=threading.active_count(),
                active_tasks=task_stats['active_tasks']
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {str(e)}")
            return PerformanceMetrics()

    async def _check_and_optimize(self) -> None:
        """Check if optimization is needed and perform it."""
        now = datetime.now()
        
        # Check if enough time has passed since last optimization
        if (now - self._last_optimization).total_seconds() < self.optimization_interval:
            return
        
        # Check memory pressure
        memory_pressure = self.memory_manager.check_memory_pressure()
        
        # Perform optimization if needed
        if memory_pressure['pressure_level'] in ['warning', 'critical']:
            await self.optimize_performance()

    async def optimize_performance(self, force: bool = False) -> Dict[str, Any]:
        """Perform comprehensive performance optimization."""
        self.logger.info("Starting performance optimization")
        start_time = time.time()
        
        optimization_results = {}
        
        # Memory optimization
        memory_result = await self.memory_manager.optimize_memory(force)
        optimization_results['memory'] = memory_result
        
        # Task management optimization
        task_stats_before = self.task_manager.get_task_stats()
        
        # If too many failed tasks, restart task manager
        if task_stats_before['total_tasks'] > 0:
            failure_rate = task_stats_before['failed_tasks'] / task_stats_before['total_tasks']
            if failure_rate > 0.2:  # More than 20% failure rate
                await self.task_manager.cancel_all_tasks()
                optimization_results['tasks'] = {
                    'cancelled_tasks': task_stats_before['active_tasks'],
                    'reason': 'High failure rate detected'
                }
        
        # Update last optimization time
        self._last_optimization = datetime.now()
        
        optimization_time = time.time() - start_time
        
        self.logger.info(f"Performance optimization completed in {optimization_time:.2f}s")
        
        return {
            'optimization_time': optimization_time,
            'results': optimization_results,
            'timestamp': self._last_optimization.isoformat()
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        # Get latest metrics
        latest_metrics = self._performance_history[-1] if self._performance_history else PerformanceMetrics()
        
        # Calculate averages over recent history
        recent_metrics = list(self._performance_history)[-10:]  # Last 10 measurements
        
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        avg_memory = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
        
        return {
            'current_metrics': latest_metrics.to_dict(),
            'averages': {
                'cpu_usage_percent': avg_cpu,
                'memory_usage_mb': avg_memory
            },
            'task_stats': self.task_manager.get_task_stats(),
            'memory_stats': self.memory_manager.get_memory_stats(),
            'monitoring': {
                'enabled': self.monitoring_enabled,
                'interval_seconds': self.monitoring_interval,
                'history_size': len(self._performance_history)
            },
            'optimization': {
                'auto_enabled': self.auto_optimization_enabled,
                'last_optimization': self._last_optimization.isoformat(),
                'interval_seconds': self.optimization_interval
            }
        }

    async def cleanup(self) -> None:
        """Clean up performance optimizer resources."""
        await self.stop_monitoring()
        await self.task_manager.cancel_all_tasks()
        self.logger.info("Performance optimizer cleanup completed")


# Decorator for performance monitoring
def monitor_performance(func_name: Optional[str] = None):
    """Decorator to monitor function performance."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log performance if it's slow
                if execution_time > 1.0:  # Log if takes more than 1 second
                    logger = setup_logger("performance_monitor")
                    logger.info(f"Function {name} took {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger = setup_logger("performance_monitor")
                logger.error(f"Function {name} failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            name = func_name or f"{func.__module__}.{func.__name__}"
            
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Log performance if it's slow
                if execution_time > 1.0:  # Log if takes more than 1 second
                    logger = setup_logger("performance_monitor")
                    logger.info(f"Function {name} took {execution_time:.2f}s")
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                logger = setup_logger("performance_monitor")
                logger.error(f"Function {name} failed after {execution_time:.2f}s: {str(e)}")
                raise
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Context manager for resource management
@asynccontextmanager
async def managed_resources(optimizer: PerformanceOptimizer, resource_name: str):
    """Context manager for automatic resource management."""
    logger = setup_logger("resource_manager")
    
    try:
        logger.debug(f"Acquiring resources for {resource_name}")
        
        # Check resource availability before starting
        memory_pressure = optimizer.memory_manager.check_memory_pressure()
        if memory_pressure['pressure_level'] == 'critical':
            await optimizer.memory_manager.optimize_memory()
        
        yield
        
    except Exception as e:
        logger.error(f"Error in resource management for {resource_name}: {str(e)}")
        raise
        
    finally:
        logger.debug(f"Releasing resources for {resource_name}")
        
        # Perform cleanup if needed
        memory_pressure = optimizer.memory_manager.check_memory_pressure()
        if memory_pressure['pressure_level'] in ['warning', 'critical']:
            await optimizer.memory_manager.optimize_memory()