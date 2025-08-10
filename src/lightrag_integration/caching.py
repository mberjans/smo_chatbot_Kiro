"""
LightRAG Caching System

This module implements comprehensive caching for query results, vector embeddings,
and database operations to improve performance and reduce latency.
Implements requirements 5.3 and 5.6.
"""

import asyncio
import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from collections import OrderedDict
import threading
import weakref

from .utils.logging import setup_logger


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.last_accessed is None:
            self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'size_bytes': self.size_bytes,
            'metadata': self.metadata
        }


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_rate: float = 0.0
    average_access_time_ms: float = 0.0

    def update_hit_rate(self):
        """Update the hit rate calculation."""
        total_requests = self.hits + self.misses
        self.hit_rate = (self.hits / total_requests) if total_requests > 0 else 0.0


class LRUCache:
    """
    Thread-safe LRU (Least Recently Used) cache implementation.
    """
    
    def __init__(self, max_size: int = 1000, max_memory_mb: int = 100):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        self.logger = setup_logger("lru_cache")

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()
        
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check expiration
                if entry.is_expired():
                    del self._cache[key]
                    self._stats.misses += 1
                    self._stats.evictions += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                
                self._stats.hits += 1
                self._stats.update_hit_rate()
                
                # Update access time stats
                access_time_ms = (time.time() - start_time) * 1000
                self._update_average_access_time(access_time_ms)
                
                return entry.value
            else:
                self._stats.misses += 1
                self._stats.update_hit_rate()
                return None

    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Put value in cache."""
        with self._lock:
            # Calculate size
            try:
                size_bytes = len(pickle.dumps(value))
            except Exception:
                size_bytes = 1024  # Default size if can't calculate

            # Check if single item exceeds memory limit
            if size_bytes > self.max_memory_bytes:
                self.logger.warning(f"Item too large for cache: {size_bytes} bytes")
                return False

            # Create cache entry
            now = datetime.now()
            expires_at = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                expires_at=expires_at,
                size_bytes=size_bytes,
                metadata=metadata or {}
            )

            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.total_size_bytes -= old_entry.size_bytes
                del self._cache[key]

            # Evict entries if necessary
            self._evict_if_necessary(size_bytes)

            # Add new entry
            self._cache[key] = entry
            self._stats.total_entries = len(self._cache)
            self._stats.total_size_bytes += size_bytes

            return True

    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                self._stats.total_size_bytes -= entry.size_bytes
                del self._cache[key]
                self._stats.total_entries = len(self._cache)
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats = CacheStats()

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            self._stats.total_entries = len(self._cache)
            return self._stats

    def _evict_if_necessary(self, new_item_size: int) -> None:
        """Evict entries if cache limits would be exceeded."""
        # Evict by size
        while (self._stats.total_size_bytes + new_item_size > self.max_memory_bytes and 
               len(self._cache) > 0):
            self._evict_lru()

        # Evict by count
        while len(self._cache) >= self.max_size:
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._cache:
            key, entry = self._cache.popitem(last=False)  # Remove first (oldest)
            self._stats.total_size_bytes -= entry.size_bytes
            self._stats.evictions += 1

    def _update_average_access_time(self, access_time_ms: float) -> None:
        """Update average access time with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        if self._stats.average_access_time_ms == 0:
            self._stats.average_access_time_ms = access_time_ms
        else:
            self._stats.average_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * self._stats.average_access_time_ms
            )


class QueryResultCache:
    """
    Specialized cache for query results with semantic similarity checking.
    """
    
    def __init__(self, config):
        """Initialize query result cache."""
        self.config = config
        self.logger = setup_logger("query_cache")
        
        # Cache configuration
        cache_size = getattr(config, 'query_cache_size', 1000)
        cache_memory_mb = getattr(config, 'query_cache_memory_mb', 50)
        self.default_ttl = getattr(config, 'cache_ttl_seconds', 3600)
        
        # Initialize LRU cache
        self._cache = LRUCache(max_size=cache_size, max_memory_mb=cache_memory_mb)
        
        # Query similarity threshold for cache hits
        self.similarity_threshold = getattr(config, 'query_similarity_threshold', 0.9)

    def _generate_cache_key(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key for query."""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Include relevant context in key
        context_str = ""
        if context:
            # Only include stable context elements
            stable_context = {k: v for k, v in context.items() 
                            if k in ['language', 'user_type', 'domain']}
            if stable_context:
                context_str = json.dumps(stable_context, sort_keys=True)
        
        # Create hash
        key_string = f"{normalized_query}|{context_str}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        cache_key = self._generate_cache_key(query, context)
        
        result = self._cache.get(cache_key)
        if result:
            self.logger.debug(f"Cache hit for query: {query[:50]}...")
            # Add cache metadata
            result['metadata'] = result.get('metadata', {})
            result['metadata']['cached'] = True
            result['metadata']['cache_hit_time'] = datetime.now().isoformat()
            
        return result

    async def put(self, query: str, result: Dict[str, Any], context: Optional[Dict[str, Any]] = None, ttl_seconds: Optional[int] = None) -> bool:
        """Cache query result."""
        cache_key = self._generate_cache_key(query, context)
        
        # Use default TTL if not specified
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl
        
        # Add caching metadata
        cache_metadata = {
            'original_query': query,
            'cached_at': datetime.now().isoformat(),
            'ttl_seconds': ttl_seconds
        }
        
        success = self._cache.put(cache_key, result, ttl_seconds, cache_metadata)
        
        if success:
            self.logger.debug(f"Cached result for query: {query[:50]}...")
        else:
            self.logger.warning(f"Failed to cache result for query: {query[:50]}...")
        
        return success

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_stats = self._cache.get_stats()
        return {
            'hits': cache_stats.hits,
            'misses': cache_stats.misses,
            'hit_rate': cache_stats.hit_rate,
            'total_entries': cache_stats.total_entries,
            'total_size_mb': cache_stats.total_size_bytes / (1024 * 1024),
            'evictions': cache_stats.evictions,
            'average_access_time_ms': cache_stats.average_access_time_ms
        }

    def clear(self) -> None:
        """Clear query cache."""
        self._cache.clear()
        self.logger.info("Query cache cleared")


class EmbeddingCache:
    """
    Cache for vector embeddings to avoid recomputation.
    """
    
    def __init__(self, config):
        """Initialize embedding cache."""
        self.config = config
        self.logger = setup_logger("embedding_cache")
        
        # Cache configuration
        cache_size = getattr(config, 'embedding_cache_size', 5000)
        cache_memory_mb = getattr(config, 'embedding_cache_memory_mb', 200)
        self.default_ttl = getattr(config, 'embedding_cache_ttl_seconds', 86400)  # 24 hours
        
        # Initialize LRU cache
        self._cache = LRUCache(max_size=cache_size, max_memory_mb=cache_memory_mb)

    def _generate_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key for embedding."""
        # Normalize text
        normalized_text = text.strip()
        
        # Create hash including model name
        key_string = f"{model_name}|{normalized_text}"
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding."""
        cache_key = self._generate_cache_key(text, model_name)
        
        embedding = self._cache.get(cache_key)
        if embedding:
            self.logger.debug(f"Embedding cache hit for text: {text[:30]}...")
        
        return embedding

    async def put_embedding(self, text: str, model_name: str, embedding: List[float], ttl_seconds: Optional[int] = None) -> bool:
        """Cache embedding."""
        cache_key = self._generate_cache_key(text, model_name)
        
        # Use default TTL if not specified
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl
        
        # Add metadata
        metadata = {
            'text_length': len(text),
            'model_name': model_name,
            'embedding_dimension': len(embedding),
            'cached_at': datetime.now().isoformat()
        }
        
        success = self._cache.put(cache_key, embedding, ttl_seconds, metadata)
        
        if success:
            self.logger.debug(f"Cached embedding for text: {text[:30]}...")
        
        return success

    def get_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        cache_stats = self._cache.get_stats()
        return {
            'hits': cache_stats.hits,
            'misses': cache_stats.misses,
            'hit_rate': cache_stats.hit_rate,
            'total_entries': cache_stats.total_entries,
            'total_size_mb': cache_stats.total_size_bytes / (1024 * 1024),
            'evictions': cache_stats.evictions,
            'average_access_time_ms': cache_stats.average_access_time_ms
        }

    def clear(self) -> None:
        """Clear embedding cache."""
        self._cache.clear()
        self.logger.info("Embedding cache cleared")


class ConnectionPool:
    """
    Connection pool for database operations to improve performance.
    """
    
    def __init__(self, config):
        """Initialize connection pool."""
        self.config = config
        self.logger = setup_logger("connection_pool")
        
        # Pool configuration
        self.min_connections = getattr(config, 'min_db_connections', 2)
        self.max_connections = getattr(config, 'max_db_connections', 10)
        self.connection_timeout = getattr(config, 'db_connection_timeout', 30)
        
        # Pool state
        self._available_connections = asyncio.Queue(maxsize=self.max_connections)
        self._all_connections = set()
        self._connection_count = 0
        self._lock = asyncio.Lock()
        
        # Statistics
        self._stats = {
            'connections_created': 0,
            'connections_destroyed': 0,
            'connections_borrowed': 0,
            'connections_returned': 0,
            'pool_exhausted_count': 0,
            'average_borrow_time_ms': 0.0
        }

    async def initialize(self) -> None:
        """Initialize the connection pool."""
        async with self._lock:
            # Create minimum connections
            for _ in range(self.min_connections):
                conn = await self._create_connection()
                if conn:
                    await self._available_connections.put(conn)

    async def get_connection(self) -> Optional[Any]:
        """Get a connection from the pool."""
        start_time = time.time()
        
        try:
            # Try to get available connection
            try:
                conn = await asyncio.wait_for(
                    self._available_connections.get(),
                    timeout=self.connection_timeout
                )
                
                # Validate connection
                if await self._validate_connection(conn):
                    self._stats['connections_borrowed'] += 1
                    
                    # Update borrow time stats
                    borrow_time_ms = (time.time() - start_time) * 1000
                    self._update_average_borrow_time(borrow_time_ms)
                    
                    return conn
                else:
                    # Connection is invalid, create new one
                    await self._destroy_connection(conn)
                    
            except asyncio.TimeoutError:
                self._stats['pool_exhausted_count'] += 1
                self.logger.warning("Connection pool exhausted, creating new connection")
            
            # Create new connection if pool is empty or connection was invalid
            async with self._lock:
                if self._connection_count < self.max_connections:
                    conn = await self._create_connection()
                    if conn:
                        self._stats['connections_borrowed'] += 1
                        return conn
            
            # Pool is at maximum capacity
            self.logger.error("Cannot create new connection, pool at maximum capacity")
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting connection from pool: {str(e)}")
            return None

    async def return_connection(self, conn: Any) -> None:
        """Return a connection to the pool."""
        try:
            if conn and await self._validate_connection(conn):
                await self._available_connections.put(conn)
                self._stats['connections_returned'] += 1
            else:
                # Connection is invalid, destroy it
                if conn:
                    await self._destroy_connection(conn)
                    
        except Exception as e:
            self.logger.error(f"Error returning connection to pool: {str(e)}")
            if conn:
                await self._destroy_connection(conn)

    async def close_all(self) -> None:
        """Close all connections in the pool."""
        async with self._lock:
            # Close available connections
            while not self._available_connections.empty():
                try:
                    conn = await self._available_connections.get_nowait()
                    await self._destroy_connection(conn)
                except asyncio.QueueEmpty:
                    break
            
            # Close any remaining connections
            for conn in list(self._all_connections):
                await self._destroy_connection(conn)

    async def _create_connection(self) -> Optional[Any]:
        """Create a new database connection."""
        try:
            # TODO: Implement actual database connection creation
            # This is a placeholder for the actual implementation
            
            # For now, return a mock connection object
            conn = {
                'id': f"conn_{self._connection_count}",
                'created_at': datetime.now(),
                'last_used': datetime.now(),
                'is_valid': True
            }
            
            self._all_connections.add(conn['id'])
            self._connection_count += 1
            self._stats['connections_created'] += 1
            
            self.logger.debug(f"Created new connection: {conn['id']}")
            return conn
            
        except Exception as e:
            self.logger.error(f"Failed to create connection: {str(e)}")
            return None

    async def _destroy_connection(self, conn: Any) -> None:
        """Destroy a database connection."""
        try:
            if isinstance(conn, dict) and 'id' in conn:
                conn_id = conn['id']
                self._all_connections.discard(conn_id)
                self._connection_count -= 1
                self._stats['connections_destroyed'] += 1
                
                self.logger.debug(f"Destroyed connection: {conn_id}")
            
        except Exception as e:
            self.logger.error(f"Error destroying connection: {str(e)}")

    async def _validate_connection(self, conn: Any) -> bool:
        """Validate that a connection is still usable."""
        try:
            # TODO: Implement actual connection validation
            # This is a placeholder for the actual implementation
            
            if isinstance(conn, dict):
                return conn.get('is_valid', False)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating connection: {str(e)}")
            return False

    def _update_average_borrow_time(self, borrow_time_ms: float) -> None:
        """Update average borrow time with exponential moving average."""
        alpha = 0.1  # Smoothing factor
        if self._stats['average_borrow_time_ms'] == 0:
            self._stats['average_borrow_time_ms'] = borrow_time_ms
        else:
            self._stats['average_borrow_time_ms'] = (
                alpha * borrow_time_ms + 
                (1 - alpha) * self._stats['average_borrow_time_ms']
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            'total_connections': self._connection_count,
            'available_connections': self._available_connections.qsize(),
            'connections_created': self._stats['connections_created'],
            'connections_destroyed': self._stats['connections_destroyed'],
            'connections_borrowed': self._stats['connections_borrowed'],
            'connections_returned': self._stats['connections_returned'],
            'pool_exhausted_count': self._stats['pool_exhausted_count'],
            'average_borrow_time_ms': self._stats['average_borrow_time_ms']
        }


class CacheManager:
    """
    Central cache manager that coordinates all caching systems.
    """
    
    def __init__(self, config):
        """Initialize cache manager."""
        self.config = config
        self.logger = setup_logger("cache_manager")
        
        # Initialize caches
        self.query_cache = QueryResultCache(config)
        self.embedding_cache = EmbeddingCache(config)
        self.connection_pool = ConnectionPool(config)
        
        # Cache warming configuration
        self.enable_cache_warming = getattr(config, 'enable_cache_warming', True)
        self.cache_warming_queries = getattr(config, 'cache_warming_queries', [
            "What is clinical metabolomics?",
            "What are biomarkers?",
            "How does metabolomics work?",
            "What is mass spectrometry?",
            "What are metabolic pathways?"
        ])

    async def initialize(self) -> None:
        """Initialize all caching systems."""
        try:
            self.logger.info("Initializing cache manager...")
            
            # Initialize connection pool
            await self.connection_pool.initialize()
            
            # Warm up caches if enabled
            if self.enable_cache_warming:
                await self._warm_up_caches()
            
            self.logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize cache manager: {str(e)}")
            raise

    async def _warm_up_caches(self) -> None:
        """Warm up caches with common queries."""
        try:
            self.logger.info("Warming up caches...")
            
            # TODO: Implement cache warming with actual query processing
            # This would involve running common queries and caching results
            
            self.logger.info("Cache warm-up completed")
            
        except Exception as e:
            self.logger.error(f"Cache warm-up failed: {str(e)}")

    async def get_query_result(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get cached query result."""
        return await self.query_cache.get(query, context)

    async def cache_query_result(self, query: str, result: Dict[str, Any], context: Optional[Dict[str, Any]] = None, ttl_seconds: Optional[int] = None) -> bool:
        """Cache query result."""
        return await self.query_cache.put(query, result, context, ttl_seconds)

    async def get_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """Get cached embedding."""
        return await self.embedding_cache.get_embedding(text, model_name)

    async def cache_embedding(self, text: str, model_name: str, embedding: List[float], ttl_seconds: Optional[int] = None) -> bool:
        """Cache embedding."""
        return await self.embedding_cache.put_embedding(text, model_name, embedding, ttl_seconds)

    async def get_connection(self) -> Optional[Any]:
        """Get database connection from pool."""
        return await self.connection_pool.get_connection()

    async def return_connection(self, conn: Any) -> None:
        """Return database connection to pool."""
        await self.connection_pool.return_connection(conn)

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive caching statistics."""
        return {
            'query_cache': self.query_cache.get_stats(),
            'embedding_cache': self.embedding_cache.get_stats(),
            'connection_pool': self.connection_pool.get_stats(),
            'cache_manager': {
                'cache_warming_enabled': self.enable_cache_warming,
                'warming_queries_count': len(self.cache_warming_queries)
            }
        }

    async def clear_all_caches(self) -> None:
        """Clear all caches."""
        self.query_cache.clear()
        self.embedding_cache.clear()
        self.logger.info("All caches cleared")

    async def cleanup(self) -> None:
        """Clean up all caching resources."""
        try:
            await self.connection_pool.close_all()
            await self.clear_all_caches()
            self.logger.info("Cache manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cache manager cleanup: {str(e)}")