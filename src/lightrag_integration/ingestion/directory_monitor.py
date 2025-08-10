"""
Directory Monitor for Papers Ingestion

This module implements file system monitoring for the papers directory,
automatically detecting new PDF files and triggering ingestion.
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..utils.logging import setup_logger


class MonitorStatus(Enum):
    """Status of the directory monitor."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class FileEvent:
    """Represents a file system event."""
    file_path: str
    event_type: str  # "created", "modified", "deleted"
    timestamp: datetime
    file_size: int = 0
    is_pdf: bool = False


@dataclass
class MonitorStats:
    """Statistics for directory monitoring."""
    start_time: Optional[datetime] = None
    files_detected: int = 0
    files_processed: int = 0
    files_failed: int = 0
    last_scan_time: Optional[datetime] = None
    scan_count: int = 0
    processing_queue_size: int = 0
    errors: List[str] = field(default_factory=list)


class DirectoryMonitor:
    """
    Monitors the papers directory for new PDF files and triggers processing.
    
    This class implements both polling-based monitoring (for cross-platform compatibility)
    and can be extended with native file system watchers for better performance.
    """
    
    def __init__(self, 
                 papers_directory: str,
                 ingestion_callback: Callable[[List[str]], Any],
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the directory monitor.
        
        Args:
            papers_directory: Path to the papers directory to monitor
            ingestion_callback: Async callback function to call when new files are detected
            config: Optional configuration dictionary
        """
        self.papers_directory = Path(papers_directory)
        self.ingestion_callback = ingestion_callback
        self.config = config or {}
        
        # Configuration parameters
        self.scan_interval = self.config.get('scan_interval', 30)  # seconds
        self.batch_size = self.config.get('batch_size', 10)
        self.max_file_age = self.config.get('max_file_age_hours', 24)  # hours
        self.enable_recursive = self.config.get('enable_recursive', False)
        self.file_extensions = self.config.get('file_extensions', ['.pdf'])
        
        # State management
        self.status = MonitorStatus.STOPPED
        self.stats = MonitorStats()
        self.logger = setup_logger("directory_monitor")
        
        # File tracking
        self._known_files: Set[str] = set()
        self._processing_queue: List[str] = []
        self._last_scan_files: Dict[str, float] = {}  # file_path -> modification_time
        
        # Control flags
        self._stop_event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None
        
        self.logger.info(f"Directory monitor initialized for: {self.papers_directory}")
    
    async def start(self) -> None:
        """Start the directory monitoring."""
        if self.status in [MonitorStatus.RUNNING, MonitorStatus.STARTING]:
            self.logger.warning("Directory monitor is already running or starting")
            return
        
        self.status = MonitorStatus.STARTING
        self.logger.info("Starting directory monitor...")
        
        try:
            # Ensure papers directory exists
            self.papers_directory.mkdir(parents=True, exist_ok=True)
            
            # Initialize stats
            self.stats.start_time = datetime.now()
            self.stats.errors.clear()
            
            # Perform initial scan
            await self._initial_scan()
            
            # Start monitoring task
            self._stop_event.clear()
            self._monitor_task = asyncio.create_task(self._monitor_loop())
            
            self.status = MonitorStatus.RUNNING
            self.logger.info("Directory monitor started successfully")
            
        except Exception as e:
            self.status = MonitorStatus.ERROR
            error_msg = f"Failed to start directory monitor: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.stats.errors.append(error_msg)
            raise
    
    async def stop(self) -> None:
        """Stop the directory monitoring."""
        if self.status == MonitorStatus.STOPPED:
            self.logger.info("Directory monitor is already stopped")
            return
        
        self.status = MonitorStatus.STOPPING
        self.logger.info("Stopping directory monitor...")
        
        try:
            # Signal stop to monitoring loop
            self._stop_event.set()
            
            # Wait for monitor task to complete
            if self._monitor_task and not self._monitor_task.done():
                try:
                    await asyncio.wait_for(self._monitor_task, timeout=10.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Monitor task did not stop gracefully, cancelling...")
                    self._monitor_task.cancel()
                    try:
                        await self._monitor_task
                    except asyncio.CancelledError:
                        pass
            
            # Process any remaining files in queue
            if self._processing_queue:
                self.logger.info(f"Processing {len(self._processing_queue)} remaining files...")
                await self._process_batch(self._processing_queue.copy())
                self._processing_queue.clear()
            
            self.status = MonitorStatus.STOPPED
            self.logger.info("Directory monitor stopped successfully")
            
        except Exception as e:
            self.status = MonitorStatus.ERROR
            error_msg = f"Error stopping directory monitor: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.stats.errors.append(error_msg)
            raise
    
    async def force_scan(self) -> Dict[str, Any]:
        """
        Force an immediate scan of the directory.
        
        Returns:
            Dictionary with scan results
        """
        self.logger.info("Performing forced directory scan...")
        
        try:
            scan_start = datetime.now()
            new_files = await self._scan_directory()
            scan_duration = (datetime.now() - scan_start).total_seconds()
            
            result = {
                'scan_duration': scan_duration,
                'new_files_found': len(new_files),
                'new_files': new_files,
                'total_known_files': len(self._known_files),
                'queue_size': len(self._processing_queue)
            }
            
            if new_files:
                self.logger.info(f"Force scan found {len(new_files)} new files")
                # Add to processing queue
                self._processing_queue.extend(new_files)
                
                # Process immediately if not running
                if self.status != MonitorStatus.RUNNING:
                    await self._process_batch(new_files)
            
            return result
            
        except Exception as e:
            error_msg = f"Error during forced scan: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.stats.errors.append(error_msg)
            return {'error': error_msg}
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get current monitor status and statistics.
        
        Returns:
            Dictionary with status information
        """
        uptime = None
        if self.stats.start_time:
            uptime = (datetime.now() - self.stats.start_time).total_seconds()
        
        return {
            'status': self.status.value,
            'uptime_seconds': uptime,
            'papers_directory': str(self.papers_directory),
            'papers_directory_exists': self.papers_directory.exists(),
            'scan_interval': self.scan_interval,
            'batch_size': self.batch_size,
            'stats': {
                'files_detected': self.stats.files_detected,
                'files_processed': self.stats.files_processed,
                'files_failed': self.stats.files_failed,
                'scan_count': self.stats.scan_count,
                'processing_queue_size': len(self._processing_queue),
                'known_files_count': len(self._known_files),
                'last_scan_time': self.stats.last_scan_time.isoformat() if self.stats.last_scan_time else None,
                'error_count': len(self.stats.errors),
                'recent_errors': self.stats.errors[-5:] if self.stats.errors else []
            }
        }
    
    async def _initial_scan(self) -> None:
        """Perform initial scan to populate known files."""
        self.logger.info("Performing initial directory scan...")
        
        try:
            # Get all existing PDF files
            existing_files = await self._get_all_pdf_files()
            
            # Add to known files without processing (they're already there)
            for file_path in existing_files:
                self._known_files.add(file_path)
                # Store modification time
                try:
                    stat = Path(file_path).stat()
                    self._last_scan_files[file_path] = stat.st_mtime
                except OSError:
                    pass
            
            self.logger.info(f"Initial scan found {len(existing_files)} existing PDF files")
            
        except Exception as e:
            error_msg = f"Error during initial scan: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.stats.errors.append(error_msg)
            raise
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("Starting directory monitoring loop...")
        
        while not self._stop_event.is_set():
            try:
                # Perform directory scan
                new_files = await self._scan_directory()
                
                if new_files:
                    self.logger.info(f"Detected {len(new_files)} new files")
                    self.stats.files_detected += len(new_files)
                    
                    # Add to processing queue
                    self._processing_queue.extend(new_files)
                
                # Process batch if queue is large enough or has been waiting
                if (len(self._processing_queue) >= self.batch_size or 
                    (self._processing_queue and self._should_process_queue())):
                    
                    batch = self._processing_queue[:self.batch_size]
                    self._processing_queue = self._processing_queue[self.batch_size:]
                    
                    await self._process_batch(batch)
                
                # Update queue size stat
                self.stats.processing_queue_size = len(self._processing_queue)
                
                # Wait for next scan
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.scan_interval)
                    break  # Stop event was set
                except asyncio.TimeoutError:
                    continue  # Timeout is expected, continue monitoring
                
            except Exception as e:
                error_msg = f"Error in monitoring loop: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.stats.errors.append(error_msg)
                
                # Wait a bit before retrying to avoid tight error loops
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
                    break
                except asyncio.TimeoutError:
                    continue
        
        self.logger.info("Directory monitoring loop stopped")
    
    async def _scan_directory(self) -> List[str]:
        """
        Scan directory for new or modified PDF files.
        
        Returns:
            List of new file paths
        """
        scan_start = datetime.now()
        new_files = []
        
        try:
            # Get all current PDF files
            current_files = await self._get_all_pdf_files()
            current_files_dict = {}
            
            # Check modification times
            for file_path in current_files:
                try:
                    stat = Path(file_path).stat()
                    current_files_dict[file_path] = stat.st_mtime
                    
                    # Check if file is new or modified
                    if (file_path not in self._known_files or 
                        file_path not in self._last_scan_files or
                        stat.st_mtime > self._last_scan_files[file_path]):
                        
                        # Check file age to avoid processing very old files on first run
                        file_age_hours = (time.time() - stat.st_mtime) / 3600
                        if file_age_hours <= self.max_file_age:
                            new_files.append(file_path)
                            self._known_files.add(file_path)
                
                except OSError as e:
                    self.logger.warning(f"Could not stat file {file_path}: {str(e)}")
            
            # Update last scan files
            self._last_scan_files = current_files_dict
            
            # Update stats
            self.stats.last_scan_time = datetime.now()
            self.stats.scan_count += 1
            
            scan_duration = (datetime.now() - scan_start).total_seconds()
            self.logger.debug(f"Directory scan completed in {scan_duration:.2f}s, found {len(new_files)} new files")
            
            return new_files
            
        except Exception as e:
            error_msg = f"Error scanning directory: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.stats.errors.append(error_msg)
            return []
    
    async def _get_all_pdf_files(self) -> List[str]:
        """Get all PDF files in the papers directory."""
        pdf_files = []
        
        try:
            if not self.papers_directory.exists():
                return pdf_files
            
            # Use glob pattern based on recursive setting
            if self.enable_recursive:
                pattern = "**/*.pdf"
            else:
                pattern = "*.pdf"
            
            for pdf_file in self.papers_directory.glob(pattern):
                if pdf_file.is_file() and pdf_file.suffix.lower() in self.file_extensions:
                    pdf_files.append(str(pdf_file))
            
            return pdf_files
            
        except Exception as e:
            self.logger.error(f"Error getting PDF files: {str(e)}")
            return []
    
    def _should_process_queue(self) -> bool:
        """Determine if processing queue should be processed now."""
        if not self._processing_queue:
            return False
        
        # Process if queue has been waiting for more than 2 scan intervals
        if self.stats.last_scan_time:
            time_since_last_scan = (datetime.now() - self.stats.last_scan_time).total_seconds()
            return time_since_last_scan > (self.scan_interval * 2)
        
        return True
    
    async def _process_batch(self, file_paths: List[str]) -> None:
        """
        Process a batch of files through the ingestion callback.
        
        Args:
            file_paths: List of file paths to process
        """
        if not file_paths:
            return
        
        self.logger.info(f"Processing batch of {len(file_paths)} files")
        
        try:
            # Call the ingestion callback
            result = await self.ingestion_callback(file_paths)
            
            # Update stats based on result
            if isinstance(result, dict):
                successful = result.get('successful', 0)
                failed = result.get('failed', 0)
                
                self.stats.files_processed += successful
                self.stats.files_failed += failed
                
                if failed > 0:
                    error_msg = f"Batch processing had {failed} failures"
                    self.logger.warning(error_msg)
                    self.stats.errors.append(error_msg)
                
                self.logger.info(f"Batch processing completed: {successful} successful, {failed} failed")
            else:
                # Assume all files were processed successfully if no detailed result
                self.stats.files_processed += len(file_paths)
                self.logger.info(f"Batch processing completed for {len(file_paths)} files")
            
        except Exception as e:
            error_msg = f"Error processing batch: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            self.stats.errors.append(error_msg)
            self.stats.files_failed += len(file_paths)
    
    async def cleanup(self) -> None:
        """Clean up resources."""
        try:
            await self.stop()
            self.logger.info("Directory monitor cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


class BatchProcessor:
    """
    Handles batch processing of multiple PDF documents with progress tracking.
    """
    
    def __init__(self, ingestion_pipeline, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the batch processor.
        
        Args:
            ingestion_pipeline: The PDF ingestion pipeline instance
            config: Optional configuration dictionary
        """
        self.ingestion_pipeline = ingestion_pipeline
        self.config = config or {}
        
        # Configuration
        self.batch_size = self.config.get('batch_size', 5)
        self.max_concurrent = self.config.get('max_concurrent', 3)
        self.progress_callback = self.config.get('progress_callback')
        
        self.logger = setup_logger("batch_processor")
    
    async def process_files(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Process multiple files in batches with progress tracking.
        
        Args:
            file_paths: List of file paths to process
        
        Returns:
            Dictionary with processing results and statistics
        """
        if not file_paths:
            return {
                'total_files': 0,
                'successful': 0,
                'failed': 0,
                'results': [],
                'processing_time': 0.0,
                'errors': []
            }
        
        start_time = datetime.now()
        self.logger.info(f"Starting batch processing of {len(file_paths)} files")
        
        # Initialize results tracking
        results = []
        successful_count = 0
        failed_count = 0
        errors = []
        
        try:
            # Process files in batches
            for i in range(0, len(file_paths), self.batch_size):
                batch = file_paths[i:i + self.batch_size]
                batch_num = (i // self.batch_size) + 1
                total_batches = (len(file_paths) + self.batch_size - 1) // self.batch_size
                
                self.logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} files)")
                
                # Process batch with concurrency control
                batch_results = await self._process_batch_concurrent(batch)
                
                # Collect results
                for result in batch_results:
                    results.append(result)
                    if result.get('success', False):
                        successful_count += 1
                    else:
                        failed_count += 1
                        if 'error_message' in result:
                            errors.append(result['error_message'])
                
                # Report progress
                progress = {
                    'completed_files': len(results),
                    'total_files': len(file_paths),
                    'successful': successful_count,
                    'failed': failed_count,
                    'progress_percent': (len(results) / len(file_paths)) * 100,
                    'current_batch': batch_num,
                    'total_batches': total_batches
                }
                
                if self.progress_callback:
                    try:
                        await self.progress_callback(progress)
                    except Exception as e:
                        self.logger.warning(f"Progress callback error: {str(e)}")
                
                self.logger.info(f"Batch {batch_num} completed: {len([r for r in batch_results if r.get('success')])} successful")
        
        except Exception as e:
            error_msg = f"Error during batch processing: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        final_result = {
            'total_files': len(file_paths),
            'successful': successful_count,
            'failed': failed_count,
            'results': results,
            'processing_time': processing_time,
            'errors': errors,
            'average_time_per_file': processing_time / len(file_paths) if file_paths else 0
        }
        
        self.logger.info(
            f"Batch processing completed in {processing_time:.2f}s: "
            f"{successful_count} successful, {failed_count} failed"
        )
        
        return final_result
    
    async def _process_batch_concurrent(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of files with controlled concurrency.
        
        Args:
            file_paths: List of file paths in the batch
        
        Returns:
            List of processing results
        """
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_single_file(file_path: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.ingestion_pipeline.process_file(file_path)
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {str(e)}")
                    return {
                        'success': False,
                        'file_path': file_path,
                        'error_message': str(e),
                        'processing_time': 0.0,
                        'timestamp': datetime.now().isoformat()
                    }
        
        # Process all files in the batch concurrently
        tasks = [process_single_file(file_path) for file_path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'file_path': file_paths[i],
                    'error_message': str(result),
                    'processing_time': 0.0,
                    'timestamp': datetime.now().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results