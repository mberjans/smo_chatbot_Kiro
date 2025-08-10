"""
Basic Integration Test for Directory Monitor Task

This module contains a simple integration test to verify the directory monitoring
and batch processing functionality works correctly.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

from ..component import LightRAGComponent
from ..config.settings import LightRAGConfig


@pytest.mark.asyncio
async def test_component_directory_monitoring():
    """Test that the LightRAG component can initialize directory monitoring."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test configuration
        config = LightRAGConfig()
        config.papers_directory = str(Path(temp_dir) / "papers")
        config.knowledge_graph_path = str(Path(temp_dir) / "kg")
        config.vector_store_path = str(Path(temp_dir) / "vectors")
        config.cache_directory = str(Path(temp_dir) / "cache")
        
        # Create papers directory
        Path(config.papers_directory).mkdir(parents=True, exist_ok=True)
        
        component = LightRAGComponent(config)
        
        try:
            # Initialize component
            await component.initialize()
            
            # Test that directory monitoring methods are available
            assert hasattr(component, 'start_directory_monitoring')
            assert hasattr(component, 'stop_directory_monitoring')
            assert hasattr(component, 'force_directory_scan')
            assert hasattr(component, 'get_monitoring_status')
            assert hasattr(component, 'get_progress_report')
            assert hasattr(component, 'batch_process_files')
            
            # Test getting monitoring status
            status = await component.get_monitoring_status()
            assert 'status' in status
            assert 'papers_directory' in status
            
            # Test getting progress report
            progress = await component.get_progress_report()
            assert 'timestamp' in progress
            
            # Test force scan on empty directory
            scan_result = await component.force_directory_scan()
            assert 'new_files_found' in scan_result
            assert scan_result['new_files_found'] == 0
            
            # Create a test PDF file
            test_pdf = Path(config.papers_directory) / "test.pdf"
            test_pdf.write_text("Test PDF content")
            
            # Test force scan with PDF file
            scan_result = await component.force_directory_scan()
            assert scan_result['new_files_found'] == 1
            assert len(scan_result['new_files']) == 1
            
            # Test batch processing (will use mock pipeline)
            batch_result = await component.batch_process_files([str(test_pdf)])
            assert 'total_files' in batch_result
            assert batch_result['total_files'] == 1
            
        finally:
            await component.cleanup()


@pytest.mark.asyncio
async def test_directory_monitoring_workflow():
    """Test the complete directory monitoring workflow."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test configuration
        config = LightRAGConfig()
        config.papers_directory = str(Path(temp_dir) / "papers")
        config.knowledge_graph_path = str(Path(temp_dir) / "kg")
        config.vector_store_path = str(Path(temp_dir) / "vectors")
        config.cache_directory = str(Path(temp_dir) / "cache")
        
        # Create papers directory
        papers_dir = Path(config.papers_directory)
        papers_dir.mkdir(parents=True, exist_ok=True)
        
        component = LightRAGComponent(config)
        
        try:
            # Initialize component
            await component.initialize()
            
            # Start directory monitoring
            await component.start_directory_monitoring()
            
            # Check that monitoring is running
            status = await component.get_monitoring_status()
            assert status['status'] == 'running'
            
            # Create test PDF files
            test_files = []
            for i in range(3):
                test_file = papers_dir / f"test{i}.pdf"
                test_file.write_text(f"Test PDF content {i}")
                test_files.append(str(test_file))
            
            # Wait a moment for monitoring to detect files
            await asyncio.sleep(1)
            
            # Check monitoring status again
            status = await component.get_monitoring_status()
            assert 'stats' in status
            
            # Stop directory monitoring
            await component.stop_directory_monitoring()
            
            # Check that monitoring is stopped
            status = await component.get_monitoring_status()
            assert status['status'] == 'stopped'
            
        finally:
            await component.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])