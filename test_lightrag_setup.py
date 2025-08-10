#!/usr/bin/env python3
"""
Test script to verify LightRAG setup and basic functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lightrag_integration import LightRAGComponent, LightRAGConfig


async def test_lightrag_setup():
    """Test basic LightRAG setup and functionality."""
    print("Testing LightRAG setup...")
    
    try:
        # Test configuration
        print("\n1. Testing configuration...")
        config = LightRAGConfig.from_env()
        print(f"✓ Configuration loaded successfully")
        print(f"  - Knowledge graph path: {config.knowledge_graph_path}")
        print(f"  - Papers directory: {config.papers_directory}")
        print(f"  - Embedding model: {config.embedding_model}")
        
        # Test component initialization
        print("\n2. Testing component initialization...")
        component = LightRAGComponent(config)
        await component.initialize()
        print("✓ Component initialized successfully")
        
        # Test health status
        print("\n3. Testing health status...")
        health = await component.get_health_status()
        print(f"✓ Health status: {health.overall_status.value}")
        for name, comp_health in health.components.items():
            print(f"  - {name}: {comp_health.status.value} - {comp_health.message}")
        
        # Test supported formats
        print("\n4. Testing supported formats...")
        formats = component.get_supported_formats()
        print(f"✓ Supported formats: {formats}")
        
        # Test placeholder query
        print("\n5. Testing placeholder query...")
        response = await component.query("What is clinical metabolomics?")
        print(f"✓ Query processed successfully")
        print(f"  - Answer: {response['answer'][:100]}...")
        
        # Test placeholder ingestion
        print("\n6. Testing placeholder ingestion...")
        result = await component.ingest_documents([])
        print(f"✓ Ingestion test completed")
        print(f"  - Processed files: {result['processed_files']}")
        
        # Cleanup
        await component.cleanup()
        print("\n✓ All tests passed! LightRAG setup is working correctly.")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_lightrag_setup())
    sys.exit(0 if success else 1)