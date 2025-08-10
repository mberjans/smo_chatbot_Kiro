#!/usr/bin/env python3
"""
Debug PDF Processing Step by Step
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up environment variables
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test_db')
os.environ.setdefault('NEO4J_PASSWORD', 'test_password')
os.environ.setdefault('PERPLEXITY_API', 'test_api_key_placeholder')
os.environ.setdefault('OPENAI_API_KEY', 'OPENAI_API_KEY_PLACEHOLDER')
os.environ.setdefault('GROQ_API_KEY', 'GROQ_API_KEY_PLACEHOLDER')

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_pdf_processing():
    """Debug the PDF processing step by step"""
    
    print("🔍 Debugging PDF Processing")
    print("=" * 50)
    
    try:
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        # Create configuration
        config = LightRAGConfig.from_env()
        
        # Create LightRAG component
        lightrag_component = LightRAGComponent(config)
        await lightrag_component.initialize()
        print("✅ LightRAG component initialized")
        
        # Test the _process_single_pdf method directly
        pdf_path = "papers/clinical_metabolomics_review.pdf"
        
        if not Path(pdf_path).exists():
            print("❌ PDF not found")
            return
        
        print(f"📄 Testing direct PDF processing: {pdf_path}")
        
        # Call the PDF processing method directly
        result = await lightrag_component._process_single_pdf(pdf_path)
        
        print(f"📊 PDF Processing Result:")
        print(f"   Success: {result.get('success', False)}")
        print(f"   File path: {result.get('file_path', 'N/A')}")
        print(f"   Text length: {result.get('text_length', 0)}")
        print(f"   Pages processed: {result.get('pages_processed', 0)}")
        
        if 'knowledge_graph' in result:
            kg_info = result['knowledge_graph']
            print(f"   Knowledge Graph:")
            print(f"     Graph ID: {kg_info.get('graph_id', 'N/A')}")
            print(f"     Nodes created: {kg_info.get('nodes_created', 0)}")
            print(f"     Edges created: {kg_info.get('edges_created', 0)}")
            print(f"     Entities extracted: {kg_info.get('entities_extracted', 0)}")
            print(f"     Relationships extracted: {kg_info.get('relationships_extracted', 0)}")
            print(f"     Processing time: {kg_info.get('processing_time', 0):.2f}s")
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main debug function"""
    success = await debug_pdf_processing()
    
    if success:
        print(f"\n✅ PDF processing is working!")
    else:
        print(f"\n❌ PDF processing has issues.")

if __name__ == "__main__":
    asyncio.run(main())