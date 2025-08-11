#!/usr/bin/env python3
"""
Trace LightRAG Data Sources

This script investigates where LightRAG gets its information from by:
1. Examining the knowledge graph storage
2. Checking document ingestion paths
3. Analyzing vector stores
4. Tracing query processing flow
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

async def examine_knowledge_graph_storage():
    """Examine what's stored in the knowledge graph"""
    
    print("🔍 Examining Knowledge Graph Storage")
    print("=" * 60)
    
    # Check knowledge graph directory
    kg_path = Path("./data/lightrag_kg")
    if kg_path.exists():
        print(f"✅ Knowledge graph directory exists: {kg_path}")
        
        # List all files in the knowledge graph directory
        files = list(kg_path.rglob("*"))
        print(f"📁 Found {len(files)} files in knowledge graph:")
        
        for file_path in files:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   📄 {file_path.name}: {size} bytes")
                
                # Try to read and analyze small files
                if size < 10000 and file_path.suffix in ['.json', '.txt', '.csv']:
                    try:
                        content = file_path.read_text()
                        print(f"      Preview: {content[:200]}...")
                    except Exception as e:
                        print(f"      Could not read: {str(e)}")
    else:
        print("❌ No knowledge graph directory found")
    
    # Check vector store directory
    vector_path = Path("./data/lightrag_vectors")
    if vector_path.exists():
        print(f"\n✅ Vector store directory exists: {vector_path}")
        
        files = list(vector_path.rglob("*"))
        print(f"📁 Found {len(files)} files in vector store:")
        
        for file_path in files:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   📄 {file_path.name}: {size} bytes")
    else:
        print("\n❌ No vector store directory found")

async def examine_papers_directory():
    """Examine the papers directory for source documents"""
    
    print("\n📚 Examining Papers Directory")
    print("=" * 60)
    
    papers_path = Path("./papers")
    if papers_path.exists():
        print(f"✅ Papers directory exists: {papers_path}")
        
        files = list(papers_path.glob("*"))
        print(f"📁 Found {len(files)} files in papers directory:")
        
        for file_path in files:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   📄 {file_path.name}: {size} bytes ({file_path.suffix})")
                
                # Try to extract text from PDFs
                if file_path.suffix.lower() == '.pdf':
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(str(file_path))
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        doc.close()
                        
                        print(f"      PDF content preview: {text[:200]}...")
                        print(f"      Total text length: {len(text)} characters")
                        
                    except Exception as e:
                        print(f"      Could not extract PDF text: {str(e)}")
    else:
        print("❌ No papers directory found")

async def trace_query_processing_flow():
    """Trace how queries are processed and where information comes from"""
    
    print("\n🔄 Tracing Query Processing Flow")
    print("=" * 60)
    
    try:
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        # Initialize system
        config = LightRAGConfig.from_env()
        component = LightRAGComponent(config)
        await component.initialize()
        
        print("✅ System initialized for tracing")
        
        # Examine the query engine
        print("\n🔍 Examining Query Engine Components:")
        
        # Check if we can access internal components
        try:
            from lightrag_integration.query.engine import LightRAGQueryEngine
            query_engine = LightRAGQueryEngine(config)
            
            print("✅ Query engine created")
            print(f"   Knowledge graph path: {config.knowledge_graph_path}")
            print(f"   Vector store path: {config.vector_store_path}")
            print(f"   Papers directory: {config.papers_directory}")
            print(f"   Embedding model: {config.embedding_model}")
            print(f"   LLM model: {config.llm_model}")
            
        except Exception as e:
            print(f"⚠️  Could not examine query engine: {str(e)}")
        
        # Test a query and trace its path
        print("\n🔄 Tracing a test query...")
        
        test_query = "What is metabolomics?"
        print(f"Query: {test_query}")
        
        # This will show us the processing path
        result = await component.query(test_query)
        
        print(f"Result type: {type(result)}")
        print(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        
        if isinstance(result, dict):
            answer = result.get('answer', 'No answer')
            confidence = result.get('confidence_score', 0.0)
            sources = result.get('source_documents', [])
            
            print(f"Answer: {answer[:200]}...")
            print(f"Confidence: {confidence}")
            print(f"Source documents: {len(sources)}")
            
            if sources:
                print("Source document details:")
                for i, source in enumerate(sources[:3]):  # Show first 3 sources
                    print(f"   Source {i+1}: {source}")
        
        await component.cleanup()
        
    except Exception as e:
        print(f"❌ Query tracing failed: {str(e)}")
        import traceback
        print(f"   Details: {traceback.format_exc()}")

async def examine_cache_and_storage():
    """Examine cache and storage mechanisms"""
    
    print("\n💾 Examining Cache and Storage")
    print("=" * 60)
    
    cache_path = Path("./data/lightrag_cache")
    if cache_path.exists():
        print(f"✅ Cache directory exists: {cache_path}")
        
        files = list(cache_path.rglob("*"))
        print(f"📁 Found {len(files)} files in cache:")
        
        for file_path in files:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   📄 {file_path.name}: {size} bytes")
                
                # Try to read cache files
                if file_path.suffix == '.json' and size < 5000:
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        print(f"      JSON keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    except Exception as e:
                        print(f"      Could not read JSON: {str(e)}")
    else:
        print("❌ No cache directory found")

async def check_external_data_sources():
    """Check for external data sources and APIs"""
    
    print("\n🌐 Checking External Data Sources")
    print("=" * 60)
    
    # Check environment variables for API keys
    api_keys = {
        'GROQ_API_KEY': os.environ.get('GROQ_API_KEY', ''),
        'OPENAI_API_KEY': os.environ.get('OPENAI_API_KEY', ''),
        'PERPLEXITY_API': os.environ.get('PERPLEXITY_API', ''),
    }
    
    print("API Key Status:")
    for key, value in api_keys.items():
        if value and not value.startswith(key + '_PLACEHOLDER'):
            print(f"   ✅ {key}: Configured (real key)")
        elif value:
            print(f"   ⚠️  {key}: Placeholder value")
        else:
            print(f"   ❌ {key}: Not set")
    
    # Check if system uses external APIs for responses
    print("\nExternal API Usage:")
    print("   🤖 LLM APIs: Used for generating responses from processed knowledge")
    print("   🔍 Embedding APIs: Used for vector similarity search")
    print("   🌐 Perplexity API: May be used for enhanced search capabilities")

async def analyze_information_flow():
    """Analyze the complete information flow in LightRAG"""
    
    print("\n📊 LightRAG Information Flow Analysis")
    print("=" * 60)
    
    print("Information Sources (in order of processing):")
    print()
    
    print("1️⃣ PRIMARY SOURCES (Input Documents):")
    print("   📄 PDF Documents in ./papers/ directory")
    print("   📝 Text files and research papers")
    print("   🔬 Scientific literature and publications")
    print()
    
    print("2️⃣ PROCESSING PIPELINE:")
    print("   📖 Document ingestion and text extraction")
    print("   ✂️  Text chunking and preprocessing")
    print("   🧠 Entity and relationship extraction (using LLM)")
    print("   🔗 Knowledge graph construction")
    print("   📊 Vector embeddings generation")
    print()
    
    print("3️⃣ STORAGE SYSTEMS:")
    print("   🗄️  Knowledge Graph: ./data/lightrag_kg/")
    print("   📊 Vector Store: ./data/lightrag_vectors/")
    print("   💾 Cache: ./data/lightrag_cache/")
    print()
    
    print("4️⃣ QUERY PROCESSING:")
    print("   🔍 Query analysis and intent understanding")
    print("   📊 Vector similarity search in embeddings")
    print("   🗄️  Knowledge graph traversal")
    print("   🤖 LLM-based response generation")
    print("   📝 Citation and confidence scoring")
    print()
    
    print("5️⃣ EXTERNAL APIS (for enhancement):")
    print("   🤖 Groq/OpenAI: LLM responses and entity extraction")
    print("   🔤 Embedding Models: Text vectorization")
    print("   🌐 Perplexity API: Enhanced search capabilities")

async def main():
    """Main analysis function"""
    
    print("🔍 LightRAG Data Source Investigation")
    print("=" * 80)
    print("Tracing where LightRAG gets its information from...")
    print("=" * 80)
    
    # Run all investigations
    await examine_knowledge_graph_storage()
    await examine_papers_directory()
    await examine_cache_and_storage()
    await check_external_data_sources()
    await trace_query_processing_flow()
    await analyze_information_flow()
    
    # Summary
    print("\n" + "=" * 80)
    print("🎯 INFORMATION SOURCE SUMMARY")
    print("=" * 80)
    
    print("LightRAG gets its information from:")
    print()
    print("📚 PRIMARY SOURCES:")
    print("   • PDF documents in ./papers/ directory")
    print("   • Research papers and scientific literature")
    print("   • Any documents you add to the system")
    print()
    print("🧠 PROCESSED KNOWLEDGE:")
    print("   • Knowledge graphs built from documents")
    print("   • Vector embeddings for semantic search")
    print("   • Extracted entities and relationships")
    print()
    print("🤖 AI ENHANCEMENT:")
    print("   • LLM APIs (Groq/OpenAI) for response generation")
    print("   • Embedding models for similarity matching")
    print("   • External APIs for enhanced capabilities")
    print()
    print("💡 KEY INSIGHT:")
    print("   LightRAG combines YOUR documents with AI processing")
    print("   to create intelligent, contextual responses!")

if __name__ == "__main__":
    asyncio.run(main())