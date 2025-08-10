#!/usr/bin/env python3
"""
Test Full Knowledge Graph Integration with Sample Preparation Query
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
logging.basicConfig(level=logging.INFO)  # Show more details
logger = logging.getLogger(__name__)

async def test_full_integration():
    """Test the full integration with knowledge graph construction and querying"""
    
    print("🚀 Testing Full Knowledge Graph Integration")
    print("=" * 60)
    
    try:
        # Import LightRAG components
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        # Create configuration
        config = LightRAGConfig.from_env()
        
        # Create LightRAG component
        lightrag_component = LightRAGComponent(config)
        await lightrag_component.initialize()
        print("✅ LightRAG component initialized")
        
        # Ensure PDF exists
        pdf_path = Path("papers/clinical_metabolomics_review.pdf")
        if not pdf_path.exists():
            print("❌ PDF not found")
            return False
        
        # Clear any existing knowledge graphs to start fresh
        print("\n🧹 Clearing existing knowledge graphs...")
        from lightrag_integration.ingestion.knowledge_graph import KnowledgeGraphBuilder
        kg_builder = KnowledgeGraphBuilder(config)
        
        existing_graphs = await kg_builder.list_available_graphs()
        for graph_id in existing_graphs:
            await kg_builder.delete_graph(graph_id)
        print(f"✅ Cleared {len(existing_graphs)} existing graphs")
        
        # Process PDF with knowledge graph construction
        print("\n📄 Processing PDF with Knowledge Graph Construction...")
        print("-" * 50)
        
        ingestion_result = await lightrag_component.ingest_documents([str(pdf_path)])
        
        print(f"📊 Ingestion Results:")
        print(f"   Processed files: {ingestion_result.get('processed_files', 0)}")
        print(f"   Successful: {ingestion_result.get('successful', 0)}")
        print(f"   Failed: {ingestion_result.get('failed', 0)}")
        print(f"   Processing time: {ingestion_result.get('processing_time', 0):.2f}s")
        
        if ingestion_result.get('successful', 0) == 0:
            print("❌ PDF ingestion failed")
            return False
        
        # Check knowledge graphs
        print("\n🧠 Checking Knowledge Graph Creation...")
        print("-" * 50)
        
        available_graphs = await kg_builder.list_available_graphs()
        print(f"Available graphs: {len(available_graphs)}")
        
        if not available_graphs:
            print("❌ No knowledge graphs created")
            return False
        
        # Get statistics for the first graph
        graph_id = available_graphs[0]
        stats = await kg_builder.get_graph_statistics(graph_id)
        
        print(f"✅ Knowledge Graph Created:")
        print(f"   Graph ID: {graph_id}")
        print(f"   Total nodes: {stats.get('total_nodes', 0)}")
        print(f"   Total edges: {stats.get('total_edges', 0)}")
        print(f"   Node types: {stats.get('node_types', {})}")
        print(f"   Edge types: {stats.get('edge_types', {})}")
        print(f"   Average node confidence: {stats.get('avg_node_confidence', 0):.2f}")
        print(f"   Average edge confidence: {stats.get('avg_edge_confidence', 0):.2f}")
        
        # Test querying with the sample preparation question
        print("\n🔍 Testing Sample Preparation Query...")
        print("-" * 50)
        
        question = "What does the clinical metabolomics review document say about sample preparation methods?"
        
        result = await lightrag_component.query(question)
        
        print(f"📝 Query Results:")
        print(f"   Question: {question}")
        print(f"   Answer: {result.get('answer', 'No answer provided')}")
        print(f"   Confidence: {result.get('confidence_score', 0.0):.2f}")
        print(f"   Source documents: {len(result.get('source_documents', []))}")
        print(f"   Entities used: {len(result.get('entities_used', []))}")
        print(f"   Relationships used: {len(result.get('relationships_used', []))}")
        print(f"   Processing time: {result.get('processing_time', 0.0):.2f}s")
        print(f"   Fallback used: {result.get('fallback_used', False)}")
        
        # Show entities and relationships if available
        if result.get('entities_used'):
            print(f"\n📋 Entities Used in Response:")
            for i, entity in enumerate(result['entities_used'][:5], 1):
                entity_text = entity.get('text', 'N/A') if isinstance(entity, dict) else str(entity)
                entity_type = entity.get('type', 'N/A') if isinstance(entity, dict) else 'N/A'
                print(f"   {i}. {entity_text} ({entity_type})")
        
        if result.get('relationships_used'):
            print(f"\n🔗 Relationships Used in Response:")
            for i, rel in enumerate(result['relationships_used'][:3], 1):
                rel_type = rel.get('type', 'N/A') if isinstance(rel, dict) else 'N/A'
                rel_conf = rel.get('confidence', 0) if isinstance(rel, dict) else 0
                print(f"   {i}. {rel_type} (confidence: {rel_conf:.2f})")
        
        # Test a few more metabolomics-related questions
        print("\n🧪 Testing Additional Metabolomics Questions...")
        print("-" * 50)
        
        additional_questions = [
            "What analytical techniques are mentioned in the document?",
            "What biomarkers are discussed in the metabolomics review?",
            "What diseases are associated with metabolomics in the document?"
        ]
        
        for i, q in enumerate(additional_questions, 1):
            print(f"\n{i}. {q}")
            result = await lightrag_component.query(q)
            answer = result.get('answer', 'No answer')
            confidence = result.get('confidence_score', 0.0)
            print(f"   Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"   Confidence: {confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main test function"""
    
    success = await test_full_integration()
    
    print(f"\n🏁 Final Result:")
    if success:
        print(f"✅ FULL INTEGRATION SUCCESS!")
        print(f"   Knowledge graphs are being constructed from PDF content")
        print(f"   The system can now answer questions based on the document")
        print(f"   LightRAG is operational with entity extraction and graph construction")
    else:
        print(f"❌ Integration test failed")
        print(f"   Check the logs above for specific issues")

if __name__ == "__main__":
    asyncio.run(main())