#!/usr/bin/env python3
"""
Test Knowledge Graph Construction from PDF Content
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
os.environ.setdefault('OPENAI_API_KEY', 'sk-test_key_placeholder')
os.environ.setdefault('GROQ_API_KEY', 'GROQ_API_KEY_PLACEHOLDER')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_knowledge_graph_construction():
    """Test the knowledge graph construction from PDF content"""
    
    print("🧠 Testing Knowledge Graph Construction")
    print("=" * 60)
    
    try:
        # Import LightRAG components
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        print("✅ LightRAG imports successful")
        
        # Create configuration
        config = LightRAGConfig.from_env()
        
        # Create LightRAG component
        lightrag_component = LightRAGComponent(config)
        await lightrag_component.initialize()
        print("✅ LightRAG component initialized")
        
        # Ensure PDF exists
        pdf_path = Path("clinical_metabolomics_review.pdf")
        papers_dir = Path("papers")
        papers_dir.mkdir(exist_ok=True)
        
        target_pdf = papers_dir / "clinical_metabolomics_review.pdf"
        
        if pdf_path.exists() and not target_pdf.exists():
            import shutil
            shutil.copy2(pdf_path, target_pdf)
            print(f"✅ Copied PDF to papers directory")
        elif target_pdf.exists():
            print(f"✅ PDF already in papers directory")
        else:
            print("❌ No PDF found for testing")
            return False
        
        # Test PDF ingestion with knowledge graph construction
        print("\n🔄 Testing PDF Ingestion with Knowledge Graph Construction")
        print("-" * 50)
        
        ingestion_result = await lightrag_component.ingest_documents([str(target_pdf)])
        
        print(f"📊 Ingestion Results:")
        print(f"   Processed files: {ingestion_result.get('processed_files', 0)}")
        print(f"   Successful: {ingestion_result.get('successful', 0)}")
        print(f"   Failed: {ingestion_result.get('failed', 0)}")
        print(f"   Processing time: {ingestion_result.get('processing_time', 0):.2f}s")
        
        # Check if knowledge graph was created
        if ingestion_result.get('successful', 0) > 0:
            valid_files = ingestion_result.get('valid_files', [])
            if valid_files:
                # Get knowledge graph info from the first successful file
                print(f"\n🧠 Knowledge Graph Construction Results:")
                
                # The knowledge graph info should be in the processing results
                # Let's check if we can access it through the component
                try:
                    from lightrag_integration.ingestion.knowledge_graph import KnowledgeGraphBuilder
                    kg_builder = KnowledgeGraphBuilder(config)
                    
                    # List available graphs
                    available_graphs = await kg_builder.list_available_graphs()
                    print(f"   Available graphs: {len(available_graphs)}")
                    
                    if available_graphs:
                        # Get statistics for the first graph
                        graph_id = available_graphs[0]
                        stats = await kg_builder.get_graph_statistics(graph_id)
                        
                        print(f"   Graph ID: {graph_id}")
                        print(f"   Total nodes: {stats.get('total_nodes', 0)}")
                        print(f"   Total edges: {stats.get('total_edges', 0)}")
                        print(f"   Node types: {stats.get('node_types', {})}")
                        print(f"   Edge types: {stats.get('edge_types', {})}")
                        print(f"   Average node confidence: {stats.get('avg_node_confidence', 0):.2f}")
                        print(f"   Average edge confidence: {stats.get('avg_edge_confidence', 0):.2f}")
                        
                        # Test querying the knowledge graph
                        print(f"\n🔍 Testing Knowledge Graph Query")
                        print("-" * 40)
                        
                        question = "What does the clinical metabolomics review document say about sample preparation methods?"
                        result = await lightrag_component.query(question)
                        
                        print(f"Query: {question}")
                        print(f"Answer: {result.get('answer', 'No answer provided')}")
                        print(f"Confidence: {result.get('confidence_score', 0.0):.2f}")
                        print(f"Source documents: {len(result.get('source_documents', []))}")
                        print(f"Entities used: {len(result.get('entities_used', []))}")
                        print(f"Relationships used: {len(result.get('relationships_used', []))}")
                        print(f"Processing time: {result.get('processing_time', 0.0):.2f}s")
                        
                        # Show some entities and relationships if available
                        if result.get('entities_used'):
                            print(f"\n📋 Sample Entities Used:")
                            for i, entity in enumerate(result['entities_used'][:3], 1):
                                print(f"   {i}. {entity.get('text', 'N/A')} ({entity.get('type', 'N/A')})")
                        
                        if result.get('relationships_used'):
                            print(f"\n🔗 Sample Relationships Used:")
                            for i, rel in enumerate(result['relationships_used'][:3], 1):
                                print(f"   {i}. {rel.get('type', 'N/A')} (confidence: {rel.get('confidence', 0):.2f})")
                        
                        return True
                    else:
                        print("   ⚠️  No knowledge graphs found")
                        return False
                        
                except Exception as e:
                    print(f"   ❌ Error accessing knowledge graph: {str(e)}")
                    return False
            else:
                print("   ❌ No valid files processed")
                return False
        else:
            print("   ❌ PDF ingestion failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def test_simple_entity_extraction():
    """Test the simple entity extractor directly"""
    
    print("\n🔬 Testing Simple Entity Extraction")
    print("=" * 60)
    
    try:
        from lightrag_integration.ingestion.simple_entity_extractor import SimpleEntityExtractor
        
        # Create entity extractor
        extractor = SimpleEntityExtractor()
        
        # Test text with metabolomics content
        test_text = """
        Sample preparation methods in clinical metabolomics are crucial for accurate results. 
        Glucose and lactate levels were significantly associated with diabetes. 
        Mass spectrometry (LC-MS) was used to analyze plasma samples. 
        The glycolysis pathway produces pyruvate from glucose. 
        Creatinine serves as a biomarker for kidney function.
        """
        
        print(f"📝 Test Text: {test_text.strip()}")
        
        # Extract entities and relationships
        result = await extractor.extract_entities_and_relationships(test_text, "test_context")
        
        print(f"\n📊 Extraction Results:")
        print(f"   Entities extracted: {len(result.entities)}")
        print(f"   Relationships extracted: {len(result.relationships)}")
        print(f"   Processing time: {result.processing_time:.3f}s")
        
        # Show extracted entities
        if result.entities:
            print(f"\n📋 Extracted Entities:")
            for i, entity in enumerate(result.entities[:10], 1):  # Show first 10
                print(f"   {i}. '{entity.text}' ({entity.entity_type}) - confidence: {entity.confidence_score:.2f}")
        
        # Show extracted relationships
        if result.relationships:
            print(f"\n🔗 Extracted Relationships:")
            for i, rel in enumerate(result.relationships[:5], 1):  # Show first 5
                print(f"   {i}. {rel.relationship_type} - confidence: {rel.confidence_score:.2f}")
                print(f"      Evidence: '{rel.evidence_text[:100]}...'")
        
        return len(result.entities) > 0 and len(result.relationships) > 0
        
    except Exception as e:
        print(f"❌ Entity extraction test failed: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main test function"""
    
    print("🚀 Knowledge Graph Construction Test Suite")
    print("=" * 60)
    
    # Test simple entity extraction first
    entity_test_success = await test_simple_entity_extraction()
    
    # Test full knowledge graph construction
    kg_test_success = await test_knowledge_graph_construction()
    
    print(f"\n🏁 Test Results:")
    print(f"   Entity extraction: {'✅ PASSED' if entity_test_success else '❌ FAILED'}")
    print(f"   Knowledge graph construction: {'✅ PASSED' if kg_test_success else '❌ FAILED'}")
    
    if entity_test_success and kg_test_success:
        print(f"\n🎉 All tests passed! Knowledge graph construction is working.")
        print(f"   The system can now extract entities and relationships from PDF content")
        print(f"   and build knowledge graphs for semantic querying.")
    else:
        print(f"\n⚠️  Some tests failed, but basic functionality may still work.")

if __name__ == "__main__":
    asyncio.run(main())