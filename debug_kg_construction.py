#!/usr/bin/env python3
"""
Debug Knowledge Graph Construction
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

# Configure logging to see more details
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

async def debug_pdf_processing():
    """Debug the PDF processing step by step"""
    
    print("🔍 Debugging PDF Processing and Knowledge Graph Construction")
    print("=" * 70)
    
    try:
        # Import components
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        from lightrag_integration.ingestion.simple_entity_extractor import SimpleEntityExtractor
        from lightrag_integration.ingestion.knowledge_graph import KnowledgeGraphBuilder
        
        # Create configuration
        config = LightRAGConfig.from_env()
        
        # Test 1: Direct PDF text extraction
        print("\n📄 Step 1: PDF Text Extraction")
        print("-" * 40)
        
        pdf_path = Path("papers/clinical_metabolomics_review.pdf")
        if not pdf_path.exists():
            print("❌ PDF not found")
            return
        
        import PyPDF2
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text_content = ""
            
            for page_num, page in enumerate(reader.pages[:2]):  # Just first 2 pages for testing
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"
                except Exception as e:
                    print(f"Error on page {page_num + 1}: {e}")
        
        print(f"✅ Extracted {len(text_content)} characters from PDF")
        print(f"Sample text: {text_content[:200]}...")
        
        # Test 2: Entity extraction
        print("\n🔬 Step 2: Entity Extraction")
        print("-" * 40)
        
        extractor = SimpleEntityExtractor(config)
        
        # Process a smaller chunk for testing
        test_chunk = text_content[:2000]  # First 2000 characters
        extraction_result = await extractor.extract_entities_and_relationships(
            test_chunk, "debug_test"
        )
        
        print(f"✅ Extracted {len(extraction_result.entities)} entities")
        print(f"✅ Extracted {len(extraction_result.relationships)} relationships")
        
        # Show some entities
        if extraction_result.entities:
            print(f"\nSample entities:")
            for i, entity in enumerate(extraction_result.entities[:5], 1):
                print(f"  {i}. '{entity.text}' ({entity.entity_type}) - {entity.confidence_score:.2f}")
        
        # Show some relationships
        if extraction_result.relationships:
            print(f"\nSample relationships:")
            for i, rel in enumerate(extraction_result.relationships[:3], 1):
                print(f"  {i}. {rel.relationship_type} - {rel.confidence_score:.2f}")
                print(f"     Evidence: '{rel.evidence_text[:80]}...'")
        
        # Test 3: Knowledge graph construction
        print("\n🧠 Step 3: Knowledge Graph Construction")
        print("-" * 40)
        
        kg_builder = KnowledgeGraphBuilder(config)
        document_id = "debug_test_doc"
        
        graph_result = await kg_builder.construct_graph_from_entities_and_relationships(
            extraction_result.entities,
            extraction_result.relationships,
            document_id
        )
        
        print(f"Graph construction success: {graph_result.success}")
        if graph_result.success:
            print(f"✅ Created {graph_result.nodes_created} nodes")
            print(f"✅ Created {graph_result.edges_created} edges")
            print(f"✅ Graph ID: {graph_result.graph.graph_id}")
            
            # Test 4: Graph persistence and retrieval
            print("\n💾 Step 4: Graph Persistence")
            print("-" * 40)
            
            # List available graphs
            available_graphs = await kg_builder.list_available_graphs()
            print(f"Available graphs after construction: {available_graphs}")
            
            if available_graphs:
                # Get statistics for the graph
                graph_id = graph_result.graph.graph_id
                stats = await kg_builder.get_graph_statistics(graph_id)
                print(f"Graph statistics: {stats}")
                
                # Test 5: Query the knowledge graph
                print("\n🔍 Step 5: Knowledge Graph Query")
                print("-" * 40)
                
                # Try to load and query the graph
                loaded_graph = await kg_builder.load_graph_from_storage(graph_id)
                if loaded_graph:
                    print(f"✅ Successfully loaded graph with {len(loaded_graph.nodes)} nodes and {len(loaded_graph.edges)} edges")
                    
                    # Show some nodes
                    print(f"\nSample nodes:")
                    for i, (node_id, node) in enumerate(list(loaded_graph.nodes.items())[:5], 1):
                        print(f"  {i}. {node.text} ({node.node_type}) - confidence: {node.confidence_score:.2f}")
                    
                    # Show some edges
                    if loaded_graph.edges:
                        print(f"\nSample edges:")
                        for i, (edge_id, edge) in enumerate(list(loaded_graph.edges.items())[:3], 1):
                            print(f"  {i}. {edge.edge_type} - confidence: {edge.confidence_score:.2f}")
                            print(f"     Evidence: {edge.evidence[0][:60]}..." if edge.evidence else "     No evidence")
                    
                    return True
                else:
                    print("❌ Failed to load graph from storage")
                    return False
            else:
                print("❌ No graphs found after construction")
                return False
        else:
            print(f"❌ Graph construction failed: {graph_result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Debug failed: {str(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

async def main():
    """Main debug function"""
    success = await debug_pdf_processing()
    
    if success:
        print(f"\n🎉 Debug successful! Knowledge graph construction is working.")
    else:
        print(f"\n❌ Debug revealed issues in the knowledge graph construction pipeline.")

if __name__ == "__main__":
    asyncio.run(main())