#!/usr/bin/env python3
"""
Update LightRAG Knowledge Base

Run this script whenever you add new PDF files to the papers folder.
It will automatically detect new files and add them to LightRAG's knowledge base.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def update_lightrag_knowledge():
    """Update LightRAG knowledge base with new documents"""
    
    print("🔄 Updating LightRAG Knowledge Base")
    print("=" * 60)
    
    try:
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        # Initialize LightRAG
        print("1️⃣ Initializing LightRAG system...")
        config = LightRAGConfig.from_env()
        component = LightRAGComponent(config)
        await component.initialize()
        print("✅ System initialized")
        
        # Check papers directory
        papers_dir = Path("./papers")
        if not papers_dir.exists():
            print("❌ Papers directory not found. Creating it...")
            papers_dir.mkdir(exist_ok=True)
            print("✅ Created papers directory")
            print("📝 Add your PDF files to the papers directory and run this script again")
            return False
        
        # Find all PDF files
        pdf_files = list(papers_dir.glob("*.pdf"))
        txt_files = list(papers_dir.glob("*.txt"))
        all_files = pdf_files + txt_files
        
        if not all_files:
            print("❌ No PDF or text files found in papers directory")
            print("📝 Add your PDF files to the papers directory and run this script again")
            return False
        
        print(f"2️⃣ Found {len(all_files)} files to process:")
        for file_path in all_files:
            size = file_path.stat().st_size
            print(f"   📄 {file_path.name}: {size:,} bytes")
        
        # Process all documents
        print(f"\n3️⃣ Processing {len(all_files)} documents...")
        file_paths = [str(f) for f in all_files]
        
        start_time = datetime.now()
        result = await component.ingest_documents(file_paths)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Show results
        print(f"\n4️⃣ Processing Results:")
        print(f"   ✅ Successfully processed: {result.get('successful', 0)} files")
        print(f"   ❌ Failed to process: {result.get('failed', 0)} files")
        print(f"   ⏱️  Processing time: {processing_time:.2f} seconds")
        
        if result.get('errors'):
            print(f"   ⚠️  Errors encountered:")
            for error in result.get('errors', []):
                print(f"      • {error}")
        
        # Show system statistics
        print(f"\n5️⃣ Updated System Statistics:")
        stats = component.get_statistics()
        print(f"   📊 Total queries processed: {stats.get('queries_processed', 0)}")
        print(f"   📄 Total documents ingested: {stats.get('documents_ingested', 0)}")
        print(f"   🔧 System initialized: {stats.get('is_initialized', False)}")
        
        # Test with a sample query
        print(f"\n6️⃣ Testing updated knowledge base...")
        test_query = "What topics are covered in the documents?"
        print(f"   Query: {test_query}")
        
        query_result = await component.query(test_query)
        answer = query_result.get('answer', 'No answer available')
        confidence = query_result.get('confidence_score', 0.0)
        
        if "queued" in answer.lower():
            print(f"   ⏳ Query queued for processing (system is working)")
        else:
            print(f"   📝 Answer: {answer[:200]}...")
        print(f"   📊 Confidence: {confidence:.2f}")
        
        # Cleanup
        print(f"\n7️⃣ Cleaning up...")
        await component.cleanup()
        print("✅ Cleanup completed")
        
        success = result.get('successful', 0) > 0
        
        if success:
            print(f"\n🎉 Knowledge base updated successfully!")
            print(f"   • {result.get('successful', 0)} documents added to LightRAG")
            print(f"   • You can now query the updated knowledge base")
            print(f"   • Run queries using the demo scripts or your own code")
        else:
            print(f"\n⚠️  No documents were successfully processed")
            print(f"   • Check that PDF files are valid and readable")
            print(f"   • Ensure files are not corrupted or password-protected")
        
        return success
        
    except Exception as e:
        print(f"❌ Failed to update knowledge base: {str(e)}")
        import traceback
        print(f"   Details: {traceback.format_exc()}")
        return False

async def show_current_status():
    """Show current status of the knowledge base"""
    
    print("\n📊 Current Knowledge Base Status")
    print("=" * 60)
    
    # Check directories
    papers_dir = Path("./papers")
    kg_dir = Path("./data/lightrag_kg")
    vector_dir = Path("./data/lightrag_vectors")
    cache_dir = Path("./data/lightrag_cache")
    
    print("Directory Status:")
    print(f"   📁 Papers: {'✅' if papers_dir.exists() else '❌'} {papers_dir}")
    print(f"   🗄️  Knowledge Graph: {'✅' if kg_dir.exists() else '❌'} {kg_dir}")
    print(f"   📊 Vector Store: {'✅' if vector_dir.exists() else '❌'} {vector_dir}")
    print(f"   💾 Cache: {'✅' if cache_dir.exists() else '❌'} {cache_dir}")
    
    if papers_dir.exists():
        files = list(papers_dir.glob("*"))
        print(f"\nFiles in papers directory: {len(files)}")
        for file_path in files:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   📄 {file_path.name}: {size:,} bytes")
    
    if kg_dir.exists():
        kg_files = list(kg_dir.rglob("*"))
        kg_file_count = len([f for f in kg_files if f.is_file()])
        print(f"\nKnowledge graph files: {kg_file_count}")
    
    if vector_dir.exists():
        vector_files = list(vector_dir.rglob("*"))
        vector_file_count = len([f for f in vector_files if f.is_file()])
        print(f"Vector store files: {vector_file_count}")

def show_usage():
    """Show usage instructions"""
    
    print("📖 How to Use This Script")
    print("=" * 60)
    print()
    print("1️⃣ Add PDF files to the papers directory:")
    print("   cp your_research_paper.pdf ./papers/")
    print("   cp another_document.pdf ./papers/")
    print()
    print("2️⃣ Run this script to update LightRAG:")
    print("   python update_lightrag_knowledge.py")
    print()
    print("3️⃣ Query your updated knowledge base:")
    print("   python demo_lightrag_working.py")
    print()
    print("🔄 Repeat steps 1-2 whenever you add new documents!")

async def main():
    """Main function"""
    
    print("🚀 LightRAG Knowledge Base Updater")
    print("=" * 80)
    
    # Show current status first
    await show_current_status()
    
    # Update knowledge base
    success = await update_lightrag_knowledge()
    
    # Show usage instructions
    show_usage()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 SUMMARY")
    print("=" * 80)
    
    if success:
        print("✅ Knowledge base updated successfully!")
        print("🚀 LightRAG is ready to answer questions about your documents")
        print()
        print("Next steps:")
        print("   • Test queries with: python demo_lightrag_working.py")
        print("   • Add more documents and run this script again")
        print("   • Integrate with your applications")
    else:
        print("⚠️  Knowledge base update had issues")
        print("📝 Check the error messages above and try again")
        print()
        print("Common solutions:")
        print("   • Ensure PDF files are not corrupted")
        print("   • Check that files are not password-protected")
        print("   • Verify API keys are configured correctly")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)