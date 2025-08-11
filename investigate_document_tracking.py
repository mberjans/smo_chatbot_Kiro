#!/usr/bin/env python3
"""
Investigate Document Tracking

This script investigates how LightRAG tracks which documents have been processed
to understand the duplicate detection mechanism.
"""

import os
import sys
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def examine_tracking_mechanisms():
    """Examine how documents are tracked"""
    
    print("🔍 Investigating Document Tracking Mechanisms")
    print("=" * 60)
    
    # Check cache directory for tracking files
    cache_dir = Path("./data/lightrag_cache")
    if cache_dir.exists():
        print(f"✅ Cache directory exists: {cache_dir}")
        
        files = list(cache_dir.rglob("*"))
        print(f"📁 Found {len(files)} files in cache:")
        
        for file_path in files:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   📄 {file_path.name}: {size} bytes")
                
                # Examine specific tracking files
                if file_path.name in ['progress_status.json', 'document_registry.json', 'processed_files.json']:
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        print(f"      📋 Content preview: {list(data.keys()) if isinstance(data, dict) else str(data)[:100]}")
                    except Exception as e:
                        print(f"      ⚠️  Could not read: {str(e)}")
    
    # Check knowledge graph directory
    kg_dir = Path("./data/lightrag_kg")
    if kg_dir.exists():
        print(f"\n✅ Knowledge graph directory exists: {kg_dir}")
        
        files = list(kg_dir.rglob("*"))
        print(f"📁 Found {len(files)} files in knowledge graph:")
        
        for file_path in files:
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   📄 {file_path.name}: {size} bytes")
                
                # Look for document tracking files
                if 'document' in file_path.name.lower() or 'registry' in file_path.name.lower():
                    print(f"      🎯 Potential tracking file: {file_path.name}")

async def examine_component_tracking():
    """Examine how the LightRAG component tracks documents"""
    
    print("\n🔧 Examining Component-Level Tracking")
    print("=" * 60)
    
    try:
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        # Initialize system
        config = LightRAGConfig.from_env()
        component = LightRAGComponent(config)
        await component.initialize()
        
        print("✅ Component initialized")
        
        # Check if component has tracking methods
        methods = [method for method in dir(component) if 'track' in method.lower() or 'processed' in method.lower() or 'registry' in method.lower()]
        if methods:
            print(f"📋 Found tracking-related methods: {methods}")
        else:
            print("⚠️  No obvious tracking methods found")
        
        # Check component attributes
        attributes = [attr for attr in dir(component) if not attr.startswith('_')]
        tracking_attrs = [attr for attr in attributes if any(keyword in attr.lower() for keyword in ['track', 'processed', 'registry', 'cache', 'history'])]
        
        if tracking_attrs:
            print(f"📋 Found tracking-related attributes: {tracking_attrs}")
        else:
            print("⚠️  No obvious tracking attributes found")
        
        # Check statistics for document tracking info
        stats = component.get_statistics()
        print(f"\n📊 Component Statistics:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        await component.cleanup()
        
    except Exception as e:
        print(f"❌ Component examination failed: {str(e)}")

async def examine_ingestion_process():
    """Examine the document ingestion process for tracking logic"""
    
    print("\n📖 Examining Document Ingestion Process")
    print("=" * 60)
    
    try:
        # Look at the ingestion code
        from lightrag_integration.ingestion.pdf_processor import PDFIngestionPipeline
        
        print("✅ PDF ingestion pipeline found")
        
        # Check if it has tracking methods
        pipeline_methods = [method for method in dir(PDFIngestionPipeline) if not method.startswith('_')]
        tracking_methods = [method for method in pipeline_methods if any(keyword in method.lower() for keyword in ['track', 'processed', 'exists', 'check'])]
        
        if tracking_methods:
            print(f"📋 Found potential tracking methods: {tracking_methods}")
        else:
            print("⚠️  No obvious tracking methods in PDF pipeline")
        
    except ImportError:
        print("⚠️  PDF ingestion pipeline not found or not importable")
    except Exception as e:
        print(f"❌ Ingestion examination failed: {str(e)}")

async def test_duplicate_detection():
    """Test how the system handles duplicate documents"""
    
    print("\n🧪 Testing Duplicate Detection")
    print("=" * 60)
    
    try:
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        # Initialize system
        config = LightRAGConfig.from_env()
        component = LightRAGComponent(config)
        await component.initialize()
        
        # Get list of existing files
        papers_dir = Path("./papers")
        if papers_dir.exists():
            pdf_files = list(papers_dir.glob("*.pdf"))
            if pdf_files:
                test_file = pdf_files[0]
                print(f"📄 Testing with existing file: {test_file.name}")
                
                # Try to process the same file twice
                print("🔄 First processing attempt...")
                result1 = await component.ingest_documents([str(test_file)])
                print(f"   Result 1: {result1}")
                
                print("🔄 Second processing attempt (should detect duplicate)...")
                result2 = await component.ingest_documents([str(test_file)])
                print(f"   Result 2: {result2}")
                
                # Compare results
                if result1 == result2:
                    print("⚠️  Results are identical - may not have duplicate detection")
                else:
                    print("✅ Results differ - likely has duplicate detection")
            else:
                print("❌ No PDF files found for testing")
        
        await component.cleanup()
        
    except Exception as e:
        print(f"❌ Duplicate detection test failed: {str(e)}")

async def examine_progress_tracking():
    """Examine the progress tracking system"""
    
    print("\n📊 Examining Progress Tracking System")
    print("=" * 60)
    
    # Read progress status file
    progress_file = Path("./data/lightrag_cache/progress_status.json")
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            
            print("✅ Progress status file found")
            print("📋 Progress data structure:")
            for key, value in progress_data.items():
                if isinstance(value, (list, dict)):
                    print(f"   {key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"   {key}: {value}")
            
            # Look for document-specific tracking
            if 'completed_operations' in progress_data:
                completed = progress_data['completed_operations']
                print(f"\n📄 Completed operations: {len(completed)}")
                for i, op in enumerate(completed[:3]):  # Show first 3
                    print(f"   Operation {i+1}: {op}")
            
            if 'active_operations' in progress_data:
                active = progress_data['active_operations']
                print(f"\n🔄 Active operations: {len(active)}")
                for i, op in enumerate(active[:3]):  # Show first 3
                    print(f"   Operation {i+1}: {op}")
                    
        except Exception as e:
            print(f"❌ Could not read progress file: {str(e)}")
    else:
        print("❌ No progress status file found")

async def check_lightrag_internal_tracking():
    """Check if LightRAG library has internal document tracking"""
    
    print("\n🔬 Checking LightRAG Library Internal Tracking")
    print("=" * 60)
    
    try:
        import lightrag
        print("✅ LightRAG library available")
        
        # Check if there are any document status or tracking modules
        lightrag_modules = [attr for attr in dir(lightrag) if not attr.startswith('_')]
        print(f"📋 LightRAG modules: {lightrag_modules}")
        
        # Look for document status tracking
        try:
            from lightrag.kg.json_doc_status_impl import JsonDocStatus
            print("✅ Found JsonDocStatus - this likely tracks processed documents")
            
            # Check its methods
            status_methods = [method for method in dir(JsonDocStatus) if not method.startswith('_')]
            print(f"📋 JsonDocStatus methods: {status_methods}")
            
        except ImportError:
            print("⚠️  JsonDocStatus not found")
        
        # Check working directory for LightRAG internal files
        working_dirs = [
            Path("./data/lightrag_kg"),
            Path("./data/lightrag_vectors"),
            Path("./test_lightrag_working") if Path("./test_lightrag_working").exists() else None
        ]
        
        for working_dir in working_dirs:
            if working_dir and working_dir.exists():
                print(f"\n📁 Checking LightRAG working directory: {working_dir}")
                files = list(working_dir.rglob("*"))
                for file_path in files:
                    if file_path.is_file():
                        print(f"   📄 {file_path.name}: {file_path.stat().st_size} bytes")
                        
                        # Look for document status files
                        if 'doc_status' in file_path.name or 'document' in file_path.name:
                            print(f"      🎯 Potential document tracking file!")
        
    except ImportError:
        print("❌ LightRAG library not available")
    except Exception as e:
        print(f"❌ LightRAG internal check failed: {str(e)}")

async def main():
    """Main investigation function"""
    
    print("🔍 Document Tracking Investigation")
    print("=" * 80)
    print("Investigating how LightRAG knows which PDFs have been processed...")
    print("=" * 80)
    
    # Run all investigations
    await examine_tracking_mechanisms()
    await examine_component_tracking()
    await examine_ingestion_process()
    await examine_progress_tracking()
    await check_lightrag_internal_tracking()
    await test_duplicate_detection()
    
    # Summary and conclusions
    print("\n" + "=" * 80)
    print("🎯 DOCUMENT TRACKING ANALYSIS")
    print("=" * 80)
    
    print("Based on the investigation, here's how document tracking likely works:")
    print()
    print("📋 POTENTIAL TRACKING MECHANISMS:")
    print("   1. Progress Status File (./data/lightrag_cache/progress_status.json)")
    print("   2. LightRAG Internal Document Status (JsonDocStatus)")
    print("   3. Knowledge Graph Storage (document metadata)")
    print("   4. Component-level Statistics Tracking")
    print()
    print("🔍 KEY FINDINGS:")
    print("   • Progress tracking system maintains operation history")
    print("   • LightRAG library has JsonDocStatus for document tracking")
    print("   • Component statistics track ingestion counts")
    print("   • Working directories may contain document metadata")
    print()
    print("💡 LIKELY MECHANISM:")
    print("   The system probably uses file paths, modification times,")
    print("   or content hashes to identify already-processed documents")
    print("   and avoid reprocessing them unnecessarily.")

if __name__ == "__main__":
    asyncio.run(main())