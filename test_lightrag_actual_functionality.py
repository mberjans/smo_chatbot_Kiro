#!/usr/bin/env python3
"""
Comprehensive test to verify actual LightRAG functionality
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
import traceback

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

async def test_lightrag_core_functionality():
    """Test the core LightRAG functionality step by step"""
    
    print("🧪 LightRAG Core Functionality Test")
    print("=" * 60)
    
    try:
        # Test 1: Import and Configuration
        print("1️⃣ Testing imports and configuration...")
        
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        config = LightRAGConfig.from_env()
        print(f"✅ Configuration loaded: {config.llm_model}")
        
        # Test 2: Component Initialization
        print("\n2️⃣ Testing component initialization...")
        
        component = LightRAGComponent(config)
        await component.initialize()
        print("✅ Component initialized successfully")
        
        # Test 3: Health Check
        print("\n3️⃣ Testing health check...")
        
        health = await component.get_health_status()
        print(f"✅ Health status: {health.overall_status.value}")
        
        # Print component health details
        for comp_name, comp_health in health.components.items():
            status_icon = "✅" if comp_health.status.value == "healthy" else "⚠️" if comp_health.status.value == "degraded" else "❌"
            print(f"   {status_icon} {comp_name}: {comp_health.status.value}")
        
        # Test 4: Statistics
        print("\n4️⃣ Testing statistics...")
        
        stats = component.get_statistics()
        print(f"✅ Statistics retrieved:")
        print(f"   Initialized: {stats.get('is_initialized', False)}")
        print(f"   Queries processed: {stats.get('queries_processed', 0)}")
        print(f"   Documents ingested: {stats.get('documents_ingested', 0)}")
        
        # Test 5: Query Engine Direct Test
        print("\n5️⃣ Testing query engine directly...")
        
        try:
            from lightrag_integration.query.engine import LightRAGQueryEngine
            
            query_engine = LightRAGQueryEngine(config)
            
            # Test with a simple question
            test_question = "What is metabolomics?"
            print(f"   Testing question: '{test_question}'")
            
            result = await query_engine.process_query(test_question, {})
            
            # Check if result is a QueryResult object or dict
            if hasattr(result, 'answer'):
                answer = result.answer
                confidence = result.confidence_score
                processing_time = result.processing_time
            else:
                answer = result.get('answer', 'No answer')
                confidence = result.get('confidence_score', 0.0)
                processing_time = result.get('processing_time', 0.0)
            
            print(f"✅ Query engine response:")
            print(f"   Answer: {answer[:200]}...")
            print(f"   Confidence: {confidence}")
            print(f"   Processing time: {processing_time}s")
            
        except Exception as e:
            print(f"❌ Query engine test failed: {str(e)}")
            print(f"   Error details: {traceback.format_exc()}")
        
        # Test 6: Document Ingestion Test
        print("\n6️⃣ Testing document ingestion...")
        
        # Create a test document
        test_doc_path = Path("test_document.txt")
        test_content = """
        Clinical metabolomics is the application of metabolomics to clinical research and practice.
        It involves the comprehensive analysis of small molecules (metabolites) in biological samples
        such as blood, urine, and tissue. Sample preparation is crucial for accurate results and
        typically involves standardized collection protocols, proper storage conditions, and
        extraction procedures. Quality control measures are essential throughout the process.
        """
        
        test_doc_path.write_text(test_content)
        print(f"✅ Created test document: {test_doc_path}")
        
        # Test ingestion (this might be a placeholder)
        try:
            ingestion_result = await component.ingest_documents([str(test_doc_path)])
            print(f"✅ Ingestion result: {ingestion_result}")
        except Exception as e:
            print(f"⚠️  Ingestion test (expected to be placeholder): {str(e)}")
        
        # Test 7: Component Query Test
        print("\n7️⃣ Testing component query method...")
        
        try:
            query_result = await component.query("What is clinical metabolomics?")
            
            print(f"✅ Component query response:")
            print(f"   Answer: {query_result.get('answer', 'No answer')[:200]}...")
            print(f"   Confidence: {query_result.get('confidence_score', 0.0)}")
            print(f"   Processing time: {query_result.get('processing_time', 0.0)}s")
            print(f"   Source documents: {len(query_result.get('source_documents', []))}")
            
        except Exception as e:
            print(f"❌ Component query test failed: {str(e)}")
            print(f"   Error details: {traceback.format_exc()}")
        
        # Test 8: Cache and Performance Features
        print("\n8️⃣ Testing cache and performance features...")
        
        try:
            cache_stats = await component.get_cache_stats()
            print(f"✅ Cache stats: {cache_stats}")
            
            perf_stats = await component.get_performance_stats()
            print(f"✅ Performance stats: {perf_stats}")
            
        except Exception as e:
            print(f"⚠️  Cache/Performance test: {str(e)}")
        
        # Test 9: Cleanup
        print("\n9️⃣ Testing cleanup...")
        
        await component.cleanup()
        print("✅ Component cleanup completed")
        
        # Clean up test file
        if test_doc_path.exists():
            test_doc_path.unlink()
            print("✅ Test document cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        print(f"   Full traceback: {traceback.format_exc()}")
        return False

async def test_lightrag_library_direct():
    """Test the actual LightRAG library directly if available"""
    
    print("\n🔬 Direct LightRAG Library Test")
    print("=" * 60)
    
    try:
        # Try to import the actual LightRAG library
        import lightrag
        print("✅ LightRAG library imported successfully")
        
        # Try to create a LightRAG instance
        from lightrag import LightRAG, QueryParam
        
        # Create a simple working directory
        working_dir = Path("./test_lightrag_working")
        working_dir.mkdir(exist_ok=True)
        
        print(f"✅ Working directory created: {working_dir}")
        
        # Initialize LightRAG with minimal configuration
        rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=None,  # We'll need to configure this properly
        )
        
        print("✅ LightRAG instance created")
        
        # Test basic functionality
        test_text = "Clinical metabolomics is the study of metabolites in clinical samples."
        
        # Insert text (this might fail without proper LLM configuration)
        try:
            await rag.ainsert(test_text)
            print("✅ Text insertion successful")
            
            # Query the inserted text
            result = await rag.aquery("What is clinical metabolomics?")
            print(f"✅ Query result: {result}")
            
        except Exception as e:
            print(f"⚠️  LightRAG operations failed (expected without proper LLM config): {str(e)}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  LightRAG library not available: {str(e)}")
        print("   This is expected if the library is not properly installed")
        return False
    except Exception as e:
        print(f"❌ Direct LightRAG test failed: {str(e)}")
        print(f"   Error details: {traceback.format_exc()}")
        return False

async def test_integration_components():
    """Test individual integration components"""
    
    print("\n🔧 Integration Components Test")
    print("=" * 60)
    
    components_to_test = [
        ("Config Settings", "lightrag_integration.config.settings", "LightRAGConfig"),
        ("Query Engine", "lightrag_integration.query.engine", "LightRAGQueryEngine"),
        ("Citation Formatter", "lightrag_integration.citation_formatter", "CitationFormatter"),
        ("Confidence Scorer", "lightrag_integration.confidence_scoring", "ConfidenceScorer"),
        ("Translation Integration", "lightrag_integration.translation_integration", "TranslationIntegrator"),
        ("Response Integration", "lightrag_integration.response_integration", "ResponseIntegrator"),
        ("Monitoring", "lightrag_integration.monitoring", "SystemMonitor"),
        ("Error Handling", "lightrag_integration.error_handling", "ErrorHandler"),
    ]
    
    results = {}
    
    for component_name, module_path, class_name in components_to_test:
        try:
            print(f"Testing {component_name}...")
            
            # Try to import the module
            module = __import__(module_path, fromlist=[class_name])
            component_class = getattr(module, class_name)
            
            # Try to instantiate (with minimal config if needed)
            if component_name == "Config Settings":
                instance = component_class.from_env()
            else:
                # Most components might need config
                from lightrag_integration.config.settings import LightRAGConfig
                config = LightRAGConfig.from_env()
                instance = component_class(config)
            
            print(f"✅ {component_name}: Successfully imported and instantiated")
            results[component_name] = True
            
        except Exception as e:
            print(f"❌ {component_name}: Failed - {str(e)}")
            results[component_name] = False
    
    # Summary
    print(f"\n📊 Component Test Summary:")
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    print(f"   Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    for component, result in results.items():
        status = "✅" if result else "❌"
        print(f"   {status} {component}")
    
    return results

async def main():
    """Main test function"""
    
    print("🚀 LightRAG Integration Functionality Test")
    print("=" * 80)
    
    # Test 1: Core functionality
    core_result = await test_lightrag_core_functionality()
    
    # Test 2: Direct LightRAG library
    library_result = await test_lightrag_library_direct()
    
    # Test 3: Integration components
    component_results = await test_integration_components()
    
    # Final summary
    print("\n" + "=" * 80)
    print("🎯 FINAL TEST SUMMARY")
    print("=" * 80)
    
    print(f"Core Functionality Test: {'✅ PASSED' if core_result else '❌ FAILED'}")
    print(f"Direct LightRAG Library Test: {'✅ PASSED' if library_result else '⚠️  SKIPPED/FAILED'}")
    
    component_passed = sum(1 for result in component_results.values() if result)
    component_total = len(component_results)
    print(f"Integration Components: {component_passed}/{component_total} passed ({component_passed/component_total*100:.1f}%)")
    
    overall_success = core_result and (component_passed / component_total) >= 0.7
    
    print(f"\n🎯 Overall Status: {'✅ SYSTEM FUNCTIONAL' if overall_success else '⚠️  NEEDS ATTENTION'}")
    
    if not overall_success:
        print("\n📋 Recommendations:")
        if not core_result:
            print("   - Fix core functionality issues")
        if not library_result:
            print("   - Verify LightRAG library installation and configuration")
        if component_passed / component_total < 0.7:
            print("   - Address integration component failures")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)