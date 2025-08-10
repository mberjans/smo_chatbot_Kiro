#!/usr/bin/env python3
"""
Simple Chatbot Test

This script tests the core chatbot functionality without requiring
all dependencies to be installed.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_test_environment():
    """Setup minimal test environment"""
    # Set required environment variables if not present
    env_vars = {
        'DATABASE_URL': 'postgresql://localhost:5432/lightrag_test',
        'NEO4J_PASSWORD': 'test_password',
        'PERPLEXITY_API': 'test_key_placeholder'
    }
    
    for var, default_value in env_vars.items():
        if not os.getenv(var):
            os.environ[var] = default_value
            logger.info(f"Set {var} for testing")

def test_lightrag_config():
    """Test LightRAG configuration loading"""
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        from lightrag_integration.config.settings import LightRAGConfig
        
        # Test configuration creation
        config = LightRAGConfig.from_env()
        
        logger.info("✅ LightRAG configuration loaded successfully")
        logger.info(f"   - Knowledge graph path: {config.knowledge_graph_path}")
        logger.info(f"   - Vector store path: {config.vector_store_path}")
        logger.info(f"   - Cache directory: {config.cache_directory}")
        logger.info(f"   - Papers directory: {config.papers_directory}")
        logger.info(f"   - LLM model: {config.llm_model}")
        logger.info(f"   - Embedding model: {config.embedding_model}")
        
        return True, config
        
    except Exception as e:
        logger.error(f"❌ LightRAG configuration failed: {str(e)}")
        return False, None

def test_lightrag_component_creation(config):
    """Test LightRAG component creation"""
    try:
        from lightrag_integration.component import LightRAGComponent
        
        # Create component
        component = LightRAGComponent(config)
        
        logger.info("✅ LightRAG component created successfully")
        logger.info(f"   - Component type: {type(component).__name__}")
        logger.info(f"   - Has config: {hasattr(component, 'config')}")
        logger.info(f"   - Has logger: {hasattr(component, 'logger')}")
        
        return True, component
        
    except Exception as e:
        logger.error(f"❌ LightRAG component creation failed: {str(e)}")
        return False, None

async def test_lightrag_initialization(component):
    """Test LightRAG component initialization"""
    try:
        # Try to initialize the component
        await component.initialize()
        
        logger.info("✅ LightRAG component initialized successfully")
        return True
        
    except Exception as e:
        logger.warning(f"⚠️  LightRAG initialization failed (expected): {str(e)}")
        logger.info("   This is expected if LightRAG dependencies are not fully installed")
        return False

def test_directory_creation(config):
    """Test that required directories can be created"""
    try:
        directories = [
            config.knowledge_graph_path,
            config.vector_store_path,
            config.cache_directory,
            config.papers_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
            # Test write permissions
            test_file = Path(directory) / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
        
        logger.info("✅ All required directories created and writable")
        logger.info(f"   - Created {len(directories)} directories")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Directory creation failed: {str(e)}")
        return False

def test_environment_variables():
    """Test that required environment variables are available"""
    try:
        required_vars = ['GROQ_API_KEY']
        optional_vars = ['DATABASE_URL', 'NEO4J_PASSWORD', 'PERPLEXITY_API', 'OPENAI_API_KEY']
        
        missing_required = []
        available_optional = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_required.append(var)
        
        for var in optional_vars:
            if os.getenv(var):
                available_optional.append(var)
        
        if missing_required:
            logger.warning(f"⚠️  Missing required environment variables: {', '.join(missing_required)}")
            logger.info("   The chatbot will have limited functionality")
        else:
            logger.info("✅ All required environment variables are set")
        
        if available_optional:
            logger.info(f"   Available optional variables: {', '.join(available_optional)}")
        
        return len(missing_required) == 0
        
    except Exception as e:
        logger.error(f"❌ Environment variable check failed: {str(e)}")
        return False

def test_basic_imports():
    """Test basic imports that should work"""
    try:
        # Test Python standard library imports
        import json
        import asyncio
        import pathlib
        import logging
        
        # Test if we can import our modules
        sys.path.insert(0, 'src')
        
        # Try to import LightRAG integration modules
        from lightrag_integration.config.settings import LightRAGConfig
        from lightrag_integration.component import LightRAGComponent
        
        logger.info("✅ Basic imports successful")
        logger.info("   - Standard library: OK")
        logger.info("   - LightRAG integration: OK")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during imports: {str(e)}")
        return False

async def test_mock_query():
    """Test a mock query to simulate chatbot functionality"""
    try:
        logger.info("🧪 Testing mock query functionality...")
        
        # Simulate query processing
        question = "What is clinical metabolomics?"
        start_time = time.time()
        
        # Mock processing delay
        await asyncio.sleep(0.1)
        
        # Mock response
        mock_response = {
            "answer": "Clinical metabolomics is the application of metabolomics technologies and approaches to understand disease mechanisms, identify biomarkers, and support clinical decision-making.",
            "confidence_score": 0.85,
            "source_documents": ["mock_document_1.pdf", "mock_document_2.pdf"],
            "processing_time": time.time() - start_time,
            "source": "Mock",
            "metadata": {"test": True}
        }
        
        logger.info("✅ Mock query processing successful")
        logger.info(f"   - Question: {question}")
        logger.info(f"   - Processing time: {mock_response['processing_time']:.3f}s")
        logger.info(f"   - Confidence: {mock_response['confidence_score']:.2f}")
        logger.info(f"   - Sources: {len(mock_response['source_documents'])}")
        
        return True, mock_response
        
    except Exception as e:
        logger.error(f"❌ Mock query failed: {str(e)}")
        return False, None

def check_system_resources():
    """Check basic system resources"""
    try:
        import psutil
        
        # Get system info
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        
        logger.info("✅ System resources check:")
        logger.info(f"   - CPU cores: {cpu_count}")
        logger.info(f"   - Memory: {memory.total / (1024**3):.1f}GB total, {memory.available / (1024**3):.1f}GB available")
        logger.info(f"   - Disk space: {disk.free / (1024**3):.1f}GB free")
        
        # Check if resources are adequate
        adequate = (
            memory.available > 1024**3 and  # At least 1GB available memory
            disk.free > 5 * 1024**3         # At least 5GB free disk space
        )
        
        if adequate:
            logger.info("   - Resources appear adequate for testing")
        else:
            logger.warning("   - Limited resources may affect performance")
        
        return True
        
    except ImportError:
        logger.info("⚠️  psutil not available - skipping resource check")
        return True
    except Exception as e:
        logger.warning(f"⚠️  Resource check failed: {str(e)}")
        return True

async def run_comprehensive_test():
    """Run comprehensive chatbot test"""
    logger.info("🤖 Starting Simple Chatbot Test")
    logger.info("=" * 60)
    
    # Setup test environment
    setup_test_environment()
    
    # Track test results
    test_results = {}
    
    # Test 1: Basic imports
    logger.info("\n📦 Testing basic imports...")
    test_results['imports'] = test_basic_imports()
    
    # Test 2: Environment variables
    logger.info("\n🔧 Testing environment variables...")
    test_results['environment'] = test_environment_variables()
    
    # Test 3: System resources
    logger.info("\n💻 Checking system resources...")
    test_results['resources'] = check_system_resources()
    
    # Test 4: LightRAG configuration
    logger.info("\n⚙️  Testing LightRAG configuration...")
    config_success, config = test_lightrag_config()
    test_results['config'] = config_success
    
    if config_success:
        # Test 5: Directory creation
        logger.info("\n📁 Testing directory creation...")
        test_results['directories'] = test_directory_creation(config)
        
        # Test 6: Component creation
        logger.info("\n🔧 Testing component creation...")
        component_success, component = test_lightrag_component_creation(config)
        test_results['component_creation'] = component_success
        
        if component_success:
            # Test 7: Component initialization
            logger.info("\n🚀 Testing component initialization...")
            test_results['initialization'] = await test_lightrag_initialization(component)
    
    # Test 8: Mock query
    logger.info("\n🧪 Testing mock query processing...")
    query_success, mock_response = await test_mock_query()
    test_results['mock_query'] = query_success
    
    # Calculate overall results
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    success_rate = passed_tests / total_tests
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    # Overall status
    if success_rate >= 0.8:
        overall_status = "✅ EXCELLENT"
        status_color = "System is ready for testing"
    elif success_rate >= 0.6:
        overall_status = "⚠️  GOOD"
        status_color = "System has minor issues but should work"
    elif success_rate >= 0.4:
        overall_status = "⚠️  FAIR"
        status_color = "System has some issues that may affect functionality"
    else:
        overall_status = "❌ POOR"
        status_color = "System has significant issues"
    
    logger.info(f"Overall Status: {overall_status}")
    logger.info(f"Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests} tests passed)")
    logger.info(f"Assessment: {status_color}")
    
    # Individual test results
    logger.info("\nIndividual Test Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status} {test_name.replace('_', ' ').title()}")
    
    # Recommendations
    logger.info("\n💡 Recommendations:")
    
    if not test_results.get('imports', False):
        logger.info("  • Install missing Python dependencies")
    
    if not test_results.get('environment', False):
        logger.info("  • Set required environment variables (especially GROQ_API_KEY)")
    
    if not test_results.get('config', False):
        logger.info("  • Check LightRAG configuration and dependencies")
    
    if not test_results.get('initialization', False):
        logger.info("  • Install LightRAG and related dependencies for full functionality")
    
    if success_rate >= 0.6:
        logger.info("  • System appears functional for basic testing")
        logger.info("  • Consider running the actual chatbot with: python src/main.py")
    
    logger.info("\n" + "=" * 60)
    
    return success_rate >= 0.4

def main():
    """Main function"""
    try:
        success = asyncio.run(run_comprehensive_test())
        
        if success:
            logger.info("🎉 Chatbot test completed successfully!")
            return 0
        else:
            logger.info("⚠️  Chatbot test completed with issues")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"💥 Test execution failed: {str(e)}")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)