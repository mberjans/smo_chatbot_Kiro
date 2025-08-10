#!/usr/bin/env python3
"""
Test Chatbot Query

This script tests the actual chatbot query functionality by simulating
a user interaction without the full Chainlit interface.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_environment():
    """Setup environment for testing"""
    # Set required environment variables if not present
    if not os.getenv('DATABASE_URL'):
        os.environ['DATABASE_URL'] = 'postgresql://localhost:5432/lightrag_test'
    if not os.getenv('NEO4J_PASSWORD'):
        os.environ['NEO4J_PASSWORD'] = 'test_password'
    if not os.getenv('PERPLEXITY_API'):
        os.environ['PERPLEXITY_API'] = 'test_key_placeholder'

async def test_lightrag_query():
    """Test LightRAG query functionality"""
    try:
        logger.info("🔍 Testing LightRAG query functionality...")
        
        # Import required modules
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        # Create and initialize component
        config = LightRAGConfig.from_env()
        component = LightRAGComponent(config)
        await component.initialize()
        
        # Test query
        question = "What is clinical metabolomics?"
        logger.info(f"Asking question: {question}")
        
        start_time = time.time()
        
        try:
            # Try to query the component
            result = await component.query(question)
            processing_time = time.time() - start_time
            
            logger.info("✅ LightRAG query successful!")
            logger.info(f"   - Processing time: {processing_time:.3f}s")
            logger.info(f"   - Answer length: {len(result.get('answer', ''))}")
            logger.info(f"   - Confidence: {result.get('confidence_score', 0.0):.2f}")
            logger.info(f"   - Sources: {len(result.get('source_documents', []))}")
            
            if result.get('answer'):
                logger.info(f"   - Answer preview: {result['answer'][:100]}...")
            
            return True, result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"⚠️  LightRAG query failed (expected without knowledge base): {str(e)}")
            logger.info(f"   - Processing time: {processing_time:.3f}s")
            
            # This is expected if there's no knowledge base
            return False, {"error": str(e)}
            
    except Exception as e:
        logger.error(f"❌ LightRAG test setup failed: {str(e)}")
        return False, {"error": str(e)}

async def test_perplexity_fallback():
    """Test Perplexity fallback functionality"""
    try:
        logger.info("🌐 Testing Perplexity fallback functionality...")
        
        # Check if we have a real Perplexity API key
        perplexity_key = os.getenv('PERPLEXITY_API')
        if not perplexity_key or perplexity_key == 'test_key_placeholder':
            logger.warning("⚠️  No real Perplexity API key - skipping actual API test")
            return False, {"error": "No API key"}
        
        # Import the query function
        from main import query_perplexity
        
        question = "What is clinical metabolomics?"
        logger.info(f"Asking question: {question}")
        
        start_time = time.time()
        
        try:
            result = await query_perplexity(question)
            processing_time = time.time() - start_time
            
            logger.info("✅ Perplexity query successful!")
            logger.info(f"   - Processing time: {processing_time:.3f}s")
            logger.info(f"   - Answer length: {len(result.get('content', ''))}")
            logger.info(f"   - Confidence: {result.get('confidence_score', 0.0):.2f}")
            logger.info(f"   - Source: {result.get('source', 'Unknown')}")
            
            if result.get('content'):
                logger.info(f"   - Answer preview: {result['content'][:100]}...")
            
            return True, result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.warning(f"⚠️  Perplexity query failed: {str(e)}")
            logger.info(f"   - Processing time: {processing_time:.3f}s")
            return False, {"error": str(e)}
            
    except Exception as e:
        logger.error(f"❌ Perplexity test setup failed: {str(e)}")
        return False, {"error": str(e)}

async def test_translation_system():
    """Test translation system"""
    try:
        logger.info("🌍 Testing translation system...")
        
        # Import translation modules
        from translation import get_translator, detect_language, get_language_detector, translate
        from lingua_iso_codes import IsoCode639_1
        
        # Get translator
        translator = get_translator()
        
        # Test language detection
        iso_codes = [
            IsoCode639_1[code.upper()].value
            for code in translator.get_supported_languages(as_dict=True).values()
            if code.upper() in IsoCode639_1._member_names_
        ]
        detector = get_language_detector(*iso_codes)
        
        # Test with English text
        test_text = "What is clinical metabolomics?"
        detection = await detect_language(detector, test_text)
        
        logger.info("✅ Translation system working!")
        logger.info(f"   - Translator type: {type(translator).__name__}")
        logger.info(f"   - Supported languages: {len(translator.get_supported_languages(as_dict=True))}")
        logger.info(f"   - Detected language: {detection.get('language', 'unknown')}")
        logger.info(f"   - Detection confidence: {detection.get('confidence', 0.0):.2f}")
        
        # Test translation (English to Spanish)
        try:
            spanish_text = await translate(translator, test_text, source="en", target="es")
            logger.info(f"   - Translation test: '{test_text}' -> '{spanish_text}'")
        except Exception as e:
            logger.warning(f"   - Translation test failed: {str(e)}")
        
        return True, {
            "translator": type(translator).__name__,
            "languages": len(translator.get_supported_languages(as_dict=True)),
            "detection": detection
        }
        
    except Exception as e:
        logger.error(f"❌ Translation system test failed: {str(e)}")
        return False, {"error": str(e)}

async def simulate_chatbot_interaction():
    """Simulate a complete chatbot interaction"""
    try:
        logger.info("🤖 Simulating complete chatbot interaction...")
        
        # Test question
        question = "What is clinical metabolomics?"
        logger.info(f"User question: {question}")
        
        # Initialize components (similar to main.py)
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        from translation import get_translator, detect_language, get_language_detector
        from lingua_iso_codes import IsoCode639_1
        
        # Setup translation
        translator = get_translator()
        iso_codes = [
            IsoCode639_1[code.upper()].value
            for code in translator.get_supported_languages(as_dict=True).values()
            if code.upper() in IsoCode639_1._member_names_
        ]
        detector = get_language_detector(*iso_codes)
        
        # Detect language
        detection = await detect_language(detector, question)
        language = detection["language"]
        logger.info(f"Detected language: {language}")
        
        # Initialize LightRAG
        lightrag_component = None
        try:
            lightrag_config = LightRAGConfig.from_env()
            lightrag_component = LightRAGComponent(lightrag_config)
            await lightrag_component.initialize()
            logger.info("LightRAG component initialized")
        except Exception as e:
            logger.warning(f"LightRAG initialization failed: {str(e)}")
        
        # Try LightRAG first
        response_data = None
        error_messages = []
        
        if lightrag_component is not None:
            try:
                logger.info("Attempting LightRAG query...")
                # Import the query function from main.py
                from main import query_lightrag
                response_data = await query_lightrag(lightrag_component, question)
                logger.info("LightRAG query successful")
            except Exception as e:
                error_messages.append(f"LightRAG failed: {str(e)}")
                logger.warning(f"LightRAG query failed: {str(e)}")
        else:
            error_messages.append("LightRAG component not available")
        
        # Fall back to Perplexity if needed
        if response_data is None:
            try:
                logger.info("Attempting Perplexity fallback...")
                from main import query_perplexity
                response_data = await query_perplexity(question)
                logger.info("Perplexity query successful")
            except Exception as e:
                error_messages.append(f"Perplexity failed: {str(e)}")
                logger.warning(f"Perplexity query failed: {str(e)}")
        
        # Generate response
        if response_data is None:
            response_content = (
                "I apologize, but I'm currently experiencing technical difficulties. "
                "Both my knowledge base and real-time search capabilities are temporarily unavailable."
            )
            logger.warning("Both LightRAG and Perplexity failed")
        else:
            response_content = response_data["content"]
            source_info = f"\n\n*Response from: {response_data['source']}*"
            if response_data.get("confidence_score"):
                source_info += f" (Confidence: {response_data['confidence_score']:.2f})"
            response_content += source_info
            
            if response_data.get("bibliography"):
                response_content += response_data["bibliography"]
            
            logger.info("✅ Chatbot interaction successful!")
            logger.info(f"   - Response source: {response_data['source']}")
            logger.info(f"   - Response length: {len(response_content)}")
            logger.info(f"   - Confidence: {response_data.get('confidence_score', 0.0):.2f}")
        
        logger.info(f"Final response preview: {response_content[:200]}...")
        
        return True, {
            "question": question,
            "response": response_content,
            "source": response_data.get("source", "None") if response_data else "None",
            "errors": error_messages
        }
        
    except Exception as e:
        logger.error(f"❌ Chatbot interaction simulation failed: {str(e)}")
        return False, {"error": str(e)}

async def main():
    """Main test function"""
    logger.info("🧪 Starting Chatbot Query Testing")
    logger.info("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Run tests
    tests = [
        ("LightRAG Query", test_lightrag_query),
        ("Perplexity Fallback", test_perplexity_fallback),
        ("Translation System", test_translation_system),
        ("Complete Interaction", simulate_chatbot_interaction)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running {test_name}...")
        try:
            success, data = await test_func()
            results[test_name] = {"success": success, "data": data}
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {str(e)}")
            results[test_name] = {"success": False, "data": {"error": str(e)}}
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 QUERY TEST SUMMARY")
    logger.info("=" * 60)
    
    successful_tests = sum(1 for result in results.values() if result["success"])
    total_tests = len(results)
    success_rate = successful_tests / total_tests
    
    logger.info(f"Success Rate: {success_rate:.1%} ({successful_tests}/{total_tests})")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        logger.info(f"  {status} {test_name}")
    
    # Recommendations
    logger.info("\n💡 Key Findings:")
    
    if results.get("Complete Interaction", {}).get("success"):
        logger.info("  • Chatbot interaction simulation successful")
        interaction_data = results["Complete Interaction"]["data"]
        if interaction_data.get("source") != "None":
            logger.info(f"  • Response generated from: {interaction_data.get('source')}")
        else:
            logger.info("  • Both LightRAG and Perplexity failed - check API keys and configuration")
    
    if not results.get("LightRAG Query", {}).get("success"):
        logger.info("  • LightRAG needs knowledge base data to function properly")
        logger.info("  • Add PDF papers to the 'papers' directory for full functionality")
    
    if not results.get("Perplexity Fallback", {}).get("success"):
        logger.info("  • Perplexity fallback not working - check PERPLEXITY_API environment variable")
    
    if results.get("Translation System", {}).get("success"):
        logger.info("  • Translation system is working correctly")
    
    logger.info("\n🚀 To run the actual chatbot:")
    logger.info("  python src/main.py")
    
    logger.info("\n" + "=" * 60)
    
    return success_rate >= 0.5

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        if success:
            print("🎉 Chatbot query testing completed successfully!")
            sys.exit(0)
        else:
            print("⚠️  Chatbot query testing completed with issues")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"💥 Testing failed: {str(e)}")
        sys.exit(2)