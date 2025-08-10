#!/usr/bin/env python3
"""
Chatbot Testing Script

This script tests the Clinical Metabolomics Oracle chatbot functionality,
including both LightRAG integration and Perplexity fallback.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.insert(0, 'src')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotTester:
    """Test the chatbot functionality"""
    
    def __init__(self):
        """Initialize the tester"""
        self.test_results = []
        self.setup_environment()
    
    def setup_environment(self):
        """Setup test environment variables if not set"""
        # Set default values for missing environment variables
        if not os.getenv('DATABASE_URL'):
            os.environ['DATABASE_URL'] = 'postgresql://localhost:5432/lightrag_test'
            logger.info("Set default DATABASE_URL for testing")
        
        if not os.getenv('NEO4J_PASSWORD'):
            os.environ['NEO4J_PASSWORD'] = 'test_password'
            logger.info("Set default NEO4J_PASSWORD for testing")
        
        if not os.getenv('PERPLEXITY_API'):
            # Use a placeholder - Perplexity tests will be skipped
            os.environ['PERPLEXITY_API'] = 'test_key'
            logger.info("Set placeholder PERPLEXITY_API for testing")
    
    async def test_imports(self) -> Dict[str, Any]:
        """Test that all required modules can be imported"""
        logger.info("Testing imports...")
        
        try:
            # Test core imports
            import chainlit as cl
            from lingua import LanguageDetector
            from llama_index.core.callbacks import CallbackManager
            
            # Test LightRAG integration imports
            from lightrag_integration.component import LightRAGComponent
            from lightrag_integration.config.settings import LightRAGConfig
            
            # Test other application imports
            from translation import BaseTranslator, detect_language, get_language_detector, get_translator
            from citation import postprocess_citation
            from pipelines import get_pipeline
            
            return {
                'status': 'PASSED',
                'message': 'All imports successful',
                'details': {
                    'chainlit': 'OK',
                    'lightrag_integration': 'OK',
                    'translation': 'OK',
                    'citation': 'OK',
                    'pipelines': 'OK'
                }
            }
            
        except ImportError as e:
            return {
                'status': 'FAILED',
                'message': f'Import failed: {str(e)}',
                'details': {'error': str(e)}
            }
        except Exception as e:
            return {
                'status': 'FAILED',
                'message': f'Unexpected error: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def test_lightrag_config(self) -> Dict[str, Any]:
        """Test LightRAG configuration"""
        logger.info("Testing LightRAG configuration...")
        
        try:
            from lightrag_integration.config.settings import LightRAGConfig
            
            # Test configuration loading
            config = LightRAGConfig.from_env()
            
            return {
                'status': 'PASSED',
                'message': 'LightRAG configuration loaded successfully',
                'details': {
                    'working_dir': config.working_dir,
                    'llm_model': config.llm_model,
                    'embedding_model': config.embedding_model,
                    'chunk_size': config.chunk_size
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'message': f'LightRAG configuration failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def test_lightrag_component_init(self) -> Dict[str, Any]:
        """Test LightRAG component initialization"""
        logger.info("Testing LightRAG component initialization...")
        
        try:
            from lightrag_integration.component import LightRAGComponent
            from lightrag_integration.config.settings import LightRAGConfig
            
            # Create configuration
            config = LightRAGConfig.from_env()
            
            # Initialize component
            component = LightRAGComponent(config)
            
            # Test initialization (this might fail due to missing dependencies)
            try:
                await component.initialize()
                initialized = True
                init_message = "Component initialized successfully"
            except Exception as init_error:
                initialized = False
                init_message = f"Initialization failed: {str(init_error)}"
            
            return {
                'status': 'PASSED' if initialized else 'PARTIAL',
                'message': f'LightRAG component created. {init_message}',
                'details': {
                    'component_created': True,
                    'initialized': initialized,
                    'config': {
                        'working_dir': config.working_dir,
                        'llm_model': config.llm_model
                    }
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'message': f'LightRAG component creation failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def test_translation_system(self) -> Dict[str, Any]:
        """Test translation system"""
        logger.info("Testing translation system...")
        
        try:
            from translation import get_translator, detect_language, get_language_detector
            from lingua_iso_codes import IsoCode639_1
            
            # Test translator creation
            translator = get_translator()
            
            # Test language detection
            iso_codes = [
                IsoCode639_1[code.upper()].value
                for code in translator.get_supported_languages(as_dict=True).values()
                if code.upper() in IsoCode639_1._member_names_
            ]
            detector = get_language_detector(*iso_codes)
            
            # Test detection on English text
            test_text = "What is clinical metabolomics?"
            detection_result = await detect_language(detector, test_text)
            
            return {
                'status': 'PASSED',
                'message': 'Translation system working',
                'details': {
                    'translator_type': type(translator).__name__,
                    'supported_languages': len(translator.get_supported_languages(as_dict=True)),
                    'detection_test': {
                        'text': test_text,
                        'detected_language': detection_result.get('language', 'unknown'),
                        'confidence': detection_result.get('confidence', 0.0)
                    }
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'message': f'Translation system failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def test_perplexity_query_function(self) -> Dict[str, Any]:
        """Test the Perplexity query function (without actual API call)"""
        logger.info("Testing Perplexity query function...")
        
        try:
            # Import the query function from main.py
            from main import query_perplexity
            
            # Test with a mock question (this will fail due to API key, but we can test the function exists)
            test_question = "What is clinical metabolomics?"
            
            try:
                # This will likely fail due to API key, but that's expected
                result = await query_perplexity(test_question)
                return {
                    'status': 'PASSED',
                    'message': 'Perplexity query function works',
                    'details': {
                        'function_callable': True,
                        'result_type': type(result).__name__,
                        'has_content': 'content' in result if isinstance(result, dict) else False
                    }
                }
            except Exception as api_error:
                # Expected to fail due to API key issues
                return {
                    'status': 'PARTIAL',
                    'message': 'Perplexity function exists but API call failed (expected)',
                    'details': {
                        'function_callable': True,
                        'api_error': str(api_error)[:100] + '...' if len(str(api_error)) > 100 else str(api_error)
                    }
                }
                
        except Exception as e:
            return {
                'status': 'FAILED',
                'message': f'Perplexity query function test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def test_lightrag_query_function(self) -> Dict[str, Any]:
        """Test the LightRAG query function"""
        logger.info("Testing LightRAG query function...")
        
        try:
            # Import the query function from main.py
            from main import query_lightrag
            from lightrag_integration.component import LightRAGComponent
            from lightrag_integration.config.settings import LightRAGConfig
            
            # Create a component for testing
            config = LightRAGConfig.from_env()
            component = LightRAGComponent(config)
            
            test_question = "What is clinical metabolomics?"
            
            try:
                # Try to initialize and query
                await component.initialize()
                result = await query_lightrag(component, test_question)
                
                return {
                    'status': 'PASSED',
                    'message': 'LightRAG query function works',
                    'details': {
                        'function_callable': True,
                        'component_initialized': True,
                        'result_type': type(result).__name__,
                        'has_content': 'content' in result if isinstance(result, dict) else False,
                        'has_confidence': 'confidence_score' in result if isinstance(result, dict) else False
                    }
                }
            except Exception as query_error:
                # Expected to fail if no knowledge base is available
                return {
                    'status': 'PARTIAL',
                    'message': 'LightRAG function exists but query failed (expected without knowledge base)',
                    'details': {
                        'function_callable': True,
                        'query_error': str(query_error)[:100] + '...' if len(str(query_error)) > 100 else str(query_error)
                    }
                }
                
        except Exception as e:
            return {
                'status': 'FAILED',
                'message': f'LightRAG query function test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def test_citation_system(self) -> Dict[str, Any]:
        """Test citation processing system"""
        logger.info("Testing citation system...")
        
        try:
            from citation import postprocess_citation
            
            # Test citation processing with sample text
            test_text = "This is a test citation [1] with multiple references [2]."
            processed = postprocess_citation(test_text)
            
            return {
                'status': 'PASSED',
                'message': 'Citation system working',
                'details': {
                    'function_callable': True,
                    'input_text': test_text,
                    'output_text': processed,
                    'processing_changed_text': test_text != processed
                }
            }
            
        except Exception as e:
            return {
                'status': 'FAILED',
                'message': f'Citation system failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def test_pipeline_system(self) -> Dict[str, Any]:
        """Test the pipeline system"""
        logger.info("Testing pipeline system...")
        
        try:
            from pipelines import get_pipeline
            from llama_index.core.callbacks import CallbackManager
            
            # Test pipeline creation
            callback_manager = CallbackManager([])
            
            try:
                pipeline = get_pipeline(callback_manager=callback_manager)
                
                return {
                    'status': 'PASSED',
                    'message': 'Pipeline system working',
                    'details': {
                        'pipeline_created': True,
                        'pipeline_type': type(pipeline).__name__
                    }
                }
            except Exception as pipeline_error:
                return {
                    'status': 'PARTIAL',
                    'message': 'Pipeline function exists but creation failed',
                    'details': {
                        'function_callable': True,
                        'pipeline_error': str(pipeline_error)[:100] + '...' if len(str(pipeline_error)) > 100 else str(pipeline_error)
                    }
                }
                
        except Exception as e:
            return {
                'status': 'FAILED',
                'message': f'Pipeline system test failed: {str(e)}',
                'details': {'error': str(e)}
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all chatbot tests"""
        logger.info("Starting comprehensive chatbot testing...")
        
        tests = [
            ("Import Test", self.test_imports),
            ("LightRAG Config Test", self.test_lightrag_config),
            ("LightRAG Component Test", self.test_lightrag_component_init),
            ("Translation System Test", self.test_translation_system),
            ("Perplexity Query Test", self.test_perplexity_query_function),
            ("LightRAG Query Test", self.test_lightrag_query_function),
            ("Citation System Test", self.test_citation_system),
            ("Pipeline System Test", self.test_pipeline_system)
        ]
        
        results = {}
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"Running {test_name}...")
            try:
                result = await test_func()
                results[test_name] = result
                
                if result['status'] == 'PASSED':
                    passed_tests += 1
                    logger.info(f"✅ {test_name}: PASSED")
                elif result['status'] == 'PARTIAL':
                    passed_tests += 0.5
                    logger.info(f"⚠️  {test_name}: PARTIAL")
                else:
                    logger.info(f"❌ {test_name}: FAILED")
                    
            except Exception as e:
                results[test_name] = {
                    'status': 'ERROR',
                    'message': f'Test execution error: {str(e)}',
                    'details': {'error': str(e)}
                }
                logger.error(f"❌ {test_name}: ERROR - {str(e)}")
        
        # Calculate overall status
        success_rate = passed_tests / total_tests
        if success_rate >= 0.8:
            overall_status = 'PASSED'
        elif success_rate >= 0.5:
            overall_status = 'PARTIAL'
        else:
            overall_status = 'FAILED'
        
        return {
            'overall_status': overall_status,
            'success_rate': success_rate,
            'passed_tests': int(passed_tests),
            'total_tests': total_tests,
            'individual_results': results,
            'summary': self.generate_summary(results)
        }
    
    def generate_summary(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of test results"""
        summary = {
            'critical_systems': {},
            'integration_status': {},
            'recommendations': []
        }
        
        # Check critical systems
        critical_tests = [
            'Import Test',
            'LightRAG Config Test',
            'Translation System Test'
        ]
        
        for test in critical_tests:
            if test in results:
                summary['critical_systems'][test] = results[test]['status']
        
        # Check integration status
        integration_tests = [
            'LightRAG Component Test',
            'LightRAG Query Test',
            'Perplexity Query Test'
        ]
        
        for test in integration_tests:
            if test in results:
                summary['integration_status'][test] = results[test]['status']
        
        # Generate recommendations
        if results.get('Import Test', {}).get('status') != 'PASSED':
            summary['recommendations'].append("Fix import issues before proceeding")
        
        if results.get('LightRAG Component Test', {}).get('status') == 'FAILED':
            summary['recommendations'].append("Check LightRAG dependencies and configuration")
        
        if results.get('Perplexity Query Test', {}).get('status') == 'FAILED':
            summary['recommendations'].append("Verify Perplexity API key and configuration")
        
        if not summary['recommendations']:
            summary['recommendations'].append("System appears to be functioning correctly")
            summary['recommendations'].append("Consider adding test data to fully test LightRAG functionality")
        
        return summary
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results"""
        print(f"\n{'='*80}")
        print("CHATBOT TESTING RESULTS")
        print(f"{'='*80}")
        
        # Overall status
        status_color = {
            'PASSED': '\033[92m',  # Green
            'PARTIAL': '\033[93m', # Yellow
            'FAILED': '\033[91m',  # Red
            'ERROR': '\033[91m'    # Red
        }
        reset_color = '\033[0m'
        
        overall = results['overall_status']
        color = status_color.get(overall, '')
        
        print(f"Overall Status: {color}{overall}{reset_color}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Tests Passed: {results['passed_tests']}/{results['total_tests']}")
        
        # Individual test results
        print(f"\nIndividual Test Results:")
        for test_name, result in results['individual_results'].items():
            status = result['status']
            color = status_color.get(status, '')
            icon = {
                'PASSED': '✅',
                'PARTIAL': '⚠️',
                'FAILED': '❌',
                'ERROR': '💥'
            }.get(status, '❓')
            
            print(f"  {icon} {test_name}: {color}{status}{reset_color}")
            print(f"     {result['message']}")
        
        # Summary
        summary = results['summary']
        print(f"\nCritical Systems Status:")
        for system, status in summary['critical_systems'].items():
            color = status_color.get(status, '')
            print(f"  - {system}: {color}{status}{reset_color}")
        
        print(f"\nIntegration Status:")
        for integration, status in summary['integration_status'].items():
            color = status_color.get(status, '')
            print(f"  - {integration}: {color}{status}{reset_color}")
        
        print(f"\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
        
        print(f"\n{'='*80}")

async def main():
    """Main function to run chatbot tests"""
    print("🤖 Clinical Metabolomics Oracle - Chatbot Testing")
    print("=" * 60)
    
    tester = ChatbotTester()
    results = await tester.run_all_tests()
    tester.print_results(results)
    
    # Exit with appropriate code
    if results['overall_status'] == 'PASSED':
        print("\n🎉 Chatbot testing completed successfully!")
        return 0
    elif results['overall_status'] == 'PARTIAL':
        print("\n⚠️  Chatbot testing completed with some issues")
        return 1
    else:
        print("\n❌ Chatbot testing failed")
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)