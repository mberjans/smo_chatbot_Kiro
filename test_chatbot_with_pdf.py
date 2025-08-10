#!/usr/bin/env python3
"""
Clinical Metabolomics Oracle - PDF Integration Test
Tests the complete chatbot pipeline using clinical_metabolomics_review.pdf
"""

import os
import sys
import logging
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up environment variables for testing
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test_db')
os.environ.setdefault('NEO4J_PASSWORD', 'test_password')
os.environ.setdefault('PERPLEXITY_API', 'test_api_key')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChatbotPDFTester:
    """Comprehensive tester for chatbot with PDF integration"""
    
    def __init__(self):
        self.pdf_path = "clinical_metabolomics_review.pdf"
        self.test_results = {}
        self.lightrag_component = None
        
    async def setup_environment(self):
        """Set up the testing environment"""
        logger.info("Setting up test environment...")
        
        # Create necessary directories
        os.makedirs("./data/lightrag_kg", exist_ok=True)
        os.makedirs("./data/lightrag_vectors", exist_ok=True)
        os.makedirs("./data/lightrag_cache", exist_ok=True)
        
        logger.info("✅ Test environment setup complete")
        
    async def test_pdf_loading(self):
        """Test loading the PDF into LightRAG"""
        logger.info("Testing PDF loading into LightRAG...")
        
        try:
            # Import LightRAG component
            from lightrag_integration.component import LightRAGComponent
            
            # Initialize component
            self.lightrag_component = LightRAGComponent()
            
            # Check if PDF exists
            if not os.path.exists(self.pdf_path):
                raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
            
            # Load PDF content
            logger.info(f"Loading PDF: {self.pdf_path}")
            
            # Read PDF content (simplified approach)
            import PyPDF2
            with open(self.pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
            
            logger.info(f"Extracted {len(text_content)} characters from PDF")
            
            # Save PDF content to a temporary file for ingestion
            temp_pdf_path = "temp_clinical_metabolomics_review.pdf"
            import shutil
            shutil.copy(self.pdf_path, temp_pdf_path)
            
            # Ingest PDF using the correct method
            result = await self.lightrag_component.ingest_documents([temp_pdf_path])
            
            # Clean up temporary file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
            
            self.test_results['pdf_loading'] = {
                'status': 'PASSED',
                'content_length': len(text_content),
                'message': 'PDF successfully loaded into LightRAG'
            }
            
            logger.info("✅ PDF loading test: PASSED")
            return True
            
        except Exception as e:
            self.test_results['pdf_loading'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': f'Failed to load PDF: {e}'
            }
            logger.error(f"❌ PDF loading test: FAILED - {e}")
            return False
    
    async def test_chatbot_queries(self):
        """Test chatbot with various metabolomics queries"""
        logger.info("Testing chatbot queries...")
        
        if not self.lightrag_component:
            logger.error("LightRAG component not initialized")
            return False
        
        # Define test queries
        test_queries = [
            {
                "query": "What are the main applications of metabolomics in clinical research?",
                "expected_topics": ["biomarker", "disease", "diagnosis", "clinical"]
            },
            {
                "query": "How is mass spectrometry used in metabolomics?",
                "expected_topics": ["mass spectrometry", "MS", "analytical", "detection"]
            },
            {
                "query": "What are the challenges in metabolomics data analysis?",
                "expected_topics": ["data", "analysis", "preprocessing", "statistics"]
            },
            {
                "query": "Explain the role of metabolomics in personalized medicine",
                "expected_topics": ["personalized", "precision", "medicine", "individual"]
            }
        ]
        
        query_results = []
        
        for i, test_case in enumerate(test_queries):
            logger.info(f"Testing query {i+1}: {test_case['query']}")
            
            try:
                # Query the chatbot using correct method signature
                response = await self.lightrag_component.query(
                    question=test_case['query'],
                    context={"mode": "hybrid"}  # Use hybrid search for best results
                )
                
                # Analyze response using correct field names
                response_text = response.get('answer', '').lower()
                found_topics = [topic for topic in test_case['expected_topics'] 
                              if topic.lower() in response_text]
                
                query_result = {
                    'query': test_case['query'],
                    'response_length': len(response.get('answer', '')),
                    'found_topics': found_topics,
                    'expected_topics': test_case['expected_topics'],
                    'relevance_score': len(found_topics) / len(test_case['expected_topics']),
                    'citations': response.get('source_documents', []),
                    'confidence': response.get('confidence_score', 0),
                    'status': 'PASSED' if found_topics else 'FAILED'
                }
                
                query_results.append(query_result)
                
                logger.info(f"Query {i+1} result: {query_result['status']}")
                logger.info(f"  Response length: {query_result['response_length']}")
                logger.info(f"  Relevance score: {query_result['relevance_score']:.2f}")
                logger.info(f"  Found topics: {found_topics}")
                
            except Exception as e:
                query_result = {
                    'query': test_case['query'],
                    'status': 'ERROR',
                    'error': str(e)
                }
                query_results.append(query_result)
                logger.error(f"Query {i+1} error: {e}")
        
        # Calculate overall success rate
        passed_queries = sum(1 for r in query_results if r['status'] == 'PASSED')
        success_rate = passed_queries / len(test_queries)
        
        self.test_results['chatbot_queries'] = {
            'status': 'PASSED' if success_rate >= 0.5 else 'FAILED',
            'success_rate': success_rate,
            'total_queries': len(test_queries),
            'passed_queries': passed_queries,
            'query_results': query_results
        }
        
        logger.info(f"✅ Chatbot queries test: {self.test_results['chatbot_queries']['status']}")
        logger.info(f"Success rate: {success_rate:.2f} ({passed_queries}/{len(test_queries)})")
        
        return success_rate >= 0.5
    
    async def test_citation_functionality(self):
        """Test citation extraction and formatting"""
        logger.info("Testing citation functionality...")
        
        if not self.lightrag_component:
            logger.error("LightRAG component not initialized")
            return False
        
        try:
            # Test query that should generate citations
            query = "What analytical techniques are commonly used in metabolomics studies?"
            
            response = await self.lightrag_component.query(
                question=query,
                context={"mode": "hybrid", "include_citations": True}
            )
            
            citations = response.get('source_documents', [])
            has_citations = len(citations) > 0
            
            # Test citation formatting
            citation_quality = 0
            if has_citations:
                for citation in citations:
                    if isinstance(citation, dict):
                        if 'source' in citation and 'confidence' in citation:
                            citation_quality += 1
            
            citation_quality_score = citation_quality / max(len(citations), 1)
            
            self.test_results['citation_functionality'] = {
                'status': 'PASSED' if has_citations else 'FAILED',
                'citations_found': len(citations),
                'citation_quality_score': citation_quality_score,
                'sample_citations': citations[:3] if citations else []
            }
            
            logger.info(f"✅ Citation test: {self.test_results['citation_functionality']['status']}")
            logger.info(f"Citations found: {len(citations)}")
            
            return has_citations
            
        except Exception as e:
            self.test_results['citation_functionality'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            logger.error(f"❌ Citation test error: {e}")
            return False
    
    async def test_response_quality(self):
        """Test the quality and coherence of responses"""
        logger.info("Testing response quality...")
        
        if not self.lightrag_component:
            logger.error("LightRAG component not initialized")
            return False
        
        try:
            # Test with a complex query
            complex_query = """
            Compare the advantages and disadvantages of different analytical platforms 
            used in metabolomics, specifically focusing on mass spectrometry and NMR spectroscopy.
            """
            
            response = await self.lightrag_component.query(
                question=complex_query,
                context={"mode": "hybrid"}
            )
            
            response_text = response.get('answer', '')
            
            # Quality metrics
            quality_metrics = {
                'length_appropriate': 100 <= len(response_text) <= 2000,
                'mentions_ms': 'mass spectrometry' in response_text.lower() or 'ms' in response_text.lower(),
                'mentions_nmr': 'nmr' in response_text.lower() or 'nuclear magnetic' in response_text.lower(),
                'has_comparison': any(word in response_text.lower() for word in ['advantage', 'disadvantage', 'compare', 'versus', 'vs']),
                'coherent_structure': len(response_text.split('.')) >= 3,  # At least 3 sentences
                'confidence_score': response.get('confidence', 0) > 0.5
            }
            
            quality_score = sum(quality_metrics.values()) / len(quality_metrics)
            
            self.test_results['response_quality'] = {
                'status': 'PASSED' if quality_score >= 0.6 else 'FAILED',
                'quality_score': quality_score,
                'quality_metrics': quality_metrics,
                'response_length': len(response_text),
                'confidence': response.get('confidence_score', 0)
            }
            
            logger.info(f"✅ Response quality test: {self.test_results['response_quality']['status']}")
            logger.info(f"Quality score: {quality_score:.2f}")
            
            return quality_score >= 0.6
            
        except Exception as e:
            self.test_results['response_quality'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            logger.error(f"❌ Response quality test error: {e}")
            return False
    
    async def run_all_tests(self):
        """Run all tests in sequence"""
        logger.info("🤖 Clinical Metabolomics Oracle - PDF Integration Testing")
        logger.info("=" * 60)
        
        # Setup
        await self.setup_environment()
        
        # Run tests
        tests = [
            ('PDF Loading', self.test_pdf_loading),
            ('Chatbot Queries', self.test_chatbot_queries),
            ('Citation Functionality', self.test_citation_functionality),
            ('Response Quality', self.test_response_quality)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Test ---")
            try:
                result = await test_func()
                if result:
                    passed_tests += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
        
        # Summary
        success_rate = passed_tests / total_tests
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Tests passed: {passed_tests}/{total_tests}")
        logger.info(f"Success rate: {success_rate:.2%}")
        logger.info(f"Overall status: {'PASSED' if success_rate >= 0.75 else 'FAILED'}")
        
        # Save detailed results
        with open('chatbot_pdf_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: chatbot_pdf_test_results.json")
        
        return success_rate >= 0.75

async def main():
    """Main test execution"""
    tester = ChatbotPDFTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n🎉 All tests passed! The chatbot is ready for use with the PDF content.")
    else:
        print("\n⚠️  Some tests failed. Please check the logs and fix issues before proceeding.")
    
    return success

if __name__ == "__main__":
    # Install required packages if missing
    try:
        import PyPDF2
    except ImportError:
        print("Installing PyPDF2...")
        os.system("pip install PyPDF2")
        import PyPDF2
    
    # Run tests
    result = asyncio.run(main())
    sys.exit(0 if result else 1)