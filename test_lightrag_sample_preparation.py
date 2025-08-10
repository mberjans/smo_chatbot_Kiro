#!/usr/bin/env python3
"""
Test LightRAG with specific question about sample preparation methods
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up environment variables if not already set
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test_db')
os.environ.setdefault('NEO4J_PASSWORD', 'test_password')
os.environ.setdefault('PERPLEXITY_API', 'test_api_key_placeholder')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test_key_placeholder')
os.environ.setdefault('GROQ_API_KEY', 'test_groq_key')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_lightrag_query():
    """Test LightRAG with the specific sample preparation question"""
    
    print("🧪 Testing LightRAG with Sample Preparation Question")
    print("=" * 60)
    
    try:
        # Import LightRAG components
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        print("✅ LightRAG imports successful")
        
        # Create configuration
        config = LightRAGConfig.from_env()
        print(f"✅ Configuration loaded: {config.to_dict()}")
        
        # Create LightRAG component
        lightrag_component = LightRAGComponent(config)
        print("✅ LightRAG component created")
        
        # Initialize the component
        print("🔄 Initializing LightRAG component...")
        await lightrag_component.initialize()
        print("✅ LightRAG component initialized")
        
        # Check if PDF is available
        pdf_path = Path("clinical_metabolomics_review.pdf")
        if pdf_path.exists():
            print(f"✅ PDF found: {pdf_path} ({pdf_path.stat().st_size} bytes)")
        else:
            print(f"⚠️  PDF not found at {pdf_path}")
            # Check papers directory
            papers_dir = Path("papers")
            if papers_dir.exists():
                pdf_files = list(papers_dir.glob("*.pdf"))
                if pdf_files:
                    print(f"✅ Found PDFs in papers/: {[f.name for f in pdf_files]}")
                else:
                    print("⚠️  No PDFs found in papers/ directory")
            else:
                print("⚠️  Papers directory not found")
        
        # Test the specific question
        question = "What does the clinical metabolomics review document say about sample preparation methods?"
        
        print(f"\n🔍 Testing Question: '{question}'")
        print("-" * 60)
        
        # Query LightRAG
        result = await lightrag_component.query(question)
        
        print("\n📊 LightRAG Response:")
        print("=" * 60)
        print(f"Answer: {result.get('answer', 'No answer provided')}")
        print(f"Confidence Score: {result.get('confidence_score', 0.0)}")
        print(f"Processing Time: {result.get('processing_time', 0.0)} seconds")
        print(f"Source Documents: {len(result.get('source_documents', []))}")
        print(f"Entities Used: {len(result.get('entities_used', []))}")
        print(f"Relationships Used: {len(result.get('relationships_used', []))}")
        
        # Show source documents if available
        if result.get('source_documents'):
            print(f"\n📚 Source Documents:")
            for i, doc in enumerate(result['source_documents'][:3], 1):  # Show first 3
                print(f"  {i}. {doc}")
        
        # Show entities if available
        if result.get('entities_used'):
            print(f"\n🏷️  Entities Used:")
            for i, entity in enumerate(result['entities_used'][:5], 1):  # Show first 5
                print(f"  {i}. {entity}")
        
        # Show metadata
        if result.get('metadata'):
            print(f"\n📋 Metadata:")
            for key, value in result['metadata'].items():
                print(f"  {key}: {value}")
        
        # Check if fallback was used
        if result.get('fallback_used'):
            print(f"\n⚠️  Fallback Method Used: {result.get('fallback_method', 'Unknown')}")
        else:
            print(f"\n✅ Primary LightRAG processing successful!")
        
        return result
        
    except ImportError as e:
        print(f"❌ Import Error: {str(e)}")
        print("   LightRAG integration components not available")
        return None
        
    except Exception as e:
        print(f"❌ Error during LightRAG test: {str(e)}")
        import traceback
        print(f"   Traceback: {traceback.format_exc()}")
        return None

async def test_pdf_content():
    """Test if PDF content is accessible"""
    
    print("\n📄 Testing PDF Content Access")
    print("=" * 60)
    
    # Check for PDF files
    pdf_locations = [
        "clinical_metabolomics_review.pdf",
        "papers/clinical_metabolomics_review.pdf",
        Path("papers").glob("*.pdf")
    ]
    
    for location in pdf_locations[:2]:  # Check first two locations
        pdf_path = Path(location)
        if pdf_path.exists():
            print(f"✅ Found PDF: {pdf_path}")
            print(f"   Size: {pdf_path.stat().st_size:,} bytes")
            
            # Try to read first few lines
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    print(f"   Pages: {len(reader.pages)}")
                    
                    # Extract first page text sample
                    if len(reader.pages) > 0:
                        first_page = reader.pages[0]
                        text = first_page.extract_text()
                        print(f"   First 200 chars: {text[:200]}...")
                        
                        # Look for sample preparation mentions
                        if "sample preparation" in text.lower():
                            print("   ✅ 'Sample preparation' found in first page")
                        else:
                            print("   ⚠️  'Sample preparation' not found in first page")
                            
            except Exception as e:
                print(f"   ❌ Error reading PDF: {str(e)}")
            
            return pdf_path
    
    # Check papers directory for any PDFs
    papers_dir = Path("papers")
    if papers_dir.exists():
        pdf_files = list(papers_dir.glob("*.pdf"))
        if pdf_files:
            print(f"✅ Found {len(pdf_files)} PDF(s) in papers/:")
            for pdf in pdf_files:
                print(f"   - {pdf.name} ({pdf.stat().st_size:,} bytes)")
            return pdf_files[0]
    
    print("❌ No PDF files found")
    return None

async def main():
    """Main test function"""
    
    print("🚀 LightRAG Sample Preparation Test")
    print("=" * 60)
    
    # Test PDF content first
    pdf_found = await test_pdf_content()
    
    if not pdf_found:
        print("\n❌ Cannot test LightRAG without PDF content")
        print("   Please ensure clinical_metabolomics_review.pdf is available")
        return
    
    # Test LightRAG query
    result = await test_lightrag_query()
    
    if result:
        print(f"\n🎯 Test Summary:")
        print(f"   Question: 'What does the clinical metabolomics review document say about sample preparation methods?'")
        print(f"   Answer Length: {len(result.get('answer', ''))}")
        print(f"   Confidence: {result.get('confidence_score', 0.0):.2f}")
        print(f"   Source: {'LightRAG' if not result.get('fallback_used') else 'Fallback'}")
        print(f"   Success: {'✅ Yes' if result.get('answer') else '❌ No'}")
        
        # Return the answer for the user
        if result.get('answer'):
            print(f"\n📝 LightRAG Answer:")
            print("-" * 40)
            print(result['answer'])
            print("-" * 40)
    else:
        print(f"\n❌ LightRAG test failed")

if __name__ == "__main__":
    asyncio.run(main())