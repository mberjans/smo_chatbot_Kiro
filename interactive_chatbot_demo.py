#!/usr/bin/env python3
"""
Interactive Clinical Metabolomics Oracle Demo
Demonstrates the chatbot functionality with the loaded PDF content
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up environment variables for testing
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test_db')
os.environ.setdefault('NEO4J_PASSWORD', 'test_password')
os.environ.setdefault('PERPLEXITY_API', 'test_api_key')

# Configure logging to be less verbose for demo
logging.basicConfig(level=logging.WARNING)

class InteractiveChatbotDemo:
    """Interactive demo of the Clinical Metabolomics Oracle"""
    
    def __init__(self):
        self.lightrag_component = None
        self.session_start = datetime.now()
        self.query_count = 0
        
    async def initialize(self):
        """Initialize the chatbot component"""
        print("🤖 Clinical Metabolomics Oracle - Interactive Demo")
        print("=" * 60)
        print("Initializing chatbot... Please wait...")
        
        try:
            from lightrag_integration.component import LightRAGComponent
            self.lightrag_component = LightRAGComponent()
            await self.lightrag_component.initialize()
            print("✅ Chatbot initialized successfully!")
            print()
            return True
        except Exception as e:
            print(f"❌ Failed to initialize chatbot: {e}")
            return False
    
    async def load_pdf_content(self):
        """Load the clinical metabolomics PDF"""
        print("📄 Loading clinical metabolomics review PDF...")
        
        try:
            # Copy PDF to temporary location for ingestion
            import shutil
            temp_pdf = "temp_demo_pdf.pdf"
            shutil.copy("clinical_metabolomics_review.pdf", temp_pdf)
            
            # Ingest the PDF
            result = await self.lightrag_component.ingest_documents([temp_pdf])
            
            # Clean up
            if os.path.exists(temp_pdf):
                os.remove(temp_pdf)
            
            if result.get('successful', 0) > 0:
                print("✅ PDF content loaded successfully!")
            else:
                print("⚠️  PDF loading completed with fallback processing")
            
            print()
            return True
            
        except Exception as e:
            print(f"❌ Failed to load PDF: {e}")
            return False
    
    async def process_query(self, question: str):
        """Process a user query and display results"""
        if not question.strip():
            return
        
        self.query_count += 1
        print(f"\n🔍 Query #{self.query_count}: {question}")
        print("-" * 50)
        
        try:
            # Process the query
            start_time = datetime.now()
            response = await self.lightrag_component.query(
                question=question,
                context={"mode": "hybrid", "include_citations": True}
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            answer = response.get('answer', 'No answer available')
            confidence = response.get('confidence_score', 0.0)
            citations = response.get('source_documents', [])
            fallback_used = response.get('fallback_used', False)
            
            print(f"💬 Answer:")
            print(f"   {answer}")
            print()
            print(f"📊 Confidence Score: {confidence:.2f}")
            print(f"⏱️  Processing Time: {processing_time:.2f} seconds")
            print(f"🔄 Fallback Used: {'Yes' if fallback_used else 'No'}")
            
            if citations:
                print(f"📚 Citations: {len(citations)} found")
                for i, citation in enumerate(citations[:3], 1):
                    if isinstance(citation, dict):
                        source = citation.get('source', 'Unknown')
                        print(f"   {i}. {source}")
            else:
                print("📚 Citations: None available")
            
            print()
            
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print()
    
    def display_sample_queries(self):
        """Display sample queries for users to try"""
        print("💡 Sample Questions to Try:")
        print("   1. What are the main applications of metabolomics in clinical research?")
        print("   2. How is mass spectrometry used in metabolomics?")
        print("   3. What are the challenges in metabolomics data analysis?")
        print("   4. Explain the role of metabolomics in personalized medicine")
        print("   5. What analytical techniques are commonly used in metabolomics?")
        print("   6. How do you handle missing data in metabolomics studies?")
        print("   7. What is the difference between targeted and untargeted metabolomics?")
        print()
    
    async def run_interactive_session(self):
        """Run the interactive chat session"""
        print("🎯 Ready for questions! Type 'quit', 'exit', or 'q' to end the session.")
        print("   Type 'help' or 'samples' to see sample questions.")
        print()
        
        while True:
            try:
                # Get user input
                user_input = input("❓ Your question: ").strip()
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() in ['help', 'samples']:
                    self.display_sample_queries()
                    continue
                elif not user_input:
                    continue
                
                # Process the query
                await self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted by user.")
                break
            except EOFError:
                print("\n\n👋 Session ended.")
                break
            except Exception as e:
                print(f"\n❌ Unexpected error: {e}")
                continue
    
    def display_session_summary(self):
        """Display session summary"""
        session_duration = (datetime.now() - self.session_start).total_seconds()
        print("\n" + "=" * 60)
        print("📊 Session Summary")
        print("=" * 60)
        print(f"⏱️  Session Duration: {session_duration:.1f} seconds")
        print(f"🔢 Total Queries: {self.query_count}")
        if self.query_count > 0:
            print(f"⚡ Average Query Time: {session_duration/self.query_count:.1f} seconds")
        print("\n🙏 Thank you for using the Clinical Metabolomics Oracle!")
        print("   For more information, check the documentation in src/lightrag_integration/docs/")
        print()

async def main():
    """Main demo function"""
    demo = InteractiveChatbotDemo()
    
    # Initialize the system
    if not await demo.initialize():
        print("Failed to initialize. Exiting...")
        return False
    
    # Load PDF content
    if not await demo.load_pdf_content():
        print("Failed to load PDF content. Exiting...")
        return False
    
    # Run interactive session
    try:
        await demo.run_interactive_session()
    finally:
        demo.display_session_summary()
    
    return True

if __name__ == "__main__":
    print("Starting Clinical Metabolomics Oracle Interactive Demo...")
    print("Press Ctrl+C at any time to exit.\n")
    
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        sys.exit(1)