#!/usr/bin/env python3
"""
Start the Clinical Metabolomics Oracle Chatbot Website
Sets up environment variables and launches the Chainlit web interface
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_environment():
    """Set up required environment variables"""
    print("🔧 Setting up environment variables...")
    
    # Set default environment variables for demo/testing
    env_vars = {
        'DATABASE_URL': 'postgresql://test:test@localhost:5432/test_db',
        'NEO4J_PASSWORD': 'test_password',
        'PERPLEXITY_API': 'test_api_key_placeholder',  # You'll need to replace this with a real key
        'OPENAI_API_KEY': 'OPENAI_API_KEY_PLACEHOLDER',   # Optional - for OpenAI fallback
    }
    
    # Only set if not already present
    for key, value in env_vars.items():
        if not os.getenv(key):
            os.environ[key] = value
            print(f"  ✅ Set {key}")
        else:
            print(f"  ✅ {key} already set")
    
    print()

def check_dependencies():
    """Check if required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    required_packages = [
        'chainlit',
        'fastapi',
        'uvicorn',
        'lightrag-hku',
        'llama-index',
        'lingua-language-detector'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                print(f"  ✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to install {package}: {e}")
                return False
    
    print()
    return True

def prepare_pdf_content():
    """Ensure PDF content is available for the chatbot"""
    print("📄 Preparing PDF content...")
    
    pdf_file = "clinical_metabolomics_review.pdf"
    if os.path.exists(pdf_file):
        print(f"  ✅ Found {pdf_file}")
        
        # Create papers directory if it doesn't exist
        papers_dir = Path("papers")
        papers_dir.mkdir(exist_ok=True)
        
        # Copy PDF to papers directory if not already there
        papers_pdf = papers_dir / pdf_file
        if not papers_pdf.exists():
            import shutil
            shutil.copy(pdf_file, papers_pdf)
            print(f"  ✅ Copied PDF to papers directory")
        else:
            print(f"  ✅ PDF already in papers directory")
    else:
        print(f"  ⚠️  {pdf_file} not found - chatbot will use fallback processing")
    
    print()

def start_chainlit_app():
    """Start the Chainlit application"""
    print("🚀 Starting Clinical Metabolomics Oracle Chatbot...")
    print("=" * 60)
    print("🌐 The chatbot website will be available at:")
    print("   http://localhost:8000")
    print()
    print("🔑 Login credentials:")
    print("   Username: admin")
    print("   Password: admin123")
    print()
    print("   OR")
    print()
    print("   Username: testing") 
    print("   Password: ku9R_3")
    print()
    print("💡 Features available:")
    print("   ✅ PDF-based knowledge retrieval")
    print("   ✅ Multi-language support")
    print("   ✅ Citation and confidence scoring")
    print("   ✅ Fallback to Perplexity AI")
    print("   ✅ Real-time translation")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        # Change to src directory and run chainlit
        os.chdir('src')
        
        # Start Chainlit application
        subprocess.run([
            sys.executable, '-m', 'chainlit', 'run', 'main.py',
            '--host', '0.0.0.0',
            '--port', '8000'
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start chatbot: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    
    return True

def start_fastapi_app():
    """Alternative: Start the FastAPI application with Chainlit mounted"""
    print("🚀 Starting Clinical Metabolomics Oracle via FastAPI...")
    print("=" * 60)
    print("🌐 The chatbot website will be available at:")
    print("   http://localhost:8000/chat")
    print("   API docs: http://localhost:8000/docs")
    print()
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    try:
        # Change to src directory and run uvicorn
        os.chdir('src')
        
        # Start FastAPI application with Chainlit mounted
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 'app:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload'
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n\n👋 Chatbot stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to start chatbot: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False
    
    return True

def main():
    """Main function to start the chatbot website"""
    print("🤖 Clinical Metabolomics Oracle - Website Launcher")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please install missing packages manually.")
        return False
    
    # Prepare content
    prepare_pdf_content()
    
    # Ask user which mode to use
    print("🎯 Choose startup mode:")
    print("  1. Chainlit only (recommended)")
    print("  2. FastAPI with Chainlit mounted")
    print()
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == '1':
            return start_chainlit_app()
        elif choice == '2':
            return start_fastapi_app()
        else:
            print("Please enter 1 or 2")

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Startup cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Startup failed: {e}")
        sys.exit(1)