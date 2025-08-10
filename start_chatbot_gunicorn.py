#!/usr/bin/env python3
"""
Start Clinical Metabolomics Oracle with Gunicorn
Non-blocking startup that runs the server in the background
"""

import os
import sys
import subprocess
import time
import signal
import requests
from pathlib import Path

def setup_environment():
    """Set up environment variables"""
    env_vars = {
        'DATABASE_URL': 'postgresql://test:test@localhost:5432/test_db',
        'NEO4J_PASSWORD': 'test_password',
        'PERPLEXITY_API': 'test_api_key_placeholder',
        'OPENAI_API_KEY': 'sk-test_key_placeholder',
    }
    
    for key, value in env_vars.items():
        if not os.getenv(key):
            os.environ[key] = value

def check_port_available(port=8000):
    """Check if port is available"""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('localhost', port))
            return True
    except OSError:
        return False

def start_gunicorn_server():
    """Start the Gunicorn server in the background"""
    print("🤖 Clinical Metabolomics Oracle - Gunicorn Startup")
    print("=" * 55)
    
    # Setup environment
    setup_environment()
    print("✅ Environment variables configured")
    
    # Create papers directory and copy PDF
    os.makedirs("papers", exist_ok=True)
    if os.path.exists("clinical_metabolomics_review.pdf"):
        import shutil
        shutil.copy("clinical_metabolomics_review.pdf", "papers/")
        print("✅ PDF content prepared")
    
    # Check if port is available
    if not check_port_available(8000):
        print("⚠️  Port 8000 is already in use. Trying to stop existing server...")
        try:
            # Try to stop existing server
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("🔄 Existing server found - will start on different port")
                port = 8001
            else:
                port = 8000
        except:
            port = 8000
    else:
        port = 8000
    
    # Start Gunicorn server
    print(f"🚀 Starting Gunicorn server on port {port}...")
    
    cmd = [
        sys.executable, "-m", "gunicorn",
        "--config", "gunicorn.conf.py",
        "--bind", f"0.0.0.0:{port}",
        "--daemon",  # Run in background
        "src.wsgi:application"
    ]
    
    try:
        # Start the server from root directory
        process = subprocess.Popen(
            cmd,
            cwd=os.getcwd(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for startup
        time.sleep(3)
        
        # Check if server is running
        max_retries = 10
        for i in range(max_retries):
            try:
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Server started successfully!")
                    print()
                    print("🌐 Access Information:")
                    print(f"   Website: http://localhost:{port}")
                    print(f"   Chat Interface: http://localhost:{port}/chat")
                    print(f"   API Docs: http://localhost:{port}/docs")
                    print()
                    print("🔑 Login Credentials:")
                    print("   Username: admin | Password: admin123")
                    print("   OR Username: testing | Password: ku9R_3")
                    print()
                    print("💡 Features Available:")
                    print("   ✅ PDF-based knowledge retrieval")
                    print("   ✅ LightRAG integration with fallback")
                    print("   ✅ User authentication")
                    print("   ✅ Real-time chat interface")
                    print("   ✅ API endpoints")
                    print()
                    print("🛑 To stop the server:")
                    print(f"   python3 stop_chatbot_gunicorn.py")
                    print()
                    
                    # Save server info
                    with open(".chatbot_server_info", "w") as f:
                        f.write(f"port={port}\n")
                        f.write(f"pid={process.pid}\n")
                        f.write(f"url=http://localhost:{port}\n")
                    
                    return True
                    
            except requests.exceptions.RequestException:
                if i < max_retries - 1:
                    print(f"⏳ Waiting for server startup... ({i+1}/{max_retries})")
                    time.sleep(2)
                else:
                    print("❌ Server failed to start properly")
                    return False
        
        return False
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def main():
    """Main function"""
    try:
        success = start_gunicorn_server()
        if success:
            print("🎉 Clinical Metabolomics Oracle is now running!")
            print("   The server is running in the background.")
            print("   You can continue using this terminal for other tasks.")
        else:
            print("❌ Failed to start the server. Check the logs above.")
            return False
            
    except KeyboardInterrupt:
        print("\n⚠️  Startup interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)