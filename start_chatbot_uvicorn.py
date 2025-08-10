#!/usr/bin/env python3
"""
Start Clinical Metabolomics Oracle with Uvicorn
Non-blocking startup that runs the server in the background
"""

import os
import sys
import subprocess
import time
import requests
import signal

def setup_environment():
    """Set up environment variables"""
    env_vars = {
        'DATABASE_URL': 'postgresql://test:test@localhost:5432/test_db',
        'NEO4J_PASSWORD': 'test_password',
        'PERPLEXITY_API': 'test_api_key_placeholder',
        'OPENAI_API_KEY': 'OPENAI_API_KEY_PLACEHOLDER',
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

def start_uvicorn_server():
    """Start the Uvicorn server in the background"""
    print("🤖 Clinical Metabolomics Oracle - Uvicorn Startup")
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
    
    # Find available port
    port = 8000
    if not check_port_available(port):
        port = 8001
        if not check_port_available(port):
            port = 8002
    
    print(f"🚀 Starting Uvicorn server on port {port}...")
    
    # Start Uvicorn server in background
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.wsgi:application",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--log-level", "info",
        "--access-log"
    ]
    
    try:
        # Start the server process in background
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Save process info
        with open(".chatbot_server_info", "w") as f:
            f.write(f"port={port}\n")
            f.write(f"pid={process.pid}\n")
            f.write(f"url=http://localhost:{port}\n")
            f.write(f"type=uvicorn\n")
        
        # Wait for startup
        max_retries = 15
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
                    print("   python3 stop_chatbot_server.py")
                    print()
                    print("📊 Server Status:")
                    health_data = response.json()
                    print(f"   Status: {health_data.get('status', 'unknown')}")
                    print(f"   Service: {health_data.get('service', 'unknown')}")
                    print(f"   Working Directory: {health_data.get('working_directory', 'unknown')}")
                    print()
                    
                    return True
                    
            except requests.exceptions.RequestException:
                if i < max_retries - 1:
                    print(f"⏳ Waiting for server startup... ({i+1}/{max_retries})")
                    time.sleep(2)
                else:
                    print("❌ Server failed to start - checking process...")
                    
                    # Check if process is still running
                    if process.poll() is None:
                        print("⚠️  Process is running but not responding to HTTP requests")
                        # Try to get some output
                        try:
                            stdout, stderr = process.communicate(timeout=1)
                            if stdout:
                                print(f"STDOUT: {stdout}")
                            if stderr:
                                print(f"STDERR: {stderr}")
                        except subprocess.TimeoutExpired:
                            print("Process is still running...")
                    else:
                        print(f"❌ Process exited with code: {process.returncode}")
                        stdout, stderr = process.communicate()
                        if stdout:
                            print(f"STDOUT: {stdout}")
                        if stderr:
                            print(f"STDERR: {stderr}")
                    
                    return False
        
        return False
        
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

def main():
    """Main function"""
    try:
        success = start_uvicorn_server()
        if success:
            print("🎉 Clinical Metabolomics Oracle is now running!")
            print("   The server is running in the background.")
            print("   You can continue using this terminal for other tasks.")
            print("   Check the server status anytime with: curl http://localhost:8000/health")
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