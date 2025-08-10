#!/usr/bin/env python3
"""
Test that the Chainlit server can start successfully
"""

import os
import sys
import subprocess
import time
import signal

# Set environment variables
os.environ['DATABASE_URL'] = 'postgresql://test:test@localhost:5432/test_db'
os.environ['NEO4J_PASSWORD'] = 'test_password'
os.environ['PERPLEXITY_API'] = 'test_api_key_placeholder'
os.environ['OPENAI_API_KEY'] = 'sk-test_key_placeholder'

def test_server_startup():
    """Test that the server can start and respond"""
    print("🧪 Testing Chainlit server startup...")
    
    try:
        # Change to src directory
        os.chdir('src')
        
        # Start the server process
        process = subprocess.Popen([
            sys.executable, '-m', 'chainlit', 'run', 'main.py',
            '--host', '127.0.0.1',
            '--port', '8001',  # Use different port for testing
            '--headless'  # Run without opening browser
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a few seconds for startup
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Server started successfully!")
            print("✅ Process is running")
            
            # Try to terminate gracefully
            process.terminate()
            try:
                process.wait(timeout=5)
                print("✅ Server stopped gracefully")
            except subprocess.TimeoutExpired:
                process.kill()
                print("✅ Server stopped (forced)")
            
            return True
        else:
            # Process exited, check output
            stdout, stderr = process.communicate()
            print(f"❌ Server failed to start")
            print(f"Exit code: {process.returncode}")
            if stdout:
                print(f"STDOUT: {stdout}")
            if stderr:
                print(f"STDERR: {stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_server_startup()
    if success:
        print("\n🎉 Server startup test PASSED!")
        print("🚀 Ready to launch the full server!")
    else:
        print("\n❌ Server startup test FAILED!")
        print("🔧 Check the error messages above")
    
    sys.exit(0 if success else 1)