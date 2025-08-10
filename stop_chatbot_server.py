#!/usr/bin/env python3
"""
Stop Clinical Metabolomics Oracle Server (Uvicorn or Gunicorn)
"""

import os
import sys
import signal
import requests
import subprocess

def stop_server():
    """Stop the server"""
    print("🛑 Stopping Clinical Metabolomics Oracle...")
    
    # Try to read server info
    server_info = {}
    if os.path.exists(".chatbot_server_info"):
        try:
            with open(".chatbot_server_info", "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        server_info[key] = value
            print(f"📋 Found server info: {server_info}")
        except Exception as e:
            print(f"⚠️  Could not read server info: {e}")
    
    # Try to stop via PID
    if "pid" in server_info:
        try:
            pid = int(server_info["pid"])
            print(f"📋 Attempting to stop process {pid}...")
            
            # Send SIGTERM first
            os.kill(pid, signal.SIGTERM)
            print("✅ Sent SIGTERM to server process")
            
            # Wait a moment and check if it's still running
            import time
            time.sleep(3)
            
            try:
                os.kill(pid, 0)  # Check if process still exists
                print("⚠️  Process still running, sending SIGKILL...")
                os.kill(pid, signal.SIGKILL)
                time.sleep(1)
                
                # Check again
                try:
                    os.kill(pid, 0)
                    print("❌ Process still running after SIGKILL")
                except ProcessLookupError:
                    print("✅ Server process stopped successfully")
                    
            except ProcessLookupError:
                print("✅ Server process stopped successfully")
                
        except Exception as e:
            print(f"❌ Failed to stop via PID: {e}")
    
    # Try to stop via process name
    try:
        # Stop uvicorn processes
        result = subprocess.run(
            ["pkill", "-f", "uvicorn.*wsgi:application"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Stopped uvicorn processes")
        
        # Stop gunicorn processes
        result = subprocess.run(
            ["pkill", "-f", "gunicorn.*wsgi:application"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Stopped gunicorn processes")
            
    except Exception as e:
        print(f"⚠️  pkill not available: {e}")
    
    # Check if server is still responding
    port = server_info.get("port", "8000")
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            print(f"⚠️  Server still responding on port {port}")
            print("   You may need to manually kill the process")
        else:
            print("✅ Server no longer responding")
    except requests.exceptions.RequestException:
        print("✅ Server no longer responding")
    
    # Clean up files
    if os.path.exists(".chatbot_server_info"):
        os.remove(".chatbot_server_info")
        print("✅ Cleaned up server info file")
    
    # Clean up PID files
    for pid_file in ["/tmp/gunicorn_cmo.pid", "/tmp/uvicorn_cmo.pid"]:
        if os.path.exists(pid_file):
            os.remove(pid_file)
            print(f"✅ Cleaned up {pid_file}")
    
    print()
    print("🎯 Server shutdown complete!")
    print("   You can start it again with:")
    print("     python3 start_chatbot_uvicorn.py")

def main():
    """Main function"""
    try:
        stop_server()
        return True
    except KeyboardInterrupt:
        print("\n⚠️  Stop operation interrupted")
        return False
    except Exception as e:
        print(f"❌ Error stopping server: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)