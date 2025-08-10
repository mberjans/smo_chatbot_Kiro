#!/usr/bin/env python3
"""
Stop Clinical Metabolomics Oracle Gunicorn Server
"""

import os
import sys
import signal
import requests
import subprocess

def stop_server():
    """Stop the Gunicorn server"""
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
        except Exception as e:
            print(f"⚠️  Could not read server info: {e}")
    
    # Try to stop via PID file
    pid_file = "/tmp/gunicorn_cmo.pid"
    if os.path.exists(pid_file):
        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
            
            print(f"📋 Found server PID: {pid}")
            os.kill(pid, signal.SIGTERM)
            print("✅ Sent SIGTERM to server process")
            
            # Wait a moment and check if it's still running
            import time
            time.sleep(2)
            
            try:
                os.kill(pid, 0)  # Check if process still exists
                print("⚠️  Process still running, sending SIGKILL...")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                print("✅ Server process stopped successfully")
                
        except Exception as e:
            print(f"❌ Failed to stop via PID file: {e}")
    
    # Try to stop via pkill
    try:
        result = subprocess.run(
            ["pkill", "-f", "clinical_metabolomics_oracle"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ Stopped server via pkill")
        else:
            print("ℹ️  No processes found via pkill")
    except Exception as e:
        print(f"⚠️  pkill not available: {e}")
    
    # Try to check if server is still responding
    port = server_info.get("port", "8000")
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=2)
        if response.status_code == 200:
            print(f"⚠️  Server still responding on port {port}")
        else:
            print("✅ Server no longer responding")
    except requests.exceptions.RequestException:
        print("✅ Server no longer responding")
    
    # Clean up server info file
    if os.path.exists(".chatbot_server_info"):
        os.remove(".chatbot_server_info")
        print("✅ Cleaned up server info file")
    
    # Clean up PID file
    if os.path.exists(pid_file):
        os.remove(pid_file)
        print("✅ Cleaned up PID file")
    
    print()
    print("🎯 Server shutdown complete!")
    print("   You can start it again with: python3 start_chatbot_gunicorn.py")

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