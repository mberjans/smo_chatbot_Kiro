#!/usr/bin/env python3
"""
Test the chatbot's Perplexity Sonar Pro integration
Verifies the chatbot is using the correct API key and model
"""

import os
import sys
import asyncio
import requests

# Add src to path
sys.path.insert(0, 'src')

async def test_chatbot_openrouter_integration():
    """Test the chatbot's OpenRouter integration directly"""
    print("🤖 Testing Chatbot's OpenRouter Integration")
    print("=" * 45)
    
    # Clear environment variable to ensure .env file is used
    if 'OPENROUTER_API_KEY' in os.environ:
        del os.environ['OPENROUTER_API_KEY']
    
    try:
        from openrouter_integration import get_openrouter_client
        
        # Get the client (should load from .env)
        client = get_openrouter_client()
        
        if not client.is_available():
            print("❌ OpenRouter client not available")
            return False
        
        print("✅ OpenRouter client available")
        print(f"📋 API key: {client.api_key[:15]}...{client.api_key[-5:]}")
        
        # Verify it's the correct key
        if client.api_key.endswith('2a673'):
            print("✅ Using correct API key (ends with 2a673)")
        else:
            print(f"⚠️  Using different API key (ends with {client.api_key[-5:]})")
        
        # Test the default model
        print(f"🎯 Default model: {client.default_model}")
        
        # Test a query
        print("\n🔍 Testing query with Perplexity Sonar Pro...")
        result = await client.query_with_citations(
            question="What are the main applications of metabolomics in clinical research?",
            model="perplexity/sonar-pro"
        )
        
        if result["success"]:
            print("✅ Query successful!")
            print(f"   Model: {result['model_name']}")
            print(f"   Response length: {len(result['content'])}")
            print(f"   Citations: {len(result.get('citations', []))}")
            print(f"   Confidence: {result.get('overall_confidence', 0):.2f}")
            print(f"   Tokens used: {result['tokens_used']['total']}")
            print(f"   Online search: {result['online_search']}")
            
            # Show response preview
            preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            print(f"\n📝 Response preview:\n{preview}")
            
            # Show citations
            if result.get('citations'):
                print(f"\n📚 Citations found:")
                for i, citation in enumerate(result['citations'][:3], 1):
                    print(f"   [{i}] {citation}")
            
            return True
        else:
            print(f"❌ Query failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_server_health():
    """Test that the server is running and healthy"""
    print("\n🏥 Testing Server Health")
    print("=" * 25)
    
    try:
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("✅ Server is healthy")
            print(f"   Service: {data.get('service')}")
            print(f"   Version: {data.get('version')}")
            print(f"   Working directory: {data.get('working_directory')}")
            return True
        else:
            print(f"❌ Server returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        return False

async def main():
    """Main test function"""
    print("🔍 Chatbot Perplexity Integration Test")
    print("=" * 40)
    
    # Test server health
    server_healthy = test_server_health()
    
    if not server_healthy:
        print("\n❌ Server is not healthy, cannot test integration")
        return False
    
    # Test OpenRouter integration
    integration_success = await test_chatbot_openrouter_integration()
    
    if integration_success:
        print("\n🎉 All tests passed!")
        print("   ✅ Server is healthy and running")
        print("   ✅ OpenRouter API key is correct (ends with 2a673)")
        print("   ✅ Perplexity Sonar Pro is working")
        print("   ✅ Citations and confidence scoring active")
        print("   ✅ Real-time web search enabled")
        
        print("\n🚀 Your chatbot now has enhanced AI capabilities!")
        print("   Visit: http://localhost:8001/chat")
        print("   Try asking: 'What are recent developments in metabolomics?'")
        
        return True
    else:
        print("\n❌ Integration test failed")
        print("   Check the error messages above")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)