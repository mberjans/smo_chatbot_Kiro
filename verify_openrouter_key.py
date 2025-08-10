#!/usr/bin/env python3
"""
Simple OpenRouter API Key Verification
Quick test to verify if an OpenRouter API key works with perplexity/sonar-pro
"""

import os
import sys
import asyncio
import requests

def test_api_key(api_key: str) -> bool:
    """Test if API key works with a simple request"""
    print(f"🔑 Testing API key: {api_key[:15]}...{api_key[-5:]}")
    
    try:
        # Simple test with minimal payload
        payload = {
            "model": "perplexity/sonar-pro",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 5
        }
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost:8001",
            "X-Title": "Clinical Metabolomics Oracle"
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            print(f"✅ Success! Response: '{content}'")
            print(f"📈 Tokens used: {usage.get('total_tokens', 'unknown')}")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

async def test_with_our_integration(api_key: str) -> bool:
    """Test with our integration code"""
    print(f"\n🤖 Testing with our integration...")
    
    try:
        sys.path.insert(0, 'src')
        from openrouter_integration import OpenRouterClient
        
        client = OpenRouterClient(api_key=api_key)
        
        if not client.is_available():
            print("❌ Client not available")
            return False
        
        result = await client.query_perplexity(
            question="What is metabolomics?",
            model="perplexity/sonar-pro",
            max_tokens=50
        )
        
        if result["success"]:
            print("✅ Integration test successful!")
            print(f"📝 Response preview: {result['content'][:100]}...")
            print(f"📊 Tokens used: {result['tokens_used']['total']}")
            return True
        else:
            print(f"❌ Integration test failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Integration test exception: {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 OpenRouter API Key Verification")
    print("=" * 40)
    
    # Get API key from environment or user input
    api_key = os.getenv('OPENROUTER_API_KEY')
    
    if not api_key or not api_key.strip():
        print("⚠️  No API key found in OPENROUTER_API_KEY environment variable")
        print("🔑 Please provide your OpenRouter API key:")
        try:
            api_key = input("API Key: ").strip()
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return False
    
    if not api_key:
        print("❌ No API key provided")
        return False
    
    # Validate format
    if not api_key.startswith('sk-or-v1-'):
        print("⚠️  API key doesn't start with 'sk-or-v1-' (expected OpenRouter format)")
        print("   Make sure you're using an OpenRouter API key, not OpenAI")
    
    # Test direct API call
    direct_success = test_api_key(api_key)
    
    if direct_success:
        print("\n🎉 API key is working!")
        
        # Test integration
        integration_success = asyncio.run(test_with_our_integration(api_key))
        
        if integration_success:
            print("\n✅ Full integration test passed!")
            print("🚀 Your chatbot is ready to use Perplexity Sonar Pro!")
            
            # Show how to set up
            print("\n📋 To activate in your chatbot:")
            print("1. Set the API key in environment:")
            print(f"   export OPENROUTER_API_KEY='{api_key}'")
            print("2. Or add to src/.env file:")
            print(f"   OPENROUTER_API_KEY=\"{api_key}\"")
            print("3. Restart the chatbot server")
            
            return True
        else:
            print("\n⚠️  Direct API works but integration has issues")
            return False
    else:
        print("\n❌ API key is not working")
        print("💡 Possible issues:")
        print("   • API key might be invalid or expired")
        print("   • Account might not be activated")
        print("   • Insufficient credits")
        print("   • Network connectivity issues")
        print("\n🔗 Check your OpenRouter account at: https://openrouter.ai/")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Verification cancelled")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)