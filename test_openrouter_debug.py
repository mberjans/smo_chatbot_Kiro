#!/usr/bin/env python3
"""
Debug OpenRouter API Connection
Comprehensive testing and debugging for OpenRouter/Perplexity integration
"""

import os
import sys
import asyncio
import json
import requests

# Add src to path
sys.path.insert(0, 'src')

async def test_openrouter_api_directly(api_key: str):
    """Test OpenRouter API directly with HTTP requests"""
    print("🔍 Testing OpenRouter API directly...")
    
    # Test 1: List available models
    print("\n1️⃣ Testing model list endpoint...")
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            models_data = response.json()
            perplexity_models = [m for m in models_data.get('data', []) if 'perplexity' in m.get('id', '').lower()]
            print(f"   ✅ Found {len(perplexity_models)} Perplexity models")
            
            # Check if sonar-pro is available
            sonar_pro = next((m for m in perplexity_models if 'sonar-pro' in m.get('id', '')), None)
            if sonar_pro:
                print(f"   ✅ perplexity/sonar-pro is available")
                print(f"      Context length: {sonar_pro.get('context_length', 'unknown')}")
            else:
                print(f"   ⚠️  perplexity/sonar-pro not found in model list")
                print("   Available Perplexity models:")
                for model in perplexity_models[:5]:  # Show first 5
                    print(f"      - {model.get('id')}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
    
    # Test 2: Simple chat completion
    print("\n2️⃣ Testing chat completion with perplexity/sonar-pro...")
    try:
        payload = {
            "model": "perplexity/sonar-pro",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Respond briefly."
                },
                {
                    "role": "user", 
                    "content": "What is 2+2?"
                }
            ],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=30
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            print(f"   ✅ Success!")
            print(f"   Response: {content}")
            print(f"   Tokens used: {usage.get('total_tokens', 'unknown')}")
        else:
            print(f"   ❌ Error: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")

async def test_with_openai_client(api_key: str):
    """Test using OpenAI client (which our integration uses)"""
    print("\n🔧 Testing with OpenAI client (our integration method)...")
    
    try:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        print("   ✅ OpenAI client initialized")
        
        # Test simple completion
        response = await client.chat.completions.create(
            model="perplexity/sonar-pro",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is metabolomics? Answer in one sentence."}
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        content = response.choices[0].message.content
        usage = response.usage
        
        print(f"   ✅ Success!")
        print(f"   Response: {content}")
        print(f"   Tokens: {usage.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False

async def test_our_integration(api_key: str):
    """Test our OpenRouter integration"""
    print("\n🤖 Testing our OpenRouter integration...")
    
    try:
        from openrouter_integration import OpenRouterClient
        
        # Initialize with explicit API key
        client = OpenRouterClient(api_key=api_key)
        
        if not client.is_available():
            print("   ❌ Client not available")
            return False
        
        print("   ✅ Client initialized")
        
        # Test query
        result = await client.query_perplexity(
            question="What is metabolomics?",
            model="perplexity/sonar-pro",
            max_tokens=100
        )
        
        if result["success"]:
            print("   ✅ Query successful!")
            print(f"   Model: {result['model']}")
            print(f"   Response length: {len(result['content'])}")
            print(f"   Tokens: {result['tokens_used']['total']}")
            print(f"   Preview: {result['content'][:100]}...")
            return True
        else:
            print(f"   ❌ Query failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_api_key_format(api_key: str):
    """Check if API key has correct format"""
    print("🔑 API Key Analysis")
    print("=" * 20)
    
    print(f"Length: {len(api_key)}")
    print(f"Starts with: {api_key[:10]}...")
    print(f"Ends with: ...{api_key[-5:]}")
    
    if api_key.startswith('sk-or-v1-'):
        print("✅ Correct OpenRouter format")
    else:
        print("❌ Incorrect format (should start with 'sk-or-v1-')")
    
    if len(api_key) > 50:
        print("✅ Reasonable length")
    else:
        print("⚠️  Unusually short for API key")

async def main():
    """Main debug function"""
    print("🔍 OpenRouter API Debug Tool")
    print("=" * 40)
    
    # Get API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ No OPENROUTER_API_KEY found in environment")
        return False
    
    # Analyze API key
    check_api_key_format(api_key)
    
    # Test API directly
    await test_openrouter_api_directly(api_key)
    
    # Test with OpenAI client
    openai_success = await test_with_openai_client(api_key)
    
    # Test our integration
    integration_success = await test_our_integration(api_key)
    
    # Summary
    print("\n" + "=" * 40)
    print("🎯 Test Summary")
    print("=" * 40)
    print(f"OpenAI Client: {'✅ Working' if openai_success else '❌ Failed'}")
    print(f"Our Integration: {'✅ Working' if integration_success else '❌ Failed'}")
    
    if openai_success and integration_success:
        print("\n🎉 All tests passed!")
        print("   OpenRouter/Perplexity integration is working correctly")
        print("   The chatbot should now have enhanced AI capabilities")
        return True
    else:
        print("\n⚠️  Some tests failed")
        print("   Check API key validity and network connection")
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Debug interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        sys.exit(1)