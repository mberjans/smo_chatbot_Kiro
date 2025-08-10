#!/usr/bin/env python3
"""
Test OpenRouter with different headers and configurations
"""

import os
import sys
import asyncio
import json
import requests

async def test_with_different_headers(api_key: str):
    """Test with different header configurations"""
    print("🔧 Testing different header configurations...")
    
    headers_configs = [
        {
            "name": "Basic headers",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        },
        {
            "name": "With HTTP-Referer",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost:8001"
            }
        },
        {
            "name": "With X-Title",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Title": "Clinical Metabolomics Oracle"
            }
        },
        {
            "name": "With both Referer and Title",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost:8001",
                "X-Title": "Clinical Metabolomics Oracle"
            }
        }
    ]
    
    payload = {
        "model": "perplexity/sonar-pro",
        "messages": [
            {"role": "user", "content": "What is 2+2?"}
        ],
        "max_tokens": 20
    }
    
    for config in headers_configs:
        print(f"\n📋 Testing: {config['name']}")
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=config['headers'],
                json=payload,
                timeout=10
            )
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                print(f"   ✅ Success: {content}")
                return config['headers']
            else:
                print(f"   ❌ Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    return None

async def test_different_models(api_key: str):
    """Test different Perplexity models to see which ones work"""
    print("\n🤖 Testing different Perplexity models...")
    
    models_to_test = [
        "perplexity/sonar-pro",
        "perplexity/llama-3.1-sonar-small-128k-online",
        "perplexity/llama-3.1-sonar-large-128k-online",
        "perplexity/sonar-small-online",
        "perplexity/sonar-medium-online"
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:8001",
        "X-Title": "Clinical Metabolomics Oracle"
    }
    
    working_models = []
    
    for model in models_to_test:
        print(f"\n📋 Testing model: {model}")
        
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 10
        }
        
        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                print(f"   ✅ Working: {content}")
                working_models.append(model)
            else:
                print(f"   ❌ Error: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    return working_models

async def check_account_status(api_key: str):
    """Check account status and credits"""
    print("\n💳 Checking account status...")
    
    try:
        # Try to get account info
        response = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Account info retrieved")
            print(f"   Data: {json.dumps(data, indent=2)}")
        else:
            print(f"   ❌ Error: {response.text}")
            
    except Exception as e:
        print(f"   ❌ Exception: {e}")

async def main():
    """Main test function"""
    print("🔍 OpenRouter Advanced Debug")
    print("=" * 35)
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return False
    
    print(f"🔑 Using API key: {api_key[:15]}...{api_key[-5:]}")
    
    # Check account status
    await check_account_status(api_key)
    
    # Test different headers
    working_headers = await test_with_different_headers(api_key)
    
    if working_headers:
        print(f"\n✅ Found working headers configuration!")
        print("   Will update integration to use these headers")
    else:
        print(f"\n⚠️  No working header configuration found")
        
        # Test different models
        working_models = await test_different_models(api_key)
        
        if working_models:
            print(f"\n✅ Found {len(working_models)} working models:")
            for model in working_models:
                print(f"   - {model}")
        else:
            print(f"\n❌ No working models found")
            print("   This suggests an account or API key issue")
    
    return working_headers is not None

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted")
        sys.exit(0)