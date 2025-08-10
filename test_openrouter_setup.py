#!/usr/bin/env python3
"""
Test OpenRouter Setup for Clinical Metabolomics Oracle
Shows how to configure and test OpenRouter/Perplexity integration
"""

import os
import sys
import asyncio

# Add src to path
sys.path.insert(0, 'src')

def check_openrouter_setup():
    """Check OpenRouter configuration"""
    print("🔧 OpenRouter Configuration Check")
    print("=" * 40)
    
    # Check .env file
    env_file = "src/.env"
    if os.path.exists(env_file):
        print(f"✅ Found .env file: {env_file}")
        
        with open(env_file, 'r') as f:
            content = f.read()
            
        if 'OPENROUTER_API_KEY' in content:
            print("✅ OPENROUTER_API_KEY field exists in .env")
            
            # Check if it has a value
            for line in content.split('\n'):
                if line.startswith('OPENROUTER_API_KEY'):
                    if '=' in line:
                        key_value = line.split('=', 1)[1].strip().strip('"')
                        if key_value:
                            print(f"✅ API key is set (length: {len(key_value)})")
                            return True
                        else:
                            print("⚠️  API key is empty")
                            return False
        else:
            print("❌ OPENROUTER_API_KEY not found in .env")
            return False
    else:
        print(f"❌ .env file not found: {env_file}")
        return False

async def test_openrouter_integration():
    """Test OpenRouter integration"""
    print("\n🧪 Testing OpenRouter Integration")
    print("=" * 40)
    
    try:
        from openrouter_integration import OpenRouterClient
        
        client = OpenRouterClient()
        
        if not client.is_available():
            print("❌ OpenRouter client not available")
            print("   Reason: API key not set or invalid")
            print("\n💡 To fix this:")
            print("   1. Get an API key from https://openrouter.ai/")
            print("   2. Add it to src/.env file:")
            print('      OPENROUTER_API_KEY="your_api_key_here"')
            print("   3. Restart the chatbot server")
            return False
        
        print("✅ OpenRouter client initialized")
        
        # Test connection
        print("🔗 Testing connection...")
        test_result = await client.test_connection()
        
        if test_result["success"]:
            print("✅ Connection successful!")
            print(f"   Model: {test_result.get('model')}")
            print(f"   Tokens used: {test_result.get('tokens_used', {}).get('total', 0)}")
            
            # Test a real query
            print("\n🤖 Testing query...")
            result = await client.query_with_citations(
                "What is metabolomics?",
                model="perplexity/llama-3.1-sonar-small-128k-online"  # Use smaller model for testing
            )
            
            if result["success"]:
                print("✅ Query successful!")
                print(f"   Response length: {len(result['content'])}")
                print(f"   Citations: {len(result.get('citations', []))}")
                print(f"   Confidence: {result.get('overall_confidence', 0):.2f}")
                print(f"   Model: {result.get('model_name')}")
                
                # Show first 200 characters of response
                content = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
                print(f"\n📝 Sample response:\n{content}")
                
                return True
            else:
                print(f"❌ Query failed: {result.get('error')}")
                return False
        else:
            print(f"❌ Connection failed: {test_result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

def show_available_models():
    """Show available Perplexity models"""
    print("\n📋 Available Perplexity Models via OpenRouter")
    print("=" * 50)
    
    try:
        from openrouter_integration import OpenRouterClient
        client = OpenRouterClient()
        models = client.get_available_models()
        
        for model_id, info in models.items():
            print(f"\n🤖 {info['name']}")
            print(f"   ID: {model_id}")
            print(f"   Context: {info['context_length']:,} tokens")
            print(f"   Online Search: {'Yes' if info['online'] else 'No'}")
            print(f"   Description: {info['description']}")
            
    except Exception as e:
        print(f"❌ Could not load models: {e}")

def show_setup_instructions():
    """Show setup instructions"""
    print("\n📖 OpenRouter Setup Instructions")
    print("=" * 40)
    print("1. Visit https://openrouter.ai/ and create an account")
    print("2. Go to the API Keys section and create a new key")
    print("3. Copy the API key")
    print("4. Edit src/.env file and set:")
    print('   OPENROUTER_API_KEY="your_api_key_here"')
    print("5. Restart the chatbot server:")
    print("   python3 stop_chatbot_server.py")
    print("   python3 start_chatbot_uvicorn.py")
    print("\n💰 Pricing Information:")
    print("   • OpenRouter charges per token used")
    print("   • Perplexity models typically cost $0.001-0.005 per 1K tokens")
    print("   • Online search models may cost slightly more")
    print("   • Check current pricing at https://openrouter.ai/models")

async def main():
    """Main test function"""
    print("🤖 Clinical Metabolomics Oracle - OpenRouter Setup")
    print("=" * 60)
    
    # Check configuration
    config_ok = check_openrouter_setup()
    
    if config_ok:
        # Test integration
        test_ok = await test_openrouter_integration()
        
        if test_ok:
            print("\n🎉 OpenRouter integration is working!")
            print("   Your chatbot now has access to Perplexity AI with real-time web search.")
        else:
            print("\n⚠️  OpenRouter integration has issues.")
            print("   Check your API key and network connection.")
    else:
        print("\n⚠️  OpenRouter not configured.")
        show_setup_instructions()
    
    # Always show available models
    show_available_models()
    
    print("\n" + "=" * 60)
    print("🔗 Useful Links:")
    print("   • OpenRouter: https://openrouter.ai/")
    print("   • API Keys: https://openrouter.ai/keys")
    print("   • Models: https://openrouter.ai/models")
    print("   • Pricing: https://openrouter.ai/models")

if __name__ == "__main__":
    asyncio.run(main())