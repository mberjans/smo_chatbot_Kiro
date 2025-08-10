#!/usr/bin/env python3
"""
Test .env file loading and API key verification
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_env_loading():
    """Test that .env files are loaded correctly"""
    print("🔧 Testing .env file loading...")
    
    # Load dotenv manually to test
    try:
        from dotenv import load_dotenv
        
        # Load root .env
        root_env = Path('.env')
        if root_env.exists():
            load_dotenv(root_env)
            print(f"✅ Loaded root .env file: {root_env}")
        else:
            print(f"❌ Root .env file not found: {root_env}")
        
        # Load src/.env
        src_env = Path('src/.env')
        if src_env.exists():
            load_dotenv(src_env)
            print(f"✅ Loaded src .env file: {src_env}")
        else:
            print(f"❌ Src .env file not found: {src_env}")
            
    except ImportError:
        print("❌ python-dotenv not available")
        return False
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print(f"✅ OPENROUTER_API_KEY loaded (length: {len(api_key)})")
        print(f"   Starts with: {api_key[:15]}...")
        print(f"   Ends with: ...{api_key[-5:]}")
        
        if api_key.endswith('2a673'):
            print("✅ API key ends with '2a673' as expected")
            return api_key
        else:
            print(f"⚠️  API key ends with '{api_key[-5:]}', expected '2a673'")
            return api_key
    else:
        print("❌ OPENROUTER_API_KEY not found")
        return None

async def test_openrouter_with_env_key():
    """Test OpenRouter integration with .env loaded key"""
    print("\n🤖 Testing OpenRouter integration with .env key...")
    
    try:
        from openrouter_integration import OpenRouterClient
        
        # Initialize client (should load from .env)
        client = OpenRouterClient()
        
        if not client.is_available():
            print("❌ OpenRouter client not available")
            return False
        
        print("✅ OpenRouter client initialized")
        print(f"📋 API key loaded: {client.api_key[:15]}...{client.api_key[-5:]}")
        
        # Test connection
        test_result = await client.test_connection()
        
        if test_result["success"]:
            print("✅ Connection test successful!")
            print(f"   Model: {test_result.get('model')}")
            print(f"   Tokens used: {test_result.get('tokens_used', {}).get('total', 0)}")
            
            # Test Sonar Pro specifically
            print("\n🎯 Testing Perplexity Sonar Pro...")
            result = await client.query_perplexity(
                question="What is metabolomics?",
                model="perplexity/sonar-pro",
                max_tokens=100
            )
            
            if result["success"]:
                print("✅ Sonar Pro query successful!")
                print(f"   Model: {result['model_name']}")
                print(f"   Response length: {len(result['content'])}")
                print(f"   Tokens used: {result['tokens_used']['total']}")
                print(f"   Online search: {result['online_search']}")
                
                # Show response preview
                preview = result['content'][:150] + "..." if len(result['content']) > 150 else result['content']
                print(f"\n📝 Response preview:\n{preview}")
                
                return True
            else:
                print(f"❌ Sonar Pro query failed: {result.get('error')}")
                return False
        else:
            print(f"❌ Connection test failed: {test_result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔍 Environment Loading and API Key Test")
    print("=" * 45)
    
    # Test .env loading
    api_key = test_env_loading()
    
    if not api_key:
        print("\n❌ Cannot proceed without API key")
        return False
    
    # Test OpenRouter integration
    success = asyncio.run(test_openrouter_with_env_key())
    
    if success:
        print("\n🎉 All tests passed!")
        print("   ✅ .env file loading works")
        print("   ✅ API key is valid")
        print("   ✅ Perplexity Sonar Pro is working")
        print("   ✅ Integration is ready for production")
        
        print("\n🚀 Next steps:")
        print("   1. Restart the chatbot server to use the new API key")
        print("   2. Test enhanced queries at the chat interface")
        print("   3. Enjoy professional-grade AI responses!")
        
    else:
        print("\n❌ Some tests failed")
        print("   Check the error messages above")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)