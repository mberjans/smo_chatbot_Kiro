#!/usr/bin/env python3
"""
Test Perplexity Sonar Pro via OpenRouter
Tests the specific perplexity/sonar-pro model with API key
"""

import os
import sys
import asyncio
import json

# Add src to path
sys.path.insert(0, 'src')

async def test_sonar_pro_with_api_key(api_key: str):
    """Test Perplexity Sonar Pro with provided API key"""
    print("🧪 Testing Perplexity Sonar Pro via OpenRouter")
    print("=" * 50)
    
    try:
        from openrouter_integration import OpenRouterClient
        
        # Initialize client with provided API key
        client = OpenRouterClient(api_key=api_key)
        
        if not client.is_available():
            print("❌ OpenRouter client not available")
            return False
        
        print("✅ OpenRouter client initialized")
        print(f"📋 Default model: {client.default_model}")
        
        # Test connection with sonar-pro specifically
        print("\n🔗 Testing connection with perplexity/sonar-pro...")
        
        result = await client.query_perplexity(
            question="What is metabolomics?",
            model="perplexity/sonar-pro",
            max_tokens=200
        )
        
        if result["success"]:
            print("✅ Connection successful!")
            print(f"   Model: {result['model_name']}")
            print(f"   Online search: {result['online_search']}")
            print(f"   Tokens used: {result['tokens_used']['total']}")
            print(f"   Response length: {len(result['content'])}")
            
            # Show response preview
            content_preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
            print(f"\n📝 Response preview:\n{content_preview}")
            
            # Test with citations
            print("\n🔍 Testing with citations...")
            citation_result = await client.query_with_citations(
                question="What are the main applications of metabolomics in clinical research?",
                model="perplexity/sonar-pro"
            )
            
            if citation_result["success"]:
                print("✅ Citation query successful!")
                print(f"   Citations found: {len(citation_result.get('citations', []))}")
                print(f"   Overall confidence: {citation_result.get('overall_confidence', 0):.2f}")
                print(f"   Tokens used: {citation_result['tokens_used']['total']}")
                
                # Show citations
                if citation_result.get('citations'):
                    print("\n📚 Citations:")
                    for i, citation in enumerate(citation_result['citations'][:3], 1):
                        print(f"   [{i}] {citation}")
                
                return True
            else:
                print(f"❌ Citation query failed: {citation_result.get('error')}")
                return False
        else:
            print(f"❌ Connection failed: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def get_api_key_from_user():
    """Get API key from user input or environment"""
    # First check environment
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key and api_key.strip():
        print(f"✅ Found API key in environment (length: {len(api_key)})")
        return api_key.strip()
    
    # Check .env file
    env_file = "src/.env"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                if line.startswith('OPENROUTER_API_KEY'):
                    if '=' in line:
                        key_value = line.split('=', 1)[1].strip().strip('"')
                        if key_value:
                            print(f"✅ Found API key in .env file (length: {len(key_value)})")
                            return key_value
    
    # Ask user for API key
    print("⚠️  No API key found in environment or .env file")
    print("🔑 Please provide your OpenRouter API key to test:")
    print("   (Get one from https://openrouter.ai/keys)")
    
    try:
        api_key = input("Enter API key: ").strip()
        if api_key:
            return api_key
        else:
            print("❌ No API key provided")
            return None
    except KeyboardInterrupt:
        print("\n❌ Test cancelled by user")
        return None

async def test_model_availability():
    """Test which Perplexity models are available"""
    print("\n📋 Available Perplexity Models")
    print("=" * 40)
    
    try:
        from openrouter_integration import OpenRouterClient
        client = OpenRouterClient()
        models = client.get_available_models()
        
        for model_id, info in models.items():
            status = "🎯 DEFAULT" if model_id == client.default_model else "  "
            print(f"{status} {info['name']}")
            print(f"     ID: {model_id}")
            print(f"     Context: {info['context_length']:,} tokens")
            print(f"     Online: {'Yes' if info['online'] else 'No'}")
            print(f"     Description: {info['description']}")
            print()
            
    except Exception as e:
        print(f"❌ Could not load models: {e}")

def show_integration_status():
    """Show current integration status"""
    print("🔧 Integration Status Check")
    print("=" * 30)
    
    # Check if files exist
    files_to_check = [
        "src/openrouter_integration.py",
        "src/main_simple.py",
        "src/.env"
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - Missing")
    
    # Check if server is running
    try:
        import requests
        response = requests.get("http://localhost:8001/health", timeout=2)
        if response.status_code == 200:
            print("✅ Chatbot server running on port 8001")
        else:
            print("⚠️  Chatbot server responding but with issues")
    except:
        try:
            response = requests.get("http://localhost:8000/health", timeout=2)
            if response.status_code == 200:
                print("✅ Chatbot server running on port 8000")
            else:
                print("⚠️  Chatbot server responding but with issues")
        except:
            print("❌ Chatbot server not responding")

async def main():
    """Main test function"""
    print("🤖 Clinical Metabolomics Oracle - Perplexity Sonar Pro Test")
    print("=" * 65)
    
    # Show integration status
    show_integration_status()
    
    # Show available models
    await test_model_availability()
    
    # Get API key
    api_key = get_api_key_from_user()
    
    if not api_key:
        print("\n❌ Cannot test without API key")
        print("\n💡 To test with API key:")
        print("   1. Get API key from https://openrouter.ai/keys")
        print("   2. Set OPENROUTER_API_KEY in environment or src/.env")
        print("   3. Run this test again")
        return False
    
    # Test the API
    success = await test_sonar_pro_with_api_key(api_key)
    
    if success:
        print("\n🎉 Perplexity Sonar Pro integration working!")
        print("   ✅ API key valid")
        print("   ✅ Model accessible")
        print("   ✅ Citations working")
        print("   ✅ Ready for production use")
        
        print("\n🚀 Next steps:")
        print("   1. Add API key to src/.env file:")
        print(f'      OPENROUTER_API_KEY="{api_key[:10]}..."')
        print("   2. Restart chatbot server:")
        print("      python3 stop_chatbot_server.py")
        print("      python3 start_chatbot_uvicorn.py")
        print("   3. Test at http://localhost:8001/chat")
        
    else:
        print("\n❌ Perplexity Sonar Pro test failed")
        print("   Check API key and network connection")
    
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)