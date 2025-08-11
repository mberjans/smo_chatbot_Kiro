#!/usr/bin/env python3
"""
Reality check for LightRAG integration - what actually works vs what's documented
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set up environment variables
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test_db')
os.environ.setdefault('NEO4J_PASSWORD', 'test_password')
os.environ.setdefault('PERPLEXITY_API', 'test_api_key_placeholder')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test_key_placeholder')
os.environ.setdefault('GROQ_API_KEY', 'GROQ_API_KEY_PLACEHOLDER')

# Configure logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise
logger = logging.getLogger(__name__)

def test_reality_vs_documentation():
    """Test what actually exists vs what's documented"""
    
    print("🔍 LightRAG Integration Reality Check")
    print("=" * 60)
    
    # Check what files actually exist
    print("\n📁 File System Reality Check:")
    
    files_to_check = [
        # Core files
        "src/lightrag_integration/__init__.py",
        "src/lightrag_integration/component.py",
        "src/lightrag_integration/config/settings.py",
        
        # Query system
        "src/lightrag_integration/query/engine.py",
        "src/lightrag_integration/routing/router.py",
        
        # Integration components
        "src/lightrag_integration/citation_formatter.py",
        "src/lightrag_integration/confidence_scoring.py",
        "src/lightrag_integration/translation_integration.py",
        "src/lightrag_integration/response_integration.py",
        "src/lightrag_integration/monitoring.py",
        "src/lightrag_integration/error_handling.py",
        
        # Testing
        "src/lightrag_integration/testing/execute_final_integration_tests.py",
        "src/lightrag_integration/testing/run_final_integration_tests.py",
        
        # Actual LightRAG library
        "requirements.txt",
    ]
    
    existing_files = []
    missing_files = []
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    print(f"\n📊 File System Summary:")
    print(f"   Existing: {len(existing_files)}/{len(files_to_check)} ({len(existing_files)/len(files_to_check)*100:.1f}%)")
    
    # Check requirements.txt for actual LightRAG dependency
    print(f"\n📦 Dependencies Check:")
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read()
            
        if "lightrag" in requirements.lower():
            print("✅ LightRAG dependency found in requirements.txt")
            # Extract the line
            for line in requirements.split('\n'):
                if 'lightrag' in line.lower():
                    print(f"   {line.strip()}")
        else:
            print("❌ LightRAG dependency NOT found in requirements.txt")
            
    except FileNotFoundError:
        print("❌ requirements.txt not found")
    
    # Check if LightRAG is actually installed
    print(f"\n🐍 Python Package Check:")
    
    try:
        import lightrag
        print("✅ LightRAG library is installed")
        print(f"   Version: {getattr(lightrag, '__version__', 'unknown')}")
    except ImportError:
        print("❌ LightRAG library is NOT installed")
    
    # Check what components can actually be imported
    print(f"\n🔧 Component Import Check:")
    
    components = [
        ("LightRAGComponent", "lightrag_integration.component", "LightRAGComponent"),
        ("LightRAGConfig", "lightrag_integration.config.settings", "LightRAGConfig"),
        ("Query Engine", "lightrag_integration.query.engine", "LightRAGQueryEngine"),
        ("Citation Formatter", "lightrag_integration.citation_formatter", "CitationFormatter"),
        ("Confidence Scorer", "lightrag_integration.confidence_scoring", "ConfidenceScorer"),
        ("Translation Integration", "lightrag_integration.translation_integration", "TranslationIntegrator"),
        ("Response Integration", "lightrag_integration.response_integration", "ResponseIntegrator"),
        ("Monitoring", "lightrag_integration.monitoring", "SystemMonitor"),
        ("Error Handling", "lightrag_integration.error_handling", "ErrorHandler"),
    ]
    
    working_components = []
    broken_components = []
    
    for name, module_path, class_name in components:
        try:
            module = __import__(module_path, fromlist=[class_name])
            if hasattr(module, class_name):
                working_components.append(name)
                print(f"✅ {name}: Can import {class_name}")
            else:
                broken_components.append(name)
                print(f"❌ {name}: Module exists but no {class_name} class")
        except ImportError as e:
            broken_components.append(name)
            print(f"❌ {name}: Import failed - {str(e)}")
    
    print(f"\n📊 Component Import Summary:")
    print(f"   Working: {len(working_components)}/{len(components)} ({len(working_components)/len(components)*100:.1f}%)")
    
    return {
        'files_existing': len(existing_files),
        'files_total': len(files_to_check),
        'components_working': len(working_components),
        'components_total': len(components),
        'lightrag_installed': 'lightrag' in sys.modules or check_lightrag_installed(),
        'working_components': working_components,
        'broken_components': broken_components
    }

def check_lightrag_installed():
    """Check if LightRAG is actually installed"""
    try:
        import lightrag
        return True
    except ImportError:
        return False

async def test_what_actually_works():
    """Test what functionality actually works"""
    
    print("\n🧪 Functional Reality Check")
    print("=" * 60)
    
    try:
        # Test basic component creation
        print("1️⃣ Testing basic component creation...")
        
        from lightrag_integration.component import LightRAGComponent
        from lightrag_integration.config.settings import LightRAGConfig
        
        config = LightRAGConfig.from_env()
        component = LightRAGComponent(config)
        
        print("✅ Component created successfully")
        
        # Test initialization
        print("\n2️⃣ Testing component initialization...")
        
        await component.initialize()
        print("✅ Component initialized successfully")
        
        # Test health check
        print("\n3️⃣ Testing health check...")
        
        health = await component.get_health_status()
        print(f"✅ Health check works: {health.overall_status.value}")
        
        # Test query with no knowledge base
        print("\n4️⃣ Testing query with empty knowledge base...")
        
        try:
            result = await component.query("What is metabolomics?")
            print(f"✅ Query works (empty KB): {result.get('answer', 'No answer')[:100]}...")
            print(f"   Confidence: {result.get('confidence_score', 0.0)}")
        except Exception as e:
            print(f"❌ Query failed: {str(e)}")
        
        # Test statistics
        print("\n5️⃣ Testing statistics...")
        
        stats = component.get_statistics()
        print(f"✅ Statistics work: {len(stats)} metrics")
        
        # Test cleanup
        print("\n6️⃣ Testing cleanup...")
        
        await component.cleanup()
        print("✅ Cleanup works")
        
        return True
        
    except Exception as e:
        print(f"❌ Functional test failed: {str(e)}")
        import traceback
        print(f"   Details: {traceback.format_exc()}")
        return False

def analyze_documentation_vs_reality(reality_check, functional_test):
    """Analyze the gap between documentation and reality"""
    
    print("\n📊 DOCUMENTATION vs REALITY ANALYSIS")
    print("=" * 60)
    
    # Documentation claims
    documented_features = [
        "Complete LightRAG integration",
        "Knowledge graph RAG",
        "Multi-language support",
        "Citation formatting",
        "Confidence scoring",
        "Performance optimization",
        "Comprehensive testing",
        "Production deployment ready"
    ]
    
    # Reality assessment
    reality_assessment = {
        "Complete LightRAG integration": reality_check['lightrag_installed'] and functional_test,
        "Knowledge graph RAG": False,  # No actual LightRAG library
        "Multi-language support": "Translation Integration" in reality_check['working_components'],
        "Citation formatting": "Citation Formatter" in reality_check['working_components'],
        "Confidence scoring": "Confidence Scorer" in reality_check['working_components'],
        "Performance optimization": functional_test,  # Basic performance features work
        "Comprehensive testing": reality_check['files_existing'] > reality_check['files_total'] * 0.8,
        "Production deployment ready": False  # Without actual LightRAG, not ready
    }
    
    print("Feature Analysis:")
    working_features = 0
    
    for feature in documented_features:
        status = reality_assessment.get(feature, False)
        icon = "✅" if status else "❌"
        print(f"   {icon} {feature}")
        if status:
            working_features += 1
    
    print(f"\nReality Score: {working_features}/{len(documented_features)} ({working_features/len(documented_features)*100:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    if not reality_check['lightrag_installed']:
        print("   🔴 CRITICAL: Install actual LightRAG library")
        print("      pip install lightrag-hku")
    
    if reality_check['components_working'] < reality_check['components_total'] * 0.7:
        print("   🟡 IMPORTANT: Fix broken integration components")
        for component in reality_check['broken_components']:
            print(f"      - Fix {component}")
    
    if working_features < len(documented_features) * 0.5:
        print("   🟡 IMPORTANT: Update documentation to reflect actual capabilities")
    
    if not reality_assessment["Production deployment ready"]:
        print("   🔴 CRITICAL: System is NOT ready for production deployment")
        print("      - Need actual LightRAG integration")
        print("      - Need working knowledge graph functionality")
    
    return {
        'reality_score': working_features / len(documented_features),
        'critical_issues': not reality_check['lightrag_installed'],
        'deployment_ready': reality_assessment["Production deployment ready"]
    }

async def main():
    """Main test function"""
    
    print("🎯 LightRAG Integration Reality Check")
    print("=" * 80)
    
    # Check what exists vs what's documented
    reality_check = test_reality_vs_documentation()
    
    # Test what actually works
    functional_test = await test_what_actually_works()
    
    # Analyze the gap
    analysis = analyze_documentation_vs_reality(reality_check, functional_test)
    
    # Final verdict
    print("\n" + "=" * 80)
    print("🎯 FINAL VERDICT")
    print("=" * 80)
    
    if analysis['reality_score'] >= 0.8:
        verdict = "✅ SYSTEM MOSTLY FUNCTIONAL"
        color = "green"
    elif analysis['reality_score'] >= 0.5:
        verdict = "⚠️  SYSTEM PARTIALLY FUNCTIONAL"
        color = "yellow"
    else:
        verdict = "❌ SYSTEM NOT FUNCTIONAL"
        color = "red"
    
    print(f"Overall Status: {verdict}")
    print(f"Reality Score: {analysis['reality_score']:.1%}")
    print(f"Deployment Ready: {'❌ NO' if not analysis['deployment_ready'] else '✅ YES'}")
    
    if analysis['critical_issues']:
        print(f"\n🚨 CRITICAL ISSUES DETECTED:")
        print(f"   - LightRAG library not installed")
        print(f"   - Knowledge graph functionality not available")
        print(f"   - Production deployment claims are inaccurate")
    
    print(f"\n📋 NEXT STEPS:")
    if not reality_check['lightrag_installed']:
        print(f"   1. Install LightRAG library: pip install lightrag-hku")
    print(f"   2. Fix broken integration components")
    print(f"   3. Test actual knowledge graph functionality")
    print(f"   4. Update documentation to reflect reality")
    
    return analysis['deployment_ready']

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)