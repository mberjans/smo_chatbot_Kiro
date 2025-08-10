#!/bin/bash

# Clinical Metabolomics Oracle - Simplified Launch Script

echo "🤖 Clinical Metabolomics Oracle - Simplified Launch"
echo "=================================================="

# Set environment variables
export DATABASE_URL="postgresql://test:test@localhost:5432/test_db"
export NEO4J_PASSWORD="test_password"
export PERPLEXITY_API="test_api_key_placeholder"
export OPENAI_API_KEY="sk-test_key_placeholder"

echo "✅ Environment variables set"

# Create papers directory and copy PDF if available
mkdir -p papers
if [ -f "clinical_metabolomics_review.pdf" ]; then
    cp clinical_metabolomics_review.pdf papers/ 2>/dev/null || true
    echo "✅ PDF content prepared"
fi

echo ""
echo "🚀 Starting simplified chatbot website..."
echo "🌐 Website will be available at: http://localhost:8000"
echo "🔑 Login: admin / admin123 (or testing / ku9R_3)"
echo "🛑 Press Ctrl+C to stop"
echo ""
echo "💡 This simplified version includes:"
echo "   ✅ Core chatbot functionality"
echo "   ✅ LightRAG integration (with fallback)"
echo "   ✅ User authentication"
echo "   ✅ Clean web interface"
echo "   ✅ PDF knowledge base support"
echo ""

# Change to src directory and start Chainlit with simplified main
cd src
python3 -m chainlit run main_simple.py --host 0.0.0.0 --port 8000