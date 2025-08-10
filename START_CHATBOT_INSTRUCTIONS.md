# 🤖 Clinical Metabolomics Oracle - Website Startup Instructions

## Quick Start

The chatbot website is ready to launch! Here are the steps to start it:

### Option 1: Using the Launch Script (Recommended)
```bash
./launch_chatbot.sh
```

### Option 2: Manual Startup
```bash
# Set environment variables
export DATABASE_URL="postgresql://test:test@localhost:5432/test_db"
export NEO4J_PASSWORD="test_password"
export PERPLEXITY_API="test_api_key_placeholder"
export OPENAI_API_KEY="sk-test_key_placeholder"

# Start the chatbot
cd src
python3 -m chainlit run main.py --host 0.0.0.0 --port 8000
```

### Option 3: Using the Python Launcher
```bash
python3 start_chatbot_website.py
```

## Access Information

Once started, the chatbot will be available at:
- **Website**: http://localhost:8000
- **Login Credentials**:
  - Username: `admin` / Password: `admin123`
  - OR Username: `testing` / Password: `ku9R_3`

## Features Available

✅ **PDF-based Knowledge Retrieval**: Loaded with clinical_metabolomics_review.pdf  
✅ **Multi-language Support**: Automatic language detection and translation  
✅ **Citation System**: Source references with confidence scores  
✅ **Fallback Processing**: Uses Perplexity AI when primary system unavailable  
✅ **Real-time Translation**: Supports multiple languages  
✅ **User Authentication**: Secure login system  
✅ **Responsive Design**: Works on desktop and mobile  

## System Architecture

The chatbot uses a hybrid approach:

1. **Primary**: LightRAG integration with PDF content
2. **Fallback**: Perplexity AI for real-time search
3. **Translation**: Multi-language support with Google Translate and OPUS-MT
4. **Interface**: Chainlit web framework with FastAPI backend

## Sample Questions to Try

Once logged in, try asking:

- "What are the main applications of metabolomics in clinical research?"
- "How is mass spectrometry used in metabolomics studies?"
- "What are the challenges in metabolomics data analysis?"
- "Explain the difference between targeted and untargeted metabolomics"
- "How can metabolomics contribute to personalized medicine?"

## Troubleshooting

### If the chatbot doesn't start:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check that Python 3.8+ is being used
3. Verify environment variables are set correctly

### If you get authentication errors:
- Use the provided login credentials
- Check that the password authentication is working

### If queries fail:
- The system will automatically fall back to Perplexity AI
- Check the console logs for detailed error information
- Ensure the PDF file is in the `papers/` directory

## Stopping the Server

Press `Ctrl+C` in the terminal to stop the chatbot server.

## Next Steps

1. **Replace API Keys**: Update `PERPLEXITY_API` with a real Perplexity API key for enhanced functionality
2. **Add More PDFs**: Place additional research papers in the `papers/` directory
3. **Customize Interface**: Modify `src/.chainlit/config.toml` for UI customization
4. **Deploy**: Use the Docker configuration in `src/lightrag_integration/deployment/` for production

---

**Ready to start?** Run `./launch_chatbot.sh` and visit http://localhost:8000!