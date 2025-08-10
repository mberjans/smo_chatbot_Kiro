#!/usr/bin/env python3
"""
Clinical Metabolomics Oracle - Gunicorn ASGI Application
FastAPI app with Chainlit mounted for production deployment
"""

import os
import sys
import logging
from pathlib import Path

# Set up environment variables
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test_db')
os.environ.setdefault('NEO4J_PASSWORD', 'test_password')
os.environ.setdefault('PERPLEXITY_API', 'test_api_key_placeholder')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test_key_placeholder')

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from chainlit.utils import mount_chainlit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Clinical Metabolomics Oracle",
    description="AI-powered chatbot for clinical metabolomics research",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - redirect to chat interface"""
    return RedirectResponse(url="/chat")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Clinical Metabolomics Oracle",
        "version": "1.0.0"
    }

@app.get("/api/info")
async def api_info():
    """API information endpoint"""
    return {
        "name": "Clinical Metabolomics Oracle API",
        "description": "AI-powered chatbot for clinical metabolomics research",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "docs": "/docs"
        }
    }

# Mount Chainlit application
try:
    # Check for main files in current directory and src directory
    main_files = [
        "main_simple.py",
        "src/main_simple.py", 
        "main.py",
        "src/main.py"
    ]
    
    mounted = False
    for main_file in main_files:
        if os.path.exists(main_file):
            # Extract just the filename for Chainlit
            target_file = os.path.basename(main_file)
            mount_chainlit(app=app, target=target_file, path="/chat")
            logger.info(f"Mounted Chainlit with {main_file}")
            mounted = True
            break
    
    if not mounted:
        raise FileNotFoundError(f"No main files found. Checked: {main_files}")
        
except Exception as e:
    logger.error(f"Failed to mount Chainlit: {e}")
    
    @app.get("/chat")
    async def chat_error():
        raise HTTPException(
            status_code=503,
            detail="Chat service temporarily unavailable. Please try again later."
        )

# Application factory for Gunicorn
def create_app():
    """Application factory for Gunicorn"""
    return app

# For direct running (development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app_gunicorn:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )