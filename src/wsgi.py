#!/usr/bin/env python3
"""
WSGI entry point for Clinical Metabolomics Oracle
Ensures proper working directory and imports
"""

import os
import sys
from pathlib import Path

# Get the directory containing this file (src/)
current_dir = Path(__file__).parent.absolute()

# Change to the src directory
os.chdir(current_dir)

# Add src to Python path
sys.path.insert(0, str(current_dir))

# Set up environment variables
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test_db')
os.environ.setdefault('NEO4J_PASSWORD', 'test_password')
os.environ.setdefault('PERPLEXITY_API', 'test_api_key_placeholder')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test_key_placeholder')

# Now import the FastAPI app
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from chainlit.utils import mount_chainlit
import logging

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
        "version": "1.0.0",
        "working_directory": str(os.getcwd())
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
    # Check for main files in current directory (src/)
    if os.path.exists("main_simple.py"):
        mount_chainlit(app=app, target="main_simple.py", path="/chat")
        logger.info("Mounted Chainlit with main_simple.py")
    elif os.path.exists("main.py"):
        mount_chainlit(app=app, target="main.py", path="/chat")
        logger.info("Mounted Chainlit with main.py")
    else:
        logger.error("No main files found in src directory")
        
        @app.get("/chat")
        async def chat_error():
            raise HTTPException(
                status_code=503,
                detail="Chat service temporarily unavailable. Main application file not found."
            )
        
except Exception as e:
    logger.error(f"Failed to mount Chainlit: {e}")
    
    @app.get("/chat")
    async def chat_error():
        raise HTTPException(
            status_code=503,
            detail=f"Chat service temporarily unavailable: {str(e)}"
        )

# Export the app for Gunicorn
application = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)