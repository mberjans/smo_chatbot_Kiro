#!/usr/bin/env python3
"""
Simplified Clinical Metabolomics Oracle - Chainlit Application
Focuses on core chatbot functionality with LightRAG integration
"""

import os
import sys
import time
import logging
import asyncio
from typing import Optional

import chainlit as cl

# Set up environment variables if not already set
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost:5432/test_db')
os.environ.setdefault('NEO4J_PASSWORD', 'test_password')
os.environ.setdefault('PERPLEXITY_API', 'test_api_key_placeholder')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test_key_placeholder')
os.environ.setdefault('OPENROUTER_API_KEY', '')

# Add src to path for imports
sys.path.insert(0, os.path.dirname(__file__))

# LightRAG integration
try:
    from lightrag_integration.component import LightRAGComponent
    from lightrag_integration.config.settings import LightRAGConfig
    LIGHTRAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LightRAG not available: {e}")
    LIGHTRAG_AVAILABLE = False

# OpenRouter integration
try:
    from openrouter_integration import get_openrouter_client, query_perplexity_via_openrouter
    OPENROUTER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"OpenRouter integration not available: {e}")
    OPENROUTER_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Simple authentication callback"""
    valid_credentials = [
        ("admin", "admin123"),
        ("testing", "ku9R_3")
    ]
    
    if (username, password) in valid_credentials:
        return cl.User(
            identifier=username,
            metadata={"role": "user", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session"""
    logger.info("Starting new chat session")
    
    # Initialize LightRAG component if available
    lightrag_component = None
    if LIGHTRAG_AVAILABLE:
        try:
            lightrag_config = LightRAGConfig.from_env()
            lightrag_component = LightRAGComponent(lightrag_config)
            await lightrag_component.initialize()
            logger.info("LightRAG component initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LightRAG component: {str(e)}")
    
    cl.user_session.set("lightrag_component", lightrag_component)
    
    # Initialize OpenRouter client if available
    openrouter_client = None
    if OPENROUTER_AVAILABLE:
        try:
            openrouter_client = get_openrouter_client()
            if openrouter_client.is_available():
                # Test connection
                test_result = await openrouter_client.test_connection()
                if test_result["success"]:
                    logger.info("OpenRouter/Perplexity integration initialized successfully")
                else:
                    logger.warning(f"OpenRouter connection test failed: {test_result.get('error')}")
                    openrouter_client = None
            else:
                logger.warning("OpenRouter API key not available")
                openrouter_client = None
        except Exception as e:
            logger.error(f"Failed to initialize OpenRouter client: {str(e)}")
            openrouter_client = None
    
    cl.user_session.set("openrouter_client", openrouter_client)
    
    # Display welcome message
    welcome_elements = [
        cl.Text(
            name="Welcome to Clinical Metabolomics Oracle",
            content=(
                "I'm a specialized AI assistant designed to help you with clinical metabolomics research. "
                "I can access scientific literature and provide evidence-based responses with citations.\n\n"
                "I use multiple AI systems including:\n"
                "• LightRAG with PDF knowledge base\n"
                "• Perplexity AI via OpenRouter (with real-time web search)\n"
                "• Intelligent fallback responses\n\n"
                "To get started, simply ask me any question about clinical metabolomics!"
            ),
            display='inline'
        ),
        cl.Text(
            name='Important Disclaimer',
            content=(
                'The Clinical Metabolomics Oracle is an automated question answering tool, and is not intended '
                'to replace the advice of a qualified healthcare professional. Content generated is for '
                'informational purposes only, and is not advice for the treatment or diagnosis of any condition.'
            ),
            display='inline'
        )
    ]
    
    await cl.Message(
        content="",
        elements=welcome_elements,
        author="CMO",
    ).send()
    
    # Get user agreement
    res = await cl.AskActionMessage(
        content='Do you understand the purpose and limitations of the Clinical Metabolomics Oracle?',
        actions=[
            cl.Action(
                name='I Understand',
                label='I Understand',
                description='Agree and continue',
                payload={"response": "agree"}
            ),
            cl.Action(
                name='Disagree',
                label='Disagree',
                description='Disagree to terms of service',
                payload={"response": "disagree"}
            )
        ],
        timeout=300,
        author="CMO",
    ).send()
    
    if res and res.get("label") == "I Understand":
        await cl.Message(
            content=(
                "Great! I'm ready to help you with clinical metabolomics questions. "
                "You can ask me about:\n\n"
                "• Applications of metabolomics in clinical research\n"
                "• Analytical techniques (MS, NMR, etc.)\n"
                "• Data analysis challenges and methods\n"
                "• Biomarker discovery and validation\n"
                "• Personalized medicine applications\n\n"
                "What would you like to know?"
            ),
            author="CMO",
        ).send()
    else:
        await cl.Message(
            content="You must agree to the terms of service to continue using the system.",
            author="CMO",
        ).send()

async def query_lightrag(lightrag_component: LightRAGComponent, question: str) -> dict:
    """Query LightRAG component"""
    try:
        result = await lightrag_component.query(question)
        
        content = result.get("answer", "No answer available")
        confidence = result.get("confidence_score", 0.0)
        sources = result.get("source_documents", [])
        processing_time = result.get("processing_time", 0.0)
        
        # Format sources
        bibliography = ""
        if sources:
            bibliography += "\n\n**Sources:**\n"
            for i, source in enumerate(sources[:5], 1):  # Limit to 5 sources
                bibliography += f"[{i}] {source}\n"
        
        return {
            "content": content,
            "bibliography": bibliography,
            "confidence_score": confidence,
            "processing_time": processing_time,
            "source": "LightRAG Knowledge Base",
            "success": True
        }
        
    except Exception as e:
        logger.error(f"LightRAG query failed: {str(e)}")
        return {
            "content": "",
            "bibliography": "",
            "confidence_score": 0.0,
            "processing_time": 0.0,
            "source": "LightRAG",
            "success": False,
            "error": str(e)
        }

async def query_openrouter_perplexity(openrouter_client, question: str) -> dict:
    """Query Perplexity via OpenRouter"""
    try:
        result = await openrouter_client.query_with_citations(
            question=question,
            include_sources=True
        )
        
        if result["success"]:
            content = result["content"]
            
            # Format citations
            bibliography = ""
            if result.get("citations"):
                bibliography += "\n\n**Sources (via Perplexity):**\n"
                for i, citation in enumerate(result["citations"], 1):
                    bibliography += f"[{i}] {citation}\n"
            
            return {
                "content": content,
                "bibliography": bibliography,
                "confidence_score": result.get("overall_confidence", 0.8),
                "processing_time": 0.0,  # OpenRouter doesn't provide timing
                "source": f"OpenRouter/{result.get('model_name', 'Perplexity')}",
                "success": True,
                "tokens_used": result.get("tokens_used", {}),
                "online_search": result.get("online_search", True)
            }
        else:
            return {
                "content": "",
                "bibliography": "",
                "confidence_score": 0.0,
                "processing_time": 0.0,
                "source": "OpenRouter/Perplexity",
                "success": False,
                "error": result.get("error", "Unknown error")
            }
            
    except Exception as e:
        logger.error(f"OpenRouter query failed: {str(e)}")
        return {
            "content": "",
            "bibliography": "",
            "confidence_score": 0.0,
            "processing_time": 0.0,
            "source": "OpenRouter/Perplexity",
            "success": False,
            "error": str(e)
        }

def create_fallback_response(question: str) -> dict:
    """Create a fallback response when systems are unavailable"""
    fallback_responses = {
        "metabolomics applications": (
            "Clinical metabolomics has several key applications including biomarker discovery, "
            "disease diagnosis, drug development, and personalized medicine. It's particularly "
            "useful for understanding metabolic pathways and identifying disease-specific metabolic signatures."
        ),
        "mass spectrometry": (
            "Mass spectrometry (MS) is a fundamental analytical technique in metabolomics. "
            "It provides high sensitivity and specificity for metabolite identification and quantification. "
            "Common MS approaches include LC-MS, GC-MS, and direct infusion MS."
        ),
        "data analysis": (
            "Metabolomics data analysis involves several challenges including data preprocessing, "
            "normalization, statistical analysis, and metabolite identification. Common approaches "
            "include multivariate analysis, pathway analysis, and machine learning methods."
        ),
        "personalized medicine": (
            "Metabolomics contributes to personalized medicine by identifying individual metabolic "
            "profiles that can guide treatment decisions, predict drug responses, and monitor "
            "therapeutic outcomes based on a patient's unique metabolic signature."
        )
    }
    
    # Simple keyword matching for fallback responses
    question_lower = question.lower()
    for key, response in fallback_responses.items():
        if key in question_lower:
            return {
                "content": response,
                "bibliography": "\n\n*Note: This is a general response. For detailed information with citations, please ensure the system is properly configured.*",
                "confidence_score": 0.3,
                "processing_time": 0.0,
                "source": "Fallback Knowledge",
                "success": True
            }
    
    # Default fallback
    return {
        "content": (
            "I apologize, but I'm currently operating in limited mode. "
            "I can provide general information about clinical metabolomics, but for detailed, "
            "citation-backed responses, please ensure the system is properly configured with "
            "access to the scientific literature database."
        ),
        "bibliography": "",
        "confidence_score": 0.1,
        "processing_time": 0.0,
        "source": "Fallback Mode",
        "success": True
    }

@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming messages"""
    start_time = time.time()
    question = message.content.strip()
    
    if not question:
        await cl.Message(
            content="Please ask a question about clinical metabolomics.",
            author="CMO"
        ).send()
        return
    
    # Show thinking message
    thinking_msg = cl.Message(content="🤔 Thinking...", author="CMO")
    await thinking_msg.send()
    
    # Get components
    lightrag_component = cl.user_session.get("lightrag_component")
    openrouter_client = cl.user_session.get("openrouter_client")
    
    # Try to get response with priority: LightRAG -> OpenRouter/Perplexity -> Fallback
    response_data = None
    error_messages = []
    
    # 1. Try LightRAG first
    if lightrag_component:
        logger.info("Attempting LightRAG query")
        response_data = await query_lightrag(lightrag_component, question)
        
        if not response_data.get("success"):
            error_messages.append(f"LightRAG: {response_data.get('error', 'Unknown error')}")
            logger.warning(f"LightRAG failed: {response_data.get('error', 'Unknown error')}")
            response_data = None
    else:
        error_messages.append("LightRAG: Component not available")
    
    # 2. Try OpenRouter/Perplexity if LightRAG failed
    if not response_data and openrouter_client:
        logger.info("Attempting OpenRouter/Perplexity query")
        response_data = await query_openrouter_perplexity(openrouter_client, question)
        
        if not response_data.get("success"):
            error_messages.append(f"OpenRouter: {response_data.get('error', 'Unknown error')}")
            logger.warning(f"OpenRouter failed: {response_data.get('error', 'Unknown error')}")
            response_data = None
    elif not response_data:
        error_messages.append("OpenRouter: Client not available")
    
    # 3. Use basic fallback if all else fails
    if not response_data:
        logger.info("Using basic fallback response")
        response_data = create_fallback_response(question)
        response_data["error_messages"] = error_messages
    
    # Format final response
    response_content = response_data["content"]
    
    # Add metadata
    processing_time = time.time() - start_time
    metadata_info = (
        f"\n\n---\n"
        f"**Source:** {response_data['source']}\n"
        f"**Confidence:** {response_data['confidence_score']:.2f}\n"
        f"**Processing Time:** {processing_time:.2f}s"
    )
    
    # Add additional info for OpenRouter responses
    if "OpenRouter" in response_data.get("source", ""):
        if response_data.get("tokens_used"):
            tokens = response_data["tokens_used"]
            metadata_info += f"\n**Tokens Used:** {tokens.get('total', 0)}"
        if response_data.get("online_search"):
            metadata_info += f"\n**Online Search:** Enabled"
    
    # Add error information if available
    if response_data.get("error_messages"):
        metadata_info += f"\n**Fallback Reason:** Multiple systems unavailable"
    
    # Add bibliography if available
    if response_data.get("bibliography"):
        response_content += response_data["bibliography"]
    
    response_content += metadata_info
    
    # Update the thinking message with the response
    thinking_msg.content = response_content
    await thinking_msg.update()

@cl.author_rename
def rename(orig_author: str):
    """Rename authors for display"""
    rename_dict = {"Chatbot": "CMO"}
    return rename_dict.get(orig_author, orig_author)

if __name__ == "__main__":
    # This won't be called when run via chainlit, but useful for testing imports
    print("✅ Clinical Metabolomics Oracle - Simplified Version Ready")
    print("🚀 Run with: chainlit run main_simple.py")