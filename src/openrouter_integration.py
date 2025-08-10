#!/usr/bin/env python3
"""
OpenRouter API Integration for Clinical Metabolomics Oracle
Provides access to Perplexity and other models via OpenRouter
"""

import os
import logging
import asyncio
from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI
import json
from pathlib import Path

# Load environment variables from .env files
try:
    from dotenv import load_dotenv
    
    # Load from root .env file first (where the API key is)
    root_env = Path(__file__).parent.parent / '.env'
    if root_env.exists():
        load_dotenv(root_env)
        
    # Then load from src/.env (for other variables)
    src_env = Path(__file__).parent / '.env'
    if src_env.exists():
        load_dotenv(src_env)
        
except ImportError:
    pass

logger = logging.getLogger(__name__)

class OpenRouterClient:
    """
    OpenRouter API client for accessing Perplexity and other models
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key. If None, reads from environment.
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        
        if not self.api_key:
            logger.warning("OpenRouter API key not found. OpenRouter features will be disabled.")
            self.client = None
            return
        
        # Initialize OpenAI client with OpenRouter base URL
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://localhost:8001",
                "X-Title": "Clinical Metabolomics Oracle"
            }
        )
        
        # Available Perplexity models via OpenRouter
        self.perplexity_models = {
            "perplexity/sonar-pro": {
                "name": "Perplexity Sonar Pro",
                "context_length": 127072,
                "online": True,
                "description": "Professional Perplexity model with web search capabilities"
            },
            "perplexity/llama-3.1-sonar-small-128k-online": {
                "name": "Perplexity Llama 3.1 Sonar Small (Online)",
                "context_length": 127072,
                "online": True,
                "description": "Fast online model with web search capabilities"
            },
            "perplexity/llama-3.1-sonar-large-128k-online": {
                "name": "Perplexity Llama 3.1 Sonar Large (Online)",
                "context_length": 127072,
                "online": True,
                "description": "Powerful online model with web search capabilities"
            },
            "perplexity/llama-3.1-sonar-huge-128k-online": {
                "name": "Perplexity Llama 3.1 Sonar Huge (Online)",
                "context_length": 127072,
                "online": True,
                "description": "Most capable online model with web search capabilities"
            }
        }
        
        # Default model - use Sonar Pro as requested
        self.default_model = "perplexity/sonar-pro"
        
        logger.info(f"OpenRouter client initialized with {len(self.perplexity_models)} Perplexity models")
    
    def is_available(self) -> bool:
        """Check if OpenRouter client is available"""
        return self.client is not None
    
    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get available Perplexity models"""
        return self.perplexity_models
    
    async def query_perplexity(
        self,
        question: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Query Perplexity model via OpenRouter
        
        Args:
            question: User question
            model: Model to use (defaults to default_model)
            system_prompt: System prompt for the model
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_available():
            raise RuntimeError("OpenRouter client not available. Check API key.")
        
        model = model or self.default_model
        
        if model not in self.perplexity_models:
            raise ValueError(f"Model {model} not available. Available models: {list(self.perplexity_models.keys())}")
        
        # Default system prompt for clinical metabolomics
        if system_prompt is None:
            system_prompt = (
                "You are an expert in clinical metabolomics research. Provide accurate, "
                "evidence-based responses with proper citations. Focus on scientific accuracy "
                "and include confidence indicators for your statements. When possible, "
                "reference peer-reviewed sources and current research findings."
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        try:
            logger.info(f"Querying OpenRouter with model: {model}")
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            # Extract response content
            content = response.choices[0].message.content
            
            # Calculate basic metrics
            usage = response.usage
            model_info = self.perplexity_models[model]
            
            result = {
                "content": content,
                "model": model,
                "model_name": model_info["name"],
                "online_search": model_info["online"],
                "tokens_used": {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "total": usage.total_tokens
                },
                "temperature": temperature,
                "max_tokens": max_tokens,
                "success": True,
                "source": "OpenRouter/Perplexity"
            }
            
            logger.info(f"OpenRouter query successful. Tokens used: {usage.total_tokens}")
            return result
            
        except Exception as e:
            logger.error(f"OpenRouter query failed: {str(e)}")
            return {
                "content": "",
                "model": model,
                "success": False,
                "error": str(e),
                "source": "OpenRouter/Perplexity"
            }
    
    async def query_with_citations(
        self,
        question: str,
        model: Optional[str] = None,
        include_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query with enhanced citation formatting
        
        Args:
            question: User question
            model: Model to use
            include_sources: Whether to request source citations
            
        Returns:
            Dictionary with formatted response and citations
        """
        # Enhanced system prompt for citations
        citation_prompt = (
            "You are an expert in clinical metabolomics research. Provide accurate, "
            "evidence-based responses with proper citations. For each claim or fact, "
            "include a confidence score (0.0-1.0) in parentheses. When citing sources, "
            "use the format [Source: description] at the end of relevant statements. "
            "Focus on peer-reviewed research and current findings in metabolomics."
        )
        
        if include_sources:
            citation_prompt += (
                " Include specific source references where possible, and indicate "
                "the reliability of each source."
            )
        
        result = await self.query_perplexity(
            question=question,
            model=model,
            system_prompt=citation_prompt,
            temperature=0.1
        )
        
        if result["success"]:
            # Parse citations and confidence scores from content
            content = result["content"]
            citations = self._extract_citations(content)
            confidence_scores = self._extract_confidence_scores(content)
            
            # Calculate overall confidence
            if confidence_scores:
                overall_confidence = sum(confidence_scores) / len(confidence_scores)
            else:
                overall_confidence = 0.7  # Default confidence
            
            result.update({
                "citations": citations,
                "confidence_scores": confidence_scores,
                "overall_confidence": overall_confidence,
                "formatted_content": self._format_content_with_citations(content)
            })
        
        return result
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citations from content"""
        import re
        
        # Look for [Source: ...] patterns
        citation_pattern = r'\[Source: ([^\]]+)\]'
        citations = re.findall(citation_pattern, content)
        
        return citations
    
    def _extract_confidence_scores(self, content: str) -> List[float]:
        """Extract confidence scores from content"""
        import re
        
        # Look for confidence scores in parentheses
        confidence_pattern = r'\((\d+\.?\d*)\)'
        matches = re.findall(confidence_pattern, content)
        
        scores = []
        for match in matches:
            try:
                score = float(match)
                if 0.0 <= score <= 1.0:
                    scores.append(score)
            except ValueError:
                continue
        
        return scores
    
    def _format_content_with_citations(self, content: str) -> str:
        """Format content with better citation display"""
        import re
        
        # Replace [Source: ...] with numbered citations
        citations = []
        
        def replace_citation(match):
            citation = match.group(1)
            citations.append(citation)
            return f"[{len(citations)}]"
        
        # Replace citations with numbers
        formatted_content = re.sub(r'\[Source: ([^\]]+)\]', replace_citation, content)
        
        # Add bibliography if citations exist
        if citations:
            formatted_content += "\n\n**Sources:**\n"
            for i, citation in enumerate(citations, 1):
                formatted_content += f"[{i}] {citation}\n"
        
        return formatted_content
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test OpenRouter connection"""
        if not self.is_available():
            return {
                "success": False,
                "error": "OpenRouter client not available",
                "api_key_present": bool(self.api_key)
            }
        
        try:
            # Simple test query
            result = await self.query_perplexity(
                question="What is metabolomics?",
                max_tokens=100
            )
            
            return {
                "success": result["success"],
                "model": result.get("model"),
                "tokens_used": result.get("tokens_used", {}),
                "api_key_present": True,
                "available_models": len(self.perplexity_models)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "api_key_present": bool(self.api_key)
            }

# Global client instance
_openrouter_client = None

def get_openrouter_client() -> OpenRouterClient:
    """Get global OpenRouter client instance"""
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = OpenRouterClient()
    return _openrouter_client

async def query_perplexity_via_openrouter(
    question: str,
    model: Optional[str] = None,
    include_citations: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to query Perplexity via OpenRouter
    
    Args:
        question: User question
        model: Perplexity model to use
        include_citations: Whether to include citations
        
    Returns:
        Dictionary with response and metadata
    """
    client = get_openrouter_client()
    
    if include_citations:
        return await client.query_with_citations(question, model)
    else:
        return await client.query_perplexity(question, model)

if __name__ == "__main__":
    # Test the OpenRouter integration
    async def test_openrouter():
        client = OpenRouterClient()
        
        print("🧪 Testing OpenRouter Integration")
        print("=" * 40)
        
        # Test connection
        connection_test = await client.test_connection()
        print(f"Connection test: {connection_test}")
        
        if connection_test["success"]:
            # Test query
            result = await client.query_with_citations(
                "What are the main applications of metabolomics in clinical research?"
            )
            
            print(f"\nQuery result:")
            print(f"Success: {result['success']}")
            if result["success"]:
                print(f"Model: {result['model_name']}")
                print(f"Content length: {len(result['content'])}")
                print(f"Citations: {len(result.get('citations', []))}")
                print(f"Confidence: {result.get('overall_confidence', 0):.2f}")
        else:
            print("❌ OpenRouter not available - check API key")
    
    asyncio.run(test_openrouter())