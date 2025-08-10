"""
Integration example showing how to use QueryRouter with the existing system.

This example demonstrates how to integrate the QueryRouter into the existing
Chainlit application to provide intelligent routing between LightRAG and Perplexity.
"""

import asyncio
import logging
from typing import Dict, Any, Optional


class IntegrationExample:
    """
    Example integration of QueryRouter with the existing system.
    
    This class shows how to modify the existing message handler to use
    intelligent routing instead of simple fallback.
    """
    
    def __init__(self):
        """Initialize the integration example."""
        self.logger = logging.getLogger(__name__)
        
        # These would be initialized from the actual system
        self.lightrag_component = None
        self.query_router = None
        self.translator = None
        self.detector = None
    
    async def initialize_routing_system(self):
        """
        Initialize the routing system components.
        
        This method shows how to set up the routing system in the existing
        Chainlit application startup.
        """
        try:
            # Step 1: Initialize LightRAG component (already done in main.py)
            # self.lightrag_component = LightRAGComponent(config)
            
            # Step 2: Create query functions for the router
            lightrag_query_func = self._create_lightrag_query_func()
            perplexity_query_func = self._create_perplexity_query_func()
            
            # Step 3: Initialize classifier and router
            # In real implementation, this would use actual LLM and config
            # from llama_index.llms.groq import Groq
            # llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
            # classifier = QueryClassifier(llm, config)
            # self.query_router = QueryRouter(classifier, lightrag_query_func, perplexity_query_func)
            
            self.logger.info("Routing system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize routing system: {str(e)}")
            return False
    
    def _create_lightrag_query_func(self):
        """Create LightRAG query function for the router."""
        async def query_lightrag(query: str) -> Dict[str, Any]:
            """
            Query LightRAG component and return formatted response.
            
            This is the same function from main.py but adapted for the router.
            """
            if not self.lightrag_component:
                raise Exception("LightRAG component not available")
            
            try:
                result = await self.lightrag_component.query(query)
                
                # Format response with citations if available
                content = result["answer"]
                bibliography = ""
                
                if result.get("source_documents"):
                    bibliography += "\n\n**Sources:**\n"
                    for i, doc in enumerate(result["source_documents"], 1):
                        confidence = result.get("confidence_breakdown", {}).get(
                            doc, result.get("confidence_score", 0.0)
                        )
                        bibliography += f"[{i}]: {doc}\n      (Confidence: {confidence:.2f})\n"
                
                return {
                    "content": content,
                    "bibliography": bibliography,
                    "confidence_score": result.get("confidence_score", 0.0),
                    "processing_time": result.get("processing_time", 0.0),
                    "source": "LightRAG",
                    "metadata": result.get("metadata", {})
                }
                
            except Exception as e:
                self.logger.error(f"LightRAG query failed: {str(e)}")
                raise
        
        return query_lightrag
    
    def _create_perplexity_query_func(self):
        """Create Perplexity query function for the router."""
        async def query_perplexity(query: str) -> Dict[str, Any]:
            """
            Query Perplexity API and return formatted response.
            
            This is the same function from main.py but adapted for the router.
            """
            try:
                import requests
                import re
                import os
                
                PERPLEXITY_API = os.environ["PERPLEXITY_API"]
                url = "https://api.perplexity.ai/chat/completions"

                payload = {
                    "model": "sonar",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert in clinical metabolomics. You respond to"
                                "user queries in a helpful manner, with a focus on correct"
                                "scientific detail. Include peer-reviewed sources for all claims."
                                "For each source/claim, provide a confidence score from 0.0-1.0, formatted as (confidence score: X.X)"
                                "Respond in a single paragraph, never use lists unless explicitly asked."
                            ),
                        },
                        {
                            "role": "user",
                            "content": query,
                        },
                    ],
                    "temperature": 0.1,
                    "search_domain_filter": [
                        "-wikipedia.org",
                    ],
                }
                headers = {
                    "Authorization": f"Bearer {PERPLEXITY_API}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    response_data = response.json()
                    content = response_data['choices'][0]['message']['content']
                    citations = response_data.get('citations', [])
                    
                    # Format bibliography (same logic as main.py)
                    bibliography_dict = {}
                    if citations:
                        counter = 1
                        for citation in citations:
                            bibliography_dict[str(counter)] = [citation]
                            counter += 1
                    
                    # Extract confidence scores from text
                    pattern = r"confidence score:\s*([0-9.]+)(?:\s*\)\s*((?:\[\d+\]\s*)+)|\s+based on\s+(\[\d+\]))"
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    for score, refs1, refs2 in matches:
                        confidence = score
                        refs = refs1 if refs1 else refs2
                        ref_nums = re.findall(r"\[(\d+)\]", refs)
                        for num in ref_nums:
                            if num in bibliography_dict:
                                bibliography_dict[num].append(confidence)
                    
                    # Format bibliography
                    bibliography = ""
                    references = "\n\n\n**References:**\n"
                    further_reading = "\n**Further Reading:**\n"
                    for key, value in bibliography_dict.items():
                        if len(value) > 1:
                            references += f"[{key}]: {value[0]} \n      (Confidence: {value[1]})\n"
                        else:
                            further_reading += f"[{key}]: {value[0]} \n"
                    if references != "\n\n\n**References:**\n":
                        bibliography += references
                    if further_reading != "\n**Further Reading:**\n":
                        bibliography += further_reading
                    
                    # Clean confidence scores from content
                    clean_pattern = r"\(\s*confidence score:\s*[0-9.]+\s*\)"
                    content = re.sub(clean_pattern, "", content, flags=re.IGNORECASE)
                    content = re.sub(r'\s+', ' ', content)
                    
                    return {
                        "content": content,
                        "bibliography": bibliography,
                        "confidence_score": 0.8,  # Default confidence for Perplexity
                        "processing_time": 0.0,
                        "source": "Perplexity",
                        "metadata": {"citations_count": len(citations)}
                    }
                else:
                    raise Exception(f"Perplexity API error: {response.status_code}, {response.text}")
                    
            except Exception as e:
                self.logger.error(f"Perplexity query failed: {str(e)}")
                raise
        
        return query_perplexity
    
    async def handle_message_with_routing(self, message_content: str, language: str = "en") -> Dict[str, Any]:
        """
        Handle a message using intelligent routing.
        
        This method replaces the simple fallback logic in the existing on_message
        handler with intelligent routing.
        
        Args:
            message_content: The user's message content
            language: Detected or specified language
            
        Returns:
            Dictionary with response data
        """
        try:
            # Step 1: Translate to English if needed (same as existing logic)
            content = message_content
            if language != "en" and language is not None and self.translator:
                # content = await translate(self.translator, content, source=language, target="en")
                pass
            
            # Step 2: Use intelligent routing instead of simple fallback
            if self.query_router:
                self.logger.info("Using intelligent routing for query")
                routed_response = await self.query_router.route_query(content)
                
                # Step 3: Translate response back if needed
                response_content = routed_response.content
                if language != "en" and language is not None and self.translator:
                    # response_content = await translate(self.translator, response_content, source="en", target=language)
                    pass
                
                # Step 4: Format final response
                return {
                    "content": response_content,
                    "bibliography": routed_response.bibliography,
                    "confidence_score": routed_response.confidence_score,
                    "source": routed_response.source,
                    "sources_used": routed_response.sources_used,
                    "routing_decision": routed_response.routing_decision,
                    "processing_time": routed_response.processing_time,
                    "errors": routed_response.errors,
                    "metadata": {
                        **routed_response.metadata,
                        "original_language": language,
                        "routing_used": True
                    }
                }
            
            else:
                # Fallback to original logic if router not available
                self.logger.warning("Query router not available, using fallback logic")
                return await self._fallback_message_handling(content, language)
                
        except Exception as e:
            self.logger.error(f"Message handling failed: {str(e)}")
            return {
                "content": "I apologize, but I'm currently unable to process your query due to technical issues. Please try again later.",
                "bibliography": "",
                "confidence_score": 0.0,
                "source": "Error",
                "sources_used": [],
                "errors": [str(e)],
                "metadata": {"error_response": True}
            }
    
    async def _fallback_message_handling(self, content: str, language: str) -> Dict[str, Any]:
        """
        Fallback message handling using the original logic.
        
        This maintains the existing behavior when the routing system is not available.
        """
        response_data = None
        error_messages = []
        
        # Try LightRAG first, then fall back to Perplexity (original logic)
        if self.lightrag_component is not None:
            try:
                self.logger.info("Attempting LightRAG query")
                lightrag_func = self._create_lightrag_query_func()
                response_data = await lightrag_func(content)
                self.logger.info(f"LightRAG query successful with confidence: {response_data.get('confidence_score', 0.0)}")
            except Exception as e:
                error_messages.append(f"LightRAG failed: {str(e)}")
                self.logger.warning(f"LightRAG query failed, falling back to Perplexity: {str(e)}")
        else:
            error_messages.append("LightRAG component not available")
            self.logger.info("LightRAG component not available, using Perplexity")
        
        # Fall back to Perplexity if LightRAG failed or unavailable
        if response_data is None:
            try:
                self.logger.info("Attempting Perplexity query")
                perplexity_func = self._create_perplexity_query_func()
                response_data = await perplexity_func(content)
                self.logger.info("Perplexity query successful")
            except Exception as e:
                error_messages.append(f"Perplexity failed: {str(e)}")
                self.logger.error(f"Perplexity query also failed: {str(e)}")
        
        if response_data is None:
            return {
                "content": "I apologize, but I'm currently unable to process your query.",
                "bibliography": "",
                "confidence_score": 0.0,
                "source": "Error",
                "sources_used": [],
                "errors": error_messages,
                "metadata": {"fallback_response": True}
            }
        
        response_data["sources_used"] = [response_data["source"].lower()]
        response_data["errors"] = error_messages
        response_data["metadata"] = {**response_data.get("metadata", {}), "fallback_logic": True}
        
        return response_data
    
    def get_routing_metrics(self) -> Optional[Dict[str, Any]]:
        """Get routing metrics if router is available."""
        if self.query_router:
            return self.query_router.get_routing_metrics()
        return None


def show_integration_modifications():
    """
    Show the key modifications needed to integrate routing into main.py.
    """
    print("=== Integration Modifications for main.py ===\n")
    
    print("1. Add imports at the top of main.py:")
    print("""
from lightrag_integration.routing import QueryRouter, QueryClassifier
from llama_index.llms.groq import Groq  # or your preferred LLM
""")
    
    print("\n2. Modify the on_chat_start function to initialize routing:")
    print("""
@cl.on_chat_start
async def on_chat_start(accepted: bool = False):
    # ... existing initialization code ...
    
    # Initialize LightRAG component (already exists)
    try:
        lightrag_config = LightRAGConfig.from_env()
        lightrag_component = LightRAGComponent(lightrag_config)
        await lightrag_component.initialize()
        cl.user_session.set("lightrag_component", lightrag_component)
        
        # NEW: Initialize routing system
        llm = Groq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
        classifier = QueryClassifier(llm, lightrag_config)
        
        # Create query functions
        async def lightrag_query_func(query: str):
            return await query_lightrag(lightrag_component, query)
        
        query_router = QueryRouter(
            classifier=classifier,
            lightrag_query_func=lightrag_query_func,
            perplexity_query_func=query_perplexity,
            config=lightrag_config
        )
        cl.user_session.set("query_router", query_router)
        
        logging.info("Routing system initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize routing system: {str(e)}")
        cl.user_session.set("query_router", None)
    
    # ... rest of existing code ...
""")
    
    print("\n3. Modify the on_message function to use routing:")
    print("""
@cl.on_message
async def on_message(message: cl.Message):
    start = time.time()
    detector: LanguageDetector = cl.user_session.get("detector")
    translator: BaseTranslator = cl.user_session.get("translator")
    query_router: QueryRouter = cl.user_session.get("query_router")
    content = message.content

    await cl.Message(content="Thinking...", author="CMO").send()

    # Language detection and translation (existing logic)
    language = cl.user_session.get("language")
    if not language or language == "auto":
        detection = await detect_language(detector, content)
        language = detection["language"]
    if language != "en" and language is not None:
        content = await translate(translator, content, source=language, target="en")

    # NEW: Use intelligent routing instead of simple fallback
    response_data = None
    
    if query_router is not None:
        try:
            logging.info("Using intelligent routing")
            routed_response = await query_router.route_query(content)
            
            response_data = {
                "content": routed_response.content,
                "bibliography": routed_response.bibliography,
                "confidence_score": routed_response.confidence_score,
                "source": routed_response.source,
                "processing_time": routed_response.processing_time,
                "routing_info": {
                    "strategy": routed_response.routing_decision.strategy.value,
                    "sources_used": routed_response.sources_used,
                    "classification": routed_response.routing_decision.classification.query_type.value,
                    "confidence": routed_response.routing_decision.confidence_score
                }
            }
            
            if routed_response.errors:
                logging.warning(f"Routing errors: {routed_response.errors}")
                
        except Exception as e:
            logging.error(f"Routing failed: {str(e)}")
            # Fall back to original logic
            query_router = None
    
    # Fallback to original logic if routing not available
    if response_data is None:
        # ... existing fallback logic ...
        pass
    
    # ... rest of existing message processing ...
""")
    
    print("\n4. Optional: Add routing metrics endpoint:")
    print("""
@cl.on_settings_update
async def setup_agent(settings):
    # ... existing settings logic ...
    
    # Show routing metrics if available
    query_router = cl.user_session.get("query_router")
    if query_router:
        metrics = query_router.get_routing_metrics()
        logging.info(f"Routing metrics: {metrics}")
""")
    
    print("\n=== Benefits of This Integration ===")
    print("• Intelligent query classification and routing")
    print("• Automatic fallback mechanisms")
    print("• Response combination for hybrid queries")
    print("• Comprehensive metrics and logging")
    print("• Maintains backward compatibility")
    print("• Configurable routing strategies")


if __name__ == "__main__":
    show_integration_modifications()