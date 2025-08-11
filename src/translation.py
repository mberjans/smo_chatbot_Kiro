import os
#from functools import cache
from functools import lru_cache
from typing import Dict, Any, Optional

import chainlit as cl
import hanzidentifier
from deep_translator import GoogleTranslator
from deep_translator.base import BaseTranslator
from lingua import IsoCode639_1, Language, LanguageDetector, LanguageDetectorBuilder

from translators.opusmt import OpusMTTranslator

# Import LightRAG translation integration
try:
    from lightrag_integration.translation_integration import LightRAGTranslationIntegrator
    from lightrag_integration.response_integration import ProcessedResponse, CombinedResponse
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False


@lru_cache(maxsize=None)
def get_language_detector(*iso_codes: IsoCode639_1):
    return LanguageDetectorBuilder.from_iso_codes_639_1(*iso_codes).build()



@lru_cache(maxsize=None)
def get_translator(translator: str = "google"):
    if translator == "google":
        return GoogleTranslator(source="auto", target="en")
    elif translator == "opusmt":
        return OpusMTTranslator(source="fr", target="en")
    else:
        return GoogleTranslator(source="auto", target="en")
    

def _detect_language(detector: LanguageDetector, content: str, threshold: float = 0.5):
    confidence_values = detector.compute_language_confidence_values(content)
    language = Language.ENGLISH
    for confidence_value in confidence_values:
        if confidence_value.value > threshold:
            language = confidence_value.language
            break
    iso_code = language.iso_code_639_1.name if language else None
    iso_code = iso_code.lower() if iso_code else None
    if iso_code == "zh":
        if hanzidentifier.is_traditional(content):
            iso_code = "zh-TW"
        else:
            iso_code = "zh-CN"
    confidence_values_dict = {
        confidence_value.language.name: confidence_value.value for confidence_value in confidence_values[:5]
    }
    return {
        "language": iso_code,
        "confidence_values": confidence_values_dict,
    }


@cl.step
async def detect_language(detector: LanguageDetector, content: str):
    return _detect_language(detector, content)


def _translate(translator: BaseTranslator, content: str, source: str = "auto", target: str = "en"):
    translator.source = source
    translator.target = target
    translated = translator.translate(content)
    return translated


@cl.step
async def translate(translator: BaseTranslator, content: str, source: str = "auto", target: str = "en"):
    return _translate(translator, content, source, target)


# LightRAG-specific translation functions
_lightrag_integrator = None

def get_lightrag_translation_integrator(translator: BaseTranslator = None, detector: LanguageDetector = None):
    """Get or create LightRAG translation integrator instance."""
    global _lightrag_integrator
    
    if not LIGHTRAG_AVAILABLE:
        return None
    
    if _lightrag_integrator is None or translator is not None:
        _lightrag_integrator = LightRAGTranslationIntegrator(
            translator=translator,
            language_detector=detector
        )
    
    return _lightrag_integrator


async def translate_lightrag_response(
    response_data: Dict[str, Any], 
    translator: BaseTranslator, 
    detector: LanguageDetector,
    target_language: str,
    source_language: str = "auto"
) -> Dict[str, Any]:
    """
    Translate LightRAG response data while preserving structure and metadata.
    
    Args:
        response_data: LightRAG response dictionary
        translator: Translation service
        detector: Language detection service
        target_language: Target language code
        source_language: Source language code (auto-detect if "auto")
        
    Returns:
        Translated response dictionary
    """
    if not LIGHTRAG_AVAILABLE:
        # Fallback to simple content translation
        if "content" in response_data:
            response_data["content"] = await translate(
                translator, response_data["content"], source_language, target_language
            )
        return response_data
    
    try:
        integrator = get_lightrag_translation_integrator(translator, detector)
        
        # Check if this is a ProcessedResponse or CombinedResponse structure
        if "sources_used" in response_data and "combination_strategy" in response_data:
            # This is a CombinedResponse structure
            combined_response = CombinedResponse(
                content=response_data.get("content", ""),
                bibliography=response_data.get("bibliography", ""),
                confidence_score=response_data.get("confidence_score", 0.0),
                processing_time=response_data.get("processing_time", 0.0),
                sources_used=response_data.get("sources_used", []),
                combination_strategy=response_data.get("combination_strategy", ""),
                metadata=response_data.get("metadata", {}),
                quality_assessment=response_data.get("quality_assessment", {}),
                citations=response_data.get("citations", [])
            )
            
            translated_dict = await integrator.translate_combined_response(
                combined_response, target_language, source_language
            )
            return translated_dict
            
        else:
            # This is a regular response - convert to ProcessedResponse for translation
            from lightrag_integration.response_integration import ResponseSource
            
            # Determine source type
            source_name = response_data.get("source", "").lower()
            if "lightrag" in source_name:
                source_type = ResponseSource.LIGHTRAG
            elif "perplexity" in source_name:
                source_type = ResponseSource.PERPLEXITY
            else:
                source_type = ResponseSource.LIGHTRAG  # Default
            
            processed_response = ProcessedResponse(
                content=response_data.get("content", ""),
                bibliography=response_data.get("bibliography", ""),
                confidence_score=response_data.get("confidence_score", 0.0),
                source=source_type,
                processing_time=response_data.get("processing_time", 0.0),
                metadata=response_data.get("metadata", {}),
                quality_score=response_data.get("quality_score", 0.8),
                citations=response_data.get("citations", [])
            )
            
            translated_response = await integrator.translate_processed_response(
                processed_response, target_language, source_language
            )
            
            # Convert back to dictionary format
            translated_dict = translated_response.to_dict()
            
            # Preserve original fields that might not be in ProcessedResponse
            for key, value in response_data.items():
                if key not in translated_dict:
                    translated_dict[key] = value
            
            return translated_dict
            
    except Exception as e:
        # Fallback to simple content translation on error
        import logging
        logging.error(f"LightRAG translation failed, using fallback: {str(e)}")
        
        if "content" in response_data:
            response_data["content"] = await translate(
                translator, response_data["content"], source_language, target_language
            )
        
        # Add error metadata
        if "metadata" not in response_data:
            response_data["metadata"] = {}
        response_data["metadata"]["translation_error"] = str(e)
        
        return response_data


def is_lightrag_response(response_data: Dict[str, Any]) -> bool:
    """
    Check if response data is from LightRAG system.
    
    Args:
        response_data: Response dictionary to check
        
    Returns:
        True if response appears to be from LightRAG
    """
    if not isinstance(response_data, dict):
        return False
    
    # Check for LightRAG-specific fields
    lightrag_indicators = [
        "source_documents",
        "entities_used", 
        "relationships_used",
        "formatted_response",
        "confidence_breakdown"
    ]
    
    # Check source field
    source = response_data.get("source", "").lower()
    if "lightrag" in source:
        return True
    
    # Check for LightRAG-specific metadata
    metadata = response_data.get("metadata", {})
    if metadata.get("original_source") == "lightrag":
        return True
    
    # Check for presence of LightRAG-specific fields
    return any(field in response_data for field in lightrag_indicators)
