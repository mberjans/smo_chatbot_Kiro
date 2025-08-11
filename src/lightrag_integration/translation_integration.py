"""
Translation Integration for LightRAG Responses

This module extends the existing translation system to handle LightRAG-specific
response formats and metadata. Implements requirements 4.1 and 4.4.
"""

import re
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

from .utils.logging import setup_logger
from .response_integration import ProcessedResponse, CombinedResponse, ResponseSource

# Import translation functions (will be imported dynamically to avoid dependency issues)
def _detect_language(*args, **kwargs):
    """Dynamic import wrapper for _detect_language."""
    from translation import _detect_language as detect_func
    return detect_func(*args, **kwargs)

def _translate(*args, **kwargs):
    """Dynamic import wrapper for _translate."""
    from translation import _translate as translate_func
    return translate_func(*args, **kwargs)


@dataclass
class TranslatedResponse:
    """Response after translation processing."""
    content: str
    bibliography: str
    confidence_score: float
    source: ResponseSource
    processing_time: float
    metadata: Dict[str, Any]
    quality_score: float
    citations: List[Dict[str, Any]]
    original_language: str
    target_language: str
    translation_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "content": self.content,
            "bibliography": self.bibliography,
            "confidence_score": self.confidence_score,
            "source": self.source.value if hasattr(self.source, 'value') else str(self.source),
            "processing_time": self.processing_time,
            "metadata": self.metadata,
            "quality_score": self.quality_score,
            "citations": self.citations,
            "original_language": self.original_language,
            "target_language": self.target_language,
            "translation_confidence": self.translation_confidence
        }


class TranslationIntegrator:
    """Alias for LightRAGTranslationIntegrator for backward compatibility."""
    
    def __init__(self, config=None):
        self._integrator = LightRAGTranslationIntegrator()
    
    def __getattr__(self, name):
        return getattr(self._integrator, name)

class LightRAGTranslationIntegrator:
    """
    Integrates LightRAG responses with the existing translation system.
    
    This class handles translation of LightRAG responses while preserving
    citations, metadata, and confidence scores.
    """
    
    def __init__(self, translator=None, language_detector=None):
        """
        Initialize the translation integrator.
        
        Args:
            translator: Translation service (from existing system)
            language_detector: Language detection service (from existing system)
        """
        self.translator = translator
        self.language_detector = language_detector
        self.logger = setup_logger("lightrag_translation")
        
        # Translation quality thresholds
        self.high_translation_confidence = 0.8
        self.medium_translation_confidence = 0.6
        
        self.logger.info("LightRAG Translation Integrator initialized")
    
    async def translate_processed_response(
        self, 
        response: ProcessedResponse, 
        target_language: str,
        source_language: str = "auto"
    ) -> TranslatedResponse:
        """
        Translate a ProcessedResponse to the target language.
        
        Args:
            response: ProcessedResponse to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if "auto")
            
        Returns:
            TranslatedResponse with translated content
        """
        try:
            start_time = time.time()
            
            # Detect source language if needed
            if source_language == "auto" and self.language_detector:
                detected = await self._detect_language(response.content)
                source_language = detected.get("language", "en")
            elif source_language == "auto":
                source_language = "en"  # Default fallback
            
            # Skip translation if source and target are the same
            if source_language == target_language:
                return self._create_untranslated_response(response, source_language, target_language)
            
            # Translate main content
            translated_content = await self._translate_content(
                response.content, source_language, target_language
            )
            
            # Translate bibliography (preserving structure)
            translated_bibliography = await self._translate_bibliography(
                response.bibliography, source_language, target_language
            )
            
            # Update metadata with translation info
            updated_metadata = {
                **response.metadata,
                "translation": {
                    "source_language": source_language,
                    "target_language": target_language,
                    "translation_time": time.time() - start_time
                }
            }
            
            # Calculate translation confidence
            translation_confidence = self._calculate_translation_confidence(
                response.content, translated_content, response.confidence_score
            )
            
            translated_response = TranslatedResponse(
                content=translated_content,
                bibliography=translated_bibliography,
                confidence_score=response.confidence_score,
                source=response.source,
                processing_time=response.processing_time + (time.time() - start_time),
                metadata=updated_metadata,
                quality_score=response.quality_score,
                citations=response.citations,  # Citations remain in original language
                original_language=source_language,
                target_language=target_language,
                translation_confidence=translation_confidence
            )
            
            self.logger.info(f"Response translated from {source_language} to {target_language}")
            return translated_response
            
        except Exception as e:
            self.logger.error(f"Error translating response: {str(e)}", exc_info=True)
            
            # Return original response with error metadata
            return TranslatedResponse(
                content=response.content,
                bibliography=response.bibliography,
                confidence_score=response.confidence_score,
                source=response.source,
                processing_time=response.processing_time,
                metadata={**response.metadata, "translation_error": str(e)},
                quality_score=response.quality_score,
                citations=response.citations,
                original_language=source_language,
                target_language=target_language,
                translation_confidence=0.0
            )
    
    async def translate_combined_response(
        self, 
        response: CombinedResponse, 
        target_language: str,
        source_language: str = "auto"
    ) -> Dict[str, Any]:
        """
        Translate a CombinedResponse to the target language.
        
        Args:
            response: CombinedResponse to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if "auto")
            
        Returns:
            Dictionary with translated combined response
        """
        try:
            start_time = time.time()
            
            # Detect source language if needed
            if source_language == "auto" and self.language_detector:
                detected = await self._detect_language(response.content)
                source_language = detected.get("language", "en")
            elif source_language == "auto":
                source_language = "en"  # Default fallback
            
            # Skip translation if source and target are the same
            if source_language == target_language:
                return {
                    **response.to_dict(),
                    "original_language": source_language,
                    "target_language": target_language,
                    "translation_confidence": 1.0
                }
            
            # Translate main content
            translated_content = await self._translate_content(
                response.content, source_language, target_language
            )
            
            # Translate bibliography (preserving structure)
            translated_bibliography = await self._translate_bibliography(
                response.bibliography, source_language, target_language
            )
            
            # Update metadata with translation info
            updated_metadata = {
                **response.metadata,
                "translation": {
                    "source_language": source_language,
                    "target_language": target_language,
                    "translation_time": time.time() - start_time
                }
            }
            
            # Calculate translation confidence
            translation_confidence = self._calculate_translation_confidence(
                response.content, translated_content, response.confidence_score
            )
            
            translated_response = {
                "content": translated_content,
                "bibliography": translated_bibliography,
                "confidence_score": response.confidence_score,
                "processing_time": response.processing_time + (time.time() - start_time),
                "sources_used": [source.value for source in response.sources_used],
                "combination_strategy": response.combination_strategy,
                "metadata": updated_metadata,
                "quality_assessment": response.quality_assessment,
                "citations": response.citations,  # Citations remain in original language
                "original_language": source_language,
                "target_language": target_language,
                "translation_confidence": translation_confidence
            }
            
            self.logger.info(f"Combined response translated from {source_language} to {target_language}")
            return translated_response
            
        except Exception as e:
            self.logger.error(f"Error translating combined response: {str(e)}", exc_info=True)
            
            # Return original response with error metadata
            return {
                **response.to_dict(),
                "original_language": source_language,
                "target_language": target_language,
                "translation_confidence": 0.0,
                "translation_error": str(e)
            }
    
    async def _detect_language(self, content: str) -> Dict[str, Any]:
        """Detect language of content using the existing language detector."""
        try:
            if self.language_detector:
                # Use the existing language detection function
                return _detect_language(self.language_detector, content)
            else:
                # Fallback to simple heuristics
                return {"language": "en", "confidence_values": {"English": 0.5}}
        except Exception as e:
            self.logger.warning(f"Language detection failed: {str(e)}")
            return {"language": "en", "confidence_values": {"English": 0.5}}
    
    async def _translate_content(self, content: str, source_lang: str, target_lang: str) -> str:
        """Translate content while preserving citations and formatting."""
        if not content or not self.translator:
            return content
        
        try:
            # Extract citation markers to preserve them
            citation_pattern = r'\[(\d+(?:,\s*\d+)*(?:-\d+)*)\]'
            citations = re.findall(citation_pattern, content)
            
            # Replace citations with placeholders
            placeholder_content = content
            citation_placeholders = {}
            for i, citation in enumerate(citations):
                placeholder = f"__CITATION_{i}__"
                citation_placeholders[placeholder] = f"[{citation}]"
                placeholder_content = placeholder_content.replace(f"[{citation}]", placeholder, 1)
            
            # Translate the content with placeholders
            translated_content = _translate(
                self.translator, 
                placeholder_content, 
                source=source_lang, 
                target=target_lang
            )
            
            # Restore citation markers
            for placeholder, citation in citation_placeholders.items():
                translated_content = translated_content.replace(placeholder, citation)
            
            return translated_content
            
        except Exception as e:
            self.logger.error(f"Content translation failed: {str(e)}")
            return content  # Return original on error
    
    async def _translate_bibliography(self, bibliography: str, source_lang: str, target_lang: str) -> str:
        """Translate bibliography while preserving structure and citations."""
        if not bibliography or not self.translator:
            return bibliography
        
        try:
            # Split bibliography into sections
            sections = bibliography.split('\n\n')
            translated_sections = []
            
            for section in sections:
                if not section.strip():
                    translated_sections.append(section)
                    continue
                
                # Check if this is a header section (contains **)
                if '**' in section:
                    # Translate headers but preserve formatting
                    header_pattern = r'\*\*(.*?)\*\*'
                    headers = re.findall(header_pattern, section)
                    translated_section = section
                    
                    for header in headers:
                        if header.strip():
                            translated_header = _translate(
                                self.translator, 
                                header, 
                                source=source_lang, 
                                target=target_lang
                            )
                            translated_section = translated_section.replace(
                                f"**{header}**", 
                                f"**{translated_header}**"
                            )
                    
                    translated_sections.append(translated_section)
                else:
                    # For citation lines, only translate the descriptive parts
                    lines = section.split('\n')
                    translated_lines = []
                    
                    for line in lines:
                        if line.strip().startswith('[') and ']:' in line:
                            # This is a citation line - preserve citation format
                            citation_match = re.match(r'(\[\d+\]:\s*)(.*?)(\s*\(Confidence:.*\))?$', line)
                            if citation_match:
                                prefix, source_text, confidence_suffix = citation_match.groups()
                                
                                # Don't translate URLs or DOIs
                                if not (source_text.startswith(('http://', 'https://')) or 'doi:' in source_text.lower()):
                                    translated_source = _translate(
                                        self.translator, 
                                        source_text, 
                                        source=source_lang, 
                                        target=target_lang
                                    )
                                    line = prefix + translated_source + (confidence_suffix or '')
                            
                            translated_lines.append(line)
                        else:
                            # Regular line - translate if not empty
                            if line.strip():
                                translated_line = _translate(
                                    self.translator, 
                                    line, 
                                    source=source_lang, 
                                    target=target_lang
                                )
                                translated_lines.append(translated_line)
                            else:
                                translated_lines.append(line)
                    
                    translated_sections.append('\n'.join(translated_lines))
            
            return '\n\n'.join(translated_sections)
            
        except Exception as e:
            self.logger.error(f"Bibliography translation failed: {str(e)}")
            return bibliography  # Return original on error
    
    def _calculate_translation_confidence(
        self, 
        original_content: str, 
        translated_content: str, 
        original_confidence: float
    ) -> float:
        """Calculate confidence score for the translation."""
        try:
            # Simple heuristics for translation confidence
            confidence_factors = []
            
            # Length similarity factor
            original_length = len(original_content.split())
            translated_length = len(translated_content.split())
            
            if original_length > 0:
                length_ratio = min(translated_length, original_length) / max(translated_length, original_length)
                confidence_factors.append(length_ratio)
            else:
                confidence_factors.append(0.5)
            
            # Citation preservation factor
            original_citations = len(re.findall(r'\[\d+\]', original_content))
            translated_citations = len(re.findall(r'\[\d+\]', translated_content))
            
            if original_citations > 0:
                citation_preservation = min(translated_citations, original_citations) / original_citations
                confidence_factors.append(citation_preservation)
            else:
                confidence_factors.append(1.0)  # No citations to preserve
            
            # Base confidence factor (from original response)
            confidence_factors.append(original_confidence)
            
            # Calculate weighted average
            translation_confidence = sum(confidence_factors) / len(confidence_factors)
            
            return min(translation_confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating translation confidence: {str(e)}")
            return 0.5  # Default moderate confidence
    
    def _create_untranslated_response(
        self, 
        response: ProcessedResponse, 
        source_language: str, 
        target_language: str
    ) -> TranslatedResponse:
        """Create a TranslatedResponse for cases where no translation is needed."""
        return TranslatedResponse(
            content=response.content,
            bibliography=response.bibliography,
            confidence_score=response.confidence_score,
            source=response.source,
            processing_time=response.processing_time,
            metadata={
                **response.metadata,
                "translation": {
                    "source_language": source_language,
                    "target_language": target_language,
                    "translation_time": 0.0,
                    "skipped_reason": "same_language"
                }
            },
            quality_score=response.quality_score,
            citations=response.citations,
            original_language=source_language,
            target_language=target_language,
            translation_confidence=1.0
        )
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages from the translator."""
        try:
            if self.translator and hasattr(self.translator, 'get_supported_languages'):
                return list(self.translator.get_supported_languages(as_dict=True).values())
            else:
                # Return common language codes as fallback
                return ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh-cn', 'ar']
        except Exception as e:
            self.logger.error(f"Error getting supported languages: {str(e)}")
            return ['en']  # Fallback to English only
    
    def is_translation_needed(self, source_language: str, target_language: str) -> bool:
        """Check if translation is needed between source and target languages."""
        if not source_language or not target_language:
            return False
        
        # Normalize language codes
        source_normalized = source_language.lower().replace('_', '-')
        target_normalized = target_language.lower().replace('_', '-')
        
        # Handle common variations
        language_mappings = {
            'zh-cn': 'zh',
            'zh-tw': 'zh',
            'en-us': 'en',
            'en-gb': 'en'
        }
        
        source_normalized = language_mappings.get(source_normalized, source_normalized)
        target_normalized = language_mappings.get(target_normalized, target_normalized)
        
        return source_normalized != target_normalized