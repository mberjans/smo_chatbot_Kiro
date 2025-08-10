"""
Unit tests for Translation Integration

Tests the LightRAGTranslationIntegrator class and related functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from .translation_integration import (
    LightRAGTranslationIntegrator,
    TranslatedResponse
)
from .response_integration import ProcessedResponse, CombinedResponse, ResponseSource


class TestLightRAGTranslationIntegrator:
    """Test cases for LightRAGTranslationIntegrator class."""
    
    @pytest.fixture
    def mock_translator(self):
        """Create a mock translator."""
        translator = Mock()
        translator.get_supported_languages.return_value = {
            'english': 'en',
            'spanish': 'es',
            'french': 'fr'
        }
        return translator
    
    @pytest.fixture
    def mock_language_detector(self):
        """Create a mock language detector."""
        return Mock()
    
    @pytest.fixture
    def integrator(self, mock_translator, mock_language_detector):
        """Create a LightRAGTranslationIntegrator instance for testing."""
        return LightRAGTranslationIntegrator(
            translator=mock_translator,
            language_detector=mock_language_detector
        )
    
    @pytest.fixture
    def sample_processed_response(self):
        """Sample ProcessedResponse for testing."""
        return ProcessedResponse(
            content="Clinical metabolomics is the application of metabolomics to clinical research [1].",
            bibliography="**References:**\n[1]: Research Paper Title\n      (Confidence: 0.8)",
            confidence_score=0.85,
            source=ResponseSource.LIGHTRAG,
            processing_time=0.5,
            metadata={"query_type": "definition"},
            quality_score=0.8,
            citations=[{"id": "1", "source": "Research Paper Title", "confidence": 0.8}]
        )
    
    @pytest.fixture
    def sample_combined_response(self):
        """Sample CombinedResponse for testing."""
        return CombinedResponse(
            content="Clinical metabolomics is a growing field [1]. Recent advances include new techniques [2].",
            bibliography="**References:**\n[1]: Knowledge Base Paper\n[2]: Recent Web Article",
            confidence_score=0.8,
            processing_time=1.0,
            sources_used=[ResponseSource.LIGHTRAG, ResponseSource.PERPLEXITY],
            combination_strategy="parallel_sections",
            metadata={"combined": True},
            quality_assessment={"combined_quality": 0.8},
            citations=[
                {"id": "1", "source": "Knowledge Base Paper"},
                {"id": "2", "source": "Recent Web Article"}
            ]
        )
    
    @pytest.mark.asyncio
    async def test_translate_processed_response_same_language(self, integrator, sample_processed_response):
        """Test translation when source and target languages are the same."""
        with patch.object(integrator, '_detect_language', return_value={"language": "en"}):
            translated = await integrator.translate_processed_response(
                sample_processed_response, 
                target_language="en"
            )
        
        assert isinstance(translated, TranslatedResponse)
        assert translated.content == sample_processed_response.content
        assert translated.original_language == "en"
        assert translated.target_language == "en"
        assert translated.translation_confidence == 1.0
    
    @pytest.mark.asyncio
    async def test_translate_processed_response_different_language(self, integrator, sample_processed_response):
        """Test translation when source and target languages are different."""
        with patch.object(integrator, '_detect_language', return_value={"language": "en"}), \
             patch.object(integrator, '_translate_content', return_value="La metabolómica clínica es la aplicación [1]."), \
             patch.object(integrator, '_translate_bibliography', return_value="**Referencias:**\n[1]: Título del Artículo"):
            
            translated = await integrator.translate_processed_response(
                sample_processed_response, 
                target_language="es"
            )
        
        assert isinstance(translated, TranslatedResponse)
        assert translated.content == "La metabolómica clínica es la aplicación [1]."
        assert translated.original_language == "en"
        assert translated.target_language == "es"
        assert translated.translation_confidence > 0
    
    @pytest.mark.asyncio
    async def test_translate_combined_response(self, integrator, sample_combined_response):
        """Test translation of combined response."""
        with patch.object(integrator, '_detect_language', return_value={"language": "en"}), \
             patch.object(integrator, '_translate_content', return_value="Contenido traducido [1] [2]."), \
             patch.object(integrator, '_translate_bibliography', return_value="**Referencias:**\n[1]: Papel traducido"):
            
            translated = await integrator.translate_combined_response(
                sample_combined_response, 
                target_language="es"
            )
        
        assert isinstance(translated, dict)
        assert translated["content"] == "Contenido traducido [1] [2]."
        assert translated["original_language"] == "en"
        assert translated["target_language"] == "es"
        assert "translation_confidence" in translated
    
    @pytest.mark.asyncio
    async def test_detect_language(self, integrator):
        """Test language detection."""
        # Mock the language detector directly
        integrator.language_detector = Mock()
        
        # Mock the _detect_language function import
        with patch('lightrag_integration.translation_integration._detect_language') as mock_detect:
            mock_detect.return_value = {"language": "es", "confidence_values": {"Spanish": 0.9}}
            
            result = await integrator._detect_language("Hola mundo")
            
            assert result["language"] == "es"
            assert "confidence_values" in result
    
    @pytest.mark.asyncio
    async def test_translate_content_with_citations(self, integrator):
        """Test content translation preserving citations."""
        content = "This is a test [1] with multiple citations [2, 3]."
        
        # Mock the _translate function
        def mock_translate(translator, text, source, target):
            # Simple mock translation
            return text.replace("This is a test", "Esta es una prueba").replace("with multiple citations", "con múltiples citas")
        
        # Ensure translator is set
        integrator.translator = Mock()
        
        with patch('lightrag_integration.translation_integration._translate', side_effect=mock_translate):
            translated = await integrator._translate_content(content, "en", "es")
            
            # Citations should be preserved
            assert "[1]" in translated
            assert "[2, 3]" in translated
            assert "Esta es una prueba" in translated
    
    @pytest.mark.asyncio
    async def test_translate_bibliography_structure_preservation(self, integrator):
        """Test bibliography translation preserving structure."""
        bibliography = """**References:**
[1]: Research Paper Title
      (Confidence: 0.8)
[2]: Another Study
      (Confidence: 0.7)

**Further Reading:**
[3]: Additional Resource"""
        
        def mock_translate(translator, text, source, target):
            # Simple mock translation for headers and titles
            translations = {
                "References": "Referencias",
                "Further Reading": "Lectura Adicional",
                "Research Paper Title": "Título del Artículo de Investigación",
                "Another Study": "Otro Estudio",
                "Additional Resource": "Recurso Adicional"
            }
            return translations.get(text, text)
        
        # Ensure translator is set
        integrator.translator = Mock()
        
        with patch('lightrag_integration.translation_integration._translate', side_effect=mock_translate):
            translated = await integrator._translate_bibliography(bibliography, "en", "es")
            
            # Structure should be preserved
            assert "[1]:" in translated
            assert "[2]:" in translated
            assert "[3]:" in translated
            assert "(Confidence: 0.8)" in translated
            # Check that basic structure is maintained
            assert "References" in translated or "Referencias" in translated
    
    def test_calculate_translation_confidence(self, integrator):
        """Test translation confidence calculation."""
        original = "This is a test sentence with citations [1]."
        translated = "Esta es una oración de prueba con citas [1]."
        
        confidence = integrator._calculate_translation_confidence(original, translated, 0.8)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident
    
    def test_calculate_translation_confidence_missing_citations(self, integrator):
        """Test translation confidence when citations are missing."""
        original = "This is a test sentence with citations [1] [2]."
        translated = "Esta es una oración de prueba con citas [1]."  # Missing [2]
        
        confidence = integrator._calculate_translation_confidence(original, translated, 0.8)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence < 0.8  # Should be lower due to missing citation
    
    def test_get_supported_languages(self, integrator):
        """Test getting supported languages."""
        languages = integrator.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert 'en' in languages or 'es' in languages or 'fr' in languages
    
    def test_is_translation_needed(self, integrator):
        """Test translation necessity check."""
        # Same language
        assert not integrator.is_translation_needed("en", "en")
        assert not integrator.is_translation_needed("en-us", "en-gb")  # Variants
        assert not integrator.is_translation_needed("zh-cn", "zh-tw")  # Chinese variants
        
        # Different languages
        assert integrator.is_translation_needed("en", "es")
        assert integrator.is_translation_needed("fr", "de")
        
        # Edge cases
        assert not integrator.is_translation_needed("", "en")
        assert not integrator.is_translation_needed("en", "")
        assert not integrator.is_translation_needed(None, "en")
    
    def test_create_untranslated_response(self, integrator, sample_processed_response):
        """Test creation of untranslated response."""
        untranslated = integrator._create_untranslated_response(
            sample_processed_response, "en", "en"
        )
        
        assert isinstance(untranslated, TranslatedResponse)
        assert untranslated.content == sample_processed_response.content
        assert untranslated.translation_confidence == 1.0
        assert untranslated.original_language == "en"
        assert untranslated.target_language == "en"
        assert "translation" in untranslated.metadata
        assert untranslated.metadata["translation"]["skipped_reason"] == "same_language"
    
    @pytest.mark.asyncio
    async def test_error_handling_in_translation(self, integrator, sample_processed_response):
        """Test error handling during translation."""
        with patch.object(integrator, '_detect_language', side_effect=Exception("Detection failed")):
            translated = await integrator.translate_processed_response(
                sample_processed_response, 
                target_language="es"
            )
        
        assert isinstance(translated, TranslatedResponse)
        assert translated.content == sample_processed_response.content  # Should return original
        assert translated.translation_confidence == 0.0
        assert "translation_error" in translated.metadata
    
    def test_translated_response_to_dict(self, integrator):
        """Test TranslatedResponse to_dict method."""
        response = TranslatedResponse(
            content="Test content",
            bibliography="Test bibliography",
            confidence_score=0.8,
            source=ResponseSource.LIGHTRAG,
            processing_time=0.5,
            metadata={"test": "data"},
            quality_score=0.7,
            citations=[{"id": "1", "source": "test.pdf"}],
            original_language="en",
            target_language="es",
            translation_confidence=0.9
        )
        
        response_dict = response.to_dict()
        
        assert response_dict["content"] == "Test content"
        assert response_dict["source"] == "lightrag"
        assert response_dict["original_language"] == "en"
        assert response_dict["target_language"] == "es"
        assert response_dict["translation_confidence"] == 0.9


class TestTranslationIntegrationEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def integrator_no_translator(self):
        """Create integrator without translator."""
        return LightRAGTranslationIntegrator(translator=None, language_detector=None)
    
    @pytest.mark.asyncio
    async def test_translation_without_translator(self, integrator_no_translator):
        """Test translation when no translator is available."""
        response = ProcessedResponse(
            content="Test content",
            bibliography="Test bib",
            confidence_score=0.8,
            source=ResponseSource.LIGHTRAG,
            processing_time=0.1,
            metadata={},
            quality_score=0.7,
            citations=[]
        )
        
        translated = await integrator_no_translator.translate_processed_response(
            response, target_language="es"
        )
        
        assert translated.content == "Test content"  # Should remain unchanged
        assert translated.original_language == "en"  # Default fallback
        assert translated.target_language == "es"
    
    @pytest.mark.asyncio
    async def test_translate_empty_content(self, integrator_no_translator):
        """Test translation of empty content."""
        content = ""
        translated = await integrator_no_translator._translate_content(content, "en", "es")
        assert translated == ""
    
    @pytest.mark.asyncio
    async def test_translate_empty_bibliography(self, integrator_no_translator):
        """Test translation of empty bibliography."""
        bibliography = ""
        translated = await integrator_no_translator._translate_bibliography(bibliography, "en", "es")
        assert translated == ""
    
    def test_confidence_calculation_edge_cases(self, integrator_no_translator):
        """Test confidence calculation with edge cases."""
        # Empty content
        confidence = integrator_no_translator._calculate_translation_confidence("", "", 0.5)
        assert 0.0 <= confidence <= 1.0
        
        # Very different lengths
        original = "Short"
        translated = "This is a much longer translation that doesn't match the original length at all"
        confidence = integrator_no_translator._calculate_translation_confidence(original, translated, 0.8)
        assert confidence < 0.8  # Should be penalized for length mismatch
    
    def test_get_supported_languages_no_translator(self, integrator_no_translator):
        """Test getting supported languages when no translator is available."""
        languages = integrator_no_translator.get_supported_languages()
        assert isinstance(languages, list)
        assert 'en' in languages  # Should at least have English fallback


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])