"""
Integration tests for LightRAG translation with main translation system

Tests the integration between LightRAG translation components and the main
translation.py module.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

# Mock chainlit to avoid import issues
import sys
sys.modules['chainlit'] = Mock()

from lightrag_integration.response_integration import ProcessedResponse, ResponseSource


class TestMainTranslationIntegration:
    """Test integration with main translation system."""
    
    @pytest.fixture
    def mock_translator(self):
        """Create a mock translator."""
        translator = Mock()
        translator.get_supported_languages.return_value = {
            'english': 'en',
            'spanish': 'es'
        }
        return translator
    
    @pytest.fixture
    def mock_detector(self):
        """Create a mock language detector."""
        return Mock()
    
    @pytest.fixture
    def sample_lightrag_response(self):
        """Sample LightRAG response for testing."""
        return {
            "content": "Clinical metabolomics is the application of metabolomics [1].",
            "bibliography": "**References:**\n[1]: Research Paper",
            "confidence_score": 0.85,
            "source": "LightRAG",
            "processing_time": 0.5,
            "metadata": {"query_type": "definition"},
            "quality_score": 0.8,
            "citations": [{"id": "1", "source": "Research Paper"}]
        }
    
    @pytest.mark.asyncio
    async def test_translate_lightrag_response_function(self, mock_translator, mock_detector, sample_lightrag_response):
        """Test the translate_lightrag_response function."""
        # Import here to avoid chainlit import issues
        from translation import translate_lightrag_response
        
        # Mock the translation integrator
        with patch('lightrag_integration.translation_integration.LightRAGTranslationIntegrator') as mock_integrator_class:
            mock_integrator = Mock()
            mock_translated_response = Mock()
            mock_translated_response.to_dict.return_value = {
                **sample_lightrag_response,
                "content": "La metabolómica clínica es la aplicación [1].",
                "original_language": "en",
                "target_language": "es",
                "translation_confidence": 0.9
            }
            mock_integrator.translate_processed_response = AsyncMock(return_value=mock_translated_response)
            mock_integrator_class.return_value = mock_integrator
            
            result = await translate_lightrag_response(
                mock_translator,
                mock_detector,
                sample_lightrag_response,
                "es"
            )
            
            assert result["content"] == "La metabolómica clínica es la aplicación [1]."
            assert result["original_language"] == "en"
            assert result["target_language"] == "es"
            assert result["translation_confidence"] == 0.9
    
    @pytest.mark.asyncio
    async def test_translate_lightrag_response_error_handling(self, mock_translator, mock_detector, sample_lightrag_response):
        """Test error handling in translate_lightrag_response."""
        from translation import translate_lightrag_response
        
        # Mock the translation integrator to raise an exception
        with patch('lightrag_integration.translation_integration.LightRAGTranslationIntegrator') as mock_integrator_class:
            mock_integrator_class.side_effect = Exception("Translation failed")
            
            result = await translate_lightrag_response(
                mock_translator,
                mock_detector,
                sample_lightrag_response,
                "es"
            )
            
            # Should return original response with error info
            assert result["content"] == sample_lightrag_response["content"]
            assert "translation_error" in result
            assert result["translation_error"] == "Translation failed"
    
    def test_is_lightrag_response_positive(self):
        """Test is_lightrag_response with LightRAG response."""
        from translation import is_lightrag_response
        
        lightrag_response = {
            "content": "Test content",
            "source": "LightRAG",
            "source_documents": ["doc1.pdf"],
            "entities_used": [{"id": "1", "text": "entity"}]
        }
        
        assert is_lightrag_response(lightrag_response) is True
    
    def test_is_lightrag_response_by_indicators(self):
        """Test is_lightrag_response detection by indicators."""
        from translation import is_lightrag_response
        
        lightrag_response = {
            "content": "Test content",
            "source_documents": ["doc1.pdf"],
            "entities_used": [{"id": "1", "text": "entity"}],
            "relationships_used": [{"id": "1", "type": "related_to"}]
        }
        
        assert is_lightrag_response(lightrag_response) is True
    
    def test_is_lightrag_response_negative(self):
        """Test is_lightrag_response with non-LightRAG response."""
        from translation import is_lightrag_response
        
        perplexity_response = {
            "content": "Test content",
            "source": "Perplexity",
            "bibliography": "References"
        }
        
        assert is_lightrag_response(perplexity_response) is False
    
    def test_is_lightrag_response_insufficient_indicators(self):
        """Test is_lightrag_response with insufficient indicators."""
        from translation import is_lightrag_response
        
        ambiguous_response = {
            "content": "Test content",
            "source_documents": ["doc1.pdf"]  # Only one indicator
        }
        
        assert is_lightrag_response(ambiguous_response) is False


class TestTranslationIntegrationEdgeCases:
    """Test edge cases for translation integration."""
    
    @pytest.mark.asyncio
    async def test_translate_unrecognized_format(self):
        """Test translation with unrecognized response format."""
        from translation import translate_lightrag_response
        
        mock_translator = Mock()
        mock_detector = Mock()
        
        unrecognized_response = {
            "unknown_field": "unknown_value"
        }
        
        result = await translate_lightrag_response(
            mock_translator,
            mock_detector,
            unrecognized_response,
            "es"
        )
        
        # Should return original response unchanged
        assert result == unrecognized_response
    
    def test_is_lightrag_response_empty_dict(self):
        """Test is_lightrag_response with empty dictionary."""
        from translation import is_lightrag_response
        
        assert is_lightrag_response({}) is False
    
    def test_is_lightrag_response_none_values(self):
        """Test is_lightrag_response with None values."""
        from translation import is_lightrag_response
        
        response_with_nones = {
            "content": None,
            "source": None,
            "source_documents": None
        }
        
        assert is_lightrag_response(response_with_nones) is False


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])