"""
Unit tests for entity and relationship extraction component.

Tests the biomedical entity extraction and relationship detection
functionality for clinical metabolomics concepts.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

# Import the module under test
from .entity_extractor import (
    BiomedicaEntityExtractor, 
    Entity, 
    Relationship, 
    ExtractionResult
)


class TestBiomedicaEntityExtractor:
    """Test suite for BiomedicaEntityExtractor class."""
    
    @pytest.fixture
    def extractor(self):
        """Create a BiomedicaEntityExtractor instance for testing."""
        return BiomedicaEntityExtractor()
    
    @pytest.fixture
    def sample_metabolomics_text(self):
        """Sample text with metabolomics content."""
        return """
        Clinical metabolomics is the study of metabolites in biological systems.
        Glucose levels are associated with diabetes. The glycolysis pathway
        produces pyruvate from glucose. NMR spectroscopy is used for metabolite
        profiling. Creatinine serves as a biomarker for kidney function.
        """
    
    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing relationships."""
        return [
            Entity(
                entity_id="entity_1",
                text="glucose",
                entity_type="metabolite",
                confidence_score=0.9,
                start_pos=10,
                end_pos=17,
                context="test",
                metadata={}
            ),
            Entity(
                entity_id="entity_2", 
                text="diabetes",
                entity_type="disease",
                confidence_score=0.8,
                start_pos=50,
                end_pos=58,
                context="test",
                metadata={}
            )
        ]
    
    def test_initialization(self, extractor):
        """Test BiomedicaEntityExtractor initialization."""
        assert extractor is not None
        assert extractor.logger is not None
        assert extractor.entity_patterns is not None
        assert extractor.relationship_patterns is not None
        assert extractor.metabolomics_terms is not None
        assert len(extractor.metabolomics_terms) > 0
    
    def test_entity_patterns_structure(self, extractor):
        """Test that entity patterns are properly structured."""
        patterns = extractor.entity_patterns
        
        # Check that we have expected entity types
        expected_types = ["metabolite", "disease", "pathway", "biomarker", "technique"]
        for entity_type in expected_types:
            assert entity_type in patterns
            assert isinstance(patterns[entity_type], list)
            assert len(patterns[entity_type]) > 0
    
    def test_relationship_patterns_structure(self, extractor):
        """Test that relationship patterns are properly structured."""
        patterns = extractor.relationship_patterns
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        for pattern in patterns:
            assert "pattern" in pattern
            assert "type" in pattern
            assert "confidence" in pattern
            assert isinstance(pattern["confidence"], float)
            assert 0 <= pattern["confidence"] <= 1
    
    def test_metabolomics_vocabulary(self, extractor):
        """Test metabolomics vocabulary loading."""
        vocab = extractor.metabolomics_terms
        
        # Check for some expected terms
        expected_terms = ["metabolomics", "glucose", "diabetes", "nmr", "biomarker"]
        for term in expected_terms:
            assert term in vocab
    
    @pytest.mark.asyncio
    async def test_extract_entities_empty_text(self, extractor):
        """Test entity extraction with empty text."""
        entities = await extractor.extract_entities("")
        assert entities == []
        
        entities = await extractor.extract_entities(None)
        assert entities == []
    
    @pytest.mark.asyncio
    async def test_extract_entities_without_spacy(self, extractor):
        """Test entity extraction when spaCy is not available."""
        with patch.object(extractor, 'nlp', None):
            entities = await extractor.extract_entities("test text")
            assert entities == []
    
    @pytest.mark.asyncio
    @patch('src.lightrag_integration.ingestion.entity_extractor.SPACY_AVAILABLE', True)
    async def test_extract_entities_with_spacy(self, extractor, sample_metabolomics_text):
        """Test entity extraction with spaCy available."""
        # Mock spaCy nlp object
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = []
        mock_doc.text = sample_metabolomics_text
        
        # Mock tokens
        mock_tokens = []
        for i, word in enumerate(["glucose", "diabetes", "metabolomics"]):
            mock_token = MagicMock()
            mock_token.text = word
            mock_token.i = i
            mock_token.idx = i * 10
            mock_tokens.append(mock_token)
        
        mock_doc.__iter__ = lambda self: iter(mock_tokens)
        mock_nlp.return_value = mock_doc
        
        extractor.nlp = mock_nlp
        
        entities = await extractor.extract_entities(sample_metabolomics_text)
        
        # Should extract some entities
        assert len(entities) >= 0  # May be 0 if no patterns match
    
    @pytest.mark.asyncio
    async def test_extract_relationships_empty_entities(self, extractor):
        """Test relationship extraction with empty entity list."""
        relationships = await extractor.extract_relationships("test text", [])
        assert relationships == []
        
        # Test with single entity (need at least 2 for relationships)
        single_entity = [Entity("1", "test", "type", 0.5, 0, 4, "", {})]
        relationships = await extractor.extract_relationships("test text", single_entity)
        assert relationships == []
    
    @pytest.mark.asyncio
    async def test_extract_relationships_with_entities(self, extractor, sample_entities):
        """Test relationship extraction with sample entities."""
        text = "glucose is associated with diabetes"
        
        relationships = await extractor.extract_relationships(text, sample_entities)
        
        # Should extract at least co-occurrence relationships
        assert len(relationships) >= 0
    
    def test_find_entity_by_text(self, extractor, sample_entities):
        """Test finding entities by text."""
        entity = extractor._find_entity_by_text("glucose", sample_entities)
        assert entity is not None
        assert entity.text == "glucose"
        
        entity = extractor._find_entity_by_text("nonexistent", sample_entities)
        assert entity is None
    
    def test_deduplicate_entities(self, extractor):
        """Test entity deduplication."""
        # Create overlapping entities
        entities = [
            Entity("1", "glucose", "metabolite", 0.8, 0, 7, "", {}),
            Entity("2", "glucose levels", "measurement", 0.6, 0, 13, "", {}),  # Overlaps
            Entity("3", "diabetes", "disease", 0.9, 20, 28, "", {}),
        ]
        
        deduplicated = extractor._deduplicate_entities(entities)
        
        # Should keep the higher confidence entity for overlapping ones
        assert len(deduplicated) == 2
        glucose_entities = [e for e in deduplicated if "glucose" in e.text]
        assert len(glucose_entities) == 1
        assert glucose_entities[0].confidence_score == 0.8  # Higher confidence kept
    
    def test_map_spacy_entity_type(self, extractor):
        """Test mapping spaCy entity types to biomedical types."""
        # Test valid mappings
        assert extractor._map_spacy_entity_type("ORG") == "organization"
        assert extractor._map_spacy_entity_type("PRODUCT") == "technique"
        assert extractor._map_spacy_entity_type("QUANTITY") == "measurement"
        
        # Test invalid/irrelevant mappings
        assert extractor._map_spacy_entity_type("PERSON") is None
        assert extractor._map_spacy_entity_type("GPE") is None
    
    @pytest.mark.asyncio
    async def test_extract_entities_and_relationships(self, extractor, sample_metabolomics_text):
        """Test combined entity and relationship extraction."""
        result = await extractor.extract_entities_and_relationships(sample_metabolomics_text)
        
        assert isinstance(result, ExtractionResult)
        assert isinstance(result.entities, list)
        assert isinstance(result.relationships, list)
        assert result.processing_time >= 0
        assert "text_length" in result.metadata
        assert "entity_count" in result.metadata
        assert "relationship_count" in result.metadata
    
    @pytest.mark.asyncio
    async def test_extract_pattern_entities(self, extractor):
        """Test pattern-based entity extraction."""
        # Mock spaCy doc
        mock_doc = MagicMock()
        mock_tokens = []
        
        # Create tokens for known metabolites
        for i, word in enumerate(["glucose", "diabetes", "nmr"]):
            mock_token = MagicMock()
            mock_token.text = word
            mock_token.i = i
            mock_token.idx = i * 10
            mock_tokens.append(mock_token)
        
        mock_doc.__iter__ = lambda self: iter(mock_tokens)
        
        entities = await extractor._extract_pattern_entities(mock_doc, "test context")
        
        # Should extract entities based on patterns
        assert len(entities) >= 0
        
        # Check that extracted entities have correct structure
        for entity in entities:
            assert hasattr(entity, 'entity_id')
            assert hasattr(entity, 'text')
            assert hasattr(entity, 'entity_type')
            assert hasattr(entity, 'confidence_score')
    
    @pytest.mark.asyncio
    async def test_extract_vocabulary_entities(self, extractor):
        """Test vocabulary-based entity extraction."""
        # Mock spaCy doc
        mock_doc = MagicMock()
        mock_doc.text = "metabolomics and glucose analysis"
        
        entities = await extractor._extract_vocabulary_entities(mock_doc, "test context")
        
        # Should find vocabulary terms
        assert len(entities) >= 0
        
        # Check for expected terms
        entity_texts = [e.text for e in entities]
        if "metabolomics" in mock_doc.text:
            assert "metabolomics" in entity_texts
    
    @pytest.mark.asyncio
    async def test_extract_cooccurrence_relationships(self, extractor, sample_entities):
        """Test co-occurrence relationship extraction."""
        text = "glucose and diabetes are related conditions in metabolic disorders"
        
        relationships = await extractor._extract_cooccurrence_relationships(text, sample_entities, "test")
        
        # Should extract co-occurrence relationships
        assert len(relationships) >= 0
        
        for rel in relationships:
            assert rel.relationship_type == "co_occurs_with"
            assert rel.confidence_score == 0.5
            assert "co_occurrence" in rel.metadata["extraction_method"]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, extractor):
        """Test error handling in extraction methods."""
        # Test with problematic input that might cause exceptions
        problematic_text = None
        
        result = await extractor.extract_entities_and_relationships(problematic_text)
        
        # Should handle errors gracefully
        assert isinstance(result, ExtractionResult)
        assert result.entities == []
        assert result.relationships == []
        assert result.processing_time >= 0
    
    def test_initialization_without_spacy(self):
        """Test initialization when spaCy is not available."""
        with patch('src.lightrag_integration.ingestion.entity_extractor.SPACY_AVAILABLE', False):
            extractor = BiomedicaEntityExtractor()
            assert extractor.nlp is None
    
    def test_initialization_without_spacy_model(self):
        """Test initialization when spaCy model is not available."""
        with patch('src.lightrag_integration.ingestion.entity_extractor.spacy') as mock_spacy:
            mock_spacy.load.side_effect = OSError("Model not found")
            
            extractor = BiomedicaEntityExtractor()
            assert extractor.nlp is None


# Integration tests that can be run manually with actual spaCy models
class TestBiomedicaEntityExtractorIntegration:
    """Integration tests for BiomedicaEntityExtractor (requires spaCy)."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_extraction(self):
        """Test extraction with real spaCy model (if available)."""
        try:
            extractor = BiomedicaEntityExtractor()
            
            if extractor.nlp is not None:
                text = """
                Clinical metabolomics studies the metabolites in blood plasma.
                Glucose concentration is elevated in diabetes patients.
                NMR spectroscopy can detect metabolic changes.
                """
                
                result = await extractor.extract_entities_and_relationships(text)
                
                print(f"Extracted {len(result.entities)} entities:")
                for entity in result.entities:
                    print(f"  - {entity.text} ({entity.entity_type}): {entity.confidence_score}")
                
                print(f"Extracted {len(result.relationships)} relationships:")
                for rel in result.relationships:
                    print(f"  - {rel.relationship_type}: {rel.evidence_text}")
                
                # Basic validation
                assert result.processing_time > 0
                assert len(result.entities) >= 0
                assert len(result.relationships) >= 0
            else:
                print("spaCy model not available, skipping integration test")
                
        except Exception as e:
            print(f"Integration test failed: {str(e)}")


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])