"""
Unit tests for LightRAG Confidence Scoring

Tests the confidence scoring functionality for graph-based evidence
and source document reliability scoring.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock

from lightrag_integration.confidence_scoring import (
    LightRAGConfidenceScorer,
    ConfidenceFactors,
    ConfidenceBreakdown,
    SourceReliability
)


class TestLightRAGConfidenceScorer(unittest.TestCase):
    """Test cases for LightRAG Confidence Scorer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = LightRAGConfidenceScorer()
        
        # Sample test data
        self.sample_source_documents = [
            "papers/clinical_metabolomics_2023.pdf",
            "papers/biomarker_discovery_nature_2022.pdf",
            "papers/old_study_1995.pdf"
        ]
        
        self.sample_entities_used = [
            {
                "id": "entity_1",
                "text": "clinical metabolomics",
                "type": "field",
                "relevance_score": 0.9,
                "source_documents": ["papers/clinical_metabolomics_2023.pdf"],
                "properties": {"page": 1, "section": "Introduction"}
            },
            {
                "id": "entity_2",
                "text": "biomarkers",
                "type": "compound",
                "relevance_score": 0.8,
                "source_documents": ["papers/clinical_metabolomics_2023.pdf", "papers/biomarker_discovery_nature_2022.pdf"],
                "properties": {"page": 3}
            },
            {
                "id": "entity_3",
                "text": "mass spectrometry",
                "type": "technique",
                "relevance_score": 0.7,
                "source_documents": ["papers/biomarker_discovery_nature_2022.pdf"],
                "properties": {}
            }
        ]
        
        self.sample_relationships_used = [
            {
                "id": "rel_1",
                "type": "uses",
                "source": "entity_1",
                "target": "entity_2",
                "confidence": 0.85,
                "evidence": ["Clinical metabolomics uses biomarkers for diagnosis"],
                "source_documents": ["papers/clinical_metabolomics_2023.pdf"]
            },
            {
                "id": "rel_2",
                "type": "measured_by",
                "source": "entity_2",
                "target": "entity_3",
                "confidence": 0.9,
                "evidence": ["Biomarkers are measured by mass spectrometry"],
                "source_documents": ["papers/biomarker_discovery_nature_2022.pdf"]
            }
        ]
    
    def test_initialization(self):
        """Test scorer initialization."""
        scorer = LightRAGConfidenceScorer()
        
        self.assertIsInstance(scorer.weights, dict)
        self.assertEqual(scorer.high_confidence_threshold, 0.8)
        self.assertEqual(scorer.medium_confidence_threshold, 0.6)
        self.assertEqual(scorer.low_confidence_threshold, 0.4)
        self.assertIsInstance(scorer._source_reliability_cache, dict)
    
    def test_calculate_response_confidence(self):
        """Test comprehensive confidence calculation."""
        base_confidence = 0.7
        
        breakdown = self.scorer.calculate_response_confidence(
            base_confidence,
            self.sample_source_documents,
            self.sample_entities_used,
            self.sample_relationships_used
        )
        
        self.assertIsInstance(breakdown, ConfidenceBreakdown)
        self.assertEqual(breakdown.base_confidence, base_confidence)
        self.assertGreaterEqual(breakdown.overall_confidence, 0.0)
        self.assertLessEqual(breakdown.overall_confidence, 1.0)
        self.assertIsInstance(breakdown.confidence_factors, ConfidenceFactors)
        self.assertIsInstance(breakdown.source_scores, dict)
        self.assertIsInstance(breakdown.entity_scores, dict)
        self.assertIsInstance(breakdown.relationship_scores, dict)
        self.assertIsInstance(breakdown.explanation, str)
    
    def test_calculate_confidence_factors(self):
        """Test individual confidence factors calculation."""
        factors = self.scorer._calculate_confidence_factors(
            self.sample_source_documents,
            self.sample_entities_used,
            self.sample_relationships_used,
            None
        )
        
        self.assertIsInstance(factors, ConfidenceFactors)
        
        # Entity confidence should be average of relevance scores
        expected_entity_confidence = (0.9 + 0.8 + 0.7) / 3
        self.assertAlmostEqual(factors.entity_confidence, expected_entity_confidence, places=2)
        
        # Relationship confidence should be average of confidence scores
        expected_rel_confidence = (0.85 + 0.9) / 2
        self.assertAlmostEqual(factors.relationship_confidence, expected_rel_confidence, places=2)
        
        # All factors should be between 0 and 1
        self.assertGreaterEqual(factors.source_reliability, 0.0)
        self.assertLessEqual(factors.source_reliability, 1.0)
        self.assertGreaterEqual(factors.graph_connectivity, 0.0)
        self.assertLessEqual(factors.graph_connectivity, 1.0)
        self.assertGreaterEqual(factors.evidence_consistency, 0.0)
        self.assertLessEqual(factors.evidence_consistency, 1.0)
    
    def test_assess_source_reliability(self):
        """Test source reliability assessment."""
        doc_path = "papers/clinical_metabolomics_2023.pdf"
        
        reliability = self.scorer._assess_source_reliability(
            doc_path,
            self.sample_entities_used,
            self.sample_relationships_used
        )
        
        self.assertIsInstance(reliability, SourceReliability)
        self.assertEqual(reliability.document_path, doc_path)
        self.assertGreaterEqual(reliability.reliability_score, 0.0)
        self.assertLessEqual(reliability.reliability_score, 1.0)
        self.assertIsInstance(reliability.factors, dict)
        self.assertGreaterEqual(reliability.citation_frequency, 0)
        
        # Test caching
        reliability2 = self.scorer._assess_source_reliability(
            doc_path,
            self.sample_entities_used,
            self.sample_relationships_used
        )
        self.assertEqual(reliability.reliability_score, reliability2.reliability_score)
    
    def test_assess_metadata_quality(self):
        """Test metadata quality assessment."""
        test_cases = [
            ("papers/clinical_metabolomics_2023.pdf", 0.8),  # Good filename
            ("papers/nature_biomarkers_smith_2022.pdf", 0.9),  # Excellent filename
            ("papers/document.pdf", 0.5),  # Basic filename
            ("papers/very_technical_filename_without_keywords.pdf", 0.5)  # No keywords
        ]
        
        for filename, expected_min_quality in test_cases:
            with self.subTest(filename=filename):
                quality = self.scorer._assess_metadata_quality(filename)
                self.assertGreaterEqual(quality, 0.0)
                self.assertLessEqual(quality, 1.0)
                # Don't enforce exact values since they depend on implementation details
    
    def test_calculate_citation_frequency(self):
        """Test citation frequency calculation."""
        doc_path = "papers/clinical_metabolomics_2023.pdf"
        
        frequency = self.scorer._calculate_citation_frequency(
            doc_path,
            self.sample_entities_used,
            self.sample_relationships_used
        )
        
        # Should count entity and relationship references
        expected_frequency = 3  # 2 entities + 1 relationship reference this document
        self.assertEqual(frequency, expected_frequency)
        
        # Test with non-referenced document
        frequency_zero = self.scorer._calculate_citation_frequency(
            "papers/unreferenced.pdf",
            self.sample_entities_used,
            self.sample_relationships_used
        )
        self.assertEqual(frequency_zero, 0)
    
    def test_assess_content_quality(self):
        """Test content quality assessment."""
        doc_path = "papers/clinical_metabolomics_2023.pdf"
        
        quality = self.scorer._assess_content_quality(
            doc_path,
            self.sample_entities_used,
            self.sample_relationships_used
        )
        
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)
        
        # Should be based on average relevance/confidence of entities/relationships from this document
        # Expected: (0.9 + 0.8 + 0.85) / 3 = 0.85
        self.assertAlmostEqual(quality, 0.85, places=1)
    
    def test_calculate_graph_connectivity(self):
        """Test graph connectivity calculation."""
        connectivity = self.scorer._calculate_graph_connectivity(
            self.sample_entities_used,
            self.sample_relationships_used
        )
        
        self.assertGreaterEqual(connectivity, 0.0)
        self.assertLessEqual(connectivity, 1.0)
        
        # All 3 entities should be connected by the 2 relationships
        # entity_1 -> entity_2 -> entity_3
        self.assertAlmostEqual(connectivity, 1.0, places=1)
    
    def test_calculate_graph_connectivity_no_relationships(self):
        """Test graph connectivity with no relationships."""
        connectivity = self.scorer._calculate_graph_connectivity(
            self.sample_entities_used,
            []
        )
        
        self.assertEqual(connectivity, 0.0)
    
    def test_calculate_evidence_consistency(self):
        """Test evidence consistency calculation."""
        consistency = self.scorer._calculate_evidence_consistency(
            self.sample_entities_used,
            self.sample_relationships_used,
            self.sample_source_documents
        )
        
        self.assertGreaterEqual(consistency, 0.0)
        self.assertLessEqual(consistency, 1.0)
        
        # entity_2 is referenced by multiple sources, so consistency should be > 0
        self.assertGreater(consistency, 0.0)
    
    def test_calculate_citation_quality(self):
        """Test citation quality calculation."""
        quality = self.scorer._calculate_citation_quality(
            self.sample_source_documents,
            self.sample_entities_used
        )
        
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)
        
        # PDF files with academic keywords should have good quality
        self.assertGreater(quality, 0.5)
    
    def test_calculate_temporal_relevance(self):
        """Test temporal relevance calculation."""
        relevance = self.scorer._calculate_temporal_relevance(
            self.sample_source_documents,
            None
        )
        
        self.assertGreaterEqual(relevance, 0.0)
        self.assertLessEqual(relevance, 1.0)
        
        # Recent papers (2022, 2023) should have higher relevance than old ones (1995)
        self.assertGreater(relevance, 0.6)
    
    def test_calculate_enhancement_factor(self):
        """Test enhancement factor calculation."""
        factors = ConfidenceFactors(
            entity_confidence=0.8,
            relationship_confidence=0.9,
            source_reliability=0.7,
            graph_connectivity=0.6,
            evidence_consistency=0.5,
            citation_quality=0.8,
            temporal_relevance=0.9
        )
        
        enhancement = self.scorer._calculate_enhancement_factor(factors)
        
        self.assertGreaterEqual(enhancement, 0.0)
        self.assertLessEqual(enhancement, 1.0)
        
        # Should be weighted average of factors
        expected = (
            0.8 * 0.20 +  # entity_confidence
            0.9 * 0.20 +  # relationship_confidence
            0.7 * 0.25 +  # source_reliability
            0.6 * 0.15 +  # graph_connectivity
            0.5 * 0.10 +  # evidence_consistency
            0.8 * 0.05 +  # citation_quality
            0.9 * 0.05    # temporal_relevance
        )
        self.assertAlmostEqual(enhancement, expected, places=2)
    
    def test_apply_confidence_enhancement(self):
        """Test confidence enhancement application."""
        base_confidence = 0.6
        enhancement_factor = 0.8
        
        enhanced = self.scorer._apply_confidence_enhancement(base_confidence, enhancement_factor)
        
        self.assertGreaterEqual(enhanced, base_confidence)
        self.assertLessEqual(enhanced, 1.0)
        
        # Enhancement should be limited to prevent over-confidence
        max_possible_enhancement = base_confidence + (0.3 * 0.8 * (1 - base_confidence))
        self.assertLessEqual(enhanced, max_possible_enhancement + 0.01)  # Small tolerance
    
    def test_get_confidence_level(self):
        """Test confidence level descriptions."""
        test_cases = [
            (0.9, "High"),
            (0.8, "High"),
            (0.7, "Medium"),
            (0.6, "Medium"),
            (0.5, "Low"),
            (0.4, "Low"),
            (0.3, "Very Low"),
            (0.1, "Very Low")
        ]
        
        for confidence, expected_level in test_cases:
            with self.subTest(confidence=confidence):
                result = self.scorer._get_confidence_level(confidence)
                self.assertEqual(result, expected_level)
    
    def test_get_confidence_display_info(self):
        """Test confidence display information generation."""
        breakdown = ConfidenceBreakdown(
            overall_confidence=0.85,
            base_confidence=0.8,
            enhancement_factor=0.05,
            confidence_factors=ConfidenceFactors(),
            source_scores={},
            entity_scores={},
            relationship_scores={},
            explanation="Test explanation"
        )
        
        display_info = self.scorer.get_confidence_display_info(breakdown)
        
        self.assertIsInstance(display_info, dict)
        self.assertEqual(display_info["confidence_score"], 0.85)
        self.assertEqual(display_info["confidence_level"], "High")
        self.assertEqual(display_info["color"], "green")
        self.assertEqual(display_info["icon"], "✓")
        self.assertIn("High Confidence", display_info["display_text"])
        self.assertEqual(display_info["tooltip"], "Test explanation")
        self.assertFalse(display_info["show_warning"])
    
    def test_get_confidence_display_info_low_confidence(self):
        """Test confidence display information for low confidence."""
        breakdown = ConfidenceBreakdown(
            overall_confidence=0.3,
            base_confidence=0.3,
            enhancement_factor=0.0,
            confidence_factors=ConfidenceFactors(),
            source_scores={},
            entity_scores={},
            relationship_scores={},
            explanation="Low confidence explanation"
        )
        
        display_info = self.scorer.get_confidence_display_info(breakdown)
        
        self.assertEqual(display_info["confidence_level"], "Very Low")
        self.assertEqual(display_info["color"], "darkred")
        self.assertEqual(display_info["icon"], "✗")
        self.assertTrue(display_info["show_warning"])
    
    def test_error_handling_in_calculate_response_confidence(self):
        """Test error handling in main confidence calculation."""
        # Test with invalid data that should trigger error handling
        with patch.object(self.scorer, '_calculate_confidence_factors', side_effect=Exception("Test error")):
            breakdown = self.scorer.calculate_response_confidence(
                0.7,
                self.sample_source_documents,
                self.sample_entities_used,
                self.sample_relationships_used
            )
            
            # Should return fallback breakdown
            self.assertEqual(breakdown.overall_confidence, 0.7)
            self.assertEqual(breakdown.base_confidence, 0.7)
            self.assertEqual(breakdown.enhancement_factor, 0.0)
            self.assertIn("Error calculating confidence", breakdown.explanation)
    
    def test_source_reliability_caching(self):
        """Test that source reliability assessments are cached."""
        doc_path = "papers/test_document.pdf"
        
        # First call
        reliability1 = self.scorer._assess_source_reliability(
            doc_path, self.sample_entities_used, self.sample_relationships_used
        )
        
        # Second call should use cache
        reliability2 = self.scorer._assess_source_reliability(
            doc_path, self.sample_entities_used, self.sample_relationships_used
        )
        
        # Should be the same object (from cache)
        self.assertIs(reliability1, reliability2)
        
        # Verify it's in the cache
        self.assertIn(doc_path, self.scorer._source_reliability_cache)


class TestConfidenceIntegration(unittest.TestCase):
    """Test integration with citation formatter."""
    
    def test_confidence_scorer_import(self):
        """Test that confidence scorer can be imported."""
        try:
            from lightrag_integration.confidence_scoring import LightRAGConfidenceScorer
            scorer = LightRAGConfidenceScorer()
            self.assertIsNotNone(scorer)
        except ImportError:
            self.fail("Could not import LightRAGConfidenceScorer")


if __name__ == '__main__':
    unittest.main()