"""
Tests for Clinical Metabolomics Test Suite

This module contains unit tests for the clinical metabolomics test suite,
validating the test framework itself and ensuring proper functionality.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from .clinical_metabolomics_suite import (
    ClinicalMetabolomicsTestSuite,
    TestQuestion,
    TestResult,
    TestSuiteResult,
    run_mvp_validation_test
)
from ..component import LightRAGComponent
from ..config.settings import LightRAGConfig


class TestClinicalMetabolomicsTestSuite:
    """Test cases for the clinical metabolomics test suite."""
    
    def test_test_dataset_creation(self):
        """Test that the test dataset is created properly."""
        test_suite = ClinicalMetabolomicsTestSuite()
        
        # Check that test questions are created
        assert len(test_suite.test_questions) > 0
        
        # Check that the primary test question exists
        primary_question = next(
            (q for q in test_suite.test_questions 
             if "What is clinical metabolomics?" in q.question),
            None
        )
        assert primary_question is not None
        assert primary_question.category == "definition"
        assert primary_question.difficulty == "basic"
        assert len(primary_question.expected_keywords) > 0
        assert len(primary_question.expected_concepts) > 0
    
    def test_test_papers_creation(self):
        """Test creation of test papers dataset."""
        test_suite = ClinicalMetabolomicsTestSuite()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            papers = test_suite.create_test_papers_dataset(temp_dir)
            
            # Check that papers were created
            assert len(papers) > 0
            
            # Check that files exist
            for paper_path in papers:
                assert Path(paper_path).exists()
                assert Path(paper_path).stat().st_size > 0
            
            # Check content of first paper
            first_paper = Path(papers[0])
            content = first_paper.read_text()
            assert "clinical metabolomics" in content.lower()
            assert "metabolites" in content.lower()
    
    @pytest.mark.asyncio
    async def test_single_test_execution(self):
        """Test execution of a single test question."""
        test_suite = ClinicalMetabolomicsTestSuite()
        
        # Create mock component
        mock_component = AsyncMock(spec=LightRAGComponent)
        mock_component.query.return_value = {
            "answer": "Clinical metabolomics is the study of metabolites in biological systems for clinical applications. It involves the identification and quantification of small molecules to understand disease mechanisms and discover biomarkers for diagnosis and personalized medicine.",
            "confidence_score": 0.85,
            "source_documents": ["test_paper.txt"],
            "processing_time": 1.2
        }
        
        # Get test question
        test_question = test_suite.test_questions[0]  # "What is clinical metabolomics?"
        
        # Run single test
        result = await test_suite.run_single_test(mock_component, test_question)
        
        # Validate result
        assert isinstance(result, TestResult)
        assert result.question == test_question.question
        assert result.confidence_score == 0.85
        assert result.processing_time > 0
        assert len(result.keyword_matches) > 0
        assert len(result.concept_matches) > 0
        assert result.accuracy_score > 0
        assert result.passed is True
        assert result.error is None
    
    @pytest.mark.asyncio
    async def test_single_test_failure(self):
        """Test handling of test failures."""
        test_suite = ClinicalMetabolomicsTestSuite()
        
        # Create mock component that raises an exception
        mock_component = AsyncMock(spec=LightRAGComponent)
        mock_component.query.side_effect = Exception("Test error")
        
        # Get test question
        test_question = test_suite.test_questions[0]
        
        # Run single test
        result = await test_suite.run_single_test(mock_component, test_question)
        
        # Validate error handling
        assert isinstance(result, TestResult)
        assert result.passed is False
        assert result.error == "Test error"
        assert result.accuracy_score == 0.0
        assert result.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_test_suite_execution(self):
        """Test execution of the complete test suite."""
        test_suite = ClinicalMetabolomicsTestSuite()
        
        # Create mock component with varied responses
        mock_component = AsyncMock(spec=LightRAGComponent)
        
        def mock_query_response(question):
            if "clinical metabolomics" in question.lower():
                return {
                    "answer": "Clinical metabolomics is the comprehensive study of metabolites in biological systems for clinical research and healthcare applications.",
                    "confidence_score": 0.9,
                    "source_documents": ["paper1.txt"],
                    "processing_time": 1.0
                }
            else:
                return {
                    "answer": "This is a general response about metabolomics research and applications.",
                    "confidence_score": 0.7,
                    "source_documents": ["paper2.txt"],
                    "processing_time": 1.5
                }
        
        mock_component.query.side_effect = mock_query_response
        
        # Run test suite with subset of questions
        test_questions = test_suite.test_questions[:3]  # Test with first 3 questions
        result = await test_suite.run_test_suite(mock_component, test_questions)
        
        # Validate results
        assert isinstance(result, TestSuiteResult)
        assert result.total_questions == 3
        assert result.passed_questions >= 0
        assert result.failed_questions >= 0
        assert result.passed_questions + result.failed_questions == result.total_questions
        assert len(result.results) == 3
        assert result.average_accuracy >= 0
        assert result.average_confidence >= 0
        assert result.average_processing_time > 0
        assert "category_breakdown" in result.summary
        assert "difficulty_breakdown" in result.summary
    
    def test_report_generation(self):
        """Test generation of test reports."""
        test_suite = ClinicalMetabolomicsTestSuite()
        
        # Create mock test results
        from datetime import datetime
        
        mock_results = [
            TestResult(
                question="What is clinical metabolomics?",
                answer="Clinical metabolomics is the study of metabolites...",
                confidence_score=0.9,
                processing_time=1.2,
                keyword_matches=["metabolomics", "clinical", "metabolites"],
                concept_matches=["biomarker discovery"],
                accuracy_score=0.85,
                passed=True
            ),
            TestResult(
                question="What are metabolites?",
                answer="Metabolites are small molecules...",
                confidence_score=0.8,
                processing_time=1.0,
                keyword_matches=["metabolites", "small molecules"],
                concept_matches=["biochemical pathways"],
                accuracy_score=0.75,
                passed=True
            )
        ]
        
        mock_suite_result = TestSuiteResult(
            timestamp=datetime.now(),
            total_questions=2,
            passed_questions=2,
            failed_questions=0,
            average_accuracy=0.8,
            average_confidence=0.85,
            average_processing_time=1.1,
            results=mock_results,
            summary={
                "pass_rate": 1.0,
                "accuracy_distribution": {"min": 0.75, "max": 0.85, "median": 0.8, "std_dev": 0.05},
                "confidence_distribution": {"min": 0.8, "max": 0.9, "median": 0.85, "std_dev": 0.05},
                "performance": {"min_time": 1.0, "max_time": 1.2, "median_time": 1.1},
                "category_breakdown": {"definition": {"pass_rate": 1.0, "average_accuracy": 0.8, "total": 2}},
                "difficulty_breakdown": {"basic": {"pass_rate": 1.0, "average_accuracy": 0.8, "total": 2}}
            }
        )
        
        # Generate report
        report = test_suite.generate_report(mock_suite_result)
        
        # Validate report content
        assert "CLINICAL METABOLOMICS TEST SUITE REPORT" in report
        assert "Total Questions: 2" in report
        assert "Passed: 2" in report
        assert "Failed: 0" in report
        assert "Average Accuracy:" in report
        assert "CATEGORY BREAKDOWN:" in report
        assert "DIFFICULTY BREAKDOWN:" in report
        assert "DETAILED RESULTS:" in report
    
    def test_json_results_saving(self):
        """Test saving results as JSON."""
        test_suite = ClinicalMetabolomicsTestSuite()
        
        # Create mock test result
        from datetime import datetime
        
        mock_result = TestSuiteResult(
            timestamp=datetime.now(),
            total_questions=1,
            passed_questions=1,
            failed_questions=0,
            average_accuracy=0.8,
            average_confidence=0.85,
            average_processing_time=1.1,
            results=[
                TestResult(
                    question="Test question",
                    answer="Test answer",
                    confidence_score=0.85,
                    processing_time=1.1,
                    keyword_matches=["test"],
                    concept_matches=["testing"],
                    accuracy_score=0.8,
                    passed=True
                )
            ],
            summary={"test": "data"}
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_file = Path(temp_dir) / "test_results.json"
            
            # Save results
            test_suite.save_results_json(mock_result, str(output_file))
            
            # Validate file was created and contains valid JSON
            assert output_file.exists()
            
            with open(output_file) as f:
                loaded_data = json.load(f)
            
            assert loaded_data["total_questions"] == 1
            assert loaded_data["passed_questions"] == 1
            assert "timestamp" in loaded_data
            assert len(loaded_data["results"]) == 1


@pytest.mark.asyncio
async def test_mvp_validation_integration():
    """Test the complete MVP validation workflow."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test configuration
        config = LightRAGConfig()
        config.papers_directory = str(Path(temp_dir) / "papers")
        config.knowledge_graph_path = str(Path(temp_dir) / "kg")
        config.vector_store_path = str(Path(temp_dir) / "vectors")
        config.cache_directory = str(Path(temp_dir) / "cache")
        
        # Mock the component to avoid actual LightRAG initialization
        original_component_init = LightRAGComponent.__init__
        original_component_initialize = LightRAGComponent.initialize
        original_component_ingest = LightRAGComponent.ingest_documents
        original_component_query = LightRAGComponent.query
        original_component_cleanup = LightRAGComponent.cleanup
        
        def mock_init(self, config):
            self.config = config
            self._initialized = False
        
        async def mock_initialize(self):
            self._initialized = True
        
        async def mock_ingest(self, documents):
            return {"processed": len(documents), "success": True}
        
        async def mock_query(self, question, context=None):
            return {
                "answer": f"Mock answer for: {question}",
                "confidence_score": 0.8,
                "source_documents": ["mock_paper.txt"],
                "processing_time": 1.0
            }
        
        async def mock_cleanup(self):
            pass
        
        try:
            # Apply mocks
            LightRAGComponent.__init__ = mock_init
            LightRAGComponent.initialize = mock_initialize
            LightRAGComponent.ingest_documents = mock_ingest
            LightRAGComponent.query = mock_query
            LightRAGComponent.cleanup = mock_cleanup
            
            # Run MVP validation test
            results = await run_mvp_validation_test(config)
            
            # Validate results
            assert isinstance(results, TestSuiteResult)
            assert results.total_questions > 0
            assert results.average_processing_time > 0
            
        finally:
            # Restore original methods
            LightRAGComponent.__init__ = original_component_init
            LightRAGComponent.initialize = original_component_initialize
            LightRAGComponent.ingest_documents = original_component_ingest
            LightRAGComponent.query = original_component_query
            LightRAGComponent.cleanup = original_component_cleanup


if __name__ == "__main__":
    pytest.main([__file__, "-v"])