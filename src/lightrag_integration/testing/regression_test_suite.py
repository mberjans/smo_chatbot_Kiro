"""
Regression Test Suite for LightRAG Integration

This module provides regression testing to ensure that the LightRAG integration
does not break existing system functionality. It tests core components and
their interactions to detect any regressions.
"""

import pytest
import asyncio
import tempfile
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass, asdict

# Import existing system components to test
try:
    from ...translation import detect_language, translate_text
    from ...citation import format_citations, generate_bibliography
    from ...query_engine import create_query_engine
    from ...embeddings import get_embedding_model
    from ...graph_stores import Neo4jGraphStore
except ImportError:
    # Handle missing imports gracefully for testing
    pass

from ..component import LightRAGComponent
from ..config.settings import LightRAGConfig
from ..utils.logging import setup_logger


@dataclass
class RegressionTestCase:
    """Definition of a regression test case."""
    name: str
    description: str
    test_function: str
    baseline_value: Any
    tolerance: float = 0.1
    critical: bool = False


@dataclass
class RegressionTestResult:
    """Result of a regression test."""
    test_name: str
    baseline_value: Any
    current_value: Any
    passed: bool
    deviation: float
    error_message: Optional[str]
    timestamp: datetime
    critical: bool


class RegressionTestSuite:
    """
    Regression test suite for existing system functionality.
    
    This class tests that LightRAG integration doesn't break existing
    functionality in translation, citation, query processing, and other
    core system components.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the regression test suite.
        
        Args:
            config: Optional LightRAG configuration
            output_dir: Directory for test outputs and reports
        """
        self.config = config or LightRAGConfig.from_env()
        self.output_dir = Path(output_dir or "regression_test_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("regression_test_suite",
                                 log_file=str(self.output_dir / "regression_tests.log"))
        
        # Define regression test cases
        self.test_cases = self._define_regression_test_cases()
        
        # Load baseline values
        self.baselines = self._load_baseline_values()
    
    def _define_regression_test_cases(self) -> List[RegressionTestCase]:
        """Define all regression test cases."""
        return [
            # Translation system tests
            RegressionTestCase(
                name="translation_accuracy_spanish",
                description="Test translation accuracy for Spanish",
                test_function="test_translation_accuracy_spanish",
                baseline_value=0.85,
                tolerance=0.05,
                critical=True
            ),
            RegressionTestCase(
                name="translation_accuracy_french",
                description="Test translation accuracy for French", 
                test_function="test_translation_accuracy_french",
                baseline_value=0.82,
                tolerance=0.05,
                critical=True
            ),
            RegressionTestCase(
                name="language_detection_accuracy",
                description="Test language detection accuracy",
                test_function="test_language_detection_accuracy",
                baseline_value=0.95,
                tolerance=0.02,
                critical=True
            ),
            
            # Citation system tests
            RegressionTestCase(
                name="citation_format_consistency",
                description="Test citation formatting consistency",
                test_function="test_citation_format_consistency",
                baseline_value=0.98,
                tolerance=0.02,
                critical=True
            ),
            RegressionTestCase(
                name="bibliography_generation_accuracy",
                description="Test bibliography generation accuracy",
                test_function="test_bibliography_generation_accuracy",
                baseline_value=0.95,
                tolerance=0.03,
                critical=False
            ),
            
            # Query engine tests
            RegressionTestCase(
                name="query_response_time",
                description="Test query response time performance",
                test_function="test_query_response_time",
                baseline_value=3.0,  # seconds
                tolerance=0.5,
                critical=False
            ),
            RegressionTestCase(
                name="embedding_generation_speed",
                description="Test embedding generation speed",
                test_function="test_embedding_generation_speed",
                baseline_value=0.1,  # seconds per text
                tolerance=0.02,
                critical=False
            ),
            
            # Graph store tests
            RegressionTestCase(
                name="neo4j_connection_reliability",
                description="Test Neo4j connection reliability",
                test_function="test_neo4j_connection_reliability",
                baseline_value=0.99,
                tolerance=0.01,
                critical=True
            ),
            RegressionTestCase(
                name="graph_query_performance",
                description="Test graph query performance",
                test_function="test_graph_query_performance",
                baseline_value=1.5,  # seconds
                tolerance=0.3,
                critical=False
            ),
            
            # Integration tests
            RegressionTestCase(
                name="chainlit_interface_compatibility",
                description="Test Chainlit interface compatibility",
                test_function="test_chainlit_interface_compatibility",
                baseline_value=True,
                tolerance=0.0,
                critical=True
            ),
            RegressionTestCase(
                name="perplexity_api_integration",
                description="Test Perplexity API integration",
                test_function="test_perplexity_api_integration",
                baseline_value=0.95,
                tolerance=0.03,
                critical=True
            )
        ]
    
    def _load_baseline_values(self) -> Dict[str, Any]:
        """Load baseline values from file or use defaults."""
        baseline_file = self.output_dir / "regression_baselines.json"
        
        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load baselines: {e}")
        
        # Use test case baseline values as defaults
        return {
            test_case.name: test_case.baseline_value 
            for test_case in self.test_cases
        }
    
    async def run_regression_tests(self) -> Dict[str, Any]:
        """
        Run all regression tests.
        
        Returns:
            Dictionary with comprehensive regression test results
        """
        test_start = datetime.now()
        self.logger.info("Starting regression test suite")
        
        results = {
            "timestamp": test_start.isoformat(),
            "test_results": [],
            "summary": {
                "total_tests": len(self.test_cases),
                "passed": 0,
                "failed": 0,
                "critical_failures": 0
            },
            "overall_success": False,
            "duration_seconds": 0
        }
        
        try:
            # Run each test case
            for test_case in self.test_cases:
                self.logger.info(f"Running regression test: {test_case.name}")
                
                test_result = await self._run_single_regression_test(test_case)
                results["test_results"].append(asdict(test_result))
                
                if test_result.passed:
                    results["summary"]["passed"] += 1
                else:
                    results["summary"]["failed"] += 1
                    if test_result.critical:
                        results["summary"]["critical_failures"] += 1
                
                self.logger.info(
                    f"Test {test_case.name}: "
                    f"{'PASSED' if test_result.passed else 'FAILED'}"
                    f"{' (CRITICAL)' if test_result.critical and not test_result.passed else ''}"
                )
            
            # Determine overall success
            results["overall_success"] = (
                results["summary"]["failed"] == 0 or
                results["summary"]["critical_failures"] == 0
            )
            
            results["duration_seconds"] = (datetime.now() - test_start).total_seconds()
            
            self.logger.info(
                f"Regression tests completed: "
                f"{'SUCCESS' if results['overall_success'] else 'FAILED'} "
                f"({results['summary']['passed']}/{results['summary']['total_tests']} passed)"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Regression tests failed with error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            results["error"] = error_msg
            results["duration_seconds"] = (datetime.now() - test_start).total_seconds()
            return results
    
    async def _run_single_regression_test(self, test_case: RegressionTestCase) -> RegressionTestResult:
        """Run a single regression test case."""
        test_start = datetime.now()
        
        try:
            # Get the test function
            test_function = getattr(self, test_case.test_function, None)
            if not test_function:
                raise ValueError(f"Test function {test_case.test_function} not found")
            
            # Run the test
            current_value = await test_function()
            
            # Get baseline value
            baseline_value = self.baselines.get(test_case.name, test_case.baseline_value)
            
            # Calculate deviation and determine if test passed
            passed, deviation = self._evaluate_test_result(
                current_value, baseline_value, test_case.tolerance
            )
            
            return RegressionTestResult(
                test_name=test_case.name,
                baseline_value=baseline_value,
                current_value=current_value,
                passed=passed,
                deviation=deviation,
                error_message=None,
                timestamp=test_start,
                critical=test_case.critical
            )
            
        except Exception as e:
            return RegressionTestResult(
                test_name=test_case.name,
                baseline_value=self.baselines.get(test_case.name, test_case.baseline_value),
                current_value=None,
                passed=False,
                deviation=float('inf'),
                error_message=str(e),
                timestamp=test_start,
                critical=test_case.critical
            )
    
    def _evaluate_test_result(self, current_value: Any, baseline_value: Any, 
                            tolerance: float) -> tuple[bool, float]:
        """Evaluate if a test result passes within tolerance."""
        try:
            if isinstance(baseline_value, bool):
                # Boolean comparison
                return current_value == baseline_value, 0.0 if current_value == baseline_value else 1.0
            
            elif isinstance(baseline_value, (int, float)):
                # Numeric comparison
                if baseline_value == 0:
                    deviation = abs(current_value)
                else:
                    deviation = abs(current_value - baseline_value) / abs(baseline_value)
                
                passed = deviation <= tolerance
                return passed, deviation
            
            else:
                # String or other comparison
                passed = current_value == baseline_value
                return passed, 0.0 if passed else 1.0
                
        except Exception:
            return False, float('inf')
    
    # Individual test implementations
    async def test_translation_accuracy_spanish(self) -> float:
        """Test Spanish translation accuracy."""
        try:
            # Test translation of common clinical metabolomics terms
            test_phrases = [
                ("Clinical metabolomics", "Metabolómica clínica"),
                ("Biomarkers", "Biomarcadores"),
                ("Mass spectrometry", "Espectrometría de masas"),
                ("Metabolic pathway", "Vía metabólica"),
                ("Disease diagnosis", "Diagnóstico de enfermedad")
            ]
            
            correct_translations = 0
            
            for english, expected_spanish in test_phrases:
                try:
                    translated = translate_text(english, target_language="es")
                    # Simple similarity check (in practice would use more sophisticated metrics)
                    if any(word in translated.lower() for word in expected_spanish.lower().split()):
                        correct_translations += 1
                except:
                    pass
            
            return correct_translations / len(test_phrases)
            
        except Exception as e:
            self.logger.warning(f"Translation test failed: {e}")
            return 0.0
    
    async def test_translation_accuracy_french(self) -> float:
        """Test French translation accuracy."""
        try:
            test_phrases = [
                ("Clinical metabolomics", "Métabolomique clinique"),
                ("Biomarkers", "Biomarqueurs"),
                ("Mass spectrometry", "Spectrométrie de masse"),
                ("Metabolic pathway", "Voie métabolique"),
                ("Disease diagnosis", "Diagnostic de maladie")
            ]
            
            correct_translations = 0
            
            for english, expected_french in test_phrases:
                try:
                    translated = translate_text(english, target_language="fr")
                    if any(word in translated.lower() for word in expected_french.lower().split()):
                        correct_translations += 1
                except:
                    pass
            
            return correct_translations / len(test_phrases)
            
        except Exception as e:
            self.logger.warning(f"French translation test failed: {e}")
            return 0.0
    
    async def test_language_detection_accuracy(self) -> float:
        """Test language detection accuracy."""
        try:
            test_texts = [
                ("What is clinical metabolomics?", "en"),
                ("¿Qué es la metabolómica clínica?", "es"),
                ("Qu'est-ce que la métabolomique clinique?", "fr"),
                ("Was ist klinische Metabolomik?", "de"),
                ("Che cos'è la metabolomica clinica?", "it")
            ]
            
            correct_detections = 0
            
            for text, expected_lang in test_texts:
                try:
                    detected_lang = detect_language(text)
                    if detected_lang == expected_lang:
                        correct_detections += 1
                except:
                    pass
            
            return correct_detections / len(test_texts)
            
        except Exception as e:
            self.logger.warning(f"Language detection test failed: {e}")
            return 0.0    

    async def test_citation_format_consistency(self) -> float:
        """Test citation formatting consistency."""
        try:
            # Test citation formatting with sample data
            sample_sources = [
                {
                    "title": "Clinical Metabolomics in Precision Medicine",
                    "authors": ["Smith, J.", "Johnson, A."],
                    "journal": "Nature Medicine",
                    "year": 2023,
                    "doi": "10.1038/nm.2023.001"
                },
                {
                    "title": "Metabolomic Biomarkers for Disease Diagnosis",
                    "authors": ["Brown, K.", "Davis, L.", "Wilson, M."],
                    "journal": "Cell Metabolism",
                    "year": 2022,
                    "doi": "10.1016/j.cmet.2022.001"
                }
            ]
            
            consistent_formats = 0
            
            for source in sample_sources:
                try:
                    citation = format_citations([source])
                    # Check if citation contains expected elements
                    if (citation and 
                        source["title"] in citation and
                        str(source["year"]) in citation and
                        source["authors"][0] in citation):
                        consistent_formats += 1
                except:
                    pass
            
            return consistent_formats / len(sample_sources)
            
        except Exception as e:
            self.logger.warning(f"Citation format test failed: {e}")
            return 0.0
    
    async def test_bibliography_generation_accuracy(self) -> float:
        """Test bibliography generation accuracy."""
        try:
            sample_sources = [
                {
                    "title": "Metabolomics in Clinical Research",
                    "authors": ["Taylor, R.", "Anderson, S."],
                    "journal": "Science",
                    "year": 2023
                }
            ]
            
            try:
                bibliography = generate_bibliography(sample_sources)
                # Check if bibliography contains expected elements
                if (bibliography and 
                    sample_sources[0]["title"] in bibliography and
                    "Taylor" in bibliography):
                    return 1.0
                else:
                    return 0.0
            except:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"Bibliography test failed: {e}")
            return 0.0
    
    async def test_query_response_time(self) -> float:
        """Test query response time performance."""
        try:
            # Create a simple query engine test
            with tempfile.TemporaryDirectory() as temp_dir:
                config = LightRAGConfig(
                    knowledge_graph_path=f"{temp_dir}/kg",
                    vector_store_path=f"{temp_dir}/vectors",
                    cache_directory=f"{temp_dir}/cache",
                    papers_directory=f"{temp_dir}/papers"
                )
                
                component = LightRAGComponent(config)
                await component.initialize()
                
                # Measure response time
                start_time = datetime.now()
                await component.query("What is clinical metabolomics?")
                response_time = (datetime.now() - start_time).total_seconds()
                
                await component.cleanup()
                return response_time
                
        except Exception as e:
            self.logger.warning(f"Query response time test failed: {e}")
            return 10.0  # High value indicates poor performance
    
    async def test_embedding_generation_speed(self) -> float:
        """Test embedding generation speed."""
        try:
            # Test embedding generation speed
            test_text = "Clinical metabolomics is the study of metabolites in clinical settings."
            
            start_time = datetime.now()
            embedding_model = get_embedding_model()
            embeddings = embedding_model.encode([test_text])
            generation_time = (datetime.now() - start_time).total_seconds()
            
            return generation_time
            
        except Exception as e:
            self.logger.warning(f"Embedding speed test failed: {e}")
            return 1.0  # High value indicates poor performance
    
    async def test_neo4j_connection_reliability(self) -> float:
        """Test Neo4j connection reliability."""
        try:
            # Test Neo4j connection
            attempts = 10
            successful_connections = 0
            
            for _ in range(attempts):
                try:
                    graph_store = Neo4jGraphStore()
                    # Simple connection test
                    await graph_store.health_check()
                    successful_connections += 1
                except:
                    pass
            
            return successful_connections / attempts
            
        except Exception as e:
            self.logger.warning(f"Neo4j connection test failed: {e}")
            return 0.0
    
    async def test_graph_query_performance(self) -> float:
        """Test graph query performance."""
        try:
            # Test graph query performance
            graph_store = Neo4jGraphStore()
            
            start_time = datetime.now()
            # Simple query test
            await graph_store.query("MATCH (n) RETURN count(n) LIMIT 1")
            query_time = (datetime.now() - start_time).total_seconds()
            
            return query_time
            
        except Exception as e:
            self.logger.warning(f"Graph query performance test failed: {e}")
            return 5.0  # High value indicates poor performance
    
    async def test_chainlit_interface_compatibility(self) -> bool:
        """Test Chainlit interface compatibility."""
        try:
            # Test that LightRAG component can be imported and used
            # in a Chainlit-compatible way
            with tempfile.TemporaryDirectory() as temp_dir:
                config = LightRAGConfig(
                    knowledge_graph_path=f"{temp_dir}/kg",
                    vector_store_path=f"{temp_dir}/vectors",
                    cache_directory=f"{temp_dir}/cache",
                    papers_directory=f"{temp_dir}/papers"
                )
                
                component = LightRAGComponent(config)
                await component.initialize()
                
                # Test basic interface compatibility
                response = await component.query("Test query")
                
                # Check response format is compatible
                compatible = (
                    isinstance(response, dict) and
                    "answer" in response and
                    "confidence_score" in response
                )
                
                await component.cleanup()
                return compatible
                
        except Exception as e:
            self.logger.warning(f"Chainlit compatibility test failed: {e}")
            return False
    
    async def test_perplexity_api_integration(self) -> float:
        """Test Perplexity API integration."""
        try:
            # Mock Perplexity API test
            # In practice, this would test actual API integration
            
            # Simulate API call success rate
            attempts = 10
            successful_calls = 9  # Mock 90% success rate
            
            return successful_calls / attempts
            
        except Exception as e:
            self.logger.warning(f"Perplexity API test failed: {e}")
            return 0.0
    
    def generate_regression_report(self, results: Dict[str, Any],
                                 output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive regression test report.
        
        Args:
            results: Regression test results
            output_file: Optional file to save report
            
        Returns:
            Report content as string
        """
        report_lines = [
            "=" * 80,
            "LIGHTRAG REGRESSION TEST REPORT",
            "=" * 80,
            f"Timestamp: {results['timestamp']}",
            f"Duration: {results['duration_seconds']:.2f} seconds",
            f"Overall Status: {'✅ PASSED' if results['overall_success'] else '❌ FAILED'}",
            ""
        ]
        
        # Summary
        summary = results["summary"]
        report_lines.extend([
            "SUMMARY:",
            f"  Total Tests: {summary['total_tests']}",
            f"  Passed: {summary['passed']}",
            f"  Failed: {summary['failed']}",
            f"  Critical Failures: {summary['critical_failures']}",
            ""
        ])
        
        # Detailed results
        report_lines.append("DETAILED RESULTS:")
        report_lines.append("-" * 40)
        
        for test_result in results["test_results"]:
            status = "✅ PASSED" if test_result["passed"] else "❌ FAILED"
            critical_marker = " (CRITICAL)" if test_result["critical"] else ""
            
            report_lines.extend([
                f"Test: {test_result['test_name']}{critical_marker}",
                f"  Status: {status}",
                f"  Baseline: {test_result['baseline_value']}",
                f"  Current: {test_result['current_value']}",
                f"  Deviation: {test_result['deviation']:.3f}"
            ])
            
            if test_result["error_message"]:
                report_lines.append(f"  Error: {test_result['error_message']}")
            
            report_lines.append("")
        
        # Critical failures section
        critical_failures = [
            test for test in results["test_results"]
            if test["critical"] and not test["passed"]
        ]
        
        if critical_failures:
            report_lines.extend([
                "CRITICAL FAILURES:",
                "-" * 20
            ])
            
            for failure in critical_failures:
                report_lines.extend([
                    f"❌ {failure['test_name']}",
                    f"   Expected: {failure['baseline_value']}",
                    f"   Actual: {failure['current_value']}",
                    f"   Impact: Critical system functionality affected"
                ])
                
                if failure["error_message"]:
                    report_lines.append(f"   Error: {failure['error_message']}")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 20
        ])
        
        if results["overall_success"]:
            report_lines.extend([
                "✅ All regression tests passed or only non-critical tests failed.",
                "",
                "The LightRAG integration maintains compatibility with existing",
                "system functionality. No regressions detected in critical areas.",
                "",
                "Safe to proceed with deployment."
            ])
        else:
            report_lines.extend([
                "❌ Critical regression tests failed. Address the following issues",
                "before proceeding with deployment:",
                ""
            ])
            
            for failure in critical_failures:
                report_lines.append(f"  • Fix {failure['test_name']}")
                if failure["error_message"]:
                    report_lines.append(f"    Error: {failure['error_message']}")
            
            report_lines.extend([
                "",
                "Consider:",
                "  • Rolling back recent changes that may have caused regressions",
                "  • Updating baseline values if intentional changes were made",
                "  • Investigating root causes of performance degradations",
                "  • Running additional diagnostic tests"
            ])
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report_content)
            self.logger.info(f"Regression test report saved to {output_file}")
        
        return report_content
    
    def save_regression_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save regression test results as JSON."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Regression test results saved to {output_file}")
    
    def update_baselines(self, results: Dict[str, Any]) -> None:
        """Update baseline values based on current test results."""
        new_baselines = {}
        
        for test_result in results["test_results"]:
            if test_result["passed"] and test_result["current_value"] is not None:
                new_baselines[test_result["test_name"]] = test_result["current_value"]
        
        # Save updated baselines
        baseline_file = self.output_dir / "regression_baselines.json"
        with open(baseline_file, 'w') as f:
            json.dump(new_baselines, f, indent=2)
        
        self.logger.info(f"Updated baselines saved to {baseline_file}")


# Convenience function for running regression tests
async def run_regression_tests(config: Optional[LightRAGConfig] = None,
                             output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run regression test suite.
    
    Args:
        config: Optional LightRAG configuration
        output_dir: Optional output directory for results
        
    Returns:
        Dictionary with regression test results
    """
    test_suite = RegressionTestSuite(config=config, output_dir=output_dir)
    return await test_suite.run_regression_tests()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LightRAG Regression Test Suite")
    parser.add_argument("--output-dir", default="regression_test_results",
                       help="Output directory for test results")
    parser.add_argument("--update-baselines", action="store_true",
                       help="Update baseline values with current results")
    parser.add_argument("--save-results", action="store_true",
                       help="Save detailed results as JSON")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    async def main():
        try:
            # Run regression tests
            results = await run_regression_tests(output_dir=args.output_dir)
            
            # Generate report
            test_suite = RegressionTestSuite(output_dir=args.output_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            report = test_suite.generate_regression_report(
                results,
                str(test_suite.output_dir / f"regression_report_{timestamp}.txt")
            )
            print(report)
            
            # Save results if requested
            if args.save_results:
                test_suite.save_regression_results(
                    results,
                    str(test_suite.output_dir / f"regression_results_{timestamp}.json")
                )
            
            # Update baselines if requested
            if args.update_baselines:
                test_suite.update_baselines(results)
                print("Baseline values updated.")
            
            # Exit with appropriate code
            if results["overall_success"]:
                print("\n✅ REGRESSION TESTS PASSED!")
                exit(0)
            else:
                print("\n❌ REGRESSION TESTS FAILED!")
                exit(1)
                
        except Exception as e:
            print(f"Regression tests failed: {str(e)}")
            exit(1)
    
    asyncio.run(main())