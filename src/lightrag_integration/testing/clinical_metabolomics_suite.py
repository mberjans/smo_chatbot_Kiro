"""
Clinical Metabolomics Test Suite

This module provides comprehensive testing for clinical metabolomics knowledge
validation, including test datasets, accuracy measurement, and automated testing
pipelines for MVP validation.
"""

import asyncio
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import statistics

from ..component import LightRAGComponent
from ..config.settings import LightRAGConfig


@dataclass
class TestQuestion:
    """Represents a test question with expected answer criteria."""
    question: str
    expected_keywords: List[str]
    expected_concepts: List[str]
    minimum_confidence: float
    category: str
    difficulty: str  # "basic", "intermediate", "advanced"
    description: str


@dataclass
class TestResult:
    """Represents the result of a single test question."""
    question: str
    answer: str
    confidence_score: float
    processing_time: float
    keyword_matches: List[str]
    concept_matches: List[str]
    accuracy_score: float
    passed: bool
    error: Optional[str] = None


@dataclass
class TestSuiteResult:
    """Represents the results of an entire test suite run."""
    timestamp: datetime
    total_questions: int
    passed_questions: int
    failed_questions: int
    average_accuracy: float
    average_confidence: float
    average_processing_time: float
    results: List[TestResult]
    summary: Dict[str, Any]


class ClinicalMetabolomicsTestSuite:
    """
    Comprehensive test suite for clinical metabolomics knowledge validation.
    
    This class provides methods to create test datasets, run validation tests,
    measure accuracy, and generate reports for MVP validation.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None):
        """Initialize the test suite with configuration."""
        self.config = config or LightRAGConfig.from_env()
        self.logger = logging.getLogger(__name__)
        self.test_questions = self._create_test_dataset()
        
    def _create_test_dataset(self) -> List[TestQuestion]:
        """
        Create a comprehensive test dataset for clinical metabolomics.
        
        Returns:
            List of test questions covering various aspects of clinical metabolomics.
        """
        return [
            # Basic definition questions
            TestQuestion(
                question="What is clinical metabolomics?",
                expected_keywords=[
                    "metabolomics", "metabolites", "clinical", "biomarkers", 
                    "disease", "diagnosis", "small molecules", "metabolism"
                ],
                expected_concepts=[
                    "metabolite profiling", "biomarker discovery", "disease diagnosis",
                    "personalized medicine", "metabolic pathways"
                ],
                minimum_confidence=0.7,
                category="definition",
                difficulty="basic",
                description="Core definition of clinical metabolomics"
            ),
            
            TestQuestion(
                question="What are metabolites in the context of clinical research?",
                expected_keywords=[
                    "metabolites", "small molecules", "endogenous", "exogenous",
                    "biochemical", "pathways", "intermediates", "products"
                ],
                expected_concepts=[
                    "metabolic intermediates", "biochemical pathways", "cellular metabolism",
                    "bioactive compounds"
                ],
                minimum_confidence=0.6,
                category="definition",
                difficulty="basic",
                description="Definition and role of metabolites"
            ),
            
            # Application questions
            TestQuestion(
                question="How is metabolomics used in disease diagnosis?",
                expected_keywords=[
                    "diagnosis", "biomarkers", "disease", "metabolic", "signature",
                    "profiling", "pattern", "classification"
                ],
                expected_concepts=[
                    "biomarker discovery", "disease classification", "metabolic signatures",
                    "diagnostic accuracy", "clinical validation"
                ],
                minimum_confidence=0.6,
                category="application",
                difficulty="intermediate",
                description="Application of metabolomics in disease diagnosis"
            ),
            
            TestQuestion(
                question="What role does metabolomics play in personalized medicine?",
                expected_keywords=[
                    "personalized", "precision", "medicine", "individual", "treatment",
                    "response", "therapy", "patient"
                ],
                expected_concepts=[
                    "precision medicine", "treatment response", "drug metabolism",
                    "patient stratification", "therapeutic monitoring"
                ],
                minimum_confidence=0.6,
                category="application",
                difficulty="intermediate",
                description="Role in personalized/precision medicine"
            ),
            
            # Technical questions
            TestQuestion(
                question="What analytical techniques are used in clinical metabolomics?",
                expected_keywords=[
                    "mass spectrometry", "NMR", "chromatography", "LC-MS", "GC-MS",
                    "analytical", "techniques", "methods"
                ],
                expected_concepts=[
                    "analytical methods", "mass spectrometry", "nuclear magnetic resonance",
                    "chromatographic separation", "metabolite identification"
                ],
                minimum_confidence=0.6,
                category="technical",
                difficulty="intermediate",
                description="Analytical techniques and methods"
            ),
            
            TestQuestion(
                question="What are the main challenges in clinical metabolomics?",
                expected_keywords=[
                    "challenges", "standardization", "reproducibility", "validation",
                    "complexity", "interpretation", "variability"
                ],
                expected_concepts=[
                    "analytical reproducibility", "data standardization", "clinical validation",
                    "biological variability", "technical challenges"
                ],
                minimum_confidence=0.6,
                category="challenges",
                difficulty="advanced",
                description="Main challenges and limitations"
            ),
            
            # Sample and workflow questions
            TestQuestion(
                question="What types of biological samples are used in clinical metabolomics?",
                expected_keywords=[
                    "samples", "blood", "urine", "plasma", "serum", "tissue",
                    "saliva", "biological", "specimens"
                ],
                expected_concepts=[
                    "biological samples", "sample types", "specimen collection",
                    "sample preparation", "biofluid analysis"
                ],
                minimum_confidence=0.6,
                category="methodology",
                difficulty="basic",
                description="Types of biological samples used"
            ),
            
            TestQuestion(
                question="What is the typical workflow in a clinical metabolomics study?",
                expected_keywords=[
                    "workflow", "study", "design", "collection", "analysis",
                    "processing", "interpretation", "validation"
                ],
                expected_concepts=[
                    "study design", "sample collection", "data processing",
                    "statistical analysis", "biomarker validation"
                ],
                minimum_confidence=0.6,
                category="methodology",
                difficulty="intermediate",
                description="Typical study workflow and processes"
            ),
            
            # Advanced applications
            TestQuestion(
                question="How does metabolomics contribute to drug development?",
                expected_keywords=[
                    "drug", "development", "pharmaceutical", "toxicity", "efficacy",
                    "mechanism", "safety", "screening"
                ],
                expected_concepts=[
                    "drug development", "pharmacometabolomics", "toxicity assessment",
                    "mechanism of action", "drug safety"
                ],
                minimum_confidence=0.6,
                category="application",
                difficulty="advanced",
                description="Role in pharmaceutical drug development"
            ),
            
            TestQuestion(
                question="What is the difference between targeted and untargeted metabolomics?",
                expected_keywords=[
                    "targeted", "untargeted", "approach", "specific", "comprehensive",
                    "hypothesis", "discovery", "quantitative"
                ],
                expected_concepts=[
                    "targeted analysis", "untargeted profiling", "hypothesis-driven",
                    "discovery-based", "quantitative analysis"
                ],
                minimum_confidence=0.6,
                category="methodology",
                difficulty="intermediate",
                description="Comparison of targeted vs untargeted approaches"
            )
        ]
    
    def create_test_papers_dataset(self, output_dir: str) -> List[str]:
        """
        Create mock clinical metabolomics papers for testing.
        
        Args:
            output_dir: Directory to create test papers in
            
        Returns:
            List of created paper file paths
        """
        papers_dir = Path(output_dir)
        papers_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock PDF content (as text files for testing)
        test_papers = [
            {
                "filename": "clinical_metabolomics_overview.txt",
                "content": """
                Clinical Metabolomics: A Comprehensive Overview
                
                Clinical metabolomics is the comprehensive study of small molecules (metabolites) 
                in biological systems within the context of clinical research and healthcare. 
                This field focuses on the identification and quantification of metabolites in 
                biological samples such as blood, urine, tissue, and other biofluids to understand 
                disease mechanisms, discover biomarkers, and support personalized medicine approaches.
                
                Metabolites are small molecules (typically <1500 Da) that are intermediates and 
                products of cellular metabolism. They include endogenous compounds produced by 
                the organism and exogenous compounds from environmental sources, diet, or medications. 
                The metabolome represents the complete set of metabolites present in a biological 
                system at a given time.
                
                Clinical metabolomics applications include:
                1. Disease diagnosis and prognosis
                2. Biomarker discovery and validation
                3. Drug development and pharmacometabolomics
                4. Personalized medicine and treatment monitoring
                5. Understanding disease mechanisms and pathways
                
                The field employs various analytical techniques including mass spectrometry (MS), 
                nuclear magnetic resonance (NMR) spectroscopy, and chromatographic methods such as 
                liquid chromatography-mass spectrometry (LC-MS) and gas chromatography-mass 
                spectrometry (GC-MS).
                """
            },
            {
                "filename": "metabolomics_biomarkers_disease.txt",
                "content": """
                Metabolomics Biomarkers in Disease Diagnosis
                
                Metabolomics has emerged as a powerful approach for biomarker discovery in 
                clinical research. Metabolic biomarkers are metabolites whose levels change 
                in response to disease states, providing valuable information for diagnosis, 
                prognosis, and treatment monitoring.
                
                The process of biomarker discovery involves:
                1. Sample collection from patients and controls
                2. Metabolite profiling using analytical techniques
                3. Statistical analysis to identify discriminatory metabolites
                4. Validation in independent cohorts
                5. Clinical implementation and regulatory approval
                
                Metabolic signatures or patterns of multiple metabolites often provide 
                better diagnostic accuracy than single biomarkers. These signatures can 
                reflect complex biological processes and disease mechanisms.
                
                Applications in various diseases include:
                - Cancer: Altered metabolism in tumor cells
                - Diabetes: Glucose and lipid metabolism changes
                - Cardiovascular disease: Lipid and amino acid alterations
                - Neurological disorders: Neurotransmitter and energy metabolism
                - Kidney disease: Uremic toxins and metabolic waste products
                """
            },
            {
                "filename": "analytical_methods_metabolomics.txt",
                "content": """
                Analytical Methods in Clinical Metabolomics
                
                Clinical metabolomics relies on sophisticated analytical techniques to 
                identify and quantify metabolites in biological samples. The choice of 
                analytical method depends on the research question, sample type, and 
                target metabolites.
                
                Mass Spectrometry (MS):
                - High sensitivity and specificity
                - Coupled with chromatographic separation (LC-MS, GC-MS)
                - Enables identification and quantification of diverse metabolites
                - Suitable for both targeted and untargeted approaches
                
                Nuclear Magnetic Resonance (NMR) Spectroscopy:
                - Non-destructive and highly reproducible
                - Provides structural information
                - Quantitative without need for standards
                - Limited sensitivity compared to MS
                
                Sample Preparation:
                - Critical for reproducible results
                - Includes protein precipitation, extraction, derivatization
                - Must preserve metabolite integrity
                - Standardized protocols essential for clinical applications
                
                Data Processing and Analysis:
                - Peak detection and alignment
                - Metabolite identification using databases
                - Statistical analysis and pattern recognition
                - Quality control and validation procedures
                """
            },
            {
                "filename": "personalized_medicine_metabolomics.txt",
                "content": """
                Metabolomics in Personalized Medicine
                
                Personalized medicine aims to tailor medical treatment to individual 
                characteristics, and metabolomics plays a crucial role in this approach. 
                The metabolome reflects the interaction between genetic, environmental, 
                and lifestyle factors, making it an ideal tool for personalized healthcare.
                
                Applications in Personalized Medicine:
                
                1. Treatment Response Prediction:
                   - Metabolic profiles can predict drug response
                   - Identification of responders vs. non-responders
                   - Optimization of drug dosing
                
                2. Patient Stratification:
                   - Classification of patients into subgroups
                   - Disease subtypes with different metabolic profiles
                   - Targeted therapy selection
                
                3. Therapeutic Monitoring:
                   - Real-time assessment of treatment effects
                   - Early detection of adverse reactions
                   - Adjustment of treatment protocols
                
                4. Pharmacometabolomics:
                   - Study of drug metabolism and effects
                   - Individual variations in drug processing
                   - Prediction of drug toxicity and efficacy
                
                Challenges include standardization of methods, validation in diverse 
                populations, integration with other omics data, and translation to 
                clinical practice.
                """
            },
            {
                "filename": "challenges_clinical_metabolomics.txt",
                "content": """
                Challenges and Future Directions in Clinical Metabolomics
                
                Despite significant advances, clinical metabolomics faces several 
                challenges that must be addressed for successful translation to 
                clinical practice.
                
                Technical Challenges:
                1. Analytical Reproducibility:
                   - Instrument variability and drift
                   - Need for standardized protocols
                   - Quality control measures
                
                2. Metabolite Identification:
                   - Limited reference standards
                   - Incomplete metabolite databases
                   - Structural elucidation difficulties
                
                3. Data Processing and Integration:
                   - Complex data analysis pipelines
                   - Integration with clinical data
                   - Standardized data formats
                
                Biological Challenges:
                1. Biological Variability:
                   - Inter-individual differences
                   - Temporal variations
                   - Environmental influences
                
                2. Sample-related Issues:
                   - Sample collection and storage
                   - Matrix effects
                   - Contamination risks
                
                Clinical Translation Challenges:
                1. Validation Requirements:
                   - Large-scale validation studies
                   - Regulatory approval processes
                   - Clinical utility demonstration
                
                2. Implementation:
                   - Cost-effectiveness considerations
                   - Integration into clinical workflows
                   - Training and education needs
                
                Future directions include improved analytical methods, better 
                bioinformatics tools, standardization initiatives, and integration 
                with other omics technologies.
                """
            }
        ]
        
        created_files = []
        for paper in test_papers:
            file_path = papers_dir / paper["filename"]
            file_path.write_text(paper["content"])
            created_files.append(str(file_path))
            
        self.logger.info(f"Created {len(created_files)} test papers in {output_dir}")
        return created_files
    
    async def run_single_test(self, component: LightRAGComponent, 
                            test_question: TestQuestion) -> TestResult:
        """
        Run a single test question and evaluate the result.
        
        Args:
            component: Initialized LightRAG component
            test_question: The test question to evaluate
            
        Returns:
            TestResult with evaluation metrics
        """
        start_time = datetime.now()
        
        try:
            # Query the component
            response = await component.query(test_question.question)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract response data
            answer = response.get("answer", "")
            confidence_score = response.get("confidence_score", 0.0)
            
            # Evaluate keyword matches
            answer_lower = answer.lower()
            keyword_matches = [
                keyword for keyword in test_question.expected_keywords
                if keyword.lower() in answer_lower
            ]
            
            # Evaluate concept matches (more flexible matching)
            concept_matches = []
            for concept in test_question.expected_concepts:
                concept_words = concept.lower().split()
                if any(word in answer_lower for word in concept_words):
                    concept_matches.append(concept)
            
            # Calculate accuracy score
            keyword_score = len(keyword_matches) / len(test_question.expected_keywords)
            concept_score = len(concept_matches) / len(test_question.expected_concepts)
            confidence_bonus = min(confidence_score, 1.0) * 0.2  # Up to 20% bonus for confidence
            
            accuracy_score = (keyword_score * 0.4 + concept_score * 0.4 + confidence_bonus)
            accuracy_score = min(accuracy_score, 1.0)  # Cap at 100%
            
            # Determine if test passed
            passed = (
                accuracy_score >= 0.6 and  # At least 60% accuracy
                confidence_score >= test_question.minimum_confidence and
                len(answer.strip()) > 50  # Minimum answer length
            )
            
            return TestResult(
                question=test_question.question,
                answer=answer,
                confidence_score=confidence_score,
                processing_time=processing_time,
                keyword_matches=keyword_matches,
                concept_matches=concept_matches,
                accuracy_score=accuracy_score,
                passed=passed
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Test failed for question '{test_question.question}': {str(e)}")
            
            return TestResult(
                question=test_question.question,
                answer="",
                confidence_score=0.0,
                processing_time=processing_time,
                keyword_matches=[],
                concept_matches=[],
                accuracy_score=0.0,
                passed=False,
                error=str(e)
            )
    
    async def run_test_suite(self, component: LightRAGComponent, 
                           questions: Optional[List[TestQuestion]] = None) -> TestSuiteResult:
        """
        Run the complete test suite and generate results.
        
        Args:
            component: Initialized LightRAG component
            questions: Optional list of questions to test (defaults to all)
            
        Returns:
            TestSuiteResult with comprehensive metrics
        """
        questions = questions or self.test_questions
        results = []
        
        self.logger.info(f"Running test suite with {len(questions)} questions")
        
        # Run all tests
        for i, question in enumerate(questions, 1):
            self.logger.info(f"Running test {i}/{len(questions)}: {question.question[:50]}...")
            result = await self.run_single_test(component, question)
            results.append(result)
            
            # Log result
            status = "PASSED" if result.passed else "FAILED"
            self.logger.info(
                f"Test {i} {status}: accuracy={result.accuracy_score:.2f}, "
                f"confidence={result.confidence_score:.2f}, time={result.processing_time:.2f}s"
            )
        
        # Calculate summary statistics
        passed_results = [r for r in results if r.passed]
        failed_results = [r for r in results if not r.passed]
        
        accuracy_scores = [r.accuracy_score for r in results]
        confidence_scores = [r.confidence_score for r in results if r.confidence_score > 0]
        processing_times = [r.processing_time for r in results]
        
        # Create summary
        summary = {
            "pass_rate": len(passed_results) / len(results) if results else 0,
            "accuracy_distribution": {
                "min": min(accuracy_scores) if accuracy_scores else 0,
                "max": max(accuracy_scores) if accuracy_scores else 0,
                "median": statistics.median(accuracy_scores) if accuracy_scores else 0,
                "std_dev": statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0
            },
            "confidence_distribution": {
                "min": min(confidence_scores) if confidence_scores else 0,
                "max": max(confidence_scores) if confidence_scores else 0,
                "median": statistics.median(confidence_scores) if confidence_scores else 0,
                "std_dev": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0
            },
            "performance": {
                "min_time": min(processing_times) if processing_times else 0,
                "max_time": max(processing_times) if processing_times else 0,
                "median_time": statistics.median(processing_times) if processing_times else 0
            },
            "category_breakdown": self._analyze_by_category(results, questions),
            "difficulty_breakdown": self._analyze_by_difficulty(results, questions)
        }
        
        return TestSuiteResult(
            timestamp=datetime.now(),
            total_questions=len(results),
            passed_questions=len(passed_results),
            failed_questions=len(failed_results),
            average_accuracy=statistics.mean(accuracy_scores) if accuracy_scores else 0,
            average_confidence=statistics.mean(confidence_scores) if confidence_scores else 0,
            average_processing_time=statistics.mean(processing_times) if processing_times else 0,
            results=results,
            summary=summary
        )
    
    def _analyze_by_category(self, results: List[TestResult], 
                           questions: List[TestQuestion]) -> Dict[str, Dict[str, float]]:
        """Analyze results by question category."""
        categories = {}
        
        for result, question in zip(results, questions):
            if question.category not in categories:
                categories[question.category] = {"passed": 0, "total": 0, "accuracy": []}
            
            categories[question.category]["total"] += 1
            categories[question.category]["accuracy"].append(result.accuracy_score)
            
            if result.passed:
                categories[question.category]["passed"] += 1
        
        # Calculate summary statistics for each category
        for category in categories:
            data = categories[category]
            data["pass_rate"] = data["passed"] / data["total"]
            data["average_accuracy"] = statistics.mean(data["accuracy"])
            del data["accuracy"]  # Remove raw data
        
        return categories
    
    def _analyze_by_difficulty(self, results: List[TestResult], 
                             questions: List[TestQuestion]) -> Dict[str, Dict[str, float]]:
        """Analyze results by question difficulty."""
        difficulties = {}
        
        for result, question in zip(results, questions):
            if question.difficulty not in difficulties:
                difficulties[question.difficulty] = {"passed": 0, "total": 0, "accuracy": []}
            
            difficulties[question.difficulty]["total"] += 1
            difficulties[question.difficulty]["accuracy"].append(result.accuracy_score)
            
            if result.passed:
                difficulties[question.difficulty]["passed"] += 1
        
        # Calculate summary statistics for each difficulty
        for difficulty in difficulties:
            data = difficulties[difficulty]
            data["pass_rate"] = data["passed"] / data["total"]
            data["average_accuracy"] = statistics.mean(data["accuracy"])
            del data["accuracy"]  # Remove raw data
        
        return difficulties
    
    def generate_report(self, result: TestSuiteResult, output_file: Optional[str] = None) -> str:
        """
        Generate a comprehensive test report.
        
        Args:
            result: Test suite results
            output_file: Optional file path to save report
            
        Returns:
            Report content as string
        """
        report_lines = [
            "=" * 80,
            "CLINICAL METABOLOMICS TEST SUITE REPORT",
            "=" * 80,
            f"Timestamp: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Questions: {result.total_questions}",
            f"Passed: {result.passed_questions} ({result.passed_questions/result.total_questions*100:.1f}%)",
            f"Failed: {result.failed_questions} ({result.failed_questions/result.total_questions*100:.1f}%)",
            "",
            "OVERALL METRICS:",
            f"  Average Accuracy: {result.average_accuracy:.3f}",
            f"  Average Confidence: {result.average_confidence:.3f}",
            f"  Average Processing Time: {result.average_processing_time:.3f}s",
            "",
            "ACCURACY DISTRIBUTION:",
            f"  Min: {result.summary['accuracy_distribution']['min']:.3f}",
            f"  Max: {result.summary['accuracy_distribution']['max']:.3f}",
            f"  Median: {result.summary['accuracy_distribution']['median']:.3f}",
            f"  Std Dev: {result.summary['accuracy_distribution']['std_dev']:.3f}",
            "",
            "PERFORMANCE METRICS:",
            f"  Min Time: {result.summary['performance']['min_time']:.3f}s",
            f"  Max Time: {result.summary['performance']['max_time']:.3f}s",
            f"  Median Time: {result.summary['performance']['median_time']:.3f}s",
            "",
            "CATEGORY BREAKDOWN:",
        ]
        
        for category, data in result.summary['category_breakdown'].items():
            report_lines.extend([
                f"  {category.upper()}:",
                f"    Pass Rate: {data['pass_rate']:.1%}",
                f"    Average Accuracy: {data['average_accuracy']:.3f}",
                f"    Total Questions: {data['total']}",
                ""
            ])
        
        report_lines.extend([
            "DIFFICULTY BREAKDOWN:",
        ])
        
        for difficulty, data in result.summary['difficulty_breakdown'].items():
            report_lines.extend([
                f"  {difficulty.upper()}:",
                f"    Pass Rate: {data['pass_rate']:.1%}",
                f"    Average Accuracy: {data['average_accuracy']:.3f}",
                f"    Total Questions: {data['total']}",
                ""
            ])
        
        report_lines.extend([
            "DETAILED RESULTS:",
            "-" * 40,
        ])
        
        for i, test_result in enumerate(result.results, 1):
            status = "PASS" if test_result.passed else "FAIL"
            report_lines.extend([
                f"{i}. {status} - {test_result.question}",
                f"   Accuracy: {test_result.accuracy_score:.3f}",
                f"   Confidence: {test_result.confidence_score:.3f}",
                f"   Time: {test_result.processing_time:.3f}s",
                f"   Keywords Found: {len(test_result.keyword_matches)}",
                f"   Concepts Found: {len(test_result.concept_matches)}",
            ])
            
            if test_result.error:
                report_lines.append(f"   Error: {test_result.error}")
            
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report_content)
            self.logger.info(f"Report saved to {output_file}")
        
        return report_content
    
    def save_results_json(self, result: TestSuiteResult, output_file: str) -> None:
        """Save test results as JSON for further analysis."""
        # Convert dataclass to dict for JSON serialization
        result_dict = asdict(result)
        
        # Convert datetime to string
        result_dict['timestamp'] = result.timestamp.isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Results saved to {output_file}")


async def run_mvp_validation_test(config: Optional[LightRAGConfig] = None) -> TestSuiteResult:
    """
    Run the MVP validation test for clinical metabolomics.
    
    This is the main entry point for automated testing pipeline.
    
    Args:
        config: Optional LightRAG configuration
        
    Returns:
        TestSuiteResult with validation results
    """
    # Create test suite
    test_suite = ClinicalMetabolomicsTestSuite(config)
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up test configuration
        test_config = config or LightRAGConfig.from_env()
        test_config.papers_directory = str(Path(temp_dir) / "papers")
        test_config.knowledge_graph_path = str(Path(temp_dir) / "kg")
        test_config.vector_store_path = str(Path(temp_dir) / "vectors")
        test_config.cache_directory = str(Path(temp_dir) / "cache")
        
        # Create test papers
        test_suite.create_test_papers_dataset(test_config.papers_directory)
        
        # Initialize component
        component = LightRAGComponent(test_config)
        
        try:
            await component.initialize()
            
            # Ingest test papers
            papers = list(Path(test_config.papers_directory).glob("*.txt"))
            if papers:
                await component.ingest_documents([str(p) for p in papers])
            
            # Run test suite
            results = await test_suite.run_test_suite(component)
            
            return results
            
        finally:
            await component.cleanup()


if __name__ == "__main__":
    # Run MVP validation test
    async def main():
        logging.basicConfig(level=logging.INFO)
        results = await run_mvp_validation_test()
        
        # Generate and print report
        test_suite = ClinicalMetabolomicsTestSuite()
        report = test_suite.generate_report(results)
        print(report)
        
        # Save results
        test_suite.save_results_json(results, "mvp_validation_results.json")
    
    asyncio.run(main())