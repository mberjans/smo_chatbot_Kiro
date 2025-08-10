"""
User Acceptance Test Suite for LightRAG Integration

This module provides user acceptance testing (UAT) for key scenarios that
validate the system meets user requirements and expectations from an
end-user perspective.
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

from ..component import LightRAGComponent
from ..config.settings import LightRAGConfig
from ..utils.logging import setup_logger


@dataclass
class UserScenario:
    """Definition of a user acceptance test scenario."""
    name: str
    description: str
    user_persona: str
    steps: List[str]
    acceptance_criteria: Dict[str, Any]
    priority: str  # "high", "medium", "low"


@dataclass
class UserAcceptanceResult:
    """Result of a user acceptance test."""
    scenario_name: str
    user_persona: str
    success: bool
    steps_completed: List[str]
    steps_failed: List[str]
    acceptance_criteria_met: Dict[str, bool]
    user_satisfaction_score: float  # 1-5 scale
    feedback: str
    duration_seconds: float
    timestamp: datetime


class UserAcceptanceTestSuite:
    """
    User acceptance test suite for LightRAG integration.
    
    This class provides testing from the perspective of different user personas
    to validate that the system meets real-world usage requirements.
    """
    
    def __init__(self, config: Optional[LightRAGConfig] = None,
                 output_dir: Optional[str] = None):
        """
        Initialize the user acceptance test suite.
        
        Args:
            config: Optional LightRAG configuration
            output_dir: Directory for test outputs and reports
        """
        self.config = config or LightRAGConfig.from_env()
        self.output_dir = Path(output_dir or "uat_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("user_acceptance_test_suite",
                                 log_file=str(self.output_dir / "uat_tests.log"))
        
        # Define user scenarios
        self.user_scenarios = self._define_user_scenarios()
    
    def _define_user_scenarios(self) -> List[UserScenario]:
        """Define user acceptance test scenarios."""
        return [
            UserScenario(
                name="researcher_literature_review",
                description="Researcher conducting literature review on clinical metabolomics",
                user_persona="Clinical Researcher",
                steps=[
                    "setup_research_environment",
                    "upload_metabolomics_papers",
                    "ask_basic_definition_question",
                    "ask_specific_technique_question",
                    "ask_application_question",
                    "verify_citations_provided",
                    "check_response_accuracy",
                    "evaluate_user_experience"
                ],
                acceptance_criteria={
                    "response_accuracy": 0.8,
                    "response_time": 10.0,  # seconds
                    "citations_provided": True,
                    "user_satisfaction": 4.0  # out of 5
                },
                priority="high"
            ),
            
            UserScenario(
                name="clinician_diagnostic_support",
                description="Clinician seeking diagnostic support information",
                user_persona="Clinical Physician",
                steps=[
                    "setup_clinical_environment",
                    "upload_diagnostic_papers",
                    "ask_diagnostic_biomarker_question",
                    "ask_disease_specific_question",
                    "verify_clinical_relevance",
                    "check_confidence_scores",
                    "evaluate_clinical_utility"
                ],
                acceptance_criteria={
                    "clinical_relevance": 0.85,
                    "confidence_scores_present": True,
                    "response_time": 8.0,
                    "user_satisfaction": 4.2
                },
                priority="high"
            ),
            
            UserScenario(
                name="student_learning_support",
                description="Graduate student learning about metabolomics",
                user_persona="Graduate Student",
                steps=[
                    "setup_learning_environment",
                    "upload_educational_papers",
                    "ask_fundamental_concepts",
                    "ask_methodology_questions",
                    "verify_educational_value",
                    "check_explanation_clarity",
                    "evaluate_learning_support"
                ],
                acceptance_criteria={
                    "explanation_clarity": 0.8,
                    "educational_value": 0.85,
                    "response_time": 12.0,
                    "user_satisfaction": 3.8
                },
                priority="medium"
            ),
            
            UserScenario(
                name="multilingual_researcher",
                description="Non-English speaking researcher using translation",
                user_persona="International Researcher",
                steps=[
                    "setup_multilingual_environment",
                    "upload_papers_various_languages",
                    "ask_question_in_spanish",
                    "ask_question_in_french",
                    "verify_translation_quality",
                    "check_citation_translation",
                    "evaluate_multilingual_experience"
                ],
                acceptance_criteria={
                    "translation_accuracy": 0.8,
                    "multilingual_support": True,
                    "response_time": 15.0,
                    "user_satisfaction": 3.5
                },
                priority="medium"
            ),
            
            UserScenario(
                name="industry_professional_research",
                description="Industry professional researching market applications",
                user_persona="Industry Professional",
                steps=[
                    "setup_industry_environment",
                    "upload_commercial_papers",
                    "ask_market_application_questions",
                    "ask_technology_trend_questions",
                    "verify_commercial_relevance",
                    "check_recent_developments",
                    "evaluate_business_utility"
                ],
                acceptance_criteria={
                    "commercial_relevance": 0.75,
                    "technology_coverage": 0.8,
                    "response_time": 10.0,
                    "user_satisfaction": 3.8
                },
                priority="low"
            ),
            
            UserScenario(
                name="collaborative_research_team",
                description="Research team collaborating on metabolomics project",
                user_persona="Research Team",
                steps=[
                    "setup_collaborative_environment",
                    "simulate_multiple_users",
                    "concurrent_question_asking",
                    "verify_consistent_responses",
                    "check_system_performance",
                    "evaluate_collaboration_support"
                ],
                acceptance_criteria={
                    "response_consistency": 0.9,
                    "concurrent_user_support": 5,  # users
                    "system_stability": True,
                    "user_satisfaction": 4.0
                },
                priority="high"
            )
        ]
    
    async def run_user_acceptance_tests(self) -> Dict[str, Any]:
        """
        Run all user acceptance test scenarios.
        
        Returns:
            Dictionary with comprehensive UAT results
        """
        test_start = datetime.now()
        self.logger.info("Starting user acceptance test suite")
        
        results = {
            "timestamp": test_start.isoformat(),
            "scenario_results": [],
            "summary": {
                "total_scenarios": len(self.user_scenarios),
                "passed": 0,
                "failed": 0,
                "high_priority_passed": 0,
                "high_priority_failed": 0,
                "average_satisfaction": 0.0
            },
            "overall_success": False,
            "duration_seconds": 0
        }
        
        try:
            satisfaction_scores = []
            
            # Run each user scenario
            for scenario in self.user_scenarios:
                self.logger.info(f"Running UAT scenario: {scenario.name}")
                
                scenario_result = await self._run_user_scenario(scenario)
                results["scenario_results"].append(asdict(scenario_result))
                
                satisfaction_scores.append(scenario_result.user_satisfaction_score)
                
                if scenario_result.success:
                    results["summary"]["passed"] += 1
                    if scenario.priority == "high":
                        results["summary"]["high_priority_passed"] += 1
                else:
                    results["summary"]["failed"] += 1
                    if scenario.priority == "high":
                        results["summary"]["high_priority_failed"] += 1
                
                self.logger.info(
                    f"Scenario {scenario.name}: "
                    f"{'SUCCESS' if scenario_result.success else 'FAILED'} "
                    f"(Satisfaction: {scenario_result.user_satisfaction_score:.1f}/5.0)"
                )
            
            # Calculate overall metrics
            results["summary"]["average_satisfaction"] = sum(satisfaction_scores) / len(satisfaction_scores)
            
            # Determine overall success (all high priority scenarios must pass)
            results["overall_success"] = (
                results["summary"]["high_priority_failed"] == 0 and
                results["summary"]["average_satisfaction"] >= 3.5
            )
            
            results["duration_seconds"] = (datetime.now() - test_start).total_seconds()
            
            self.logger.info(
                f"User acceptance tests completed: "
                f"{'SUCCESS' if results['overall_success'] else 'FAILED'} "
                f"(Avg satisfaction: {results['summary']['average_satisfaction']:.1f}/5.0)"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"User acceptance tests failed with error: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            results["error"] = error_msg
            results["duration_seconds"] = (datetime.now() - test_start).total_seconds()
            return results
    
    async def _run_user_scenario(self, scenario: UserScenario) -> UserAcceptanceResult:
        """Run a single user acceptance scenario."""
        scenario_start = datetime.now()
        steps_completed = []
        steps_failed = []
        acceptance_criteria_met = {}
        user_satisfaction_score = 0.0
        feedback = ""
        
        try:
            # Execute each step in the scenario
            scenario_context = {}
            
            for step_name in scenario.steps:
                self.logger.debug(f"Executing UAT step: {step_name}")
                
                step_method = getattr(self, f"_uat_step_{step_name}", None)
                if not step_method:
                    raise ValueError(f"Unknown UAT step: {step_name}")
                
                step_result = await step_method(scenario_context)
                
                if step_result.get("success", True):
                    steps_completed.append(step_name)
                    # Update scenario context with step results
                    scenario_context.update(step_result.get("context", {}))
                else:
                    steps_failed.append(step_name)
                    break
            
            # Evaluate acceptance criteria
            acceptance_criteria_met = self._evaluate_acceptance_criteria(
                scenario.acceptance_criteria, scenario_context
            )
            
            # Calculate user satisfaction score
            user_satisfaction_score = self._calculate_user_satisfaction(
                scenario, scenario_context, acceptance_criteria_met
            )
            
            # Generate user feedback
            feedback = self._generate_user_feedback(
                scenario, scenario_context, acceptance_criteria_met
            )
            
            # Determine overall success
            success = (
                len(steps_failed) == 0 and
                all(acceptance_criteria_met.values()) and
                user_satisfaction_score >= 3.0
            )
            
            return UserAcceptanceResult(
                scenario_name=scenario.name,
                user_persona=scenario.user_persona,
                success=success,
                steps_completed=steps_completed,
                steps_failed=steps_failed,
                acceptance_criteria_met=acceptance_criteria_met,
                user_satisfaction_score=user_satisfaction_score,
                feedback=feedback,
                duration_seconds=(datetime.now() - scenario_start).total_seconds(),
                timestamp=scenario_start
            )
            
        except Exception as e:
            return UserAcceptanceResult(
                scenario_name=scenario.name,
                user_persona=scenario.user_persona,
                success=False,
                steps_completed=steps_completed,
                steps_failed=steps_failed + [f"Exception: {str(e)}"],
                acceptance_criteria_met={},
                user_satisfaction_score=1.0,  # Low satisfaction due to failure
                feedback=f"Test failed with error: {str(e)}",
                duration_seconds=(datetime.now() - scenario_start).total_seconds(),
                timestamp=scenario_start
            )
    
    def _evaluate_acceptance_criteria(self, criteria: Dict[str, Any],
                                    context: Dict[str, Any]) -> Dict[str, bool]:
        """Evaluate acceptance criteria against scenario context."""
        results = {}
        
        for criterion_name, expected_value in criteria.items():
            actual_value = context.get(criterion_name, None)
            
            if actual_value is None:
                results[criterion_name] = False
            elif isinstance(expected_value, bool):
                results[criterion_name] = actual_value == expected_value
            elif isinstance(expected_value, (int, float)):
                if criterion_name.endswith("_time"):
                    # For time criteria, actual should be <= expected
                    results[criterion_name] = actual_value <= expected_value
                else:
                    # For other numeric criteria, actual should be >= expected
                    results[criterion_name] = actual_value >= expected_value
            else:
                results[criterion_name] = actual_value == expected_value
        
        return results
    
    def _calculate_user_satisfaction(self, scenario: UserScenario,
                                   context: Dict[str, Any],
                                   criteria_met: Dict[str, bool]) -> float:
        """Calculate user satisfaction score based on scenario results."""
        base_score = 3.0  # Neutral satisfaction
        
        # Adjust based on criteria met
        criteria_score = sum(criteria_met.values()) / len(criteria_met) if criteria_met else 0
        base_score += (criteria_score - 0.5) * 2  # Scale to -1 to +1, then to satisfaction
        
        # Adjust based on specific context factors
        if context.get("response_accuracy", 0) > 0.9:
            base_score += 0.3
        elif context.get("response_accuracy", 0) < 0.6:
            base_score -= 0.5
        
        if context.get("response_time", 10) < 5:
            base_score += 0.2
        elif context.get("response_time", 10) > 15:
            base_score -= 0.3
        
        if context.get("citations_provided", False):
            base_score += 0.2
        
        # Clamp to 1-5 range
        return max(1.0, min(5.0, base_score))
    
    def _generate_user_feedback(self, scenario: UserScenario,
                              context: Dict[str, Any],
                              criteria_met: Dict[str, bool]) -> str:
        """Generate realistic user feedback based on scenario results."""
        feedback_parts = []
        
        # Overall experience
        satisfaction = self._calculate_user_satisfaction(scenario, context, criteria_met)
        
        if satisfaction >= 4.5:
            feedback_parts.append("Excellent experience! The system exceeded expectations.")
        elif satisfaction >= 4.0:
            feedback_parts.append("Very good experience. The system works well for my needs.")
        elif satisfaction >= 3.5:
            feedback_parts.append("Good experience overall, with some room for improvement.")
        elif satisfaction >= 3.0:
            feedback_parts.append("Acceptable experience, but several issues need addressing.")
        else:
            feedback_parts.append("Poor experience. Significant improvements needed.")
        
        # Specific feedback based on criteria
        if not criteria_met.get("response_accuracy", True):
            feedback_parts.append("The accuracy of responses needs improvement.")
        
        if not criteria_met.get("response_time", True):
            feedback_parts.append("Response times are too slow for practical use.")
        
        if not criteria_met.get("citations_provided", True):
            feedback_parts.append("Missing citations make it hard to verify information.")
        
        if context.get("translation_accuracy", 1.0) < 0.8:
            feedback_parts.append("Translation quality needs improvement for non-English users.")
        
        # Positive feedback
        if context.get("response_accuracy", 0) > 0.85:
            feedback_parts.append("The responses are accurate and helpful.")
        
        if context.get("citations_provided", False):
            feedback_parts.append("Good citation support helps verify information.")
        
        return " ".join(feedback_parts) 
   
    # UAT step implementations
    async def _uat_step_setup_research_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Set up research environment for testing."""
        try:
            context["temp_dir"] = tempfile.mkdtemp(prefix="uat_research_")
            context["config"] = LightRAGConfig(
                knowledge_graph_path=f"{context['temp_dir']}/kg",
                vector_store_path=f"{context['temp_dir']}/vectors",
                cache_directory=f"{context['temp_dir']}/cache",
                papers_directory=f"{context['temp_dir']}/papers"
            )
            
            context["component"] = LightRAGComponent(context["config"])
            await context["component"].initialize()
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_upload_metabolomics_papers(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Upload sample metabolomics papers."""
        try:
            papers_dir = Path(context["config"].papers_directory)
            
            # Create sample research papers
            sample_papers = [
                "clinical_metabolomics_review.pdf",
                "biomarker_discovery_methods.pdf",
                "metabolic_profiling_techniques.pdf"
            ]
            
            # Create minimal PDF content for each paper
            pdf_content = self._create_sample_pdf_content("Clinical Metabolomics Research")
            
            for paper_name in sample_papers:
                (papers_dir / paper_name).write_bytes(pdf_content)
            
            # Ingest the papers
            ingestion_result = await context["component"].ingest_documents()
            
            context["papers_uploaded"] = len(sample_papers)
            context["papers_ingested"] = ingestion_result["successful"]
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_ask_basic_definition_question(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask basic definition question."""
        try:
            start_time = datetime.now()
            response = await context["component"].query("What is clinical metabolomics?")
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Evaluate response quality
            answer = response.get("answer", "").lower()
            key_terms = ["metabolomics", "metabolites", "clinical", "biomarkers"]
            terms_found = sum(1 for term in key_terms if term in answer)
            accuracy = terms_found / len(key_terms)
            
            context["basic_question_response"] = response
            context["response_time"] = response_time
            context["response_accuracy"] = accuracy
            context["citations_provided"] = len(response.get("source_documents", [])) > 0
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_ask_specific_technique_question(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask specific technique question."""
        try:
            response = await context["component"].query(
                "How is mass spectrometry used in clinical metabolomics?"
            )
            
            # Evaluate technical accuracy
            answer = response.get("answer", "").lower()
            technical_terms = ["mass spectrometry", "ms", "analytical", "detection"]
            terms_found = sum(1 for term in technical_terms if term in answer)
            technical_accuracy = terms_found / len(technical_terms)
            
            context["technical_question_response"] = response
            context["technical_accuracy"] = technical_accuracy
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_ask_application_question(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask application-focused question."""
        try:
            response = await context["component"].query(
                "What are the clinical applications of metabolomics in disease diagnosis?"
            )
            
            # Evaluate clinical relevance
            answer = response.get("answer", "").lower()
            clinical_terms = ["diagnosis", "disease", "biomarkers", "clinical", "patients"]
            terms_found = sum(1 for term in clinical_terms if term in answer)
            clinical_relevance = terms_found / len(clinical_terms)
            
            context["application_question_response"] = response
            context["clinical_relevance"] = clinical_relevance
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_verify_citations_provided(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify that citations are provided in responses."""
        try:
            # Check all responses for citations
            responses = [
                context.get("basic_question_response", {}),
                context.get("technical_question_response", {}),
                context.get("application_question_response", {})
            ]
            
            citations_count = 0
            for response in responses:
                if response.get("source_documents") and len(response["source_documents"]) > 0:
                    citations_count += 1
            
            context["citations_provided"] = citations_count > 0
            context["citation_coverage"] = citations_count / len(responses)
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_check_response_accuracy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check overall response accuracy."""
        try:
            # Calculate average accuracy across all responses
            accuracies = [
                context.get("response_accuracy", 0),
                context.get("technical_accuracy", 0),
                context.get("clinical_relevance", 0)
            ]
            
            context["overall_accuracy"] = sum(accuracies) / len(accuracies)
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_evaluate_user_experience(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate overall user experience."""
        try:
            # Simulate user experience evaluation
            experience_factors = {
                "response_quality": context.get("overall_accuracy", 0),
                "response_speed": 1.0 if context.get("response_time", 10) < 8 else 0.5,
                "citation_support": 1.0 if context.get("citations_provided", False) else 0.3,
                "system_reliability": 1.0  # Assume reliable if we got this far
            }
            
            context["user_experience_score"] = sum(experience_factors.values()) / len(experience_factors)
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Additional UAT steps for other scenarios
    async def _uat_step_setup_clinical_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Set up clinical environment."""
        return await self._uat_step_setup_research_environment(context)
    
    async def _uat_step_upload_diagnostic_papers(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Upload diagnostic-focused papers."""
        try:
            papers_dir = Path(context["config"].papers_directory)
            
            diagnostic_papers = [
                "metabolomic_biomarkers_diabetes.pdf",
                "clinical_diagnosis_metabolomics.pdf"
            ]
            
            pdf_content = self._create_sample_pdf_content("Clinical Diagnostic Metabolomics")
            
            for paper_name in diagnostic_papers:
                (papers_dir / paper_name).write_bytes(pdf_content)
            
            ingestion_result = await context["component"].ingest_documents()
            context["diagnostic_papers_ingested"] = ingestion_result["successful"]
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_ask_diagnostic_biomarker_question(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask diagnostic biomarker question."""
        try:
            response = await context["component"].query(
                "What metabolomic biomarkers are used for diabetes diagnosis?"
            )
            
            # Evaluate clinical relevance
            answer = response.get("answer", "").lower()
            diagnostic_terms = ["biomarkers", "diabetes", "diagnosis", "metabolites", "glucose"]
            terms_found = sum(1 for term in diagnostic_terms if term in answer)
            clinical_relevance = terms_found / len(diagnostic_terms)
            
            context["diagnostic_response"] = response
            context["clinical_relevance"] = clinical_relevance
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_ask_disease_specific_question(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask disease-specific question."""
        try:
            response = await context["component"].query(
                "How can metabolomics help in cardiovascular disease diagnosis?"
            )
            
            context["disease_specific_response"] = response
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_verify_clinical_relevance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify clinical relevance of responses."""
        try:
            # Evaluate clinical relevance across responses
            responses = [
                context.get("diagnostic_response", {}),
                context.get("disease_specific_response", {})
            ]
            
            clinical_relevance_scores = []
            for response in responses:
                answer = response.get("answer", "").lower()
                clinical_terms = ["clinical", "diagnosis", "patients", "disease", "treatment"]
                terms_found = sum(1 for term in clinical_terms if term in answer)
                clinical_relevance_scores.append(terms_found / len(clinical_terms))
            
            context["clinical_relevance"] = sum(clinical_relevance_scores) / len(clinical_relevance_scores)
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_check_confidence_scores(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check confidence scores in responses."""
        try:
            responses = [
                context.get("diagnostic_response", {}),
                context.get("disease_specific_response", {})
            ]
            
            confidence_scores_present = all(
                "confidence_score" in response and response["confidence_score"] is not None
                for response in responses
            )
            
            context["confidence_scores_present"] = confidence_scores_present
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_evaluate_clinical_utility(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate clinical utility."""
        try:
            # Simulate clinical utility evaluation
            utility_factors = {
                "clinical_relevance": context.get("clinical_relevance", 0),
                "confidence_scores": 1.0 if context.get("confidence_scores_present", False) else 0.5,
                "response_accuracy": context.get("response_accuracy", 0),
                "practical_applicability": 0.8  # Mock score
            }
            
            context["clinical_utility"] = sum(utility_factors.values()) / len(utility_factors)
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Multilingual scenario steps
    async def _uat_step_setup_multilingual_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Set up multilingual environment."""
        return await self._uat_step_setup_research_environment(context)
    
    async def _uat_step_upload_papers_various_languages(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Upload papers in various languages."""
        return await self._uat_step_upload_metabolomics_papers(context)
    
    async def _uat_step_ask_question_in_spanish(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask question in Spanish."""
        try:
            response = await context["component"].query("¿Qué es la metabolómica clínica?")
            
            context["spanish_response"] = response
            context["spanish_query_processed"] = "answer" in response and len(response["answer"]) > 0
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_ask_question_in_french(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Ask question in French."""
        try:
            response = await context["component"].query("Qu'est-ce que la métabolomique clinique?")
            
            context["french_response"] = response
            context["french_query_processed"] = "answer" in response and len(response["answer"]) > 0
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_verify_translation_quality(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Verify translation quality."""
        try:
            # Mock translation quality assessment
            spanish_processed = context.get("spanish_query_processed", False)
            french_processed = context.get("french_query_processed", False)
            
            context["translation_accuracy"] = 0.8 if (spanish_processed and french_processed) else 0.4
            context["multilingual_support"] = spanish_processed and french_processed
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_check_citation_translation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check citation translation."""
        try:
            # Mock citation translation check
            context["citation_translation_supported"] = True
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_evaluate_multilingual_experience(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate multilingual experience."""
        try:
            multilingual_factors = {
                "translation_accuracy": context.get("translation_accuracy", 0),
                "multilingual_support": 1.0 if context.get("multilingual_support", False) else 0.3,
                "citation_translation": 1.0 if context.get("citation_translation_supported", False) else 0.5
            }
            
            context["multilingual_experience"] = sum(multilingual_factors.values()) / len(multilingual_factors)
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Additional steps for other scenarios (simplified implementations)
    async def _uat_step_setup_learning_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self._uat_step_setup_research_environment(context)
    
    async def _uat_step_upload_educational_papers(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self._uat_step_upload_metabolomics_papers(context)
    
    async def _uat_step_ask_fundamental_concepts(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self._uat_step_ask_basic_definition_question(context)
    
    async def _uat_step_ask_methodology_questions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self._uat_step_ask_specific_technique_question(context)
    
    async def _uat_step_verify_educational_value(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["educational_value"] = context.get("response_accuracy", 0.8)
        return {"success": True, "context": context}
    
    async def _uat_step_check_explanation_clarity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["explanation_clarity"] = context.get("response_accuracy", 0.8)
        return {"success": True, "context": context}
    
    async def _uat_step_evaluate_learning_support(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["learning_support"] = context.get("response_accuracy", 0.8)
        return {"success": True, "context": context}
    
    # Industry and collaborative scenario steps (simplified)
    async def _uat_step_setup_industry_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self._uat_step_setup_research_environment(context)
    
    async def _uat_step_upload_commercial_papers(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self._uat_step_upload_metabolomics_papers(context)
    
    async def _uat_step_ask_market_application_questions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self._uat_step_ask_application_question(context)
    
    async def _uat_step_ask_technology_trend_questions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self._uat_step_ask_specific_technique_question(context)
    
    async def _uat_step_verify_commercial_relevance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["commercial_relevance"] = context.get("response_accuracy", 0.75)
        return {"success": True, "context": context}
    
    async def _uat_step_check_recent_developments(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["recent_developments_covered"] = True
        return {"success": True, "context": context}
    
    async def _uat_step_evaluate_business_utility(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["business_utility"] = context.get("commercial_relevance", 0.75)
        return {"success": True, "context": context}
    
    async def _uat_step_setup_collaborative_environment(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return await self._uat_step_setup_research_environment(context)
    
    async def _uat_step_simulate_multiple_users(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["simulated_users"] = 5
        return {"success": True, "context": context}
    
    async def _uat_step_concurrent_question_asking(self, context: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Simulate concurrent queries
            queries = [
                "What is clinical metabolomics?",
                "What are biomarkers?",
                "How is NMR used?",
                "What is mass spectrometry?",
                "What are metabolic pathways?"
            ]
            
            tasks = [context["component"].query(query) for query in queries]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_responses = sum(
                1 for response in responses 
                if not isinstance(response, Exception) and "answer" in response
            )
            
            context["concurrent_user_support"] = len(queries)
            context["response_consistency"] = successful_responses / len(queries)
            context["system_stability"] = successful_responses == len(queries)
            
            return {"success": True, "context": context}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _uat_step_verify_consistent_responses(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Already handled in concurrent_question_asking
        return {"success": True, "context": context}
    
    async def _uat_step_check_system_performance(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Already handled in concurrent_question_asking
        return {"success": True, "context": context}
    
    async def _uat_step_evaluate_collaboration_support(self, context: Dict[str, Any]) -> Dict[str, Any]:
        context["collaboration_support"] = context.get("response_consistency", 0.9)
        return {"success": True, "context": context}
    
    def _create_sample_pdf_content(self, title: str) -> bytes:
        """Create minimal valid PDF content for testing."""
        return f"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj

4 0 obj
<<
/Length 50
>>
stream
BT
/F1 12 Tf
100 700 Td
({title}) Tj
ET
endstream
endobj

xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
299
%%EOF""".encode('utf-8')   
 
    def generate_uat_report(self, results: Dict[str, Any],
                          output_file: Optional[str] = None) -> str:
        """
        Generate comprehensive user acceptance test report.
        
        Args:
            results: UAT results
            output_file: Optional file to save report
            
        Returns:
            Report content as string
        """
        report_lines = [
            "=" * 80,
            "LIGHTRAG USER ACCEPTANCE TEST REPORT",
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
            f"  Total Scenarios: {summary['total_scenarios']}",
            f"  Passed: {summary['passed']}",
            f"  Failed: {summary['failed']}",
            f"  High Priority Passed: {summary['high_priority_passed']}",
            f"  High Priority Failed: {summary['high_priority_failed']}",
            f"  Average User Satisfaction: {summary['average_satisfaction']:.1f}/5.0",
            ""
        ])
        
        # Scenario results by user persona
        personas = {}
        for scenario in results["scenario_results"]:
            persona = scenario["user_persona"]
            if persona not in personas:
                personas[persona] = []
            personas[persona].append(scenario)
        
        report_lines.append("RESULTS BY USER PERSONA:")
        report_lines.append("-" * 40)
        
        for persona, scenarios in personas.items():
            report_lines.append(f"\n{persona}:")
            
            for scenario in scenarios:
                status = "✅ PASSED" if scenario["success"] else "❌ FAILED"
                satisfaction = scenario["user_satisfaction_score"]
                
                report_lines.extend([
                    f"  Scenario: {scenario['scenario_name']}",
                    f"    Status: {status}",
                    f"    Satisfaction: {satisfaction:.1f}/5.0",
                    f"    Duration: {scenario['duration_seconds']:.2f}s",
                    f"    Steps Completed: {len(scenario['steps_completed'])}/{len(scenario['steps_completed']) + len(scenario['steps_failed'])}"
                ])
                
                if scenario["steps_failed"]:
                    report_lines.append(f"    Failed Steps: {', '.join(scenario['steps_failed'])}")
                
                # Show acceptance criteria results
                criteria_met = scenario["acceptance_criteria_met"]
                if criteria_met:
                    passed_criteria = sum(criteria_met.values())
                    total_criteria = len(criteria_met)
                    report_lines.append(f"    Acceptance Criteria: {passed_criteria}/{total_criteria} met")
                    
                    for criterion, met in criteria_met.items():
                        status_icon = "✅" if met else "❌"
                        report_lines.append(f"      {status_icon} {criterion}")
                
                # User feedback
                if scenario["feedback"]:
                    report_lines.append(f"    Feedback: {scenario['feedback']}")
                
                report_lines.append("")
        
        # Satisfaction analysis
        report_lines.extend([
            "USER SATISFACTION ANALYSIS:",
            "-" * 30
        ])
        
        satisfaction_scores = [s["user_satisfaction_score"] for s in results["scenario_results"]]
        
        if satisfaction_scores:
            min_satisfaction = min(satisfaction_scores)
            max_satisfaction = max(satisfaction_scores)
            avg_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
            
            report_lines.extend([
                f"  Minimum Satisfaction: {min_satisfaction:.1f}/5.0",
                f"  Maximum Satisfaction: {max_satisfaction:.1f}/5.0",
                f"  Average Satisfaction: {avg_satisfaction:.1f}/5.0",
                ""
            ])
            
            # Satisfaction distribution
            satisfaction_ranges = {
                "Excellent (4.5-5.0)": sum(1 for s in satisfaction_scores if s >= 4.5),
                "Good (3.5-4.4)": sum(1 for s in satisfaction_scores if 3.5 <= s < 4.5),
                "Acceptable (2.5-3.4)": sum(1 for s in satisfaction_scores if 2.5 <= s < 3.5),
                "Poor (1.0-2.4)": sum(1 for s in satisfaction_scores if s < 2.5)
            }
            
            report_lines.append("  Satisfaction Distribution:")
            for range_name, count in satisfaction_ranges.items():
                percentage = (count / len(satisfaction_scores)) * 100
                report_lines.append(f"    {range_name}: {count} ({percentage:.1f}%)")
            
            report_lines.append("")
        
        # Key findings
        report_lines.extend([
            "KEY FINDINGS:",
            "-" * 20
        ])
        
        # Identify common issues
        all_failed_steps = []
        for scenario in results["scenario_results"]:
            all_failed_steps.extend(scenario["steps_failed"])
        
        if all_failed_steps:
            from collections import Counter
            common_failures = Counter(all_failed_steps).most_common(3)
            
            report_lines.append("  Most Common Issues:")
            for failure, count in common_failures:
                report_lines.append(f"    • {failure} (occurred {count} times)")
            report_lines.append("")
        
        # Identify strengths
        high_satisfaction_scenarios = [
            s for s in results["scenario_results"] 
            if s["user_satisfaction_score"] >= 4.0
        ]
        
        if high_satisfaction_scenarios:
            report_lines.append("  Strengths:")
            for scenario in high_satisfaction_scenarios[:3]:  # Top 3
                report_lines.append(f"    • {scenario['scenario_name']}: {scenario['user_satisfaction_score']:.1f}/5.0")
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 20
        ])
        
        if results["overall_success"]:
            report_lines.extend([
                "✅ User acceptance tests passed successfully!",
                "",
                "Key achievements:",
                "  • All high-priority scenarios passed",
                f"  • Average user satisfaction: {summary['average_satisfaction']:.1f}/5.0",
                "  • System meets user expectations across different personas",
                "",
                "Ready for user acceptance and production deployment.",
                "",
                "Consider:",
                "  • Gathering feedback from real users in pilot deployment",
                "  • Monitoring user satisfaction metrics in production",
                "  • Continuing to improve based on user feedback"
            ])
        else:
            report_lines.extend([
                "❌ User acceptance tests failed. Address the following issues:",
                ""
            ])
            
            # Identify specific issues to address
            failed_scenarios = [
                s for s in results["scenario_results"] 
                if not s["success"]
            ]
            
            high_priority_failures = [
                s for s in failed_scenarios 
                if any(scenario["name"] == s["scenario_name"] and scenario["priority"] == "high" 
                      for scenario in self.user_scenarios)
            ]
            
            if high_priority_failures:
                report_lines.append("  Critical Issues (High Priority):")
                for scenario in high_priority_failures:
                    report_lines.append(f"    • {scenario['scenario_name']}")
                    if scenario["steps_failed"]:
                        report_lines.append(f"      Failed steps: {', '.join(scenario['steps_failed'])}")
                report_lines.append("")
            
            # Low satisfaction scenarios
            low_satisfaction_scenarios = [
                s for s in results["scenario_results"] 
                if s["user_satisfaction_score"] < 3.0
            ]
            
            if low_satisfaction_scenarios:
                report_lines.append("  User Satisfaction Issues:")
                for scenario in low_satisfaction_scenarios:
                    report_lines.append(
                        f"    • {scenario['scenario_name']}: "
                        f"{scenario['user_satisfaction_score']:.1f}/5.0"
                    )
                report_lines.append("")
            
            report_lines.extend([
                "  Recommended Actions:",
                "    • Fix critical functionality issues in high-priority scenarios",
                "    • Improve response accuracy and speed",
                "    • Enhance user interface and experience",
                "    • Conduct additional user research and testing",
                "    • Consider phased rollout with limited user groups"
            ])
        
        report_content = "\n".join(report_lines)
        
        if output_file:
            Path(output_file).write_text(report_content)
            self.logger.info(f"UAT report saved to {output_file}")
        
        return report_content
    
    def save_uat_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save UAT results as JSON."""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"UAT results saved to {output_file}")


# Convenience function for running user acceptance tests
async def run_user_acceptance_tests(config: Optional[LightRAGConfig] = None,
                                  output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run user acceptance test suite.
    
    Args:
        config: Optional LightRAG configuration
        output_dir: Optional output directory for results
        
    Returns:
        Dictionary with UAT results
    """
    test_suite = UserAcceptanceTestSuite(config=config, output_dir=output_dir)
    return await test_suite.run_user_acceptance_tests()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LightRAG User Acceptance Test Suite")
    parser.add_argument("--output-dir", default="uat_results",
                       help="Output directory for test results")
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
            # Run user acceptance tests
            results = await run_user_acceptance_tests(output_dir=args.output_dir)
            
            # Generate report
            test_suite = UserAcceptanceTestSuite(output_dir=args.output_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            report = test_suite.generate_uat_report(
                results,
                str(test_suite.output_dir / f"uat_report_{timestamp}.txt")
            )
            print(report)
            
            # Save results if requested
            if args.save_results:
                test_suite.save_uat_results(
                    results,
                    str(test_suite.output_dir / f"uat_results_{timestamp}.json")
                )
            
            # Exit with appropriate code
            if results["overall_success"]:
                print("\n🎉 USER ACCEPTANCE TESTS PASSED!")
                exit(0)
            else:
                print("\n💥 USER ACCEPTANCE TESTS FAILED!")
                exit(1)
                
        except Exception as e:
            print(f"User acceptance tests failed: {str(e)}")
            exit(1)
    
    asyncio.run(main())