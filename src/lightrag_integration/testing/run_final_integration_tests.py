#!/usr/bin/env python3
"""
Run Final Integration Tests

This script orchestrates the complete final integration and system testing
process, validating all requirements and preparing the system for production
deployment.

Usage:
    python run_final_integration_tests.py [--config CONFIG_FILE] [--verbose]
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import argparse
from datetime import datetime

# Import the final integration test suite
from .final_integration_test_suite import FinalIntegrationTestSuite, SystemTestReport

class FinalIntegrationTestRunner:
    """Runner for final integration tests with comprehensive reporting"""
    
    def __init__(self, config_path: str = None, verbose: bool = False):
        """Initialize the test runner"""
        self.config_path = config_path
        self.verbose = verbose
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        level = logging.DEBUG if self.verbose else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('final_integration_tests.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    async def run_all_tests(self) -> SystemTestReport:
        """Run all final integration tests"""
        self.logger.info("Starting final integration and system testing")
        
        try:
            # Initialize test suite
            test_suite = FinalIntegrationTestSuite(self.config_path)
            
            # Run complete system test
            report = await test_suite.run_complete_system_test()
            
            # Generate detailed reports
            await self.generate_detailed_reports(report)
            
            # Print summary
            self.print_test_summary(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Final integration testing failed: {e}")
            raise
    
    async def generate_detailed_reports(self, report: SystemTestReport):
        """Generate detailed test reports"""
        report_dir = Path("test_reports")
        report_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON report
        await self.generate_json_report(report, report_dir, timestamp)
        
        # Generate HTML report
        await self.generate_html_report(report, report_dir, timestamp)
        
        # Generate CSV summary
        await self.generate_csv_summary(report, report_dir, timestamp)
        
        # Generate deployment checklist
        await self.generate_deployment_checklist(report, report_dir, timestamp)
    
    async def generate_json_report(self, report: SystemTestReport, report_dir: Path, timestamp: str):
        """Generate JSON test report"""
        json_file = report_dir / f"final_integration_report_{timestamp}.json"
        
        # Convert report to JSON-serializable format
        report_dict = {
            "test_timestamp": report.test_timestamp.isoformat(),
            "overall_passed": report.overall_passed,
            "total_tests": report.total_tests,
            "passed_tests": report.passed_tests,
            "failed_tests": report.failed_tests,
            "overall_score": report.overall_score,
            "deployment_readiness": report.deployment_readiness,
            "requirement_validations": [
                {
                    "requirement_id": req.requirement_id,
                    "description": req.description,
                    "passed": req.passed,
                    "score": req.score,
                    "acceptance_criteria_met": req.acceptance_criteria_met,
                    "test_results": [
                        {
                            "test_name": test.test_name,
                            "passed": test.passed,
                            "score": test.score,
                            "execution_time": test.execution_time,
                            "details": test.details,
                            "error_message": test.error_message
                        }
                        for test in req.test_results
                    ]
                }
                for req in report.requirement_validations
            ],
            "performance_metrics": report.performance_metrics,
            "recommendations": report.recommendations
        }
        
        with open(json_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"JSON report saved to {json_file}")
    
    async def generate_html_report(self, report: SystemTestReport, report_dir: Path, timestamp: str):
        """Generate HTML test report"""
        html_file = report_dir / f"final_integration_report_{timestamp}.html"
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Final Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .passed {{ color: green; font-weight: bold; }}
        .failed {{ color: red; font-weight: bold; }}
        .requirement {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
        .test-result {{ margin: 5px 0; padding: 5px; background-color: #f9f9f9; }}
        .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Final Integration Test Report</h1>
        <p><strong>Test Date:</strong> {report.test_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Overall Status:</strong> <span class="{'passed' if report.overall_passed else 'failed'}">
            {'PASSED' if report.overall_passed else 'FAILED'}
        </span></p>
        <p><strong>Deployment Ready:</strong> <span class="{'passed' if report.deployment_readiness else 'failed'}">
            {'YES' if report.deployment_readiness else 'NO'}
        </span></p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests</td><td>{report.total_tests}</td></tr>
            <tr><td>Passed Tests</td><td>{report.passed_tests}</td></tr>
            <tr><td>Failed Tests</td><td>{report.failed_tests}</td></tr>
            <tr><td>Overall Score</td><td>{report.overall_score:.2%}</td></tr>
        </table>
    </div>
    
    <div class="requirements">
        <h2>Requirement Validations</h2>
        {self._generate_requirements_html(report.requirement_validations)}
    </div>
    
    <div class="performance">
        <h2>Performance Metrics</h2>
        <table>
            {self._generate_performance_table_html(report.performance_metrics)}
        </table>
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
            {chr(10).join(f'<li>{rec}</li>' for rec in report.recommendations)}
        </ul>
    </div>
</body>
</html>
        """
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved to {html_file}")
    
    def _generate_requirements_html(self, requirements):
        """Generate HTML for requirements section"""
        html = ""
        for req in requirements:
            status_class = "passed" if req.passed else "failed"
            status_text = "PASS" if req.passed else "FAIL"
            
            html += f"""
            <div class="requirement">
                <h3>{req.requirement_id}: <span class="{status_class}">{status_text}</span></h3>
                <p><strong>Description:</strong> {req.description}</p>
                <p><strong>Score:</strong> {req.score:.2%}</p>
                <p><strong>Acceptance Criteria Met:</strong> {'Yes' if req.acceptance_criteria_met else 'No'}</p>
                
                <h4>Test Results:</h4>
                {chr(10).join(f'''
                <div class="test-result">
                    <strong>{test.test_name}:</strong> 
                    <span class="{'passed' if test.passed else 'failed'}">
                        {'PASS' if test.passed else 'FAIL'}
                    </span>
                    (Score: {test.score:.2%}, Time: {test.execution_time:.2f}s)
                    {f'<br><em>Error: {test.error_message}</em>' if test.error_message else ''}
                </div>
                ''' for test in req.test_results)}
            </div>
            """
        
        return html
    
    def _generate_performance_table_html(self, metrics):
        """Generate HTML table for performance metrics"""
        html = ""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.2f}</td></tr>"
            else:
                html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        return html
    
    async def generate_csv_summary(self, report: SystemTestReport, report_dir: Path, timestamp: str):
        """Generate CSV summary of test results"""
        csv_file = report_dir / f"test_summary_{timestamp}.csv"
        
        csv_content = "Requirement ID,Description,Status,Score,Acceptance Criteria Met\n"
        
        for req in report.requirement_validations:
            status = "PASS" if req.passed else "FAIL"
            criteria_met = "Yes" if req.acceptance_criteria_met else "No"
            csv_content += f'"{req.requirement_id}","{req.description}","{status}","{req.score:.2%}","{criteria_met}"\n'
        
        with open(csv_file, 'w') as f:
            f.write(csv_content)
        
        self.logger.info(f"CSV summary saved to {csv_file}")
    
    async def generate_deployment_checklist(self, report: SystemTestReport, report_dir: Path, timestamp: str):
        """Generate deployment readiness checklist"""
        checklist_file = report_dir / f"deployment_checklist_{timestamp}.md"
        
        checklist_content = f"""# Deployment Readiness Checklist

Generated: {report.test_timestamp.strftime('%Y-%m-%d %H:%M:%S')}

## Overall Status: {'✅ READY' if report.deployment_readiness else '❌ NOT READY'}

## Test Results Summary
- Total Tests: {report.total_tests}
- Passed: {report.passed_tests}
- Failed: {report.failed_tests}
- Overall Score: {report.overall_score:.2%}

## Requirement Validation Results

"""
        
        for req in report.requirement_validations:
            status_icon = "✅" if req.passed else "❌"
            checklist_content += f"### {status_icon} Requirement {req.requirement_id}\n"
            checklist_content += f"**Description:** {req.description}\n"
            checklist_content += f"**Score:** {req.score:.2%}\n"
            checklist_content += f"**Acceptance Criteria Met:** {'Yes' if req.acceptance_criteria_met else 'No'}\n\n"
        
        checklist_content += "## Recommendations\n\n"
        for rec in report.recommendations:
            checklist_content += f"- {rec}\n"
        
        checklist_content += f"""
## Pre-Deployment Actions

### If Status is READY ✅
- [ ] Review all test results and recommendations
- [ ] Ensure monitoring and alerting are configured
- [ ] Verify backup and recovery procedures
- [ ] Schedule deployment window
- [ ] Prepare rollback plan
- [ ] Notify stakeholders

### If Status is NOT READY ❌
- [ ] Address all failed requirements
- [ ] Fix failed tests
- [ ] Re-run integration tests
- [ ] Validate performance improvements
- [ ] Update documentation as needed

## Performance Metrics
"""
        
        for key, value in report.performance_metrics.items():
            checklist_content += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        with open(checklist_file, 'w') as f:
            f.write(checklist_content)
        
        self.logger.info(f"Deployment checklist saved to {checklist_file}")
    
    def print_test_summary(self, report: SystemTestReport):
        """Print test summary to console"""
        print(f"\n{'='*80}")
        print("FINAL INTEGRATION TEST RESULTS")
        print(f"{'='*80}")
        
        # Overall status
        status_color = '\033[92m' if report.overall_passed else '\033[91m'  # Green or Red
        reset_color = '\033[0m'
        
        print(f"Overall Status: {status_color}{'PASSED' if report.overall_passed else 'FAILED'}{reset_color}")
        print(f"Deployment Ready: {status_color}{'YES' if report.deployment_readiness else 'NO'}{reset_color}")
        print(f"Test Date: {report.test_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test summary
        print(f"\nTest Summary:")
        print(f"  Total Tests: {report.total_tests}")
        print(f"  Passed: {report.passed_tests}")
        print(f"  Failed: {report.failed_tests}")
        print(f"  Overall Score: {report.overall_score:.2%}")
        
        # Requirement validations
        print(f"\nRequirement Validations:")
        for req in report.requirement_validations:
            status_icon = "✅" if req.passed else "❌"
            print(f"  {status_icon} {req.requirement_id}: {req.score:.2%} - {req.description}")
        
        # Performance metrics
        if report.performance_metrics:
            print(f"\nPerformance Metrics:")
            for key, value in report.performance_metrics.items():
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        # Recommendations
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  • {rec}")
        
        print(f"\n{'='*80}")
        print(f"Detailed reports saved to: test_reports/")
        print(f"{'='*80}\n")

async def main():
    """Main function for running final integration tests"""
    parser = argparse.ArgumentParser(
        description="Run final integration and system tests for LightRAG integration"
    )
    parser.add_argument(
        "--config", 
        help="Path to test configuration file",
        default=None
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for test reports",
        default="test_reports"
    )
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        Path(args.output_dir).mkdir(exist_ok=True)
        
        # Initialize and run test runner
        runner = FinalIntegrationTestRunner(
            config_path=args.config,
            verbose=args.verbose
        )
        
        # Run all tests
        report = await runner.run_all_tests()
        
        # Exit with appropriate code
        exit_code = 0 if report.overall_passed else 1
        
        if exit_code == 0:
            print("🎉 All tests passed! System is ready for deployment.")
        else:
            print("❌ Some tests failed. Please review the report and address issues.")
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())