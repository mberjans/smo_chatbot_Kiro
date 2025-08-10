# LightRAG Testing and Validation

This module provides comprehensive testing and validation capabilities for the LightRAG integration with the Clinical Metabolomics Oracle system. It includes both functional validation tests and performance benchmarking to ensure the MVP meets all requirements.

## Overview

The testing system consists of three main components:

1. **Clinical Metabolomics Test Suite** - Validates accuracy and functionality
2. **Performance Benchmark** - Measures response times, memory usage, and throughput
3. **Automated Testing Pipeline** - Provides continuous validation and reporting

## Components

### Clinical Metabolomics Test Suite (`clinical_metabolomics_suite.py`)

Provides comprehensive validation of clinical metabolomics knowledge and functionality:

- **Test Dataset**: 10 carefully crafted questions covering basic to advanced topics
- **Mock Papers**: Generates test papers with clinical metabolomics content
- **Accuracy Measurement**: Evaluates keyword matches, concept coverage, and confidence scores
- **Automated Validation**: Runs the critical "What is clinical metabolomics?" test

**Key Features:**
- EARS format requirements validation
- Category-based analysis (definition, application, technical, etc.)
- Difficulty-based analysis (basic, intermediate, advanced)
- Comprehensive reporting with pass/fail criteria

### Performance Benchmark (`performance_benchmark.py`)

Measures system performance across multiple dimensions:

- **Response Time Measurement**: Tracks query processing times
- **Memory Usage Monitoring**: Real-time memory consumption tracking
- **Load Testing**: Simulates concurrent users and measures throughput
- **Stress Testing**: Gradually increases load to find system limits
- **Regression Analysis**: Compares current performance to baseline

**Key Metrics:**
- Average/min/max response times
- Memory usage (current/peak/average)
- Requests per second
- 95th/99th percentile response times
- CPU utilization

### Automated Testing Pipeline (`automated_pipeline.py`)

Provides continuous validation and integration with CI/CD systems:

- **Scheduled Testing**: Run validation at regular intervals
- **Success Criteria Evaluation**: Automated pass/fail determination
- **Trend Analysis**: Track performance over time
- **Report Generation**: Human-readable and JSON reports
- **Cleanup Management**: Automatic cleanup of old test results

### MVP Test Runner (`test_runner.py`)

Unified test runner that combines all testing components:

- **Complete MVP Validation**: Runs both functional and performance tests
- **Criteria Evaluation**: Checks against MVP success criteria
- **Comprehensive Reporting**: Generates detailed validation reports
- **Result Persistence**: Saves results for analysis and tracking

## Usage

### Running Individual Test Suites

#### Clinical Metabolomics Validation
```bash
# Run validation test suite
python -m src.lightrag_integration.testing.clinical_metabolomics_suite

# Run with custom configuration
python -c "
import asyncio
from src.lightrag_integration.testing.clinical_metabolomics_suite import run_mvp_validation_test
from src.lightrag_integration.config.settings import LightRAGConfig

config = LightRAGConfig()
config.papers_directory = 'custom_papers'
results = asyncio.run(run_mvp_validation_test(config))
print(f'Pass rate: {results.passed_questions}/{results.total_questions}')
"
```

#### Performance Benchmark
```bash
# Run performance benchmark
python -m src.lightrag_integration.testing.performance_benchmark

# Run with custom configuration
python -c "
import asyncio
from src.lightrag_integration.testing.performance_benchmark import run_performance_benchmark
results = asyncio.run(run_performance_benchmark())
print(f'Average response time: {results.summary[\"average_response_time\"]:.3f}s')
"
```

### Running Complete MVP Validation

```bash
# Run complete MVP validation
python -m src.lightrag_integration.testing.test_runner

# Run with custom output directory
python -m src.lightrag_integration.testing.test_runner --output-dir my_results --save-results

# Run with verbose logging
python -m src.lightrag_integration.testing.test_runner --verbose
```

### Automated Testing Pipeline

```bash
# Run single validation
python -m src.lightrag_integration.testing.automated_pipeline --mode single

# Run continuous validation (every 24 hours)
python -m src.lightrag_integration.testing.automated_pipeline --mode continuous --interval 24

# Generate trend report for last 7 days
python -m src.lightrag_integration.testing.automated_pipeline --mode trend --trend-days 7

# Run with cleanup of old results
python -m src.lightrag_integration.testing.automated_pipeline --mode trend --cleanup-days 30
```

## MVP Success Criteria

The MVP validation uses the following success criteria:

### Functional Criteria
- **Pass Rate**: ≥85% of validation tests must pass
- **Accuracy**: Average accuracy ≥75%
- **Confidence**: Average confidence ≥60%
- **Core Question**: Must correctly answer "What is clinical metabolomics?"

### Performance Criteria
- **Response Time**: Average ≤5.0 seconds
- **Memory Usage**: Peak ≤2048 MB
- **Throughput**: ≥1.0 requests per second

## Test Questions

The clinical metabolomics test suite includes questions covering:

1. **Basic Definitions**
   - What is clinical metabolomics?
   - What are metabolites?
   - Types of biological samples

2. **Applications**
   - Disease diagnosis
   - Personalized medicine
   - Drug development

3. **Technical Aspects**
   - Analytical techniques
   - Study workflows
   - Targeted vs untargeted approaches

4. **Challenges**
   - Technical limitations
   - Standardization issues
   - Clinical translation

## Output Files

The testing system generates several types of output files:

### Validation Reports
- `validation_report_YYYYMMDD_HHMMSS.txt` - Human-readable validation report
- `validation_results_YYYYMMDD_HHMMSS.json` - Detailed validation results

### Performance Reports
- `performance_benchmark_results.json` - Detailed benchmark results
- `performance_report_YYYYMMDD_HHMMSS.txt` - Human-readable benchmark report

### MVP Reports
- `mvp_report_YYYYMMDD_HHMMSS.txt` - Complete MVP validation report
- `mvp_results_YYYYMMDD_HHMMSS.json` - Detailed MVP validation results

### Pipeline Logs
- `pipeline.log` - Automated pipeline execution logs
- `test_runner.log` - Test runner execution logs

## Integration with CI/CD

The testing system is designed for easy integration with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
name: LightRAG MVP Validation
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run MVP validation
        run: python -m src.lightrag_integration.testing.test_runner
```

## Extending the Test Suite

### Adding New Test Questions

To add new test questions, modify the `_create_test_dataset()` method in `ClinicalMetabolomicsTestSuite`:

```python
TestQuestion(
    question="Your new question?",
    expected_keywords=["keyword1", "keyword2"],
    expected_concepts=["concept1", "concept2"],
    minimum_confidence=0.6,
    category="your_category",
    difficulty="basic|intermediate|advanced",
    description="Description of what this tests"
)
```

### Adding New Performance Metrics

To add new performance metrics, extend the `PerformanceMetrics` dataclass and modify the measurement methods in `PerformanceBenchmark`.

### Customizing Success Criteria

Modify the criteria dictionaries in the respective classes:

```python
# In ClinicalMetabolomicsTestSuite
self.success_criteria = {
    "minimum_pass_rate": 0.90,  # Increase to 90%
    # ... other criteria
}

# In PerformanceBenchmark
self.thresholds = {
    "max_response_time": 3.0,  # Decrease to 3 seconds
    # ... other thresholds
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
2. **Permission Errors**: Check write permissions for output directories
3. **Memory Issues**: Reduce concurrent users in load tests for resource-constrained environments
4. **Timeout Issues**: Increase timeout values for slower systems

### Debug Mode

Run tests with verbose logging to get detailed information:

```bash
python -m src.lightrag_integration.testing.test_runner --verbose
```

### Test Data Issues

If test papers are not being created properly, check:
- Write permissions in the papers directory
- Available disk space
- File system limitations

## Requirements

The testing system requires the following additional dependencies:
- `psutil>=7.0.0` - System monitoring
- `pytest>=6.0.0` - Test framework
- `pytest-asyncio>=0.18.0` - Async test support

These are automatically installed when you install the main requirements.txt file.