#!/usr/bin/env python3
"""
Create Sample Data for Chatbot Testing

This script creates sample clinical metabolomics content to test the chatbot
functionality without requiring real PDF files.
"""

import os
import sys
from pathlib import Path

def create_sample_text_files():
    """Create sample text files with clinical metabolomics content"""
    
    # Create papers directory
    papers_dir = Path("papers")
    papers_dir.mkdir(exist_ok=True)
    
    # Sample clinical metabolomics content
    sample_papers = {
        "clinical_metabolomics_overview.txt": """
Clinical Metabolomics: An Overview

Clinical metabolomics is the application of metabolomics technologies and approaches to understand disease mechanisms, identify biomarkers, and support clinical decision-making. This field focuses on the comprehensive analysis of small molecules (metabolites) in biological samples such as blood, urine, and tissue.

Key Applications:
1. Disease Diagnosis: Metabolomics can identify metabolic signatures associated with specific diseases
2. Drug Development: Understanding how drugs affect metabolic pathways
3. Personalized Medicine: Tailoring treatments based on individual metabolic profiles
4. Biomarker Discovery: Finding metabolites that indicate disease states or treatment responses

Analytical Techniques:
- Mass Spectrometry (MS)
- Nuclear Magnetic Resonance (NMR) Spectroscopy
- Liquid Chromatography-Mass Spectrometry (LC-MS)
- Gas Chromatography-Mass Spectrometry (GC-MS)

Clinical metabolomics represents a powerful approach for advancing precision medicine and improving patient outcomes through better understanding of metabolic processes in health and disease.
        """,
        
        "metabolomics_biomarkers.txt": """
Metabolomics Biomarkers in Clinical Practice

Metabolomics biomarkers are metabolites or patterns of metabolites that can indicate biological states, disease processes, or responses to therapeutic interventions. These biomarkers have significant potential in clinical applications.

Types of Metabolomics Biomarkers:
1. Diagnostic Biomarkers: Help identify the presence of disease
2. Prognostic Biomarkers: Predict disease progression or outcomes
3. Predictive Biomarkers: Forecast response to specific treatments
4. Pharmacodynamic Biomarkers: Monitor drug effects and mechanisms

Clinical Applications:
- Cancer Detection: Metabolic signatures can distinguish between healthy and cancerous tissues
- Cardiovascular Disease: Lipid profiles and metabolic markers for heart disease risk
- Diabetes: Glucose metabolism markers and insulin resistance indicators
- Neurological Disorders: Brain metabolites associated with cognitive function

Challenges in Clinical Implementation:
- Standardization of analytical methods
- Validation across diverse populations
- Integration with existing clinical workflows
- Regulatory approval processes

The future of metabolomics biomarkers lies in their integration with other omics data and clinical information to provide comprehensive patient profiles for precision medicine.
        """,
        
        "analytical_techniques_metabolomics.txt": """
Analytical Techniques in Clinical Metabolomics

Clinical metabolomics relies on sophisticated analytical techniques to identify and quantify metabolites in biological samples. The choice of technique depends on the research question, sample type, and target metabolites.

Mass Spectrometry (MS):
- High sensitivity and specificity
- Can identify unknown metabolites
- Requires sample preparation and separation techniques
- Common platforms: LC-MS, GC-MS, CE-MS

Nuclear Magnetic Resonance (NMR):
- Non-destructive analysis
- Provides structural information
- Good for quantitative analysis
- Less sensitive than MS but highly reproducible

Liquid Chromatography-Mass Spectrometry (LC-MS):
- Most widely used in metabolomics
- Excellent for polar and semi-polar metabolites
- High throughput capabilities
- Suitable for targeted and untargeted approaches

Gas Chromatography-Mass Spectrometry (GC-MS):
- Best for volatile and derivatizable compounds
- High reproducibility and extensive metabolite libraries
- Limited to thermally stable compounds
- Requires derivatization for many metabolites

Sample Preparation Considerations:
- Sample collection and storage protocols
- Extraction methods for different metabolite classes
- Quality control and standardization
- Batch effects and normalization strategies

Data Analysis Challenges:
- Peak detection and alignment
- Metabolite identification and annotation
- Statistical analysis and interpretation
- Integration with clinical data

The continued development of analytical techniques and data analysis methods is crucial for advancing clinical metabolomics applications.
        """,
        
        "personalized_medicine_metabolomics.txt": """
Metabolomics in Personalized Medicine

Personalized medicine aims to tailor medical treatment to individual characteristics, and metabolomics plays a crucial role in this approach by providing insights into individual metabolic profiles and drug responses.

Key Concepts:
1. Metabolic Phenotyping: Characterizing individual metabolic profiles
2. Pharmacometabolomics: Understanding drug metabolism and response
3. Precision Dosing: Optimizing drug doses based on metabolic capacity
4. Treatment Selection: Choosing therapies based on metabolic biomarkers

Applications in Drug Development:
- Identifying metabolic pathways affected by drugs
- Predicting drug efficacy and toxicity
- Understanding drug resistance mechanisms
- Developing companion diagnostics

Clinical Implementation:
- Pre-treatment metabolic profiling
- Monitoring treatment response
- Adjusting therapy based on metabolic changes
- Identifying adverse drug reactions early

Examples of Success:
- Warfarin dosing based on metabolic capacity
- Cancer treatment selection using metabolic signatures
- Psychiatric medication optimization
- Diabetes management through metabolic monitoring

Challenges and Future Directions:
- Standardization of metabolomics protocols
- Integration with electronic health records
- Cost-effectiveness considerations
- Training healthcare providers in metabolomics interpretation

The integration of metabolomics into clinical practice represents a significant step toward truly personalized medicine, enabling more effective and safer treatments for individual patients.
        """,
        
        "metabolomics_data_analysis.txt": """
Data Analysis in Clinical Metabolomics

Data analysis is a critical component of clinical metabolomics studies, involving complex computational approaches to extract meaningful biological information from large-scale metabolite datasets.

Data Processing Pipeline:
1. Raw Data Processing: Peak detection, alignment, and normalization
2. Quality Control: Identifying and removing poor-quality samples or features
3. Statistical Analysis: Univariate and multivariate statistical methods
4. Metabolite Identification: Matching spectral data to metabolite databases
5. Pathway Analysis: Understanding biological significance of findings

Statistical Methods:
- Principal Component Analysis (PCA)
- Partial Least Squares Discriminant Analysis (PLS-DA)
- Random Forest and Machine Learning approaches
- Correlation analysis and network construction
- Time-series analysis for longitudinal studies

Challenges in Data Analysis:
- High dimensionality and small sample sizes
- Missing values and data imputation
- Batch effects and technical variation
- Multiple testing correction
- Biological interpretation of results

Software and Tools:
- R packages: MetaboAnalyst, XCMS, MZmine
- Commercial software: Compound Discoverer, Progenesis QI
- Database resources: HMDB, KEGG, MetaCyc
- Pathway analysis tools: MetaboAnalyst, GSEA

Quality Assurance:
- Use of quality control samples
- Validation of statistical models
- Cross-validation and external validation
- Reproducibility assessment
- Documentation of analysis workflows

Best Practices:
- Pre-registration of analysis plans
- Transparent reporting of methods
- Sharing of data and analysis code
- Collaboration between analysts and clinicians
- Continuous method development and validation

Effective data analysis is essential for translating metabolomics discoveries into clinically actionable insights.
        """
    }
    
    # Write sample files
    for filename, content in sample_papers.items():
        file_path = papers_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        print(f"Created: {file_path}")
    
    print(f"\n✅ Created {len(sample_papers)} sample text files in {papers_dir}")
    print("These files contain clinical metabolomics content for testing the chatbot.")
    
    return list(sample_papers.keys())

def create_sample_questions():
    """Create sample questions for testing"""
    
    questions = [
        "What is clinical metabolomics?",
        "What are the main analytical techniques used in metabolomics?",
        "How is metabolomics used in personalized medicine?",
        "What are metabolomics biomarkers?",
        "What are the challenges in metabolomics data analysis?",
        "How does mass spectrometry work in metabolomics?",
        "What is the role of NMR in metabolomics?",
        "How are metabolomics biomarkers used in cancer detection?",
        "What is pharmacometabolomics?",
        "What are the applications of metabolomics in drug development?"
    ]
    
    questions_file = Path("sample_questions.txt")
    with open(questions_file, 'w') as f:
        f.write("Sample Questions for Clinical Metabolomics Oracle Testing\n")
        f.write("=" * 60 + "\n\n")
        for i, question in enumerate(questions, 1):
            f.write(f"{i}. {question}\n")
    
    print(f"\n✅ Created {questions_file} with {len(questions)} sample questions")
    
    return questions

def main():
    """Main function to create sample data"""
    print("🧪 Creating Sample Data for Chatbot Testing")
    print("=" * 50)
    
    # Create sample text files
    sample_files = create_sample_text_files()
    
    # Create sample questions
    sample_questions = create_sample_questions()
    
    print("\n📋 Next Steps:")
    print("1. Run the chatbot with: python src/main.py")
    print("2. Try asking questions from sample_questions.txt")
    print("3. The chatbot should now have content to work with")
    
    print(f"\n📁 Files created:")
    for file in sample_files:
        print(f"   - papers/{file}")
    print(f"   - sample_questions.txt")
    
    print("\n🎯 Test the chatbot by asking:")
    print(f"   '{sample_questions[0]}'")

if __name__ == "__main__":
    main()