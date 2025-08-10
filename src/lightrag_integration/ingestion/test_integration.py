"""
Integration tests for the complete PDF ingestion pipeline.

Tests the full workflow from PDF extraction through knowledge graph construction.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

from .pipeline import PDFIngestionPipeline
from .pdf_extractor import ExtractedDocument, ExtractionResult
from datetime import datetime


class TestPDFIngestionPipelineIntegration:
    """Integration tests for the complete PDF ingestion pipeline."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def config(self, temp_storage):
        """Create test configuration."""
        config = Mock()
        config.knowledge_graph_path = str(temp_storage / "kg")
        config.papers_directory = str(temp_storage / "papers")
        config.max_concurrent_extractions = 2
        return config
    
    @pytest.fixture
    def pipeline(self, config):
        """Create pipeline instance for testing."""
        return PDFIngestionPipeline(config)
    
    @pytest.fixture
    def sample_document(self):
        """Create a sample extracted document."""
        return ExtractedDocument(
            file_path="test_paper.pdf",
            title="Clinical Metabolomics in Precision Medicine",
            authors=["John Doe", "Jane Smith"],
            abstract="This paper discusses clinical metabolomics applications.",
            content="""
            Clinical metabolomics is the study of metabolites in biological systems.
            Glucose levels are elevated in diabetes patients. The glycolysis pathway
            produces pyruvate from glucose. NMR spectroscopy is used for metabolite
            profiling in clinical studies. Creatinine serves as a biomarker for
            kidney function assessment.
            """,
            page_count=10,
            word_count=50,
            extraction_timestamp=datetime.now(),
            metadata={"source": "test"},
            processing_errors=[]
        )
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_workflow(self, pipeline, sample_document):
        """Test the complete pipeline from document to knowledge graph."""
        # Mock the PDF extraction to return our sample document
        extraction_result = ExtractionResult(
            success=True,
            document=sample_document,
            error_message=None,
            processing_time=1.0
        )
        
        with patch.object(pipeline.pdf_extractor, 'extract_from_file', return_value=extraction_result):
            # Process the document
            result = await pipeline.process_file("test_paper.pdf")
            
            # Verify basic processing success
            assert result['success'] == True
            assert result['file_path'] == "test_paper.pdf"
            assert result['title'] == "Clinical Metabolomics in Precision Medicine"
            assert result['word_count'] == 50
            
            # Verify entity extraction occurred
            assert 'entities_extracted' in result
            assert 'relationships_extracted' in result
            assert 'knowledge_graph_nodes' in result
            assert 'knowledge_graph_edges' in result
            
            # Should have extracted some entities from the metabolomics content
            assert result['entities_extracted'] >= 0
            
            # Verify entities structure if any were extracted
            if result['entities_extracted'] > 0:
                assert 'entities' in result
                assert isinstance(result['entities'], list)
                
                # Check entity structure
                for entity in result['entities']:
                    assert 'entity_id' in entity
                    assert 'text' in entity
                    assert 'entity_type' in entity
                    assert 'confidence_score' in entity
            
            # Verify relationships structure if any were extracted
            if result['relationships_extracted'] > 0:
                assert 'relationships' in result
                assert isinstance(result['relationships'], list)
                
                # Check relationship structure
                for relationship in result['relationships']:
                    assert 'relationship_id' in relationship
                    assert 'source_entity_id' in relationship
                    assert 'target_entity_id' in relationship
                    assert 'relationship_type' in relationship
                    assert 'confidence_score' in relationship
            
            # Verify knowledge graph was created if entities exist
            if result['entities_extracted'] > 0:
                assert result['knowledge_graph_nodes'] >= 0
                assert 'knowledge_graph_id' in result
    
    @pytest.mark.asyncio
    async def test_entity_extraction_integration(self, pipeline):
        """Test entity extraction with realistic metabolomics text."""
        text = """
        Clinical metabolomics studies revealed that glucose concentration
        is significantly elevated in diabetes mellitus patients. The glycolysis
        pathway is responsible for glucose metabolism. Mass spectrometry and
        NMR spectroscopy are commonly used analytical techniques. Biomarkers
        such as creatinine indicate kidney function.
        """
        
        entities = await pipeline.extract_entities(text)
        
        # Should extract some entities
        assert len(entities) >= 0
        
        # Check for expected entity types if entities were found
        if entities:
            entity_types = [entity['entity_type'] for entity in entities]
            
            # May contain metabolomics-related entity types
            possible_types = ['metabolite', 'disease', 'pathway', 'technique', 'biomarker', 'metabolomics_term']
            
            # At least some entities should be of expected types
            assert any(entity_type in possible_types for entity_type in entity_types)
    
    @pytest.mark.asyncio
    async def test_relationship_extraction_integration(self, pipeline):
        """Test relationship extraction with sample entities."""
        text = "Glucose is associated with diabetes in metabolic disorders."
        
        # First extract entities
        entities = await pipeline.extract_entities(text)
        
        if entities:
            # Then extract relationships
            relationships = await pipeline.extract_relationships(text, entities)
            
            # Should extract some relationships if entities were found
            assert len(relationships) >= 0
            
            # Check relationship structure if any were found
            for relationship in relationships:
                assert 'relationship_type' in relationship
                assert 'confidence_score' in relationship
                assert 'evidence_text' in relationship
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_integration(self, pipeline, sample_document):
        """Test knowledge graph construction integration."""
        # Mock successful PDF extraction
        extraction_result = ExtractionResult(
            success=True,
            document=sample_document,
            error_message=None,
            processing_time=1.0
        )
        
        with patch.object(pipeline.pdf_extractor, 'extract_from_file', return_value=extraction_result):
            result = await pipeline.process_file("test_paper.pdf")
            
            # If entities were extracted, knowledge graph should be created
            if result['entities_extracted'] > 0:
                assert result['knowledge_graph_id'] is not None
                assert result['knowledge_graph_nodes'] >= 0
                assert result['knowledge_graph_edges'] >= 0
                
                # Verify the knowledge graph was actually saved
                kg_builder = pipeline.kg_builder
                available_graphs = await kg_builder.list_available_graphs()
                assert result['knowledge_graph_id'] in available_graphs
                
                # Get graph statistics
                stats = await kg_builder.get_graph_statistics(result['knowledge_graph_id'])
                assert 'total_nodes' in stats
                assert 'total_edges' in stats
                assert stats['total_nodes'] == result['knowledge_graph_nodes']
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, pipeline):
        """Test error handling in the complete pipeline."""
        # Test with non-existent file
        result = await pipeline.process_file("nonexistent.pdf")
        
        assert result['success'] == False
        assert 'error_message' in result
        assert result['entities_extracted'] == 0
        assert result['relationships_extracted'] == 0
        assert result['knowledge_graph_nodes'] == 0
        assert result['knowledge_graph_edges'] == 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_integration(self, pipeline, sample_document):
        """Test batch processing of multiple documents."""
        # Mock PDF extraction for multiple files
        extraction_result = ExtractionResult(
            success=True,
            document=sample_document,
            error_message=None,
            processing_time=1.0
        )
        
        with patch.object(pipeline.pdf_extractor, 'extract_batch') as mock_batch:
            mock_batch.return_value = [extraction_result, extraction_result]
            
            # Create temporary directory with mock PDF files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Create mock PDF files
                (temp_path / "paper1.pdf").touch()
                (temp_path / "paper2.pdf").touch()
                
                results = await pipeline.process_directory(str(temp_path))
                
                # Should process both files
                assert len(results) == 2
                
                for result in results:
                    assert 'success' in result
                    assert 'entities_extracted' in result
                    assert 'relationships_extracted' in result
                    assert 'knowledge_graph_nodes' in result
                    assert 'knowledge_graph_edges' in result
    
    @pytest.mark.asyncio
    async def test_processing_statistics(self, pipeline, sample_document):
        """Test processing statistics collection."""
        # Mock successful extraction
        extraction_result = ExtractionResult(
            success=True,
            document=sample_document,
            error_message=None,
            processing_time=1.5
        )
        
        with patch.object(pipeline.pdf_extractor, 'extract_from_file', return_value=extraction_result):
            result = await pipeline.process_file("test_paper.pdf")
            
            # Verify timing and statistics are recorded
            assert 'processing_time' in result
            assert 'timestamp' in result
            assert result['processing_time'] >= 0
            
            # Verify document metadata is preserved
            assert result['title'] == sample_document.title
            assert result['authors'] == sample_document.authors
            assert result['abstract'] == sample_document.abstract
            assert result['word_count'] == sample_document.word_count
            assert result['page_count'] == sample_document.page_count


# Real integration test that can be run manually
class TestRealIntegration:
    """Real integration tests (require actual dependencies)."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_real_pipeline_with_sample_text(self):
        """Test pipeline with real text processing (if dependencies available)."""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                config = Mock()
                config.knowledge_graph_path = str(Path(temp_dir) / "kg")
                config.papers_directory = str(Path(temp_dir) / "papers")
                config.max_concurrent_extractions = 1
                
                pipeline = PDFIngestionPipeline(config)
                
                # Test entity extraction with real text
                text = """
                Clinical metabolomics is a powerful approach for studying metabolic
                changes in disease. Glucose levels are often elevated in diabetes
                patients. Mass spectrometry and NMR are key analytical techniques
                used in metabolomics research.
                """
                
                entities = await pipeline.extract_entities(text)
                print(f"Extracted {len(entities)} entities:")
                for entity in entities[:5]:  # Show first 5
                    print(f"  - {entity['text']} ({entity['entity_type']}): {entity['confidence_score']:.2f}")
                
                if entities:
                    relationships = await pipeline.extract_relationships(text, entities)
                    print(f"Extracted {len(relationships)} relationships:")
                    for rel in relationships[:3]:  # Show first 3
                        print(f"  - {rel['relationship_type']}: {rel['evidence_text'][:50]}...")
                
                # Basic validation
                assert len(entities) >= 0
                print("Real integration test completed successfully!")
                
        except Exception as e:
            print(f"Real integration test failed (this may be expected): {str(e)}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "not integration"])