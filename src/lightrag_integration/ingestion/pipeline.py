"""
PDF Ingestion Pipeline

This module handles PDF document processing and knowledge graph construction.
Implements the PDF text extraction component for task 3.1.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import asyncio

from ..utils.logging import setup_logger
from .pdf_extractor import PDFExtractor, ExtractionResult, ExtractedDocument
from .knowledge_graph import KnowledgeGraphBuilder


class PDFIngestionPipeline:
    """
    Pipeline for processing PDF documents and extracting knowledge.
    
    This class coordinates the PDF ingestion process, including text extraction,
    entity extraction, and knowledge graph construction.
    """
    
    def __init__(self, config):
        """Initialize the ingestion pipeline."""
        self.config = config
        self.logger = setup_logger("pdf_ingestion")
        
        # Initialize PDF extractor
        self.pdf_extractor = PDFExtractor(config)
        
        # Initialize knowledge graph builder
        self.kg_builder = KnowledgeGraphBuilder(config)
        
        self.logger.info("PDF ingestion pipeline initialized")
    
    async def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
        
        Returns:
            List of processing results for each file
        """
        self.logger.info(f"Starting directory processing: {directory_path}")
        
        try:
            # Find all PDF files in directory
            dir_path = Path(directory_path)
            if not dir_path.exists():
                self.logger.error(f"Directory does not exist: {directory_path}")
                return []
            
            pdf_files = list(dir_path.glob("*.pdf"))
            if not pdf_files:
                self.logger.warning(f"No PDF files found in directory: {directory_path}")
                return []
            
            self.logger.info(f"Found {len(pdf_files)} PDF files to process")
            
            # Process files in batches
            pdf_paths = [str(pdf_file) for pdf_file in pdf_files]
            extraction_results = await self.pdf_extractor.extract_batch(
                pdf_paths, 
                max_concurrent=getattr(self.config, 'max_concurrent_extractions', 3)
            )
            
            # Convert extraction results to processing results
            processing_results = []
            for extraction_result in extraction_results:
                processing_result = await self._convert_extraction_to_processing_result(extraction_result)
                processing_results.append(processing_result)
            
            successful = sum(1 for result in processing_results if result.get('success', False))
            self.logger.info(f"Directory processing completed: {successful}/{len(processing_results)} successful")
            
            return processing_results
            
        except Exception as e:
            self.logger.error(f"Error processing directory {directory_path}: {str(e)}", exc_info=True)
            return []
    
    async def process_file(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a single PDF file.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Processing result dictionary
        """
        self.logger.info(f"Starting file processing: {pdf_path}")
        
        try:
            # Extract text from PDF
            extraction_result = await self.pdf_extractor.extract_from_file(pdf_path)
            
            # Convert to processing result
            processing_result = await self._convert_extraction_to_processing_result(extraction_result)
            
            if processing_result.get('success', False):
                self.logger.info(f"Successfully processed file: {pdf_path}")
            else:
                self.logger.error(f"Failed to process file: {pdf_path} - {processing_result.get('error_message', 'Unknown error')}")
            
            return processing_result
            
        except Exception as e:
            error_msg = f"Unexpected error processing file {pdf_path}: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'file_path': pdf_path,
                'error_message': error_msg,
                'processing_time': 0.0,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _convert_extraction_to_processing_result(self, extraction_result: ExtractionResult) -> Dict[str, Any]:
        """
        Convert ExtractionResult to processing result dictionary.
        
        Args:
            extraction_result: Result from PDF extraction
        
        Returns:
            Processing result dictionary
        """
        base_result = {
            'success': extraction_result.success,
            'processing_time': extraction_result.processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        if extraction_result.success and extraction_result.document:
            doc = extraction_result.document
            
            # Extract entities and relationships from the document content
            entities = []
            relationships = []
            kg_result = None
            
            try:
                if doc.content and doc.content.strip():
                    self.logger.info(f"Extracting entities and relationships from {doc.file_path}")
                    
                    # Extract entities from the full content
                    entities = await self.extract_entities(doc.content)
                    
                    # Extract relationships between entities
                    if entities:
                        relationships = await self.extract_relationships(doc.content, entities)
                    
                    self.logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
                    
                    # Build knowledge graph if we have entities
                    if entities:
                        # Convert entity dictionaries back to Entity objects for KG construction
                        from .entity_extractor import Entity, Relationship as RelationshipObj
                        
                        entity_objects = []
                        for entity_dict in entities:
                            entity = Entity(
                                entity_id=entity_dict['entity_id'],
                                text=entity_dict['text'],
                                entity_type=entity_dict['entity_type'],
                                confidence_score=entity_dict['confidence_score'],
                                start_pos=entity_dict['start_pos'],
                                end_pos=entity_dict['end_pos'],
                                context=entity_dict['context'],
                                metadata=entity_dict['metadata']
                            )
                            entity_objects.append(entity)
                        
                        relationship_objects = []
                        for rel_dict in relationships:
                            relationship = RelationshipObj(
                                relationship_id=rel_dict['relationship_id'],
                                source_entity_id=rel_dict['source_entity_id'],
                                target_entity_id=rel_dict['target_entity_id'],
                                relationship_type=rel_dict['relationship_type'],
                                confidence_score=rel_dict['confidence_score'],
                                evidence_text=rel_dict['evidence_text'],
                                context=rel_dict['context'],
                                metadata=rel_dict['metadata']
                            )
                            relationship_objects.append(relationship)
                        
                        # Construct knowledge graph
                        document_id = Path(doc.file_path).stem
                        kg_result = await self.kg_builder.construct_graph_from_entities_and_relationships(
                            entity_objects, relationship_objects, document_id
                        )
                        
                        if kg_result.success:
                            self.logger.info(f"Created knowledge graph with {kg_result.nodes_created} nodes and {kg_result.edges_created} edges")
                        else:
                            self.logger.error(f"Failed to create knowledge graph: {kg_result.error_message}")
                
            except Exception as e:
                self.logger.error(f"Error extracting entities/relationships from {doc.file_path}: {str(e)}")
            
            base_result.update({
                'file_path': doc.file_path,
                'title': doc.title,
                'authors': doc.authors,
                'abstract': doc.abstract,
                'word_count': doc.word_count,
                'page_count': doc.page_count,
                'extraction_timestamp': doc.extraction_timestamp.isoformat(),
                'processing_errors': doc.processing_errors,
                'metadata': doc.metadata,
                'entities_extracted': len(entities),
                'relationships_extracted': len(relationships),
                'knowledge_graph_nodes': kg_result.nodes_created if kg_result and kg_result.success else 0,
                'knowledge_graph_edges': kg_result.edges_created if kg_result and kg_result.success else 0,
                'knowledge_graph_id': kg_result.graph.graph_id if kg_result and kg_result.success else None,
                'entities': entities,
                'relationships': relationships
            })
        else:
            base_result.update({
                'file_path': getattr(extraction_result, 'file_path', 'unknown'),
                'error_message': extraction_result.error_message
            })
        
        return base_result
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract biomedical entities from text.
        
        Args:
            text: Input text to process
        
        Returns:
            List of extracted entities
        """
        if not hasattr(self, 'entity_extractor'):
            from .entity_extractor import BiomedicaEntityExtractor
            self.entity_extractor = BiomedicaEntityExtractor(self.config)
        
        try:
            entities = await self.entity_extractor.extract_entities(text)
            
            # Convert Entity objects to dictionaries
            entity_dicts = []
            for entity in entities:
                entity_dict = {
                    'entity_id': entity.entity_id,
                    'text': entity.text,
                    'entity_type': entity.entity_type,
                    'confidence_score': entity.confidence_score,
                    'start_pos': entity.start_pos,
                    'end_pos': entity.end_pos,
                    'context': entity.context,
                    'metadata': entity.metadata
                }
                entity_dicts.append(entity_dict)
            
            self.logger.info(f"Extracted {len(entity_dicts)} entities from text")
            return entity_dicts
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return []
    
    async def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities.
        
        Args:
            text: Input text
            entities: List of extracted entities
        
        Returns:
            List of extracted relationships
        """
        if not hasattr(self, 'entity_extractor'):
            from .entity_extractor import BiomedicaEntityExtractor
            self.entity_extractor = BiomedicaEntityExtractor(self.config)
        
        try:
            # Convert entity dictionaries back to Entity objects
            from .entity_extractor import Entity
            entity_objects = []
            for entity_dict in entities:
                entity = Entity(
                    entity_id=entity_dict['entity_id'],
                    text=entity_dict['text'],
                    entity_type=entity_dict['entity_type'],
                    confidence_score=entity_dict['confidence_score'],
                    start_pos=entity_dict['start_pos'],
                    end_pos=entity_dict['end_pos'],
                    context=entity_dict['context'],
                    metadata=entity_dict['metadata']
                )
                entity_objects.append(entity)
            
            relationships = await self.entity_extractor.extract_relationships(text, entity_objects)
            
            # Convert Relationship objects to dictionaries
            relationship_dicts = []
            for relationship in relationships:
                relationship_dict = {
                    'relationship_id': relationship.relationship_id,
                    'source_entity_id': relationship.source_entity_id,
                    'target_entity_id': relationship.target_entity_id,
                    'relationship_type': relationship.relationship_type,
                    'confidence_score': relationship.confidence_score,
                    'evidence_text': relationship.evidence_text,
                    'context': relationship.context,
                    'metadata': relationship.metadata
                }
                relationship_dicts.append(relationship_dict)
            
            self.logger.info(f"Extracted {len(relationship_dicts)} relationships from text")
            return relationship_dicts
            
        except Exception as e:
            self.logger.error(f"Error extracting relationships: {str(e)}", exc_info=True)
            return []
    
    def validate_pdf_file(self, pdf_path: str) -> tuple[bool, Optional[str]]:
        """
        Validate if a PDF file can be processed.
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        return self.pdf_extractor.validate_pdf(pdf_path)
    
    async def get_processing_stats(self, directory_path: str) -> Dict[str, Any]:
        """
        Get statistics about PDFs in a directory without processing them.
        
        Args:
            directory_path: Path to directory containing PDF files
        
        Returns:
            Dictionary with statistics
        """
        try:
            dir_path = Path(directory_path)
            if not dir_path.exists():
                return {'error': 'Directory does not exist'}
            
            pdf_files = list(dir_path.glob("*.pdf"))
            
            stats = {
                'total_files': len(pdf_files),
                'valid_files': 0,
                'invalid_files': 0,
                'total_size_mb': 0.0,
                'validation_errors': []
            }
            
            for pdf_file in pdf_files:
                file_size_mb = pdf_file.stat().st_size / (1024 * 1024)
                stats['total_size_mb'] += file_size_mb
                
                is_valid, error = self.validate_pdf_file(str(pdf_file))
                if is_valid:
                    stats['valid_files'] += 1
                else:
                    stats['invalid_files'] += 1
                    stats['validation_errors'].append({
                        'file': str(pdf_file),
                        'error': error
                    })
            
            stats['total_size_mb'] = round(stats['total_size_mb'], 2)
            
            return stats
            
        except Exception as e:
            return {'error': f'Error getting stats: {str(e)}'}
    
    async def cleanup(self):
        """Clean up resources."""
        self.logger.info("PDF ingestion pipeline cleanup completed")