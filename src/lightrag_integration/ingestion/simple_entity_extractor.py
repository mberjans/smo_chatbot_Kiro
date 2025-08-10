#!/usr/bin/env python3
"""
Simple Entity and Relationship Extractor

A simplified version that doesn't require spaCy for basic entity extraction
focused on clinical metabolomics concepts.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from ..utils.logging import setup_logger


@dataclass
class Entity:
    """Represents an extracted biomedical entity."""
    entity_id: str
    text: str
    entity_type: str
    confidence_score: float
    start_pos: int
    end_pos: int
    context: str
    metadata: Dict[str, Any]


@dataclass
class Relationship:
    """Represents a relationship between two entities."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence_score: float
    evidence_text: str
    context: str
    metadata: Dict[str, Any]


@dataclass
class ExtractionResult:
    """Result of entity and relationship extraction."""
    entities: List[Entity]
    relationships: List[Relationship]
    processing_time: float
    metadata: Dict[str, Any]


class SimpleEntityExtractor:
    """
    Simple entity and relationship extractor for clinical metabolomics.
    
    This class uses regex patterns and keyword matching to extract entities
    and relationships relevant to clinical metabolomics research.
    """
    
    def __init__(self, config=None):
        """Initialize the simple entity extractor."""
        self.config = config
        self.logger = setup_logger("simple_entity_extractor")
        
        # Define metabolomics vocabulary and patterns
        self.metabolomics_terms = self._create_metabolomics_vocabulary()
        self.entity_patterns = self._create_entity_patterns()
        self.relationship_patterns = self._create_relationship_patterns()
        
        self.logger.info("SimpleEntityExtractor initialized")
    
    def _create_metabolomics_vocabulary(self) -> Dict[str, Set[str]]:
        """Create vocabulary for different entity types."""
        vocabulary = {
            "metabolite": {
                "glucose", "lactate", "pyruvate", "acetate", "citrate", "succinate",
                "alanine", "glycine", "serine", "threonine", "valine", "leucine",
                "isoleucine", "phenylalanine", "tyrosine", "tryptophan", "histidine",
                "lysine", "arginine", "creatinine", "urea", "cholesterol", "triglyceride",
                "fatty acid", "amino acid", "nucleotide", "ATP", "ADP", "AMP",
                "NADH", "NAD+", "CoA", "acetyl-CoA"
            },
            
            "disease": {
                "diabetes", "obesity", "hypertension", "cancer", "cardiovascular disease",
                "metabolic syndrome", "insulin resistance", "type 2 diabetes",
                "coronary heart disease", "stroke", "atherosclerosis", "inflammation"
            },
            
            "pathway": {
                "glycolysis", "gluconeogenesis", "citric acid cycle", "TCA cycle",
                "fatty acid oxidation", "fatty acid synthesis", "amino acid metabolism",
                "purine metabolism", "pyrimidine metabolism", "pentose phosphate pathway"
            },
            
            "biomarker": {
                "biomarker", "marker", "indicator", "predictor", "signature",
                "metabolic signature", "metabolic profile", "metabolic fingerprint"
            },
            
            "technique": {
                "NMR", "mass spectrometry", "LC-MS", "GC-MS", "HPLC", "UPLC",
                "spectroscopy", "chromatography", "metabolomics", "proteomics",
                "genomics", "nuclear magnetic resonance"
            },
            
            "sample_type": {
                "plasma", "serum", "urine", "tissue", "saliva", "breath",
                "blood", "cerebrospinal fluid", "CSF", "feces", "stool"
            },
            
            "clinical_condition": {
                "diagnosis", "prognosis", "treatment", "therapy", "intervention",
                "clinical trial", "patient", "control", "case", "cohort"
            }
        }
        
        return vocabulary
    
    def _create_entity_patterns(self) -> Dict[str, List[str]]:
        """Create regex patterns for entity recognition."""
        patterns = {
            "measurement": [
                r'\b\d+\.?\d*\s*(mg/dL|mmol/L|μM|mM|nM|pM|g/L|mg/L|μg/L)\b',
                r'\b\d+\.?\d*\s*(fold|times|%|percent)\b',
                r'\bp\s*[<>=]\s*0\.\d+\b',  # p-values
                r'\bR²?\s*[=]\s*0\.\d+\b',  # correlation coefficients
            ],
            
            "statistical": [
                r'\bp-value\b|\bp\s*[<>=]\s*\d+\.?\d*\b',
                r'\bconfidence interval\b|\bCI\b',
                r'\bstandard deviation\b|\bSD\b|\bstd\b',
                r'\bmean\b|\baverage\b|\bmedian\b',
                r'\bcorrelation\b|\bassociation\b|\bregression\b'
            ],
            
            "sample_preparation": [
                r'\bsample preparation\b|\bsample processing\b',
                r'\bextraction\b|\bpurification\b|\bcentrifugation\b',
                r'\bfiltration\b|\bdilution\b|\bdeproteinization\b',
                r'\bstorage\b|\bfreeze\b|\bthaw\b|\bstability\b'
            ]
        }
        
        return patterns
    
    def _create_relationship_patterns(self) -> List[Dict[str, Any]]:
        """Create patterns for relationship extraction."""
        patterns = [
            {
                "pattern": r'(\w+(?:\s+\w+)*)\s+(?:is|are|was|were)\s+(?:significantly\s+)?(?:associated|correlated|linked)\s+with\s+(\w+(?:\s+\w+)*)',
                "type": "associated_with",
                "confidence": 0.8
            },
            {
                "pattern": r'(\w+(?:\s+\w+)*)\s+(?:causes?|leads?\s+to|results?\s+in)\s+(\w+(?:\s+\w+)*)',
                "type": "causes",
                "confidence": 0.85
            },
            {
                "pattern": r'(\w+(?:\s+\w+)*)\s+(?:regulates?|controls?|modulates?)\s+(\w+(?:\s+\w+)*)',
                "type": "regulates",
                "confidence": 0.8
            },
            {
                "pattern": r'(\w+(?:\s+\w+)*)\s+(?:is|serves as|acts as)\s+(?:a|an)?\s*biomarker\s+(?:for|of)\s+(\w+(?:\s+\w+)*)',
                "type": "biomarker_for",
                "confidence": 0.9
            },
            {
                "pattern": r'(\w+(?:\s+\w+)*)\s+(?:pathway|metabolism)\s+(?:produces?|generates?|yields?)\s+(\w+(?:\s+\w+)*)',
                "type": "produces",
                "confidence": 0.85
            },
            {
                "pattern": r'(\w+(?:\s+\w+)*)\s+(?:levels?|concentrations?)\s+(?:were?|are)\s+(?:increased|elevated|higher)\s+in\s+(\w+(?:\s+\w+)*)',
                "type": "elevated_in",
                "confidence": 0.8
            },
            {
                "pattern": r'(\w+(?:\s+\w+)*)\s+(?:levels?|concentrations?)\s+(?:were?|are)\s+(?:decreased|reduced|lower)\s+in\s+(\w+(?:\s+\w+)*)',
                "type": "reduced_in",
                "confidence": 0.8
            }
        ]
        
        return patterns
    
    async def extract_entities(self, text: str, context: Optional[str] = None) -> List[Entity]:
        """Extract biomedical entities from text using simple patterns."""
        if not text or not text.strip():
            return []
        
        entities = []
        text_lower = text.lower()
        
        # Extract vocabulary-based entities
        for entity_type, terms in self.metabolomics_terms.items():
            for term in terms:
                term_lower = term.lower()
                start = 0
                
                while True:
                    pos = text_lower.find(term_lower, start)
                    if pos == -1:
                        break
                    
                    # Check if it's a whole word match
                    if (pos == 0 or not text[pos-1].isalnum()) and \
                       (pos + len(term) >= len(text) or not text[pos + len(term)].isalnum()):
                        
                        entity = Entity(
                            entity_id=f"{entity_type}_{len(entities)}_{pos}",
                            text=text[pos:pos + len(term)],
                            entity_type=entity_type,
                            confidence_score=0.8,
                            start_pos=pos,
                            end_pos=pos + len(term),
                            context=context or "",
                            metadata={
                                "extraction_method": "vocabulary_matching",
                                "term_category": entity_type
                            }
                        )
                        entities.append(entity)
                    
                    start = pos + 1
        
        # Extract pattern-based entities
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = Entity(
                        entity_id=f"{entity_type}_{len(entities)}_{match.start()}",
                        text=match.group(0),
                        entity_type=entity_type,
                        confidence_score=0.7,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        context=context or "",
                        metadata={
                            "extraction_method": "pattern_matching",
                            "pattern": pattern
                        }
                    )
                    entities.append(entity)
        
        # Remove overlapping entities (keep the one with higher confidence)
        entities = self._remove_overlapping_entities(entities)
        
        self.logger.info(f"Extracted {len(entities)} entities from text")
        return entities
    
    def _remove_overlapping_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove overlapping entities, keeping the one with higher confidence."""
        if not entities:
            return entities
        
        # Sort by start position
        entities.sort(key=lambda e: e.start_pos)
        
        filtered = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for i, existing in enumerate(filtered):
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Entities overlap
                    if entity.confidence_score > existing.confidence_score:
                        # Replace existing with current
                        filtered[i] = entity
                    overlaps = True
                    break
            
            if not overlaps:
                filtered.append(entity)
        
        return filtered
    
    async def extract_relationships(self, text: str, entities: List[Entity], 
                                  context: Optional[str] = None) -> List[Relationship]:
        """Extract relationships between entities using patterns."""
        if not entities or len(entities) < 2:
            return []
        
        relationships = []
        
        # Extract pattern-based relationships
        for pattern_info in self.relationship_patterns:
            pattern = pattern_info["pattern"]
            rel_type = pattern_info["type"]
            confidence = pattern_info["confidence"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source_text = match.group(1).strip()
                target_text = match.group(2).strip()
                
                # Find corresponding entities
                source_entity = self._find_best_matching_entity(source_text, entities)
                target_entity = self._find_best_matching_entity(target_text, entities)
                
                if source_entity and target_entity and source_entity != target_entity:
                    relationship = Relationship(
                        relationship_id=f"{rel_type}_{len(relationships)}",
                        source_entity_id=source_entity.entity_id,
                        target_entity_id=target_entity.entity_id,
                        relationship_type=rel_type,
                        confidence_score=confidence,
                        evidence_text=match.group(0),
                        context=context or "",
                        metadata={
                            "extraction_method": "pattern_matching",
                            "pattern": pattern,
                            "match_start": match.start(),
                            "match_end": match.end()
                        }
                    )
                    relationships.append(relationship)
        
        # Extract co-occurrence relationships for entities in close proximity
        relationships.extend(self._extract_cooccurrence_relationships(text, entities, context))
        
        self.logger.info(f"Extracted {len(relationships)} relationships")
        return relationships
    
    def _find_best_matching_entity(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """Find the best matching entity for the given text."""
        text_lower = text.lower().strip()
        
        # First, try exact match
        for entity in entities:
            if entity.text.lower().strip() == text_lower:
                return entity
        
        # Then, try partial match (entity text contains the search text)
        for entity in entities:
            if text_lower in entity.text.lower():
                return entity
        
        # Finally, try reverse partial match (search text contains entity text)
        for entity in entities:
            if entity.text.lower().strip() in text_lower:
                return entity
        
        return None
    
    def _extract_cooccurrence_relationships(self, text: str, entities: List[Entity], 
                                          context: Optional[str]) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence."""
        relationships = []
        window_size = 150  # Characters
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check if entities are within the co-occurrence window
                distance = abs(entity1.start_pos - entity2.start_pos)
                if distance <= window_size:
                    # Extract context around both entities
                    start_pos = min(entity1.start_pos, entity2.start_pos) - 30
                    end_pos = max(entity1.end_pos, entity2.end_pos) + 30
                    evidence_text = text[max(0, start_pos):min(len(text), end_pos)]
                    
                    # Determine relationship strength based on distance
                    confidence = max(0.3, 0.7 - (distance / window_size) * 0.4)
                    
                    relationship = Relationship(
                        relationship_id=f"cooccur_{i}_{j}",
                        source_entity_id=entity1.entity_id,
                        target_entity_id=entity2.entity_id,
                        relationship_type="co_occurs_with",
                        confidence_score=confidence,
                        evidence_text=evidence_text.strip(),
                        context=context or "",
                        metadata={
                            "extraction_method": "co_occurrence",
                            "distance": distance,
                            "window_size": window_size
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def extract_entities_and_relationships(self, text: str, 
                                               context: Optional[str] = None) -> ExtractionResult:
        """Extract both entities and relationships from text."""
        start_time = datetime.now()
        
        try:
            # Extract entities
            entities = await self.extract_entities(text, context)
            
            # Extract relationships
            relationships = await self.extract_relationships(text, entities, context)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ExtractionResult(
                entities=entities,
                relationships=relationships,
                processing_time=processing_time,
                metadata={
                    "text_length": len(text),
                    "entity_count": len(entities),
                    "relationship_count": len(relationships),
                    "extraction_timestamp": datetime.now().isoformat(),
                    "extraction_method": "simple_pattern_matching"
                }
            )
            
            self.logger.info(f"Extraction completed: {len(entities)} entities, {len(relationships)} relationships")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error in extraction: {str(e)}", exc_info=True)
            
            return ExtractionResult(
                entities=[],
                relationships=[],
                processing_time=processing_time,
                metadata={
                    "error": str(e),
                    "extraction_timestamp": datetime.now().isoformat()
                }
            )