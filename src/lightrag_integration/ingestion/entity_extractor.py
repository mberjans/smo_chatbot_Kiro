"""
Entity and Relationship Extraction Component

This module handles biomedical entity extraction and relationship detection
for clinical metabolomics concepts. Implements requirements 1.6 and 8.5.
"""

import asyncio
import logging
import re
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

try:
    import spacy
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

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


class BiomedicaEntityExtractor:
    """
    Biomedical entity and relationship extractor for clinical metabolomics.
    
    This class uses spaCy NLP models and custom patterns to extract entities
    and relationships relevant to clinical metabolomics research.
    """
    
    def __init__(self, config=None):
        """
        Initialize the entity extractor.
        
        Args:
            config: Optional configuration object
        """
        self.config = config
        self.logger = setup_logger("entity_extractor")
        
        # Initialize spaCy model
        self.nlp = None
        self._initialize_nlp()
        
        # Define biomedical entity patterns
        self.entity_patterns = self._create_entity_patterns()
        
        # Define relationship patterns
        self.relationship_patterns = self._create_relationship_patterns()
        
        # Clinical metabolomics vocabulary
        self.metabolomics_terms = self._load_metabolomics_vocabulary()
        
        self.logger.info("BiomedicaEntityExtractor initialized")
    
    def _initialize_nlp(self):
        """Initialize spaCy NLP model."""
        if not SPACY_AVAILABLE:
            self.logger.error("spaCy not available. Please install with: pip install spacy")
            return
        
        try:
            # Try to load the English model
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            self.logger.error("spaCy model 'en_core_web_sm' not found. Please install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def _create_entity_patterns(self) -> Dict[str, List[Dict]]:
        """Create patterns for biomedical entity recognition."""
        patterns = {
            "metabolite": [
                {"LOWER": {"IN": ["glucose", "lactate", "pyruvate", "acetate", "citrate"]}},
                {"LOWER": {"IN": ["alanine", "glycine", "serine", "threonine", "valine"]}},
                {"LOWER": {"IN": ["leucine", "isoleucine", "phenylalanine", "tyrosine"]}},
                {"LOWER": {"IN": ["tryptophan", "histidine", "lysine", "arginine"]}},
                {"LOWER": {"IN": ["creatinine", "urea", "cholesterol", "triglyceride"]}},
                {"TEXT": {"REGEX": r".*-CoA$"}},  # Coenzyme A compounds
                {"TEXT": {"REGEX": r".*ATP$|.*ADP$|.*AMP$"}},  # Nucleotides
            ],
            
            "disease": [
                {"LOWER": {"IN": ["diabetes", "obesity", "hypertension", "cancer"]}},
                {"LOWER": {"IN": ["cardiovascular", "metabolic", "syndrome"]}},
                {"LOWER": "disease"},
                {"LOWER": "disorder"},
                {"LOWER": "condition"},
            ],
            
            "pathway": [
                {"LOWER": {"IN": ["glycolysis", "gluconeogenesis", "citric", "acid", "cycle"]}},
                {"LOWER": {"IN": ["fatty", "acid", "oxidation", "synthesis"]}},
                {"LOWER": {"IN": ["amino", "acid", "metabolism"]}},
                {"LOWER": "pathway"},
                {"LOWER": "metabolism"},
            ],
            
            "biomarker": [
                {"LOWER": "biomarker"},
                {"LOWER": "marker"},
                {"LOWER": {"IN": ["indicator", "predictor", "signature"]}},
            ],
            
            "technique": [
                {"UPPER": {"IN": ["NMR", "MS", "LC-MS", "GC-MS", "HPLC"]}},
                {"LOWER": {"IN": ["spectroscopy", "chromatography", "spectrometry"]}},
                {"LOWER": {"IN": ["metabolomics", "proteomics", "genomics"]}},
            ]
        }
        
        return patterns
    
    def _create_relationship_patterns(self) -> List[Dict]:
        """Create patterns for relationship extraction."""
        patterns = [
            # Metabolite-disease relationships
            {
                "pattern": r"(\w+)\s+(?:is|are)\s+(?:associated|linked|correlated)\s+with\s+(\w+)",
                "type": "associated_with",
                "confidence": 0.8
            },
            
            # Pathway-metabolite relationships
            {
                "pattern": r"(\w+)\s+(?:pathway|metabolism)\s+(?:produces|generates|yields)\s+(\w+)",
                "type": "produces",
                "confidence": 0.9
            },
            
            # Biomarker relationships
            {
                "pattern": r"(\w+)\s+(?:is|serves as)\s+(?:a|an)?\s*biomarker\s+(?:for|of)\s+(\w+)",
                "type": "biomarker_for",
                "confidence": 0.85
            },
            
            # Regulation relationships
            {
                "pattern": r"(\w+)\s+(?:regulates|controls|modulates)\s+(\w+)",
                "type": "regulates",
                "confidence": 0.75
            },
            
            # Causation relationships
            {
                "pattern": r"(\w+)\s+(?:causes|leads to|results in)\s+(\w+)",
                "type": "causes",
                "confidence": 0.8
            }
        ]
        
        return patterns
    
    def _load_metabolomics_vocabulary(self) -> Set[str]:
        """Load clinical metabolomics vocabulary."""
        vocabulary = {
            # Core metabolomics terms
            "metabolomics", "metabolome", "metabolite", "metabolism",
            "biomarker", "pathway", "flux", "profiling",
            
            # Clinical terms
            "clinical", "diagnostic", "therapeutic", "prognostic",
            "personalized", "precision", "medicine",
            
            # Analytical techniques
            "nmr", "mass spectrometry", "chromatography", "spectroscopy",
            "lc-ms", "gc-ms", "hplc", "uplc",
            
            # Sample types
            "plasma", "serum", "urine", "tissue", "saliva", "breath",
            
            # Statistical terms
            "correlation", "association", "significance", "p-value",
            "fold-change", "regulation", "expression",
            
            # Common metabolites
            "glucose", "lactate", "pyruvate", "citrate", "acetate",
            "creatinine", "urea", "cholesterol", "amino acid",
            
            # Diseases
            "diabetes", "obesity", "cancer", "cardiovascular",
            "metabolic syndrome", "hypertension"
        }
        
        return vocabulary
    
    async def extract_entities(self, text: str, context: Optional[str] = None) -> List[Entity]:
        """
        Extract biomedical entities from text.
        
        Args:
            text: Input text to process
            context: Optional context information
        
        Returns:
            List of extracted entities
        """
        if not self.nlp:
            self.logger.error("spaCy model not available")
            return []
        
        if not text or not text.strip():
            return []
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            entities = []
            
            # Extract named entities using spaCy's built-in NER
            spacy_entities = await self._extract_spacy_entities(doc, context)
            entities.extend(spacy_entities)
            
            # Extract domain-specific entities using patterns
            pattern_entities = await self._extract_pattern_entities(doc, context)
            entities.extend(pattern_entities)
            
            # Extract metabolomics-specific terms
            vocab_entities = await self._extract_vocabulary_entities(doc, context)
            entities.extend(vocab_entities)
            
            # Remove duplicates and merge overlapping entities
            entities = self._deduplicate_entities(entities)
            
            self.logger.info(f"Extracted {len(entities)} entities from text")
            return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}", exc_info=True)
            return []
    
    async def _extract_spacy_entities(self, doc: Any, context: Optional[str]) -> List[Entity]:
        """Extract entities using spaCy's built-in NER."""
        entities = []
        
        for ent in doc.ents:
            # Map spaCy entity types to our biomedical types
            entity_type = self._map_spacy_entity_type(ent.label_)
            if entity_type:
                entity = Entity(
                    entity_id=f"spacy_{len(entities)}_{ent.start}_{ent.end}",
                    text=ent.text,
                    entity_type=entity_type,
                    confidence_score=0.7,  # Default confidence for spaCy entities
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    context=context or "",
                    metadata={
                        "spacy_label": ent.label_,
                        "spacy_description": spacy.explain(ent.label_),
                        "extraction_method": "spacy_ner"
                    }
                )
                entities.append(entity)
        
        return entities
    
    def _map_spacy_entity_type(self, spacy_label: str) -> Optional[str]:
        """Map spaCy entity labels to biomedical entity types."""
        mapping = {
            "PERSON": None,  # Usually not relevant for metabolomics
            "ORG": "organization",
            "GPE": None,  # Geopolitical entities not relevant
            "PRODUCT": "technique",  # Could be analytical instruments
            "EVENT": None,
            "WORK_OF_ART": None,
            "LAW": None,
            "LANGUAGE": None,
            "DATE": "temporal",
            "TIME": "temporal",
            "PERCENT": "measurement",
            "MONEY": None,
            "QUANTITY": "measurement",
            "ORDINAL": "measurement",
            "CARDINAL": "measurement"
        }
        
        return mapping.get(spacy_label)
    
    async def _extract_pattern_entities(self, doc: Any, context: Optional[str]) -> List[Entity]:
        """Extract entities using custom patterns."""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                # Simple pattern matching for now
                # In a more sophisticated implementation, we'd use spaCy's Matcher
                if "IN" in pattern.get("LOWER", {}):
                    terms = pattern["LOWER"]["IN"]
                    for token in doc:
                        if token.text.lower() in terms:
                            entity = Entity(
                                entity_id=f"pattern_{entity_type}_{len(entities)}_{token.i}",
                                text=token.text,
                                entity_type=entity_type,
                                confidence_score=0.8,
                                start_pos=token.idx,
                                end_pos=token.idx + len(token.text),
                                context=context or "",
                                metadata={
                                    "extraction_method": "pattern_matching",
                                    "pattern_type": entity_type
                                }
                            )
                            entities.append(entity)
        
        return entities
    
    async def _extract_vocabulary_entities(self, doc: Any, context: Optional[str]) -> List[Entity]:
        """Extract entities based on metabolomics vocabulary."""
        entities = []
        
        # Look for vocabulary terms in the text
        text_lower = doc.text.lower()
        
        for term in self.metabolomics_terms:
            if term in text_lower:
                # Find all occurrences of the term
                start = 0
                while True:
                    pos = text_lower.find(term, start)
                    if pos == -1:
                        break
                    
                    entity = Entity(
                        entity_id=f"vocab_{len(entities)}_{pos}",
                        text=term,
                        entity_type="metabolomics_term",
                        confidence_score=0.6,
                        start_pos=pos,
                        end_pos=pos + len(term),
                        context=context or "",
                        metadata={
                            "extraction_method": "vocabulary_matching",
                            "term_category": "metabolomics"
                        }
                    )
                    entities.append(entity)
                    start = pos + 1
        
        return entities
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate and overlapping entities."""
        if not entities:
            return entities
        
        # Sort entities by start position
        entities.sort(key=lambda e: e.start_pos)
        
        deduplicated = []
        for entity in entities:
            # Check for overlap with existing entities
            overlaps = False
            for existing in deduplicated:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Entities overlap, keep the one with higher confidence
                    if entity.confidence_score > existing.confidence_score:
                        deduplicated.remove(existing)
                        deduplicated.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                deduplicated.append(entity)
        
        return deduplicated
    
    async def extract_relationships(self, text: str, entities: List[Entity], 
                                  context: Optional[str] = None) -> List[Relationship]:
        """
        Extract relationships between entities.
        
        Args:
            text: Input text
            entities: List of extracted entities
            context: Optional context information
        
        Returns:
            List of extracted relationships
        """
        if not entities or len(entities) < 2:
            return []
        
        try:
            relationships = []
            
            # Extract relationships using patterns
            pattern_relationships = await self._extract_pattern_relationships(text, entities, context)
            relationships.extend(pattern_relationships)
            
            # Extract co-occurrence relationships
            cooccurrence_relationships = await self._extract_cooccurrence_relationships(text, entities, context)
            relationships.extend(cooccurrence_relationships)
            
            # Extract syntactic relationships
            if self.nlp:
                syntactic_relationships = await self._extract_syntactic_relationships(text, entities, context)
                relationships.extend(syntactic_relationships)
            
            self.logger.info(f"Extracted {len(relationships)} relationships")
            return relationships
            
        except Exception as e:
            self.logger.error(f"Error extracting relationships: {str(e)}", exc_info=True)
            return []
    
    async def _extract_pattern_relationships(self, text: str, entities: List[Entity], 
                                           context: Optional[str]) -> List[Relationship]:
        """Extract relationships using predefined patterns."""
        relationships = []
        
        for pattern_info in self.relationship_patterns:
            pattern = pattern_info["pattern"]
            rel_type = pattern_info["type"]
            confidence = pattern_info["confidence"]
            
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source_text = match.group(1)
                target_text = match.group(2)
                
                # Find corresponding entities
                source_entity = self._find_entity_by_text(source_text, entities)
                target_entity = self._find_entity_by_text(target_text, entities)
                
                if source_entity and target_entity:
                    relationship = Relationship(
                        relationship_id=f"pattern_{rel_type}_{len(relationships)}",
                        source_entity_id=source_entity.entity_id,
                        target_entity_id=target_entity.entity_id,
                        relationship_type=rel_type,
                        confidence_score=confidence,
                        evidence_text=match.group(0),
                        context=context or "",
                        metadata={
                            "extraction_method": "pattern_matching",
                            "pattern": pattern
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def _extract_cooccurrence_relationships(self, text: str, entities: List[Entity], 
                                                context: Optional[str]) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence."""
        relationships = []
        
        # Define co-occurrence window (number of characters)
        window_size = 100
        
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check if entities are within the co-occurrence window
                distance = abs(entity1.start_pos - entity2.start_pos)
                if distance <= window_size:
                    # Extract context around both entities
                    start_pos = min(entity1.start_pos, entity2.start_pos) - 20
                    end_pos = max(entity1.end_pos, entity2.end_pos) + 20
                    evidence_text = text[max(0, start_pos):min(len(text), end_pos)]
                    
                    relationship = Relationship(
                        relationship_id=f"cooccur_{i}_{j}",
                        source_entity_id=entity1.entity_id,
                        target_entity_id=entity2.entity_id,
                        relationship_type="co_occurs_with",
                        confidence_score=0.5,  # Lower confidence for co-occurrence
                        evidence_text=evidence_text,
                        context=context or "",
                        metadata={
                            "extraction_method": "co_occurrence",
                            "distance": distance
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    async def _extract_syntactic_relationships(self, text: str, entities: List[Entity], 
                                             context: Optional[str]) -> List[Relationship]:
        """Extract relationships using syntactic analysis."""
        if not self.nlp:
            return []
        
        relationships = []
        doc = self.nlp(text)
        
        # Look for dependency relationships between entity tokens
        entity_tokens = {}
        for entity in entities:
            # Find tokens that correspond to this entity
            for token in doc:
                if (token.idx >= entity.start_pos and 
                    token.idx + len(token.text) <= entity.end_pos):
                    entity_tokens[token.i] = entity
        
        for token in doc:
            if token.i in entity_tokens:
                source_entity = entity_tokens[token.i]
                
                # Look at dependency relationships
                for child in token.children:
                    if child.i in entity_tokens:
                        target_entity = entity_tokens[child.i]
                        
                        relationship = Relationship(
                            relationship_id=f"syntax_{token.i}_{child.i}",
                            source_entity_id=source_entity.entity_id,
                            target_entity_id=target_entity.entity_id,
                            relationship_type=f"syntactic_{child.dep_}",
                            confidence_score=0.6,
                            evidence_text=f"{token.text} {child.dep_} {child.text}",
                            context=context or "",
                            metadata={
                                "extraction_method": "syntactic_analysis",
                                "dependency": child.dep_
                            }
                        )
                        relationships.append(relationship)
        
        return relationships
    
    def _find_entity_by_text(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """Find an entity by its text content."""
        text_lower = text.lower()
        for entity in entities:
            if entity.text.lower() == text_lower:
                return entity
        return None
    
    async def extract_entities_and_relationships(self, text: str, 
                                               context: Optional[str] = None) -> ExtractionResult:
        """
        Extract both entities and relationships from text.
        
        Args:
            text: Input text to process
            context: Optional context information
        
        Returns:
            ExtractionResult containing entities and relationships
        """
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
                    "extraction_timestamp": datetime.now().isoformat()
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