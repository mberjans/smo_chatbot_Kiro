"""
LightRAG Integration Module

This module provides integration between LightRAG and the Clinical Metabolomics Oracle system.
It includes components for PDF ingestion, knowledge graph construction, and query processing.
"""

from .component import LightRAGComponent
from .config.settings import LightRAGConfig

__version__ = "1.0.0"
__all__ = ["LightRAGComponent", "LightRAGConfig"]