"""
LightRAG Configuration Settings

This module defines the configuration dataclass and environment variable loading
for the LightRAG integration component.
"""

import os
import json
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging


@dataclass
class LightRAGConfig:
    """Configuration settings for LightRAG integration."""
    
    # Storage configuration
    knowledge_graph_path: str = field(default_factory=lambda: os.getenv("LIGHTRAG_KG_PATH", "./data/lightrag_kg"))
    vector_store_path: str = field(default_factory=lambda: os.getenv("LIGHTRAG_VECTOR_PATH", "./data/lightrag_vectors"))
    cache_directory: str = field(default_factory=lambda: os.getenv("LIGHTRAG_CACHE_DIR", "./data/lightrag_cache"))
    
    # Processing configuration
    chunk_size: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_CHUNK_SIZE", "1000")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_CHUNK_OVERLAP", "200")))
    max_entities_per_chunk: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_ENTITIES", "50")))
    
    # Model configuration
    embedding_model: str = field(default_factory=lambda: os.getenv("LIGHTRAG_EMBEDDING_MODEL", "intfloat/e5-base-v2"))
    llm_model: str = field(default_factory=lambda: os.getenv("LIGHTRAG_LLM_MODEL", "groq:Llama-3.3-70b-Versatile"))
    
    # Performance configuration
    batch_size: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_BATCH_SIZE", "32")))
    max_concurrent_requests: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_CONCURRENT", "10")))
    cache_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_CACHE_TTL", "3600")))
    
    # Papers directory
    papers_directory: str = field(default_factory=lambda: os.getenv("LIGHTRAG_PAPERS_DIR", "./papers"))
    
    # API Keys and credentials
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    
    # Logging configuration
    log_level: str = field(default_factory=lambda: os.getenv("LIGHTRAG_LOG_LEVEL", "INFO"))
    log_file: Optional[str] = field(default_factory=lambda: os.getenv("LIGHTRAG_LOG_FILE"))
    
    # Feature flags
    enable_entity_extraction: bool = field(default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_ENTITY_EXTRACTION", "true").lower() == "true")
    enable_relationship_extraction: bool = field(default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_RELATIONSHIP_EXTRACTION", "true").lower() == "true")
    enable_caching: bool = field(default_factory=lambda: os.getenv("LIGHTRAG_ENABLE_CACHING", "true").lower() == "true")
    
    # Advanced processing options
    min_entity_confidence: float = field(default_factory=lambda: float(os.getenv("LIGHTRAG_MIN_ENTITY_CONFIDENCE", "0.7")))
    min_relationship_confidence: float = field(default_factory=lambda: float(os.getenv("LIGHTRAG_MIN_RELATIONSHIP_CONFIDENCE", "0.6")))
    max_document_size_mb: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_DOCUMENT_SIZE_MB", "50")))
    
    # Query processing configuration
    max_query_length: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_MAX_QUERY_LENGTH", "1000")))
    default_top_k: int = field(default_factory=lambda: int(os.getenv("LIGHTRAG_DEFAULT_TOP_K", "10")))
    similarity_threshold: float = field(default_factory=lambda: float(os.getenv("LIGHTRAG_SIMILARITY_THRESHOLD", "0.5")))
    
    def __post_init__(self):
        """Ensure directories exist after initialization and validate configuration."""
        # Create necessary directories
        for path_attr in ["knowledge_graph_path", "vector_store_path", "cache_directory", "papers_directory"]:
            path = getattr(self, path_attr)
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Set up logging configuration if log_file is not specified
        if not self.log_file:
            self.log_file = str(Path(self.cache_directory) / "lightrag.log")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_env(cls) -> "LightRAGConfig":
        """Create configuration from environment variables."""
        return cls()
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "LightRAGConfig":
        """
        Create configuration from a config file.
        
        Supports JSON and YAML formats. Environment variables take precedence
        over config file values.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            LightRAGConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file format is unsupported or invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load config file
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                elif config_path.suffix.lower() in ['.yml', '.yaml']:
                    file_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"Invalid config file format: {e}")
        
        # Create instance with file config as defaults, but allow env vars to override
        instance = cls()
        
        # Update with file config values only if env var is not set
        for key, value in file_config.items():
            if hasattr(instance, key):
                # Check if environment variable exists for this config
                env_key = f"LIGHTRAG_{key.upper()}"
                if env_key not in os.environ:
                    setattr(instance, key, value)
        
        return instance
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LightRAGConfig":
        """
        Create configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            LightRAGConfig instance
        """
        # Start with environment defaults
        instance = cls()
        
        # Update with provided values
        for key, value in config_dict.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
            else:
                logging.warning(f"Unknown configuration key: {key}")
        
        # Trigger __post_init__ manually since we modified attributes after creation
        instance.__post_init__()
        
        return instance
    
    def validate(self) -> None:
        """
        Validate configuration parameters.
        
        Raises:
            ValueError: If any configuration parameter is invalid
        """
        # Validate processing parameters
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.max_entities_per_chunk <= 0:
            raise ValueError("max_entities_per_chunk must be positive")
        
        # Validate performance parameters
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be positive")
        if self.cache_ttl_seconds < 0:
            raise ValueError("cache_ttl_seconds cannot be negative")
        
        # Validate model configuration
        if not self.embedding_model.strip():
            raise ValueError("embedding_model cannot be empty")
        if not self.llm_model.strip():
            raise ValueError("llm_model cannot be empty")
        
        # Validate paths
        for path_attr in ["knowledge_graph_path", "vector_store_path", "cache_directory", "papers_directory"]:
            path = getattr(self, path_attr)
            if not path.strip():
                raise ValueError(f"{path_attr} cannot be empty")
        
        # Validate confidence thresholds
        if not (0.0 <= self.min_entity_confidence <= 1.0):
            raise ValueError("min_entity_confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.min_relationship_confidence <= 1.0):
            raise ValueError("min_relationship_confidence must be between 0.0 and 1.0")
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be between 0.0 and 1.0")
        
        # Validate size limits
        if self.max_document_size_mb <= 0:
            raise ValueError("max_document_size_mb must be positive")
        if self.max_query_length <= 0:
            raise ValueError("max_query_length must be positive")
        if self.default_top_k <= 0:
            raise ValueError("default_top_k must be positive")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of: {valid_log_levels}")
        
        # Validate API keys if LLM model requires them
        if self.llm_model.startswith("groq:") and not self.groq_api_key:
            raise ValueError("groq_api_key is required when using Groq models")
        if self.llm_model.startswith("openai:") and not self.openai_api_key:
            raise ValueError("openai_api_key is required when using OpenAI models")
    
    def get_effective_config(self) -> Dict[str, Any]:
        """
        Get the effective configuration with resolved paths and validated values.
        
        Returns:
            Dictionary containing the effective configuration
        """
        config = self.to_dict()
        
        # Resolve relative paths to absolute paths
        for path_key in ["knowledge_graph_path", "vector_store_path", "cache_directory", "papers_directory"]:
            config[path_key] = str(Path(config[path_key]).resolve())
        
        if self.log_file:
            config["log_file"] = str(Path(self.log_file).resolve())
        
        return config
    
    def save_to_file(self, config_path: Union[str, Path], format: str = "json") -> None:
        """
        Save configuration to a file.
        
        Args:
            config_path: Path where to save the configuration
            format: File format ("json" or "yaml")
            
        Raises:
            ValueError: If format is not supported
        """
        config_path = Path(config_path)
        config_dict = self.to_dict()
        
        # Remove sensitive information
        sensitive_keys = ["groq_api_key", "openai_api_key"]
        for key in sensitive_keys:
            if key in config_dict and config_dict[key]:
                config_dict[key] = "***REDACTED***"
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if format.lower() == "json":
                json.dump(config_dict, f, indent=2, sort_keys=True)
            elif format.lower() in ["yaml", "yml"]:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def __str__(self) -> str:
        """String representation of configuration (with sensitive data redacted)."""
        config = self.to_dict()
        
        # Redact sensitive information
        sensitive_keys = ["groq_api_key", "openai_api_key"]
        for key in sensitive_keys:
            if key in config and config[key]:
                config[key] = "***REDACTED***"
        
        return f"LightRAGConfig({config})"