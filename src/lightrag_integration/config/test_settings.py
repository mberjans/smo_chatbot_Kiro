"""
Unit tests for LightRAG configuration system.
"""

import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from .settings import LightRAGConfig


class TestLightRAGConfig:
    """Test cases for LightRAGConfig class."""
    
    def test_default_initialization(self):
        """Test that configuration initializes with default values."""
        config = LightRAGConfig()
        
        # Test default values
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_entities_per_chunk == 50
        assert config.embedding_model == "intfloat/e5-base-v2"
        assert config.llm_model == "groq:Llama-3.3-70b-Versatile"
        assert config.batch_size == 32
        assert config.max_concurrent_requests == 10
        assert config.cache_ttl_seconds == 3600
        assert config.papers_directory == "./papers"
        assert config.log_level == "INFO"
        assert config.enable_entity_extraction is True
        assert config.enable_relationship_extraction is True
        assert config.enable_caching is True
        assert config.min_entity_confidence == 0.7
        assert config.min_relationship_confidence == 0.6
        assert config.max_document_size_mb == 50
        assert config.max_query_length == 1000
        assert config.default_top_k == 10
        assert config.similarity_threshold == 0.5
    
    def test_environment_variable_override(self):
        """Test that environment variables override default values."""
        env_vars = {
            "LIGHTRAG_CHUNK_SIZE": "2000",
            "LIGHTRAG_CHUNK_OVERLAP": "400",
            "LIGHTRAG_MAX_ENTITIES": "100",
            "LIGHTRAG_EMBEDDING_MODEL": "custom-model",
            "LIGHTRAG_LLM_MODEL": "custom-llm",
            "LIGHTRAG_BATCH_SIZE": "64",
            "LIGHTRAG_MAX_CONCURRENT": "20",
            "LIGHTRAG_CACHE_TTL": "7200",
            "LIGHTRAG_PAPERS_DIR": "./custom_papers",
            "LIGHTRAG_LOG_LEVEL": "DEBUG",
            "LIGHTRAG_ENABLE_ENTITY_EXTRACTION": "false",
            "LIGHTRAG_ENABLE_RELATIONSHIP_EXTRACTION": "false",
            "LIGHTRAG_ENABLE_CACHING": "false",
            "LIGHTRAG_MIN_ENTITY_CONFIDENCE": "0.8",
            "LIGHTRAG_MIN_RELATIONSHIP_CONFIDENCE": "0.7",
            "LIGHTRAG_MAX_DOCUMENT_SIZE_MB": "100",
            "LIGHTRAG_MAX_QUERY_LENGTH": "2000",
            "LIGHTRAG_DEFAULT_TOP_K": "20",
            "LIGHTRAG_SIMILARITY_THRESHOLD": "0.6",
            "GROQ_API_KEY": "test-groq-key",
            "OPENAI_API_KEY": "test-openai-key"
        }
        
        with patch.dict(os.environ, env_vars):
            config = LightRAGConfig()
            
            assert config.chunk_size == 2000
            assert config.chunk_overlap == 400
            assert config.max_entities_per_chunk == 100
            assert config.embedding_model == "custom-model"
            assert config.llm_model == "custom-llm"
            assert config.batch_size == 64
            assert config.max_concurrent_requests == 20
            assert config.cache_ttl_seconds == 7200
            assert config.papers_directory == "./custom_papers"
            assert config.log_level == "DEBUG"
            assert config.enable_entity_extraction is False
            assert config.enable_relationship_extraction is False
            assert config.enable_caching is False
            assert config.min_entity_confidence == 0.8
            assert config.min_relationship_confidence == 0.7
            assert config.max_document_size_mb == 100
            assert config.max_query_length == 2000
            assert config.default_top_k == 20
            assert config.similarity_threshold == 0.6
            assert config.groq_api_key == "test-groq-key"
            assert config.openai_api_key == "test-openai-key"
    
    def test_from_env_class_method(self):
        """Test from_env class method."""
        with patch.dict(os.environ, {"LIGHTRAG_CHUNK_SIZE": "1500"}):
            config = LightRAGConfig.from_env()
            assert config.chunk_size == 1500
    
    def test_from_dict_class_method(self):
        """Test from_dict class method."""
        config_dict = {
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "embedding_model": "test-model",
            "unknown_key": "should_be_ignored"  # This should be ignored with a warning
        }
        
        config = LightRAGConfig.from_dict(config_dict)
        assert config.chunk_size == 1500
        assert config.chunk_overlap == 300
        assert config.embedding_model == "test-model"
        # unknown_key should not be set
        assert not hasattr(config, "unknown_key")
    
    def test_from_file_json(self):
        """Test loading configuration from JSON file."""
        config_data = {
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "embedding_model": "json-model",
            "enable_caching": False
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            config = LightRAGConfig.from_file(temp_path)
            assert config.chunk_size == 1500
            assert config.chunk_overlap == 300
            assert config.embedding_model == "json-model"
            assert config.enable_caching is False
        finally:
            os.unlink(temp_path)
    
    def test_from_file_yaml(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
chunk_size: 1500
chunk_overlap: 300
embedding_model: yaml-model
enable_caching: false
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            config = LightRAGConfig.from_file(temp_path)
            assert config.chunk_size == 1500
            assert config.chunk_overlap == 300
            assert config.embedding_model == "yaml-model"
            assert config.enable_caching is False
        finally:
            os.unlink(temp_path)
    
    def test_from_file_nonexistent(self):
        """Test loading from non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            LightRAGConfig.from_file("/nonexistent/path/config.json")
    
    def test_from_file_invalid_format(self):
        """Test loading from file with invalid format raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("invalid content")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported config file format"):
                LightRAGConfig.from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_from_file_invalid_json(self):
        """Test loading from file with invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Invalid config file format"):
                LightRAGConfig.from_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_from_file_env_override(self):
        """Test that environment variables override file values."""
        config_data = {
            "chunk_size": 1500,
            "embedding_model": "file-model"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        try:
            # Environment variable should override file value
            with patch.dict(os.environ, {"LIGHTRAG_CHUNK_SIZE": "2000"}):
                config = LightRAGConfig.from_file(temp_path)
                assert config.chunk_size == 2000  # From env var
                assert config.embedding_model == "file-model"  # From file
        finally:
            os.unlink(temp_path)
    
    def test_validation_success(self):
        """Test that valid configuration passes validation."""
        config = LightRAGConfig()
        config.validate()  # Should not raise any exception
    
    def test_validation_chunk_size_invalid(self):
        """Test validation fails for invalid chunk_size."""
        config = LightRAGConfig()
        config.chunk_size = 0
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            config.validate()
        
        config.chunk_size = -100
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            config.validate()
    
    def test_validation_chunk_overlap_invalid(self):
        """Test validation fails for invalid chunk_overlap."""
        config = LightRAGConfig()
        config.chunk_overlap = -1
        with pytest.raises(ValueError, match="chunk_overlap cannot be negative"):
            config.validate()
        
        config.chunk_overlap = config.chunk_size
        with pytest.raises(ValueError, match="chunk_overlap must be less than chunk_size"):
            config.validate()
    
    def test_validation_confidence_thresholds(self):
        """Test validation of confidence thresholds."""
        config = LightRAGConfig()
        
        # Test invalid entity confidence
        config.min_entity_confidence = -0.1
        with pytest.raises(ValueError, match="min_entity_confidence must be between 0.0 and 1.0"):
            config.validate()
        
        config.min_entity_confidence = 1.1
        with pytest.raises(ValueError, match="min_entity_confidence must be between 0.0 and 1.0"):
            config.validate()
        
        # Reset and test relationship confidence
        config.min_entity_confidence = 0.7
        config.min_relationship_confidence = -0.1
        with pytest.raises(ValueError, match="min_relationship_confidence must be between 0.0 and 1.0"):
            config.validate()
        
        config.min_relationship_confidence = 1.1
        with pytest.raises(ValueError, match="min_relationship_confidence must be between 0.0 and 1.0"):
            config.validate()
    
    def test_validation_api_keys(self):
        """Test validation of API keys for different models."""
        config = LightRAGConfig()
        
        # Test Groq model without API key
        config.llm_model = "groq:test-model"
        config.groq_api_key = None
        with pytest.raises(ValueError, match="groq_api_key is required when using Groq models"):
            config.validate()
        
        # Test OpenAI model without API key
        config.llm_model = "openai:test-model"
        config.openai_api_key = None
        with pytest.raises(ValueError, match="openai_api_key is required when using OpenAI models"):
            config.validate()
    
    def test_validation_log_level(self):
        """Test validation of log level."""
        config = LightRAGConfig()
        config.log_level = "INVALID"
        with pytest.raises(ValueError, match="log_level must be one of"):
            config.validate()
    
    def test_validation_empty_strings(self):
        """Test validation fails for empty string values."""
        config = LightRAGConfig()
        
        config.embedding_model = ""
        with pytest.raises(ValueError, match="embedding_model cannot be empty"):
            config.validate()
        
        config.embedding_model = "valid-model"
        config.llm_model = "   "  # Whitespace only
        with pytest.raises(ValueError, match="llm_model cannot be empty"):
            config.validate()
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LightRAGConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "chunk_size" in config_dict
        assert "embedding_model" in config_dict
        assert config_dict["chunk_size"] == config.chunk_size
        assert config_dict["embedding_model"] == config.embedding_model
    
    def test_get_effective_config(self):
        """Test getting effective configuration with resolved paths."""
        config = LightRAGConfig()
        effective = config.get_effective_config()
        
        # Paths should be resolved to absolute paths
        assert Path(effective["knowledge_graph_path"]).is_absolute()
        assert Path(effective["vector_store_path"]).is_absolute()
        assert Path(effective["cache_directory"]).is_absolute()
        assert Path(effective["papers_directory"]).is_absolute()
    
    def test_save_to_file_json(self):
        """Test saving configuration to JSON file."""
        config = LightRAGConfig()
        config.groq_api_key = "secret-key"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_to_file(temp_path, format="json")
            
            # Verify file was created and contains expected data
            with open(temp_path, 'r') as f:
                saved_data = json.load(f)
            
            assert saved_data["chunk_size"] == config.chunk_size
            assert saved_data["groq_api_key"] == "***REDACTED***"  # Should be redacted
        finally:
            os.unlink(temp_path)
    
    def test_save_to_file_yaml(self):
        """Test saving configuration to YAML file."""
        config = LightRAGConfig()
        config.openai_api_key = "secret-key"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            temp_path = f.name
        
        try:
            config.save_to_file(temp_path, format="yaml")
            
            # Verify file was created
            assert Path(temp_path).exists()
            
            # Read and verify content
            with open(temp_path, 'r') as f:
                content = f.read()
                assert "chunk_size:" in content
                assert "***REDACTED***" in content  # API key should be redacted
        finally:
            os.unlink(temp_path)
    
    def test_save_to_file_invalid_format(self):
        """Test saving with invalid format raises ValueError."""
        config = LightRAGConfig()
        
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                config.save_to_file(temp_path, format="xml")
        finally:
            if Path(temp_path).exists():
                os.unlink(temp_path)
    
    def test_string_representation(self):
        """Test string representation redacts sensitive data."""
        config = LightRAGConfig()
        config.groq_api_key = "secret-key"
        config.openai_api_key = "another-secret"
        
        str_repr = str(config)
        assert "***REDACTED***" in str_repr
        assert "secret-key" not in str_repr
        assert "another-secret" not in str_repr
        assert "LightRAGConfig" in str_repr
    
    def test_post_init_creates_directories(self):
        """Test that __post_init__ creates necessary directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dict = {
                "knowledge_graph_path": str(Path(temp_dir) / "kg"),
                "vector_store_path": str(Path(temp_dir) / "vectors"),
                "cache_directory": str(Path(temp_dir) / "cache"),
                "papers_directory": str(Path(temp_dir) / "papers")
            }
            
            # Create instance which should trigger __post_init__
            config = LightRAGConfig.from_dict(config_dict)
            
            # Verify directories were created
            assert Path(config.knowledge_graph_path).exists()
            assert Path(config.vector_store_path).exists()
            assert Path(config.cache_directory).exists()
            assert Path(config.papers_directory).exists()
    
    def test_post_init_sets_log_file(self):
        """Test that __post_init__ sets log_file if not specified."""
        config = LightRAGConfig()
        assert config.log_file is not None
        assert config.log_file.endswith("lightrag.log")
        # Check that the log file path contains the cache directory name
        assert "lightrag_cache" in config.log_file


if __name__ == "__main__":
    pytest.main([__file__])