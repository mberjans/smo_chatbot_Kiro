#!/usr/bin/env python3
"""
Configuration Management Script for LightRAG Integration

This script manages configuration across different environments and provides
utilities for configuration validation, migration, and deployment.
"""

import os
import sys
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import subprocess
import shutil
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from lightrag_integration.config.settings import LightRAGConfig
except ImportError:
    print("Warning: Could not import LightRAGConfig. Some features may not work.")
    LightRAGConfig = None

@dataclass
class DeploymentConfig:
    """Deployment configuration settings"""
    environment: str
    database_url: str
    neo4j_password: str
    api_keys: Dict[str, str]
    lightrag_settings: Dict[str, Any]
    monitoring_enabled: bool = True
    log_level: str = "INFO"
    backup_enabled: bool = True
    security_settings: Dict[str, Any] = None

    def __post_init__(self):
        if self.security_settings is None:
            self.security_settings = {}

class ConfigManager:
    """Manages configuration for different deployment environments"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Environment configurations
        self.environments = ["development", "staging", "production"]
        
    def create_environment_config(self, environment: str, template: bool = False) -> Path:
        """Create configuration file for specific environment"""
        config_file = self.config_dir / f"{environment}.yaml"
        
        if config_file.exists() and not template:
            self.logger.warning(f"Configuration file {config_file} already exists")
            return config_file
        
        # Default configuration template
        default_config = {
            "environment": environment,
            "database": {
                "url": f"postgresql://user:password@localhost:5432/lightrag_{environment}",
                "pool_size": 10,
                "max_overflow": 20
            },
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "password": "change_me",
                "database": "neo4j"
            },
            "api_keys": {
                "groq": "your_groq_api_key",
                "openai": "your_openai_api_key",
                "perplexity": "your_perplexity_api_key"
            },
            "lightrag": {
                "knowledge_graph_path": f"data/{environment}/lightrag_kg",
                "vector_store_path": f"data/{environment}/lightrag_vectors",
                "cache_directory": f"data/{environment}/lightrag_cache",
                "embedding_model": "intfloat/e5-base-v2",
                "llm_model": "groq:Llama-3.3-70b-Versatile",
                "batch_size": 32 if environment == "production" else 16,
                "max_concurrent_requests": 10 if environment == "production" else 5,
                "cache_ttl_seconds": 3600
            },
            "monitoring": {
                "enabled": True,
                "interval": 60,
                "metrics_retention_days": 30 if environment == "production" else 7,
                "prometheus_enabled": environment == "production",
                "grafana_enabled": environment == "production"
            },
            "logging": {
                "level": "INFO" if environment == "production" else "DEBUG",
                "file_rotation": True,
                "max_file_size": "50MB",
                "backup_count": 10 if environment == "production" else 5
            },
            "security": {
                "jwt_secret": "generate_secure_secret_key",
                "token_expiry": 3600,
                "rate_limiting": {
                    "enabled": True,
                    "requests_per_minute": 60,
                    "burst_size": 10
                },
                "cors": {
                    "enabled": True,
                    "origins": ["http://localhost:3000"] if environment == "development" else []
                }
            },
            "backup": {
                "enabled": True,
                "schedule": "0 2 * * *",  # Daily at 2 AM
                "retention_days": 30 if environment == "production" else 7,
                "compression": True
            }
        }
        
        # Environment-specific adjustments
        if environment == "development":
            default_config["lightrag"]["batch_size"] = 8
            default_config["monitoring"]["prometheus_enabled"] = False
            default_config["monitoring"]["grafana_enabled"] = False
            
        elif environment == "staging":
            default_config["lightrag"]["batch_size"] = 16
            default_config["monitoring"]["prometheus_enabled"] = True
            default_config["monitoring"]["grafana_enabled"] = False
            
        # Write configuration file
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Created configuration file: {config_file}")
        return config_file
    
    def validate_config(self, config_file: Path) -> bool:
        """Validate configuration file"""
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Required sections
            required_sections = ["database", "neo4j", "api_keys", "lightrag"]
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"Missing required section: {section}")
                    return False
            
            # Validate database URL
            db_url = config["database"]["url"]
            if not db_url.startswith("postgresql://"):
                self.logger.error("Invalid database URL format")
                return False
            
            # Validate API keys (check if they're not default values)
            api_keys = config["api_keys"]
            default_keys = ["your_groq_api_key", "your_openai_api_key", "your_perplexity_api_key"]
            for key, value in api_keys.items():
                if value in default_keys:
                    self.logger.warning(f"API key '{key}' appears to be a default value")
            
            # Validate LightRAG paths
            lightrag_config = config["lightrag"]
            required_paths = ["knowledge_graph_path", "vector_store_path", "cache_directory"]
            for path_key in required_paths:
                if path_key not in lightrag_config:
                    self.logger.error(f"Missing LightRAG path: {path_key}")
                    return False
            
            self.logger.info(f"Configuration validation passed: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def generate_env_file(self, config_file: Path, output_file: Path = None) -> Path:
        """Generate .env file from configuration"""
        if output_file is None:
            output_file = Path(".env")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        env_vars = []
        
        # Database configuration
        env_vars.append(f"DATABASE_URL={config['database']['url']}")
        
        # Neo4j configuration
        env_vars.append(f"NEO4J_URI={config['neo4j']['uri']}")
        env_vars.append(f"NEO4J_PASSWORD={config['neo4j']['password']}")
        
        # API keys
        for key, value in config["api_keys"].items():
            env_vars.append(f"{key.upper()}_API_KEY={value}")
        
        # LightRAG configuration
        lightrag = config["lightrag"]
        env_vars.extend([
            f"LIGHTRAG_KG_PATH={lightrag['knowledge_graph_path']}",
            f"LIGHTRAG_VECTOR_PATH={lightrag['vector_store_path']}",
            f"LIGHTRAG_CACHE_PATH={lightrag['cache_directory']}",
            f"LIGHTRAG_EMBEDDING_MODEL={lightrag['embedding_model']}",
            f"LIGHTRAG_LLM_MODEL={lightrag['llm_model']}",
            f"LIGHTRAG_BATCH_SIZE={lightrag['batch_size']}",
            f"LIGHTRAG_MAX_CONCURRENT={lightrag['max_concurrent_requests']}",
            f"LIGHTRAG_CACHE_TTL={lightrag['cache_ttl_seconds']}"
        ])
        
        # Monitoring configuration
        if "monitoring" in config:
            monitoring = config["monitoring"]
            env_vars.extend([
                f"ENABLE_MONITORING={str(monitoring.get('enabled', True)).lower()}",
                f"MONITORING_INTERVAL={monitoring.get('interval', 60)}",
                f"PROMETHEUS_ENABLED={str(monitoring.get('prometheus_enabled', False)).lower()}",
                f"GRAFANA_ENABLED={str(monitoring.get('grafana_enabled', False)).lower()}"
            ])
        
        # Logging configuration
        if "logging" in config:
            logging_config = config["logging"]
            env_vars.append(f"LOG_LEVEL={logging_config.get('level', 'INFO')}")
        
        # Security configuration
        if "security" in config:
            security = config["security"]
            env_vars.extend([
                f"JWT_SECRET={security.get('jwt_secret', 'change_me')}",
                f"TOKEN_EXPIRY={security.get('token_expiry', 3600)}"
            ])
        
        # Environment
        env_vars.append(f"DEPLOYMENT_ENV={config.get('environment', 'development')}")
        
        # Write .env file
        with open(output_file, 'w') as f:
            f.write("# Generated environment configuration\n")
            f.write(f"# Generated on: {datetime.now().isoformat()}\n")
            f.write(f"# Source: {config_file}\n\n")
            
            for var in env_vars:
                f.write(f"{var}\n")
        
        self.logger.info(f"Generated .env file: {output_file}")
        return output_file
    
    def migrate_config(self, old_version: str, new_version: str, config_file: Path) -> bool:
        """Migrate configuration from old version to new version"""
        backup_file = config_file.with_suffix(f".{old_version}.backup")
        shutil.copy2(config_file, backup_file)
        
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # Version-specific migrations
            if old_version == "1.0" and new_version == "1.1":
                # Add new monitoring section if missing
                if "monitoring" not in config:
                    config["monitoring"] = {
                        "enabled": True,
                        "interval": 60,
                        "metrics_retention_days": 30
                    }
                
                # Add new security settings
                if "security" not in config:
                    config["security"] = {
                        "jwt_secret": "generate_secure_secret_key",
                        "token_expiry": 3600,
                        "rate_limiting": {
                            "enabled": True,
                            "requests_per_minute": 60,
                            "burst_size": 10
                        }
                    }
            
            # Write updated configuration
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.info(f"Migrated configuration from {old_version} to {new_version}")
            self.logger.info(f"Backup saved as: {backup_file}")
            return True
            
        except Exception as e:
            # Restore backup on failure
            shutil.copy2(backup_file, config_file)
            self.logger.error(f"Configuration migration failed: {e}")
            return False
    
    def compare_configs(self, config1: Path, config2: Path) -> Dict[str, Any]:
        """Compare two configuration files"""
        with open(config1, 'r') as f:
            cfg1 = yaml.safe_load(f)
        
        with open(config2, 'r') as f:
            cfg2 = yaml.safe_load(f)
        
        differences = {}
        
        def compare_dict(d1, d2, path=""):
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences[current_path] = {"status": "added", "value": d2[key]}
                elif key not in d2:
                    differences[current_path] = {"status": "removed", "value": d1[key]}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dict(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences[current_path] = {
                        "status": "changed",
                        "old_value": d1[key],
                        "new_value": d2[key]
                    }
        
        compare_dict(cfg1, cfg2)
        return differences
    
    def deploy_config(self, environment: str, dry_run: bool = False) -> bool:
        """Deploy configuration for specific environment"""
        config_file = self.config_dir / f"{environment}.yaml"
        
        if not config_file.exists():
            self.logger.error(f"Configuration file not found: {config_file}")
            return False
        
        if not self.validate_config(config_file):
            self.logger.error("Configuration validation failed")
            return False
        
        if dry_run:
            self.logger.info("DRY RUN: Configuration deployment simulation")
            self.logger.info(f"Would deploy: {config_file}")
            return True
        
        try:
            # Generate .env file
            env_file = self.generate_env_file(config_file)
            
            # Create data directories
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            lightrag_config = config["lightrag"]
            for path_key in ["knowledge_graph_path", "vector_store_path", "cache_directory"]:
                path = Path(lightrag_config[path_key])
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Created directory: {path}")
            
            # Copy logging configuration
            logging_config_file = self.config_dir.parent / "deployment" / "logging.yaml"
            if logging_config_file.exists():
                target_logging_config = Path("config") / "logging.yaml"
                target_logging_config.parent.mkdir(exist_ok=True)
                shutil.copy2(logging_config_file, target_logging_config)
                self.logger.info(f"Deployed logging configuration: {target_logging_config}")
            
            self.logger.info(f"Successfully deployed configuration for {environment}")
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration deployment failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="LightRAG Configuration Manager")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create environment configuration")
    create_parser.add_argument("environment", choices=["development", "staging", "production"])
    create_parser.add_argument("--template", action="store_true", help="Create template even if file exists")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate configuration")
    validate_parser.add_argument("config_file", help="Configuration file to validate")
    
    # Generate env command
    env_parser = subparsers.add_parser("generate-env", help="Generate .env file")
    env_parser.add_argument("config_file", help="Source configuration file")
    env_parser.add_argument("--output", help="Output .env file path")
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Migrate configuration")
    migrate_parser.add_argument("config_file", help="Configuration file to migrate")
    migrate_parser.add_argument("--from-version", required=True, help="Source version")
    migrate_parser.add_argument("--to-version", required=True, help="Target version")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare configurations")
    compare_parser.add_argument("config1", help="First configuration file")
    compare_parser.add_argument("config2", help="Second configuration file")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy configuration")
    deploy_parser.add_argument("environment", choices=["development", "staging", "production"])
    deploy_parser.add_argument("--dry-run", action="store_true", help="Simulate deployment")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    config_manager = ConfigManager(args.config_dir)
    
    if args.command == "create":
        config_manager.create_environment_config(args.environment, args.template)
    
    elif args.command == "validate":
        config_file = Path(args.config_file)
        if config_manager.validate_config(config_file):
            print("✓ Configuration is valid")
        else:
            print("✗ Configuration validation failed")
            sys.exit(1)
    
    elif args.command == "generate-env":
        config_file = Path(args.config_file)
        output_file = Path(args.output) if args.output else None
        config_manager.generate_env_file(config_file, output_file)
    
    elif args.command == "migrate":
        config_file = Path(args.config_file)
        if config_manager.migrate_config(args.from_version, args.to_version, config_file):
            print("✓ Configuration migration completed")
        else:
            print("✗ Configuration migration failed")
            sys.exit(1)
    
    elif args.command == "compare":
        config1 = Path(args.config1)
        config2 = Path(args.config2)
        differences = config_manager.compare_configs(config1, config2)
        
        if not differences:
            print("✓ Configurations are identical")
        else:
            print("Configuration differences:")
            for path, diff in differences.items():
                if diff["status"] == "added":
                    print(f"  + {path}: {diff['value']}")
                elif diff["status"] == "removed":
                    print(f"  - {path}: {diff['value']}")
                elif diff["status"] == "changed":
                    print(f"  ~ {path}: {diff['old_value']} → {diff['new_value']}")
    
    elif args.command == "deploy":
        if config_manager.deploy_config(args.environment, args.dry_run):
            print(f"✓ Configuration deployed for {args.environment}")
        else:
            print(f"✗ Configuration deployment failed for {args.environment}")
            sys.exit(1)

if __name__ == "__main__":
    main()