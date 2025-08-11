#!/usr/bin/env python3
"""
Monitoring setup and configuration script for LightRAG integration
"""

import os
import sys
import json
import yaml
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil

class MonitoringSetup:
    """Sets up monitoring infrastructure for LightRAG"""
    
    def __init__(self, config_dir: str = "config", deployment_dir: str = None):
        self.config_dir = Path(config_dir)
        self.deployment_dir = Path(deployment_dir) if deployment_dir else Path(__file__).parent.parent
        self.config_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Monitoring components
        self.components = {
            'prometheus': {
                'port': 9090,
                'config_file': 'prometheus.yml',
                'data_dir': 'prometheus_data'
            },
            'grafana': {
                'port': 3000,
                'config_dir': 'grafana',
                'data_dir': 'grafana_data'
            },
            'alertmanager': {
                'port': 9093,
                'config_file': 'alertmanager.yml',
                'data_dir': 'alertmanager_data'
            },
            'node_exporter': {
                'port': 9100,
                'enabled': True
            }
        }
    
    def setup_prometheus(self, environment: str = 'production') -> Dict[str, Any]:
        """Setup Prometheus monitoring"""
        self.logger.info("Setting up Prometheus monitoring...")
        
        # Create Prometheus configuration
        prometheus_config = {
            'global': {
                'scrape_interval': '15s',
                'evaluation_interval': '15s'
            },
            'rule_files': [
                'alert_rules.yml'
            ],
            'alerting': {
                'alertmanagers': [
                    {
                        'static_configs': [
                            {'targets': ['alertmanager:9093']}
                        ]
                    }
                ]
            },
            'scrape_configs': [
                {
                    'job_name': 'lightrag-oracle',
                    'static_configs': [
                        {'targets': ['lightrag-oracle:8000']}
                    ],
                    'metrics_path': '/metrics',
                    'scrape_interval': '30s',
                    'scrape_timeout': '10s'
                },
                {
                    'job_name': 'postgres',
                    'static_configs': [
                        {'targets': ['postgres:5432']}
                    ],
                    'scrape_interval': '30s'
                },
                {
                    'job_name': 'neo4j',
                    'static_configs': [
                        {'targets': ['neo4j:2004']}
                    ],
                    'scrape_interval': '30s'
                },
                {
                    'job_name': 'redis',
                    'static_configs': [
                        {'targets': ['redis:6379']}
                    ],
                    'scrape_interval': '30s'
                },
                {
                    'job_name': 'node-exporter',
                    'static_configs': [
                        {'targets': ['node-exporter:9100']}
                    ],
                    'scrape_interval': '30s'
                },
                {
                    'job_name': 'prometheus',
                    'static_configs': [
                        {'targets': ['localhost:9090']}
                    ]
                }
            ]
        }
        
        # Environment-specific adjustments
        if environment == 'development':
            prometheus_config['global']['scrape_interval'] = '30s'
            prometheus_config['global']['evaluation_interval'] = '30s'
        elif environment == 'staging':
            prometheus_config['global']['scrape_interval'] = '20s'
            prometheus_config['global']['evaluation_interval'] = '20s'
        
        # Write Prometheus configuration
        prometheus_config_path = self.deployment_dir / 'prometheus.yml'
        with open(prometheus_config_path, 'w') as f:
            yaml.dump(prometheus_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Prometheus configuration written to: {prometheus_config_path}")
        
        return {
            'status': 'success',
            'config_path': str(prometheus_config_path),
            'port': self.components['prometheus']['port']
        }
    
    def setup_grafana(self, environment: str = 'production') -> Dict[str, Any]:
        """Setup Grafana dashboards and data sources"""
        self.logger.info("Setting up Grafana dashboards...")
        
        grafana_dir = self.deployment_dir / 'grafana'
        grafana_dir.mkdir(exist_ok=True)
        
        # Create datasources directory
        datasources_dir = grafana_dir / 'datasources'
        datasources_dir.mkdir(exist_ok=True)
        
        # Create dashboards directory
        dashboards_dir = grafana_dir / 'dashboards'
        dashboards_dir.mkdir(exist_ok=True)
        
        # Prometheus datasource configuration
        prometheus_datasource = {
            'apiVersion': 1,
            'datasources': [
                {
                    'name': 'Prometheus',
                    'type': 'prometheus',
                    'access': 'proxy',
                    'url': 'http://prometheus:9090',
                    'isDefault': True,
                    'editable': True
                }
            ]
        }
        
        datasource_path = datasources_dir / 'prometheus.yml'
        with open(datasource_path, 'w') as f:
            yaml.dump(prometheus_datasource, f, default_flow_style=False, indent=2)
        
        # Dashboard provisioning configuration
        dashboard_config = {
            'apiVersion': 1,
            'providers': [
                {
                    'name': 'lightrag-dashboards',
                    'orgId': 1,
                    'folder': '',
                    'type': 'file',
                    'disableDeletion': False,
                    'updateIntervalSeconds': 10,
                    'allowUiUpdates': True,
                    'options': {
                        'path': '/etc/grafana/provisioning/dashboards'
                    }
                }
            ]
        }
        
        dashboard_config_path = dashboards_dir / 'dashboard.yml'
        with open(dashboard_config_path, 'w') as f:
            yaml.dump(dashboard_config, f, default_flow_style=False, indent=2)
        
        # Create LightRAG dashboard
        lightrag_dashboard = self._create_lightrag_dashboard()
        dashboard_json_path = dashboards_dir / 'lightrag-dashboard.json'
        with open(dashboard_json_path, 'w') as f:
            json.dump(lightrag_dashboard, f, indent=2)
        
        # Create system dashboard
        system_dashboard = self._create_system_dashboard()
        system_dashboard_path = dashboards_dir / 'system-dashboard.json'
        with open(system_dashboard_path, 'w') as f:
            json.dump(system_dashboard, f, indent=2)
        
        self.logger.info(f"Grafana configuration written to: {grafana_dir}")
        
        return {
            'status': 'success',
            'config_dir': str(grafana_dir),
            'port': self.components['grafana']['port'],
            'dashboards': ['lightrag-dashboard.json', 'system-dashboard.json']
        }
    
    def setup_alertmanager(self, environment: str = 'production') -> Dict[str, Any]:
        """Setup Alertmanager for alert routing"""
        self.logger.info("Setting up Alertmanager...")
        
        # Alertmanager configuration
        alertmanager_config = {
            'global': {
                'smtp_smarthost': 'localhost:587',
                'smtp_from': 'alerts@lightrag-oracle.com'
            },
            'route': {
                'group_by': ['alertname'],
                'group_wait': '10s',
                'group_interval': '10s',
                'repeat_interval': '1h',
                'receiver': 'web.hook'
            },
            'receivers': [
                {
                    'name': 'web.hook',
                    'email_configs': [
                        {
                            'to': 'admin@lightrag-oracle.com',
                            'subject': 'LightRAG Alert: {{ .GroupLabels.alertname }}',
                            'body': '''
Alert: {{ .GroupLabels.alertname }}
Status: {{ .Status }}
Summary: {{ .CommonAnnotations.summary }}
Description: {{ .CommonAnnotations.description }}

Details:
{{ range .Alerts }}
- Alert: {{ .Annotations.summary }}
  Status: {{ .Status }}
  Labels: {{ .Labels }}
{{ end }}
'''
                        }
                    ]
                }
            ]
        }
        
        # Environment-specific adjustments
        if environment == 'development':
            alertmanager_config['route']['repeat_interval'] = '5m'
        elif environment == 'staging':
            alertmanager_config['route']['repeat_interval'] = '30m'
        
        # Write Alertmanager configuration
        alertmanager_config_path = self.deployment_dir / 'alertmanager.yml'
        with open(alertmanager_config_path, 'w') as f:
            yaml.dump(alertmanager_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Alertmanager configuration written to: {alertmanager_config_path}")
        
        return {
            'status': 'success',
            'config_path': str(alertmanager_config_path),
            'port': self.components['alertmanager']['port']
        }
    
    def setup_logging(self, environment: str = 'production') -> Dict[str, Any]:
        """Setup centralized logging configuration"""
        self.logger.info("Setting up logging configuration...")
        
        # Enhanced logging configuration
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'detailed': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d [%(funcName)s]: %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                },
                'json': {
                    'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                    'format': '%(asctime)s %(name)s %(levelname)s %(message)s %(pathname)s %(lineno)d'
                },
                'access': {
                    'format': '%(asctime)s [ACCESS] %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'INFO' if environment == 'production' else 'DEBUG',
                    'formatter': 'standard',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'DEBUG',
                    'formatter': 'detailed',
                    'filename': 'logs/lightrag.log',
                    'maxBytes': 52428800,  # 50MB
                    'backupCount': 10,
                    'encoding': 'utf8'
                },
                'error_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'ERROR',
                    'formatter': 'detailed',
                    'filename': 'logs/lightrag_errors.log',
                    'maxBytes': 52428800,  # 50MB
                    'backupCount': 10,
                    'encoding': 'utf8'
                },
                'performance_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'json',
                    'filename': 'logs/lightrag_performance.log',
                    'maxBytes': 52428800,  # 50MB
                    'backupCount': 5,
                    'encoding': 'utf8'
                },
                'security_file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'WARNING',
                    'formatter': 'detailed',
                    'filename': 'logs/lightrag_security.log',
                    'maxBytes': 52428800,  # 50MB
                    'backupCount': 10,
                    'encoding': 'utf8'
                }
            },
            'loggers': {
                'lightrag_integration': {
                    'level': 'DEBUG' if environment != 'production' else 'INFO',
                    'handlers': ['console', 'file', 'error_file'],
                    'propagate': False
                },
                'performance': {
                    'level': 'INFO',
                    'handlers': ['performance_file'],
                    'propagate': False
                },
                'security': {
                    'level': 'WARNING',
                    'handlers': ['security_file', 'error_file'],
                    'propagate': False
                },
                'uvicorn': {
                    'level': 'INFO',
                    'handlers': ['file'],
                    'propagate': False
                },
                'chainlit': {
                    'level': 'INFO',
                    'handlers': ['file'],
                    'propagate': False
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file']
            }
        }
        
        # Environment-specific adjustments
        if environment == 'development':
            logging_config['handlers']['console']['level'] = 'DEBUG'
            logging_config['loggers']['lightrag_integration']['level'] = 'DEBUG'
        elif environment == 'staging':
            logging_config['handlers']['file']['backupCount'] = 5
            logging_config['handlers']['error_file']['backupCount'] = 5
        
        # Write logging configuration
        logging_config_path = self.deployment_dir / 'logging.yaml'
        with open(logging_config_path, 'w') as f:
            yaml.dump(logging_config, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Logging configuration written to: {logging_config_path}")
        
        return {
            'status': 'success',
            'config_path': str(logging_config_path),
            'log_files': [
                'logs/lightrag.log',
                'logs/lightrag_errors.log',
                'logs/lightrag_performance.log',
                'logs/lightrag_security.log'
            ]
        }
    
    def _create_lightrag_dashboard(self) -> Dict[str, Any]:
        """Create LightRAG-specific Grafana dashboard"""
        return {
            "dashboard": {
                "id": None,
                "title": "LightRAG Oracle Dashboard",
                "tags": ["lightrag", "oracle"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "Query Response Time",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "histogram_quantile(0.95, lightrag_query_duration_seconds_bucket)",
                                "legendFormat": "95th percentile"
                            },
                            {
                                "expr": "histogram_quantile(0.50, lightrag_query_duration_seconds_bucket)",
                                "legendFormat": "50th percentile"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Seconds",
                                "min": 0
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 0,
                            "y": 0
                        }
                    },
                    {
                        "id": 2,
                        "title": "Query Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(lightrag_queries_total[5m])",
                                "legendFormat": "Queries/sec"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Queries/sec",
                                "min": 0
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 12,
                            "y": 0
                        }
                    },
                    {
                        "id": 3,
                        "title": "Error Rate",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "rate(lightrag_errors_total[5m])",
                                "legendFormat": "Errors/sec"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Errors/sec",
                                "min": 0
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 0,
                            "y": 8
                        }
                    },
                    {
                        "id": 4,
                        "title": "Cache Hit Rate",
                        "type": "singlestat",
                        "targets": [
                            {
                                "expr": "lightrag_cache_hit_rate",
                                "legendFormat": "Hit Rate"
                            }
                        ],
                        "valueName": "current",
                        "format": "percent",
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 12,
                            "y": 8
                        }
                    },
                    {
                        "id": 5,
                        "title": "Active Connections",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "lightrag_active_connections",
                                "legendFormat": "Active Connections"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Connections",
                                "min": 0
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 24,
                            "x": 0,
                            "y": 16
                        }
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
    
    def _create_system_dashboard(self) -> Dict[str, Any]:
        """Create system metrics Grafana dashboard"""
        return {
            "dashboard": {
                "id": None,
                "title": "System Metrics Dashboard",
                "tags": ["system", "infrastructure"],
                "timezone": "browser",
                "panels": [
                    {
                        "id": 1,
                        "title": "CPU Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                                "legendFormat": "CPU Usage %"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Percent",
                                "min": 0,
                                "max": 100
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 0,
                            "y": 0
                        }
                    },
                    {
                        "id": 2,
                        "title": "Memory Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
                                "legendFormat": "Memory Usage %"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Percent",
                                "min": 0,
                                "max": 100
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 12,
                            "y": 0
                        }
                    },
                    {
                        "id": 3,
                        "title": "Disk Usage",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100",
                                "legendFormat": "Disk Usage % - {{ mountpoint }}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Percent",
                                "min": 0,
                                "max": 100
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 0,
                            "y": 8
                        }
                    },
                    {
                        "id": 4,
                        "title": "Network I/O",
                        "type": "graph",
                        "targets": [
                            {
                                "expr": "irate(node_network_receive_bytes_total[5m])",
                                "legendFormat": "Receive - {{ device }}"
                            },
                            {
                                "expr": "irate(node_network_transmit_bytes_total[5m])",
                                "legendFormat": "Transmit - {{ device }}"
                            }
                        ],
                        "yAxes": [
                            {
                                "label": "Bytes/sec",
                                "min": 0
                            }
                        ],
                        "gridPos": {
                            "h": 8,
                            "w": 12,
                            "x": 12,
                            "y": 8
                        }
                    }
                ],
                "time": {
                    "from": "now-1h",
                    "to": "now"
                },
                "refresh": "30s"
            }
        }
    
    def setup_all(self, environment: str = 'production') -> Dict[str, Any]:
        """Setup all monitoring components"""
        self.logger.info(f"Setting up all monitoring components for {environment} environment...")
        
        results = {}
        
        try:
            # Setup Prometheus
            results['prometheus'] = self.setup_prometheus(environment)
            
            # Setup Grafana
            results['grafana'] = self.setup_grafana(environment)
            
            # Setup Alertmanager
            results['alertmanager'] = self.setup_alertmanager(environment)
            
            # Setup Logging
            results['logging'] = self.setup_logging(environment)
            
            # Create monitoring directories
            monitoring_dirs = ['prometheus_data', 'grafana_data', 'alertmanager_data', 'logs']
            for dir_name in monitoring_dirs:
                dir_path = Path(dir_name)
                dir_path.mkdir(exist_ok=True)
                self.logger.info(f"Created directory: {dir_path}")
            
            results['status'] = 'success'
            results['environment'] = environment
            results['directories_created'] = monitoring_dirs
            
            self.logger.info("All monitoring components setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Monitoring setup failed: {e}")
            results['status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def validate_setup(self) -> Dict[str, Any]:
        """Validate monitoring setup"""
        self.logger.info("Validating monitoring setup...")
        
        validation_results = {}
        
        # Check configuration files
        config_files = {
            'prometheus': self.deployment_dir / 'prometheus.yml',
            'alertmanager': self.deployment_dir / 'alertmanager.yml',
            'logging': self.deployment_dir / 'logging.yaml',
            'alert_rules': self.deployment_dir / 'alert_rules.yml'
        }
        
        for name, path in config_files.items():
            validation_results[f'{name}_config'] = {
                'exists': path.exists(),
                'path': str(path),
                'readable': path.is_file() and os.access(path, os.R_OK) if path.exists() else False
            }
        
        # Check directories
        directories = {
            'grafana': self.deployment_dir / 'grafana',
            'prometheus_data': Path('prometheus_data'),
            'grafana_data': Path('grafana_data'),
            'alertmanager_data': Path('alertmanager_data'),
            'logs': Path('logs')
        }
        
        for name, path in directories.items():
            validation_results[f'{name}_dir'] = {
                'exists': path.exists(),
                'path': str(path),
                'writable': os.access(path, os.W_OK) if path.exists() else False
            }
        
        # Overall validation status
        all_valid = all(
            result.get('exists', False) and result.get('readable', True)
            for result in validation_results.values()
        )
        
        validation_results['overall_status'] = 'valid' if all_valid else 'invalid'
        
        return validation_results

def main():
    parser = argparse.ArgumentParser(description="LightRAG Monitoring Setup")
    parser.add_argument('--config-dir', default='config', help='Configuration directory')
    parser.add_argument('--deployment-dir', help='Deployment directory')
    parser.add_argument('--environment', choices=['development', 'staging', 'production'], 
                       default='production', help='Deployment environment')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup commands
    setup_parser = subparsers.add_parser('setup', help='Setup monitoring components')
    setup_parser.add_argument('--component', choices=['prometheus', 'grafana', 'alertmanager', 'logging', 'all'],
                             default='all', help='Component to setup')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate monitoring setup')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    monitoring_setup = MonitoringSetup(args.config_dir, args.deployment_dir)
    
    try:
        if args.command == 'setup':
            if args.component == 'all':
                result = monitoring_setup.setup_all(args.environment)
            elif args.component == 'prometheus':
                result = monitoring_setup.setup_prometheus(args.environment)
            elif args.component == 'grafana':
                result = monitoring_setup.setup_grafana(args.environment)
            elif args.component == 'alertmanager':
                result = monitoring_setup.setup_alertmanager(args.environment)
            elif args.component == 'logging':
                result = monitoring_setup.setup_logging(args.environment)
            
            print(json.dumps(result, indent=2))
            
        elif args.command == 'validate':
            result = monitoring_setup.validate_setup()
            print(json.dumps(result, indent=2))
            
            if result['overall_status'] != 'valid':
                sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()