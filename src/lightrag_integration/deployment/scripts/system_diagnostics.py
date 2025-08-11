#!/usr/bin/env python3
"""
Comprehensive system diagnostics for LightRAG integration
"""

import os
import sys
import json
import time
import psutil
import socket
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import tempfile
import platform

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class SystemDiagnostics:
    """Comprehensive system diagnostics for LightRAG"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Diagnostic categories
        self.categories = [
            'system_info',
            'resource_usage',
            'network_connectivity',
            'process_status',
            'file_system',
            'database_connectivity',
            'application_health',
            'log_analysis',
            'performance_metrics'
        ]
    
    def run_full_diagnostics(self) -> Dict[str, Any]:
        """Run complete system diagnostics"""
        self.logger.info("Starting comprehensive system diagnostics...")
        
        start_time = time.time()
        results = {
            'timestamp': datetime.now().isoformat(),
            'diagnostics': {}
        }
        
        # Run each diagnostic category
        for category in self.categories:
            self.logger.info(f"Running {category} diagnostics...")
            
            try:
                method_name = f"diagnose_{category}"
                if hasattr(self, method_name):
                    method = getattr(self, method_name)
                    results['diagnostics'][category] = method()
                else:
                    results['diagnostics'][category] = {
                        'status': 'skipped',
                        'reason': f'Method {method_name} not implemented'
                    }
            except Exception as e:
                self.logger.error(f"Error in {category} diagnostics: {e}")
                results['diagnostics'][category] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Calculate overall health score
        results['overall_health'] = self._calculate_health_score(results['diagnostics'])
        results['duration'] = time.time() - start_time
        
        self.logger.info(f"Diagnostics completed in {results['duration']:.2f}s")
        self.logger.info(f"Overall health score: {results['overall_health']['score']}/100")
        
        return results
    
    def diagnose_system_info(self) -> Dict[str, Any]:
        """Gather system information"""
        try:
            return {
                'status': 'success',
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor(),
                    'architecture': platform.architecture(),
                    'hostname': socket.gethostname(),
                    'fqdn': socket.getfqdn()
                },
                'python': {
                    'version': platform.python_version(),
                    'implementation': platform.python_implementation(),
                    'executable': sys.executable,
                    'path': sys.path[:3]  # First 3 entries
                },
                'environment': {
                    'user': os.getenv('USER', 'unknown'),
                    'home': os.getenv('HOME', 'unknown'),
                    'path': os.getenv('PATH', '').split(':')[:5],  # First 5 entries
                    'deployment_env': os.getenv('DEPLOYMENT_ENV', 'unknown')
                },
                'uptime': self._get_system_uptime()
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def diagnose_resource_usage(self) -> Dict[str, Any]:
        """Analyze system resource usage"""
        try:
            # CPU information
            cpu_info = {
                'count': psutil.cpu_count(),
                'count_logical': psutil.cpu_count(logical=True),
                'usage_percent': psutil.cpu_percent(interval=1),
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None,
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            }
            
            # Memory information
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            memory_info = {
                'virtual': {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent,
                    'free': memory.free
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent
                }
            }
            
            # Disk information
            disk_info = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info[partition.mountpoint] = {
                        'device': partition.device,
                        'fstype': partition.fstype,
                        'total': usage.total,
                        'used': usage.used,
                        'free': usage.free,
                        'percent': (usage.used / usage.total) * 100
                    }
                except PermissionError:
                    disk_info[partition.mountpoint] = {
                        'device': partition.device,
                        'fstype': partition.fstype,
                        'error': 'Permission denied'
                    }
            
            # Network information
            network_info = {}
            for interface, addresses in psutil.net_if_addrs().items():
                network_info[interface] = {
                    'addresses': [addr._asdict() for addr in addresses],
                    'stats': psutil.net_if_stats()[interface]._asdict() if interface in psutil.net_if_stats() else None
                }
            
            # Determine status based on resource usage
            status = 'healthy'
            if cpu_info['usage_percent'] > 90 or memory_info['virtual']['percent'] > 95:
                status = 'critical'
            elif cpu_info['usage_percent'] > 80 or memory_info['virtual']['percent'] > 85:
                status = 'warning'
            
            return {
                'status': status,
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info,
                'network': network_info
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def diagnose_network_connectivity(self) -> Dict[str, Any]:
        """Test network connectivity"""
        try:
            connectivity_tests = {
                'localhost': self._test_connection('localhost', 8000),
                'database': self._test_database_connectivity(),
                'external': self._test_external_connectivity()
            }
            
            # Determine overall status
            failed_tests = sum(1 for test in connectivity_tests.values() 
                             if test.get('status') != 'success')
            
            if failed_tests == 0:
                status = 'healthy'
            elif failed_tests <= len(connectivity_tests) // 2:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'tests': connectivity_tests
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def diagnose_process_status(self) -> Dict[str, Any]:
        """Check process status"""
        try:
            processes = {}
            
            # Find LightRAG processes
            lightrag_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'status', 'cpu_percent', 'memory_percent']):
                try:
                    if any('main.py' in str(cmd) for cmd in proc.info['cmdline'] or []):
                        lightrag_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'status': proc.info['status'],
                            'cpu_percent': proc.info['cpu_percent'],
                            'memory_percent': proc.info['memory_percent'],
                            'cmdline': ' '.join(proc.info['cmdline'] or [])
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            processes['lightrag'] = lightrag_processes
            
            # Check for database processes
            db_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'status']):
                try:
                    if proc.info['name'] in ['postgres', 'neo4j', 'redis-server']:
                        db_processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'status': proc.info['status']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            processes['databases'] = db_processes
            
            # Determine status
            status = 'healthy'
            if not lightrag_processes:
                status = 'critical'
            elif any(proc['status'] != 'running' for proc in lightrag_processes):
                status = 'warning'
            
            return {
                'status': status,
                'processes': processes,
                'total_processes': len(psutil.pids())
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def diagnose_file_system(self) -> Dict[str, Any]:
        """Check file system status"""
        try:
            required_paths = [
                'data/lightrag_kg',
                'data/lightrag_vectors',
                'data/lightrag_cache',
                'papers',
                'logs',
                'config'
            ]
            
            path_status = {}
            issues = []
            
            for path_str in required_paths:
                path = Path(path_str)
                
                status_info = {
                    'exists': path.exists(),
                    'is_directory': path.is_dir() if path.exists() else False,
                    'readable': os.access(path, os.R_OK) if path.exists() else False,
                    'writable': os.access(path, os.W_OK) if path.exists() else False,
                    'size': 0,
                    'file_count': 0
                }
                
                if path.exists() and path.is_dir():
                    try:
                        files = list(path.rglob('*'))
                        status_info['file_count'] = len([f for f in files if f.is_file()])
                        status_info['size'] = sum(f.stat().st_size for f in files if f.is_file())
                    except Exception as e:
                        status_info['error'] = str(e)
                        issues.append(f"Could not analyze {path}: {e}")
                
                if not status_info['exists']:
                    issues.append(f"Required path missing: {path}")
                elif not status_info['readable'] or not status_info['writable']:
                    issues.append(f"Permission issues with: {path}")
                
                path_status[path_str] = status_info
            
            # Check log files
            log_files = ['logs/lightrag.log', 'logs/lightrag_errors.log']
            for log_file in log_files:
                if Path(log_file).exists():
                    size = Path(log_file).stat().st_size
                    if size > 100 * 1024 * 1024:  # 100MB
                        issues.append(f"Large log file: {log_file} ({size} bytes)")
            
            status = 'critical' if any('missing' in issue for issue in issues) else \
                    'warning' if issues else 'healthy'
            
            return {
                'status': status,
                'paths': path_status,
                'issues': issues
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def diagnose_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity"""
        try:
            databases = {}
            
            # PostgreSQL test
            try:
                result = subprocess.run(
                    ['pg_isready', '-q'],
                    capture_output=True,
                    timeout=10
                )
                databases['postgresql'] = {
                    'status': 'healthy' if result.returncode == 0 else 'unhealthy',
                    'available': result.returncode == 0,
                    'response_time': None
                }
            except subprocess.TimeoutExpired:
                databases['postgresql'] = {
                    'status': 'timeout',
                    'available': False,
                    'error': 'Connection timeout'
                }
            except FileNotFoundError:
                databases['postgresql'] = {
                    'status': 'unknown',
                    'available': False,
                    'error': 'pg_isready not found'
                }
            
            # Neo4j test
            neo4j_password = os.getenv('NEO4J_PASSWORD', '')
            if neo4j_password:
                try:
                    start_time = time.time()
                    result = subprocess.run(
                        ['cypher-shell', '-u', 'neo4j', '-p', neo4j_password, 'RETURN 1'],
                        capture_output=True,
                        timeout=10,
                        input='',
                        text=True
                    )
                    response_time = time.time() - start_time
                    
                    databases['neo4j'] = {
                        'status': 'healthy' if result.returncode == 0 else 'unhealthy',
                        'available': result.returncode == 0,
                        'response_time': response_time
                    }
                except subprocess.TimeoutExpired:
                    databases['neo4j'] = {
                        'status': 'timeout',
                        'available': False,
                        'error': 'Connection timeout'
                    }
                except FileNotFoundError:
                    databases['neo4j'] = {
                        'status': 'unknown',
                        'available': False,
                        'error': 'cypher-shell not found'
                    }
            else:
                databases['neo4j'] = {
                    'status': 'unknown',
                    'available': False,
                    'error': 'NEO4J_PASSWORD not set'
                }
            
            # Determine overall status
            available_dbs = sum(1 for db in databases.values() if db['available'])
            total_dbs = len(databases)
            
            if available_dbs == total_dbs:
                status = 'healthy'
            elif available_dbs > 0:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'databases': databases
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def diagnose_application_health(self) -> Dict[str, Any]:
        """Check application health"""
        try:
            if not REQUESTS_AVAILABLE:
                return {
                    'status': 'unknown',
                    'error': 'requests library not available'
                }
            
            health_checks = {}
            
            # Basic health check
            try:
                start_time = time.time()
                response = requests.get('http://localhost:8000/health', timeout=10)
                response_time = time.time() - start_time
                
                health_checks['basic'] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'data': response.json() if response.status_code == 200 else None
                }
            except requests.exceptions.ConnectionError:
                health_checks['basic'] = {
                    'status': 'down',
                    'error': 'Connection refused'
                }
            except requests.exceptions.Timeout:
                health_checks['basic'] = {
                    'status': 'timeout',
                    'error': 'Request timeout'
                }
            except Exception as e:
                health_checks['basic'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Detailed health check
            try:
                response = requests.get('http://localhost:8000/health/detailed', timeout=10)
                health_checks['detailed'] = {
                    'status': 'healthy' if response.status_code == 200 else 'unhealthy',
                    'status_code': response.status_code,
                    'data': response.json() if response.status_code == 200 else None
                }
            except Exception as e:
                health_checks['detailed'] = {
                    'status': 'error',
                    'error': str(e)
                }
            
            # Metrics endpoint
            try:
                response = requests.get('http://localhost:8000/metrics', timeout=5)
                health_checks['metrics'] = {
                    'status': 'available' if response.status_code == 200 else 'unavailable',
                    'status_code': response.status_code
                }
            except Exception as e:
                health_checks['metrics'] = {
                    'status': 'unavailable',
                    'error': str(e)
                }
            
            # Determine overall status
            if health_checks['basic']['status'] == 'healthy':
                status = 'healthy'
            elif health_checks['basic']['status'] in ['timeout', 'unhealthy']:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'health_checks': health_checks
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def diagnose_log_analysis(self) -> Dict[str, Any]:
        """Analyze log files"""
        try:
            log_files = {
                'main': 'logs/lightrag.log',
                'errors': 'logs/lightrag_errors.log',
                'performance': 'logs/lightrag_performance.log',
                'security': 'logs/lightrag_security.log'
            }
            
            log_analysis = {}
            
            for log_type, log_path in log_files.items():
                path = Path(log_path)
                
                if not path.exists():
                    log_analysis[log_type] = {
                        'status': 'missing',
                        'path': log_path
                    }
                    continue
                
                try:
                    # Basic file info
                    stat = path.stat()
                    
                    # Read recent entries
                    with open(path, 'r') as f:
                        lines = f.readlines()
                    
                    recent_lines = lines[-100:] if len(lines) > 100 else lines
                    
                    # Count error levels
                    error_count = sum(1 for line in recent_lines if 'ERROR' in line)
                    warning_count = sum(1 for line in recent_lines if 'WARNING' in line)
                    
                    log_analysis[log_type] = {
                        'status': 'healthy',
                        'path': log_path,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'total_lines': len(lines),
                        'recent_errors': error_count,
                        'recent_warnings': warning_count,
                        'last_entries': recent_lines[-5:] if recent_lines else []
                    }
                    
                    # Determine status based on error rate
                    if error_count > 10:
                        log_analysis[log_type]['status'] = 'critical'
                    elif error_count > 5 or warning_count > 20:
                        log_analysis[log_type]['status'] = 'warning'
                    
                except Exception as e:
                    log_analysis[log_type] = {
                        'status': 'error',
                        'path': log_path,
                        'error': str(e)
                    }
            
            # Overall log status
            statuses = [info.get('status', 'unknown') for info in log_analysis.values()]
            if 'critical' in statuses:
                overall_status = 'critical'
            elif 'warning' in statuses:
                overall_status = 'warning'
            elif 'error' in statuses:
                overall_status = 'warning'
            else:
                overall_status = 'healthy'
            
            return {
                'status': overall_status,
                'logs': log_analysis
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def diagnose_performance_metrics(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        try:
            metrics = {}
            
            # System performance
            cpu_times = psutil.cpu_times()
            metrics['cpu'] = {
                'user': cpu_times.user,
                'system': cpu_times.system,
                'idle': cpu_times.idle,
                'usage_percent': psutil.cpu_percent(interval=1)
            }
            
            # Memory performance
            memory = psutil.virtual_memory()
            metrics['memory'] = {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'cached': getattr(memory, 'cached', 0),
                'buffers': getattr(memory, 'buffers', 0)
            }
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                metrics['disk_io'] = {
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count,
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_time': disk_io.read_time,
                    'write_time': disk_io.write_time
                }
            
            # Network I/O
            network_io = psutil.net_io_counters()
            if network_io:
                metrics['network_io'] = {
                    'bytes_sent': network_io.bytes_sent,
                    'bytes_recv': network_io.bytes_recv,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv,
                    'errin': network_io.errin,
                    'errout': network_io.errout,
                    'dropin': network_io.dropin,
                    'dropout': network_io.dropout
                }
            
            # Performance score calculation
            performance_score = 100
            
            if metrics['cpu']['usage_percent'] > 80:
                performance_score -= 20
            elif metrics['cpu']['usage_percent'] > 60:
                performance_score -= 10
            
            if metrics['memory']['percent'] > 85:
                performance_score -= 20
            elif metrics['memory']['percent'] > 70:
                performance_score -= 10
            
            status = 'healthy'
            if performance_score < 60:
                status = 'critical'
            elif performance_score < 80:
                status = 'warning'
            
            return {
                'status': status,
                'performance_score': performance_score,
                'metrics': metrics
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _test_connection(self, host: str, port: int, timeout: int = 5) -> Dict[str, Any]:
        """Test network connection to host:port"""
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            response_time = time.time() - start_time
            sock.close()
            
            return {
                'status': 'success' if result == 0 else 'failed',
                'host': host,
                'port': port,
                'response_time': response_time,
                'error_code': result if result != 0 else None
            }
        except Exception as e:
            return {
                'status': 'error',
                'host': host,
                'port': port,
                'error': str(e)
            }
    
    def _test_database_connectivity(self) -> Dict[str, Any]:
        """Test database connectivity"""
        results = {}
        
        # Test PostgreSQL
        results['postgresql'] = self._test_connection('localhost', 5432)
        
        # Test Neo4j
        results['neo4j'] = self._test_connection('localhost', 7687)
        
        # Test Redis (if configured)
        results['redis'] = self._test_connection('localhost', 6379)
        
        return results
    
    def _test_external_connectivity(self) -> Dict[str, Any]:
        """Test external connectivity"""
        external_hosts = [
            ('google.com', 80),
            ('github.com', 443),
            ('api.groq.com', 443)
        ]
        
        results = {}
        for host, port in external_hosts:
            results[host] = self._test_connection(host, port)
        
        return results
    
    def _get_system_uptime(self) -> Optional[float]:
        """Get system uptime in seconds"""
        try:
            return time.time() - psutil.boot_time()
        except Exception:
            return None
    
    def _calculate_health_score(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall health score"""
        scores = {
            'system_info': 10,
            'resource_usage': 20,
            'network_connectivity': 15,
            'process_status': 20,
            'file_system': 10,
            'database_connectivity': 15,
            'application_health': 20,
            'log_analysis': 5,
            'performance_metrics': 15
        }
        
        total_score = 0
        max_score = sum(scores.values())
        
        for category, max_points in scores.items():
            if category in diagnostics:
                status = diagnostics[category].get('status', 'error')
                
                if status in ['success', 'healthy']:
                    points = max_points
                elif status in ['warning']:
                    points = max_points * 0.7
                elif status in ['critical', 'unhealthy']:
                    points = max_points * 0.3
                else:
                    points = 0
                
                total_score += points
        
        percentage = (total_score / max_score) * 100
        
        if percentage >= 90:
            overall_status = 'excellent'
        elif percentage >= 80:
            overall_status = 'good'
        elif percentage >= 70:
            overall_status = 'fair'
        elif percentage >= 50:
            overall_status = 'poor'
        else:
            overall_status = 'critical'
        
        return {
            'score': round(percentage, 1),
            'status': overall_status,
            'max_score': max_score,
            'actual_score': round(total_score, 1)
        }
    
    def generate_report(self, results: Dict[str, Any], format_type: str = 'human') -> str:
        """Generate diagnostic report"""
        if format_type == 'json':
            return json.dumps(results, indent=2, default=str)
        
        # Human-readable report
        report = []
        report.append("=" * 80)
        report.append("LightRAG System Diagnostics Report")
        report.append("=" * 80)
        report.append(f"Generated: {results['timestamp']}")
        report.append(f"Duration: {results['duration']:.2f} seconds")
        report.append(f"Overall Health Score: {results['overall_health']['score']}/100 ({results['overall_health']['status'].upper()})")
        report.append("")
        
        # Summary of each category
        for category, data in results['diagnostics'].items():
            status = data.get('status', 'unknown').upper()
            status_icon = {
                'SUCCESS': '✓', 'HEALTHY': '✓', 'EXCELLENT': '✓', 'GOOD': '✓',
                'WARNING': '⚠', 'FAIR': '⚠',
                'CRITICAL': '✗', 'ERROR': '✗', 'UNHEALTHY': '✗', 'POOR': '✗',
                'UNKNOWN': '?', 'SKIPPED': '-'
            }.get(status, '?')
            
            report.append(f"{status_icon} {category.replace('_', ' ').title()}: {status}")
            
            # Add specific details for critical issues
            if status in ['CRITICAL', 'ERROR', 'UNHEALTHY']:
                if 'error' in data:
                    report.append(f"    Error: {data['error']}")
                if 'issues' in data:
                    for issue in data['issues'][:3]:  # Show first 3 issues
                        report.append(f"    Issue: {issue}")
        
        report.append("")
        report.append("Recommendations:")
        
        # Generate recommendations based on issues
        recommendations = self._generate_recommendations(results['diagnostics'])
        for rec in recommendations:
            report.append(f"• {rec}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _generate_recommendations(self, diagnostics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on diagnostic results"""
        recommendations = []
        
        # Resource usage recommendations
        if 'resource_usage' in diagnostics:
            resource_data = diagnostics['resource_usage']
            if resource_data.get('status') in ['warning', 'critical']:
                if 'cpu' in resource_data and resource_data['cpu'].get('usage_percent', 0) > 80:
                    recommendations.append("High CPU usage detected. Consider reducing batch sizes or concurrent requests.")
                
                if 'memory' in resource_data and resource_data['memory']['virtual'].get('percent', 0) > 85:
                    recommendations.append("High memory usage detected. Consider restarting the application or increasing system memory.")
        
        # Process status recommendations
        if 'process_status' in diagnostics:
            process_data = diagnostics['process_status']
            if process_data.get('status') == 'critical':
                recommendations.append("LightRAG application is not running. Check logs and restart the service.")
        
        # Database connectivity recommendations
        if 'database_connectivity' in diagnostics:
            db_data = diagnostics['database_connectivity']
            if db_data.get('status') in ['warning', 'critical']:
                recommendations.append("Database connectivity issues detected. Check database services and network connectivity.")
        
        # File system recommendations
        if 'file_system' in diagnostics:
            fs_data = diagnostics['file_system']
            if fs_data.get('issues'):
                recommendations.append("File system issues detected. Check directory permissions and disk space.")
        
        # Log analysis recommendations
        if 'log_analysis' in diagnostics:
            log_data = diagnostics['log_analysis']
            if log_data.get('status') in ['warning', 'critical']:
                recommendations.append("High error rate in logs detected. Review error logs for specific issues.")
        
        # Application health recommendations
        if 'application_health' in diagnostics:
            app_data = diagnostics['application_health']
            if app_data.get('status') in ['warning', 'critical']:
                recommendations.append("Application health issues detected. Check application logs and restart if necessary.")
        
        if not recommendations:
            recommendations.append("System appears to be healthy. Continue regular monitoring.")
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description="LightRAG System Diagnostics")
    parser.add_argument('--format', choices=['human', 'json'], default='human', help='Output format')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--category', choices=[
        'system_info', 'resource_usage', 'network_connectivity', 'process_status',
        'file_system', 'database_connectivity', 'application_health', 'log_analysis',
        'performance_metrics'
    ], help='Run specific diagnostic category only')
    
    args = parser.parse_args()
    
    diagnostics = SystemDiagnostics(args.verbose)
    
    if args.category:
        # Run specific category
        method_name = f"diagnose_{args.category}"
        if hasattr(diagnostics, method_name):
            method = getattr(diagnostics, method_name)
            result = {
                'timestamp': datetime.now().isoformat(),
                'category': args.category,
                'result': method()
            }
        else:
            print(f"Unknown category: {args.category}", file=sys.stderr)
            sys.exit(1)
    else:
        # Run full diagnostics
        result = diagnostics.run_full_diagnostics()
    
    # Generate output
    if args.format == 'json':
        output = json.dumps(result, indent=2, default=str)
    else:
        if args.category:
            output = json.dumps(result, indent=2, default=str)
        else:
            output = diagnostics.generate_report(result, args.format)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Diagnostics report written to: {args.output}")
    else:
        print(output)
    
    # Exit with appropriate code based on health score
    if not args.category and 'overall_health' in result:
        score = result['overall_health']['score']
        if score < 50:
            sys.exit(2)  # Critical
        elif score < 80:
            sys.exit(1)  # Warning
        else:
            sys.exit(0)  # Healthy

if __name__ == "__main__":
    main()