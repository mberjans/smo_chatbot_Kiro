#!/usr/bin/env python3
"""
Comprehensive health check script for LightRAG integration
"""

import asyncio
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import requests
import psutil
import subprocess

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from lightrag_integration.monitoring import HealthMonitor
    from lightrag_integration.config.settings import LightRAGConfig
except ImportError as e:
    print(f"Warning: Could not import LightRAG modules: {e}")
    HealthMonitor = None
    LightRAGConfig = None

class HealthChecker:
    """Comprehensive health checker for LightRAG system"""
    
    def __init__(self, config_path: str = None, verbose: bool = False):
        self.config_path = config_path
        self.verbose = verbose
        self.results = {}
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage"""
        self.logger.info("Checking system resources...")
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage('.')
            
            # Load average (Unix-like systems only)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            # Process information
            process_info = None
            try:
                # Find LightRAG process
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    if any('main.py' in str(cmd) for cmd in proc.info['cmdline'] or []):
                        process_info = {
                            'pid': proc.info['pid'],
                            'memory_percent': proc.memory_percent(),
                            'cpu_percent': proc.cpu_percent(),
                            'status': proc.status()
                        }
                        break
            except Exception as e:
                self.logger.warning(f"Could not get process info: {e}")
            
            result = {
                'status': 'healthy',
                'cpu_percent': cpu_percent,
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'disk': {
                    'total': disk.total,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'load_average': load_avg,
                'process': process_info
            }
            
            # Determine overall status
            if cpu_percent > 90 or memory.percent > 95 or (disk.used / disk.total) > 0.95:
                result['status'] = 'critical'
            elif cpu_percent > 80 or memory.percent > 85 or (disk.used / disk.total) > 0.85:
                result['status'] = 'warning'
            
            return result
            
        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def check_application_health(self, host: str = 'localhost', port: int = 8000) -> Dict[str, Any]:
        """Check application health via HTTP endpoints"""
        self.logger.info(f"Checking application health at {host}:{port}...")
        
        try:
            base_url = f"http://{host}:{port}"
            
            # Basic health check
            response = requests.get(f"{base_url}/health", timeout=10)
            if response.status_code != 200:
                return {
                    'status': 'unhealthy',
                    'error': f"Health endpoint returned {response.status_code}"
                }
            
            health_data = response.json()
            
            # Detailed health check
            detailed_response = requests.get(f"{base_url}/health/detailed", timeout=10)
            detailed_data = {}
            if detailed_response.status_code == 200:
                detailed_data = detailed_response.json()
            
            # Metrics check
            metrics_available = False
            try:
                metrics_response = requests.get(f"{base_url}/metrics", timeout=5)
                metrics_available = metrics_response.status_code == 200
            except:
                pass
            
            return {
                'status': health_data.get('status', 'unknown'),
                'basic_health': health_data,
                'detailed_health': detailed_data,
                'metrics_available': metrics_available,
                'response_time': response.elapsed.total_seconds()
            }
            
        except requests.exceptions.ConnectionError:
            return {
                'status': 'down',
                'error': 'Connection refused - application may not be running'
            }
        except requests.exceptions.Timeout:
            return {
                'status': 'timeout',
                'error': 'Request timed out - application may be overloaded'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def check_database_connections(self) -> Dict[str, Any]:
        """Check database connectivity"""
        self.logger.info("Checking database connections...")
        
        results = {}
        
        # PostgreSQL check
        try:
            result = subprocess.run(
                ['pg_isready', '-q'],
                capture_output=True,
                timeout=10
            )
            results['postgresql'] = {
                'status': 'healthy' if result.returncode == 0 else 'unhealthy',
                'available': result.returncode == 0
            }
        except subprocess.TimeoutExpired:
            results['postgresql'] = {
                'status': 'timeout',
                'available': False,
                'error': 'Connection timeout'
            }
        except FileNotFoundError:
            results['postgresql'] = {
                'status': 'unknown',
                'available': False,
                'error': 'pg_isready command not found'
            }
        except Exception as e:
            results['postgresql'] = {
                'status': 'error',
                'available': False,
                'error': str(e)
            }
        
        # Neo4j check
        try:
            neo4j_password = os.getenv('NEO4J_PASSWORD', '')
            if neo4j_password:
                result = subprocess.run(
                    ['cypher-shell', '-u', 'neo4j', '-p', neo4j_password, 'RETURN 1'],
                    capture_output=True,
                    timeout=10,
                    input='',
                    text=True
                )
                results['neo4j'] = {
                    'status': 'healthy' if result.returncode == 0 else 'unhealthy',
                    'available': result.returncode == 0
                }
            else:
                results['neo4j'] = {
                    'status': 'unknown',
                    'available': False,
                    'error': 'NEO4J_PASSWORD not set'
                }
        except subprocess.TimeoutExpired:
            results['neo4j'] = {
                'status': 'timeout',
                'available': False,
                'error': 'Connection timeout'
            }
        except FileNotFoundError:
            results['neo4j'] = {
                'status': 'unknown',
                'available': False,
                'error': 'cypher-shell command not found'
            }
        except Exception as e:
            results['neo4j'] = {
                'status': 'error',
                'available': False,
                'error': str(e)
            }
        
        return results
    
    def check_file_system(self) -> Dict[str, Any]:
        """Check file system permissions and directories"""
        self.logger.info("Checking file system...")
        
        required_dirs = [
            'data/lightrag_kg',
            'data/lightrag_vectors',
            'data/lightrag_cache',
            'papers',
            'logs'
        ]
        
        results = {
            'status': 'healthy',
            'directories': {}
        }
        
        for dir_path in required_dirs:
            path = Path(dir_path)
            
            dir_info = {
                'exists': path.exists(),
                'is_directory': path.is_dir() if path.exists() else False,
                'readable': False,
                'writable': False,
                'size': 0
            }
            
            if path.exists():
                try:
                    dir_info['readable'] = os.access(path, os.R_OK)
                    dir_info['writable'] = os.access(path, os.W_OK)
                    
                    # Calculate directory size
                    if path.is_dir():
                        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        dir_info['size'] = total_size
                        dir_info['file_count'] = len(list(path.rglob('*')))
                        
                except Exception as e:
                    dir_info['error'] = str(e)
                    results['status'] = 'warning'
            else:
                results['status'] = 'warning'
            
            results['directories'][dir_path] = dir_info
        
        return results
    
    def check_configuration(self) -> Dict[str, Any]:
        """Check configuration validity"""
        self.logger.info("Checking configuration...")
        
        try:
            if LightRAGConfig is None:
                return {
                    'status': 'unknown',
                    'error': 'LightRAGConfig not available'
                }
            
            # Try to load configuration
            config = LightRAGConfig()
            
            # Check required environment variables
            required_vars = [
                'DATABASE_URL',
                'NEO4J_PASSWORD',
                'GROQ_API_KEY'
            ]
            
            missing_vars = []
            for var in required_vars:
                if not os.getenv(var):
                    missing_vars.append(var)
            
            # Check configuration paths
            paths_exist = {
                'knowledge_graph_path': Path(config.knowledge_graph_path).exists(),
                'vector_store_path': Path(config.vector_store_path).exists(),
                'cache_directory': Path(config.cache_directory).exists()
            }
            
            status = 'healthy'
            if missing_vars:
                status = 'warning'
            
            return {
                'status': status,
                'missing_environment_variables': missing_vars,
                'configuration_paths': paths_exist,
                'batch_size': config.batch_size,
                'max_concurrent_requests': config.max_concurrent_requests,
                'cache_ttl_seconds': config.cache_ttl_seconds
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def check_lightrag_component(self) -> Dict[str, Any]:
        """Check LightRAG component health"""
        self.logger.info("Checking LightRAG component...")
        
        try:
            if HealthMonitor is None or LightRAGConfig is None:
                return {
                    'status': 'unknown',
                    'error': 'LightRAG modules not available'
                }
            
            config = LightRAGConfig()
            monitor = HealthMonitor(config)
            
            # Get health status
            health = await monitor.get_health_status()
            
            # Get performance metrics
            metrics = await monitor.get_performance_metrics()
            
            return {
                'status': health.status,
                'components': health.components,
                'metrics': metrics,
                'uptime': health.uptime,
                'last_check': health.last_check.isoformat() if health.last_check else None
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def run_comprehensive_check(self, host: str = 'localhost', port: int = 8000) -> Dict[str, Any]:
        """Run all health checks"""
        self.logger.info("Starting comprehensive health check...")
        
        start_time = time.time()
        
        # Run all checks
        self.results = {
            'timestamp': time.time(),
            'system_resources': self.check_system_resources(),
            'application_health': self.check_application_health(host, port),
            'database_connections': self.check_database_connections(),
            'file_system': self.check_file_system(),
            'configuration': self.check_configuration()
        }
        
        # Run async LightRAG component check
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.results['lightrag_component'] = loop.run_until_complete(
                self.check_lightrag_component()
            )
            loop.close()
        except Exception as e:
            self.results['lightrag_component'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Calculate overall status
        statuses = []
        for check_name, check_result in self.results.items():
            if isinstance(check_result, dict) and 'status' in check_result:
                statuses.append(check_result['status'])
            elif isinstance(check_result, dict):
                # For nested results like database_connections
                for sub_result in check_result.values():
                    if isinstance(sub_result, dict) and 'status' in sub_result:
                        statuses.append(sub_result['status'])
        
        # Determine overall status
        if 'error' in statuses or 'critical' in statuses:
            overall_status = 'critical'
        elif 'unhealthy' in statuses or 'down' in statuses:
            overall_status = 'unhealthy'
        elif 'warning' in statuses or 'timeout' in statuses:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        self.results['overall_status'] = overall_status
        self.results['check_duration'] = time.time() - start_time
        
        self.logger.info(f"Health check completed in {self.results['check_duration']:.2f}s")
        self.logger.info(f"Overall status: {overall_status}")
        
        return self.results
    
    def print_results(self, format_type: str = 'human'):
        """Print health check results"""
        if format_type == 'json':
            print(json.dumps(self.results, indent=2, default=str))
            return
        
        # Human-readable format
        print("=" * 60)
        print("LightRAG Integration Health Check Report")
        print("=" * 60)
        print(f"Timestamp: {time.ctime(self.results['timestamp'])}")
        print(f"Overall Status: {self.results['overall_status'].upper()}")
        print(f"Check Duration: {self.results['check_duration']:.2f}s")
        print()
        
        # System Resources
        print("System Resources:")
        sys_res = self.results['system_resources']
        if sys_res['status'] != 'error':
            print(f"  Status: {sys_res['status']}")
            print(f"  CPU Usage: {sys_res['cpu_percent']:.1f}%")
            print(f"  Memory Usage: {sys_res['memory']['percent']:.1f}%")
            print(f"  Disk Usage: {sys_res['disk']['percent']:.1f}%")
            if sys_res.get('process'):
                proc = sys_res['process']
                print(f"  Process PID: {proc['pid']}")
                print(f"  Process Memory: {proc['memory_percent']:.1f}%")
                print(f"  Process CPU: {proc['cpu_percent']:.1f}%")
        else:
            print(f"  Error: {sys_res['error']}")
        print()
        
        # Application Health
        print("Application Health:")
        app_health = self.results['application_health']
        print(f"  Status: {app_health['status']}")
        if 'response_time' in app_health:
            print(f"  Response Time: {app_health['response_time']:.3f}s")
        if 'metrics_available' in app_health:
            print(f"  Metrics Available: {app_health['metrics_available']}")
        if 'error' in app_health:
            print(f"  Error: {app_health['error']}")
        print()
        
        # Database Connections
        print("Database Connections:")
        db_conns = self.results['database_connections']
        for db_name, db_info in db_conns.items():
            print(f"  {db_name.title()}: {db_info['status']} ({'Available' if db_info['available'] else 'Unavailable'})")
            if 'error' in db_info:
                print(f"    Error: {db_info['error']}")
        print()
        
        # File System
        print("File System:")
        fs_info = self.results['file_system']
        print(f"  Status: {fs_info['status']}")
        for dir_name, dir_info in fs_info['directories'].items():
            status_icon = "✓" if dir_info['exists'] and dir_info['readable'] and dir_info['writable'] else "✗"
            print(f"  {status_icon} {dir_name}: {'OK' if dir_info['exists'] else 'Missing'}")
            if dir_info['exists'] and 'file_count' in dir_info:
                print(f"    Files: {dir_info['file_count']}, Size: {dir_info['size']} bytes")
        print()
        
        # Configuration
        print("Configuration:")
        config_info = self.results['configuration']
        print(f"  Status: {config_info['status']}")
        if config_info.get('missing_environment_variables'):
            print(f"  Missing Variables: {', '.join(config_info['missing_environment_variables'])}")
        if 'batch_size' in config_info:
            print(f"  Batch Size: {config_info['batch_size']}")
            print(f"  Max Concurrent: {config_info['max_concurrent_requests']}")
            print(f"  Cache TTL: {config_info['cache_ttl_seconds']}s")
        print()
        
        # LightRAG Component
        print("LightRAG Component:")
        component_info = self.results['lightrag_component']
        print(f"  Status: {component_info['status']}")
        if 'uptime' in component_info:
            print(f"  Uptime: {component_info['uptime']:.2f}s")
        if 'components' in component_info:
            for comp_name, comp_status in component_info['components'].items():
                print(f"  {comp_name}: {comp_status}")
        if 'error' in component_info:
            print(f"  Error: {component_info['error']}")
        
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="LightRAG Health Check")
    parser.add_argument('--host', default='localhost', help='Application host')
    parser.add_argument('--port', type=int, default=8000, help='Application port')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--format', choices=['human', 'json'], default='human', help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', help='Output file path')
    
    args = parser.parse_args()
    
    # Import os here to avoid issues with earlier imports
    import os
    
    checker = HealthChecker(args.config, args.verbose)
    results = checker.run_comprehensive_check(args.host, args.port)
    
    if args.output:
        with open(args.output, 'w') as f:
            if args.format == 'json':
                json.dump(results, f, indent=2, default=str)
            else:
                # Redirect stdout to file for human format
                import sys
                old_stdout = sys.stdout
                sys.stdout = f
                checker.print_results(args.format)
                sys.stdout = old_stdout
    else:
        checker.print_results(args.format)
    
    # Exit with appropriate code
    overall_status = results['overall_status']
    if overall_status in ['critical', 'error']:
        sys.exit(2)
    elif overall_status in ['unhealthy', 'warning']:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()