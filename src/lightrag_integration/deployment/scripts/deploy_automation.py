#!/usr/bin/env python3
"""
Comprehensive deployment automation for LightRAG integration
"""

import os
import sys
import json
import yaml
import time
import argparse
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import tempfile
import shutil

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

class DeploymentAutomation:
    """Automated deployment orchestrator for LightRAG"""
    
    def __init__(self, environment: str = 'production', config_dir: str = 'config'):
        self.environment = environment
        self.config_dir = Path(config_dir)
        self.deployment_dir = Path(__file__).parent.parent
        self.project_root = self.deployment_dir.parent.parent.parent
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Deployment phases
        self.phases = [
            'pre_deployment_checks',
            'backup_current_system',
            'setup_environment',
            'install_dependencies',
            'configure_system',
            'setup_monitoring',
            'deploy_application',
            'run_health_checks',
            'post_deployment_tasks'
        ]
        
        # Deployment state
        self.deployment_state = {
            'start_time': None,
            'current_phase': None,
            'completed_phases': [],
            'failed_phases': [],
            'rollback_required': False
        }
    
    def deploy(self, skip_phases: List[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        """Execute full deployment"""
        self.logger.info(f"Starting automated deployment for {self.environment} environment")
        
        if dry_run:
            self.logger.info("DRY RUN MODE: No actual changes will be made")
        
        self.deployment_state['start_time'] = datetime.now()
        skip_phases = skip_phases or []
        
        deployment_result = {
            'environment': self.environment,
            'start_time': self.deployment_state['start_time'].isoformat(),
            'phases': {},
            'overall_status': 'in_progress',
            'dry_run': dry_run
        }
        
        try:
            for phase in self.phases:
                if phase in skip_phases:
                    self.logger.info(f"Skipping phase: {phase}")
                    deployment_result['phases'][phase] = {
                        'status': 'skipped',
                        'message': 'Phase skipped by user request'
                    }
                    continue
                
                self.deployment_state['current_phase'] = phase
                self.logger.info(f"Executing phase: {phase}")
                
                try:
                    phase_method = getattr(self, phase)
                    phase_result = phase_method(dry_run)
                    
                    deployment_result['phases'][phase] = phase_result
                    
                    if phase_result['status'] == 'success':
                        self.deployment_state['completed_phases'].append(phase)
                    else:
                        self.deployment_state['failed_phases'].append(phase)
                        if phase_result.get('critical', False):
                            self.logger.error(f"Critical failure in phase {phase}, stopping deployment")
                            self.deployment_state['rollback_required'] = True
                            break
                        else:
                            self.logger.warning(f"Non-critical failure in phase {phase}, continuing")
                
                except Exception as e:
                    self.logger.error(f"Exception in phase {phase}: {e}")
                    deployment_result['phases'][phase] = {
                        'status': 'error',
                        'error': str(e),
                        'critical': True
                    }
                    self.deployment_state['failed_phases'].append(phase)
                    self.deployment_state['rollback_required'] = True
                    break
            
            # Determine overall status
            if self.deployment_state['rollback_required']:
                deployment_result['overall_status'] = 'failed'
                if not dry_run:
                    self.logger.info("Initiating rollback due to critical failures")
                    rollback_result = self.rollback()
                    deployment_result['rollback'] = rollback_result
            elif self.deployment_state['failed_phases']:
                deployment_result['overall_status'] = 'partial_success'
            else:
                deployment_result['overall_status'] = 'success'
            
            deployment_result['end_time'] = datetime.now().isoformat()
            deployment_result['duration'] = (datetime.now() - self.deployment_state['start_time']).total_seconds()
            deployment_result['completed_phases'] = self.deployment_state['completed_phases']
            deployment_result['failed_phases'] = self.deployment_state['failed_phases']
            
            self.logger.info(f"Deployment completed with status: {deployment_result['overall_status']}")
            
            return deployment_result
            
        except Exception as e:
            self.logger.error(f"Deployment failed with exception: {e}")
            deployment_result['overall_status'] = 'error'
            deployment_result['error'] = str(e)
            deployment_result['end_time'] = datetime.now().isoformat()
            return deployment_result
    
    def pre_deployment_checks(self, dry_run: bool = False) -> Dict[str, Any]:
        """Pre-deployment system checks"""
        self.logger.info("Running pre-deployment checks...")
        
        checks = {
            'system_requirements': self._check_system_requirements(),
            'dependencies': self._check_dependencies(),
            'permissions': self._check_permissions(),
            'disk_space': self._check_disk_space(),
            'network_connectivity': self._check_network_connectivity()
        }
        
        failed_checks = [name for name, result in checks.items() 
                        if result.get('status') != 'success']
        
        if failed_checks:
            return {
                'status': 'failed',
                'critical': True,
                'message': f"Pre-deployment checks failed: {', '.join(failed_checks)}",
                'checks': checks
            }
        
        return {
            'status': 'success',
            'message': 'All pre-deployment checks passed',
            'checks': checks
        }
    
    def backup_current_system(self, dry_run: bool = False) -> Dict[str, Any]:
        """Create backup of current system"""
        self.logger.info("Creating system backup...")
        
        if dry_run:
            return {
                'status': 'success',
                'message': 'Backup would be created (dry run)',
                'backup_path': '/tmp/dry_run_backup'
            }
        
        try:
            # Use backup system script
            backup_script = self.deployment_dir / 'scripts' / 'backup_system.py'
            
            cmd = [
                sys.executable, str(backup_script),
                'create', '--type', 'full'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if result.returncode != 0:
                return {
                    'status': 'failed',
                    'critical': False,  # Backup failure is not critical for deployment
                    'error': result.stderr,
                    'message': 'Backup creation failed, but deployment can continue'
                }
            
            backup_info = json.loads(result.stdout)
            
            return {
                'status': 'success',
                'message': 'System backup created successfully',
                'backup_path': backup_info.get('backup_path'),
                'backup_info': backup_info
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'failed',
                'critical': False,
                'error': 'Backup creation timed out',
                'message': 'Backup timed out, but deployment can continue'
            }
        except Exception as e:
            return {
                'status': 'failed',
                'critical': False,
                'error': str(e),
                'message': 'Backup failed, but deployment can continue'
            }
    
    def setup_environment(self, dry_run: bool = False) -> Dict[str, Any]:
        """Setup deployment environment"""
        self.logger.info("Setting up deployment environment...")
        
        tasks = []
        
        # Create virtual environment
        venv_path = self.project_root / 'venv'
        if not venv_path.exists() or not dry_run:
            if not dry_run:
                subprocess.run([sys.executable, '-m', 'venv', str(venv_path)], check=True)
            tasks.append('Created Python virtual environment')
        
        # Create necessary directories
        directories = [
            'data/lightrag_kg',
            'data/lightrag_vectors', 
            'data/lightrag_cache',
            'papers',
            'logs',
            'config'
        ]
        
        for dir_path in directories:
            full_path = self.project_root / dir_path
            if not dry_run:
                full_path.mkdir(parents=True, exist_ok=True)
            tasks.append(f'Created directory: {dir_path}')
        
        # Setup environment file
        env_file = self.project_root / '.env'
        if not env_file.exists():
            env_example = self.project_root / '.env.example'
            if env_example.exists() and not dry_run:
                shutil.copy2(env_example, env_file)
            tasks.append('Created .env file from template')
        
        return {
            'status': 'success',
            'message': 'Environment setup completed',
            'tasks': tasks
        }
    
    def install_dependencies(self, dry_run: bool = False) -> Dict[str, Any]:
        """Install system dependencies"""
        self.logger.info("Installing dependencies...")
        
        tasks = []
        
        # Python dependencies
        if not dry_run:
            venv_python = self.project_root / 'venv' / 'bin' / 'python'
            venv_pip = self.project_root / 'venv' / 'bin' / 'pip'
            
            # Upgrade pip
            subprocess.run([str(venv_pip), 'install', '--upgrade', 'pip'], check=True)
            tasks.append('Upgraded pip')
            
            # Install requirements
            requirements_file = self.project_root / 'requirements.txt'
            if requirements_file.exists():
                subprocess.run([str(venv_pip), 'install', '-r', str(requirements_file)], check=True)
                tasks.append('Installed Python dependencies')
        else:
            tasks.append('Would install Python dependencies')
        
        # Node.js dependencies
        package_json = self.project_root / 'package.json'
        if package_json.exists():
            if not dry_run:
                subprocess.run(['npm', 'ci', '--production'], cwd=self.project_root, check=True)
            tasks.append('Installed Node.js dependencies')
        
        return {
            'status': 'success',
            'message': 'Dependencies installed successfully',
            'tasks': tasks
        }
    
    def configure_system(self, dry_run: bool = False) -> Dict[str, Any]:
        """Configure system settings"""
        self.logger.info("Configuring system...")
        
        tasks = []
        
        # Generate configuration
        config_manager_script = self.deployment_dir / 'config_manager.py'
        
        if not dry_run:
            # Create environment configuration
            cmd = [
                sys.executable, str(config_manager_script),
                'create', self.environment
            ]
            subprocess.run(cmd, check=True)
            tasks.append(f'Created {self.environment} configuration')
            
            # Generate .env file
            config_file = self.config_dir / f'{self.environment}.yaml'
            if config_file.exists():
                cmd = [
                    sys.executable, str(config_manager_script),
                    'generate-env', str(config_file)
                ]
                subprocess.run(cmd, check=True)
                tasks.append('Generated .env file from configuration')
        else:
            tasks.append(f'Would create {self.environment} configuration')
            tasks.append('Would generate .env file')
        
        # Database migrations
        if not dry_run:
            try:
                subprocess.run(['npx', 'prisma', 'generate'], cwd=self.project_root, check=True)
                subprocess.run(['npx', 'prisma', 'migrate', 'deploy'], cwd=self.project_root, check=True)
                tasks.append('Ran database migrations')
            except subprocess.CalledProcessError as e:
                return {
                    'status': 'failed',
                    'critical': True,
                    'error': str(e),
                    'message': 'Database migration failed'
                }
        else:
            tasks.append('Would run database migrations')
        
        return {
            'status': 'success',
            'message': 'System configuration completed',
            'tasks': tasks
        }
    
    def setup_monitoring(self, dry_run: bool = False) -> Dict[str, Any]:
        """Setup monitoring infrastructure"""
        self.logger.info("Setting up monitoring...")
        
        if dry_run:
            return {
                'status': 'success',
                'message': 'Monitoring setup would be configured (dry run)',
                'tasks': ['Would setup Prometheus', 'Would setup Grafana', 'Would setup logging']
            }
        
        try:
            monitoring_script = self.deployment_dir / 'scripts' / 'monitoring_setup.py'
            
            cmd = [
                sys.executable, str(monitoring_script),
                'setup', '--component', 'all',
                '--environment', self.environment
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                return {
                    'status': 'failed',
                    'critical': False,  # Monitoring setup failure is not critical
                    'error': result.stderr,
                    'message': 'Monitoring setup failed, but deployment can continue'
                }
            
            monitoring_info = json.loads(result.stdout)
            
            return {
                'status': 'success',
                'message': 'Monitoring setup completed',
                'monitoring_info': monitoring_info
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'critical': False,
                'error': str(e),
                'message': 'Monitoring setup failed, but deployment can continue'
            }
    
    def deploy_application(self, dry_run: bool = False) -> Dict[str, Any]:
        """Deploy the application"""
        self.logger.info("Deploying application...")
        
        tasks = []
        
        if self.environment == 'production':
            # Setup systemd service
            service_file = Path('/etc/systemd/system/lightrag-oracle.service')
            
            if not dry_run:
                try:
                    # Create service file
                    service_content = self._generate_systemd_service()
                    
                    # Write service file (requires sudo)
                    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                        f.write(service_content)
                        temp_service_file = f.name
                    
                    subprocess.run(['sudo', 'cp', temp_service_file, str(service_file)], check=True)
                    os.unlink(temp_service_file)
                    
                    # Reload systemd and enable service
                    subprocess.run(['sudo', 'systemctl', 'daemon-reload'], check=True)
                    subprocess.run(['sudo', 'systemctl', 'enable', 'lightrag-oracle'], check=True)
                    
                    tasks.append('Created and enabled systemd service')
                    
                except subprocess.CalledProcessError as e:
                    return {
                        'status': 'failed',
                        'critical': True,
                        'error': str(e),
                        'message': 'Failed to setup systemd service'
                    }
            else:
                tasks.append('Would create and enable systemd service')
        
        # Start application
        if not dry_run:
            if self.environment == 'production':
                try:
                    subprocess.run(['sudo', 'systemctl', 'start', 'lightrag-oracle'], check=True)
                    tasks.append('Started LightRAG Oracle service')
                except subprocess.CalledProcessError as e:
                    return {
                        'status': 'failed',
                        'critical': True,
                        'error': str(e),
                        'message': 'Failed to start LightRAG Oracle service'
                    }
            else:
                # For non-production, start in background
                venv_python = self.project_root / 'venv' / 'bin' / 'python'
                main_script = self.project_root / 'src' / 'main.py'
                
                # Start application in background
                with open(self.project_root / 'logs' / 'app.log', 'w') as log_file:
                    process = subprocess.Popen(
                        [str(venv_python), str(main_script)],
                        stdout=log_file,
                        stderr=subprocess.STDOUT,
                        cwd=self.project_root
                    )
                
                # Save PID
                with open(self.project_root / '.app.pid', 'w') as f:
                    f.write(str(process.pid))
                
                tasks.append(f'Started application in background (PID: {process.pid})')
        else:
            tasks.append('Would start LightRAG Oracle application')
        
        return {
            'status': 'success',
            'message': 'Application deployment completed',
            'tasks': tasks
        }
    
    def run_health_checks(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run post-deployment health checks"""
        self.logger.info("Running health checks...")
        
        if dry_run:
            return {
                'status': 'success',
                'message': 'Health checks would be performed (dry run)',
                'checks': {'application': 'would_check', 'database': 'would_check'}
            }
        
        # Wait for application to start
        time.sleep(10)
        
        try:
            health_check_script = self.deployment_dir / 'scripts' / 'health_check.py'
            
            cmd = [
                sys.executable, str(health_check_script),
                '--format', 'json'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode != 0:
                return {
                    'status': 'failed',
                    'critical': True,
                    'error': result.stderr,
                    'message': 'Health checks failed'
                }
            
            health_data = json.loads(result.stdout)
            
            overall_status = health_data.get('overall_status', 'unknown')
            if overall_status in ['critical', 'error']:
                return {
                    'status': 'failed',
                    'critical': True,
                    'health_data': health_data,
                    'message': f'Health checks failed with status: {overall_status}'
                }
            elif overall_status in ['warning', 'unhealthy']:
                return {
                    'status': 'warning',
                    'critical': False,
                    'health_data': health_data,
                    'message': f'Health checks passed with warnings: {overall_status}'
                }
            
            return {
                'status': 'success',
                'message': 'All health checks passed',
                'health_data': health_data
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'critical': True,
                'error': str(e),
                'message': 'Health check execution failed'
            }
    
    def post_deployment_tasks(self, dry_run: bool = False) -> Dict[str, Any]:
        """Post-deployment cleanup and notifications"""
        self.logger.info("Running post-deployment tasks...")
        
        tasks = []
        
        # Cleanup temporary files
        if not dry_run:
            temp_files = list(self.project_root.glob('*.tmp'))
            for temp_file in temp_files:
                temp_file.unlink()
            if temp_files:
                tasks.append(f'Cleaned up {len(temp_files)} temporary files')
        
        # Update deployment record
        deployment_record = {
            'environment': self.environment,
            'deployment_time': datetime.now().isoformat(),
            'version': self._get_version_info(),
            'status': 'completed'
        }
        
        if not dry_run:
            record_file = self.project_root / '.deployment_record.json'
            with open(record_file, 'w') as f:
                json.dump(deployment_record, f, indent=2)
            tasks.append('Updated deployment record')
        else:
            tasks.append('Would update deployment record')
        
        # Send notifications (if configured)
        if not dry_run:
            self._send_deployment_notification('success')
            tasks.append('Sent deployment notification')
        else:
            tasks.append('Would send deployment notification')
        
        return {
            'status': 'success',
            'message': 'Post-deployment tasks completed',
            'tasks': tasks,
            'deployment_record': deployment_record
        }
    
    def rollback(self) -> Dict[str, Any]:
        """Rollback deployment"""
        self.logger.info("Initiating deployment rollback...")
        
        try:
            # Stop current application
            if self.environment == 'production':
                subprocess.run(['sudo', 'systemctl', 'stop', 'lightrag-oracle'], 
                             capture_output=True)
            else:
                # Kill background process
                pid_file = self.project_root / '.app.pid'
                if pid_file.exists():
                    with open(pid_file, 'r') as f:
                        pid = int(f.read().strip())
                    try:
                        os.kill(pid, 15)  # SIGTERM
                    except ProcessLookupError:
                        pass
            
            # Restore from backup
            last_backup_file = self.project_root / '.last_backup'
            if last_backup_file.exists():
                with open(last_backup_file, 'r') as f:
                    backup_path = f.read().strip()
                
                backup_script = self.deployment_dir / 'scripts' / 'backup_system.py'
                cmd = [sys.executable, str(backup_script), 'restore', backup_path]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    return {
                        'status': 'success',
                        'message': 'Rollback completed successfully',
                        'backup_path': backup_path
                    }
                else:
                    return {
                        'status': 'failed',
                        'error': result.stderr,
                        'message': 'Rollback failed'
                    }
            else:
                return {
                    'status': 'failed',
                    'error': 'No backup available for rollback',
                    'message': 'Rollback not possible - no backup found'
                }
                
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'message': 'Rollback failed with exception'
            }
    
    def _check_system_requirements(self) -> Dict[str, Any]:
        """Check system requirements"""
        try:
            # Check Python version
            python_version = sys.version_info
            if python_version < (3, 8):
                return {
                    'status': 'failed',
                    'error': f'Python 3.8+ required, found {python_version.major}.{python_version.minor}'
                }
            
            # Check available memory
            import psutil
            memory = psutil.virtual_memory()
            if memory.total < 4 * 1024 * 1024 * 1024:  # 4GB
                return {
                    'status': 'warning',
                    'message': 'Less than 4GB RAM available'
                }
            
            return {'status': 'success'}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _check_dependencies(self) -> Dict[str, Any]:
        """Check required dependencies"""
        try:
            # Check for required commands
            required_commands = ['node', 'npm', 'pg_isready']
            missing_commands = []
            
            for cmd in required_commands:
                result = subprocess.run(['which', cmd], capture_output=True)
                if result.returncode != 0:
                    missing_commands.append(cmd)
            
            if missing_commands:
                return {
                    'status': 'failed',
                    'error': f'Missing required commands: {", ".join(missing_commands)}'
                }
            
            return {'status': 'success'}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _check_permissions(self) -> Dict[str, Any]:
        """Check file permissions"""
        try:
            # Check write permissions for project directory
            if not os.access(self.project_root, os.W_OK):
                return {
                    'status': 'failed',
                    'error': f'No write permission for project directory: {self.project_root}'
                }
            
            return {'status': 'success'}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space"""
        try:
            import psutil
            disk_usage = psutil.disk_usage(str(self.project_root))
            
            # Check for at least 10GB free space
            if disk_usage.free < 10 * 1024 * 1024 * 1024:  # 10GB
                return {
                    'status': 'failed',
                    'error': f'Insufficient disk space: {disk_usage.free / (1024**3):.1f}GB available, 10GB required'
                }
            
            return {'status': 'success'}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity"""
        try:
            import socket
            
            # Test connection to common services
            test_hosts = [
                ('google.com', 80),
                ('github.com', 443)
            ]
            
            for host, port in test_hosts:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5)
                result = sock.connect_ex((host, port))
                sock.close()
                
                if result != 0:
                    return {
                        'status': 'warning',
                        'message': f'Could not connect to {host}:{port}'
                    }
            
            return {'status': 'success'}
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    def _generate_systemd_service(self) -> str:
        """Generate systemd service file content"""
        return f"""[Unit]
Description=Clinical Metabolomics Oracle with LightRAG
After=network.target postgresql.service neo4j.service
Wants=postgresql.service neo4j.service

[Service]
Type=simple
User={os.getenv('USER', 'oracle')}
Group={os.getenv('USER', 'oracle')}
WorkingDirectory={self.project_root}
Environment=PATH={self.project_root}/venv/bin
ExecStart={self.project_root}/venv/bin/python src/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths={self.project_root}

[Install]
WantedBy=multi-user.target
"""
    
    def _get_version_info(self) -> Dict[str, Any]:
        """Get version information"""
        try:
            # Try to get git commit
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            git_commit = result.stdout.strip() if result.returncode == 0 else 'unknown'
            
            return {
                'git_commit': git_commit,
                'python_version': sys.version,
                'deployment_time': datetime.now().isoformat()
            }
            
        except Exception:
            return {
                'git_commit': 'unknown',
                'python_version': sys.version,
                'deployment_time': datetime.now().isoformat()
            }
    
    def _send_deployment_notification(self, status: str):
        """Send deployment notification"""
        # This would integrate with notification systems like Slack, email, etc.
        # For now, just log the notification
        self.logger.info(f"Deployment notification: {status} deployment to {self.environment}")

def main():
    parser = argparse.ArgumentParser(description="LightRAG Deployment Automation")
    parser.add_argument('--environment', choices=['development', 'staging', 'production'],
                       default='production', help='Deployment environment')
    parser.add_argument('--config-dir', default='config', help='Configuration directory')
    parser.add_argument('--skip-phases', nargs='+', help='Phases to skip')
    parser.add_argument('--dry-run', action='store_true', help='Simulate deployment')
    parser.add_argument('--output', help='Output file for deployment report')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Deploy command
    deploy_parser = subparsers.add_parser('deploy', help='Run full deployment')
    
    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback deployment')
    
    args = parser.parse_args()
    
    if not args.command:
        args.command = 'deploy'  # Default to deploy
    
    automation = DeploymentAutomation(args.environment, args.config_dir)
    
    try:
        if args.command == 'deploy':
            result = automation.deploy(args.skip_phases, args.dry_run)
        elif args.command == 'rollback':
            result = automation.rollback()
        
        # Output result
        output = json.dumps(result, indent=2, default=str)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Deployment report written to: {args.output}")
        else:
            print(output)
        
        # Exit with appropriate code
        if result.get('overall_status') in ['failed', 'error']:
            sys.exit(1)
        elif result.get('overall_status') == 'partial_success':
            sys.exit(2)
        else:
            sys.exit(0)
        
    except Exception as e:
        print(f"Deployment automation failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()