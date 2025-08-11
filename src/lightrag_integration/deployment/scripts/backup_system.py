#!/usr/bin/env python3
"""
Comprehensive backup system for LightRAG integration
"""

import os
import sys
import json
import shutil
import tarfile
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import tempfile

class BackupManager:
    """Manages backups for LightRAG system"""
    
    def __init__(self, backup_dir: str = "/backups", retention_days: int = 30):
        self.backup_dir = Path(backup_dir)
        self.retention_days = retention_days
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Backup components
        self.components = {
            'data': {
                'paths': ['data/lightrag_kg', 'data/lightrag_vectors', 'data/lightrag_cache'],
                'required': True
            },
            'papers': {
                'paths': ['papers'],
                'required': False
            },
            'config': {
                'paths': ['config', '.env'],
                'required': True
            },
            'logs': {
                'paths': ['logs'],
                'required': False
            },
            'database': {
                'type': 'postgresql',
                'required': True
            },
            'neo4j': {
                'type': 'neo4j',
                'required': True
            }
        }
    
    def create_backup(self, backup_type: str = 'full', components: List[str] = None) -> Dict[str, any]:
        """Create a backup of specified components"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"lightrag_backup_{backup_type}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        
        self.logger.info(f"Creating {backup_type} backup: {backup_name}")
        
        try:
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Create backup manifest
            manifest = {
                'backup_name': backup_name,
                'backup_type': backup_type,
                'timestamp': timestamp,
                'created_at': datetime.now().isoformat(),
                'components': {},
                'status': 'in_progress'
            }
            
            # Determine which components to backup
            if components is None:
                components = list(self.components.keys())
            
            # Backup each component
            for component_name in components:
                if component_name not in self.components:
                    self.logger.warning(f"Unknown component: {component_name}")
                    continue
                
                self.logger.info(f"Backing up component: {component_name}")
                
                try:
                    if component_name == 'database':
                        result = self._backup_postgresql(backup_path)
                    elif component_name == 'neo4j':
                        result = self._backup_neo4j(backup_path)
                    else:
                        result = self._backup_files(component_name, backup_path)
                    
                    manifest['components'][component_name] = result
                    
                except Exception as e:
                    self.logger.error(f"Failed to backup {component_name}: {e}")
                    manifest['components'][component_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Save manifest
            manifest['status'] = 'completed'
            manifest_path = backup_path / 'manifest.json'
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            # Update last backup reference
            last_backup_file = Path('.last_backup')
            with open(last_backup_file, 'w') as f:
                f.write(str(backup_path))
            
            self.logger.info(f"Backup completed: {backup_path}")
            
            return {
                'status': 'success',
                'backup_path': str(backup_path),
                'manifest': manifest
            }
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _backup_files(self, component_name: str, backup_path: Path) -> Dict[str, any]:
        """Backup file-based components"""
        component = self.components[component_name]
        paths = component['paths']
        
        backed_up_paths = []
        total_size = 0
        
        for path_str in paths:
            path = Path(path_str)
            
            if not path.exists():
                if component['required']:
                    raise FileNotFoundError(f"Required path not found: {path}")
                else:
                    self.logger.warning(f"Optional path not found: {path}")
                    continue
            
            # Create tar archive for this path
            archive_name = f"{component_name}_{path.name}.tar.gz"
            archive_path = backup_path / archive_name
            
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(path, arcname=path.name)
            
            size = archive_path.stat().st_size
            total_size += size
            
            backed_up_paths.append({
                'path': str(path),
                'archive': archive_name,
                'size': size
            })
            
            self.logger.info(f"Archived {path} -> {archive_name} ({size} bytes)")
        
        return {
            'status': 'success',
            'paths': backed_up_paths,
            'total_size': total_size,
            'archive_count': len(backed_up_paths)
        }
    
    def _backup_postgresql(self, backup_path: Path) -> Dict[str, any]:
        """Backup PostgreSQL database"""
        database_name = os.getenv('DATABASE_NAME', 'clinical_metabolomics_oracle')
        backup_file = backup_path / 'postgresql_backup.sql'
        
        try:
            # Use pg_dump to create backup
            cmd = ['pg_dump', database_name]
            
            with open(backup_file, 'w') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stderr
                )
            
            size = backup_file.stat().st_size
            
            return {
                'status': 'success',
                'backup_file': 'postgresql_backup.sql',
                'size': size,
                'database': database_name
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("PostgreSQL backup timed out")
        except subprocess.CalledProcessError as e:
            raise Exception(f"PostgreSQL backup failed: {e.stderr}")
        except Exception as e:
            raise Exception(f"PostgreSQL backup error: {e}")
    
    def _backup_neo4j(self, backup_path: Path) -> Dict[str, any]:
        """Backup Neo4j database"""
        backup_file = backup_path / 'neo4j_backup.dump'
        
        try:
            # Use neo4j-admin dump to create backup
            cmd = [
                'neo4j-admin', 'dump',
                '--database=neo4j',
                f'--to={backup_file}'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stderr
                )
            
            size = backup_file.stat().st_size if backup_file.exists() else 0
            
            return {
                'status': 'success',
                'backup_file': 'neo4j_backup.dump',
                'size': size,
                'database': 'neo4j'
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("Neo4j backup timed out")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Neo4j backup failed: {e.stderr}")
        except FileNotFoundError:
            raise Exception("neo4j-admin command not found")
        except Exception as e:
            raise Exception(f"Neo4j backup error: {e}")
    
    def restore_backup(self, backup_path: str, components: List[str] = None, dry_run: bool = False) -> Dict[str, any]:
        """Restore from backup"""
        backup_dir = Path(backup_path)
        
        if not backup_dir.exists():
            raise FileNotFoundError(f"Backup directory not found: {backup_dir}")
        
        # Load manifest
        manifest_path = backup_dir / 'manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(f"Backup manifest not found: {manifest_path}")
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        self.logger.info(f"Restoring backup: {manifest['backup_name']}")
        
        if dry_run:
            self.logger.info("DRY RUN: No actual restoration will be performed")
        
        # Determine which components to restore
        if components is None:
            components = list(manifest['components'].keys())
        
        restore_results = {}
        
        for component_name in components:
            if component_name not in manifest['components']:
                self.logger.warning(f"Component not found in backup: {component_name}")
                continue
            
            self.logger.info(f"Restoring component: {component_name}")
            
            try:
                if dry_run:
                    restore_results[component_name] = {
                        'status': 'dry_run',
                        'message': 'Would restore this component'
                    }
                else:
                    if component_name == 'database':
                        result = self._restore_postgresql(backup_dir)
                    elif component_name == 'neo4j':
                        result = self._restore_neo4j(backup_dir)
                    else:
                        result = self._restore_files(component_name, backup_dir)
                    
                    restore_results[component_name] = result
                
            except Exception as e:
                self.logger.error(f"Failed to restore {component_name}: {e}")
                restore_results[component_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return {
            'status': 'success',
            'backup_name': manifest['backup_name'],
            'restore_results': restore_results,
            'dry_run': dry_run
        }
    
    def _restore_files(self, component_name: str, backup_dir: Path) -> Dict[str, any]:
        """Restore file-based components"""
        component = self.components[component_name]
        paths = component['paths']
        
        restored_paths = []
        
        for path_str in paths:
            path = Path(path_str)
            archive_name = f"{component_name}_{path.name}.tar.gz"
            archive_path = backup_dir / archive_name
            
            if not archive_path.exists():
                self.logger.warning(f"Archive not found: {archive_name}")
                continue
            
            # Remove existing path if it exists
            if path.exists():
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            
            # Extract archive
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(path.parent)
            
            restored_paths.append(str(path))
            self.logger.info(f"Restored {path} from {archive_name}")
        
        return {
            'status': 'success',
            'restored_paths': restored_paths
        }
    
    def _restore_postgresql(self, backup_dir: Path) -> Dict[str, any]:
        """Restore PostgreSQL database"""
        backup_file = backup_dir / 'postgresql_backup.sql'
        
        if not backup_file.exists():
            raise FileNotFoundError("PostgreSQL backup file not found")
        
        database_name = os.getenv('DATABASE_NAME', 'clinical_metabolomics_oracle')
        
        try:
            # Restore database
            cmd = ['psql', database_name]
            
            with open(backup_file, 'r') as f:
                result = subprocess.run(
                    cmd,
                    stdin=f,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes timeout
                )
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stderr
                )
            
            return {
                'status': 'success',
                'database': database_name,
                'message': 'PostgreSQL database restored successfully'
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("PostgreSQL restore timed out")
        except subprocess.CalledProcessError as e:
            raise Exception(f"PostgreSQL restore failed: {e.stderr}")
    
    def _restore_neo4j(self, backup_dir: Path) -> Dict[str, any]:
        """Restore Neo4j database"""
        backup_file = backup_dir / 'neo4j_backup.dump'
        
        if not backup_file.exists():
            raise FileNotFoundError("Neo4j backup file not found")
        
        try:
            # Restore database
            cmd = [
                'neo4j-admin', 'load',
                '--database=neo4j',
                f'--from={backup_file}'
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stderr
                )
            
            return {
                'status': 'success',
                'database': 'neo4j',
                'message': 'Neo4j database restored successfully'
            }
            
        except subprocess.TimeoutExpired:
            raise Exception("Neo4j restore timed out")
        except subprocess.CalledProcessError as e:
            raise Exception(f"Neo4j restore failed: {e.stderr}")
        except FileNotFoundError:
            raise Exception("neo4j-admin command not found")
    
    def list_backups(self) -> List[Dict[str, any]]:
        """List available backups"""
        backups = []
        
        for backup_dir in self.backup_dir.iterdir():
            if not backup_dir.is_dir():
                continue
            
            manifest_path = backup_dir / 'manifest.json'
            if not manifest_path.exists():
                continue
            
            try:
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                
                # Calculate total size
                total_size = 0
                for file_path in backup_dir.rglob('*'):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                
                backups.append({
                    'name': manifest['backup_name'],
                    'path': str(backup_dir),
                    'type': manifest.get('backup_type', 'unknown'),
                    'created_at': manifest.get('created_at'),
                    'components': list(manifest.get('components', {}).keys()),
                    'size': total_size,
                    'status': manifest.get('status', 'unknown')
                })
                
            except Exception as e:
                self.logger.warning(f"Could not read backup manifest {manifest_path}: {e}")
        
        # Sort by creation time (newest first)
        backups.sort(key=lambda x: x['created_at'], reverse=True)
        
        return backups
    
    def cleanup_old_backups(self, dry_run: bool = False) -> Dict[str, any]:
        """Remove old backups based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        backups = self.list_backups()
        old_backups = []
        
        for backup in backups:
            try:
                created_at = datetime.fromisoformat(backup['created_at'])
                if created_at < cutoff_date:
                    old_backups.append(backup)
            except (ValueError, TypeError):
                self.logger.warning(f"Could not parse date for backup: {backup['name']}")
        
        if not old_backups:
            return {
                'status': 'success',
                'message': 'No old backups to clean up',
                'removed_count': 0,
                'dry_run': dry_run
            }
        
        removed_count = 0
        total_size_freed = 0
        
        for backup in old_backups:
            backup_path = Path(backup['path'])
            
            if dry_run:
                self.logger.info(f"Would remove old backup: {backup['name']}")
            else:
                try:
                    shutil.rmtree(backup_path)
                    self.logger.info(f"Removed old backup: {backup['name']}")
                    removed_count += 1
                    total_size_freed += backup['size']
                except Exception as e:
                    self.logger.error(f"Failed to remove backup {backup['name']}: {e}")
        
        return {
            'status': 'success',
            'removed_count': removed_count,
            'total_size_freed': total_size_freed,
            'dry_run': dry_run
        }
    
    def verify_backup(self, backup_path: str) -> Dict[str, any]:
        """Verify backup integrity"""
        backup_dir = Path(backup_path)
        
        if not backup_dir.exists():
            return {
                'status': 'failed',
                'error': f'Backup directory not found: {backup_dir}'
            }
        
        # Load manifest
        manifest_path = backup_dir / 'manifest.json'
        if not manifest_path.exists():
            return {
                'status': 'failed',
                'error': 'Backup manifest not found'
            }
        
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            return {
                'status': 'failed',
                'error': f'Could not read manifest: {e}'
            }
        
        verification_results = {}
        
        # Verify each component
        for component_name, component_info in manifest.get('components', {}).items():
            if component_info.get('status') != 'success':
                verification_results[component_name] = {
                    'status': 'skipped',
                    'reason': 'Component backup was not successful'
                }
                continue
            
            try:
                if component_name in ['database', 'neo4j']:
                    # Verify database backup files exist
                    if component_name == 'database':
                        backup_file = backup_dir / 'postgresql_backup.sql'
                    else:
                        backup_file = backup_dir / 'neo4j_backup.dump'
                    
                    if backup_file.exists():
                        verification_results[component_name] = {
                            'status': 'valid',
                            'size': backup_file.stat().st_size
                        }
                    else:
                        verification_results[component_name] = {
                            'status': 'invalid',
                            'error': 'Backup file not found'
                        }
                else:
                    # Verify file archives
                    component = self.components.get(component_name, {})
                    paths = component.get('paths', [])
                    
                    valid_archives = 0
                    total_archives = 0
                    
                    for path_str in paths:
                        path = Path(path_str)
                        archive_name = f"{component_name}_{path.name}.tar.gz"
                        archive_path = backup_dir / archive_name
                        
                        total_archives += 1
                        
                        if archive_path.exists():
                            # Test archive integrity
                            try:
                                with tarfile.open(archive_path, 'r:gz') as tar:
                                    tar.getnames()  # This will fail if archive is corrupted
                                valid_archives += 1
                            except Exception:
                                pass
                    
                    verification_results[component_name] = {
                        'status': 'valid' if valid_archives == total_archives else 'invalid',
                        'valid_archives': valid_archives,
                        'total_archives': total_archives
                    }
                    
            except Exception as e:
                verification_results[component_name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        # Overall verification status
        all_valid = all(
            result['status'] in ['valid', 'skipped']
            for result in verification_results.values()
        )
        
        return {
            'status': 'valid' if all_valid else 'invalid',
            'backup_name': manifest['backup_name'],
            'verification_results': verification_results
        }

def main():
    parser = argparse.ArgumentParser(description="LightRAG Backup Manager")
    parser.add_argument('--backup-dir', default='/backups', help='Backup directory')
    parser.add_argument('--retention-days', type=int, default=30, help='Backup retention in days')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create backup command
    create_parser = subparsers.add_parser('create', help='Create backup')
    create_parser.add_argument('--type', choices=['full', 'incremental', 'data-only'], default='full')
    create_parser.add_argument('--components', nargs='+', help='Components to backup')
    
    # Restore backup command
    restore_parser = subparsers.add_parser('restore', help='Restore backup')
    restore_parser.add_argument('backup_path', help='Path to backup directory')
    restore_parser.add_argument('--components', nargs='+', help='Components to restore')
    restore_parser.add_argument('--dry-run', action='store_true', help='Simulate restore')
    
    # List backups command
    list_parser = subparsers.add_parser('list', help='List available backups')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old backups')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Simulate cleanup')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify backup integrity')
    verify_parser.add_argument('backup_path', help='Path to backup directory')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    backup_manager = BackupManager(args.backup_dir, args.retention_days)
    
    try:
        if args.command == 'create':
            result = backup_manager.create_backup(args.type, args.components)
            print(json.dumps(result, indent=2))
            
        elif args.command == 'restore':
            result = backup_manager.restore_backup(args.backup_path, args.components, args.dry_run)
            print(json.dumps(result, indent=2))
            
        elif args.command == 'list':
            backups = backup_manager.list_backups()
            print(json.dumps(backups, indent=2))
            
        elif args.command == 'cleanup':
            result = backup_manager.cleanup_old_backups(args.dry_run)
            print(json.dumps(result, indent=2))
            
        elif args.command == 'verify':
            result = backup_manager.verify_backup(args.backup_path)
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()