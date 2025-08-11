#!/usr/bin/env python3
"""
Extract package names from requirements.txt without versions and install them
to prevent version conflicts in the local environment.
"""

import re
import subprocess
import sys
from pathlib import Path

def extract_package_names(requirements_file):
    """Extract package names from requirements.txt without version constraints"""
    
    packages = []
    
    with open(requirements_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue
        
        # Handle git repositories
        if line.startswith('git+'):
            # Extract package name from git URL
            if 'LightRAG' in line:
                packages.append('lightrag-hku')  # Use the PyPI package instead
            continue
        
        # Remove version constraints (==, >=, <=, >, <, ~=, !=)
        package_name = re.split(r'[><=!~]', line)[0].strip()
        
        # Remove any extra characters
        package_name = package_name.replace(' ', '')
        
        if package_name:
            packages.append(package_name)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_packages = []
    for pkg in packages:
        if pkg not in seen:
            seen.add(pkg)
            unique_packages.append(pkg)
    
    return unique_packages

def install_packages(packages):
    """Install packages using pip"""
    
    print(f"📦 Installing {len(packages)} packages without version constraints...")
    print("=" * 60)
    
    # Create the pip install command
    cmd = [sys.executable, '-m', 'pip', 'install'] + packages
    
    print(f"🔄 Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Installation completed successfully!")
        print()
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed with exit code {e.returncode}")
        print()
        print("STDOUT:")
        print(e.stdout)
        print()
        print("STDERR:")
        print(e.stderr)
        
        return False

def main():
    """Main function"""
    
    print("🚀 Package Installation Without Version Constraints")
    print("=" * 80)
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print(f"❌ Requirements file not found: {requirements_file}")
        return False
    
    # Extract package names
    print("🔍 Extracting package names from requirements.txt...")
    packages = extract_package_names(requirements_file)
    
    print(f"✅ Found {len(packages)} unique packages:")
    for i, pkg in enumerate(packages, 1):
        print(f"  {i:2d}. {pkg}")
    
    print()
    
    # Install packages
    success = install_packages(packages)
    
    if success:
        print("🎉 All packages installed successfully!")
        print()
        print("💡 Note: Packages were installed with their latest compatible versions")
        print("   to avoid version conflicts. You can check installed versions with:")
        print("   pip list")
    else:
        print("❌ Package installation failed. Check the error messages above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)