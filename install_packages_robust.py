#!/usr/bin/env python3
"""
Robust package installation script that handles problematic packages
and installs packages without version constraints to prevent conflicts.
"""

import subprocess
import sys
from pathlib import Path

# Packages that commonly have build issues
PROBLEMATIC_PACKAGES = {
    'sentencepiece',  # Has CMake build issues
    'faiss-cpu',      # Large package with potential conflicts
    'faiss-gpu',      # GPU-specific package
}

# Alternative packages for problematic ones
ALTERNATIVES = {
    'sentencepiece': 'tokenizers',  # Use tokenizers instead
}

def read_packages_from_file(file_path):
    """Read package names from file"""
    packages = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                packages.append(line)
    return packages

def install_package(package_name):
    """Install a single package"""
    print(f"🔄 Installing {package_name}...")
    
    try:
        cmd = [sys.executable, '-m', 'pip', 'install', package_name]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {package_name} installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package_name}")
        print(f"   Error: {e.stderr.strip()}")
        return False

def install_packages_batch(packages, batch_size=5):
    """Install packages in batches"""
    successful = []
    failed = []
    
    for i in range(0, len(packages), batch_size):
        batch = packages[i:i + batch_size]
        print(f"\n📦 Installing batch {i//batch_size + 1}: {', '.join(batch)}")
        
        try:
            cmd = [sys.executable, '-m', 'pip', 'install'] + batch
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"✅ Batch installed successfully: {', '.join(batch)}")
            successful.extend(batch)
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Batch installation failed, trying individual packages...")
            # Try installing each package individually
            for pkg in batch:
                if install_package(pkg):
                    successful.append(pkg)
                else:
                    failed.append(pkg)
    
    return successful, failed

def main():
    """Main installation function"""
    
    print("🚀 Robust Package Installation (No Version Constraints)")
    print("=" * 80)
    
    # Read packages from file
    packages_file = Path("packages_no_versions.txt")
    if not packages_file.exists():
        print(f"❌ Package list file not found: {packages_file}")
        return False
    
    packages = read_packages_from_file(packages_file)
    print(f"📋 Found {len(packages)} packages to install")
    
    # Filter out problematic packages and suggest alternatives
    filtered_packages = []
    skipped_packages = []
    
    for pkg in packages:
        if pkg in PROBLEMATIC_PACKAGES:
            skipped_packages.append(pkg)
            if pkg in ALTERNATIVES:
                alternative = ALTERNATIVES[pkg]
                print(f"⚠️  Skipping {pkg}, will try alternative: {alternative}")
                filtered_packages.append(alternative)
            else:
                print(f"⚠️  Skipping problematic package: {pkg}")
        else:
            filtered_packages.append(pkg)
    
    print(f"📦 Installing {len(filtered_packages)} packages (skipped {len(skipped_packages)})")
    print()
    
    # Install packages in batches
    successful, failed = install_packages_batch(filtered_packages)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 INSTALLATION SUMMARY")
    print("=" * 80)
    
    print(f"✅ Successfully installed: {len(successful)} packages")
    for pkg in successful:
        print(f"   • {pkg}")
    
    if failed:
        print(f"\n❌ Failed to install: {len(failed)} packages")
        for pkg in failed:
            print(f"   • {pkg}")
    
    if skipped_packages:
        print(f"\n⚠️  Skipped problematic packages: {len(skipped_packages)}")
        for pkg in skipped_packages:
            print(f"   • {pkg}")
    
    # Try to install failed packages with alternatives
    if failed:
        print(f"\n🔄 Attempting to install failed packages with --no-deps...")
        retry_successful = []
        for pkg in failed:
            try:
                cmd = [sys.executable, '-m', 'pip', 'install', '--no-deps', pkg]
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"✅ {pkg} installed with --no-deps")
                retry_successful.append(pkg)
            except subprocess.CalledProcessError:
                print(f"❌ {pkg} still failed with --no-deps")
        
        if retry_successful:
            successful.extend(retry_successful)
            failed = [pkg for pkg in failed if pkg not in retry_successful]
    
    success_rate = len(successful) / len(filtered_packages) * 100 if filtered_packages else 0
    print(f"\n🎯 Overall success rate: {success_rate:.1f}% ({len(successful)}/{len(filtered_packages)})")
    
    if success_rate >= 80:
        print("🎉 Installation mostly successful!")
        return True
    elif success_rate >= 50:
        print("⚠️  Partial success - some packages may need manual installation")
        return True
    else:
        print("❌ Many packages failed to install - check your environment")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)