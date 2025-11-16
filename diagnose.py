#!/usr/bin/env python
"""
Diagnostic Script - Test Your Environment

This script checks if all dependencies are properly installed.
Run this FIRST before running main.py

Usage:
    python diagnose.py
"""

import sys
import importlib

print("="*70)
print("ENVIRONMENT DIAGNOSTIC")
print("="*70)

print(f"\nPython: {sys.version}")
print(f"Executable: {sys.executable}")

# List of required packages
packages = [
    'numpy',
    'pandas',
    'scipy',
    'sklearn',
    'yaml',
    'matplotlib',
    'seaborn',
    'lightgbm',
]

print("\nChecking dependencies...")
print("-" * 70)

all_ok = True
for package in packages:
    try:
        mod = importlib.import_module(package)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {package:20s} OK (version: {version})")
    except ImportError as e:
        print(f"✗ {package:20s} MISSING - {e}")
        all_ok = False

print("-" * 70)

if all_ok:
    print("\n✓ All dependencies installed!")
    print("\nYou can now run:")
    print("  python main.py")
else:
    print("\n✗ Some dependencies are missing!")
    print("\nFix with:")
    print("  pip install -r requirements.txt")
    print("\nOr manually:")
    print("  pip install numpy pandas scipy scikit-learn pyyaml matplotlib seaborn lightgbm")

print("\n" + "="*70)
