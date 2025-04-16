#!/usr/bin/env python3
"""
Installation script for the Gene Set Enrichment Pipeline.
Supports both uv and pip as dependency managers.
"""

import os
import subprocess
import sys
from pathlib import Path

def check_uv_installed():
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def install_with_uv(dev=False):
    """Install dependencies using uv."""
    print("Installing with uv...")
    cmd = ["uv", "pip", "install", "-e", "."]
    if dev:
        cmd.append(".[dev]")
    subprocess.run(cmd, check=True)

def install_with_pip(dev=False):
    """Install dependencies using pip."""
    print("Installing with pip...")
    cmd = [sys.executable, "-m", "pip", "install", "-e", "."]
    if dev:
        cmd.append(".[dev]")
    subprocess.run(cmd, check=True)

def main():
    """Main installation function."""
    dev = "--dev" in sys.argv
    use_uv = "--uv" in sys.argv
    use_pip = "--pip" in sys.argv

    if use_uv and use_pip:
        print("Error: Cannot use both --uv and --pip flags")
        sys.exit(1)

    if use_uv or (not use_pip and check_uv_installed()):
        install_with_uv(dev)
    else:
        install_with_pip(dev)

if __name__ == "__main__":
    main() 