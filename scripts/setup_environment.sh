#!/bin/bash
# Setup script for MSEA development environment
set -e

echo "Setting up Multimodal Self-Evaluating Agents environment..."

# Check Python version
python3 --version || { echo "Python 3.10+ required"; exit 1; }

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package
pip install -e ".[dev]"

# Create necessary directories
mkdir -p results
mkdir -p logs

echo "Setup complete! Activate with: source .venv/bin/activate"
