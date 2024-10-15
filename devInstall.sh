#!/bin/bash

# Create and activate virtual environment
# Replace `python` with `path` to desired python version (optional prefix can also be `python3` or similiar)
python -m venv .venv

# Use created virtual venv (VSCode prompts for automatic usage of venv)
source venv/bin/activate

# # Install package in editable mode (for development)
pip install -e .

# # Optional: Install custom dependencies
# # Link for custom installation: https://pytorch.org/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
