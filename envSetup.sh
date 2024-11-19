#!/bin/bash

# Detect the operating system and handle virtual environment setup
if [[ "$OSTYPE" == "linux-gnu"* || "$OSTYPE" == "darwin"* ]]; then
    echo "Detected Linux/MacOS"
    python3 -m venv .venv
    if [[ ! -d ".venv" ]]; then
        echo "Error: Virtual environment was not created successfully."
        exit 1
    fi
    source .venv/bin/activate

elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "Detected Windows"
    python -m venv .venv
    if [[ ! -d ".venv" ]]; then
        echo "Error: Virtual environment was not created successfully."
        exit 1
    fi
    # Use forward slashes for compatibility with Git Bash
    source .venv/Scripts/activate

else
    echo "Unsupported operating system: $OSTYPE"
    exit 1
fi

# Install package in editable mode (for development)
pip install -e .

# Optional: Install custom dependencies
# Link for custom installation: https://pytorch.org/
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
