#!/bin/bash

# Ensure the script is executed in Bash
if [ -z "$BASH_VERSION" ]; then
    exec bash "$0" "$@"
fi

# Check if Conda is available
if command -v conda &> /dev/null; then
    echo "Conda found. Setting up environment..."
    . ~/anaconda3/etc/profile.d/conda.sh

    # Initialize and activate Conda environment
    conda init bash
    . ~/.bashrc
    conda create --name fid-gan python=3.7 -y
    conda activate fid-gan
else
    echo "Conda not found. Using Python venv instead."
    
    # Ensure Python 3.7 is installed
    if ! command -v python3.7 &> /dev/null; then
        echo "Python 3.7 is not installed. Please install it first."
        exit 1
    fi

    # Create and activate a virtual environment with venv
    python3.7 -m venv fid-gan-venv
    source fid-gan/bin/activate
    echo "Virtual environment (venv) activated."
fi

# Install project dependencies
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt file not found!"
    exit 1
fi

echo "Environment setup complete."
