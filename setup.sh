#!/bin/bash

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/cache
mkdir -p data/processed
mkdir -p models
mkdir -p logs

# Download and prepare dataset
python src/data/prepare_dataset.py

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_HOME=$(pwd)/data/cache

echo "Setup completed! You can now train the model using:"
echo "python src/train.py --model_name roberta-base --batch_size 32 --max_epochs 10" 