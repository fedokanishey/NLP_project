"""
Configuration for NLP Text Generation Model
"""

import os

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Dataset path
DATA_FILE = os.path.join(DATA_DIR, "combined_data.txt")

# Dataset information
DATASET_INFO = {
    "words": 131850,
    "vocabulary": 11889,
    "description": "Combined dataset (English stories + Pride and Prejudice)",
}


def get_dataset_path():
    """Get path to dataset file"""
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    return DATA_FILE
