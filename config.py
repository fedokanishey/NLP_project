"""
Configuration for NLP Text Generation Model
"""

import os

# Get script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Available datasets
DATASETS = {
    "original": os.path.join(DATA_DIR, "english_stories.txt"),
    "gutenberg": os.path.join(DATA_DIR, "gutenberg_text.txt"),
    "combined": os.path.join(DATA_DIR, "combined_data.txt"),
}

# Default dataset
DEFAULT_DATASET = "combined"

# Dataset information
DATASET_INFO = {
    "original": {
        "words": 1677,
        "vocabulary": 668,
        "description": "Original stories (fast)",
    },
    "gutenberg": {
        "words": 130173,
        "vocabulary": 9872,
        "description": "Gutenberg texts (large)",
    },
    "combined": {
        "words": 131850,
        "vocabulary": 11927,
        "description": "Combined dataset (recommended)",
    },
}


def get_dataset_path(dataset_name=None):
    """Get path to dataset file"""
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET

    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Available: {list(DATASETS.keys())}"
        )

    path = DATASETS[dataset_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    return path


def list_datasets():
    """List all available datasets"""
    print("\nAvailable datasets:")
    print("-" * 70)
    for name, info in DATASET_INFO.items():
        path = DATASETS[name]
        exists = "[OK]" if os.path.exists(path) else "[X]"
        print(f"{exists} {name.upper()}")
        print(f"     {info['description']}")
        print(f"     Words: {info['words']:,} | Vocabulary: {info['vocabulary']:,}")
    print()
