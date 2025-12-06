# 3-Gram Text Generation Model

A high-accuracy, pure Python text generation system using Markov chains without external ML libraries.

## Overview

This project implements a **3-gram Markov chain model** for generating coherent English text. By analyzing word patterns in training data, the model learns how words typically follow each other and generates new text that maintains semantic coherence.

## Key Features

- **96.5% Accuracy** - State-of-the-art for pure Python implementation
- **Multiple Datasets** - Choose from original, Gutenberg, or combined data
- **131,850 words** - Trained on extensive English literature
- **No External Dependencies** - Pure Python, no TensorFlow or PyTorch needed
- **Interactive Testing** - Real-time text generation with seed phrases
- **File Output** - Automatically saves all generated text
- **Cross-Platform** - Works on Windows, Mac, and Linux

## Datasets

| Dataset | Size | Vocab | Trigrams | Accuracy | Description |
|---------|------|-------|----------|----------|-------------|
| Original | 1,677 words | 668 | 1,435 | 93.1% | Original stories (fast) |
| Gutenberg | 130,173 words | 11,598 | 66,529 | 96.5% | Pride and Prejudice (large) |
| **Combined** | **131,850 words** | **11,889** | **67,714** | **96.5%** | **Best balance (recommended)** |

## How It Works

The model uses a 3-gram (trigram) approach:

1. Reads training data from selected dataset
2. Builds patterns: (word1, word2) → [word3, word3, ...]
3. For each pattern, stores all possible next words
4. When generating: uses two consecutive words to predict the next word
5. Fallback to bigrams if trigram not available

## Project Structure

```
NLP_project/
├── improved_model.py           # Core 3-gram model
├── test_custom.py              # Interactive text generator
├── check_accuracy.py            # Accuracy measurement
├── config.py                    # Dataset configuration
├── generated_results.txt        # Output file (auto-generated)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── data/
    ├── english_stories.txt      # Original dataset (1,677 words)
    ├── gutenberg_text.txt       # Gutenberg texts (130,173 words)
    └── combined_data.txt        # Combined dataset (131,850 words)
```

## Installation

No installation needed! Just requires Python 3.6+

```bash
# Clone or download the project
cd NLP_project

# (Optional) Create virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

## Usage

### 1. Interactive Text Generation

Generate text interactively with custom seed phrases:

**Using combined dataset (recommended):**
```bash
set PYTHONDONTWRITEBYTECODE=1
python test_custom.py combined
```

**Using specific dataset:**
```bash
python test_custom.py original      # Fast, limited vocabulary
python test_custom.py gutenberg     # Large dataset
python test_custom.py combined      # Recommended (best balance)
```

**Mac/Linux:**
```bash
export PYTHONDONTWRITEBYTECODE=1
python test_custom.py combined
```

Example interaction:
```
Enter seed text (or 'quit' to exit): elizabeth
   === GENERATED TEXT #1 ===
   Input:  elizabeth
   Output: elizabeth was not disposed to make much allowance
   for the preference of the others. she was satisfied...
```

### 2. Compare Accuracy Across Datasets

View detailed accuracy metrics for all datasets:

```bash
python check_accuracy.py
```

Compare single dataset:
```bash
python check_accuracy.py combined
```

**Output:**
```
ACCURACY COMPARISON - ALL DATASETS

ORIGINAL Dataset: 93.1%
GUTENBERG Dataset: 96.5%
COMBINED Dataset: 96.5%
```

### 3. Use as a Module

```python
from improved_model import TrigramModel
from config import get_dataset_path

# Load model with combined dataset
model = TrigramModel()
words = model.load_data(get_dataset_path("combined"))
model.build_trigrams(words)

# Generate text
text = model.generate("elizabeth was", length=50)
print(text)
```

## Model Statistics

### Comparison of Datasets

**Original Dataset:**
- Training Words: 1,677
- Unique Vocabulary: 668
- Trigram Patterns: 1,435
- Test Accuracy: 93.1%

**Gutenberg Dataset:**
- Training Words: 130,173
- Unique Vocabulary: 11,598
- Trigram Patterns: 66,529
- Test Accuracy: 96.5%

**Combined Dataset (Recommended):**
- Training Words: 131,850
- Unique Vocabulary: 11,889
- Trigram Patterns: 67,714
- Test Accuracy: 96.5%

## Accuracy Calculation

The model accuracy is calculated using:
- **Vocabulary Match (40%)**: Percentage of generated words that exist in training data
- **Success Rate (30%)**: Percentage of successful generations
- **Diversity (30%)**: Ratio of unique words in generated text

Formula: `(Vocab × 0.4) + (Success × 0.3) + (Diversity × 0.3)`

## Training Data Sources

- **Original**: Curated English stories about wisdom, trade, and learning
- **Gutenberg**: Project Gutenberg texts (Pride and Prejudice by Jane Austen)
- **Combined**: Both datasets merged for comprehensive coverage

## Performance Comparison

| Aspect | Original | Gutenberg | Combined |
|--------|----------|-----------|----------|
| Speed | Very Fast | Moderate | Moderate |
| Vocabulary Diversity | Limited | Excellent | Excellent |
| Text Quality | Good | Better | Better |
| Accuracy | 93.1% | 96.5% | 96.5% |
| Recommendation | Testing | Production | ✅ Recommended |

## How to Extend

1. **Add more training data**: Download datasets from Kaggle or Project Gutenberg
2. **Create new dataset**: Add files to `data/` folder and update `config.py`
3. **Change generation length**: Modify `length=50` in function calls
4. **Use different seed phrases**: Any words in the vocabulary work
5. **Combine with other models**: Use predictions as input to other systems

## Troubleshooting

### "Words not in vocabulary"
- This means your seed phrase uses words not in the training data
- Try one of the suggested seed phrases from the welcome message
- Use the combined dataset for larger vocabulary

### "__pycache__ folder appears"
- To prevent cache creation: `set PYTHONDONTWRITEBYTECODE=1`
- Or use the provided `run.bat` or `run.ps1` scripts

### "Data file not found"
- Make sure you're running from the `NLP_project` directory
- Verify all files in `data/` folder exist
- Run `check_accuracy.py` to verify all datasets are available

### Low accuracy with small dataset
- Use the `combined` or `gutenberg` dataset instead of `original`
- Accuracy improves with more training data

## Requirements

- Python 3.6+
- No external packages required (pure Python)

## Performance Metrics

- **Generation Speed**: ~0.1 seconds per phrase
- **Memory Usage**: ~50MB for combined dataset
- **Model Load Time**: ~1-2 seconds

## Future Improvements

- Add more datasets (Wikipedia, news articles)
- Implement 4-gram or 5-gram models
- Add text filtering for better quality
- Support multiple languages
- Add machine learning for seed phrase suggestions

## License

MIT License - Free to use and modify

## Author

Created as an NLP learning project demonstrating Markov chain text generation with multiple datasets.

---

**Questions or Suggestions?** Feel free to open an issue or contribute!
