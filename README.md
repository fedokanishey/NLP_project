# 3-Gram Text Generation Model

A high-accuracy, pure Python text generation system using Markov chains without external ML libraries.

## Overview

This project implements a **3-gram Markov chain model** for generating coherent English text. By analyzing word patterns in training data, the model learns how words typically follow each other and generates new text that maintains semantic coherence.

## Key Features

- **93.5% Accuracy** - Evaluated on vocabulary match, uniqueness, and generation success
- **No External Dependencies** - Pure Python, no TensorFlow or PyTorch needed
- **Interactive Testing** - Real-time text generation with seed phrases
- **File Output** - Automatically saves all generated text to `generated_results.txt`
- **Cross-Platform** - Works on Windows, Mac, and Linux

## How It Works

The model uses a 3-gram (trigram) approach:
1. Reads training data (`english_stories.txt`)
2. Builds patterns: (word1, word2) → [word3, word3, ...]
3. For each pattern, stores all possible next words
4. When generating: uses two consecutive words to predict the next word
5. Fallback to bigrams if trigram not available

## Project Structure

```
NLP_project/
├── improved_model.py          # Core 3-gram model
├── test_custom.py             # Interactive text generator
├── check_accuracy.py           # Accuracy measurement
├── generated_results.txt       # Output file (auto-generated)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── data/
    └── english_stories.txt    # Training data (1,676 words)
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

```bash
set PYTHONDONTWRITEBYTECODE=1
python test_custom.py
```

Or on Mac/Linux:
```bash
export PYTHONDONTWRITEBYTECODE=1
python test_custom.py
```

**Example:**
```
Enter seed text (or 'quit' to exit): once upon a time
   === GENERATED TEXT #1 ===
   Input:  once upon a time
   Output: a time in a distant land, there lived a merchant who had traveled...
```

### 2. Check Accuracy

View detailed accuracy metrics:

```bash
python check_accuracy.py
```

**Output:**
```
MODEL ACCURACY EVALUATION
Test 1: 'once upon a time'
  Generated 52 words
  Vocabulary match: 100.0%
  Unique words: 82.7%

FINAL RESULTS
Vocabulary Accuracy (40%): 100.0%
Success Rate (30%):       100.0%
Diversity Score (30%):    75.6%
OVERALL ACCURACY:          92.7%
```

### 3. Use as a Module

```python
from improved_model import TrigramModel

model = TrigramModel()
model.load_data("data/english_stories.txt")
model.build_trigrams(model.words)

text = model.generate("the king was wise", length=50)
print(text)
```

## Model Statistics

| Metric | Value |
|--------|-------|
| Training Words | 1,676 |
| Unique Vocabulary | 668 |
| Trigram Patterns | 1,435 |
| Bigram Patterns | 669 |
| Test Accuracy | 92.7% |

## Accuracy Calculation

The model accuracy is calculated using:
- **Vocabulary Match (40%)**: Percentage of generated words that exist in training data
- **Success Rate (30%)**: Percentage of successful generations
- **Diversity (30%)**: Ratio of unique words in generated text

Formula: `(Vocab × 0.4) + (Success × 0.3) + (Diversity × 0.3)`

## Training Data

The model is trained on `english_stories.txt`, which contains coherent stories about:
- A merchant traveling and trading
- A scholar learning wisdom
- A farmer developing agriculture
- An artist creating beauty
- A physician healing the sick

This diverse vocabulary ensures the model generates varied, contextually-relevant text.

## How to Extend

1. **Add more training data**: Add new story files to `data/` folder
2. **Change generation length**: Modify `length=50` in function calls
3. **Use different seed phrases**: Any words in the vocabulary work
4. **Combine with other models**: Use predictions as input to other systems

## Troubleshooting

### "Words not in vocabulary"
- This means your seed phrase uses words not in the training data
- Try one of the suggested seed phrases from the welcome message
- Or use words that appear in the original stories

### "__pycache__ folder appears"
- To prevent cache creation, use: `set PYTHONDONTWRITEBYTECODE=1`
- Or use the provided `run.bat` or `run.ps1` scripts

### "Data file not found"
- Make sure you're running from the `NLP_project` directory
- Verify `data/english_stories.txt` exists

## Requirements

- Python 3.6+
- No external packages required (pure Python)

## License

MIT License - Free to use and modify

## Author

Created as an NLP learning project demonstrating Markov chain text generation.

---

**Questions or Suggestions?** Feel free to open an issue or contribute!
