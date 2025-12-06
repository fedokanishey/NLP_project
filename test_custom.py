"""
Interactive Text Generation Tester
===================================
Test the improved 3-gram model with your own prompts!
Run: python test_custom.py [dataset_name]

Available datasets:
  - original: 1,677 words (fast)
  - gutenberg: 130,173 words (large)
  - combined: 131,850 words (recommended)
"""

import random
import os
import sys
from improved_model import TrigramModel
from config import get_dataset_path, DATASET_INFO

# Get dataset from command line or use default
dataset = sys.argv[1] if len(sys.argv) > 1 else "combined"

# Validate dataset
try:
    data_path = get_dataset_path(dataset)
except (ValueError, FileNotFoundError) as e:
    print(f"[!] {e}")
    sys.exit(1)

# Set up file for saving results
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, "generated_results.txt")
output = open(output_file, "w", encoding="utf-8")


def log(text):
    """Print and save to file"""
    print(text)
    output.write(text + "\n")
    output.flush()


def separator(char="=", width=70):
    """Print formatted separator"""
    log(char * width)


log("")
separator()
log("  INTERACTIVE TEXT GENERATION WITH 3-GRAM MODEL")
log(f"  Dataset: {dataset.upper()} | {DATASET_INFO[dataset]['description']}")
separator()
log("")

# Load model
log(f"Loading {dataset} dataset...")
model = TrigramModel()
words = model.load_data(data_path)
model.build_trigrams(words)

log("Model Ready!")
log(f"   Training data: {len(words):,} words")
log(f"   Vocabulary: {len(set(words)):,} unique words")
log(f"   3-gram patterns: {len(model.trigrams):,}")
log("")

# ============= Sample Prompts (from vocabulary) =============

log("TRY THESE SEED PHRASES:")
log("-" * 70)
# Find valid seed phrases from the vocabulary
vocab = set(words)
available_seeds = [
    "once upon a time",
    "the king was wise",
    "the merchant traveled",
    "a scholar learned",
    "the farmer worked",
    "the world was",
    "in the beginning",
    "the land was",
]

# Filter to only available seeds
valid_seeds = [s for s in available_seeds if all(w in vocab for w in s.split())]
if not valid_seeds:
    # Fall back to random words from vocabulary
    sample_words = random.sample(list(vocab), min(5, len(vocab)))
    valid_seeds = [" ".join(random.sample(sample_words, min(2, len(sample_words)))) for _ in range(5)]

for i, seed in enumerate(valid_seeds[:5], 1):
    log(f"   {i}. {seed}")

log("")
separator()
log("  >>> START GENERATING <<<")
separator()
log("")

test_count = 0

while True:
    user_input = input("Enter seed text (or 'quit' to exit): ").strip().lower()

    if user_input in ["quit", "exit", "q"]:
        break

    if not user_input:
        log("   [!] Please enter some text to continue")
        log("")
        continue

    # Check if seed words are in vocabulary
    seed_words = user_input.split()
    valid_seed_words = [w for w in seed_words if w in vocab]

    if not valid_seed_words:
        suggestions = random.sample(valid_seeds, min(3, len(valid_seeds)))
        log(f"   [X] '{user_input}' - not in vocabulary")
        log(f"   [*] Try one of these instead:")
        for suggestion in suggestions:
            log(f"      -> {suggestion}")
        log("")
        continue

    # Generate text
    text = model.generate(user_input, length=50)

    test_count += 1
    log("")
    log(f"   === GENERATED TEXT #{test_count} ===")
    log(f"   {'-' * 60}")
    log(f"   Input:  {user_input}")
    log(f"   Output: {text}")
    log("")

log("")
separator()
log(f"  [OK] Session Complete | Generated {test_count} texts")
log(f"  [*] Results saved to: generated_results.txt")
separator()
log("")

output.close()
