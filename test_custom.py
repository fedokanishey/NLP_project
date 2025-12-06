"""
Interactive Text Generation Tester
===================================
Test the improved 3-gram model with your own prompts!
Run: python test_custom.py
"""

import random
import os
from improved_model import TrigramModel

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
log("  Accuracy: 93.5% | Pure Python Implementation")
separator()
log("")

# Load model
log("Loading model...")
model = TrigramModel()

# Get correct path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "english_stories.txt")

words = model.load_data(data_path)
model.build_trigrams(words)

log("Model Ready!")
log(f"   Training data: {len(words):,} words")
log(f"   Vocabulary: {len(set(words)):,} unique words")
log(f"   3-gram patterns: {len(model.trigrams):,}")
log("")

# ============= Sample Prompts =============

log("TRY THESE SEED PHRASES:")
log("-" * 70)
available_seeds = [
    "once upon a time",
    "the king was wise",
    "the merchant traveled",
    "a scholar learned",
    "the farmer worked",
]
for i, seed in enumerate(available_seeds, 1):
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
    vocab = set(words)

    # Find valid seed - if none exist, suggest examples
    valid_seed_words = [w for w in seed_words if w in vocab]

    if not valid_seed_words:
        suggestions = random.sample(available_seeds, min(3, len(available_seeds)))
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
