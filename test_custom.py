"""
Interactive Text Generation Tester
===================================
Generate coherent English text using 4-gram model
Run: python test_custom.py
"""

import random
import os
from improved_model import NGramModel
from config import get_dataset_path, DATASET_INFO

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
log("  4-GRAM TEXT GENERATION MODEL")
log(
    f"  Dataset: {DATASET_INFO['words']:,} words | Vocab: {DATASET_INFO['vocabulary']:,}"
)
separator()
log("")

# Load model
log("Loading model...")
model = NGramModel()
data_path = get_dataset_path()
words = model.load_data(data_path)
model.build_ngrams(words)

log("Model Ready!")
log(f"   4-gram patterns: {len(model.fourgrams):,}")
log(f"   Weighted selection enabled")
log("")

# Get sample seeds from vocabulary
vocab = model.vocab
sample_words = list(vocab)
if len(sample_words) > 5:
    available_seeds = [
        " ".join([sample_words[i % len(sample_words)] for i in range(j * 1, j * 2)])
        for j in range(1, 6)
    ]
else:
    available_seeds = [" ".join(sample_words[:2])]

log("TRY THESE SEED PHRASES:")
log("-" * 70)
for i, seed in enumerate(available_seeds[:5], 1):
    log(f"   {i}. {seed}")

log("")
separator()
log("  >>> GENERATE TEXT <<<")
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
