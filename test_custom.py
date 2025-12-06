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


log("\n" + "=" * 70)
log("[*] Interactive 3-gram Text Generation Tester")
log("=" * 70 + "\n")

# Load model
log("Loading improved model...\n")
model = TrigramModel()

# Get correct path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "data", "english_stories.txt")

words = model.load_data(data_path)
model.build_trigrams(words)

log(f"[OK] Model loaded!")
log(f"   - Training words: {len(words)}")
log(f"   - Vocabulary size: {len(set(words))}")
log(f"   - 3-gram patterns: {len(model.trigrams)}\n")

# ============= Sample Prompts =============

log("Sample seed phrases (try these):")
log("-" * 70)
available_seeds = [
    "once upon a time",
    "the king was wise",
    "the merchant traveled",
    "a scholar learned",
    "the farmer worked",
]
for seed in available_seeds:
    log(f"   • {seed}")

log("\n" + "=" * 70)
log("[*] Interactive Testing")
log("=" * 70 + "\n")

test_count = 0

while True:
    user_input = input("Enter seed text (or 'quit' to exit): ").strip().lower()

    if user_input in ["quit", "exit", "q"]:
        break

    if not user_input:
        log("[!] Please enter a seed text\n")
        continue

    # Check if seed words are in vocabulary
    seed_words = user_input.split()
    vocab = set(words)

    # Find valid seed - if none exist, suggest examples
    valid_seed_words = [w for w in seed_words if w in vocab]

    if not valid_seed_words:
        log(f"[!] Words '{user_input}' not found in vocabulary")
        log(
            f"   Try: {', '.join(random.sample(available_seeds, min(3, len(available_seeds))))}\n"
        )
        continue

    # Generate text
    text = model.generate(user_input, length=50)

    test_count += 1
    log(f"\n[OK] Generated Text #{test_count}:")
    log(f"   Seed: '{user_input}'")
    log(f"   Output: {text}\n")

log("=" * 70)
log(f"[DONE] Total generations: {test_count}")
log("=" * 70 + "\n")

output.close()
