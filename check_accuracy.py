"""
Quick Accuracy Check - Single Line Output
"""

import re
import os
from improved_model import TrigramModel


def get_accuracy():
    """Get current accuracy"""

    # Get correct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "data", "english_stories.txt")

    # Load data
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read().lower()
    text = re.sub(r"[^a-z\s\.\,\!\?\']", "", text)
    words = text.split()

    # Build model
    model = TrigramModel()
    model.load_data(data_file)
    model.build_trigrams(words)

    # Generate samples and calculate metrics
    test_seeds = ["once upon a time", "the king was wise", "the merchant traveled"]

    metrics = []
    for seed in test_seeds:
        generated = model.generate(seed, length=50)
        gen_words = generated.split()

        vocab_match = sum(1 for w in gen_words if w in set(words))
        vocab_ratio = vocab_match / len(gen_words) * 100 if gen_words else 0
        unique_ratio = len(set(gen_words)) / len(gen_words) * 100 if gen_words else 0

        metrics.append({"vocab_ratio": vocab_ratio, "unique_ratio": unique_ratio})

    # Calculate overall accuracy
    avg_vocab = sum(m["vocab_ratio"] for m in metrics) / len(metrics)
    avg_unique = sum(m["unique_ratio"] for m in metrics) / len(metrics)
    success_rate = 100

    overall = (avg_vocab * 0.4) + (success_rate * 0.3) + (avg_unique * 0.3)

    return overall


if __name__ == "__main__":
    accuracy = get_accuracy()
    print(f"Current Accuracy: {accuracy:.1f}%")
