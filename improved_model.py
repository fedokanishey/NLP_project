"""
Improved 3-gram Text Generation Model
=====================================
Better accuracy without TensorFlow complexity
"""

import os
import random
from collections import defaultdict
import re


class TrigramModel:
    """3-gram Markov chain model for better context"""

    def __init__(self):
        self.trigrams = defaultdict(list)
        self.bigrams = defaultdict(list)
        self.words = []
        self.vocab = set()

    def load_data(self, filepath):
        """Load and preprocess data"""
        print(f"[*] Loading data from {filepath}...")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().lower()

        # Clean text
        text = re.sub(r"[^a-z\s\.\,\!\?\']", "", text)

        # Split into words
        words = text.split()
        self.words = words
        self.vocab = set(words)

        print(f"[OK] Loaded {len(words):,} words from training data")
        print(f"[OK] Vocabulary: {len(self.vocab):,} unique words")

        return words

    def build_trigrams(self, words):
        """Build 3-gram model"""
        print("[*] Building 3-gram model...")

        # Add sentence markers
        words_marked = ["<start>"] + words + ["<end>"]

        # Build trigrams: (word1, word2) -> [word3, word3, ...]
        for i in range(len(words_marked) - 2):
            w1, w2, w3 = words_marked[i], words_marked[i + 1], words_marked[i + 2]
            key = (w1, w2)
            self.trigrams[key].append(w3)

        # Build bigrams as fallback
        for i in range(len(words_marked) - 1):
            w1, w2 = words_marked[i], words_marked[i + 1]
            key = w1
            self.bigrams[key].append(w2)

        print(f"[OK] Built {len(self.trigrams):,} trigram patterns")
        print(f"[OK] Built {len(self.bigrams):,} bigram patterns")

    def generate(self, seed_text="", length=50):
        """Generate text using 3-grams"""

        if not seed_text:
            # Start from beginning
            w1, w2 = "<start>", self.bigrams["<start>"][0]
        else:
            # Use seed text
            seeds = seed_text.lower().split()
            seeds = [re.sub(r"[^a-z\']", "", w) for w in seeds]
            seeds = [w for w in seeds if w]

            # Check if words exist in vocabulary
            vocab = self.vocab
            seeds = [w for w in seeds if w in vocab]

            if not seeds:
                # If no valid seeds, start from beginning
                w1, w2 = "<start>", self.bigrams["<start>"][0]
            elif len(seeds) >= 2:
                w1, w2 = seeds[-2], seeds[-1]
            elif len(seeds) == 1:
                w1 = "<start>"
                w2 = seeds[0]
            else:
                w1, w2 = "<start>", self.bigrams["<start>"][0]

        result = [w1, w2] if w1 != "<start>" else [w2]

        # Generate text
        for _ in range(length):
            key = (w1, w2)

            # Try trigram first
            if key in self.trigrams and self.trigrams[key]:
                w3 = random.choice(self.trigrams[key])
            # Fallback to bigram
            elif w2 in self.bigrams and self.bigrams[w2]:
                w3 = random.choice(self.bigrams[w2])
            else:
                break

            if w3 == "<end>":
                break

            result.append(w3)
            w1, w2 = w2, w3

        return " ".join(result)

    def generate_batch(self, seeds, count=3, length=50):
        """Generate multiple texts for each seed"""
        print("\n" + "=" * 70)
        print("[*] Generating Text Samples")
        print("=" * 70)

        for seed in seeds:
            print(f"\nSeed: '{seed}'")
            print("-" * 70)
            for i in range(count):
                text = self.generate(seed, length=length)
                print(f"{i+1}. {text}\n")


def main():
    print("\n" + "=" * 70)
    print("[*] Improved 3-gram Text Generation Model")
    print("=" * 70 + "\n")

    # Initialize model
    model = TrigramModel()

    # Get correct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "data", "english_stories.txt")

    if not os.path.exists(data_file):
        print(f"[!] Data file not found: {data_file}")
        return

    words = model.load_data(data_file)

    # Build model
    model.build_trigrams(words)

    # Generate samples
    seeds = ["once upon a time", "the king was wise", "the merchant traveled"]

    model.generate_batch(seeds, count=2, length=50)

    print("=" * 70)
    print("[OK] 3-gram model generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
