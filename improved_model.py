"""
Enhanced 4-gram Text Generation Model
======================================
Better accuracy with stronger context (4-gram + weighted selection)
"""

import os
import random
from collections import defaultdict, Counter
import re


class NGramModel:
    """4-gram Markov chain model with weighted selection and fallback"""

    def __init__(self):
        # 4-gram: (w1, w2, w3) -> [w4, w4, ...]
        self.fourgrams = defaultdict(list)
        self.fourgram_counts = defaultdict(Counter)
        
        # 3-gram fallback
        self.trigrams = defaultdict(list)
        self.trigram_counts = defaultdict(Counter)
        
        # 2-gram fallback
        self.bigrams = defaultdict(list)
        self.bigram_counts = defaultdict(Counter)
        
        self.words = []
        self.vocab = set()

    def load_data(self, filepath):
        """Load and preprocess data"""
        print(f"[*] Loading data from {filepath}...")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read().lower()

        # Clean text (keep punctuation for sentence boundaries)
        text = re.sub(r"[^a-z\s\.\,\!\?\']", "", text)

        # Split into words
        words = text.split()
        self.words = words
        self.vocab = set(words)

        print(f"[OK] Loaded {len(words):,} words from training data")
        print(f"[OK] Vocabulary: {len(self.vocab):,} unique words")

        return words

    def build_ngrams(self, words):
        """Build 2-gram, 3-gram, and 4-gram models"""
        print("[*] Building N-gram model...")

        # Store vocabulary
        self.vocab = set(words)

        # Add sentence markers
        words_marked = ["<s>", "<s>"] + words + ["</s>"]

        # Build 4-grams: (w1, w2, w3) -> w4
        for i in range(len(words_marked) - 3):
            w1, w2, w3, w4 = words_marked[i:i+4]
            key = (w1, w2, w3)
            self.fourgrams[key].append(w4)
            self.fourgram_counts[key][w4] += 1

        # Build 3-grams: (w1, w2) -> w3
        for i in range(len(words_marked) - 2):
            w1, w2, w3 = words_marked[i:i+3]
            key = (w1, w2)
            self.trigrams[key].append(w3)
            self.trigram_counts[key][w3] += 1

        # Build 2-grams: w1 -> w2
        for i in range(len(words_marked) - 1):
            w1, w2 = words_marked[i:i+2]
            self.bigrams[w1].append(w2)
            self.bigram_counts[w1][w2] += 1

        print(f"[OK] Built {len(self.fourgrams):,} 4-gram patterns")
        print(f"[OK] Built {len(self.trigrams):,} 3-gram patterns")
        print(f"[OK] Built {len(self.bigrams):,} 2-gram patterns")

    def _choose_weighted(self, options, counts_dict):
        """Choose weighted by frequency"""
        if not options:
            return None
        
        unique_options = list(set(options))
        if len(unique_options) == 1:
            return unique_options[0]
        
        # Weight by frequency (higher frequency = higher probability)
        weights = [counts_dict[opt] ** 1.5 for opt in unique_options]
        total = sum(weights)
        probabilities = [w / total for w in weights]
        
        return random.choices(unique_options, weights=probabilities, k=1)[0]

    def generate(self, seed_text="", length=50):
        """Generate text using 4-grams with fallback"""
        
        if not seed_text:
            # Start with <s> <s>
            w1, w2, w3 = "<s>", "<s>", self._choose_weighted(
                self.bigrams["<s>"],
                self.bigram_counts["<s>"]
            )
            if not w3 or w3 == "</s>":
                w3 = self._choose_weighted(
                    [w for w in self.vocab if len(w) > 2],
                    Counter({w: 1 for w in self.vocab if len(w) > 2})
                )
            result = [w3]
        else:
            # Parse seed text
            seeds = seed_text.lower().split()
            seeds = [re.sub(r"[^a-z\']", "", w) for w in seeds]
            seeds = [w for w in seeds if w and w in self.vocab]

            if not seeds:
                # Fallback to random start
                w1, w2 = "<s>", "<s>"
                w3 = self._choose_weighted(
                    self.bigrams["<s>"],
                    self.bigram_counts["<s>"]
                )
                result = [w3]
            elif len(seeds) == 1:
                w1, w2, w3 = "<s>", "<s>", seeds[0]
                result = [w3]
            elif len(seeds) == 2:
                w1, w2, w3 = "<s>", seeds[0], seeds[1]
                result = seeds
            else:
                w1, w2, w3 = seeds[-3], seeds[-2], seeds[-1]
                result = seeds[-2:]

        # Generate text
        for _ in range(length):
            # Try 4-gram first (strongest context)
            key4 = (w1, w2, w3)
            if key4 in self.fourgrams and self.fourgrams[key4]:
                w4 = self._choose_weighted(
                    self.fourgrams[key4],
                    self.fourgram_counts[key4]
                )
            # Fallback to 3-gram
            elif (w2, w3) in self.trigrams and self.trigrams[(w2, w3)]:
                w4 = self._choose_weighted(
                    self.trigrams[(w2, w3)],
                    self.trigram_counts[(w2, w3)]
                )
            # Fallback to 2-gram
            elif w3 in self.bigrams and self.bigrams[w3]:
                w4 = self._choose_weighted(
                    self.bigrams[w3],
                    self.bigram_counts[w3]
                )
            else:
                break

            # Stop if we hit end marker
            if w4 == "</s>":
                break

            result.append(w4)
            w1, w2, w3 = w2, w3, w4

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
    print("[*] Enhanced 4-gram Text Generation Model")
    print("=" * 70 + "\n")

    # Initialize model
    model = NGramModel()

    # Get correct path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "data", "english_stories.txt")

    if not os.path.exists(data_file):
        print(f"[!] Data file not found: {data_file}")
        return

    words = model.load_data(data_file)
    model.build_ngrams(words)

    # Generate samples
    seeds = ["once upon a time", "the king was wise", "the merchant traveled"]
    model.generate_batch(seeds, count=2, length=50)

    print("=" * 70)
    print("[OK] 4-gram model generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
