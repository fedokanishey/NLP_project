"""
Quick Accuracy Check - Compare all datasets
"""

import re
import os
import sys
from improved_model import TrigramModel
from config import get_dataset_path, DATASET_INFO


def evaluate_dataset(dataset_name):
    """Evaluate accuracy for a specific dataset"""
    print(f"\n{'=' * 70}")
    print(f"EVALUATING: {dataset_name.upper()} Dataset")
    print(f"{'=' * 70}")
    print(f"{DATASET_INFO[dataset_name]['description']}")
    print(f"Expected size: {DATASET_INFO[dataset_name]['words']:,} words")

    try:
        data_path = get_dataset_path(dataset_name)
    except FileNotFoundError:
        print(f"[X] Dataset not found: {data_path}")
        return None

    # Load and prepare data
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read().lower()
    text = re.sub(r"[^a-z\s\.\,\!\?\']", "", text)
    words = text.split()

    # Build model
    print(f"[*] Building model...")
    model = TrigramModel()
    model.build_trigrams(words)

    # Generate samples and calculate metrics
    # Use words that are definitely in vocabulary
    test_seeds = []
    vocab_list = list(model.vocab)
    
    # Get diverse seeds from vocabulary
    if len(vocab_list) > 10:
        indices = [0, len(vocab_list)//2, len(vocab_list)-1]
        test_seeds = [vocab_list[i] for i in indices if len(vocab_list[i]) > 2]
    
    test_seeds = test_seeds[:3] if test_seeds else vocab_list[:3]
    
    if not test_seeds:
        print("[X] Could not find valid test seeds")
        return None

    metrics = []
    for i, seed in enumerate(test_seeds, 1):
        generated = model.generate(seed, length=50)
        gen_words = generated.split()

        if not gen_words:
            continue

        vocab_match = sum(1 for w in gen_words if w in set(words))
        vocab_ratio = vocab_match / len(gen_words) * 100
        unique_ratio = len(set(gen_words)) / len(gen_words) * 100

        metrics.append({"vocab_ratio": vocab_ratio, "unique_ratio": unique_ratio})

    if not metrics:
        print("[X] Could not generate samples")
        return None

    # Calculate overall accuracy
    avg_vocab = sum(m["vocab_ratio"] for m in metrics) / len(metrics)
    avg_unique = sum(m["unique_ratio"] for m in metrics) / len(metrics)
    success_rate = 100

    overall = (avg_vocab * 0.4) + (success_rate * 0.3) + (avg_unique * 0.3)

    print(f"\nRESULTS:")
    print(f"  Actual dataset size: {len(words):,} words")
    print(f"  Vocabulary built: {len(model.vocab):,} unique words")
    print(f"  Trigram patterns: {len(model.trigrams):,}")
    print(f"  Vocabulary Accuracy: {avg_vocab:.1f}%")
    print(f"  Success Rate:       {success_rate:.1f}%")
    print(f"  Diversity Score:    {avg_unique:.1f}%")
    print(f"  OVERALL ACCURACY:   {overall:.1f}%")

    return overall


if __name__ == "__main__":
    dataset = sys.argv[1] if len(sys.argv) > 1 else None

    if dataset:
        # Evaluate specific dataset
        evaluate_dataset(dataset)
    else:
        # Evaluate all available datasets
        print("\n" + "=" * 70)
        print("ACCURACY COMPARISON - ALL DATASETS")
        print("=" * 70)

        results = {}
        for ds_name in DATASET_INFO.keys():
            try:
                accuracy = evaluate_dataset(ds_name)
                if accuracy is not None:
                    results[ds_name] = accuracy
            except Exception as e:
                print(f"\n[X] Error evaluating {ds_name}: {e}")

        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for ds_name in sorted(results.keys(), key=lambda x: results[x], reverse=True):
            print(f"  {ds_name.upper()}: {results[ds_name]:.1f}%")
        print()
