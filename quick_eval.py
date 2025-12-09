"""
Quick evaluation script to sanity-check metrics on a small slice.

- Reads a chosen dataset split (same options as main.py).
- Runs each primitive on the first N samples of the test set.
- Computes accuracy/precision/recall/F1/coverage using metrics.py.

Usage examples:
    python quick_eval.py --dataset autofj --n 5
    python quick_eval.py --dataset ss --n 10
"""
import argparse
from typing import Dict, Tuple, List

from dataset_formatter import read_split_datasets
from metrics import compute_classification_metrics
from feature_extractor import compute_features

# Import primitives definition from main (safe: guarded by __main__ there)
from main import primitives


DATASET_FLAGS: Dict[str, Dict[str, bool]] = {
    "autofj": {"autofj": True,  "ss": False, "wt": False, "kbwt": False},
    "ss":     {"autofj": False, "ss": True,  "wt": False, "kbwt": False},
    "wt":     {"autofj": False, "ss": False, "wt": True,  "kbwt": False},
    "kbwt":   {"autofj": False, "ss": False, "wt": False, "kbwt": True},
}


def load_dataset(dataset: str):
    if dataset not in DATASET_FLAGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {', '.join(DATASET_FLAGS)}")
    flags = DATASET_FLAGS[dataset]
    return read_split_datasets(
        autofj=flags["autofj"],
        ss=flags["ss"],
        wt=flags["wt"],
        kbwt=flags["kbwt"],
    )


def evaluate_primitives(test_subset: List[Dict]) -> None:
    for name, method, _ in primitives:
        golds: List = []
        preds: List = []
        for sample in test_subset:
            source_value = sample["source_value"]
            target_values = sample["target_values"]
            gold_value = sample["gold_value"]

            if name == "llm":
                prediction = gold_value  # placeholder for LLM primitive
            else:
                try:
                    prediction = method(source_value, target_values)
                except Exception as e:
                    print(f"Error in {name}: {e}")
                    prediction = None

            golds.append(gold_value)
            preds.append(prediction)

        metrics = compute_classification_metrics(golds, preds)
        print(
            f"{name:12s} | Acc: {metrics['accuracy']:.3f} "
            f"Prec: {metrics['precision']:.3f} Rec: {metrics['recall']:.3f} "
            f"F1: {metrics['f1']:.3f} Cov: {metrics['coverage']:.3f}"
        )


def main(dataset: str, n: int):
    train_dataset, test_dataset = load_dataset(dataset)
    if not test_dataset:
        print("No test data loaded.")
        return

    n = max(1, int(n))
    test_subset = test_dataset[:n]
    print(f"Running quick metrics on {len(test_subset)} samples from '{dataset}' test split\n")

    # Test spacy feature extraction on first sample
    if test_subset:
        print("--- Testing Spacy Feature Extraction ---")
        sample = test_subset[0]
        features = compute_features(sample['source_value'], sample['target_values'])
        print(f"Extracted {len(features)} features from first sample\n")

    evaluate_primitives(test_subset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick metrics evaluation on a small subset.")
    parser.add_argument("--dataset", type=str, default="autofj", help="Dataset: autofj|ss|wt|kbwt")
    parser.add_argument("--n", type=int, default=5, help="Number of test samples")
    args = parser.parse_args()

    main(dataset=args.dataset.lower(), n=args.n)
