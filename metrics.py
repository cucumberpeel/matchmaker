from typing import Dict, Iterable, List, Tuple, Callable, Union

Pred = Union[str, None]
Gold = Union[str, None]


def _counts(
    golds: Iterable[Gold],
    preds: Iterable[Pred],
    is_correct: Callable[[Gold, Pred], bool] = lambda g, p: g is not None and p == g,
) -> Dict[str, int]:
    tp = fp = fn = tn = 0
    for gold, pred in zip(golds, preds):
        if gold is None:
            if pred is None:
                tn += 1        # correctly predicted "no match"
            else:
                fp += 1        # predicted a match where none exists
        else:
            if pred is None:
                fn += 1        # missed an existing match
            elif is_correct(gold, pred):
                tp += 1
            else:
                fp += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def compute_classification_metrics(
    golds: List[Gold],
    preds: List[Pred],
    is_correct: Callable[[Gold, Pred], bool] = lambda g, p: g is not None and p == g,
) -> Dict[str, float]:
    """Compute accuracy/precision/recall/F1 (+ coverage) for match predictions."""
    if len(golds) != len(preds):
        raise ValueError("golds and preds must have the same length")

    c = _counts(golds, preds, is_correct=is_correct)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]

    pos_total = tp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / pos_total if pos_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0

    coverage = (total - tn - fn) / total if total else 0.0  # fraction with a non-None pred among matchable rows

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": coverage,
    }


def compute_per_method_metrics(
    method_results: Dict[str, Tuple[List[Gold], List[Pred]]],
    is_correct: Callable[[Gold, Pred], bool] = lambda g, p: g is not None and p == g,
) -> Dict[str, Dict[str, float]]:
    return {name: compute_classification_metrics(g, p, is_correct=is_correct)
            for name, (g, p) in method_results.items()}
