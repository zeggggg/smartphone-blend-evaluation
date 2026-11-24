"""Compare selection-set and held-out AUC after tuning a two-model blend."""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def read_predictions(path):
    data = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    required = {"label", "pred_a", "pred_b"}
    if not required.issubset(data.dtype.names or ()):
        raise ValueError("Expected CSV columns label,pred_a,pred_b")
    return data["label"], data["pred_a"], data["pred_b"]


def demo_predictions():
    rng = np.random.default_rng(19)
    signal = rng.normal(size=2000)
    labels = rng.binomial(1, 1 / (1 + np.exp(-signal)))
    return labels, signal + rng.normal(0, 0.4, len(signal)), signal + rng.normal(0, 0.4, len(signal))


def evaluate(labels, a, b, size, repeats, seed):
    if set(np.unique(labels)) != {0, 1} or not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Expected binary labels and finite prediction scores")
    if repeats < 1 or size < 2 or size > len(labels) - 2:
        raise ValueError("Choose positive repeats and leave at least two rows in each subset")

    indices = np.arange(len(labels))
    rng = np.random.default_rng(seed)
    # Evaluate weights closest to the reference first to break exact ties.
    weights = sorted((i / 20 for i in range(21)), key=lambda w: (abs(w - 0.5), w))
    reference = 0.5 * a + 0.5 * b
    selection_gains, held_out_gains = [], []
    for _ in range(repeats):
        choose, hold = train_test_split(
            indices, train_size=size, stratify=labels,
            random_state=int(rng.integers(2**31)),
        )
        if len(np.unique(labels[choose])) != 2 or len(np.unique(labels[hold])) != 2:
            raise ValueError("Both subsets need examples of both classes to compute AUC")
        a_choose, b_choose = a[choose], b[choose]
        selection_scores = [
            roc_auc_score(labels[choose], w * a_choose + (1 - w) * b_choose)
            for w in weights
        ]
        best = int(np.argmax(selection_scores))
        w = weights[best]
        selection_gains.append(selection_scores[best] - selection_scores[0])
        tuned = w * a[hold] + (1 - w) * b[hold]
        held_out_gains.append(roc_auc_score(labels[hold], tuned)
                              - roc_auc_score(labels[hold], reference[hold]))
    print(f"Mean selection AUC gain: {np.mean(selection_gains):+.6f}")
    print(f"Mean held-out AUC gain:  {np.mean(held_out_gains):+.6f}")
    print(f"Worse than 50/50 on held-out rows: {sum(g < 0 for g in held_out_gains)}/{repeats}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("csv", nargs="?")
    inputs.add_argument("--demo", action="store_true")
    parser.add_argument("--selection-size", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        sample = demo_predictions() if args.demo else read_predictions(args.csv)
        evaluate(*sample, args.selection_size, args.repeats, args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
