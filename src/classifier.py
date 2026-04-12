"""SAE-based spelling/grammar error classifier.

Two-phase design:
  1. train()   — learn which SAE features indicate errors (from clean/error pairs)
  2. predict() — given a single text's features, predict whether it contains errors

The interface and internals will evolve between experiments.
"""

from collections import Counter
from dataclasses import dataclass, field


# --------------- Trained model ---------------

@dataclass
class TrainedClassifier:
    """Learned error-detection model: a set of error-indicative SAE features per layer."""
    error_features: dict[int, set[int]]  # {layer: {feature_ids}}
    layers: list[int]
    min_active_features: int = 1  # how many error features must fire to predict "has errors"

    @property
    def total_features(self) -> int:
        return sum(len(fids) for fids in self.error_features.values())

    def summary(self) -> str:
        parts = [f"layer {l}: {len(self.error_features.get(l, set()))} features"
                 for l in self.layers]
        return f"TrainedClassifier({', '.join(parts)}, threshold={self.min_active_features})"


# --------------- Training ---------------

def train(
    pair_features: list[dict],
    layers: list[int],
    min_pair_ratio: float = 0.5,
) -> TrainedClassifier:
    """Learn error-indicative features from training pairs.

    A feature is "error-indicative" at a layer if it fires in the error text
    but NOT in the clean text, in >= min_pair_ratio of training pairs.

    Args:
        pair_features: list of {"clean": text_features, "error": text_features}
            where text_features = {"tokens": [...], "features": {layer: {fid: [...]}}}
        layers: which layers to analyze.
        min_pair_ratio: fraction of pairs a feature must be error-only in.

    Returns:
        TrainedClassifier with identified error features.
    """
    n_pairs = len(pair_features)
    min_pairs = max(2, int(n_pairs * min_pair_ratio))

    error_features: dict[int, set[int]] = {}
    for layer in layers:
        counter: Counter[int] = Counter()
        for pf in pair_features:
            clean_fids = set(pf["clean"]["features"].get(layer, {}).keys())
            error_fids = set(pf["error"]["features"].get(layer, {}).keys())
            for fid in error_fids - clean_fids:
                counter[fid] += 1
        error_features[layer] = {fid for fid, cnt in counter.items() if cnt >= min_pairs}

    return TrainedClassifier(error_features=error_features, layers=layers)


# --------------- Prediction ---------------

@dataclass
class Prediction:
    """Result of running the classifier on a single text."""
    has_errors: bool
    total_hits: int
    hits_per_layer: dict[int, int] = field(default_factory=dict)


def predict(text_features: dict, classifier: TrainedClassifier) -> Prediction:
    """Predict whether a text contains errors.

    Args:
        text_features: output of extract_text_features() —
            {"tokens": [...], "features": {layer: {fid: [...]}}}
        classifier: trained classifier.
    """
    hits_per_layer = {}
    total_hits = 0
    for layer in classifier.layers:
        text_fids = set(text_features["features"].get(layer, {}).keys())
        hits = text_fids & classifier.error_features.get(layer, set())
        hits_per_layer[layer] = len(hits)
        total_hits += len(hits)

    return Prediction(
        has_errors=total_hits >= classifier.min_active_features,
        total_hits=total_hits,
        hits_per_layer=hits_per_layer,
    )


# --------------- Evaluation ---------------

@dataclass
class Metrics:
    """Classification metrics from evaluating on test pairs."""
    tp: int = 0  # error text correctly detected
    fp: int = 0  # clean text wrongly flagged
    tn: int = 0  # clean text correctly passed
    fn: int = 0  # error text missed

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def confusion_str(self) -> str:
        return (
            f"              Predicted\n"
            f"              Error  Clean\n"
            f"  Actual Error  {self.tp:>4}  {self.fn:>4}\n"
            f"  Actual Clean  {self.fp:>4}  {self.tn:>4}"
        )

    def summary(self) -> str:
        return (
            f"Accuracy={self.accuracy:.1%}  Precision={self.precision:.1%}  "
            f"Recall={self.recall:.1%}  F1={self.f1:.1%}  "
            f"(TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn})"
        )

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def evaluate(test_pair_features: list[dict], classifier: TrainedClassifier) -> Metrics:
    """Evaluate classifier on test pairs.

    For each pair:
      - error text → should predict has_errors=True  (TP if correct, FN if missed)
      - clean text → should predict has_errors=False (TN if correct, FP if flagged)
    """
    metrics = Metrics()
    for pf in test_pair_features:
        error_pred = predict(pf["error"], classifier)
        if error_pred.has_errors:
            metrics.tp += 1
        else:
            metrics.fn += 1

        clean_pred = predict(pf["clean"], classifier)
        if clean_pred.has_errors:
            metrics.fp += 1
        else:
            metrics.tn += 1

    return metrics

    return result
