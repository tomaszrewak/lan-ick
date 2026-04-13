"""SAE-based token-level spelling error classifier.

Three-phase design:
  1. select_features()  — find SAE features that indicate errors
  2. train()            — train logistic regression on token-level activation vectors
  3. predict_sentence() — score each token, aggregate to sentence-level P(error)

The classifier takes SAE activations at selected features as input and outputs
P(error) ∈ [0, 1] per token. Sentence-level prediction uses max token score.

Single output for now; designed so adding more outputs (spelling vs grammar)
later is just adding a head.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import TextPair, token_to_word_index


# --------------- Feature selection ---------------

def select_features(
    pair_features: list[dict],
    layers: list[int],
    min_pair_ratio: float = 0.5,
) -> dict[int, set[int]]:
    """Find SAE features that are error-indicative.

    A feature is "error-indicative" at a layer if it fires in the error text
    but NOT in the clean text, in >= min_pair_ratio of training pairs.
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

    return error_features


def _build_feature_index(error_features: dict[int, set[int]]) -> list[tuple[int, int]]:
    """Build ordered list of (layer, feature_id) for consistent vector indexing."""
    index = []
    for layer in sorted(error_features.keys()):
        for fid in sorted(error_features[layer]):
            index.append((layer, fid))
    return index


# --------------- Token activation vectors ---------------

def _token_activation_vector(
    text_features: dict,
    token_pos: int,
    feature_index: list[tuple[int, int]],
) -> np.ndarray:
    """Build activation vector for a single token position.

    Returns array of shape (n_features,) with activation values.
    """
    vec = np.zeros(len(feature_index), dtype=np.float32)
    for i, (layer, fid) in enumerate(feature_index):
        hits = text_features["features"].get(layer, {}).get(fid, [])
        for pos, act, _tok in hits:
            if pos == token_pos:
                vec[i] = act
                break
    return vec


# --------------- Training ---------------

@dataclass
class TrainedClassifier:
    """Token-level logistic regression error classifier."""
    error_features: dict[int, set[int]]
    feature_index: list[tuple[int, int]]
    layers: list[int]
    model: LogisticRegression = None
    sentence_threshold: float = 0.5

    @property
    def total_features(self) -> int:
        return len(self.feature_index)

    def summary(self) -> str:
        parts = [f"layer {l}: {len(self.error_features.get(l, set()))} features"
                 for l in self.layers]
        return (f"TrainedClassifier({', '.join(parts)}, "
                f"total={self.total_features}, threshold={self.sentence_threshold})")


def train(
    train_pair_features: list[dict],
    train_pairs: list[TextPair],
    layers: list[int],
    error_features: dict[int, set[int]],
) -> TrainedClassifier:
    """Train token-level logistic regression.

    Collects token activation vectors from error texts, labels tokens
    belonging to corrupted words as 1, others as 0. Fits LR.
    """
    feature_index = _build_feature_index(error_features)
    n_features = len(feature_index)

    if n_features == 0:
        raise ValueError("No error features selected — cannot train classifier")

    X_rows = []
    y_rows = []

    for pf, pair in zip(train_pair_features, train_pairs):
        # Error text: label tokens by whether their word was corrupted
        error_feats = pf["error"]
        error_tokens = error_feats["tokens"]
        word_map = token_to_word_index(pair.error, error_tokens)
        error_word_set = set(pair.error_word_indices)

        for pos in range(len(error_tokens)):
            vec = _token_activation_vector(error_feats, pos, feature_index)
            word_idx = word_map[pos]
            label = 1 if (word_idx is not None and word_idx in error_word_set) else 0
            X_rows.append(vec)
            y_rows.append(label)

        # Clean text: all tokens are label=0
        clean_feats = pf["clean"]
        clean_tokens = clean_feats["tokens"]
        for pos in range(len(clean_tokens)):
            vec = _token_activation_vector(clean_feats, pos, feature_index)
            X_rows.append(vec)
            y_rows.append(0)

    X = np.array(X_rows)
    y = np.array(y_rows)

    print(f"  Training LR on {len(X)} tokens ({y.sum()} error, {len(y) - y.sum()} clean)")

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X, y)

    return TrainedClassifier(
        error_features=error_features,
        feature_index=feature_index,
        layers=layers,
        model=lr,
    )


# --------------- Prediction ---------------

@dataclass
class TokenPrediction:
    """Per-token error probability."""
    position: int
    token: str
    p_error: float
    word_index: int | None = None


@dataclass
class SentencePrediction:
    """Sentence-level prediction with token details."""
    has_errors: bool
    max_p_error: float
    token_predictions: list[TokenPrediction] = field(default_factory=list)


def predict_tokens(
    text_features: dict,
    classifier: TrainedClassifier,
) -> list[TokenPrediction]:
    """Score each token in a text for P(error)."""
    tokens = text_features["tokens"]
    predictions = []

    for pos in range(len(tokens)):
        vec = _token_activation_vector(text_features, pos, classifier.feature_index)
        p_error = classifier.model.predict_proba(vec.reshape(1, -1))[0, 1]
        predictions.append(TokenPrediction(
            position=pos,
            token=tokens[pos],
            p_error=p_error,
        ))

    return predictions


def predict_sentence(
    text_features: dict,
    classifier: TrainedClassifier,
) -> SentencePrediction:
    """Predict whether a text contains errors. Uses max token P(error)."""
    token_preds = predict_tokens(text_features, classifier)
    max_p = max(tp.p_error for tp in token_preds) if token_preds else 0.0
    return SentencePrediction(
        has_errors=max_p >= classifier.sentence_threshold,
        max_p_error=max_p,
        token_predictions=token_preds,
    )


# --------------- Evaluation ---------------

@dataclass
class Metrics:
    """Classification metrics from evaluating on test pairs."""
    tp: int = 0  # error sentence correctly detected
    fp: int = 0  # clean sentence wrongly flagged
    tn: int = 0  # clean sentence correctly passed
    fn: int = 0  # error sentence missed

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


def evaluate(
    test_pair_features: list[dict],
    classifier: TrainedClassifier,
) -> Metrics:
    """Evaluate classifier on test pairs at sentence level.

    For each pair:
      - error text → should predict has_errors=True  (TP if correct, FN if missed)
      - clean text → should predict has_errors=False (TN if correct, FP if flagged)
    """
    metrics = Metrics()
    for pf in test_pair_features:
        error_pred = predict_sentence(pf["error"], classifier)
        if error_pred.has_errors:
            metrics.tp += 1
        else:
            metrics.fn += 1

        clean_pred = predict_sentence(pf["clean"], classifier)
        if clean_pred.has_errors:
            metrics.fp += 1
        else:
            metrics.tn += 1

    return metrics

    return result
