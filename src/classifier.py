"""SAE-based multi-class error classifier.

Three-phase design:
  1. select_features()  — find SAE features that indicate errors
  2. train()            — train logistic regression on token-level activation vectors
  3. predict_sentence() — score each token, aggregate to sentence-level

The classifier outputs per-token probabilities for 6 error types + clean.
Sentence-level prediction uses max P(any error) across tokens.
"""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import ErrorType, TextPair, token_to_word_index


# --------------- Label mapping ---------------

LABEL_CLEAN = 0
ERROR_TYPE_TO_LABEL = {
    ErrorType.SPELLING: 1,
    ErrorType.WORD_CHOICE: 2,
    ErrorType.GRAMMAR: 3,
    ErrorType.WORD_ORDER: 4,
    ErrorType.MISSING_WORD: 5,
    ErrorType.EXTRA_WORD: 6,
}
LABEL_TO_ERROR_TYPE = {v: k for k, v in ERROR_TYPE_TO_LABEL.items()}


# --------------- Feature selection ---------------

def select_features(
    pair_features: list[dict],
    layers: list[int],
    train_pairs: list[TextPair],
    min_pair_ratio: float = 0.5,
) -> dict[int, set[int]]:
    """Find SAE features that are error-indicative, per error type.

    A feature is selected if it fires in error but NOT clean text in
    >= min_pair_ratio of pairs *of at least one error type*. This prevents
    dilution when different error types activate different features.
    """
    from src.data import ErrorType

    # Group pair indices by error type
    type_indices: dict[ErrorType, list[int]] = {}
    for i, pair in enumerate(train_pairs):
        type_indices.setdefault(pair.error_type, []).append(i)

    error_features: dict[int, set[int]] = {layer: set() for layer in layers}

    for et, indices in type_indices.items():
        n_type = len(indices)
        min_pairs = max(2, int(n_type * min_pair_ratio))

        for layer in layers:
            counter: Counter[int] = Counter()
            for idx in indices:
                pf = pair_features[idx]
                clean_fids = set(pf["clean"]["features"].get(layer, {}).keys())
                error_fids = set(pf["error"]["features"].get(layer, {}).keys())
                for fid in error_fids - clean_fids:
                    counter[fid] += 1
            selected = {fid for fid, cnt in counter.items() if cnt >= min_pairs}
            error_features[layer] |= selected

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
    """Train multi-class token-level logistic regression.

    Labels: 0=clean, 1-6 for each ErrorType.
    """
    feature_index = _build_feature_index(error_features)
    n_features = len(feature_index)

    if n_features == 0:
        raise ValueError("No error features selected — cannot train classifier")

    X_rows = []
    y_rows = []

    for pf, pair in zip(train_pair_features, train_pairs):
        # Error text: label tokens by their error type
        error_feats = pf["error"]
        error_tokens = error_feats["tokens"]
        word_map = token_to_word_index(pair.error, error_tokens)

        for pos in range(len(error_tokens)):
            vec = _token_activation_vector(error_feats, pos, feature_index)
            word_idx = word_map[pos]
            if word_idx is not None and word_idx in pair.error_word_labels:
                label = ERROR_TYPE_TO_LABEL[pair.error_word_labels[word_idx]]
            else:
                label = LABEL_CLEAN
            X_rows.append(vec)
            y_rows.append(label)

        # Clean text: all tokens are label=0
        clean_feats = pf["clean"]
        clean_tokens = clean_feats["tokens"]
        for pos in range(len(clean_tokens)):
            vec = _token_activation_vector(clean_feats, pos, feature_index)
            X_rows.append(vec)
            y_rows.append(LABEL_CLEAN)

    X = np.array(X_rows)
    y = np.array(y_rows)

    n_error = (y != LABEL_CLEAN).sum()
    type_counts = {LABEL_TO_ERROR_TYPE[l].value: (y == l).sum()
                   for l in range(1, 7) if (y == l).sum() > 0}
    print(f"  Training LR on {len(X)} tokens ({n_error} error, {len(y) - n_error} clean)")
    print(f"  Per-type tokens: {type_counts}")

    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X, y)

    return TrainedClassifier(
        error_features=error_features,
        feature_index=feature_index,
        layers=layers,
        model=lr,
    )


# --------------- Shared dataclasses ---------------

@dataclass
class TokenPrediction:
    """Per-token error probability with type breakdown."""
    position: int
    token: str
    p_error: float
    error_probs: dict[ErrorType, float] = field(default_factory=dict)
    predicted_type: ErrorType | None = None
    word_index: int | None = None


@dataclass
class SentencePrediction:
    """Sentence-level prediction with token details."""
    has_errors: bool
    max_p_error: float
    predicted_type: ErrorType | None = None
    token_predictions: list[TokenPrediction] = field(default_factory=list)


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


# --------------- OVR Training ---------------

@dataclass
class OVRClassifier:
    """One-vs-rest: independent binary LR per error type."""
    error_features: dict[int, set[int]]
    feature_index: list[tuple[int, int]]
    layers: list[int]
    models: dict[ErrorType, LogisticRegression] = field(default_factory=dict)
    thresholds: dict[ErrorType, float] = field(default_factory=dict)

    @property
    def total_features(self) -> int:
        return len(self.feature_index)

    def summary(self) -> str:
        parts = [f"layer {l}: {len(self.error_features.get(l, set()))} features"
                 for l in self.layers]
        types = ", ".join(et.value for et in self.models)
        return (f"OVRClassifier({', '.join(parts)}, total={self.total_features}, "
                f"types=[{types}])")


def train_ovr(
    train_pair_features: list[dict],
    train_pairs: list[TextPair],
    layers: list[int],
    error_features: dict[int, set[int]],
) -> OVRClassifier:
    """Train one binary LR per error type.

    For each type, label=1 for tokens of that type, label=0 for everything else.
    """
    feature_index = _build_feature_index(error_features)
    if not feature_index:
        raise ValueError("No error features selected — cannot train classifier")

    # Build token dataset once: (vector, word_idx, error_type_or_None) per token
    token_data: list[tuple[np.ndarray, ErrorType | None]] = []

    for pf, pair in zip(train_pair_features, train_pairs):
        # Error text tokens
        error_feats = pf["error"]
        error_tokens = error_feats["tokens"]
        word_map = token_to_word_index(pair.error, error_tokens)
        for pos in range(len(error_tokens)):
            vec = _token_activation_vector(error_feats, pos, feature_index)
            word_idx = word_map[pos]
            if word_idx is not None and word_idx in pair.error_word_labels:
                token_data.append((vec, pair.error_word_labels[word_idx]))
            else:
                token_data.append((vec, None))

        # Clean text tokens — all None (clean)
        clean_feats = pf["clean"]
        for pos in range(len(clean_feats["tokens"])):
            vec = _token_activation_vector(clean_feats, pos, feature_index)
            token_data.append((vec, None))

    X_all = np.array([td[0] for td in token_data])
    labels_all = [td[1] for td in token_data]

    print(f"  OVR training on {len(X_all)} tokens")

    models = {}
    for et in ErrorType:
        y = np.array([1 if lbl == et else 0 for lbl in labels_all])
        n_pos = y.sum()
        if n_pos < 2:
            print(f"  {et.value}: skipped (only {n_pos} positive tokens)")
            continue
        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        lr.fit(X_all, y)
        print(f"  {et.value}: {n_pos} positive tokens, trained")
        models[et] = lr

    return OVRClassifier(
        error_features=error_features,
        feature_index=feature_index,
        layers=layers,
        models=models,
        thresholds={et: 0.5 for et in models},
    )


# --------------- OVR Prediction ---------------

def predict_tokens_ovr(
    text_features: dict,
    classifier: OVRClassifier,
) -> list[TokenPrediction]:
    """Score each token with all OVR classifiers."""
    tokens = text_features["tokens"]
    predictions = []

    for pos in range(len(tokens)):
        vec = _token_activation_vector(text_features, pos, classifier.feature_index)
        vec_2d = vec.reshape(1, -1)

        error_probs = {}
        for et, model in classifier.models.items():
            p = float(model.predict_proba(vec_2d)[0, 1])
            error_probs[et] = p

        # p_error = max probability across all types
        p_error = max(error_probs.values()) if error_probs else 0.0
        predicted_type = max(error_probs, key=error_probs.get) if error_probs else None

        predictions.append(TokenPrediction(
            position=pos,
            token=tokens[pos],
            p_error=float(p_error),
            error_probs=error_probs,
            predicted_type=predicted_type,
        ))

    return predictions


def predict_sentence_ovr(
    text_features: dict,
    classifier: OVRClassifier,
) -> SentencePrediction:
    """Predict using OVR classifiers with per-type thresholds.

    A sentence has errors if any token exceeds that type's threshold.
    """
    token_preds = predict_tokens_ovr(text_features, classifier)
    if not token_preds:
        return SentencePrediction(has_errors=False, max_p_error=0.0)

    # Check each type's threshold independently
    best_type = None
    best_p = 0.0
    for tp in token_preds:
        for et, p in tp.error_probs.items():
            if p > best_p:
                best_p = p
                best_type = et

    # Has errors if the best score exceeds its type's threshold
    has_errors = best_type is not None and best_p >= classifier.thresholds.get(best_type, 0.5)

    max_pred = max(token_preds, key=lambda t: t.p_error)
    return SentencePrediction(
        has_errors=has_errors,
        max_p_error=max_pred.p_error,
        predicted_type=best_type if has_errors else None,
        token_predictions=token_preds,
    )


def evaluate_ovr(
    test_pair_features: list[dict],
    classifier: OVRClassifier,
) -> Metrics:
    """Evaluate OVR classifier on test pairs (binary: error vs clean)."""
    metrics = Metrics()
    for pf in test_pair_features:
        error_pred = predict_sentence_ovr(pf["error"], classifier)
        if error_pred.has_errors:
            metrics.tp += 1
        else:
            metrics.fn += 1

        clean_pred = predict_sentence_ovr(pf["clean"], classifier)
        if clean_pred.has_errors:
            metrics.fp += 1
        else:
            metrics.tn += 1

    return metrics


def predict_tokens(
    text_features: dict,
    classifier: TrainedClassifier,
) -> list[TokenPrediction]:
    """Score each token for P(error) and per-type probabilities."""
    tokens = text_features["tokens"]
    classes = classifier.model.classes_
    predictions = []

    for pos in range(len(tokens)):
        vec = _token_activation_vector(text_features, pos, classifier.feature_index)
        probas = classifier.model.predict_proba(vec.reshape(1, -1))[0]

        error_probs = {}
        p_clean = 0.0
        for i, cls in enumerate(classes):
            if cls == LABEL_CLEAN:
                p_clean = probas[i]
            elif cls in LABEL_TO_ERROR_TYPE:
                error_probs[LABEL_TO_ERROR_TYPE[cls]] = float(probas[i])

        p_error = 1.0 - p_clean
        predicted_type = max(error_probs, key=error_probs.get) if error_probs else None

        predictions.append(TokenPrediction(
            position=pos,
            token=tokens[pos],
            p_error=float(p_error),
            error_probs=error_probs,
            predicted_type=predicted_type,
        ))

    return predictions


def predict_sentence(
    text_features: dict,
    classifier: TrainedClassifier,
) -> SentencePrediction:
    """Predict whether a text contains errors. Uses max token P(error)."""
    token_preds = predict_tokens(text_features, classifier)
    if not token_preds:
        return SentencePrediction(has_errors=False, max_p_error=0.0)
    max_pred = max(token_preds, key=lambda t: t.p_error)
    has_errors = max_pred.p_error >= classifier.sentence_threshold
    return SentencePrediction(
        has_errors=has_errors,
        max_p_error=max_pred.p_error,
        predicted_type=max_pred.predicted_type if has_errors else None,
        token_predictions=token_preds,
    )


# --------------- Evaluation (multi-class) ---------------

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
