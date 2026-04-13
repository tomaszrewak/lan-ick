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


def select_features_per_type(
    pair_features: list[dict],
    layers: list[int],
    train_pairs: list[TextPair],
    min_pair_ratio: float = 0.5,
) -> dict[ErrorType, dict[int, set[int]]]:
    """Select SAE features independently for each error type.

    Returns a dict mapping each ErrorType to its own {layer: set(fid)} dict.
    """
    from src.data import ErrorType

    type_indices: dict[ErrorType, list[int]] = {}
    for i, pair in enumerate(train_pairs):
        type_indices.setdefault(pair.error_type, []).append(i)

    per_type: dict[ErrorType, dict[int, set[int]]] = {}

    for et, indices in type_indices.items():
        n_type = len(indices)
        min_pairs = max(2, int(n_type * min_pair_ratio))
        feats: dict[int, set[int]] = {layer: set() for layer in layers}

        for layer in layers:
            counter: Counter[int] = Counter()
            for idx in indices:
                pf = pair_features[idx]
                clean_fids = set(pf["clean"]["features"].get(layer, {}).keys())
                error_fids = set(pf["error"]["features"].get(layer, {}).keys())
                for fid in error_fids - clean_fids:
                    counter[fid] += 1
            feats[layer] = {fid for fid, cnt in counter.items() if cnt >= min_pairs}

        per_type[et] = feats

    return per_type


# --------------- Alternative feature selection methods ---------------

PerTypeFeatures = dict[ErrorType, dict[int, set[int]]]


def _get_token_activations(text_features: dict, layer: int, pos: int) -> dict[int, float]:
    """Get {fid: activation} for a specific token position in a layer."""
    result = {}
    for fid, hits in text_features["features"].get(layer, {}).items():
        for p, act, _tok in hits:
            if p == pos:
                result[fid] = act
                break
    return result


def _get_error_token_positions(pair: TextPair, text_features: dict) -> list[int]:
    """Get token positions that correspond to error words."""
    tokens = text_features["tokens"]
    word_map = token_to_word_index(pair.error, tokens)
    positions = []
    for pos in range(len(tokens)):
        word_idx = word_map[pos]
        if word_idx is not None and word_idx in pair.error_word_labels:
            positions.append(pos)
    return positions


def select_features_relaxed(
    pair_features: list[dict],
    layers: list[int],
    train_pairs: list[TextPair],
    min_pair_ratio: float = 0.3,
) -> PerTypeFeatures:
    """Same as select_features_per_type but with a lower threshold."""
    return select_features_per_type(pair_features, layers, train_pairs, min_pair_ratio)


def select_features_paired_token_diff(
    pair_features: list[dict],
    layers: list[int],
    train_pairs: list[TextPair],
    min_pair_ratio: float = 0.3,
) -> PerTypeFeatures:
    """Compare activations at error word positions: error text vs clean text.

    For each pair, at each error word's token position, find features that
    activate MORE in the error text than the clean text at the SAME position.
    A feature is selected if this happens in >= min_pair_ratio of the type's pairs.
    """
    type_indices: dict[ErrorType, list[int]] = {}
    for i, pair in enumerate(train_pairs):
        type_indices.setdefault(pair.error_type, []).append(i)

    per_type: PerTypeFeatures = {}

    for et, indices in type_indices.items():
        n_type = len(indices)
        min_pairs = max(2, int(n_type * min_pair_ratio))
        feats: dict[int, set[int]] = {layer: set() for layer in layers}

        for layer in layers:
            # Count: how many pairs have this feature activate more at error positions?
            counter: Counter[int] = Counter()
            for idx in indices:
                pf = pair_features[idx]
                pair = train_pairs[idx]
                error_positions = _get_error_token_positions(pair, pf["error"])

                # For each error position, compare error vs clean activations
                diff_fids: set[int] = set()
                for pos in error_positions:
                    error_acts = _get_token_activations(pf["error"], layer, pos)
                    # Use same position in clean text (may not align perfectly but
                    # for most error types the token count is similar)
                    clean_acts = _get_token_activations(pf["clean"], layer, pos) \
                        if pos < len(pf["clean"]["tokens"]) else {}

                    for fid, e_act in error_acts.items():
                        c_act = clean_acts.get(fid, 0.0)
                        if e_act > c_act:
                            diff_fids.add(fid)

                for fid in diff_fids:
                    counter[fid] += 1

            feats[layer] = {fid for fid, cnt in counter.items() if cnt >= min_pairs}

        per_type[et] = feats

    return per_type


def select_features_magnitude_diff(
    pair_features: list[dict],
    layers: list[int],
    train_pairs: list[TextPair],
    top_k: int = 20,
) -> PerTypeFeatures:
    """Select features with largest mean activation difference (error - clean).

    For each error type, compute per-feature mean activation across error word
    tokens minus mean activation across clean tokens. Take top_k features.
    """
    type_indices: dict[ErrorType, list[int]] = {}
    for i, pair in enumerate(train_pairs):
        type_indices.setdefault(pair.error_type, []).append(i)

    per_type: PerTypeFeatures = {}

    for et, indices in type_indices.items():
        feats: dict[int, set[int]] = {layer: set() for layer in layers}

        for layer in layers:
            # Collect activation sums for error word tokens and clean tokens
            error_sums: Counter[int] = Counter()
            error_counts: Counter[int] = Counter()
            clean_sums: Counter[int] = Counter()
            clean_counts: Counter[int] = Counter()

            for idx in indices:
                pf = pair_features[idx]
                pair = train_pairs[idx]
                error_positions = set(_get_error_token_positions(pair, pf["error"]))

                # Error word token activations
                for fid, hits in pf["error"]["features"].get(layer, {}).items():
                    for pos, act, _tok in hits:
                        if pos in error_positions:
                            error_sums[fid] += act
                            error_counts[fid] += 1

                # All clean token activations (from same pairs)
                for fid, hits in pf["clean"]["features"].get(layer, {}).items():
                    for pos, act, _tok in hits:
                        clean_sums[fid] += act
                        clean_counts[fid] += 1

            # Compute mean diff for each feature seen in error text
            all_fids = set(error_sums.keys()) | set(clean_sums.keys())
            diffs = {}
            for fid in all_fids:
                e_mean = error_sums[fid] / error_counts[fid] if error_counts[fid] else 0
                c_mean = clean_sums[fid] / clean_counts[fid] if clean_counts[fid] else 0
                diffs[fid] = e_mean - c_mean

            # Take top_k with positive diff
            sorted_fids = sorted(
                [(fid, d) for fid, d in diffs.items() if d > 0],
                key=lambda x: x[1], reverse=True,
            )
            feats[layer] = {fid for fid, _ in sorted_fids[:top_k]}

        per_type[et] = feats

    return per_type


def select_features_ttest(
    pair_features: list[dict],
    layers: list[int],
    train_pairs: list[TextPair],
    p_threshold: float = 0.05,
    min_samples: int = 5,
) -> PerTypeFeatures:
    """Select features via Welch's t-test on error-word vs clean activations.

    For each feature, collect activation values at error word positions
    vs activation values at all clean text positions. Select features
    where error activations are significantly higher (one-sided).
    """
    from scipy import stats

    type_indices: dict[ErrorType, list[int]] = {}
    for i, pair in enumerate(train_pairs):
        type_indices.setdefault(pair.error_type, []).append(i)

    per_type: PerTypeFeatures = {}

    for et, indices in type_indices.items():
        feats: dict[int, set[int]] = {layer: set() for layer in layers}

        for layer in layers:
            # Collect per-feature activation lists
            error_acts: dict[int, list[float]] = {}
            clean_acts: dict[int, list[float]] = {}

            for idx in indices:
                pf = pair_features[idx]
                pair = train_pairs[idx]
                error_positions = set(_get_error_token_positions(pair, pf["error"]))
                n_clean_tokens = len(pf["clean"]["tokens"])

                # Error text: collect activations at error word positions
                for fid, hits in pf["error"]["features"].get(layer, {}).items():
                    for pos, act, _tok in hits:
                        if pos in error_positions:
                            error_acts.setdefault(fid, []).append(act)

                # Clean text: collect all activations
                for fid, hits in pf["clean"]["features"].get(layer, {}).items():
                    for pos, act, _tok in hits:
                        clean_acts.setdefault(fid, []).append(act)

            # Also add zeros for error positions where feature didn't fire
            # (to avoid selection bias toward rare high-activation features)
            # Skip this for simplicity — we're comparing "when it fires"

            # T-test: error activations > clean activations?
            all_fids = set(error_acts.keys())
            for fid in all_fids:
                e = error_acts.get(fid, [])
                c = clean_acts.get(fid, [])
                if len(e) < min_samples or len(c) < min_samples:
                    continue
                # One-sided: error > clean
                t_stat, p_val = stats.ttest_ind(e, c, equal_var=False, alternative="greater")
                if p_val < p_threshold:
                    feats[layer].add(fid)

        per_type[et] = feats

    return per_type


def select_features_top_k_error(
    pair_features: list[dict],
    layers: list[int],
    train_pairs: list[TextPair],
    top_k: int = 20,
) -> PerTypeFeatures:
    """Select features with highest mean activation at error word positions.

    No comparison to clean text — purely based on which features fire
    most strongly at error positions. May capture features that fire in
    both error and clean text but are still informative for the classifier.
    """
    type_indices: dict[ErrorType, list[int]] = {}
    for i, pair in enumerate(train_pairs):
        type_indices.setdefault(pair.error_type, []).append(i)

    per_type: PerTypeFeatures = {}

    for et, indices in type_indices.items():
        feats: dict[int, set[int]] = {layer: set() for layer in layers}

        for layer in layers:
            activation_sums: Counter[int] = Counter()
            activation_counts: Counter[int] = Counter()

            for idx in indices:
                pf = pair_features[idx]
                pair = train_pairs[idx]
                error_positions = set(_get_error_token_positions(pair, pf["error"]))

                for fid, hits in pf["error"]["features"].get(layer, {}).items():
                    for pos, act, _tok in hits:
                        if pos in error_positions:
                            activation_sums[fid] += act
                            activation_counts[fid] += 1

            # Rank by mean activation, take top_k
            means = {
                fid: activation_sums[fid] / activation_counts[fid]
                for fid in activation_counts
            }
            sorted_fids = sorted(means.items(), key=lambda x: x[1], reverse=True)
            feats[layer] = {fid for fid, _ in sorted_fids[:top_k]}

        per_type[et] = feats

    return per_type


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
    """One-vs-rest: independent binary LR per error type.

    Each type has its own feature set and feature index.
    """
    per_type_features: dict[ErrorType, dict[int, set[int]]]
    per_type_index: dict[ErrorType, list[tuple[int, int]]]
    layers: list[int]
    models: dict[ErrorType, LogisticRegression] = field(default_factory=dict)
    thresholds: dict[ErrorType, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        for et in self.models:
            n = len(self.per_type_index.get(et, []))
            lines.append(f"{et.value}: {n} features")
        return f"OVRClassifier({', '.join(lines)})"


def train_ovr(
    train_pair_features: list[dict],
    train_pairs: list[TextPair],
    layers: list[int],
    per_type_features: dict[ErrorType, dict[int, set[int]]],
) -> OVRClassifier:
    """Train one binary LR per error type, each with its own feature set.

    For each type, label=1 for tokens of that type, label=0 for everything else.
    Each classifier sees only features selected for its type.
    """
    per_type_index = {
        et: _build_feature_index(feats)
        for et, feats in per_type_features.items()
    }

    # Count total tokens for logging
    n_tokens = sum(
        len(pf["error"]["tokens"]) + len(pf["clean"]["tokens"])
        for pf in train_pair_features
    )
    print(f"  OVR training on {n_tokens} tokens")

    models = {}
    for et in ErrorType:
        feat_index = per_type_index.get(et, [])
        if not feat_index:
            print(f"  {et.value}: skipped (0 features selected)")
            continue

        # Build feature vectors using this type's feature index
        X = []
        y = []
        for pf, pair in zip(train_pair_features, train_pairs):
            error_feats = pf["error"]
            error_tokens = error_feats["tokens"]
            word_map = token_to_word_index(pair.error, error_tokens)
            for pos in range(len(error_tokens)):
                vec = _token_activation_vector(error_feats, pos, feat_index)
                word_idx = word_map[pos]
                if word_idx is not None and word_idx in pair.error_word_labels:
                    is_this_type = pair.error_word_labels[word_idx] == et
                else:
                    is_this_type = False
                X.append(vec)
                y.append(1 if is_this_type else 0)

            clean_feats = pf["clean"]
            for pos in range(len(clean_feats["tokens"])):
                vec = _token_activation_vector(clean_feats, pos, feat_index)
                X.append(vec)
                y.append(0)

        X_arr = np.array(X)
        y_arr = np.array(y)
        n_pos = y_arr.sum()
        if n_pos < 2:
            print(f"  {et.value}: skipped (only {n_pos} positive tokens)")
            continue

        lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        lr.fit(X_arr, y_arr)
        print(f"  {et.value}: {n_pos} positive tokens, {len(feat_index)} features, trained")
        models[et] = lr

    return OVRClassifier(
        per_type_features=per_type_features,
        per_type_index=per_type_index,
        layers=layers,
        models=models,
        thresholds={et: 0.5 for et in models},
    )


# --------------- OVR Prediction ---------------

def predict_tokens_ovr(
    text_features: dict,
    classifier: OVRClassifier,
) -> list[TokenPrediction]:
    """Score each token with all OVR classifiers, each using its own features."""
    tokens = text_features["tokens"]
    n_tokens = len(tokens)

    # Pre-build position lookup: pos_acts[layer][fid][pos] = activation
    pos_acts: dict[int, dict[int, dict[int, float]]] = {}
    for layer, fid_dict in text_features["features"].items():
        pos_acts[layer] = {}
        for fid, hits in fid_dict.items():
            pos_acts[layer][fid] = {pos: act for pos, act, _tok in hits}

    # For each error type, build full token matrix and batch predict
    all_probs: dict = {}  # et -> np.ndarray of shape (n_tokens,)
    for et, model in classifier.models.items():
        feat_index = classifier.per_type_index[et]
        n_feats = len(feat_index)
        X = np.zeros((n_tokens, n_feats), dtype=np.float32)
        for i, (layer, fid) in enumerate(feat_index):
            fid_pos = pos_acts.get(layer, {}).get(fid, {})
            for pos in range(n_tokens):
                if pos in fid_pos:
                    X[pos, i] = fid_pos[pos]
        all_probs[et] = model.predict_proba(X)[:, 1]

    predictions = []
    for pos in range(n_tokens):
        error_probs = {et: float(all_probs[et][pos]) for et in classifier.models}
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
