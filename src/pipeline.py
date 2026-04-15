"""Shared pipeline: build classifier from cached data, run inference on text."""

from src.cache import cached
from src.data import (
    ErrorType, generate_synthetic_pairs, split_train_test,
    token_to_word_index, last_token_positions,
)
from src.model import extract_text_features
from src.classifier import (
    OVRClassifier, select_features_position_aware_topn,
    train_ovr, predict_tokens_ovr,
)

# --------------- Default parameters ---------------

LAYERS = [7, 13, 17, 22]
N_PAIRS = 6000
MIN_WORDS = 8
MAX_WORDS = 20
DATA_SEED = 42
TRAIN_RATIO = 0.75
SPLIT_SEED = 42
DATA_VERSION = "v6"
EXTRACT_VERSION = "v3"
DATA_CACHE_KEY = f"{DATA_VERSION}_n{N_PAIRS}"
EXTRACT_CACHE_KEY = f"{EXTRACT_VERSION}_{DATA_VERSION}_n{N_PAIRS}_layers={'_'.join(map(str, LAYERS))}_w16k"
TOP_N = 100
FP_BUDGET = 0.05


# --------------- Load data ---------------

def load_data():
    """Load pairs and features from cache (or generate/extract if needed).

    Returns (all_pairs, all_features).
    """
    all_pairs = cached(
        "synthetic_pairs", DATA_CACHE_KEY,
        lambda: generate_synthetic_pairs(N_PAIRS, MIN_WORDS, MAX_WORDS, DATA_SEED),
    )

    def extract_all():
        results = []
        for i, pair in enumerate(all_pairs):
            print(f"  [{i+1}/{len(all_pairs)}] {pair.error[:50]}...")
            clean_feats = extract_text_features(pair.clean, LAYERS)
            error_feats = extract_text_features(pair.error, LAYERS)
            results.append({"clean": clean_feats, "error": error_feats})
        return results

    all_features = cached("pair_features", EXTRACT_CACHE_KEY, extract_all)
    return all_pairs, all_features


# --------------- Build classifier ---------------

def build_classifier() -> OVRClassifier:
    """Load cached data, select features, train OVR, compute thresholds.

    Returns a ready-to-use OVRClassifier with per-type thresholds set.
    """
    all_pairs, all_features = load_data()
    train_idx, test_idx = split_train_test(all_pairs, TRAIN_RATIO, SPLIT_SEED)
    train_pairs = [all_pairs[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    train_features = [all_features[i] for i in train_idx]
    test_features = [all_features[i] for i in test_idx]

    per_type_feats = select_features_position_aware_topn(
        train_features, LAYERS, train_pairs, top_n=TOP_N,
    )

    classifier = train_ovr(train_features, train_pairs, LAYERS, per_type_feats)

    # Compute thresholds on test clean texts (FP budget)
    clean_scores: list[dict[ErrorType, float]] = []
    for pf, pair in zip(test_features, test_pairs):
        tpreds = predict_tokens_ovr(pf["clean"], classifier, text=pair.clean)
        scores = {}
        for et in classifier.models:
            scores[et] = max((tp.error_probs.get(et, 0.0) for tp in tpreds), default=0.0)
        clean_scores.append(scores)

    n_clean = len(clean_scores)
    max_fp = int(n_clean * FP_BUDGET)
    for et in classifier.models:
        best_t = 1.0
        for t_int in range(50, 100):
            t = t_int / 100
            fp_count = sum(1 for cs in clean_scores if cs.get(et, 0) >= t)
            if fp_count <= max_fp:
                best_t = t
                break
        classifier.thresholds[et] = best_t

    return classifier, all_pairs, all_features, train_idx, test_idx


# --------------- Inference ---------------

def check_text(text: str, classifier: OVRClassifier) -> dict:
    """Run error detection on text, return per-word results.

    Returns {"words": [{"word": "...", "errors": {"spelling": 0.95, ...}}, ...]}
    """
    if not text.strip():
        return {"words": []}

    features = extract_text_features(text, LAYERS)
    token_preds = predict_tokens_ovr(features, classifier, text=text)

    words = text.split()
    word_map = token_to_word_index(text, features["tokens"])
    last_tok = last_token_positions(word_map)

    word_results = []
    for w_idx, word in enumerate(words):
        tok_pos = last_tok.get(w_idx)
        errors = {}
        if tok_pos is not None:
            for tp in token_preds:
                if tp.position == tok_pos:
                    for et, prob in tp.error_probs.items():
                        threshold = classifier.thresholds.get(et, 1.0)
                        if prob >= threshold:
                            errors[et.value] = round(prob, 3)
                    break
        word_results.append({"word": word, "errors": errors})

    return {"words": word_results}
