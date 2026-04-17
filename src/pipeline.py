"""Shared pipeline: build classifier from cached data, run inference on text."""

import random

from src.cache import cached
from src.data import (
    ErrorType, generate_synthetic_pairs, split_train_test,
    token_to_word_index, last_token_positions,
)
from src.model import extract_text_features
from src.classifier import (
    OVRClassifier, select_features_position_aware_topn,
    train_ovr, predict_tokens_ovr, calibrate_greedy_f05,
)

# --------------- Default parameters ---------------

LAYERS = [7, 13, 17, 22]
N_PAIRS = 6000
MIN_WORDS = 8
MAX_WORDS = 20
DATA_SEED = 42
TRAIN_RATIO = 0.75
SPLIT_SEED = 42
CALIB_RATIO = 0.2          # fraction of training fold reserved for threshold calibration
CALIB_SEED = 4242
DATA_VERSION = "v10"
EXTRACT_VERSION = "v4"
DATA_CACHE_KEY = f"{DATA_VERSION}_n{N_PAIRS}"
EXTRACT_CACHE_KEY = f"{EXTRACT_VERSION}_{DATA_VERSION}_n{N_PAIRS}_layers={'_'.join(map(str, LAYERS))}_w16k"
TOP_N = 100


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


def calibration_split(train_idx: list[int], calib_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Split a training index list into (fit_idx, calib_idx) deterministically."""
    shuffled = list(train_idx)
    random.Random(seed).shuffle(shuffled)
    n_calib = max(1, int(len(shuffled) * calib_ratio))
    return sorted(shuffled[n_calib:]), sorted(shuffled[:n_calib])


def _score_clean_error(pairs, features, classifier):
    """Return (error_scores, clean_scores) as lists of {ErrorType: max_prob} dicts."""
    error_scores, clean_scores = [], []
    for pf, pair in zip(features, pairs):
        for key, text, out in [("error", pair.error, error_scores),
                               ("clean", pair.clean, clean_scores)]:
            tpreds = predict_tokens_ovr(pf[key], classifier, text=text)
            out.append({
                et: max((tp.error_probs.get(et, 0.0) for tp in tpreds), default=0.0)
                for et in classifier.models
            })
    return error_scores, clean_scores


# --------------- Build classifier ---------------

def build_classifier() -> tuple[OVRClassifier, list, list, list[int], list[int]]:
    """Load cached data, select features, train OVR, calibrate thresholds.

    Training fold is split 80/20 into fit + calibration. Features and LR are
    fit on the fit split; per-type thresholds are set by greedy F0.5 coordinate
    descent on the calibration split (see `calibrate_greedy_f05`). The test
    split is untouched.

    Returns (classifier, all_pairs, all_features, train_idx, test_idx).
    """
    all_pairs, all_features = load_data()
    train_idx, test_idx = split_train_test(all_pairs, TRAIN_RATIO, SPLIT_SEED)
    fit_idx, calib_idx = calibration_split(train_idx, CALIB_RATIO, CALIB_SEED)

    fit_pairs = [all_pairs[i] for i in fit_idx]
    fit_features = [all_features[i] for i in fit_idx]
    calib_pairs = [all_pairs[i] for i in calib_idx]
    calib_features = [all_features[i] for i in calib_idx]

    per_type_feats = select_features_position_aware_topn(
        fit_features, LAYERS, fit_pairs, top_n=TOP_N,
    )
    classifier = train_ovr(fit_features, fit_pairs, LAYERS, per_type_feats)

    calib_err, calib_clean = _score_clean_error(calib_pairs, calib_features, classifier)
    calibrate_greedy_f05(classifier, calib_err, calib_clean)

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
