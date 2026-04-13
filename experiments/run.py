"""Experiment 8: SAE width comparison (16k vs 65k vs 262k).

Compare error detection performance across SAE widths.
Uses layers available in all widths: [7, 13, 17, 22].
Loops over widths, extracting features (cached), training, evaluating.
"""

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.cache import cached
from src.data import ErrorType, generate_synthetic_pairs, split_train_test
from src.model import extract_text_features, clear_sae_cache
from src.classifier import (
    select_features_per_type, train_ovr, predict_tokens_ovr, Metrics,
)

# --------------- Parameters ---------------

LAYERS = [7, 13, 17, 22]  # Available in all widths (res-all + res-subset)
WIDTHS = ["16k", "65k", "262k"]
N_PAIRS = 300
MIN_WORDS = 8
MAX_WORDS = 20
DATA_SEED = 42
TRAIN_RATIO = 0.75
SPLIT_SEED = 42
MIN_PAIR_RATIO = 0.3

DATA_VERSION = "v4"
EXTRACT_VERSION = "v2"

THRESHOLDS = [0.5, 0.8, 0.9, 0.95]


# --------------- Main ---------------

def main():
    # 1. Load data (shared across all widths)
    all_pairs = cached(
        "synthetic_pairs", DATA_VERSION,
        lambda: generate_synthetic_pairs(N_PAIRS, MIN_WORDS, MAX_WORDS, DATA_SEED),
    )
    print(f"Loaded {len(all_pairs)} pairs")

    train_idx, test_idx = split_train_test(all_pairs, TRAIN_RATIO, SPLIT_SEED)
    train_pairs = [all_pairs[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    print(f"Split: {len(train_pairs)} train, {len(test_pairs)} test")

    # Collect results per width for final comparison
    all_width_results: dict[str, dict] = {}

    for width in WIDTHS:
        print(f"\n{'='*70}")
        print(f"WIDTH: {width}")
        print(f"{'='*70}")

        # Clear SAE cache from previous width to free VRAM
        clear_sae_cache()

        # 2. Extract features (cached per width)
        cache_key = f"{EXTRACT_VERSION}_v4_n{N_PAIRS}_layers={'_'.join(map(str, LAYERS))}_w{width}"

        def make_extractor(w):
            def extract_all():
                results = []
                for i, pair in enumerate(all_pairs):
                    print(f"  [{i+1}/{len(all_pairs)}] {pair.error[:50]}...")
                    clean_feats = extract_text_features(pair.clean, LAYERS, w)
                    error_feats = extract_text_features(pair.error, LAYERS, w)
                    results.append({"clean": clean_feats, "error": error_feats})
                return results
            return extract_all

        all_features = cached("pair_features", cache_key, make_extractor(width))
        train_features = [all_features[i] for i in train_idx]
        test_features = [all_features[i] for i in test_idx]

        # 3. Select features per type
        per_type_feats = select_features_per_type(
            train_features, LAYERS, train_pairs, min_pair_ratio=MIN_PAIR_RATIO,
        )
        feat_counts = {}
        for et in ErrorType:
            feats = per_type_feats.get(et, {})
            total = sum(len(v) for v in feats.values())
            feat_counts[et] = total
            by_layer = ", ".join(f"L{l}:{len(feats.get(l, set()))}" for l in LAYERS)
            print(f"  {et.value}: {total} features ({by_layer})")

        # 4. Train OVR classifiers
        classifier = train_ovr(train_features, train_pairs, LAYERS, per_type_feats)

        # 5. Pre-compute scores
        error_scores: list[dict[ErrorType, float]] = []
        clean_scores: list[dict[ErrorType, float]] = []

        for pf in test_features:
            for text_key, scores_list in [("error", error_scores), ("clean", clean_scores)]:
                tpreds = predict_tokens_ovr(pf[text_key], classifier)
                scores = {}
                for et in classifier.models:
                    scores[et] = max((tp.error_probs.get(et, 0.0) for tp in tpreds), default=0.0)
                scores_list.append(scores)

        # 6. Evaluate at all thresholds
        width_result = {}
        for threshold in THRESHOLDS:
            type_metrics = {}
            for et in ErrorType:
                if et not in classifier.models:
                    type_metrics[et] = {"n": 0, "det": 0, "det_pct": 0, "fp": 0, "fp_pct": 0}
                    continue
                type_idx = [i for i, p in enumerate(test_pairs) if p.error_type == et]
                detected = sum(1 for i in type_idx if error_scores[i].get(et, 0) >= threshold)
                fp = sum(1 for cs in clean_scores if cs.get(et, 0) >= threshold)
                n = len(type_idx)
                type_metrics[et] = {
                    "n": n, "det": detected,
                    "det_pct": round(detected / n * 100, 1) if n else 0,
                    "fp": fp, "fp_pct": round(fp / len(test_features) * 100, 1),
                }

            tp = sum(1 for es in error_scores if any(
                es.get(et, 0) >= threshold for et in classifier.models))
            fn = len(error_scores) - tp
            fp_total = sum(1 for cs in clean_scores if any(
                cs.get(et, 0) >= threshold for et in classifier.models))
            tn = len(clean_scores) - fp_total
            combined = Metrics(tp=tp, fp=fp_total, tn=tn, fn=fn)

            width_result[threshold] = {"per_type": type_metrics, "combined": combined}

        all_width_results[width] = {"feat_counts": feat_counts, "thresholds": width_result}

        # Print per-width summary at threshold=0.9
        t09 = width_result[0.9]
        print(f"\n  At threshold=0.9:")
        print(f"  {'Type':<15} {'Feats':>6} {'Det':>5} {'FP':>5}")
        print(f"  {'-'*35}")
        for et in ErrorType:
            n_feats = feat_counts.get(et, 0)
            tm = t09["per_type"][et]
            det_str = f"{tm['det_pct']:.0f}%" if tm['n'] else "—"
            fp_str = f"{tm['fp_pct']:.1f}%"
            print(f"  {et.value:<15} {n_feats:>6} {det_str:>5} {fp_str:>5}")
        cm = t09["combined"]
        print(f"  Combined: F1={cm.f1:.1%}  P={cm.precision:.1%}  R={cm.recall:.1%}")

    # 7. Final comparison table
    for threshold in [0.9, 0.95]:
        print(f"\n{'='*70}")
        print(f"COMPARISON (threshold={threshold})")
        print(f"{'='*70}")

        header = f"  {'Width':<8}"
        for et in ErrorType:
            header += f"  {et.value[:5]:>5}"
        header += f"  {'F1':>5} {'P':>5} {'FP#':>4}"
        print(header)
        print(f"  {'-'*60}")

        for width in WIDTHS:
            wr = all_width_results[width]
            t = wr["thresholds"][threshold]
            row = f"  {width:<8}"
            for et in ErrorType:
                tm = t["per_type"][et]
                if tm["n"]:
                    row += f"  {tm['det_pct']:>4.0f}%"
                else:
                    row += f"     —"
            cm = t["combined"]
            row += f"  {cm.f1:>4.1%} {cm.precision:>4.1%} {cm.fp:>4}"
            print(row)

        # Feature count comparison
        print(f"\n  Feature counts:")
        header2 = f"  {'Width':<8}"
        for et in ErrorType:
            header2 += f"  {et.value[:5]:>5}"
        print(header2)
        print(f"  {'-'*46}")
        for width in WIDTHS:
            row = f"  {width:<8}"
            for et in ErrorType:
                cnt = all_width_results[width]["feat_counts"].get(et, 0)
                row += f"  {cnt:>5}"
            print(row)


if __name__ == "__main__":
    main()
