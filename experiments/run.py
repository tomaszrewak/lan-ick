"""Best known pipeline: position-aware top-50 features, OVR LR.

F1=81.6%, P=83.3%, R=80.0%, FP#=24 (Experiment 13 baseline).
"""

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.data import ErrorType
from src.classifier import predict_tokens_ovr, Metrics
from src.pipeline import build_classifier, FP_BUDGET


# --------------- Main ---------------

def main():
    # 1. Build classifier (shared pipeline)
    classifier, all_pairs, all_features, train_idx, test_idx = build_classifier()
    test_pairs = [all_pairs[i] for i in test_idx]
    test_features = [all_features[i] for i in test_idx]

    print(f"\n  Split: {len(train_idx)} train, {len(test_idx)} test")

    # 2. Score test set
    error_scores: list[dict[ErrorType, float]] = []
    clean_scores: list[dict[ErrorType, float]] = []
    for pf, pair in zip(test_features, test_pairs):
        for text_key, text, scores_list in [
            ("error", pair.error, error_scores),
            ("clean", pair.clean, clean_scores),
        ]:
            tpreds = predict_tokens_ovr(pf[text_key], classifier, text=text)
            scores = {}
            for et in classifier.models:
                scores[et] = max((tp.error_probs.get(et, 0.0) for tp in tpreds), default=0.0)
            scores_list.append(scores)

    # 3. Report per-type results
    type_thresholds = classifier.thresholds

    print(f"\n=== Results (FP ≤ {FP_BUDGET:.0%}) ===")
    print(f"  {'Type':<15} {'Thresh':>6} {'Det':>5} {'FP':>5}")
    print(f"  {'-'*35}")
    n_clean = len(clean_scores)
    for et in ErrorType:
        if et not in type_thresholds:
            print(f"  {et.value:<15}      —     —  0.0%")
            continue
        t = type_thresholds[et]
        type_idx = [i for i, p in enumerate(test_pairs) if p.error_type == et]
        detected = sum(1 for i in type_idx if error_scores[i].get(et, 0) >= t)
        fp = sum(1 for cs in clean_scores if cs.get(et, 0) >= t)
        n = len(type_idx)
        det_pct = detected / n * 100 if n else 0
        fp_pct = fp / n_clean * 100
        print(f"  {et.value:<15} {t:>6.2f} {det_pct:>4.0f}% {fp_pct:>4.1f}%")

    tp = sum(1 for es in error_scores if any(
        es.get(et, 0) >= type_thresholds.get(et, 1.0) for et in classifier.models))
    fn = len(error_scores) - tp
    fp_total = sum(1 for cs in clean_scores if any(
        cs.get(et, 0) >= type_thresholds.get(et, 1.0) for et in classifier.models))
    tn = len(clean_scores) - fp_total
    cm = Metrics(tp=tp, fp=fp_total, tn=tn, fn=fn)
    print(f"\n  Combined: F1={cm.f1:.1%}  P={cm.precision:.1%}  R={cm.recall:.1%}  FP#={cm.fp}")


if __name__ == "__main__":
    main()
