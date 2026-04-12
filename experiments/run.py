"""Experiment: Baseline error-only feature detection with Gemma 3 1B + GemmaScope 2.

Replicates our earlier Gemma 2 harness results on the newer model/SAE stack.
Reads top-to-bottom: data → model → per-pair analysis → cross-pair summary.
"""

import json
from collections import Counter
from pathlib import Path

from src.cache import cached, TEMP_DIR
from src.data import generate_test_pairs, TextPair
from src.model import tokenize, get_hidden_states, encode_with_sae

# --------------- Experiment parameters ---------------

LAYERS = [5, 10, 13, 17, 22]
WIDTH = "16k"
DATA_VERSION = "v1"           # bump when generate_test_pairs changes
COMPARE_VERSION = "v1"        # bump when compare_pair logic changes
MIN_PAIRS_FOR_CONSISTENCY = 4  # feature must be error-only in this many pairs

RESULTS_DIR = TEMP_DIR / "results"

# Cache key for comparisons encodes the model parameters so changing
# LAYERS or WIDTH auto-invalidates without a manual version bump.
COMPARE_CACHE_KEY = f"{COMPARE_VERSION}_layers={'_'.join(map(str, LAYERS))}_w{WIDTH}"


# --------------- Per-pair comparison ---------------

def compare_pair(pair: TextPair, layers: list[int], width: str) -> dict:
    """Compare SAE activations between clean and error text across layers."""

    def get_features(text, layers, width):
        inputs, str_tokens = tokenize(text)
        hidden_states = get_hidden_states(inputs)
        by_layer = {}
        for layer in layers:
            sae_acts = encode_with_sae(hidden_states[layer + 1], layer, width)
            features = {}
            nonzero = sae_acts.nonzero()
            for pos, feat_idx in nonzero:
                fid = feat_idx.item()
                if fid not in features:
                    features[fid] = {}
                features[fid][pos.item()] = (sae_acts[pos, feat_idx].item(), str_tokens[pos.item()])
            by_layer[layer] = (features, str_tokens)
        return by_layer

    clean_by_layer = get_features(pair.clean, layers, width)
    error_by_layer = get_features(pair.error, layers, width)

    comparison = {}
    for layer in layers:
        clean_feats, clean_tokens = clean_by_layer[layer]
        error_feats, error_tokens = error_by_layer[layer]

        only_in_error = set(error_feats.keys()) - set(clean_feats.keys())
        only_in_clean = set(clean_feats.keys()) - set(error_feats.keys())
        in_both = set(clean_feats.keys()) & set(error_feats.keys())

        comparison[layer] = {
            "clean_tokens": clean_tokens,
            "error_tokens": error_tokens,
            "clean_feature_count": len(clean_feats),
            "error_feature_count": len(error_feats),
            "only_in_error": len(only_in_error),
            "only_in_clean": len(only_in_clean),
            "in_both": len(in_both),
            "only_in_error_ids": sorted(only_in_error),
            "error_only_detail": {
                feat_id: {
                    pos: {"activation": act, "token": tok}
                    for pos, (act, tok) in error_feats[feat_id].items()
                }
                for feat_id in only_in_error
            },
        }

    return comparison


# --------------- Cross-pair analysis ---------------

def cross_pair_analysis(all_results: list[dict], layers: list[int], min_pairs: int):
    """Find features consistently error-only across pairs."""
    summary = {}
    for layer in layers:
        error_only_counter = Counter()
        error_only_positions = {}

        for i, res in enumerate(all_results):
            c = res["comparison"][layer]
            for feat_id in c["only_in_error_ids"]:
                error_only_counter[feat_id] += 1
                if feat_id not in error_only_positions:
                    error_only_positions[feat_id] = []
                for pos, info in c["error_only_detail"].get(feat_id, {}).items():
                    error_only_positions[feat_id].append({
                        "pair": i, "pos": pos,
                        "token": info["token"], "activation": info["activation"],
                    })

        consistent = {f: cnt for f, cnt in error_only_counter.items() if cnt >= min_pairs}
        summary[layer] = {
            "total_consistent": len(consistent),
            "features": sorted(
                [
                    {
                        "feature_id": fid,
                        "pair_count": cnt,
                        "avg_activation": sum(p["activation"] for p in error_only_positions[fid])
                                         / len(error_only_positions[fid]),
                        "sample_tokens": [
                            f"{p['token']}({p['activation']:.1f})"
                            for p in error_only_positions[fid][:5]
                        ],
                    }
                    for fid, cnt in consistent.items()
                ],
                key=lambda x: -x["pair_count"],
            ),
        }

    return summary


# --------------- Main ---------------

def main():
    # 1. Load test data (cached)
    test_pairs = cached("test_pairs", DATA_VERSION, generate_test_pairs)
    print(f"Loaded {len(test_pairs)} test pairs")

    # 2. Run per-pair comparisons (cached — this is the expensive LLM part)
    def compute_all_comparisons():
        results = []
        for i, pair in enumerate(test_pairs):
            print(f"  [{i+1}/{len(test_pairs)}] Processing: {pair.error[:50]}...")
            comparison = compare_pair(pair, LAYERS, WIDTH)
            results.append({
                "pair_index": i,
                "clean": pair.clean,
                "error": pair.error,
                "comparison": comparison,
            })
        return results

    all_results = cached("pair_comparisons", COMPARE_CACHE_KEY, compute_all_comparisons)

    # Print per-pair summary (always, even from cache)
    for res in all_results:
        print(f"\nPair {res['pair_index']+1}: {res['clean'][:50]}...")
        for layer in LAYERS:
            c = res["comparison"][layer]
            print(f"  Layer {layer}: clean={c['clean_feature_count']} feats, "
                  f"error={c['error_feature_count']} feats, "
                  f"error_only={c['only_in_error']}")

    # 3. Cross-pair analysis (always re-runs — this is cheap and what we iterate on)
    print(f"\n{'='*60}")
    print("CROSS-PAIR ANALYSIS")
    print(f"{'='*60}")

    summary = cross_pair_analysis(all_results, LAYERS, MIN_PAIRS_FOR_CONSISTENCY)

    for layer in LAYERS:
        s = summary[layer]
        print(f"\nLayer {layer}: {s['total_consistent']} features error-only "
              f"in {MIN_PAIRS_FOR_CONSISTENCY}+ pairs")
        for feat in s["features"][:10]:
            print(f"  Feature {feat['feature_id']}: {feat['pair_count']}/{len(test_pairs)} pairs, "
                  f"avg_act={feat['avg_activation']:.1f}, "
                  f"tokens: {', '.join(feat['sample_tokens'])}")

    # 4. Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "baseline_experiment.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        if isinstance(obj, set):
            return sorted(obj)
        if isinstance(obj, float):
            return round(obj, 4)
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable({"results": all_results, "summary": summary}), f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
