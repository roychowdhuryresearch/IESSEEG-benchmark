#!/usr/bin/env python
"""
inference.py

Reads:
  - A trained XGBoost model (.joblib)
  - A CSV with columns [short_recording_id, patient_id, <label_key>]
  - epoch_length, sfreq, etc. (must match training)

For each row, we:
  1. Load the .npz
  2. Epoch it into non-overlapping windows (like in training)
  3. Extract features PER WINDOW (feature_set must match training)
  4. Use the loaded XGBoost model to predict a probability PER WINDOW, then
     aggregate window probabilities into one recording-level probability via
     --agg (mean/max/median/topk_mean), threshold at 0.5.
  5. Write out an output CSV with [short_recording_id, known_label, pred_prob, pred_label, n_windows].

Example usage:
  python inference.py \
    --model_file "brain_xgb.joblib" \
    --inference_csv "case_control_test.csv" \
    --data_root "../data/scalp_eeg_200hz_npz" \
    --out_csv "inference_results.csv" \
    --epoch_length 30 \
    --sfreq 200 \
    --feature_set regional \
    --agg mean
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from joblib import load
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils.data_utils import create_label_from_meta_csv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as feat_mod
import legacy_features


def extract_window_features_for_npz(npz_path, sfreq, epoch_length, feature_set="regional", cache_dir=None,
                                     include_asymmetry=True, include_pli=True):
    if feature_set in ("regional", "global_avg"):
        mode = "regional" if feature_set == "regional" else "global_avg"
        return feat_mod.extract_window_features_for_npz(npz_path, sfreq, epoch_length, cache_dir=cache_dir, mode=mode,
                                                          include_asymmetry=include_asymmetry, include_pli=include_pli)

    if not os.path.isfile(npz_path):
        print(f"File not found: {npz_path}")
        return None
    loaded = np.load(npz_path)
    data_array = loaded["data"]
    ep_list = legacy_features.epoch_data(data_array, sfreq, epoch_length)
    if len(ep_list) == 0:
        return None
    avg_channels = feature_set == "avg"
    all_features = [legacy_features.compute_epoch_features(ep, sfreq, avg_channels=avg_channels) for ep in ep_list]
    return np.array(all_features)


def aggregate_window_probas(window_probas, strategy="mean", topk_frac=0.2):
    """
    Pools per-window probabilities into a single recording-level score.
    Window-level abnormalities in IESS-related EEG (hypsarrhythmia bursts,
    spasms, discharges) are often paroxysmal rather than present throughout
    a recording, so plain mean-pooling can dilute a localized signal toward
    the "normal-looking" majority of windows -- this is really a Multiple
    Instance Learning setting (window=instance, recording=bag). max/topk_mean
    are MIL-style alternatives that don't require the whole bag to agree.
    """
    window_probas = np.asarray(window_probas)
    if strategy == "mean":
        return float(window_probas.mean())
    if strategy == "max":
        return float(window_probas.max())
    if strategy == "median":
        return float(np.median(window_probas))
    if strategy == "topk_mean":
        k = max(1, int(np.ceil(len(window_probas) * topk_frac)))
        top_vals = np.sort(window_probas)[-k:]
        return float(top_vals.mean())
    raise ValueError(f"Unknown aggregation strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--inference_csv", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="inference_results.csv")
    parser.add_argument("--epoch_length", type=float, default=30.0)
    parser.add_argument("--sfreq", type=float, default=200.0)
    parser.add_argument("--label_key", type=str, default="case_control_label")
    parser.add_argument("--feature_set", type=str, default="regional",
                         choices=["regional", "global_avg", "flat", "avg"],
                         help="Must match the feature_set used for training this model.")
    parser.add_argument("--agg", type=str, default="mean",
                         choices=["mean", "max", "median", "topk_mean"],
                         help="Window-probability aggregation strategy for the recording-level score.")
    parser.add_argument("--topk_frac", type=float, default=0.2,
                         help="Fraction of windows used by --agg topk_mean.")
    parser.add_argument("--feature_cache_dir", type=str, default=None,
                         help="If set, per-recording window features are memoized here "
                              "(reused across repeated --agg runs against the same test set, "
                              "and shared with training's cache if the same dir is passed).")
    parser.add_argument("--exclude_asymmetry", action="store_true",
                         help="(feature_set=regional only) must match training.")
    parser.add_argument("--exclude_pli", action="store_true",
                         help="(feature_set=regional only) must match training.")

    args = parser.parse_args()

    print(f"Loading model from {args.model_file}...")
    model = load(args.model_file)

    print(f"Reading inference CSV from {args.inference_csv}...")
    df_infer = pd.read_csv(args.inference_csv)

    labels = create_label_from_meta_csv(df_infer, args.label_key)
    df_infer["binary_label"] = labels

    results = []
    for row in tqdm(df_infer.itertuples(), total=len(df_infer), desc="Inference"):
        short_id = row.short_recording_id
        known_label = row.binary_label

        window_feats = extract_window_features_for_npz(
            f"{args.data_root}/{short_id}.npz", args.sfreq, args.epoch_length,
            feature_set=args.feature_set, cache_dir=args.feature_cache_dir,
            include_asymmetry=not args.exclude_asymmetry, include_pli=not args.exclude_pli,
        )
        if window_feats is None:
            print(f"Skipping file_id={short_id} due to short/no data: {args.data_root}/{short_id}.npz")
            continue

        window_probas = model.predict_proba(window_feats)[:, 1]
        proba = aggregate_window_probas(window_probas, strategy=args.agg, topk_frac=args.topk_frac)
        pred_label = int(proba >= 0.5)

        results.append({
            "short_recording_id": short_id,
            "known_label": known_label,
            "pred_prob": proba,
            "pred_label": pred_label,
            "n_windows": len(window_probas),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(args.out_csv, index=False)
    print(f"\nDone. Saved inference results to {args.out_csv}. Sample:\n{out_df.head(10)}")


if __name__ == "__main__":
    main()
