#!/usr/bin/env python
"""Summarize the LaBraM learning-rate sensitivity sweep.

For each learning rate and task: per-fold clip-level AUROC (the benchmark
metric), fold mean +/- sd, and -- for the response task -- the challenge
criterion (pooled subject-level AUROC with the within-fold permutation
test from scripts/claim_check.py).

Env: IESSEEG_SWEEP (result/lr_sweep dir), IESSEEG_REF (the published-run
inference dir for the 5e-4 reference), IESSEEG_LABELS, IESSEEG_OUT.
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts"))
from claim_check import permutation_test  # noqa: E402


def env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Set {name}")
    return v


def load_run(csv, meta, label_col):
    f = pd.read_csv(csv)
    rid = [c for c in f.columns if "recording_id" in c][0]
    f = f.rename(columns={rid: "short_recording_id"})
    return f.merge(meta, on="short_recording_id")


def summarize(task, lr, fold_csvs, meta, label_col):
    fold_auc, subj_frames = [], []
    for fold, csv in fold_csvs:
        df = load_run(csv, meta, label_col)
        y = (df[label_col] == "Responder").astype(int) \
            if task != "case_control" else df.known_label
        fold_auc.append(roc_auc_score(y, df.pred_prob))
        s = df.groupby("patient_id").agg(score=("pred_prob", "mean"),
                                         lab=(label_col, "first"))
        s["fold"] = fold
        subj_frames.append(s)
    row = dict(task=task, lr=lr, n_folds=len(fold_auc),
               auroc_mean=np.mean(fold_auc), auroc_sd=np.std(fold_auc, ddof=1),
               folds=" ".join(f"{a:.2f}" for a in fold_auc))
    if task != "case_control" and len(fold_auc) == 5:
        subj = pd.concat(subj_frames).reset_index()
        subj["label"] = (subj.lab == "Responder").astype(int)
        obs, p, _ = permutation_test(subj[["patient_id", "fold",
                                           "score", "label"]])
        row.update(pooled_auroc=obs, claim_p=p)
    return row


def main():
    meta = pd.read_csv(env("IESSEEG_LABELS"))[
        ["short_recording_id", "patient_id", "immediate_responder",
         "meaningful_responder"]]
    rows = []
    for task, label_col in [("meaningful_responder", "meaningful_responder"),
                            ("case_control", "meaningful_responder")]:
        # published reference (lr 5e-4)
        ref = []
        for fold in range(5):
            hits = [h for h in glob.glob(os.path.join(
                env("IESSEEG_REF"), f"{task}_fold{fold}",
                "*inference_results.csv")) if not h.endswith("_window.csv")]
            if hits:
                ref.append((fold, hits[0]))
        if len(ref) == 5:
            rows.append(summarize(task, "5e-4 (published)", ref, meta,
                                  label_col))
        for lr_dir in sorted(glob.glob(os.path.join(env("IESSEEG_SWEEP"),
                                                    "lr*"))):
            lr = os.path.basename(lr_dir)[2:]
            csvs = []
            for fold in range(5):
                c = os.path.join(lr_dir, f"{task}_fold{fold}",
                                 "inference_results.csv")
                if os.path.exists(c):
                    csvs.append((fold, c))
            if csvs:
                rows.append(summarize(task, lr, csvs, meta, label_col))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False,
                       float_format=lambda x: f"{x:.3f}"))
    out = os.environ.get("IESSEEG_OUT")
    if out:
        df.to_csv(os.path.join(out, "lr_sweep_summary.csv"), index=False)


if __name__ == "__main__":
    main()
