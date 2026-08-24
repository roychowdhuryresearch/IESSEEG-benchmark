#!/usr/bin/env python
"""Is diagnostic confidence prognostic?

BASED severity has been proposed as a treatment-response indicator, and
Section 7 shows the benchmark models' case probability tracks the abnormal
state. This script closes the loop: does a Task-1 (diagnosis) model's
case probability on a patient's pre-treatment Routine Clips predict that
patient's treatment response -- without ever training on response labels?

Leakage control: every clip probability comes from the Task-1 fold in
which that patient was held out, so no probability is produced by a model
that saw the patient.

For each of the nine baselines (and their ensemble): subject-level score =
mean case probability over the subject's Routine Clips; report AUROC of
that score for immediate and sustained response over the 50 cases, with a
two-sided Mann-Whitney p. AUROC < 0.5 means higher apparent severity
predicts NON-response.

Env: IESSEEG_RESULTS_ROOT (kfold results tree), IESSEEG_LABELS
     (final_test.csv), IESSEEG_OUT.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

MODEL_RESULT_SUBDIR = {
    "handcrafted": "handcrafted/result/inference",
    "cnn_resnet": "cnn/result/cnn/inference",
    "cnn_vit": "cnn/result/vit/inference",
    "biot": "biot/result/inference",
    "labram": "labram/result/inference",
    "cbramod": "cbramod/result/inference",
    "luna": "luna/result/inference",
    "eegpt": "eegpt/result/inference",
    "reve": "reve/result/inference",
}


def env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Set {name}")
    return v


def load_task1_probs(results_root, model):
    frames = []
    for fold in range(5):
        d = os.path.join(results_root, MODEL_RESULT_SUBDIR[model],
                         f"case_control_fold{fold}")
        hits = glob.glob(os.path.join(d, "*inference_results.csv"))
        hits = [h for h in hits if not h.endswith("_window.csv")]
        if len(hits) != 1:
            raise SystemExit(f"{model} fold{fold}: expected 1 csv, got {hits}")
        f = pd.read_csv(hits[0])
        # normalize the id column exactly as iesseeg.evaluation.metrics does
        rid = [c for c in f.columns if "recording_id" in c][0]
        f = f.rename(columns={rid: "short_recording_id"})
        f = f[["short_recording_id", "pred_prob", "pred_label", "known_label"]]
        f["fold"] = fold
        frames.append(f)
    df = pd.concat(frames)
    assert df.short_recording_id.is_unique, model
    return df


def main():
    out_dir = env("IESSEEG_OUT")
    os.makedirs(out_dir, exist_ok=True)
    labels = pd.read_csv(env("IESSEEG_LABELS"))
    meta = labels[["short_recording_id", "patient_id", "case_control_label",
                   "immediate_responder", "meaningful_responder"]]

    rows, subj_scores = [], {}
    for model in MODEL_RESULT_SUBDIR:
        probs = load_task1_probs(env("IESSEEG_RESULTS_ROOT"), model)
        m = probs.merge(meta, on="short_recording_id", validate="1:1")
        cases = m[m.case_control_label == "CASE"]
        subj = cases.groupby("patient_id").agg(
            prob=("pred_prob", "mean"),
            immediate=("immediate_responder", "first"),
            meaningful=("meaningful_responder", "first"))
        subj_scores[model] = subj.prob
        row = {"model": model, "n_cases": len(subj)}
        for task in ("immediate", "meaningful"):
            yb = (subj[task] == "Responder").astype(int)
            row[f"auroc_{task}"] = roc_auc_score(yb, subj.prob)
            u = stats.mannwhitneyu(subj.prob[yb == 1], subj.prob[yb == 0])
            row[f"p_{task}"] = u.pvalue
        rows.append(row)

    ens = pd.DataFrame(subj_scores).mean(axis=1)
    # labels are per-patient constants; bind them explicitly rather than
    # reusing the loop-leaked frame from whichever model iterated last
    labels_by_pat = meta.drop_duplicates("patient_id").set_index("patient_id")
    subj = pd.DataFrame({
        "prob": ens,
        "immediate": labels_by_pat.immediate_responder.reindex(ens.index),
        "meaningful": labels_by_pat.meaningful_responder.reindex(ens.index),
    })
    row = {"model": "ensemble_mean", "n_cases": len(subj)}
    for task in ("immediate", "meaningful"):
        yb = (subj[task] == "Responder").astype(int)
        row[f"auroc_{task}"] = roc_auc_score(yb, subj.prob)
        u = stats.mannwhitneyu(subj.prob[yb == 1], subj.prob[yb == 0])
        row[f"p_{task}"] = u.pvalue
    rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "task1_confidence_prognosis.csv"), index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
