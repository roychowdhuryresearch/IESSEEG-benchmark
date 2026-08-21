#!/usr/bin/env python
"""Does the interpretable clinical feature set track BASED severity?

The benchmark's foundation-model embeddings do not linearly encode BASED
(ridge probe R^2 <= 0, analyze_based.py). This script asks the same
questions of the 122-dimensional handcrafted feature set that powers the
GBDT baseline -- the features are clinically named, so any severity signal
found here is directly interpretable.

Every analysis is stratified by treatment condition: the pooled
correlation is confounded by the pre/post separation, and reporting it
alone would credit a feature with grading severity it does not grade.

Inputs (env):
  IESSEEG_BASED_EPOCHS   dir with bipolar22/{segment_uid}.npz  (5-min epochs,
                         0.5-50 Hz filtered, 200 Hz, 22-ch double banana)
  IESSEEG_BASED_SCORES   any of the {model}_scores.csv files from
                         score_based_epochs.py (provides segment_uid ->
                         recording_id, rater, role, based)
  IESSEEG_BENCH_SPLITS   the released splits/ dir (recordings present there
                         are pre-treatment; the rest are post-treatment)
  IESSEEG_OUT            output directory

Outputs (IESSEEG_OUT):
  based_handcrafted_features.npz   X (115 x 122), segment_uid, feature_names
  based_feature_correlations.csv   per-feature Spearman rho pooled/pre/post
                                   with BH-FDR q-values per stratum
  based_probe_comparison.csv       identical grouped ridge probe run on the
                                   feature set AND each cached embedding
"""

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "baselines", "handcrafted"))
import features as feat_mod  # noqa: E402

SFREQ = 200.0
WIN_SEC = 30.0


def env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Set {name}")
    return v


def load_scores():
    df = pd.read_csv(env("IESSEEG_BASED_SCORES"))
    cols = ["segment_uid", "recording_id", "rater", "role", "based"]
    return df.drop_duplicates("segment_uid")[cols].reset_index(drop=True)


def condition_map(recordings):
    """PRE if the recording appears in the released benchmark splits."""
    pre = set()
    for f in glob.glob(os.path.join(env("IESSEEG_BENCH_SPLITS"), "**", "*.csv"),
                       recursive=True):
        try:
            pre |= set(pd.read_csv(f, usecols=["long_recording_id"])
                       ["long_recording_id"].unique())
        except Exception:
            continue
    return {r: ("PRE" if r in pre else "POST") for r in recordings}


def extract_features(scores):
    root = os.path.join(env("IESSEEG_BASED_EPOCHS"), "bipolar22")
    win = int(WIN_SEC * SFREQ)
    rows = []
    for i, uid in enumerate(scores.segment_uid):
        z = np.load(os.path.join(root, f"{uid}.npz"))
        data = z["data"].astype(np.float64)
        n_win = data.shape[1] // win
        feats = [feat_mod.compute_window_features(data[:, k * win:(k + 1) * win], SFREQ)
                 for k in range(n_win)]
        rows.append(np.nanmean(feats, axis=0))
        if (i + 1) % 20 == 0:
            print(f"  features {i + 1}/{len(scores)}")
    return np.asarray(rows)


def bh_fdr(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(ranked)
    q[order] = np.clip(ranked, 0, 1)
    return q


def probe(X, y, groups, tag, out_rows):
    """Grouped ridge probe, identical to analyze_based.py's protocol."""
    X = np.asarray(X, dtype=float)
    ok = ~np.isnan(X).any(axis=1)
    X, y, groups = X[ok], np.asarray(y, float)[ok], np.asarray(groups)[ok]
    splitter = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    pred = np.zeros_like(y)
    for tr, te in splitter.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        rg = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), y[tr])
        pred[te] = rg.predict(sc.transform(X[te]))
    r = stats.spearmanr(pred, y)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    out_rows.append(dict(representation=tag, n=len(y), dim=X.shape[1],
                         probe_rho=r.statistic, probe_p=r.pvalue,
                         probe_r2=1 - ss_res / ss_tot,
                         probe_mae=np.mean(np.abs(y - pred))))
    return pred


def main():
    out_dir = env("IESSEEG_OUT")
    os.makedirs(out_dir, exist_ok=True)

    scores = load_scores()
    cond = condition_map(scores.recording_id.unique())
    scores["cond"] = scores.recording_id.map(cond)
    print("condition counts:", scores.cond.value_counts().to_dict())

    feat_path = os.path.join(out_dir, "based_handcrafted_features.npz")
    if os.path.exists(feat_path):
        z = np.load(feat_path, allow_pickle=True)
        assert list(z["segment_uid"]) == list(scores.segment_uid)
        X = z["X"]
        print("loaded cached features", X.shape)
    else:
        X = extract_features(scores)
        np.savez_compressed(feat_path, X=X, segment_uid=scores.segment_uid.values,
                            feature_names=np.array(feat_mod.FEATURE_NAMES))
        print("saved", feat_path, X.shape)

    names = feat_mod.FEATURE_NAMES
    y = scores.based.values.astype(float)

    # ---- per-feature correlations, stratified -------------------------
    rows = []
    for j, name in enumerate(names):
        row = {"feature": name}
        for tag, mask in [("pooled", np.ones(len(y), bool)),
                          ("pre", (scores.cond == "PRE").values),
                          ("post", (scores.cond == "POST").values)]:
            xj = X[mask, j]
            ok = ~np.isnan(xj)
            r = stats.spearmanr(xj[ok], y[mask][ok])
            row[f"rho_{tag}"], row[f"p_{tag}"] = r.statistic, r.pvalue
        rows.append(row)
    corr = pd.DataFrame(rows)
    for tag in ("pooled", "pre", "post"):
        corr[f"q_{tag}"] = bh_fdr(corr[f"p_{tag}"].values)
    corr = corr.sort_values("p_pre")
    corr.to_csv(os.path.join(out_dir, "based_feature_correlations.csv"), index=False)
    for tag in ("pooled", "pre", "post"):
        n_sig = int((corr[f"q_{tag}"] < 0.05).sum())
        print(f"{tag}: {n_sig}/122 features significant at FDR q<0.05")
    print("\ntop features by within-pre p-value:")
    print(corr.head(8)[["feature", "rho_pre", "p_pre", "q_pre", "rho_post",
                        "rho_pooled"]].to_string(index=False))

    # ---- probes: features vs cached embeddings, identical pipeline ----
    probe_rows = []
    groups = scores.recording_id.values
    reprs = {"handcrafted_122": X}
    emb_dir = os.path.dirname(env("IESSEEG_BASED_SCORES"))
    for model in ("luna", "reve", "eegpt"):
        z = np.load(os.path.join(emb_dir, f"{model}_embeddings.npz"),
                    allow_pickle=True)
        by_uid = {str(u): z["emb"][i] for i, u in enumerate(z["segment_uid"])}
        reprs[f"{model}_embedding"] = np.stack(
            [by_uid[u] for u in scores.segment_uid])

    for tag, Z in reprs.items():
        pred = probe(Z, y, groups, tag, probe_rows)
        for cond_tag in ("PRE", "POST"):
            m = (scores.cond == cond_tag).values
            r = stats.spearmanr(pred[m], y[m])
            probe_rows[-1][f"oof_rho_{cond_tag.lower()}"] = r.statistic
            probe_rows[-1][f"oof_p_{cond_tag.lower()}"] = r.pvalue
        # within-pre-only probe (fit and test entirely inside pre-treatment)
        m = (scores.cond == "PRE").values
        sub = []
        probe(Z[m], y[m], groups[m], tag + "_pre_only", sub)
        probe_rows.append(sub[0])

    # recording-level probe: average epochs within a recording, LORO
    rec = scores.assign(idx=np.arange(len(scores))).groupby("recording_id")
    rec_ids = list(rec.groups)
    Xr = np.stack([np.nanmean(X[g.idx.values], axis=0) for _, g in rec])
    yr = np.array([y[g.idx.values].mean() for _, g in rec])
    sub = []
    probe(Xr, yr, np.array(rec_ids), "handcrafted_122_recording_level", sub)
    probe_rows.append(sub[0])

    pr = pd.DataFrame(probe_rows)
    pr.to_csv(os.path.join(out_dir, "based_probe_comparison.csv"), index=False)
    print("\nprobe comparison:")
    print(pr.to_string(index=False))


if __name__ == "__main__":
    main()
