#!/usr/bin/env python
"""Three follow-up questions on the embedding <-> BASED relationship.

1. Non-linearity: is the ridge probe simply too weak an instrument?
   A distance-weighted k-NN probe (grouped by recording, same CV) answers
   whether ANY smooth function of the representation recovers severity.
2. What do the representations resemble, if not severity? Pairwise linear
   CKA between the four representations, against each representation's
   (per-dimension mean R^2) alignment with BASED.
3. Geometry at the recording level: within pre-treatment, is the distance
   between two recordings' centroids related to how far apart their
   severities are? Mantel-style Spearman with a recording-level
   permutation null.

Env: IESSEEG_BASED_SCORES, IESSEEG_BENCH_SPLITS, IESSEEG_OUT
     (IESSEEG_OUT must hold based_handcrafted_features.npz).
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Set {name}")
    return v


def condition_map(recordings):
    pre = set()
    for f in glob.glob(os.path.join(env("IESSEEG_BENCH_SPLITS"), "**", "*.csv"),
                       recursive=True):
        try:
            pre |= set(pd.read_csv(f, usecols=["long_recording_id"])
                       ["long_recording_id"].unique())
        except Exception:
            continue
    return {r: ("PRE" if r in pre else "POST") for r in recordings}


def load_reprs(scores, out_dir):
    reprs = {}
    z = np.load(os.path.join(out_dir, "based_handcrafted_features.npz"),
                allow_pickle=True)
    assert list(z["segment_uid"]) == list(scores.segment_uid)
    reprs["handcrafted_122"] = np.asarray(z["X"], float)
    emb_dir = os.path.dirname(env("IESSEEG_BASED_SCORES"))
    for model in ("labram", "luna", "reve", "eegpt"):
        e = np.load(os.path.join(emb_dir, f"{model}_embeddings.npz"),
                    allow_pickle=True)
        by = {str(u): e["emb"][i] for i, u in enumerate(e["segment_uid"])}
        reprs[f"{model}_embedding"] = np.stack(
            [by[u] for u in scores.segment_uid]).astype(float)
    return reprs


def zscore(X):
    X = X[:, ~np.isnan(X).any(axis=0)]
    sd = X.std(0)
    return np.clip((X - X.mean(0))[:, sd > 0] / sd[sd > 0], -3, 3)


def knn_probe(Z, y, groups, k=10):
    splitter = GroupKFold(n_splits=5)
    pred = np.zeros_like(y)
    for tr, te in splitter.split(Z, y, groups):
        sc = StandardScaler().fit(Z[tr])
        m = KNeighborsRegressor(n_neighbors=min(k, len(tr)),
                                weights="distance")
        m.fit(sc.transform(Z[tr]), y[tr])
        pred[te] = m.predict(sc.transform(Z[te]))
    return pred


def linear_cka(A, B):
    A = A - A.mean(0)
    B = B - B.mean(0)
    num = np.linalg.norm(B.T @ A, "fro") ** 2
    den = np.linalg.norm(A.T @ A, "fro") * np.linalg.norm(B.T @ B, "fro")
    return float(num / den)


def main():
    out_dir = env("IESSEEG_OUT")
    rng = np.random.default_rng(0)
    scores = pd.read_csv(env("IESSEEG_BASED_SCORES")).drop_duplicates("segment_uid")
    cond = condition_map(scores.recording_id.unique())
    scores["cond"] = scores.recording_id.map(cond)
    y = scores.based.values.astype(float)
    groups = scores.recording_id.values
    pre = (scores.cond == "PRE").values

    reprs = {k: zscore(v) for k, v in load_reprs(scores, out_dir).items()}

    # ---- 1. k-NN probe --------------------------------------------------
    rows = []
    for tag, Z in reprs.items():
        pred = knn_probe(Z, y, groups)
        r_all = stats.spearmanr(pred, y)
        r_pre = stats.spearmanr(pred[pre], y[pre])
        # within-pre-only refit
        pred_p = knn_probe(Z[pre], y[pre], groups[pre])
        r_ponly = stats.spearmanr(pred_p, y[pre])
        rows.append(dict(representation=tag,
                         knn_rho_pooled=r_all.statistic, knn_p_pooled=r_all.pvalue,
                         knn_rho_pre=r_pre.statistic, knn_p_pre=r_pre.pvalue,
                         knn_rho_pre_only=r_ponly.statistic,
                         knn_p_pre_only=r_ponly.pvalue))
    knn = pd.DataFrame(rows)
    knn.to_csv(os.path.join(out_dir, "severity_knn_probe.csv"), index=False)
    print("k-NN probe (grouped CV):")
    print(knn.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    # ---- 2. CKA between representations --------------------------------
    tags = list(reprs)
    cka = pd.DataFrame(index=tags, columns=tags, dtype=float)
    for i, a in enumerate(tags):
        for b in tags[i:]:
            v = linear_cka(reprs[a], reprs[b])
            cka.loc[a, b] = cka.loc[b, a] = v
    cka.to_csv(os.path.join(out_dir, "representation_cka.csv"))
    print("\nlinear CKA between representations:")
    print(cka.round(2).to_string())
    off = [cka.loc[a, b] for i, a in enumerate(tags) for b in tags[i + 1:]]
    print(f"pairwise range {min(off):.2f}-{max(off):.2f}")

    # ---- 3. centroid distance vs severity difference, within pre -------
    rows = []
    for tag, Z in reprs.items():
        recs = np.unique(groups[pre])
        C = np.stack([Z[pre][groups[pre] == r].mean(0) for r in recs])
        yr = np.array([y[pre][groups[pre] == r].mean() for r in recs])
        d_emb = pdist(C)
        d_sev = pdist(yr[:, None])
        r_obs = stats.spearmanr(d_emb, d_sev).statistic
        null = np.empty(10000)
        for i in range(10000):
            perm = rng.permutation(yr)
            null[i] = stats.spearmanr(d_emb, pdist(perm[:, None])).statistic
        p = (np.sum(np.abs(null) >= abs(r_obs)) + 1) / (len(null) + 1)
        rows.append(dict(representation=tag, mantel_rho_pre=r_obs,
                         mantel_p_perm=p, n_recordings=len(recs)))
    man = pd.DataFrame(rows)
    man.to_csv(os.path.join(out_dir, "severity_mantel_pre.csv"), index=False)
    print("\ncentroid distance vs |dBASED| within pre (Mantel, 10k perms):")
    print(man.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))


if __name__ == "__main__":
    main()
