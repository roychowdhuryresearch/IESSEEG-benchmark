#!/usr/bin/env python
"""What organizes the representation space: severity, state, or identity?

For each representation (three foundation-model embeddings plus the
122-dim clinical feature set) over the 115 expert-scored BASED epochs,
quantify how much of the geometry is explained by
  (a) recording identity   -- epochs from the same overnight study,
  (b) treatment condition  -- pre vs post,
  (c) BASED severity       -- the construct of interest,
and save 2-D PCA coordinates for the figure.

Metrics
  same_rec_1nn    fraction of epochs whose nearest neighbour (euclidean,
                  z-scored dims) comes from the same recording; chance is
                  computed exactly from the recording sizes.
  eta2_recording  between-recording share of total variance (mean over dims)
  eta2_condition  between-condition share of total variance
  r2_based        variance linearly explained by the BASED score (ridge-free
                  univariate fit per dim, averaged) -- an upper bound on how
                  much of the space is 'about' severity at all.
  sil_recording / sil_condition   silhouette scores.

Inputs (env): IESSEEG_BASED_SCORES, IESSEEG_BENCH_SPLITS, IESSEEG_OUT
              (IESSEEG_OUT must already hold based_handcrafted_features.npz)
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


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


def eta2(Z, labels):
    """Between-group share of total variance, averaged over dimensions."""
    grand = Z.mean(axis=0)
    ss_tot = ((Z - grand) ** 2).sum()
    ss_between = 0.0
    for g in np.unique(labels):
        m = labels == g
        ss_between += m.sum() * ((Z[m].mean(axis=0) - grand) ** 2).sum()
    return float(ss_between / ss_tot)


def r2_based(Z, y):
    """Mean per-dimension R^2 of a univariate linear fit on BASED."""
    yc = y - y.mean()
    denom = (yc ** 2).sum()
    beta = (Z - Z.mean(0)).T @ yc / denom
    resid = Z - Z.mean(0) - np.outer(yc, beta)
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((Z - Z.mean(0)) ** 2).sum(axis=0)
    ok = ss_tot > 0
    return float(np.mean(1 - ss_res[ok] / ss_tot[ok]))


def same_rec_1nn(Z, rec):
    D = cdist(Z, Z)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)
    frac = float(np.mean(rec[nn] == rec))
    n = len(rec)
    counts = pd.Series(rec).value_counts().values
    chance = float(np.sum(counts * (counts - 1)) / (n * (n - 1)))
    return frac, chance


def main():
    out_dir = env("IESSEEG_OUT")
    scores = pd.read_csv(env("IESSEEG_BASED_SCORES")).drop_duplicates("segment_uid")
    cond = condition_map(scores.recording_id.unique())
    scores["cond"] = scores.recording_id.map(cond)
    y = scores.based.values.astype(float)
    rec = scores.recording_id.values
    cnd = scores.cond.values

    reprs = {}
    z = np.load(os.path.join(out_dir, "based_handcrafted_features.npz"),
                allow_pickle=True)
    assert list(z["segment_uid"]) == list(scores.segment_uid)
    reprs["handcrafted_122"] = z["X"]
    emb_dir = os.path.dirname(env("IESSEEG_BASED_SCORES"))
    for model in ("labram", "luna", "reve", "eegpt"):
        e = np.load(os.path.join(emb_dir, f"{model}_embeddings.npz"),
                    allow_pickle=True)
        by_uid = {str(u): e["emb"][i] for i, u in enumerate(e["segment_uid"])}
        reprs[f"{model}_embedding"] = np.stack(
            [by_uid[u] for u in scores.segment_uid])

    rows, coords = [], {}
    for tag, X in reprs.items():
        X = np.asarray(X, float)
        col_ok = ~np.isnan(X).any(axis=0)
        X = X[:, col_ok]
        sd = X.std(axis=0)
        Z = (X - X.mean(0))[:, sd > 0] / sd[sd > 0]
        # winsorize at 3 SD, applied identically to every representation:
        # a handful of heavy-tailed feature epochs would otherwise dominate
        # both the variance decomposition and the PCA display
        Z = np.clip(Z, -3, 3)
        frac, chance = same_rec_1nn(Z, rec)
        rows.append(dict(
            representation=tag, dim=Z.shape[1],
            same_rec_1nn=frac, same_rec_1nn_chance=chance,
            eta2_recording=eta2(Z, rec), eta2_condition=eta2(Z, cnd),
            r2_based=r2_based(Z, y),
            sil_recording=silhouette_score(Z, rec),
            sil_condition=silhouette_score(Z, cnd),
        ))
        p = PCA(n_components=2).fit(Z)
        coords[f"pca_{tag}"] = p.transform(Z)
        rows[-1]["pca2_var"] = float(p.explained_variance_ratio_.sum())
        # UMAP layout for the figure: PCA keeps 39-60% of the variance here,
        # which renders as structureless blobs; the neighbour-graph layout
        # shows the cluster structure the quantitative metrics measure.
        # Metrics above are computed in the full space, never on the layout.
        try:
            from umap import UMAP
            coords[f"umap_{tag}"] = UMAP(
                n_neighbors=10, min_dist=0.4, random_state=0,
                init="pca").fit_transform(Z)
        except ImportError:
            from sklearn.manifold import TSNE
            coords[f"umap_{tag}"] = TSNE(
                n_components=2, perplexity=12, init="pca",
                random_state=0).fit_transform(Z)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "geometry_metrics.csv"), index=False)
    print(df.to_string(index=False))

    np.savez_compressed(
        os.path.join(out_dir, "geometry_pca_coords.npz"),
        segment_uid=scores.segment_uid.values, recording_id=rec,
        cond=cnd, based=y, **coords)
    print("saved 2-D layouts (PCA + UMAP)")


if __name__ == "__main__":
    main()
