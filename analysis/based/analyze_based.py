#!/usr/bin/env python
"""Do the benchmark models encode expert BASED severity?

Two questions, deliberately separated:

  Decision level -- does the case probability the benchmark actually
  reports rise with expert BASED? This is what a user of the benchmark
  would see.

  Representation level -- can BASED be linearly decoded from the model's
  penultimate embedding? A model can encode severity while its
  case/control head discards it, and only the probe would reveal that.

The probe is cross-validated with grouping by recording, so it is never
fit and tested on epochs from the same recording; without that, the seven
raters' epochs from one recording would leak across the split and inflate
the score badly.

Baseline for comparison: the models were supervised only on case/control,
and every BASED recording is a confirmed case, so any severity signal is
incidental rather than trained for.
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

MODELS = ["luna", "reve", "eegpt"]


def decision_level(results_dir, model):
    df = pd.read_csv(os.path.join(results_dir, f"{model}_scores.csv"))
    out = {"model": model, "n_epochs": len(df)}

    rho = stats.spearmanr(df.mean_prob, df.based)
    out["rho_all"] = rho.statistic
    out["p_all"] = rho.pvalue

    for role in ("Expert", "Trainee"):
        g = df[df.role == role]
        if len(g) > 3:
            r = stats.spearmanr(g.mean_prob, g.based)
            out[f"rho_{role.lower()}"] = r.statistic
            out[f"p_{role.lower()}"] = r.pvalue

    # Recording-level: average raters within a recording, which removes
    # the epoch-to-epoch variation and asks whether the model orders
    # recordings the way the panel does.
    rec = df.groupby("recording_id").agg(based=("based", "mean"), prob=("mean_prob", "mean"))
    r = stats.spearmanr(rec.prob, rec.based)
    out["rho_recording"] = r.statistic
    out["p_recording"] = r.pvalue
    out["n_recordings"] = len(rec)
    return out, df


def embedding_probe(results_dir, model, scores, n_splits=5, seed=0):
    """Ridge from embedding to BASED, grouped by recording."""
    z = np.load(os.path.join(results_dir, f"{model}_embeddings.npz"), allow_pickle=True)
    emb = z["emb"]
    uid = [str(u) for u in z["segment_uid"]]
    emb_by_uid = {u: emb[i] for i, u in enumerate(uid)}

    rows = scores.dropna(subset=["based"])
    X = np.stack([emb_by_uid[u] for u in rows.segment_uid])
    y = rows.based.values.astype(float)
    groups = rows.recording_id.values

    splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    pred = np.zeros_like(y, dtype=float)
    for train, test in splitter.split(X, y, groups):
        scaler = StandardScaler().fit(X[train])
        ridge = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(scaler.transform(X[train]), y[train])
        pred[test] = ridge.predict(scaler.transform(X[test]))

    rho = stats.spearmanr(pred, y)
    mae = np.abs(pred - y).mean()
    # R^2 against predicting the training mean, i.e. does the embedding
    # beat knowing nothing about the epoch.
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return dict(model=model, n=len(y), dim=X.shape[1], rho=rho.statistic,
                p=rho.pvalue, mae=mae, r2=1 - ss_res / ss_tot), pred, y, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--out_csv", default="results/based_alignment.csv")
    args = parser.parse_args()

    print("=" * 72)
    print("DECISION LEVEL: does the reported case probability track BASED?")
    print("=" * 72)
    decision_rows, score_frames = [], {}
    for m in args.models:
        row, df = decision_level(args.results_dir, m)
        decision_rows.append(row)
        score_frames[m] = df
        print(f"\n{m}  (n={row['n_epochs']} epochs, {row['n_recordings']} recordings)")
        print(f"  epoch level    rho={row['rho_all']:+.3f}  p={row['p_all']:.2g}")
        if "rho_expert" in row:
            print(f"    experts      rho={row['rho_expert']:+.3f}  p={row['p_expert']:.2g}")
            print(f"    trainees     rho={row['rho_trainee']:+.3f}  p={row['p_trainee']:.2g}")
        print(f"  recording level rho={row['rho_recording']:+.3f}  p={row['p_recording']:.2g}")

    print("\n" + "=" * 72)
    print("REPRESENTATION LEVEL: is BASED linearly decodable from the embedding?")
    print("(ridge, grouped by recording so no recording spans train and test)")
    print("=" * 72)
    probe_rows = []
    for m in args.models:
        row, pred, y, groups = embedding_probe(args.results_dir, m, score_frames[m])
        probe_rows.append(row)
        print(f"\n{m}  (n={row['n']}, embedding dim {row['dim']})")
        print(f"  probe rho={row['rho']:+.3f}  p={row['p']:.2g}  MAE={row['mae']:.2f} BASED points  R2={row['r2']:+.3f}")

    out = pd.DataFrame(decision_rows).merge(
        pd.DataFrame(probe_rows).rename(columns=lambda c: c if c == "model" else f"probe_{c}"),
        on="model")
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
