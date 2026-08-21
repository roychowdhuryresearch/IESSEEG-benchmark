#!/usr/bin/env python
"""Figure and stratified statistics for the BASED alignment analysis.

The headline correlation between a model's case probability and expert
BASED is confounded: pre-treatment recordings have both high BASED and
the appearance the models were trained to call "case", while
post-treatment recordings have neither. Reporting the pooled correlation
alone would credit the models with grading severity when what they are
mostly doing is separating treated from untreated recordings.

Every statistic here is therefore reported three ways: pooled, within
pre-treatment, and within post-treatment. The panel layout mirrors that.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

MODELS = [("luna", "LUNA"), ("reve", "REVE"), ("eegpt", "EEGPT")]
META = os.environ.get("IESSEEG_SUBJECT_META", "")
RESULTS = "results"


def condition_map():
    sm = pd.read_csv(META).drop_duplicates("long_recording_id")
    return sm.set_index("long_recording_id")["pre_post_treatment_label"]


def probe(emb_by_uid, rows, n_splits=5):
    """Ridge from embedding to BASED, grouped by recording."""
    X = np.stack([emb_by_uid[u] for u in rows.segment_uid])
    y = rows.based.values.astype(float)
    groups = rows.recording_id.values
    n = min(n_splits, len(np.unique(groups)))
    if n < 2 or len(np.unique(y)) < 2:
        return np.nan, np.nan
    pred = np.zeros_like(y, dtype=float)
    for tr, te in GroupKFold(n_splits=n).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        rg = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(sc.transform(X[tr]), y[tr])
        pred[te] = rg.predict(sc.transform(X[te]))
    r = stats.spearmanr(pred, y)
    return r.statistic, r.pvalue


def main():
    cond = condition_map()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    summary = []

    for col, (key, label) in enumerate(MODELS):
        scores = pd.read_csv(os.path.join(RESULTS, f"{key}_scores.csv"))
        scores["cond"] = scores.recording_id.map(cond)

        z = np.load(os.path.join(RESULTS, f"{key}_embeddings.npz"), allow_pickle=True)
        emb_by_uid = {str(u): z["emb"][i] for i, u in enumerate(z["segment_uid"])}

        # --- top row: decision level, coloured by treatment condition ---
        ax = axes[0, col]
        for c, colour, marker in [("PRE", "#c0392b", "o"), ("POST", "#2471a3", "s")]:
            g = scores[scores.cond == c]
            jitter = (np.random.default_rng(0).random(len(g)) - 0.5) * 0.28
            ax.scatter(g.based + jitter, g.mean_prob, s=26, alpha=0.75,
                       c=colour, marker=marker, edgecolors="none",
                       label=f"{c}-treatment (n={len(g)})")

        pooled = stats.spearmanr(scores.mean_prob, scores.based)
        strat = {}
        for c in ("PRE", "POST"):
            g = scores[scores.cond == c]
            strat[c] = stats.spearmanr(g.mean_prob, g.based)

        ax.set_title(f"{label}\npooled $\\rho$={pooled.statistic:+.2f} "
                     f"(within PRE {strat['PRE'].statistic:+.2f}, "
                     f"POST {strat['POST'].statistic:+.2f})", fontsize=10)
        ax.set_xlabel("expert BASED score"); ax.set_ylabel("model case probability")
        ax.set_xticks(range(6)); ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.25, linewidth=0.5)
        if col == 0:
            ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

        # --- bottom row: representation level, first two PCs ---
        ax = axes[1, col]
        uids = scores.drop_duplicates("segment_uid")
        X = np.stack([emb_by_uid[u] for u in uids.segment_uid])
        Xc = (X - X.mean(0)) / (X.std(0) + 1e-8)
        U, S, _ = np.linalg.svd(Xc - Xc.mean(0), full_matrices=False)
        pcs = U[:, :2] * S[:2]
        based_by_uid = scores.groupby("segment_uid").based.mean()
        colours = based_by_uid.loc[uids.segment_uid].values

        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=colours, cmap="viridis",
                        s=34, alpha=0.85, edgecolors="none", vmin=0, vmax=5)
        pooled_probe = probe(emb_by_uid, scores)
        pre_probe = probe(emb_by_uid, scores[scores.cond == "PRE"])
        ax.set_title(f"embedding PCA\nprobe $\\rho$={pooled_probe[0]:+.2f} pooled", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(alpha=0.25, linewidth=0.5)
        if col == 2:
            cb = fig.colorbar(sc, ax=ax); cb.set_label("expert BASED", fontsize=9)

        summary.append(dict(
            model=label, n=len(scores),
            rho_pooled=pooled.statistic, p_pooled=pooled.pvalue,
            rho_pre=strat["PRE"].statistic, p_pre=strat["PRE"].pvalue,
            rho_post=strat["POST"].statistic, p_post=strat["POST"].pvalue,
            prob_pre=scores[scores.cond == "PRE"].mean_prob.mean(),
            prob_post=scores[scores.cond == "POST"].mean_prob.mean(),
            probe_rho_pooled=pooled_probe[0], probe_p_pooled=pooled_probe[1],
            probe_rho_pre=pre_probe[0], probe_p_pre=pre_probe[1],
        ))

    fig.suptitle("Do benchmark models trained on case/control encode expert BASED severity?",
                 fontsize=12.5, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(RESULTS, "based_alignment.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"figure -> {out}")

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(RESULTS, "based_alignment_stratified.csv"), index=False)
    print("\n" + df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
