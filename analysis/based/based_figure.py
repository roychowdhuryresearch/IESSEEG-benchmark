#!/usr/bin/env python
"""Paper figure: case probability tracks treatment condition, not BASED.

Redesign of the alignment figure. Individual epochs are de-emphasized in
gray; the colored layer is the recording-level story (per-recording means,
colored by treatment condition, plus condition medians), because the
finding lives at that level: conditions separate, severity within a
condition does not. Row B shows the ridge probe's out-of-fold predictions
with the same emphasis structure.

Inputs (env):
  IESSEEG_BASED_RESULTS  dir with {model}_scores.csv + {model}_embeddings.npz
  IESSEEG_BENCH_SPLITS   released splits/ (condition inference)
  IESSEEG_OUT            output directory for the figure

The probe is recomputed here (same protocol as analyze_based.py: RidgeCV,
GroupKFold by recording) so the panel and the reported numbers can never
drift apart.
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

MODELS = [("luna", "LUNA"), ("reve", "REVE"), ("eegpt", "EEGPT")]
PRE_HUE, POST_HUE = "#2a78d6", "#eb6834"
EPOCH_GRAY = "#b9b6ae"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


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


def probe_oof(results, model, scores):
    z = np.load(os.path.join(results, f"{model}_embeddings.npz"),
                allow_pickle=True)
    by = {str(u): z["emb"][i] for i, u in enumerate(z["segment_uid"])}
    X = np.stack([by[u] for u in scores.segment_uid]).astype(float)
    y = scores.based.values.astype(float)
    groups = scores.recording_id.values
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        rg = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(
            sc.transform(X[tr]), y[tr])
        pred[te] = rg.predict(sc.transform(X[te]))
    return pred


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def strip(ax, text):
    ax.text(0.5, 1.02, text, transform=ax.transAxes, ha="center",
            fontsize=7.0, color=MUTED)


def fmt(v):
    return f"{v:+.2f}".replace("+0.", "+.").replace("-0.", "−.")


def main():
    results, out_dir = env("IESSEEG_BASED_RESULTS"), env("IESSEEG_OUT")
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 5.1),
                             sharex=True,
                             gridspec_kw=dict(hspace=0.36, wspace=0.16,
                                              left=0.065, right=0.985,
                                              top=0.90, bottom=0.15))
    fig.text(0.012, 0.945, "A", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 0.47, "B", fontsize=12, fontweight="bold", color=INK)

    for col, (key, label) in enumerate(MODELS):
        scores = pd.read_csv(os.path.join(results, f"{key}_scores.csv")) \
                   .drop_duplicates("segment_uid").reset_index(drop=True)
        cond = condition_map(scores.recording_id.unique())
        scores["cond"] = scores.recording_id.map(cond)
        y, prob = scores.based.values.astype(float), scores.mean_prob.values
        jit = y + rng.uniform(-0.13, 0.13, len(y))
        rec = scores.groupby("recording_id").agg(
            based=("based", "mean"), prob=("mean_prob", "mean"),
            cond=("cond", "first"))

        # ---- row A: probability vs BASED --------------------------------
        ax = axes[0, col]
        for tag, hue in (("PRE", PRE_HUE), ("POST", POST_HUE)):
            med = np.median(prob[scores.cond == tag])
            ax.axhline(med, color=hue, lw=1.0, alpha=0.45, zorder=1)
        ax.scatter(jit, prob, s=8, c=EPOCH_GRAY, alpha=0.55, lw=0, zorder=2)
        for tag, hue in (("PRE", PRE_HUE), ("POST", POST_HUE)):
            m = rec.cond == tag
            ax.scatter(rec.based[m], rec.prob[m], s=46, c=hue,
                       edgecolors="white", linewidths=0.9, zorder=3)
        pooled = stats.spearmanr(prob, y)
        r_pre = stats.spearmanr(prob[scores.cond == "PRE"],
                                y[scores.cond == "PRE"])
        r_post = stats.spearmanr(prob[scores.cond == "POST"],
                                 y[scores.cond == "POST"])
        strip(ax, f"pooled ρ {fmt(pooled.statistic)}   ·   "
                  f"pre {fmt(r_pre.statistic)}   ·   "
                  f"post {fmt(r_post.statistic)}")
        ax.set_title(label, fontsize=9.5, fontweight="bold", color=INK,
                     pad=16)
        ax.set_ylim(-0.04, 1.06)
        if col == 0:
            ax.set_ylabel("case probability", fontsize=8.2)
        else:
            ax.set_yticklabels([])
        style(ax)

        # ---- row B: probe predictions vs truth --------------------------
        ax = axes[1, col]
        pred = probe_oof(results, key, scores)
        r = stats.spearmanr(pred, y)
        r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        ax.plot([-0.4, 5.4], [-0.4, 5.4], color=AXIS, lw=0.9, ls=(0, (4, 3)),
                zorder=1)
        ax.scatter(jit, pred, s=8, c=EPOCH_GRAY, alpha=0.55, lw=0, zorder=2)
        rec_pred = pd.Series(pred).groupby(scores.recording_id).mean()
        for tag, hue in (("PRE", PRE_HUE), ("POST", POST_HUE)):
            m = rec.cond == tag
            ax.scatter(rec.based[m], rec_pred[rec.index][m.values], s=46,
                       c=hue, edgecolors="white", linewidths=0.9, zorder=3)
        strip(ax, f"probe ρ {fmt(r.statistic)}   ·   "
                  f"R² {fmt(r2)}")
        ax.set_xlim(-0.45, 5.45)
        ax.set_xticks(range(6))
        ax.set_xlabel("expert BASED score", fontsize=8.2)
        if col == 0:
            ax.set_ylabel("probe-predicted BASED", fontsize=8.2)
        style(ax)

    fig.legend(handles=[
        Line2D([], [], marker="o", ls="", color=PRE_HUE, markersize=7,
               markeredgecolor="white", label="recording mean — pre-treatment"),
        Line2D([], [], marker="o", ls="", color=POST_HUE, markersize=7,
               markeredgecolor="white", label="recording mean — post-treatment"),
        Line2D([], [], marker="o", ls="", color=EPOCH_GRAY, markersize=4.5,
               label="single rated epoch"),
        Line2D([], [], color=MUTED, lw=1.0, alpha=0.6,
               label="condition median (row A)")],
        loc="lower center", ncol=4, frameon=False, fontsize=7.4,
        handletextpad=0.3, columnspacing=1.6, bbox_to_anchor=(0.52, -0.005))

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_based_alignment.{ext}"),
                    bbox_inches="tight", dpi=220)
    print("wrote fig_based_alignment")


if __name__ == "__main__":
    main()
