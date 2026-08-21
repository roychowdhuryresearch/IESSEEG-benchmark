#!/usr/bin/env python
"""Figure and stratified statistics for the BASED alignment analysis.

The headline correlation between a model's case probability and expert
BASED is confounded: pre-treatment recordings have both high BASED and
the appearance the models were trained to call "case", while
post-treatment recordings have neither. Reporting the pooled correlation
alone would credit the models with grading severity when what they are
mostly doing is separating treated from untreated recordings.

Every statistic here is therefore reported three ways: pooled, within
pre-treatment, and within post-treatment. The figure is built so a reader
sees that decomposition rather than taking it on trust: each condition
gets a horizontal median reference, so the vertical gap between the two
lines is the effect that exists and their flatness is the effect that
does not.

Encoding follows the project's visualization rules. Treatment condition
is nominal with two levels, so it takes categorical slots 1 and 2 (blue,
orange; validated at protanopia dE 24.7). The BASED score is ordinal
0-5, so it takes a single-hue sequential ramp light-to-dark, not a
rainbow map, which would hide the very ordering it is meant to show.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

MODELS = [("luna", "LUNA"), ("reve", "REVE"), ("eegpt", "EEGPT")]
META = os.environ.get("IESSEEG_SUBJECT_META", "")
RESULTS = "results"

# Categorical slots 1 and 2, for the two treatment conditions.
PRE_HUE, POST_HUE = "#2a78d6", "#eb6834"
# Single-hue sequential ramp, one step per BASED level 0-5.
BASED_RAMP = ["#b7d3f6", "#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.linewidth": 0.8,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax):
    """Hairline, recessive chrome: grid one shade off the surface, no box."""
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)


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
    if not META:
        raise SystemExit("Set IESSEEG_SUBJECT_META to the subject metadata CSV.")
    cond = pd.read_csv(META).drop_duplicates("long_recording_id") \
             .set_index("long_recording_id")["pre_post_treatment_label"]

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8))
    summary, colour_scale = [], None

    for col, (key, label) in enumerate(MODELS):
        scores = pd.read_csv(os.path.join(RESULTS, f"{key}_scores.csv"))
        scores["cond"] = scores.recording_id.map(cond)
        z = np.load(os.path.join(RESULTS, f"{key}_embeddings.npz"), allow_pickle=True)
        emb_by_uid = {str(u): z["emb"][i] for i, u in enumerate(z["segment_uid"])}

        pooled = stats.spearmanr(scores.mean_prob, scores.based)
        strat = {c: stats.spearmanr(g.mean_prob, g.based)
                 for c, g in scores.groupby("cond")}

        # ---------------- A: decision level ----------------
        ax = axes[0, col]
        style(ax)
        rng = np.random.default_rng(7)
        for c, hue in (("PRE", PRE_HUE), ("POST", POST_HUE)):
            g = scores[scores.cond == c]
            jitter = (rng.random(len(g)) - 0.5) * 0.30
            # 2px surface ring keeps overlapping points readable.
            ax.scatter(g.based + jitter, g.mean_prob, s=30, c=hue, alpha=0.75,
                       linewidths=0.9, edgecolors=SURFACE, zorder=3)
            # A horizontal reference at the condition's median. A
            # median-per-BASED-level trace was tried and discarded: with
            # 2 pre-treatment points at BASED 2 and 45 at BASED 5, it
            # zigzagged and implied within-condition structure that the
            # rank correlation says is not there. A flat line states the
            # separation between conditions without inventing a trend.
            ax.axhline(g.mean_prob.median(), color=hue, linewidth=1.6,
                       alpha=0.9, zorder=2)

        ax.set_title(label, fontsize=10.5, color=INK, pad=14, fontweight="bold")
        ax.text(0.5, 1.012,
                f"pooled ρ={pooled.statistic:+.2f}   ·   "
                f"PRE {strat['PRE'].statistic:+.2f}   ·   "
                f"POST {strat['POST'].statistic:+.2f}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.6, color=INK_2)
        ax.set_xlabel("expert BASED score", color=INK_2)
        if col == 0:
            ax.set_ylabel("model case probability", color=INK_2)
        ax.set_xticks(range(6))
        ax.set_xlim(-0.55, 5.55)
        ax.set_ylim(-0.04, 1.04)

        # ---------------- B: representation level ----------------
        ax = axes[1, col]
        style(ax)
        uids = scores.drop_duplicates("segment_uid")
        X = np.stack([emb_by_uid[u] for u in uids.segment_uid])
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
        Xz = Xz - Xz.mean(0)
        U, S, _ = np.linalg.svd(Xz, full_matrices=False)
        pcs = U[:, :2] * S[:2]
        var = (S ** 2 / (S ** 2).sum())[:2] * 100

        vals = scores.groupby("segment_uid").based.mean().loc[uids.segment_uid].values
        cmap = ListedColormap(BASED_RAMP)
        norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
        colour_scale = ax.scatter(pcs[:, 0], pcs[:, 1], c=vals, cmap=cmap, norm=norm,
                                  s=34, alpha=0.92, linewidths=0.9,
                                  edgecolors=SURFACE, zorder=3)

        probe_rho, _ = probe(emb_by_uid, scores)
        ax.text(0.5, 1.012, f"embedding probe ρ={probe_rho:+.2f}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7.6, color=INK_2)
        ax.set_xlabel(f"PC1 ({var[0]:.0f}% var.)", color=INK_2)
        ax.set_ylabel(f"PC2 ({var[1]:.0f}% var.)" if col == 0 else "", color=INK_2)

        summary.append(dict(
            model=label, n=len(scores),
            rho_pooled=pooled.statistic, p_pooled=pooled.pvalue,
            rho_pre=strat["PRE"].statistic, p_pre=strat["PRE"].pvalue,
            rho_post=strat["POST"].statistic, p_post=strat["POST"].pvalue,
            prob_pre=scores[scores.cond == "PRE"].mean_prob.mean(),
            prob_post=scores[scores.cond == "POST"].mean_prob.mean(),
            probe_rho_pooled=probe_rho,
            probe_rho_pre=probe(emb_by_uid, scores[scores.cond == "PRE"])[0],
        ))

    # Identity is never colour-alone: a key for the categorical row, a
    # scale legend for the ordinal one.
    handles = [
        Line2D([], [], marker="o", linestyle="-", linewidth=1.6, color=PRE_HUE,
               markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.9,
               label="pre-treatment  (n=56)"),
        Line2D([], [], marker="o", linestyle="-", linewidth=1.6, color=POST_HUE,
               markersize=6, markeredgecolor=SURFACE, markeredgewidth=0.9,
               label="post-treatment  (n=59)"),
    ]
    leg = axes[0, 0].legend(handles=handles, loc="lower left", fontsize=7.4,
                            frameon=True, framealpha=0.96, edgecolor=GRID,
                            handlelength=2.0, borderpad=0.5,
                            title="line = condition median")
    leg.get_title().set_fontsize(7.0)
    leg.get_title().set_color(MUTED)
    for text in leg.get_texts():
        text.set_color(INK_2)

    fig.tight_layout(rect=[0.02, 0.0, 0.9, 0.935])

    cax = fig.add_axes([0.925, 0.10, 0.013, 0.32])
    cb = fig.colorbar(colour_scale, cax=cax, ticks=range(6))
    cb.set_label("expert BASED score", fontsize=8, color=INK_2)
    cb.ax.tick_params(labelsize=7.5, color=MUTED, labelcolor=MUTED)
    cb.outline.set_edgecolor(GRID)

    fig.text(0.012, 0.925, "A", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 0.455, "B", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.46, 0.972,
             "Case probability tracks treatment condition, not BASED severity",
             ha="center", fontsize=11.5, color=INK, fontweight="bold")

    out = os.path.join(RESULTS, "based_alignment.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"figure -> {out}")

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(RESULTS, "based_alignment_stratified.csv"), index=False)
    print("\n" + df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
