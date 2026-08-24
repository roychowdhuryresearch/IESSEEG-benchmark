#!/usr/bin/env python
"""Benchmark results figure (authored at NeurIPS print width, 5.5 in): nine baselines across the three tasks.

The paper's tables carry the numbers; this carries the shape of the
result, which is the actual claim: diagnosis is largely solved, both
treatment-response tasks sit on chance, and the ordering of model
families does not follow pre-training scale.

Form: a dot-and-interval plot, which is the honest reading of "a mean
with a fold-to-fold spread". Bars would imply a magnitude measured from
zero, but balanced accuracy is measured from 0.5 -- chance -- so chance
is drawn as the reference and the interval carries the uncertainty that
makes most of these differences unresolvable.

Encoding: model family is the identity dimension the paper argues about
(clinical features / trained from scratch / pre-trained), so it takes
three categorical slots, validated all-pairs at deuteranopia dE 9.2. The
aqua slot warns on surface contrast, which the direct model labels on
every row relieve. Model order is fixed across panels so a reader
compares rows, never re-reads a legend.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

RESULTS = os.environ.get("IESSEEG_BENCH_RESULTS", "")

# Fixed row order, grouped by family; never re-sorted per panel.
MODELS = [
    ("handcrafted", "GBDT + Clinical Prior", "features"),
    ("cnn_resnet", "3D ResNet-18", "scratch"),
    ("cnn_vit", "3D ViT", "scratch"),
    ("biot", "BIOT", "pretrained"),
    ("labram", "LaBraM", "pretrained"),
    ("cbramod", "CBraMod", "pretrained"),
    ("luna", "LUNA", "pretrained"),
    ("eegpt", "EEGPT", "pretrained"),
    ("reve", "REVE", "pretrained"),
]
FAMILY_HUE = {"features": "#2a78d6", "scratch": "#eb6834", "pretrained": "#1baf7a"}
FAMILY_LABEL = {"features": "clinical features",
                "scratch": "trained from scratch",
                "pretrained": "pre-trained foundation model"}

TASKS = [("case_control", "Task 1\nDiagnosis"),
         ("immediate_responder", "Task 2\nImmediate response"),
         ("meaningful_responder", "Task 3\nSustained response")]

SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.linewidth": 0.5,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 5.8, "ytick.labelsize": 6.2,
    "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 1.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def main():
    if not RESULTS:
        raise SystemExit("Set IESSEEG_BENCH_RESULTS to the results_summary.csv directory.")
    s = pd.read_csv(os.path.join(RESULTS, "results_summary.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0), sharey=True)
    ys = np.arange(len(MODELS))[::-1]

    for ax, (task, title) in zip(axes, TASKS):
        ax.grid(True, axis="x", color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)

        # Chance is the baseline that matters for balanced accuracy, so it
        # is drawn once as recessive chrome rather than left implicit.
        ax.axvline(0.5, color=AXIS, linewidth=0.7, zorder=1)

        for y, (key, label, family) in zip(ys, MODELS):
            row = s[(s.model == key) & (s.task == task)]
            if row.empty:
                continue
            m = float(row.balanced_accuracy_mean.iloc[0])
            sd = float(row.balanced_accuracy_std.iloc[0])
            hue = FAMILY_HUE[family]
            ax.plot([m - sd, m + sd], [y, y], color=hue, linewidth=1.1,
                    alpha=0.45, solid_capstyle="round", zorder=2)
            ax.plot([m], [y], marker="o", markersize=3.4, color=hue,
                    markeredgecolor=SURFACE, markeredgewidth=0.6, zorder=3)

        ax.set_title(title, fontsize=6.6, color=INK, pad=3, fontweight="bold")
        ax.set_xlim(0.28, 1.02)
        ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xticklabels([".4", ".5", ".6", ".7", ".8", ".9", "1"])
        ax.set_xlabel("balanced accuracy", color=INK_2, fontsize=6.2, labelpad=1.5)
        ax.set_ylim(-0.7, len(MODELS) - 0.3)
        # sharey leaves tick marks on the inner panels even with the left
        # spine hidden; they read as stray dashes between panels.
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for _, label, _ in MODELS], color=INK_2)
    axes[0].tick_params(axis="y", length=0)

    # "chance" is annotated once, on the panel where it is the whole story.
    axes[1].text(0.5, -0.66, "chance", fontsize=5.0, color=MUTED,
                 ha="center", va="bottom",
                 bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))

    handles = [Line2D([], [], marker="o", linestyle="-", linewidth=1.1,
                      color=FAMILY_HUE[f], markersize=3.4,
                      markeredgecolor=SURFACE, markeredgewidth=0.6,
                      label=FAMILY_LABEL[f])
               for f in ("features", "scratch", "pretrained")]
    leg = fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=5.8,
                     frameon=False, bbox_to_anchor=(0.55, -0.06),
                     handlelength=1.6, columnspacing=1.6)
    for text in leg.get_texts():
        text.set_color(INK_2)

    fig.tight_layout(rect=[0, 0.03, 1, 1.0])

    out_dir = os.environ.get("IESSEEG_OUT", "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fig_benchmark_overview.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
