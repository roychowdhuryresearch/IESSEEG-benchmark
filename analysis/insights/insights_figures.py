#!/usr/bin/env python
"""Figures for the dataset-insight analyses.

fig_representation_geometry  UMAP layouts of each representation over the
                             115 BASED epochs with per-recording hulls,
                             aligned above a matrix of the variance shares
                             of recording / condition / severity.
fig_error_concentration      Unit-dot histogram of the 200 test clips by
                             number of baselines wrong, coloured by what
                             the experts saw in the clip.
fig_confidence_prognosis     Dumbbell plot: held-out diagnostic confidence
                             as a response predictor, immediate vs
                             sustained, all nine baselines below chance.

Reads the CSV/NPZ outputs of the analysis scripts from IESSEEG_OUT and
writes PDF (paper) + PNG (inspection) into the same directory.

Style follows the paper's figure system (based_figure.py /
benchmark_figure.py): declarative title, panel letters, per-panel stat
strips, white-ringed markers, hairline chrome, identical hues.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull

PRE_HUE, POST_HUE = "#2a78d6", "#eb6834"
FEAT_HUE, ACCENT = "#2a78d6", "#eb6834"
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

REPS = [("handcrafted_122", "Clinical features"),
        ("luna_embedding", "LUNA"),
        ("reve_embedding", "REVE"),
        ("eegpt_embedding", "EEGPT")]


def env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Set {name}")
    return v


def style(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------
def fig_geometry(out_dir):
    z = np.load(os.path.join(out_dir, "geometry_pca_coords.npz"),
                allow_pickle=True)
    met = pd.read_csv(os.path.join(out_dir, "geometry_metrics.csv")) \
            .set_index("representation")
    rec, cond = z["recording_id"], z["cond"]

    fig = plt.figure(figsize=(10.6, 4.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.35, 1.0],
                          hspace=0.34, wspace=0.14,
                          left=0.075, right=0.985, top=0.83, bottom=0.06)
    fig.suptitle("Recording identity organizes every representation; "
                 "severity organizes none",
                 fontsize=11, fontweight="bold", color=INK, y=0.975)
    fig.text(0.012, 0.90, "A", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 0.245, "B", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.075, 0.315, "share of variance explained by",
             fontsize=7.6, color=MUTED)

    for i, (key, label) in enumerate(REPS):
        ax = fig.add_subplot(gs[0, i])
        P = z[f"umap_{key}"]
        # per-recording hulls, one per connected component: a recording
        # whose epochs land in two distant clumps gets two hulls rather
        # than one sliver spanning empty space
        span = np.linalg.norm(P.max(0) - P.min(0))
        for r in np.unique(rec):
            pts_r = P[rec == r]
            labels = fcluster(linkage(pts_r, "single"),
                              t=0.18 * span, criterion="distance")
            for g in np.unique(labels):
                pts = pts_r[labels == g]
                if len(pts) >= 3:
                    hull = ConvexHull(pts)
                    hp = pts[hull.vertices]
                    ax.fill(hp[:, 0], hp[:, 1], color=INK_2, alpha=0.05,
                            lw=0, zorder=1)
                    ax.plot(np.r_[hp[:, 0], hp[0, 0]],
                            np.r_[hp[:, 1], hp[0, 1]],
                            color=AXIS, lw=0.7, zorder=1)
                elif len(pts) == 2:
                    ax.plot(pts[:, 0], pts[:, 1], color=AXIS, lw=0.7,
                            zorder=1)
        for tag, hue in (("PRE", PRE_HUE), ("POST", POST_HUE)):
            m = cond == tag
            ax.scatter(P[m, 0], P[m, 1], s=21, c=hue, alpha=0.95,
                       edgecolors="white", linewidths=0.6, zorder=3)
        ax.set_title(label, fontsize=9.5, fontweight="bold", color=INK,
                     pad=16)
        knn = met.loc[key, "same_rec_1nn"]
        ax.text(0.5, 1.015, f"same-recording 1-NN {knn:.0%}  ·  chance 5%",
                transform=ax.transAxes, ha="center", fontsize=7.4,
                color=INK_2)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(GRID)
        m0, m1 = P.min(0), P.max(0)
        pad = 0.07 * (m1 - m0)
        ax.set_xlim(m0[0] - pad[0], m1[0] + pad[0])
        ax.set_ylim(m0[1] - pad[1], m1[1] + pad[1] * 1.6)
        if i == 0:
            ax.legend(handles=[
                Line2D([], [], marker="o", ls="", color=PRE_HUE,
                       markeredgecolor="white", label="pre-treatment"),
                Line2D([], [], marker="o", ls="", color=POST_HUE,
                       markeredgecolor="white", label="post-treatment"),
                Line2D([], [], marker="s", ls="", color=INK_2, alpha=0.25,
                       label="one recording")],
                loc="upper left", frameon=True, framealpha=0.85,
                edgecolor=GRID, fontsize=7.0, handletextpad=0.2,
                borderaxespad=0.25, labelspacing=0.25)

    # aligned variance-share matrix, one mini-axes per representation
    measures = [("eta2_recording", "recording identity"),
                ("eta2_condition", "treatment condition"),
                ("r2_based", "BASED severity")]
    ramp = LinearSegmentedColormap.from_list("ink", [SURFACE, INK_2])
    for i, (key, _) in enumerate(REPS):
        ax = fig.add_subplot(gs[1, i])
        vals = np.array([[met.loc[key, c]] for c, _ in measures])
        ax.imshow(vals, cmap=ramp, vmin=0, vmax=0.9, aspect="auto")
        for r, v in enumerate(vals[:, 0]):
            ax.text(0, r, f"{v:.2f}", ha="center", va="center",
                    fontsize=8.6,
                    color="white" if v > 0.45 else INK_2,
                    fontweight="bold" if r == 0 else "normal")
        ax.set_xticks([])
        if i == 0:
            ax.set_yticks(range(3), [lbl for _, lbl in measures],
                          fontsize=8.0)
            ax.tick_params(length=0)
        else:
            ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    save(fig, out_dir, "fig_representation_geometry")


# ----------------------------------------------------------------------
def fig_errors(out_dir):
    df = pd.read_csv(os.path.join(out_dir, "task1_error_matrix.csv"))
    strata = [
        ("control clip", MUTED,
         df.case_control_label == "CONTROL"),
        ("case clip — experts saw IESS features", FEAT_HUE,
         (df.case_control_label == "CASE") & (df.human_label == 1)),
        ("case clip — experts saw no IESS features", ACCENT,
         (df.case_control_label == "CASE") & (df.human_label == 0)),
    ]

    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    fig.suptitle("Task-1 errors are few, concentrated, and interpretable",
                 fontsize=11, fontweight="bold", color=INK, y=1.00)

    W, dx = 5, 0.145                      # dots per row, dot pitch
    for n in range(10):
        sub = df[df.n_wrong == n]
        stack = []
        for _, hue, m in strata:
            stack += [hue] * int((sub.index.isin(df[m].index)).sum())
        for k, hue in enumerate(stack):
            row, col = divmod(k, W)
            ax.scatter(n + (col - (W - 1) / 2) * dx, row + 0.5, s=13,
                       c=hue, edgecolors="white", linewidths=0.45,
                       zorder=3, clip_on=False)
        if stack:
            ax.text(n, len(stack) / W + 1.3, str(len(stack)),
                    ha="center", fontsize=7.6, color=INK_2)

    # brace over the concentrated tail
    hard = df[df.n_wrong >= 5]
    share = hard.n_wrong.sum() / df.n_wrong.sum()
    y0 = 5.2
    ax.plot([4.62, 8.38], [y0, y0], color=INK_2, lw=0.8)
    ax.plot([4.62, 4.62], [y0, y0 - 0.7], color=INK_2, lw=0.8)
    ax.plot([8.38, 8.38], [y0, y0 - 0.7], color=INK_2, lw=0.8)
    ax.text(6.5, y0 + 0.9,
            f"{len(hard)} clips — {share:.0%} of all errors",
            ha="center", fontsize=8.2, color=INK, fontweight="bold")

    ax.set_xlim(-0.6, 9.6)
    ax.set_ylim(0, 27.5)
    ax.set_xticks(range(10))
    ax.set_yticks([])
    ax.set_xlabel("baselines wrong on the clip (of 9)", fontsize=8.4)
    ax.spines["left"].set_visible(False)
    style(ax, grid_axis=None)
    ax.text(0.995, 0.62, "each dot = one Routine Clip (200 total)",
            transform=ax.transAxes, ha="right", fontsize=7.2, color=MUTED)
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=hue,
               markeredgecolor="white", label=lbl)
        for lbl, hue, _ in strata],
        loc="upper right", frameon=False, fontsize=7.6,
        handletextpad=0.2, labelspacing=0.3)
    save(fig, out_dir, "fig_error_concentration")


# ----------------------------------------------------------------------
def fig_prognosis(out_dir):
    df = pd.read_csv(os.path.join(out_dir,
                                  "task1_confidence_prognosis.csv"))
    order = ["handcrafted", "cnn_resnet", "cnn_vit", "biot", "labram",
             "cbramod", "luna", "eegpt", "reve"]
    disp = {"handcrafted": "GBDT + Clinical Prior",
            "cnn_resnet": "3D ResNet-18", "cnn_vit": "3D ViT",
            "biot": "BIOT", "labram": "LaBraM", "cbramod": "CBraMod",
            "luna": "LUNA", "eegpt": "EEGPT", "reve": "REVE"}
    df = df.set_index("model").loc[order]
    ys = np.arange(len(order))[::-1]

    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    fig.suptitle("Held-out diagnostic confidence anticipates relapse,\n"
                 "not response", fontsize=11, fontweight="bold",
                 color=INK, y=1.04)

    ax.axvline(0.5, color=AXIS, lw=1.0, zorder=1)
    ax.text(0.503, ys.max() + 0.9, "chance", fontsize=7.2, color=MUTED)
    for y, m in zip(ys, order):
        a, b = df.loc[m, "auroc_meaningful"], df.loc[m, "auroc_immediate"]
        ax.plot([a, b], [y, y], color=GRID, lw=1.6, zorder=2)
    ax.scatter(df.auroc_immediate, ys, s=34, facecolors=SURFACE,
               edgecolors=MUTED, linewidths=1.3, zorder=3,
               label="immediate response")
    ax.scatter(df.auroc_meaningful, ys, s=36, c=ACCENT,
               edgecolors="white", linewidths=0.7, zorder=4,
               label="sustained response")
    ax.set_yticks(ys, [disp[m] for m in order], fontsize=8.0)
    ax.set_xlim(0.30, 0.70)
    ax.set_ylim(-0.9, ys.max() + 1.4)
    ax.set_xlabel("AUROC of the Task-1 case probability as a response "
                  "predictor\n(held-out folds, 50 cases; "
                  "$<$0.5 = higher confidence, worse outcome)",
                  fontsize=7.9)
    ax.legend(loc="lower left", frameon=True, framealpha=0.85,
              edgecolor=GRID, fontsize=7.4, handletextpad=0.2,
              borderaxespad=0.3, labelspacing=0.3)
    style(ax, grid_axis="x")
    save(fig, out_dir, "fig_confidence_prognosis")


def main():
    out_dir = env("IESSEEG_OUT")
    fig_geometry(out_dir)
    fig_errors(out_dir)
    fig_prognosis(out_dir)


if __name__ == "__main__":
    main()
