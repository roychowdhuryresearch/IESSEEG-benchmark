#!/usr/bin/env python
"""Figures for the dataset-insight analyses.

fig_representation_geometry  UMAP layouts of each representation over the
                             115 BASED epochs -- points coloured by the
                             expert BASED score on a diverging ramp whose
                             poles match the paper's post/pre hues, with
                             smoothed per-recording hulls -- above hairline
                             bars of the variance shares.
fig_confidence_prognosis     Dumbbell plot: held-out diagnostic confidence
                             as a response predictor.
fig_error_concentration      (kept for the record; no longer in the paper)

Design per the paper's figure system + Rougier et al., "Ten Simple Rules
for Better Figures": no in-figure titles (the caption carries the
message), gray for context layers, colour reserved for the encoded
variable, direct labels over boxed legends.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import ConvexHull

PRE_HUE, POST_HUE = "#2a78d6", "#eb6834"
ACCENT = "#eb6834"
EPOCH_GRAY = "#b9b6ae"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

SEV_CMAP = LinearSegmentedColormap.from_list(
    "severity", [POST_HUE, "#dedbd3", PRE_HUE])
SEV_NORM = Normalize(vmin=0, vmax=5)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
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


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote", name)


def chaikin(pts, iters=3):
    """Corner-cutting smoothing of a closed polygon."""
    for _ in range(iters):
        nxt = []
        for i in range(len(pts)):
            p, q = pts[i], pts[(i + 1) % len(pts)]
            nxt.append(0.75 * p + 0.25 * q)
            nxt.append(0.25 * p + 0.75 * q)
        pts = np.asarray(nxt)
    return pts


def draw_recording_hulls(ax, P, rec):
    """Smoothed hull per connected component of each recording."""
    span = np.linalg.norm(P.max(0) - P.min(0))
    for r in np.unique(rec):
        pts_r = P[rec == r]
        labels = fcluster(linkage(pts_r, "single"),
                          t=0.18 * span, criterion="distance")
        for g in np.unique(labels):
            pts = pts_r[labels == g]
            if len(pts) < 3:
                if len(pts) == 2:
                    ax.plot(pts[:, 0], pts[:, 1], color=GRID, lw=0.8,
                            zorder=1)
                continue
            hull = pts[ConvexHull(pts).vertices]
            c = hull.mean(0)
            hull = c + (hull - c) * 1.18
            sm = chaikin(hull)
            ax.fill(sm[:, 0], sm[:, 1], color=INK_2, alpha=0.045, lw=0,
                    zorder=1)
            ax.plot(np.r_[sm[:, 0], sm[0, 0]], np.r_[sm[:, 1], sm[0, 1]],
                    color="#d9d6cf", lw=0.9, zorder=1)


# ----------------------------------------------------------------------
def fig_geometry(out_dir):
    z = np.load(os.path.join(out_dir, "geometry_pca_coords.npz"),
                allow_pickle=True)
    met = pd.read_csv(os.path.join(out_dir, "geometry_metrics.csv")) \
            .set_index("representation")
    rec, based = z["recording_id"], z["based"].astype(float)

    fig = plt.figure(figsize=(10.2, 4.35))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.6, 1.0],
                          hspace=0.30, wspace=0.12,
                          left=0.085, right=0.905, top=0.90, bottom=0.05)
    fig.text(0.015, 0.93, "A", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.015, 0.285, "B", fontsize=12, fontweight="bold", color=INK)

    for i, (key, label) in enumerate(REPS):
        ax = fig.add_subplot(gs[0, i])
        P = z[f"umap_{key}"]
        draw_recording_hulls(ax, P, rec)
        ax.scatter(P[:, 0], P[:, 1], s=17, c=based, cmap=SEV_CMAP,
                   norm=SEV_NORM, edgecolors="white", linewidths=0.5,
                   zorder=3)
        ax.set_title(label, fontsize=9.5, fontweight="bold", color=INK,
                     pad=15)
        knn = met.loc[key, "same_rec_1nn"]
        ax.text(0.5, 1.013,
                f"same-recording 1-NN {knn:.0%}  ·  chance 5%",
                transform=ax.transAxes, ha="center", fontsize=6.9,
                color=MUTED)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        m0, m1 = P.min(0), P.max(0)
        pad = 0.10 * (m1 - m0)
        ax.set_xlim(m0[0] - pad[0], m1[0] + pad[0])
        ax.set_ylim(m0[1] - pad[1], m1[1] + pad[1])

    cax = fig.add_axes([0.925, 0.47, 0.011, 0.36])
    cb = fig.colorbar(ScalarMappable(norm=SEV_NORM, cmap=SEV_CMAP), cax=cax)
    cb.set_ticks(range(6))
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(labelsize=6.8, color=AXIS, length=2)
    cax.set_title("expert\nBASED", fontsize=6.9, color=INK_2, pad=5)

    # hairline variance-share bars, aligned under each panel
    measures = [("eta2_recording", "recording identity", INK_2),
                ("eta2_condition", "treatment condition", EPOCH_GRAY),
                ("r2_based", "BASED severity", EPOCH_GRAY)]
    fig.text(0.085, 0.315, "share of total variance explained by",
             fontsize=7.2, color=MUTED)
    for i, (key, _) in enumerate(REPS):
        ax = fig.add_subplot(gs[1, i])
        ys = np.arange(3)[::-1]
        for y, (col, lbl, hue) in zip(ys, measures):
            v = met.loc[key, col]
            ax.barh(y, v, height=0.52, color=hue, zorder=2)
            ax.text(v + 0.025, y, f"{v:.2f}", va="center", fontsize=7.0,
                    color=INK_2)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.6, 2.6)
        ax.set_xticks([])
        if i == 0:
            ax.set_yticks(ys, [m[1] for m in measures], fontsize=7.4)
            ax.tick_params(length=0)
        else:
            ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    save(fig, out_dir, "fig_representation_geometry")


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

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.axvline(0.5, color=AXIS, lw=0.9, zorder=1)
    ax.text(0.503, ys.max() + 0.85, "chance", fontsize=7.0, color=MUTED)
    for y, m in zip(ys, order):
        a, b = df.loc[m, "auroc_meaningful"], df.loc[m, "auroc_immediate"]
        ax.plot([a, b], [y, y], color=GRID, lw=1.5, zorder=2)
    ax.scatter(df.auroc_immediate, ys, s=30, facecolors=SURFACE,
               edgecolors=MUTED, linewidths=1.2, zorder=3,
               label="immediate response")
    ax.scatter(df.auroc_meaningful, ys, s=32, c=ACCENT,
               edgecolors="white", linewidths=0.6, zorder=4,
               label="sustained response")
    ax.set_yticks(ys, [disp[m] for m in order], fontsize=7.8)
    ax.set_xlim(0.30, 0.70)
    ax.set_ylim(-0.9, ys.max() + 1.3)
    ax.set_xlabel("AUROC of the Task-1 case probability as a response "
                  "predictor\n(held-out folds, 50 cases; "
                  "$<$0.5 = higher confidence, worse outcome)",
                  fontsize=7.7)
    ax.legend(loc="lower left", frameon=False, fontsize=7.2,
              handletextpad=0.25, labelspacing=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, out_dir, "fig_confidence_prognosis")


def main():
    out_dir = env("IESSEEG_OUT")
    fig_geometry(out_dir)
    fig_prognosis(out_dir)


if __name__ == "__main__":
    main()
