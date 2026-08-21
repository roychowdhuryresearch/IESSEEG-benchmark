#!/usr/bin/env python
"""Figures for the dataset-insight analyses.

fig_representation_geometry  What organizes each representation space over
                             the 115 BASED epochs: PCA scatters with
                             same-recording spiders (identity made visible
                             without color), plus the variance shares of
                             recording / condition / severity.
fig_error_concentration      Task-1 errors are few and concentrated, and
                             land where the clip-level evidence is absent.
fig_confidence_prognosis     Held-out diagnostic confidence vs treatment
                             response: nothing for immediate response, a
                             consistent inverse association for sustained.

Reads the CSV/NPZ outputs of the analysis scripts from IESSEEG_OUT and
writes PDF (paper) + PNG (inspection) into the same directory.

Style follows the paper's existing figure system (based_figure.py /
benchmark_figure.py): identical hues, surface, and hairline chrome.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PRE_HUE, POST_HUE = "#2a78d6", "#eb6834"
ACCENT, MUTED_PT = "#eb6834", "#898781"
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

REPS = [("handcrafted_122", "Clinical features (122-d)"),
        ("luna_embedding", "LUNA embedding"),
        ("reve_embedding", "REVE embedding"),
        ("eegpt_embedding", "EEGPT embedding")]


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


def fig_geometry(out_dir):
    z = np.load(os.path.join(out_dir, "geometry_pca_coords.npz"),
                allow_pickle=True)
    met = pd.read_csv(os.path.join(out_dir, "geometry_metrics.csv")) \
            .set_index("representation")
    rec, cond = z["recording_id"], z["cond"]

    fig = plt.figure(figsize=(10.6, 5.4))
    gs = fig.add_gridspec(2, 12, hspace=0.52, wspace=2.2,
                          height_ratios=[1.25, 1.0])

    for i, (key, label) in enumerate(REPS):
        ax = fig.add_subplot(gs[0, i * 3:(i + 1) * 3])
        P = z[f"pca_{key}"]
        # same-recording spiders: identity shown by structure, not color
        for r in np.unique(rec):
            m = rec == r
            cx, cy = P[m, 0].mean(), P[m, 1].mean()
            for x, y in P[m]:
                ax.plot([cx, x], [cy, y], color=AXIS, lw=0.5, zorder=1)
        for tag, hue in (("PRE", PRE_HUE), ("POST", POST_HUE)):
            m = cond == tag
            ax.scatter(P[m, 0], P[m, 1], s=11, c=hue, alpha=0.85,
                       linewidths=0, zorder=2)
        ax.set_title(label, fontsize=8.5, color=INK)
        ax.set_xticks([]), ax.set_yticks([])
        style(ax, grid_axis=None)
        knn = met.loc[key, "same_rec_1nn"]
        ax.text(0.02, 0.02, f"same-recording 1-NN {knn:.0%}",
                transform=ax.transAxes, fontsize=7.2, color=INK_2)
        if i == 0:
            ax.legend(handles=[
                plt.Line2D([], [], marker="o", ls="", color=PRE_HUE,
                           label="pre-treatment"),
                plt.Line2D([], [], marker="o", ls="", color=POST_HUE,
                           label="post-treatment")],
                loc="upper right", frameon=False, fontsize=7.2,
                handletextpad=0.2, borderaxespad=0.1)

    measures = [("eta2_recording", "recording identity"),
                ("eta2_condition", "treatment condition"),
                ("r2_based", "BASED severity")]
    short = ["Clinical\nfeatures", "LUNA", "REVE", "EEGPT"]
    for j, (col, label) in enumerate(measures):
        ax = fig.add_subplot(gs[1, j * 4:(j + 1) * 4])
        vals = [met.loc[k, col] for k, _ in REPS]
        ax.bar(range(4), vals, width=0.62, color=INK_2)
        for k, v in enumerate(vals):
            ax.text(k, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.2,
                    color=INK_2)
        ax.set_xticks(range(4), short, fontsize=7.4)
        ax.set_ylim(0, 0.92)
        ax.set_title(f"variance explained by {label}", fontsize=8.2,
                     color=INK)
        if j == 0:
            ax.set_ylabel("share of total variance", fontsize=7.6)
        else:
            ax.set_yticklabels([])
        style(ax)
    save(fig, out_dir, "fig_representation_geometry")


def fig_errors(out_dir):
    df = pd.read_csv(os.path.join(out_dir, "task1_error_matrix.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.5),
                             gridspec_kw={"wspace": 0.30, "width_ratios": [1.15, 1.0]})

    ax = axes[0]
    counts = df.n_wrong.value_counts().reindex(range(10), fill_value=0)
    ax.bar(counts.index, counts.values, width=0.7, color=INK_2)
    hard = df[df.n_wrong >= 5]
    ax.bar(counts.index[counts.index >= 5],
           counts.values[counts.index >= 5], width=0.7, color=ACCENT)
    for x, v in counts.items():
        if v:
            ax.text(x, v + 2.5, str(v), ha="center", fontsize=7.2, color=INK_2)
    ax.set_xlabel("baselines wrong on the clip (of 9)", fontsize=7.8)
    ax.set_ylabel("Routine Clips", fontsize=7.8)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 145)
    share = hard.n_wrong.sum() / df.n_wrong.sum()
    ax.set_title(f"{len(hard)} clips (7%) carry {share:.0%} of all errors",
                 fontsize=8.4, color=INK)
    style(ax)

    ax = axes[1]
    strata = [
        ("Control clip  (n=100)", (df.case_control_label == "CONTROL"), INK_2),
        ("Case clip, experts saw\nIESS features  (n=92)",
         (df.case_control_label == "CASE") & (df.human_label == 1), INK_2),
        ("Case clip, experts saw\nno IESS features  (n=8)",
         (df.case_control_label == "CASE") & (df.human_label == 0), ACCENT),
    ]
    ys = np.arange(len(strata))[::-1]
    for y, (label, m, hue) in zip(ys, strata):
        v = df[m].n_wrong.mean()
        ax.barh(y, v, height=0.55, color=hue)
        ax.text(v + 0.12, y, f"{v:.1f}", va="center", fontsize=7.6, color=INK_2)
    ax.set_yticks(ys, [s[0] for s in strata], fontsize=7.4)
    ax.set_xlabel("mean baselines wrong (of 9)", fontsize=7.8)
    ax.set_xlim(0, 5.6)
    ax.set_title("errors land where clip-level\nevidence is absent",
                 fontsize=8.4, color=INK)
    style(ax, grid_axis="x")
    save(fig, out_dir, "fig_error_concentration")


def fig_prognosis(out_dir):
    df = pd.read_csv(os.path.join(out_dir, "task1_confidence_prognosis.csv"))
    order = ["handcrafted", "cnn_resnet", "cnn_vit", "biot", "labram",
             "cbramod", "luna", "eegpt", "reve"]
    disp = {"handcrafted": "GBDT + Clinical Prior", "cnn_resnet": "3D ResNet-18",
            "cnn_vit": "3D ViT", "biot": "BIOT", "labram": "LaBraM",
            "cbramod": "CBraMod", "luna": "LUNA", "eegpt": "EEGPT",
            "reve": "REVE"}
    df = df.set_index("model").loc[order]

    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    ys = np.arange(len(order))[::-1]
    ax.axvline(0.5, color=AXIS, lw=0.9)
    ax.scatter(df.auroc_immediate, ys, s=26, facecolors=SURFACE,
               edgecolors=MUTED_PT, linewidths=1.2, zorder=3,
               label="immediate response")
    ax.scatter(df.auroc_meaningful, ys, s=26, c=ACCENT, linewidths=0,
               zorder=3, label="sustained response")
    ax.set_yticks(ys, [disp[m] for m in order], fontsize=7.6)
    ax.set_xlabel("AUROC of held-out Task-1 case probability\nfor predicting response (50 cases)",
                  fontsize=7.8)
    ax.set_xlim(0.28, 0.72)
    ax.text(0.503, ys.max() + 0.55, "chance", fontsize=7.0, color=MUTED)
    ax.legend(loc="upper right", frameon=False, fontsize=7.2,
              handletextpad=0.2)
    style(ax, grid_axis="x")
    save(fig, out_dir, "fig_confidence_prognosis")


def main():
    out_dir = env("IESSEEG_OUT")
    fig_geometry(out_dir)
    fig_errors(out_dir)
    fig_prognosis(out_dir)


if __name__ == "__main__":
    main()
