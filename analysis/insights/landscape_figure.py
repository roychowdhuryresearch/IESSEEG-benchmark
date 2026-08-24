#!/usr/bin/env python
"""Positioning figure: the open-EEG landscape by population and question.

A conceptual map (data enumerated below, verified in the paper's Related
Work): columns are the question a corpus's labels pose, ordered from
annotation-time to decision-facing; rows are population age. Occupancy
collapses toward the upper left; each decision-facing column has one
occupant; IESSEEG spans three decision-facing cells in the infant row.

Authored at print size (5.5 in) per the paper's figure system.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACCENT = "#eb6834"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

COLS = ["seizure\ndetection", "staging &\ndecoding", "abnormality &\ndiagnosis",
        "severity\ngrading", "outcome\nprognosis", "treatment\nresponse"]
ROWS = ["adult", "child", "infant", "neonate"]          # top to bottom

# (col, row, label, dy) -- dy nudges stacked labels apart
CORPORA = [
    (0, 0, "TUSZ", 0), (0, 1, "CHB-MIT", 0), (0, 3, "Helsinki", 0),
    (1, 0, "Sleep-EDF, SHHS, BCI …", 0),
    (2, 0, "TUAB", 0.16), (2, 0, "TDBRAIN", -0.16), (2, 2, "EEG-IP", 0),
    (3, 3, "Cork HIE", 0),
    (4, 0, "I-CARE", 0),
    (5, 0, "TDBRAIN", 0),
]
IESSEEG_COLS = [2, 3, 5]


def main():
    out_dir = os.environ.get("IESSEEG_OUT", ".")
    fig, ax = plt.subplots(figsize=(5.5, 1.95))

    nx, ny = len(COLS), len(ROWS)
    # cell grid
    for i in range(nx + 1):
        ax.axvline(i, color=GRID, lw=0.5)
    for j in range(ny + 1):
        ax.axhline(j, color=GRID, lw=0.5)
    # decision-facing region: barely-there warm tint
    ax.axvspan(3, 6, color=ACCENT, alpha=0.045, lw=0)

    for col, row, label, dy in CORPORA:
        y = ny - 1 - row + 0.5 + dy
        ax.scatter(col + 0.5, y, s=9, c=INK_2, zorder=3, linewidths=0)
        ax.text(col + 0.5, y - 0.14, label, ha="center", va="top",
                fontsize=5.4, color=INK_2)

    # IESSEEG: three linked cells in the infant row
    y = ny - 1 - 2 + 0.5
    xs = [c + 0.5 for c in IESSEEG_COLS]
    ax.plot([min(xs), max(xs)], [y + 0.18, y + 0.18], color=ACCENT,
            lw=1.1, zorder=2)
    for x, col in zip(xs, IESSEEG_COLS):
        if col == 3:   # severity: released annotation, not yet a task
            ax.scatter(x, y + 0.18, s=16, facecolors="white",
                       edgecolors=ACCENT, linewidths=1.0, zorder=3)
        else:
            ax.scatter(x, y + 0.18, s=16, c=ACCENT, edgecolors="white",
                       linewidths=0.5, zorder=3)
    ax.text(np.mean(xs), y - 0.02, "IESSEEG (ours)", ha="center", va="top",
            fontsize=6.2, color=ACCENT, fontweight="bold")

    # region brackets above the grid
    for x0, x1, label in [(0, 3, "answerable at annotation time"),
                          (3, 6, "decision-facing")]:
        ax.plot([x0 + 0.06, x1 - 0.06], [ny + 0.22, ny + 0.22],
                color=MUTED, lw=0.7)
        ax.text((x0 + x1) / 2, ny + 0.34, label, ha="center", va="bottom",
                fontsize=5.8, color=INK_2, style="italic")

    ax.set_xticks(np.arange(nx) + 0.5, COLS, fontsize=5.9)
    ax.set_yticks(np.arange(ny) + 0.5, ROWS[::-1], fontsize=6.2)
    ax.tick_params(length=0, pad=2, labelcolor=INK_2)
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny + 0.75)
    for s in ax.spines.values():
        s.set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_landscape.{ext}"),
                    bbox_inches="tight", dpi=300)
    print("wrote fig_landscape")


if __name__ == "__main__":
    main()
