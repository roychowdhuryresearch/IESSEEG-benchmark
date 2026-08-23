#!/usr/bin/env python
"""The paper's three story figures, authored at print size.

fig_overview          Hero figure: real released traces (case vs control),
                      the cohort/label structure, and the decision chain
                      with each link's status. Replaces the clip-art
                      overview.
fig_window_timeline   LaBraM's window-level case probability across three
                      held-out Routine Clips: evidence is episodic in
                      cases, absent in expert-normal case clips, flat-low
                      in controls -- the weak-supervision mechanism, shown.
fig_severity_collapse The pooled-vs-within-condition correlation for all
                      122 clinical features and the four model
                      probabilities in one panel: pooled 'severity'
                      correlations collapse once treatment condition is
                      held fixed.

Env: IESSEEG_DATA (data/ root), IESSEEG_RESULTS_ROOT (kfold baselines),
     IESSEEG_LABELS (final_test.csv), IESSEEG_OUT (kfold_split/insights,
     holding task1_error_matrix.csv + based_feature_correlations.csv).
"""

import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

PRE_HUE, POST_HUE = "#2a78d6", "#eb6834"
ACCENT = "#eb6834"
BLUE = "#2a78d6"
GREEN = "#1baf7a"
SURFACE = "#fcfcfb"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.linewidth": 0.5,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 6, "ytick.labelsize": 6,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def env(name):
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Set {name}")
    return v


def save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", name)


def pick_clips():
    """Choose exemplar held-out clips from the error matrix + LaBraM
    window probabilities: a clean control, a correct case with episodic
    evidence, one of the case clips in which experts saw no IESS-specific
    features, and -- for the hero traces -- a Routine Clip from the
    highest-BASED expert-scored patient (guaranteed florid background)."""
    em = pd.read_csv(os.path.join(env("IESSEEG_OUT"),
                                  "task1_error_matrix.csv"))
    win = load_windows()
    stats = win.groupby("short_recording_id").pred_prob \
               .agg(["mean", "std", "min", "max"])
    em = em.merge(stats, left_on="short_recording_id", right_index=True)

    ctrl = em[(em.case_control_label == "CONTROL") & (em.n_wrong == 0)
              & (em["max"] < 0.5)].sort_values("std").iloc[0]
    episodic = em[(em.case_control_label == "CASE") & (em.human_label == 1)
                  & (em.labram == 1) & (em["min"] < 0.4)
                  & (em["max"] > 0.9)].sort_values("std").iloc[-1]
    absent = em[(em.case_control_label == "CASE") & (em.human_label == 0)] \
        .sort_values("mean").iloc[0]

    scores = pd.read_csv(os.path.join(
        os.path.dirname(env("IESSEEG_LABELS")), "..",
        "based_analysis", "results", "labram_scores.csv"))
    rec_sev = scores.groupby("recording_id").based.mean()
    labels = pd.read_csv(env("IESSEEG_LABELS"))
    sev_clips = labels[labels.long_recording_id.isin(rec_sev.index)].copy()
    sev_clips["sev"] = sev_clips.long_recording_id.map(rec_sev)
    hero_case = sev_clips.sort_values("sev").iloc[-1].short_recording_id
    return ctrl, episodic, absent, win, hero_case


def best_window(short_id, mode, dur=10.0, sf=200):
    """Scan 10-s windows: 'florid' returns the highest-amplitude window,
    'clean' the lowest-artifact one with live signal."""
    z = np.load(os.path.join(env("IESSEEG_DATA"), "baseline_test",
                             f"{short_id}.npz"), allow_pickle=True)
    data = z["data"][:8].astype(np.float64)
    n = int(dur * sf)
    scores = []
    for t0 in range(0, min(data.shape[1] - n, 10 * 60 * sf), 5 * sf):
        seg = data[:, t0:t0 + n]
        seg = seg - seg.mean(axis=1, keepdims=True)
        p95 = np.percentile(np.abs(seg), 95)
        pmax = np.abs(seg).max()
        alive = seg.std(axis=1).min() > 1.0
        clean = pmax < 3.5 * p95          # rejects electrode pops
        scores.append((t0, p95, alive, clean))
    if mode == "florid":
        cand = [s for s in scores if s[2] and s[3]]
        return max(cand, key=lambda s: s[1])[0] / sf
    cand = [s for s in scores if s[2] and s[3]]
    return min(cand, key=lambda s: s[1])[0] / sf


def load_windows():
    frames = []
    for fold in range(5):
        f = os.path.join(env("IESSEEG_RESULTS_ROOT"),
                         "labram/result/inference",
                         f"case_control_fold{fold}",
                         "inference_results_window.csv")
        frames.append(pd.read_csv(f))
    return pd.concat(frames)


# ----------------------------------------------------------------------
def draw_traces(ax, short_id, t0, n_ch=8, dur=10.0, label="", step=None,
                scalebar=False):
    z = np.load(os.path.join(env("IESSEEG_DATA"), "baseline_test",
                             f"{short_id}.npz"), allow_pickle=True)
    sf = 200
    seg = z["data"][:n_ch, int(t0 * sf):int((t0 + dur) * sf)].astype(float)
    seg = seg - seg.mean(axis=1, keepdims=True)
    if step is None:
        step = np.percentile(np.abs(seg), 99.5) * 2.0
    t = np.arange(seg.shape[1]) / sf
    for i in range(n_ch):
        ax.plot(t, np.clip(seg[i], -2.5 * step, 2.5 * step) - i * step,
                color=INK_2, lw=0.28)
    ax.set_xlim(-0.1, dur)
    y_lo = -(n_ch + (1.6 if scalebar else -0.2)) * step
    ax.set_ylim(y_lo, 1.6 * step)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(label, fontsize=6.2, color=INK, pad=3, loc="left")
    if scalebar:
        x0, y0 = dur - 1.05, -(n_ch + 1.35) * step
        ax.plot([x0, x0 + 1], [y0, y0], color=INK, lw=0.9)
        ax.plot([x0, x0], [y0, y0 + 200], color=INK, lw=0.9)
        ax.text(x0 + 0.55, y0 - 0.03 * step, "1 s", ha="center", va="top",
                fontsize=4.8, color=INK_2)
        ax.text(x0 - 0.12, y0 + 100, "200 µV", ha="right", va="center",
                fontsize=4.8, color=INK_2)
    return step


def fig_overview(out_dir, ctrl_id, case_id):
    fig = plt.figure(figsize=(5.5, 2.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.6],
                          height_ratios=[1.9, 1.0],
                          left=0.015, right=0.99, top=0.90, bottom=0.03,
                          wspace=0.06, hspace=0.28)

    # -- A: real released traces, common uV scale ----------------------
    axA = fig.add_subplot(gs[0, 0])
    axA.axis("off")
    t_case = best_window(case_id, "florid")
    t_ctrl = best_window(ctrl_id, "clean")
    inner = axA.inset_axes([0.0, 0.54, 1.0, 0.42])
    draw_traces(inner, case_id, t_case, step=150,
                label="IESS case — Routine Clip (10 s)")
    inner2 = axA.inset_axes([0.0, 0.0, 1.0, 0.42])
    draw_traces(inner2, ctrl_id, t_ctrl, label="age-matched control "
                "(same scale)", step=150, scalebar=True)
    fig.text(0.015, 0.935, "A", fontsize=9, fontweight="bold", color=INK)

    # -- B: cohort and label structure ---------------------------------
    axB = fig.add_subplot(gs[0, 1])
    axB.axis("off")
    axB.set_xlim(0, 1), axB.set_ylim(0, 1)
    fig.text(0.415, 0.935, "B", fontsize=9, fontweight="bold", color=INK)

    def box(ax, x, y, w, h, lines, sizes=None, edge=AXIS, face="white"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
            fc=face, ec=edge, lw=0.7, mutation_aspect=0.6))
        n = len(lines)
        for i, ln in enumerate(lines):
            sz = (sizes or [6.2] * n)[i]
            wt = "bold" if i == 0 else "normal"
            ax.text(x + w / 2, y + h * (1 - (i + 0.55) / n), ln,
                    ha="center", va="center", fontsize=sz, color=INK
                    if i == 0 else INK_2, fontweight=wt)

    box(axB, 0.18, 0.80, 0.64, 0.17,
        ["IESSEEG — 100 infants", "266.9 h pre-treatment video-EEG"],
        sizes=[6.8, 5.8])
    box(axB, 0.02, 0.44, 0.44, 0.24,
        ["50 IESS cases", "spasm-confirmed diagnosis",
         "NISC treatment outcomes"], sizes=[6.4, 5.4, 5.4])
    box(axB, 0.54, 0.44, 0.44, 0.24,
        ["50 controls", "age-matched,", "neurologically normal"],
        sizes=[6.4, 5.4, 5.4])
    for x in (0.24, 0.76):
        axB.add_patch(FancyArrowPatch((0.5, 0.79), (x, 0.70),
                                      arrowstyle="-", color=AXIS, lw=0.7))
    axB.text(0.24, 0.32, "32 immediate → 28 sustained responders\n"
                         "(4 relapse) · 18 non-responders",
             ha="center", fontsize=5.2, color=INK_2)
    axB.text(0.76, 0.32, "no antiepileptic therapy,\nno outcome labels",
             ha="center", fontsize=5.2, color=MUTED)
    axB.text(0.5, 0.10, "+ BASED severity annotations: 20 recordings × 7 raters "
                        "(140 expert scores)",
             ha="center", fontsize=5.4, color=INK_2, style="italic")

    # -- C: the decision chain, with status ----------------------------
    axC = fig.add_subplot(gs[1, :])
    axC.axis("off")
    axC.set_xlim(0, 1), axC.set_ylim(0, 1)
    fig.text(0.015, 0.315, "C", fontsize=9, fontweight="bold", color=INK)
    chain = [
        (0.075, "Detect", "Task 1 — diagnosis benchmark",
         "largely solvable: 0.93 balanced accuracy", BLUE),
        (0.405, "Grade severity", "BASED annotations released",
         "open: no representation grades it (§7)", GREEN),
        (0.735, "Predict response", "Tasks 2–3 — open challenges",
         "adjudicated, not ranked: registered lead (§4–5)", ACCENT),
    ]
    for x, head, sub, status, hue in chain:
        axC.add_patch(FancyBboxPatch(
            (x, 0.42), 0.19, 0.42,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            fc="white", ec=hue, lw=1.0, mutation_aspect=0.35))
        axC.text(x + 0.095, 0.72, head, ha="center", fontsize=6.6,
                 color=INK, fontweight="bold")
        axC.text(x + 0.095, 0.52, sub, ha="center", fontsize=5.2,
                 color=INK_2)
        axC.text(x + 0.095, 0.20, status, ha="center", fontsize=5.2,
                 color=hue)
    for x0 in (0.27, 0.60):
        axC.add_patch(FancyArrowPatch(
            (x0 + 0.005, 0.63), (x0 + 0.13, 0.63),
            arrowstyle="-|>", mutation_scale=7, color=INK_2, lw=0.9))
    save(fig, out_dir, "fig_overview")


# ----------------------------------------------------------------------
def fig_window_timeline(out_dir, ctrl, episodic, absent, win):
    rows = [
        (ctrl, "control — evidence absent, correctly rejected", MUTED),
        (episodic, "case — evidence is episodic, correctly detected", BLUE),
        (absent, "case clip in which experts saw no IESS features", ACCENT),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(5.5, 2.2), sharex=True,
                             gridspec_kw=dict(hspace=0.62, left=0.075,
                                              right=0.99, top=0.93,
                                              bottom=0.15))
    for ax, (clip, label, hue) in zip(axes, rows):
        w = win[win.short_recording_id == clip.short_recording_id] \
            .sort_values("start_ind")
        t = w.start_ind.values / 200 / 60
        p = w.pred_prob.values
        ax.axhline(0.5, color=AXIS, lw=0.5, ls=(0, (3, 2)))
        ax.fill_between(t, 0.5, np.maximum(p, 0.5), step="post",
                        color=hue, alpha=0.30, lw=0)
        ax.step(t, p, where="post", color=hue, lw=0.8)
        ax.set_ylim(-0.04, 1.06)
        ax.set_yticks([0, 0.5, 1], ["0", ".5", "1"], fontsize=5.2)
        ax.text(0.0, 1.06, label, transform=ax.transAxes, fontsize=5.6,
                color=INK_2, va="bottom")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[-1].set_xlim(0, 30)
    axes[-1].set_xlabel("time within the held-out 30-minute Routine Clip "
                        "(min)", fontsize=6.2, labelpad=1.5)
    axes[1].set_ylabel("LaBraM case probability", fontsize=6.0)
    save(fig, out_dir, "fig_window_timeline")


# ----------------------------------------------------------------------
def fig_severity_collapse(out_dir):
    corr = pd.read_csv(os.path.join(env("IESSEEG_OUT"),
                                    "based_feature_correlations.csv"))
    fig, ax = plt.subplots(figsize=(3.1, 2.2))
    lim = 0.75
    ax.plot([-lim, lim], [-lim, lim], color=AXIS, lw=0.6, ls=(0, (3, 2)))
    ax.axhline(0, color=AXIS, lw=0.5)
    ax.axvline(0, color=AXIS, lw=0.5)
    ax.scatter(corr.rho_pooled, corr.rho_pre, s=5, c=MUTED, alpha=0.55,
               lw=0, label="clinical feature (122)")
    models = [("LaBraM", 0.65, 0.23, BLUE, "o"),
              ("LUNA", 0.42, -0.07, GREEN, "s"),
              ("REVE", 0.17, -0.09, GREEN, "D"),
              ("EEGPT", 0.33, 0.36, GREEN, "^")]
    for name, xp, yp, hue, mk in models:
        ax.scatter(xp, yp, s=17, c=hue, marker=mk, edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax.annotate(name, (xp, yp), textcoords="offset points",
                    xytext=(3, 3), fontsize=5.2, color=INK_2)
    ax.annotate("if pooled correlation\nreflected severity",
                xy=(0.52, 0.52), xytext=(0.10, 0.62), fontsize=5.2,
                color=MUTED,
                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.5))
    ax.set_xlim(-0.3, lim)
    ax.set_ylim(-0.45, lim)
    ax.set_xlabel("Spearman ρ with BASED, pooled", fontsize=6.2,
                  labelpad=1.5)
    ax.set_ylabel("ρ within pre-treatment", fontsize=6.2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, out_dir, "fig_severity_collapse")


def main():
    out_dir = env("IESSEEG_OUT")
    ctrl, episodic, absent, win, hero_case = pick_clips()
    print("clips: ctrl", ctrl.short_recording_id,
          "episodic", episodic.short_recording_id,
          "absent", absent.short_recording_id, "hero", hero_case)
    fig_overview(out_dir, ctrl.short_recording_id, hero_case)
    fig_window_timeline(out_dir, ctrl, episodic, absent, win)
    fig_severity_collapse(out_dir)


if __name__ == "__main__":
    main()
