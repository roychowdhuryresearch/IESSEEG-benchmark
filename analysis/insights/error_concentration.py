#!/usr/bin/env python
"""Where does Task 1 fail? Error concentration across the nine baselines.

Builds the 200-clip x 9-model correctness matrix for the diagnosis task
(held-out fold predictions only) and asks whether errors are diffuse or
concentrated on a small set of recordings -- and what those recordings
have in common. Also cross-tabulates the subject-level diagnosis against
the clip-level expert consensus: Routine Clips of case subjects that the
experts labelled normal-looking are exactly the weak-supervision hazard
the task definition warns about.

Env: IESSEEG_RESULTS_ROOT, IESSEEG_LABELS, IESSEEG_OUT.
"""

import glob
import os

import numpy as np
import pandas as pd

from task1_prognosis import MODEL_RESULT_SUBDIR, env, load_task1_probs


def main():
    out_dir = env("IESSEEG_OUT")
    os.makedirs(out_dir, exist_ok=True)
    labels = pd.read_csv(env("IESSEEG_LABELS"))
    meta = labels[["short_recording_id", "patient_id", "case_control_label",
                   "human_label"]].set_index("short_recording_id")

    wide = {}
    for model in MODEL_RESULT_SUBDIR:
        frames = []
        for fold in range(5):
            d = os.path.join(env("IESSEEG_RESULTS_ROOT"),
                             MODEL_RESULT_SUBDIR[model], f"case_control_fold{fold}")
            hits = [h for h in glob.glob(os.path.join(d, "*inference_results.csv"))
                    if not h.endswith("_window.csv")]
            f = pd.read_csv(hits[0])
            rid = [c for c in f.columns if "recording_id" in c][0]
            frames.append(f.rename(columns={rid: "short_recording_id"}))
        df = pd.concat(frames).set_index("short_recording_id")
        wide[model] = (df.pred_label == df.known_label).astype(int)

    correct = pd.DataFrame(wide)
    correct = correct.join(meta)
    correct["n_wrong"] = (1 - correct[list(MODEL_RESULT_SUBDIR)]).sum(axis=1)

    print(f"clips: {len(correct)}   total errors: {int(correct.n_wrong.sum())}")
    dist = correct.n_wrong.value_counts().sort_index()
    print("\nclips by number of models wrong (0-9):")
    print(dist.to_string())

    hard = correct[correct.n_wrong >= 5]
    print(f"\nclips wrong for >=5 of 9 models: {len(hard)} "
          f"({len(hard) / len(correct):.0%} of clips, "
          f"{int(hard.n_wrong.sum())}/{int(correct.n_wrong.sum())} "
          f"= {hard.n_wrong.sum() / correct.n_wrong.sum():.0%} of all errors)")
    print(f"subjects contributing them: {hard.patient_id.nunique()}")

    # subject diagnosis vs clip-level expert consensus
    ct = pd.crosstab(correct.case_control_label, correct.human_label)
    print("\nsubject label x expert clip label:")
    print(ct.to_string())
    mism = correct[((correct.case_control_label == "CASE") & (correct.human_label == 0)) |
                   ((correct.case_control_label == "CONTROL") & (correct.human_label == 1))]
    print(f"\nclips whose expert label contradicts the subject diagnosis: {len(mism)}")
    if len(mism):
        print(f"  mean models-wrong on those clips: {mism.n_wrong.mean():.1f} "
              f"(vs {correct.n_wrong.mean():.1f} overall)")

    # error composition by stratum
    for name, m in [("case clips (expert=case)", (correct.case_control_label == "CASE") & (correct.human_label == 1)),
                    ("case clips (expert=control)", (correct.case_control_label == "CASE") & (correct.human_label == 0)),
                    ("control clips", correct.case_control_label == "CONTROL")]:
        sub = correct[m]
        if len(sub):
            print(f"{name:32s} n={len(sub):3d}  mean models-wrong {sub.n_wrong.mean():.2f}")

    correct.reset_index().to_csv(
        os.path.join(out_dir, "task1_error_matrix.csv"), index=False)
    print("\nsaved task1_error_matrix.csv")


if __name__ == "__main__":
    main()
