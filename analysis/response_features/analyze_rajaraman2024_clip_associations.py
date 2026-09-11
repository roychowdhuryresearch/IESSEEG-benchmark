#!/usr/bin/env python
"""Test whether individual-clip EEG features are associated with response.

Each patient contributes two clips to a condition/state cell. The test statistic
is the clip-level AUROC, but response labels are permuted across patients, not
across clips. The two clips from one patient therefore remain together in every
permutation. This avoids treating correlated clips as independent patients.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from reproduce_rajaraman2024 import normalize_recording_id, response_value


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
LOCAL_RESULTS = REPO / "local_results" / "rajaraman2024"


def load_clip_quantities(features_dir: Path, metadata_csv: Path) -> dict[str, pd.DataFrame]:
    metadata = pd.read_csv(metadata_csv)
    metadata = metadata.loc[metadata.case_control_label.eq("CASE")].copy()
    metadata["recording_id"] = metadata.short_recording_id.map(normalize_recording_id)
    metadata["label"] = metadata.meaningful_responder.map(response_value)
    if len(metadata) != 400 or metadata.patient_id.nunique() != 50:
        raise RuntimeError("Expected 400 Clinical Clips from 50 case patients")

    tables: dict[str, list[dict[str, object]]] = {
        "pre_awake_dfa_intercept_beta": [],
        "pre_awake_connectivity_percent": [],
        "pre_awake_r0": [],
        "post_sleep_entropy_beta": [],
        "post_awake_dfa_intercept_beta": [],
    }
    for row in metadata.itertuples(index=False):
        pre_awake = row.pre_post_treatment_label == "PRE" and row.sleep_awake_label == "AWAKE"
        post_sleep = row.pre_post_treatment_label == "POST" and row.sleep_awake_label == "SLEEP"
        post_awake = row.pre_post_treatment_label == "POST" and row.sleep_awake_label == "AWAKE"
        if not (pre_awake or post_sleep or post_awake):
            continue
        path = features_dir / f"{row.recording_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        common = {"patient_id": int(row.patient_id), "label": int(row.label)}
        if pre_awake:
            dfa = float(payload["beta_dfa_intercept_mean"])
            connectivity = float(payload["connectivity_percent_retained_pli"])
            tables["pre_awake_dfa_intercept_beta"].append({**common, "value": dfa})
            tables["pre_awake_connectivity_percent"].append(
                {**common, "value": connectivity}
            )
            tables["pre_awake_r0"].append(
                {**common, "value": -2.361 * dfa - 0.051 * connectivity}
            )
        if post_sleep:
            tables["post_sleep_entropy_beta"].append(
                {**common, "value": float(payload["beta_entropy_mean"])}
            )
        if post_awake:
            tables["post_awake_dfa_intercept_beta"].append(
                {**common, "value": float(payload["beta_dfa_intercept_mean"])}
            )

    output = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    for name, table in output.items():
        counts = table.groupby("patient_id").size()
        if len(table) != 100 or len(counts) != 50 or not counts.eq(2).all():
            raise RuntimeError(f"{name}: expected two clips from each of 50 patients")
        if table.groupby("patient_id").label.nunique().ne(1).any():
            raise RuntimeError(f"{name}: response labels differ within patient")
    return output


def clip_auc(table: pd.DataFrame) -> float:
    ranks = rankdata(table.value.to_numpy(float), method="average")
    labels = table.label.to_numpy(int)
    n_positive = int(labels.sum())
    n_negative = len(labels) - n_positive
    u_statistic = ranks[labels == 1].sum() - n_positive * (n_positive + 1) / 2.0
    return float(u_statistic / (n_positive * n_negative))


def cluster_permutation_p_values(
    tables: dict[str, pd.DataFrame], n_permutations: int, seed: int
) -> dict[str, float]:
    names = list(tables)
    patients = np.sort(tables[names[0]].patient_id.unique())
    labels = (
        tables[names[0]].groupby("patient_id").label.first().reindex(patients).to_numpy(int)
    )
    n_positive_patients = int(labels.sum())
    n_positive_clips = 2 * n_positive_patients
    n_negative_clips = 2 * (len(patients) - n_positive_patients)

    patient_rank_sums = []
    observed = []
    for name in names:
        table = tables[name].copy()
        table["rank"] = rankdata(table.value.to_numpy(float), method="average")
        rank_sums = (
            table.groupby("patient_id")["rank"].sum().reindex(patients).to_numpy(float)
        )
        patient_rank_sums.append(rank_sums)
        observed.append(clip_auc(table))
    patient_rank_sums_array = np.stack(patient_rank_sums, axis=1)
    observed_distance = np.abs(np.asarray(observed) - 0.5)

    rng = np.random.default_rng(seed)
    extreme = np.zeros(len(names), dtype=np.int64)
    completed = 0
    while completed < n_permutations:
        batch_size = min(10_000, n_permutations - completed)
        random_order = np.argsort(rng.random((batch_size, len(patients))), axis=1)
        permuted_labels = np.zeros((batch_size, len(patients)), dtype=float)
        rows = np.arange(batch_size)[:, None]
        permuted_labels[rows, random_order[:, :n_positive_patients]] = 1.0
        positive_rank_sums = permuted_labels @ patient_rank_sums_array
        u_statistics = positive_rank_sums - n_positive_clips * (n_positive_clips + 1) / 2.0
        permuted_auc = u_statistics / (n_positive_clips * n_negative_clips)
        extreme += (np.abs(permuted_auc - 0.5) >= observed_distance).sum(axis=0)
        completed += batch_size
    return {
        name: float((count + 1) / (n_permutations + 1))
        for name, count in zip(names, extreme, strict=True)
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (len(p_values) - rank) * p_values[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def summarize(
    tables: dict[str, pd.DataFrame], n_permutations: int, seed: int
) -> pd.DataFrame:
    p_values = cluster_permutation_p_values(tables, n_permutations, seed)
    rows = []
    for name, table in tables.items():
        responders = table.loc[table.label.eq(1), "value"].to_numpy(float)
        nonresponders = table.loc[table.label.eq(0), "value"].to_numpy(float)
        rows.append(
            {
                "quantity": name,
                "n_patients": table.patient_id.nunique(),
                "n_clips": len(table),
                "responder_clips": len(responders),
                "nonresponder_clips": len(nonresponders),
                "responder_median": float(np.median(responders)),
                "nonresponder_median": float(np.median(nonresponders)),
                "clip_auroc": clip_auc(table),
                "patient_clustered_permutation_p": p_values[name],
            }
        )
    result = pd.DataFrame(rows)
    result["holm_p_across_five_quantities"] = holm_adjust(
        result.patient_clustered_permutation_p.tolist()
    )
    result["permutations"] = n_permutations
    result["seed"] = seed
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--raw-features-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument(
        "--output",
        type=Path,
        default=LOCAL_RESULTS / "clip_level_associations.csv",
    )
    args = parser.parse_args()
    if args.permutations < 1:
        raise ValueError("--permutations must be positive")
    tables = load_clip_quantities(args.raw_features_dir, args.metadata_csv)
    result = summarize(tables, args.permutations, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved aggregate results to {args.output}")


if __name__ == "__main__":
    main()
