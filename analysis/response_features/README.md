# Published response-feature reproduction

This directory independently reimplements the PRE- and POST-treatment EEG
features used by Rajaraman et al. (2024) and compares the resulting
patient-level scores with the values reported in that article. It is a
same-cohort method reproduction, not an external validation study.

The analysis needs two local inputs that are not distributed here:

1. the source Clinical Clip EDF files; and
2. a governed clip-metadata CSV containing `patient_id`,
   `short_recording_id`, `case_control_label`, `pre_post_treatment_label`,
   `sleep_awake_label`, `meaningful_responder`, and `LeadtimeUKISS`.

The metadata-to-patient linkage must remain local. Both scripts therefore
require `--metadata-csv`, and generated patient-level files are written under
the Git-ignored `local_results/rajaraman2024/` directory.

## Run

From the repository root:

```bash
source /home/mingjian/miniforge3/etc/profile.d/mamba.sh
mamba activate seeg

# PLI extraction requires CUDA and exits if CUDA is unavailable.
CUDA_VISIBLE_DEVICES=3 python analysis/response_features/extract_rajaraman2024_raw_features.py \
  --features pli \
  --metadata-csv /path/to/local_clip_metadata.csv \
  --edf-dir /path/to/clinical_clip_edfs

# DFA and entropy extraction run on CPU.
python analysis/response_features/extract_rajaraman2024_raw_features.py \
  --features beta \
  --metadata-csv /path/to/local_clip_metadata.csv \
  --edf-dir /path/to/clinical_clip_edfs

python analysis/response_features/reproduce_rajaraman2024.py \
  --source raw \
  --metadata-csv /path/to/local_clip_metadata.csv
```

Extraction is resumable: each completed clip is cached as one local JSON file.
The final script averages the two matching clips for each patient and writes
the distribution and score comparisons. The reviewed, de-identified aggregate
tables from the completed run are in `reference_results/`.

See [`../../docs/rajaraman2024-reproduction.md`](../../docs/rajaraman2024-reproduction.md)
for the method correspondence, numerical findings, and limitations.

## Sources

- [Rajaraman et al. (2024)](https://doi.org/10.1016/j.clinph.2024.03.035)
- [Smith et al. (2021)](https://doi.org/10.1016/j.eplepsyres.2021.106704)
