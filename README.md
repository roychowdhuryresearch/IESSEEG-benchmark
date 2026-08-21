# IESSEEG Benchmark

Reference baselines for **IESSEEG**, a public pre-treatment EEG dataset for
Infantile Epileptic Spasms Syndrome (IESS): 266.9 hours of recordings from
100 infants (50 IESS cases, 50 age-matched controls), with awake and sleep
segments.

- Dataset: https://huggingface.co/datasets/roychowdhuryresearch/IESSEEG (CC BY 4.0)
- Preprocessing toolkit: https://github.com/roychowdhuryresearch/IESSEEG-toolbox

## Tasks

| Task | Name | Subjects | Target |
|---|---|---|---|
| 1 | Infantile Spasm Diagnosis | 100 | case vs. control |
| 2 | Immediate Treatment Response Prediction | 50 cases | responder vs. non-responder within two weeks |
| 3 | Sustained Treatment Response Prediction | 50 cases | relapse-free for at least four weeks |

All three are binary classification over 30-minute interictal clips.

## Evaluation protocol

Stratified **five-fold subject-wise cross-validation**. Subjects are split
into five patient-disjoint folds, stratified on the task label. Within a
fold, models train on the *Clinical Clips* of the remaining subjects and are
evaluated on the *Routine Clips* of the held-out subjects, so no patient
contributes to both sides.

The fold assignments are shipped in [`splits/`](splits/) rather than
regenerated per user, so results from different groups are computed on
identical partitions. Do not regenerate them to reproduce published numbers;
`scripts/create_kfold_split.py` is provided only for building splits over a
different cohort.

At inference each clip is tiled with non-overlapping windows; window
probabilities are averaged into one clip-level score and thresholded at 0.5.
Metrics are balanced accuracy, F1, and AUROC, reported as mean ± standard
deviation over the five folds.

## Baselines

| Key | Model | Montage | Window | Source |
|---|---|---|---|---|
| `handcrafted` | GBDT + Clinical Prior | 22-ch bipolar | 30 s | in-house |
| `cnn_resnet` | 3D ResNet-18 | 22-ch bipolar | 30 s | in-house |
| `cnn_vit` | 3D ViT | 22-ch bipolar | 30 s | in-house |
| `biot` | BIOT | 18-ch bipolar | 30 s | [official checkpoint](https://github.com/ycq091044/BIOT) |
| `labram` | LaBraM | 19-ch TUEG | 10 s | [official checkpoint](https://github.com/935963004/LaBraM) |
| `cbramod` | CBraMod | 19-ch TUEG | 30 s | [official checkpoint](https://github.com/wjq-learning/CBraMod) |

Foundation models are fine-tuned with their authors' recommended
hyperparameters (see [`configs/baselines.yaml`](configs/baselines.yaml)),
deliberately left untuned so the comparison measures out-of-the-box transfer
rather than a per-task hyperparameter search.

## Setup

```bash
pip install -r requirements.txt

# Where the preprocessed data trees live (see iesseeg/config.py for layout)
export IESSEEG_DATA_ROOT=/path/to/iesseeg

# Optional
export IESSEEG_PRETRAINED_DIR=/path/to/pretrained-models  # default ./pretrained-models
export IESSEEG_SCRATCH=/path/with/space                   # cache and temp files
export PYTHON_BIN=python                                  # interpreter to use
```

Preprocess the raw EDF release into the expected per-model `.npz` trees with
the [IESSEEG-toolbox](https://github.com/roychowdhuryresearch/IESSEEG-toolbox).
Pretrained checkpoints are not redistributed here; download them from each
model's official repository into `IESSEEG_PRETRAINED_DIR`:

| File | Model |
|---|---|
| `EEG-six-datasets-18-channels.ckpt` | BIOT |
| `labram-base.pth` | LaBraM |
| `pretrained_weights.pth` | CBraMod |

## Running

```bash
# Everything: train, infer, score, aggregate
bash scripts/run_benchmark.sh

# A subset of models, or a single stage
MODELS="biot cbramod" bash scripts/run_benchmark.sh
STAGE=inference bash scripts/run_benchmark.sh

# Pin a GPU (otherwise the least-busy free GPU is picked before each run)
CUDA_DEVICE=1 bash scripts/run_benchmark.sh

# Long runs: detach so they survive the shell closing
nohup bash scripts/run_benchmark.sh > run_benchmark.log 2>&1 &
```

Training is resumable: a fold whose checkpoint already exists is skipped.
Pass `FORCE_RETRAIN=1` to redo it.

Scoring and aggregation can also be run on their own:

```bash
python scripts/evaluate.py                                   # score + aggregate
python scripts/evaluate.py --latex_dir /path/to/paper/tables # also emit tables
```

This writes `results/results_all_folds.csv` (one row per model × task × fold)
and `results/results_summary.csv` (mean/std), and warns when a model × task
has fewer than five folds, so a partial rerun cannot quietly become a
headline number.

## Repository layout

```
iesseeg/              shared library
  config.py           all paths, resolved from environment variables
  data/               fold manifests and the windowed EEG dataset
  evaluation/         per-run metrics and cross-fold aggregation
baselines/            one directory per model (upstream code + runners)
configs/              protocol and per-model hyperparameters
scripts/              entry points; lib/common.sh holds shared shell setup
splits/               released fold manifests -- the fixed evaluation protocol
```

Adding a baseline means adding a directory under `baselines/` with a
`train_all.sh` and an `inference_all.sh` that source
`scripts/lib/common.sh`, then registering it in `scripts/run_benchmark.sh`
and `iesseeg/evaluation/aggregate.py`.

## Citation

```bibtex
@inproceedings{lu2025iesseeg,
  title  = {IESSEEG: A Public EEG Benchmark for Diagnosis and Treatment
            Response Prediction in Infantile Epileptic Spasms Syndrome},
  author = {Lu, Mingjian and Zhang, Yipeng and Daida, Atsuro and Kanai, Sotaro
            and Rajaraman, Rajsekar and Nariai, Hiroki and Oana, Shingo
            and Hussain, Shaun A. and Roychowdhury, Vwani},
  year   = {2025}
}
```

## License

Code is released under the MIT License (see [LICENSE](LICENSE)). The IESSEEG
dataset itself is distributed separately under CC BY 4.0. Vendored upstream
model code under `baselines/` remains under its original license.

Models trained on IESSEEG are for research use only and are not validated
for clinical decision-making.
