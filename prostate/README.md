# Prostate K-Space Classifier

Comparing MRI-derived image-domain classifiers against k-space-domain models for csPCa (PI-RADS ≥ 3).

## Project Overview
- **Image branch** now covers DWI-only, T2-only (`axt2`), and T2 + recon-DWI fusion models.
- **K-space branch** consumes post-GRAPPA k-space volumes via a KSPACE-NET-style FFTConv front-end.
- Modular data layer harmonizes metadata, HDF5 access, and preprocessing.
- Training utilities share engines, metrics, losses, logging, and configuration helpers.

## Requirements
- Python >= 3.10
- torch
- Access to H5 datasets and the labels CSV
- logging with weights and biases

## Data Inputs (required)
- Recon H5 root: directory of `<dat_filename>__<MRN>.h5` files with `metrics/{coil_combine}/{averaging_tag}/...`
- DICOM H5 roots: one or more directories of `<accession>.h5` with `axt2`, `adc`, and `b1500`
- Labels CSV: accession-level labels used by `load_accession_labels`
- K-space baseline additionally requires `kspace/post_grappa_full` in recon H5s

## Baselines & Cohort Alignment

This project supports three baselines for fair comparison:

1. **DICOM image classifier** — Uses scanner-reconstructed ADC/b1500 from DICOM-derived H5 files
2. **Reconstructed image classifier** — Uses selected recon metrics from k-space (ESC/RSS/ESPIRIT)
3. **K-space classifier** — Operates directly on post-GRAPPA k-space (same patients as #2)

`get_aligned_cohort()` in `h5_dataset.py` is the single source of truth for the label/recon/DICOM intersection.

## Recent updates (Feb 2026)
- **T2 image support added**: DICOM `axt2` can now train a single-stream T2-only baseline, and `image_source=hybrid` supports T2 + recon `adc`/`b1500` fusion on the existing aligned split.
- **Gleason supervision enabled for image and k-space training**: use `--enable-gleason-task` with grade-group threshold control via `--gleason-positive-grade-group-threshold` (default: `2`).
- **Grade-group imputation rule added**: when pathology has a non-null `biopsy_accession` but missing `highest_grade_group`, the sample is imputed as **GG=1**.
- **Image split strategies**:
  - `temporal`: patient-level temporal split (`train=0.75, val=0.15, test=0.10`) by each patient's first exam date.
  - `consecutive_patient_holdout`: contiguous temporal holdout where test is defined by `split_test_frac` tail or explicit `split_test_start_date`; train/val come from pre-test patients only (test patients are excluded from train/val to prevent leakage).
- **Fair comparison policy for `dicom_only` and `recon_only`**: aligned cohort defines the common validation/test holdout; source-cohort training excludes aligned val/test patients and excludes patients whose first scan date is on/after the aligned validation boundary date; leakage checks enforce no train-vs-holdout overlap at both patient and accession level.
- **W&B logging updated for reproducibility**: split distributions and fair-split metadata (including aligned val boundary date) are logged at step 0.

## Setup
1. Install [uv](https://github.com/astral-sh/uv).
2. Create the environment:
   ```bash
   uv sync
   ```
3. (Optional) Install development tools, including JupyterLab:
   ```bash
   uv sync --extra dev
   ```
4. (Optional) Configure W&B:
   - set `WANDB_API_KEY` and `WANDB_ENTITY`, or
   - pass `--wandb-mode offline` / `--wandb-mode disabled`.

## Notebook Workflow
From the repo root:

```bash
source .venv/bin/activate
jupyter lab
```

If you installed the development extra, the environment also includes `ipykernel` for selecting this repo's kernel in Jupyter.

## Quickstart (single runs)
Run from the repo root.

- Recon image baseline:
  ```bash
  uv run python scripts/train_image_metrics.py \
    --image-source recon \
    --h5-root /path/to/recon_h5 \
    --labels-csv /path/to/labels.csv \
    --coil-combine espirit \
    --averaging-tag b50_4_b1000_12
  ```
  Use repeated `--averaging-tags` flags instead of `--averaging-tag` to train one universal recon classifier across multiple tags.

- Recipe-backed ESPIRiT recon baseline:
  ```bash
  uv run python scripts/train_image_metrics.py \
    --image-source recon \
    --image-metrics adc_b1500 \
    --coil-combine espirit \
    --h5-root /gpfs/scratch/td2105/dwi_stream/recons_espirit_state_bank_v2 \
    --recipe-mode sampled_recipes \
    --recipes-per-accession 24 \
    --recipe-family-quotas single_dir=4 \
    --recipe-family-quotas dual_dir=8 \
    --recipe-family-quotas legacy_anchor=4 \
    --recipe-family-quotas xyz_equal=4 \
    --recipe-family-quotas xyz_asymmetric=4 \
    --labels-csv /path/to/labels.csv
  ```

- DICOM image baseline:
  ```bash
  uv run python scripts/train_image_metrics.py \
    --image-source dicom \
    --h5-root /path/to/recon_h5 \
    --dicom-h5-roots /path/to/dicom_h5_1 /path/to/dicom_h5_2 \
    --labels-csv /path/to/labels.csv \
    --cohort dicom_only
  ```

- T2-only image baseline:
  ```bash
  uv run python scripts/train_image_metrics.py \
    --image-source dicom \
    --image-metrics axt2 \
    --model-variant single_stream \
    --h5-root /path/to/recon_h5 \
    --dicom-h5-roots /path/to/dicom_h5_1 /path/to/dicom_h5_2 \
    --labels-csv /path/to/labels.csv \
    --cohort aligned
  ```

- T2 + recon-DWI fusion baseline:
  ```bash
  uv run python scripts/train_image_metrics.py \
    --image-source hybrid \
    --image-metrics adc_b1500 \
    --model-variant t2_recon_dwi_fusion \
    --h5-root /path/to/recon_h5 \
    --dicom-h5-roots /path/to/dicom_h5_1 /path/to/dicom_h5_2 \
    --labels-csv /path/to/labels.csv \
    --cohort aligned
  ```

- K-space baseline:
  ```bash
  uv run python scripts/train_kspace_post_grappa.py \
    --h5-root /path/to/recon_h5 \
    --labels-csv /path/to/labels.csv
  ```

## Recipe Modes
- `legacy_tags`: renders the current 15 path15 ESPIRiT states from `/gpfs/scratch/td2105/dwi_stream/recons_espirit_state_bank_v2`.
- `xyz_grid48`: renders all tied xyz states over `n50 in 1..4` and `n1000 in 1..12`.
- `sampled_recipes`: samples one-, two-, and three-direction recipes on the fly and writes fixed val/test manifests under each run's `output_dir/recipe_manifests/`.

## Key training scripts and flags

- `scripts/train_image_metrics.py`
  - `--image-source {recon,dicom,hybrid}`: use recon DWI metrics, DICOM volumes, or T2 + recon-DWI fusion.
  - `--model-variant {single_stream,t2_recon_dwi_fusion}`: explicit model family for new image work.
  - `--image-metrics {adc_b1500,trace_b50_b1000,axt2}`: choose the requested image metric set for the selected source.
  - `--coil-combine` and `--averaging-tag`: select one reconstruction H5 metric subgroup (e.g. `metrics/espirit/b50_4_b1000_12`).
  - `--averaging-tags`: repeat the flag to pool multiple recon averaging tags in one run; split assignment happens before tag expansion.
  - Recipe modes for bank-backed ESPIRiT runs:
    - `--recipe-mode legacy_tags`: current 15 path15 tags rendered on the fly from `/gpfs/scratch/td2105/dwi_stream/recons_espirit_state_bank_v2`
    - `--recipe-mode xyz_grid48`: all 48 tied xyz states (`n50 in 1..4`, `n1000 in 1..12`)
    - `--recipe-mode sampled_recipes`: sampled 1-, 2-, and 3-direction recipes with `--recipes-per-accession`, repeated `--recipe-family-quotas family=count`, and `--recipe-seed`
  - `--dicom-h5-roots ...`: one or more directories containing DICOM H5 files with `axt2`, `adc`, and `b1500`.
  - `--cohort {recon_only,dicom_only,aligned}`:
    - `aligned`: train/val/test all from aligned cohort.
    - `dicom_only` and `recon_only`: train on source cohort with fair filtering, validate/test on aligned holdout.
  - Split controls:
    - `--split-strategy {temporal,consecutive_patient_holdout}`
    - `--split-test-frac` (used by `consecutive_patient_holdout` when `--split-test-start-date` is not set)
    - `--split-test-start-date YYYY-MM-DD` (overrides `--split-test-frac` for reproducible fixed-date cutoff)
    - `--split-pool-train-frac` (train fraction inside the pre-test patient pool for `consecutive_patient_holdout`)
  - Backbone options: `--backbone {r3d_18,mc3_18,r2plus1d_18}` and `--backbone-pretrained` to use Kinetics weights.
  - Other common knobs: `--lr`, `--weight-decay`, `--batch-size`, `--epochs`, `--lr-scheduler`, `--lr-warmup-epochs`.
  - Logs split sizes to W&B at step 0 for reproducibility.

- `scripts/train_kspace_post_grappa.py`
  - `--cohort {recon_only,aligned}` (with `--dicom-h5-roots` for aligned) to control alignment with DICOM availability.
  - Backbone options: `--backbone {r3d_18,mc3_18,r2plus1d_18}` and `--backbone-pretrained`.
  - `--filter-mode {single_channel,multi_channel}`, `--num-kspace-filters`, plus the usual optimizer/loader flags.

- `configs/image/shared/`
  - `image_dicom.yaml`: single-stream DICOM ADC/b1500 baseline on the `dicom_only` cohort.
  - `image_t2.yaml`: aligned T2-only (`axt2`) best-config sweep.
  - `image_t2_source.yaml`: source-only T2 checkpoint for fusion transfer, trained on fair-filtered `dicom_only` data with `validation_policy: source_internal`.
- `configs/image/rss/`
  - `image_recon.yaml`: single-stream recon DWI baseline on the aligned cohort.
  - `image_recon_universal.yaml`: aligned universal recon DWI baseline across the 5 standard averaging tags.
  - `image_recon_universal_source.yaml`: source-only recon DWI checkpoint for fusion transfer, trained on fair-filtered `recon_only` data with `validation_policy: source_internal`.
  - `image_t2_recon_dwi_universal.yaml`: aligned T2 + recon-DWI fusion best-config sweep.
  - `image_t2_recon_dwi_universal_transfer_screen.yaml`: aligned RSS fusion transfer screen.
- `configs/image/espirit/`
  - `image_recon_universal.yaml`: aligned universal recon DWI baseline across the same 5 averaging tags using ESPIRiT recons.
  - `image_recon_universal_source.yaml`: ESPIRiT recon source checkpoint for fusion transfer.
  - `image_t2_recon_dwi_universal_transfer_screen.yaml`: aligned ESPIRiT fusion transfer screen using the ESPIRiT DWI source checkpoint plus the existing T2 source checkpoint.
  - `recipes/`: bank-backed ESPIRiT recipe configs for `legacy_tags`, `xyz_grid48`, and `sampled_recipes`.
- `configs/kspace/`
  - `kspace.yaml`: k-space baseline with `cohort: recon_only`.

## Grid and Slurm runs
Submit from the repo root.

- Generic / modality-only jobs:
  - `sbatch/grid_train_image_dicom.sh` -> `configs/image/shared/image_dicom.yaml`
  - `sbatch/grid_train_image_t2.sh` -> `configs/image/shared/image_t2.yaml`
  - `sbatch/grid_train_image_t2_source.sh` -> `configs/image/shared/image_t2_source.yaml`
  - `sbatch/grid_train_kspace.sh` -> `configs/kspace/kspace.yaml`
- Coil-combine specific image jobs:
  - `sbatch/image/rss/` contains RSS hardcoded entrypoints for recon-only and hybrid image runs.
  - `sbatch/image/espirit/` contains ESPIRiT hardcoded entrypoints for recon-only and hybrid image runs.

- RSS image jobs:
  - `sbatch/image/rss/grid_train_image_recon.sh` -> `configs/image/rss/image_recon.yaml`
  - `sbatch/image/rss/grid_train_image_recon_universal.sh` -> `configs/image/rss/image_recon_universal.yaml`
  - `sbatch/image/rss/grid_train_image_recon_universal_source.sh` -> `configs/image/rss/image_recon_universal_source.yaml`
  - `sbatch/image/rss/grid_train_image_t2_recon_dwi_universal.sh` -> `configs/image/rss/image_t2_recon_dwi_universal.yaml`
  - `sbatch/image/rss/grid_train_image_t2_recon_dwi_universal_transfer_screen.sh` -> `configs/image/rss/image_t2_recon_dwi_universal_transfer_screen.yaml`
- ESPIRiT image jobs:
  - `sbatch/image/espirit/grid_train_image_recon_universal.sh` -> `configs/image/espirit/image_recon_universal.yaml`
  - `sbatch/image/espirit/grid_train_image_recon_universal_source.sh` -> `configs/image/espirit/image_recon_universal_source.yaml`
  - `sbatch/image/espirit/grid_train_image_t2_recon_dwi_universal_transfer_screen.sh` -> `configs/image/espirit/image_t2_recon_dwi_universal_transfer_screen.yaml`
  - `sbatch/image/espirit/recipes/` contains the bank-backed ESPIRiT recipe wrappers for legacy-15, xyz48, and sampled runs.
  - Launch the ESPIRiT transfer-screen job only after the ESPIRiT recon source checkpoint has finished.

Example submissions:
```bash
sbatch sbatch/grid_train_image_dicom.sh
sbatch sbatch/grid_train_image_t2.sh
sbatch sbatch/grid_train_image_t2_source.sh
sbatch sbatch/grid_train_kspace.sh

sbatch sbatch/image/rss/grid_train_image_recon.sh
sbatch sbatch/image/rss/grid_train_image_recon_universal.sh
sbatch sbatch/image/rss/grid_train_image_recon_universal_source.sh
sbatch sbatch/image/rss/grid_train_image_t2_recon_dwi_universal.sh
sbatch sbatch/image/rss/grid_train_image_t2_recon_dwi_universal_transfer_screen.sh

sbatch sbatch/image/espirit/grid_train_image_recon_universal.sh
sbatch sbatch/image/espirit/grid_train_image_recon_universal_source.sh
sbatch sbatch/image/espirit/grid_train_image_t2_recon_dwi_universal_transfer_screen.sh
sbatch sbatch/image/espirit/recipes/grid_train_image_recon_universal_source_recipe_sampled.sh
```

Use `sbatch/analyze_image_checkpoints.sh` for generic image checkpoint analysis across DWI-only universal, T2-only, and T2 + recon-DWI runs.
- Supports 1-3 checkpoints per analysis job.
- Writes reusable prediction artifacts under `outputs/analysis/<run_name>/`, including:
  - `val_seed_predictions.csv`
  - `val_ensemble_predictions.csv`
  - `test_seed_predictions.csv`
  - `test_ensemble_predictions.csv`
  - `gleason_thresholds.csv`
  - `summary.json`
- Legacy compatibility aliases are still written for test predictions:
  - `seed_predictions.csv`
  - `ensemble_predictions.csv`
- Use `notebooks/recon_model_disagreement_review.ipynb` to compare two saved analysis dirs and review disagreement cases without rerunning inference.

Logs land under `logs/image_dicom/`, `logs/image_recon/`, `logs/image_recon_universal/`, `logs/image_t2/`, `logs/image_t2_recon_dwi_universal/`, and `logs/kspace/`.

## Outputs and logging
- Checkpoints and best models: `outputs/` (see run name in logs)
- W&B summary metric: `best_primary_metric` (with `best_primary_metric_name`)
- Reproducibility: set `--seed`; split policy metadata is logged at run start

## Repository Layout
```
prostate_kspace_classifier/
  ├── src/prostate_kspace_classifier/
  │   ├── config.py
  │   ├── data/
  │   ├── preproc/
  │   ├── models/
  │   ├── training/
  │   └── utils/
  ├── sbatch/
  └── scripts/
```

Each module includes docstrings and TODO markers to guide future extensions.
