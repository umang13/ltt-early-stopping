# ltt-early-stopping

Code for the ICLR experiments on risk-controlled stopping (RCS). RCS wraps a fixed acquisition trajectory, or a learned acquisition policy, with Learn-Then-Test so that a user-specified risk on the samples that actually stop is controlled at a finite-sample level.

The five experiments in the paper are:

| Paper figure | What is controlled | Entry point |
|---|---|---|
| Figure 2 | CIFAR-10 misclassification, fixed patch order | `bin/eval_compare_cifar10.py` |
| Figure 3 | FastMRI knee class-conditional FNR (ACL, meniscus) | `bin/eval_compare_knee.py` |
| Figure 4 | Prostate PI-RADS FNR, asymmetric ambiguity bands | `bin/eval_compare_prostate_pirads_asym.py` |
| Figure 5, top | Stopping vs. risk level α on knee and prostate | `bin/plot_alpha_stopping.py` |
| Figure 5, bottom | JAFA acquisition policy wrapped with RCS | `bin/eval_compare_rl_jafa_probs.py` |

Checkpoints, FastMRI volumes, and the prostate cohort are not in this tree. Several scripts still carry absolute paths from the original cluster as defaults. Use the commands below and pass your own paths.

Run the RCS commands from this directory. Run the JAFA trainer from `jafa/`.

## Setup

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e prostate
```

`wandb` is used by the trainers. Set `WANDB_MODE=disabled` to train without an account.

## 1. CIFAR-10 patch acquisition (Figure 2)

Train a ResNet-18 with ARMS on biased random-rate 4×4 patch masks. The published checkpoint name is `cifar10_baseline_epoch=49_val_accuracy=0.7238.ckpt`, which is the filename pattern from `bin/train_cifar10_baseline.py`. CIFAR-10 is downloaded into `configs/partial/cifar10_bayes_biased_random_rate_config.yaml`'s `datadir` (`./data/cifar10`).

```bash
python bin/train_cifar10_baseline.py \
  --config configs/partial/cifar10_bayes_biased_random_rate_config.yaml
```

Calibrate a max-softmax stopping rule at α = 0.1, δ = 0.05. The estimation set is 200 validation images and the calibration set is the rest of the 5,000-image validation split. Evaluation is on the 10,000-image test set. λ is a 101-point grid. Confidence and entropy baselines use threshold 0.5.

```bash
python bin/eval_compare_cifar10.py \
  --config_path configs/partial/cifar10_bayes_biased_random_rate_config.yaml \
  --checkpoint_path results/checkpoints/cifar10/bayes_biased_random_rate/cifar10_baseline_epoch=49_val_accuracy=0.7238.ckpt \
  --alpha 0.1 \
  --delta 0.05 \
  --est_size 200 \
  --seed 42 \
  --num_lambdas 101 \
  --stopping_rule max_score \
  --loss_kind argmax_miscoverage \
  --confidence_threshold 0.5 \
  --entropy_threshold 0.5 \
  --use_testset \
  --output_dir results/cifar10_compare \
  --score_cache_base results/cifar10_compare
```

Writes `results/cifar10_compare/testset/compare_cifar10_summary_max_score.json` and the two-panel figure. The published selection was λ̂ = 0.95.

The 45,000/5,000 train/val split is drawn by an unseeded `random_split` when `classifier.data.cifar10` is imported. A new process does not recover the exact validation indices from the paper run.

## 2. FastMRI knee slice FNR (Figure 3)

Edit `configs/kspace/knee_slice_bayes_distill.yaml` before training. The fields that point at the original cluster are `datadir`, `split_csv_file`, and `loss_fn_weights_filename`. The dataset is single-coil FastMRI knee, three labels (`acl`, `mtear`, `abnormal`), and the mask is random variable-density sampling in k-space.

```bash
python bin/train_classifiers.py \
  --config_path configs/kspace/knee_slice_bayes_distill.yaml
```

The paper controls class-conditional FNR at α = 0.1 for ACL and meniscus only (`--exclude_labels abnormal`). Sampling rates are 5%, 8%, 12.5%, 15%, and 25%. δ = 0.1, estimation set size 101, seed 0, binomial p-values, and the default 15-point λ grid. Confidence threshold 0.8 and entropy threshold 0.2 are the heuristic baselines.

```bash
python bin/eval_compare_knee.py \
  --config_path configs/kspace/knee_slice_bayes_distill.yaml \
  --checkpoint_path /path/to/knee_slice_bayes_distill.ckpt \
  --alpha 0.1 \
  --alpha_per_label 0.1,0.1,0.1 \
  --delta 0.1 \
  --est_size 101 \
  --seed 0 \
  --exclude_labels abnormal \
  --risk_metric fnr \
  --ltt_pvalue_method binomial \
  --sampling_rates 0.05,0.08,0.125,0.15,0.25 \
  --num_lambdas 15 \
  --confidence_threshold 0.8 \
  --entropy_threshold 0.2 \
  --use_testset \
  --output_dir results/knee_slice_fnr \
  --score_cache_base results/knee_slice_fnr
```

Writes `results/knee_slice_fnr/testset/compare_knee_summary_knee_slice_fnr.json` and per-rate score caches under `score_cache/seed_0/`. The published selection was λ̂ = 0.5.

## 3. Prostate PI-RADS FNR (Figure 4)

The stopping experiment does not retrain the scanner model. It reads an analysis directory that contains `summary.json` plus validation and test ensemble CSVs. Each CSV needs `accession_id`, `averaging_tag`, `pirads_prob`, and `pirads_label`. `summary.json` needs `averaging_tags` and `outputs.val_ensemble_predictions` / `outputs.test_ensemble_predictions`.

Those files are produced by the T2 + DWI fusion classifier in `prostate/`. From `prostate/`, after pointing the YAML paths at the recon H5s, DICOM H5s, and label CSVs:

```bash
cd prostate
python -m scripts.grid_train --config configs/image/shared/image_t2_source.yaml
python -m scripts.grid_train --config configs/image/espirit/image_recon_universal_source_path15.yaml
python -m scripts.grid_train --config configs/image/espirit/image_t2_recon_dwi_universal_transfer_path15.yaml
python -m scripts.analyze_image_checkpoints \
  --experiment-config configs/image/espirit/image_t2_recon_dwi_universal_transfer_path15.yaml \
  --checkpoint /path/to/fusion.pt \
  --output-dir outputs/analysis/path15
cd ..
```

The paper figure uses asymmetric bands with γ_lower = 1 and γ_upper = 0.5, α = 0.1, δ = 0.1, a 51-point τ grid, and 200 estimation exams (seed 42). The remaining validation exams are the calibration set (745 in the published split of 945).

```bash
python bin/eval_compare_prostate_pirads_asym.py \
  --repo_root prostate \
  --analysis_dir prostate/outputs/analysis/path15 \
  --alpha 0.1 \
  --delta 0.1 \
  --est_size 200 \
  --seed 42 \
  --num_lambdas 51 \
  --band_gamma_lower 1.0 \
  --band_gamma_upper 0.5 \
  --confidence_threshold 0.8 \
  --entropy_threshold 0.2 \
  --plot_conditional \
  --output_dir results/prostate_pirads_compare_asym
```

Writes `results/prostate_pirads_compare_asym/fnr/compare_prostate_pirads_summary_prostate_pirads_fnr_asym.json`. The published selection was τ̂ = 0.58.

## 4. Risk level sweep (Figure 5, top)

Re-selects λ̂ on the same knee score cache and the same prostate scores for α in {0.05, 0.075, 0.1, 0.125, 0.15, 0.2}. Knee δ, the controlled labels, and the utility are read from the knee summary (δ = 0.1, ACL and meniscus). Prostate δ is read from `--prostate_summary`. The published prostate panel used δ = 0.05, which is not the δ stored by the Figure 4 command. Copy that summary and set `"delta": 0.05` before this sweep, or pass a summary that already has that value. Bands in this sweep are the symmetric ambiguity bands, including for prostate.

```bash
python bin/plot_alpha_stopping.py \
  --datasets knee_slice,prostate_pirads \
  --alpha_grid 0.05,0.075,0.1,0.125,0.15,0.2 \
  --num_lambdas 51 \
  --alpha_policy scale \
  --knee_summary results/knee_slice_fnr/testset/compare_knee_summary_knee_slice_fnr.json \
  --knee_score_cache results/knee_slice_fnr/testset/score_cache/seed_0 \
  --prostate_summary /path/to/prostate_summary_delta0.05.json \
  --prostate_analysis_dir prostate/outputs/analysis/path15 \
  --prostate_seed 13 \
  --prostate_est_size 100 \
  --prostate_score_cache results/alpha_stopping_mri/prostate_scores.npz \
  --out_dir results/alpha_stopping_mri
```

Writes `results/alpha_stopping_mri/alpha_stopping_mri_panel.png` and `.pdf`, plus `alpha_stopping_results.json`. `--from_json` redraws the figure from that JSON without recomputing.

## 5. JAFA wrapper on CIFAR-10 (Figure 5, bottom)

The policy is a frozen-encoder DQN (JAFA) that acquires 4×4 patches and can stop. The paper run is grid index 2: cost −0.025, 4 patches visible at the start, seed 123. Training downloads CIFAR-10 into `jafa/data/cifar10`.

```bash
cd jafa
python scripts/run_rl_cifar_grid.py \
  --config configs/rl/rl_cifar_grid.yaml \
  --index 2
python scripts/export_cifar_score_matrices.py \
  --run_dir rl/results/train_rl_cifar_grid_fixed_split_top8/rl_cifar_frozen_fixedsplit_top8_2_random_seed_123_r_cost_m0p025_initial_visible_count_4 \
  --checkpoint trained_best_acc \
  --output_dir rl/results/score_matrices/cifar_best
cd ..
```

`run_rl_cifar_grid.py` names the directory from the config. If the path above does not match the log line the trainer prints, pass the directory it created.

RCS then thresholds the Q-margin `Q(STOP) − max Q(reveal)` at α = 0.1, δ = 0.05. The trajectory is subsampled to 10 acquisition rates. Estimation size is 100, seed 42, and the λ grid has 81 points.

```bash
python bin/eval_compare_rl_jafa_probs.py \
  --score_dir jafa/rl/results/score_matrices/cifar_best \
  --wrapper_signal q_margin_max \
  --alpha 0.1 \
  --delta 0.05 \
  --est_size 100 \
  --seed 42 \
  --num_lambdas 81 \
  --num_sampling_rates 10 \
  --loss_kind argmax_miscoverage \
  --abstain_penalty 0.01 \
  --output_dir results/cifar_rl_jafa_compare
```

Writes `results/cifar_rl_jafa_compare/q-margin-max/compare_rl_jafa_summary_max_score_q_margin_max.json`. The published selection was λ̂ ≈ 0.2125.

## Redrawing panels

`bin/replot_compare_panels.py` rebuilds the knee and prostate comparison figures from a saved summary and, for knee, the score cache.

```bash
python bin/replot_compare_panels.py knee \
  --summary results/knee_slice_fnr/testset/compare_knee_summary_knee_slice_fnr.json \
  --score_cache_dir results/knee_slice_fnr/testset/score_cache/seed_0 \
  --out_dir results/knee_slice_fnr/testset

python bin/replot_compare_panels.py prostate \
  --summary results/prostate_pirads_compare_asym/fnr/compare_prostate_pirads_summary_prostate_pirads_fnr_asym.json \
  --out_dir results/prostate_pirads_compare_asym/fnr \
  --file_tag prostate_pirads_fnr_asym
```

## Layout

- `fst/` — Hoeffding and binomial p-values, Holm step-down, ambiguity bands.
- `classifier/` — CIFAR-10 and FastMRI knee datasets, masks, and the ARMS / Bayes-distill modules.
- `bin/` — training and the five experiment scripts.
- `configs/` — the CIFAR-10 and knee-slice configs used above.
- `jafa/` — CIFAR-10 JAFA trainer and the NPZ export. It has its own `classifier/` package; run it from `jafa/` so that import resolves there.
- `prostate/` — T2, recon-DWI, and T2+DWI fusion training that produces the PI-RADS score CSVs.

Hierarchical ImageNet, knee-volume runs, and the earlier conformal-triage scripts are not part of this paper and are not included.
