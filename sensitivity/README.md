# Sensitivity DNN

This folder trains a feed-forward network that predicts drug sensitivity (`LN_IC50` from GDSC2) for cell-line/drug pairs. Each sample is a drug vector concatenated with a cell-line vector (`Cell1.csv`):

- **Model 1** (`Models/SMILES/`): drug = Morgan fingerprint (radius 2, 1024 bits) computed from SMILES with RDKit, or a one-hot drug encoding.
- **Model 2** (`Models/DrugZIP/`): drug = a precomputed vector (a DrugZip latent vector or one of the Chemical Checker signatures), selected with `--model_file`.

All runs log to Weights & Biases and write per-fold results to `results/`.

Large data files are **not** included. They are listed under [What is missing](#what-is-missing).

## Network

Both models use the same DNN (the `en_` scripts are Elastic Net baselines instead):

- Input: drug vector concatenated with the cell vector (no scaling)
- Hidden layers: [1024, 528, 256], each Dense + ReLU → BatchNormalization → Dropout(0.3)
- Output: one linear unit (`LN_IC50`)
- Training: Adam (learning rate 1e-3), MSE loss, batch size 256, up to 500 epochs with early stopping on validation loss (patience 15)
- Defaults that can be changed on the command line: `--n_splits` (10), `--epochs` (500), `--batch_size` (256), `--nbits` (1024, Model 1), `--project`/`--entity` (W&B), `--model_file` (Model 2)

## Evaluation settings

The `_cvN` suffix selects the split strategy. CV2–CV4 first hold out 20% of the groups as a test set (`GroupShuffleSplit`), then run 10-fold `GroupKFold` on the rest, so no group appears in two folds or in both train and test.

| Suffix | Split | Groups |
|---|---|---|
| (none), "CV1" | 10-fold `KFold`, shuffled | none: random pairs |
| `_cv2` | held-out test set + `GroupKFold` | `CANCER_TYPE` (unseen cancer types) |
| `_cv3` | held-out test set + `GroupKFold` | cell line (unseen cell lines) |
| `_cv4` | held-out test set + `GroupKFold` | `DRUG_ID` (unseen drugs) |

## Scripts

| Path | What it does |
|---|---|
| `Models/SMILES/model_training[_cv2/3/4].py` | Model 1. `model_training.py`, `_cv3` and `_cv4` currently use Morgan fingerprints, while `_cv2` is currently switched to one-hot drugs (`prepare_model1_ohe_features`). |
| `Models/DrugZIP/model2_training[_cv2/3/4].py` | Model 2 on any drug-vector file, selected with `--model_file`. These are the versions `train_model2.sh` runs. |
| `Models/DrugZIP/v1_*, v2_*, v3_*` | Earlier Model 2 iterations that read `dataset/Drugs/25in1-full-b256.csv` directly instead of using `--model_file`. |
| `Models/*/en_*.py` | Elastic Net baselines (`--alpha`, `--l1_ratio`) for Model 1 and Model 2. |
| `train_model1.sh` | SLURM job for Model 1 (currently runs `model_training_cv2.py`; the other variants are commented out). |
| `train_model2.sh` | SLURM job for Model 2: runs CV1–CV4 in sequence for `MODEL_FILE`. |
| `sync.sh` | copies the scripts to the cluster with `scp` (placeholders for host and paths). |
| `dataset_maker.py` | helper functions used to build `dataset/GDSC2_cnd_overlap_with_positions.csv` from the raw GDSC2 release (see below). |
| `div.py` | splits `all_drugs_inchikeys.csv` into three parts (used while preparing the SMILES files). |
| `verify_setup.py`, `model1_config.yaml` | early Model 1 setup check and config, kept for reference. They describe the first version (5 folds, smaller network, `smiles.csv`). The training scripts take their settings from command-line arguments, not from the YAML. |
| `requirements.txt` | Python dependencies (TensorFlow < 2.16, RDKit, scikit-learn, wandb, …). Note: `shap` is also needed by the non-`en_` training scripts but is not listed, so install it as well. |

The `.py` files were run from a single flat folder on the cluster (`sync.sh` copies them next to `train_model*.sh`). All scripts open `dataset/...` and `results/...` **relative to the current directory**, so run them from this folder, either after copying the chosen script here or by calling it as `python3 Models/SMILES/model_training_cv4.py`.

## Included data

| File | Used for |
|---|---|
| `dataset/GDSC2_cnd_overlap_with_positions.csv` | main table read by every training script: cell line, drug, cancer type, `LN_IC50` and `DRUG_POSITION` (row index of the drug in `all_drugs_inchikeys.csv`) |
| `dataset/Cells/Cell1.csv` | cell-line vectors (one column per cell line) |
| `dataset/Drugs/InChiKeys/syn-keys.csv` | hand-made drug name → InChIKey mapping used by `dataset_maker.drug_position_mapping()` |
| `dataset/Drugs/unique_drugs_synonyms.csv` | drug-name synonyms (reference only, not read by any script) |
| `dataset/screened_compounds_rel_8.5.csv` | GDSC compound list, release 8.5 (reference only) |
| `drugs_ordered.csv`, `drugs-ordered.txt` | ordered drug lists (reference only) |
| `results/results_Model1_SMILES_Cell_cv_results.csv` | an example results file from an early Model 1 run |

## What is missing

### Needed to run the training

| File | Needed by | Notes |
|---|---|---|
| `dataset/Drugs/SMILES/smiles_no_key_duplicates.csv` (~116 MB) | Model 1 with fingerprints | no header, `InChIKey,SMILES` per line, one line per drug in the row order of `all_drugs_inchikeys.csv` (duplicate InChIKeys removed with `dataset_maker.remove_key_duplicates_smiles()`). Too large for GitHub. |
| `/path/to/CCSynergy/<model_file>.csv` | `model2_training*.py` | the drug-vector file chosen with `--model_file` (e.g. `drugzip-25in1-full-1.0-b256`, `Sanger_Full/A1`). One line per drug in the row order of `all_drugs_inchikeys.csv`. The first comma-separated field on each line is dropped, so the vector starts at the second field. Generated by the DrugZip/CCSynergy part of this repository. |
| `dataset/Drugs/25in1-full-b256.csv` (~1.9 GB) | `v1_*`, `v2_*`, `v3_*`, `en_model2_training.py` | DrugZip latent vectors for all drugs. |
| `/path/to/CCSynergy/Sanger_Full/A1.csv` | `en_model2_training.py` | Chemical Checker A1 signature in the same format. |

One-hot runs (the current `model_training_cv2.py`) need only the two included files.

### Only needed to rebuild the dataset

These files are inputs or intermediate outputs of `dataset_maker.py` and `div.py`:

- `dataset/GDSC2_fitted_dose_response_27Oct23.xlsx`: the raw GDSC2 dose-response release, downloadable from the [GDSC website](https://www.cancerrxgene.org/downloads/bulk_download)
- `dataset/GDSC2.csv`, `GDSC2_all.csv`, `GDSC2_cell1_overlap.csv`, `GDSC2_cnd_overlap.csv`: intermediate tables
- `dataset/Drugs/InChiKeys/all_drugs_inchikeys.csv`: the 1.2M-drug InChIKey list that defines `DRUG_POSITION`. An identical copy is in `../chemcpa/all_drugs_inchikeys.csv`.
- `dataset/Drugs/InChiKeys/all.csv`, `all_part1-3.csv`, `dataset/Drugs/names_*.csv`, `dataset/Drugs/SMILES/smiles*.csv/.txt`: drug name and SMILES lookups used to build the SMILES file

The dataset was built roughly in this order:

1. `xlsx_to_csv` and `csv_curation` (keep GDSC2 rows and the needed columns)
2. `dataset_cell_check` (keep cell lines present in `Cell1.csv`)
3. `dataset_drug_check` (keep drugs with a known name)
4. `drug_position_mapping` (add `DRUG_POSITION` through `syn-keys.csv` and `all_drugs_inchikeys.csv`)
5. `remove_unmapped_entries` (drop drugs whose position is `-1`)

## Placeholders to fill in

| File | Placeholder | What to put there |
|---|---|---|
| `train_model*.sh` | `#SBATCH --account=<your-slurm-account>` | your SLURM account |
| `train_model*.sh` | `#SBATCH --partition=<your-partition>` | partition(s) to submit to |
| `train_model*.sh` | `#SBATCH --qos=<your-qos>` | QOS that allows the 10 h time limit |
| `train_model1.sh` | `WANDB_DIR="$BULK/wandb_logs"` | expects `$BULK` to point to your storage root; change it if not |
| `train_model2.sh` | `WANDB_DIR="/path/to/your/wandb_logs"` | where W&B stores its local files |
| `Models/**/*.py` | `wandb.login()` | the scripts no longer contain an API key. Set `WANDB_API_KEY` in the job environment or run `wandb login` once on the cluster. |
| `Models/DrugZIP/*.py` | `/path/to/CCSynergy/` | folder that holds the drug-vector CSVs |
| `sync.sh` | `/path/to/local/Sensitivity/`, `your-user@your-cluster-login-host`, `/path/to/cluster/Sensitivity` | your local copy, cluster login and target folder |

## Running

On the cluster, from the folder that contains `dataset/` and the training scripts:

```bash
pip install -r requirements.txt shap     # or an equivalent conda environment
mkdir -p results
sbatch train_model1.sh
MODEL_FILE=drugzip-25in1-full-1.0-b256 sbatch train_model2.sh
```

`sbatch` passes the submitting shell's environment to the job, so `MODEL_FILE` reaches `train_model2.sh` (default: `drugzip-25in1-full-1.0-b256`). The scripts save `results/<model_name>_cvN_results.csv` and `results/<model_name>_best_model_fold<k>.h5`, and log metrics and SHAP feature-importance summaries (drug vs. cell share) to W&B.
