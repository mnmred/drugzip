# CCSynergy (drug synergy prediction)

`synergy.py` is my modified version of **CCSynergy** (Hosseini & Zhou, *Briefings in Bioinformatics*, 2022, [doi:10.1093/bib/bbac588](https://doi.org/10.1093/bib/bbac588), licensed [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)). It predicts whether a drug pair is synergistic in a given cell line. Each sample is `drug A vector + drug B vector + cell-line vector`, and every pair is used in both orientations (A-B and B-A). The model is a dense network with layers 2000 → 1000 → 500.

Compared with the original, this version adds:
- alternative drug representations: concatenated latent vectors (e.g. DrugZip), one-hot and Morgan fingerprints
- fixed seeds for reproducibility
- Weights & Biases logging
- a SHAP check of how much the model relies on the drugs versus the cell line
- one cross-validation folder per scheme and fold

The data in `Data/` comes from the original CCSynergy repository (same licence).

## Folder layout

```
synergy/
├── synergy.py            model, feature functions and the CCSynergy() training/evaluation routine
├── script.py             runner for CV1 (5 folds × the cell representations listed in the script)
├── script_cv4.py         runner for CV4
├── syn.sh, syn_cv4.sh    SLURM jobs that run script.py / script_cv4.py
├── Cross_Validation.R    generates the train/validation/test index files (CV1–CV4)
├── synergy.def           Apptainer definition of the environment (built from synergy.lock.txt)
├── synergy.lock.txt      exact conda package list (Python 3.10, CPU TensorFlow 2.15, RDKit, shap, wandb)
└── Data/
    ├── Synergy_Data/         Sanger.csv (binary labels), Merck.csv (continuous scores)
    ├── Drug_Representation/  Drugs_Single_Info_<dataset>.csv + 25 Chemical Checker signatures (A1–E5) per dataset
    ├── Cell_Representation/  Cell1–Cell5.csv: the five cell-line representations (100 values per cell line)
    └── Cross_Validation_1_F1 … _F5/   CV1 index files for Sanger (seed 28)
```

## Choosing the drug features

Inside `CCSynergy()` in `synergy.py`, exactly one feature function is uncommented:

| Function | Drug representation | Data it needs |
|---|---|---|
| `Prepare_Features_Concatenated` (current) | rows of a latent-vector CSV (e.g. DrugZip), picked at the positions of the 62 Sanger drugs (`drugs_location`) | `25in1-full-b256.csv` in the working directory (**not included**, see below). The file name is set in `drug_features_filename`. |
| `Prepare_Features_Baseline` | one Chemical Checker signature (original CCSynergy) | included. Pass `FeatureVecorName=['A1', …, 'E5']`, and `DRUGindex` picks the signature. |
| `Prepare_OHE_Features` | one-hot drug encoding | included |
| `Prepare_Features_SMILES` | Morgan fingerprints (radius 2, 1024 bits) | `$BULK/Sensitivity/dataset/Drugs/SMILES/smiles_no_key_duplicates.csv` (**not included**, the same file the Sensitivity DNN uses) |

`Prepare_Features_Concatenated` and `Prepare_Features_SMILES` use drug positions that exist only for the Sanger drugs, so use them with `DATAset="Sanger"`.

In the runner scripts, `runs_names` is a **label only**. It names the W&B group and the output files, but the features come from the function selected above. Change both together.

## Cross-validation schemes

`CVtype` selects the folder `Data/Cross_Validation_<CVtype>_F<fold>/`, which must contain `Training_index.csv`, `Validation_index.csv` and `Testing_index.csv`. Each file holds a single `Index` column of 0-based rows of the feature matrix, already covering both orientations (row *i* and *i + N*).

| CVtype | What is held out | Index folders | Status |
|---|---|---|---|
| 1 | random drug pairs (pairs are split, so a pair is never in train and test) | `Cross_Validation_1_F1…F5` | **included** for Sanger (seed 28) |
| 2 | a tissue (`Tindex` 1 = pancreas, 2 = breast, 3 = digestive system) within a pair fold | `Cross_Validation_2_F<f>_T<t>` | **not supported by `synergy.py` yet**: its index path has no tissue part |
| 3 | drugs: test pairs contain at least one held-out drug | `Cross_Validation_3_F<f>` | generate with R |
| 4 | drugs, strictly: both drugs held out (mixed pairs are dropped) | `Cross_Validation_4_F<f>` | generate with R |

To generate index files, run R with the `caret` package from this folder (the script reads `Data/Synergy_Data/` relative to the working directory):

```r
source("Cross_Validation.R")
for (f in 1:5) CV4_Splitter("Sanger", 28, f)
```

Each splitter writes to the folder `synergy.py` reads. Use seed 28 to reproduce the included CV1 split.

## What is missing

| File | Needed for |
|---|---|
| `25in1-full-b256.csv` (~1.9 GB) or another latent-vector file | `Prepare_Features_Concatenated`. The file has a header row, then one row per drug in the order of `all_drugs_inchikeys.csv` (~1.2M rows) and 128 columns. Other DrugZip/AE/PCA outputs in the same format work too once `drug_features_filename` is changed. Too large for GitHub. |
| `smiles_no_key_duplicates.csv` | `Prepare_Features_SMILES` (see `../sensitivity/README.md`) |
| CV3 / CV4 index folders | `CVtype=3` / `CVtype=4` (generate as above) |
| Merck index folders | running on `Merck.csv` (the included folds are for Sanger) |

## Placeholders to fill in

| File | Placeholder | What to put there |
|---|---|---|
| `syn.sh`, `syn_cv4.sh` | `#SBATCH --account=<your-slurm-account>`, `--partition=<your-partition>`, `--qos=<your-qos>` | your SLURM settings (QOS must allow 36 h) |
| `syn.sh`, `syn_cv4.sh` | `# source activate /path/to/your/ccsynergy_env` | your environment (the line is commented out) |
| `syn.sh`, `syn_cv4.sh` | `WANDB_DIR="$BULK/wandb_logs"` | expects `$BULK` to point to your storage root |
| `script.py`, `script_cv4.py` | `wandb.login()` | no key is stored: set `WANDB_API_KEY` in the job environment or run `wandb login` once |

## Running

The runs used the Apptainer container defined in `synergy.def`, which recreates the exact environment from `synergy.lock.txt`. Build it once from this folder (`synergy.def` copies the lockfile from the current directory):

```bash
apptainer build synergy.sif synergy.def      # may need --fakeroot on your cluster
apptainer run synergy.sif script.py          # runs python inside the container
```

To use it from SLURM, replace `srun python script.py` in `syn.sh` with `srun apptainer run synergy.sif script.py`. TensorFlow in the image is the CPU build. Without the container, set up the conda environment described in the main README.

The runners use `DIR=""`, so run them from this folder:

```bash
sbatch syn.sh        # → script.py , Set up the cross validation folders first through Cross_Validation.R
```

Edit the settings at the top of the runner before submitting: `DATAset`, `TASK`, `runs_names` and the `cells` list (the cell-line representations to loop over). Each run trains on 5 folds and writes these files to the working directory:
- `Results_<dataset>_CCSynergy<cell>_<run name>_Fold<k>_CV<cv>.keras`: the best model
- `Results_<…>.csv`: real vs. predicted values on the test fold

Test AUC, PR AUC, accuracy and F1 are logged to the W&B project `CCSynergy-Paper`.

For running MF, it has 1024 sized inputs so the SHAP code needs to be adapted, uncomment lines 383-387.
