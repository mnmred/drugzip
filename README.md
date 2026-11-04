# DrugZip a multimodal task-agnostic drug representation - Code for DrugZip, CCSynergy, ChemCPA, and Sensitivity DNN

This repository contains the code accompanying the paper *DrugZip: a multimodal task-agnostic drug representation*.

DrugZip compresses the 25 bioactivity signatures of the Chemical Checker (chemistry, targets, networks, cells and clinics) into a single 128-dimensional vector per molecule, using a variational autoencoder trained on about 1.2 million molecules. The resulting representation does not depend on any particular downstream task. The repository contains the code to train DrugZip and an autoencoder baseline. It also contains the code used to evaluate the representation on three downstream tasks, comparing it with other drug representations such as one-hot encodings, Morgan fingerprints and the individual Chemical Checker signatures.

**Paper Abstract**: Small molecules are usually represented by their chemical structure, which overlooks their interactions in a broader biological context. The Chemical Checker (CC) addresses this by characterizing drugs across 25 bioactivity spaces, yielding a 3{,}200-dimensionsal profile per drug. We introduce DrugZip, a method for compressing these profiles into compact, task-agnostic embeddings, and compare it with variational and denoising autoencoders. DrugZip retains the stochastic encoder of a variational autoencoder but removes the KL regularization term. On 1.23 million compounds, we find that KL-regularized models suffer from latent space collapse, whereas both DrugZip and the denoising autoencoder (AE) learn non-collapsed representations. However, despite similar reconstruction quality, DrugZip preserves substantially more of the original bioactivity-space geometry, retaining 58\% of each drug's 10 nearest neighbors compared with 19\% for the denoising autoencoder. We evaluate the learned representations across drug synergy, sensitivity and transcriptional-response prediction. DrugZip achieves the highest mean AUC across three held-out-tissue synergy tiers, while other downstream evaluations show less consistent differences between representations. These findings demonstrate that avoiding collapse and achieving accurate reconstruction are not sufficient to assess drug embeddings and highlight the importance of evaluating their geometric properties alongside downstream predictive performance.


## Table of Contents
- [Overview](#overview)
- [Data](#data)
- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Quick Start: Test Example](#quick-start-test-example)
- [Usage](#usage)
  - [1. Generating DrugZip](#1-generating-drugzip)
  - [2. Reproducing CCSynergy](#2-reproducing-ccsynergy)
  - [3. Reproducing ChemCPA](#3-reproducing-chemcpa)
  - [4. Sensitivity DNN](#4-sensitivity-dnn)
- [Relation to the Master's Thesis](#relation-to-the-masters-thesis)
- [Citation](#citation)
- [License](#license)

## Overview

The repository covers four parts:
1. **DrugZip generation**: training the DrugZip model on the Chemical Checker signatures and exporting one vector per molecule.
2. **Drug synergy prediction (CCSynergy)**: the CCSynergy model, used to compare drug representations for predicting synergistic drug pairs.
3. **Perturbation response prediction (ChemCPA)**: fine-tuning ChemCPA on single-cell responses to drugs with different drug embeddings.
4. **Drug sensitivity prediction (Sensitivity DNN)**: a deep neural network that predicts drug response (`LN_IC50`) in cancer cell lines.

## Data

The signature data is available on the official Chemical Checker website: <https://chemicalchecker.com/downloads/signature3>.
You may download each signature, file by file, and then use `csv_signature_generator.py` to translate them into usable CSV files for the DrugZip compression.

The downstream models use further public datasets. Each folder's README says which files are included and where to get the rest:

| Folder | Data |
|---|---|
| `synergy/` | CCSynergy's Sanger drug-synergy data, cell-line representations and Chemical Checker signatures (included) |
| `chemcpa/` | ChemCPA's preprocessed sci-Plex 3 dataset (not included, ~1.4 GB) |
| `sensitivity/` | GDSC2 drug response (`LN_IC50`) and cell-line vectors (main tables included) |

Files larger than GitHub's 100 MB limit, such as the DrugZip output (`*-25in1-full-*.csv`, ~1.9 GB), the SMILES table and the container images, are not in the repository.

## Repository Structure

```
.
├── drugzip.py                   DrugZip: VAE that compresses the 25 Chemical Checker signatures of a molecule into one 128-d vector
├── ae.py                        plain autoencoder with the same architecture (AE baseline)
├── csv_signature_generator.py   converts Chemical Checker .h5 signature files into CSV
├── synergy/                     CCSynergy (drug-pair synergy prediction) + container definition
├── chemcpa/                     ChemCPA fine-tuning runs + container definitions
└── sensitivity/                 Sensitivity DNN (GDSC2 drug response) + container definition
```

## Environment Setup

To ensure strict reproducibility (particularly for the CCSynergy and DNN reproduction), please use the exact environment specifications provided below.

#### Option A: Apptainer containers (used for the experiments)

| Container | Definition | Used by |
|---|---|---|
| `synergy.sif` | `synergy/synergy.def` + `synergy/synergy.lock.txt` (an identical copy is in `sensitivity/`) | CCSynergy and the Sensitivity DNN: Python 3.10, CPU TensorFlow 2.15, RDKit, scikit-learn, shap, wandb |
| `updated_chemCPA.sif` | `chemcpa/chemCPA.def` + `chemcpa/environment.yml`, then `chemcpa/update_wandb.def` | ChemCPA: Python 3.9, PyTorch 1.12.1 with CUDA 11.3, Lightning, Hydra |

Build them from the folder that holds the definition file (each `.def` copies its environment file from the current directory):

```bash
cd synergy && apptainer build synergy.sif synergy.def && cd ..
cd chemcpa && apptainer build chemCPA.sif chemCPA.def && apptainer build updated_chemCPA.sif update_wandb.def && cd ..
```

Depending on your system, `apptainer build` may need `--fakeroot`. The ChemCPA base image is ~17 GB.

#### Option B: Conda

The setup relies on `conda` and `pip` to manage dependencies accurately. Open your terminal and execute the following:

##### 1. Create and Activate the Conda Environment

```bash
conda create -n ccsynergy -c conda-forge python=3.10 numpy=1.23.5 scipy=1.10.1 pandas=2.0.3 scikit-learn=1.3.0
conda activate ccsynergy
```

##### 2. Install the Remaining Packages

```bash
conda install -c conda-forge tensorflow=2.15 rdkit h5py matplotlib shap wandb
```

To reproduce the exact environment instead (Linux x86-64 only), create it from the lockfile:

```bash
conda create -n ccsynergy --file synergy/synergy.lock.txt
```

ChemCPA needs its own environment: `conda env create -f chemcpa/environment.yml`.

#### Weights & Biases

All training scripts log to [Weights & Biases](https://wandb.ai). No API key is stored in the repository. Run `wandb login` once or set `WANDB_API_KEY`. In `drugzip.py` and `ae.py`, also replace the `"Insert your W&B API key here"` placeholder or change the call to `wandb.login()`.

## Quick Start: Test Example

To check that DrugZip runs before using the full dataset, the repository includes a small example: the 25 Chemical Checker signatures of 200 molecules (`examples/drugzip/data/`). From the repository root:

```bash
bash examples/drugzip/run_example.sh
```

This trains DrugZip and the autoencoder baseline on those 200 molecules, in about two minutes on a CPU, and writes the vectors and models to `examples/drugzip/output/`. W&B logging is switched off. See [`examples/drugzip/README.md`](examples/drugzip/README.md) for the expected output and for running it inside the container.

The downstream folders (`synergy/`, `chemcpa/`, `sensitivity/`) include their data, or a link to it, in their own READMEs.

## Usage

The steps below describe the full workflow used in the paper.

### 1. Generating DrugZip

1. Download the 25 Chemical Checker signature files (A1–E5) and convert them to CSV:

   ```python
   from csv_signature_generator import convert_folder_h5_to_csv
   convert_folder_h5_to_csv("path/to/h5_files", "Data/Drug_Representation/ChemicalChecker_Full")
   ```

   Each CSV gets one row per molecule: the InChIKey, then the 128 signature values (`V1`–`V128`). `drugzip.py` expects the files to be named `A1.csv` … `E5.csv`.

2. Point the scripts at that folder and run DrugZip:

   ```bash
   export DRUGZIP_DATA_DIR=Data/Drug_Representation/ChemicalChecker_Full/   # keep the trailing /
   python drugzip.py
   ```

   It concatenates the 25 signatures into one 3,200-value input per molecule and trains a VAE: encoder 2688 → 2176 → 1664 → 1152 → 768 → 384 → 128-d latent, mirrored decoder, batch size 256, early stopping on a 20% validation split.
   - Without `DRUGZIP_DATA_DIR`, the scripts look in `path/to/your/data/` (`directory_path` in the script).
   - `kl_weights` sets the KL weight(s) to train with (β; `0.0` is the default).
   - The input matrix (1,230,665 molecules × 3,200 values in float64, ~31.5 GB) is cached as `input_matrix.npy` in that folder. Plan memory accordingly.

3. Outputs, for each KL weight:
   - `drugzip-25in1-full-<kl>-b256.csv`: the DrugZip vectors. It has a header row, then one 128-value row per molecule, in the same order as the input CSVs.
   - `drugzip-reconstructed-full-<kl>-b256.csv`: the decoder's reconstructions.
   - the encoder and VAE models (`.keras` and `.h5`).

   Everything is also logged to the W&B project `Paper-DrugZip`. The downstream folders assume this row order matches `chemcpa/all_drugs_inchikeys.csv`.

`ae.py` trains the deterministic autoencoder baseline on the same input (also read from `DRUGZIP_DATA_DIR`) and writes `ae-25in1-full-b256.csv`.

### 2. Reproducing CCSynergy

See [`synergy/README.md`](synergy/README.md). In short: pick the drug representation inside `CCSynergy()` in `synergy/synergy.py` (DrugZip vectors, the original Chemical Checker signatures, one-hot or Morgan fingerprints). Then run `synergy/script.py` (CV1) or `script_cv4.py` (CV4) from the `synergy/` folder, for example through `sbatch syn.sh`. The CV1 folds for Sanger are included; other folds are generated with `Cross_Validation.R`.

### 3. Reproducing ChemCPA

See [`chemcpa/README.md`](chemcpa/README.md). Three SLURM array scripts fine-tune ChemCPA on sci-Plex 3 with different drug embeddings:

| Script | Drug embedding |
|---|---|
| `run-own.sh` | Morgan fingerprints |
| `run-ohe.sh` | one-hot |
| `run-drugzip1.sh` | DrugZip vectors |

Before running, add the dataset and the missing `config/wandb/default.yaml`, and build the container (both described in that README).

### 4. Sensitivity DNN

See [`sensitivity/README.md`](sensitivity/README.md). The network predicts GDSC2 `LN_IC50` from a drug vector plus a cell-line vector:
- **Model 1:** Morgan fingerprints or one-hot drugs
- **Model 2:** DrugZip or Chemical Checker vectors

Four evaluation settings are available: random pairs (CV1), unseen cancer types (CV2), unseen cell lines (CV3) and unseen drugs (CV4). Submit with `sbatch train_model1.sh` or `MODEL_FILE=<vector file> sbatch train_model2.sh`.

## Relation to the Master's Thesis

This work is a follow-up to a master's thesis completed at TU Delft (MSc Data Science and Artificial Intelligence Technology), available in the TU Delft Education Repository:
[https://repository.tudelft.nl/record/uuid:10f9f08b-3059-4f3b-96ef-8b556a1b8404](https://repository.tudelft.nl/record/uuid:10f9f08b-3059-4f3b-96ef-8b556a1b8404)

## Citation

If you use this code, please cite the paper *DrugZip: a multimodal task-agnostic drug representation* (the full reference will be added on publication). The reproduced methods and data come from:

- **CCSynergy**: S.-R. Hosseini and X. Zhou, *CCSynergy: an integrative deep-learning framework enabling context-aware prediction of anti-cancer drug synergy*, Briefings in Bioinformatics, 2022. [doi:10.1093/bib/bbac588](https://doi.org/10.1093/bib/bbac588)
- **ChemCPA**: L. Hetzel et al., *Predicting Cellular Responses to Novel Drug Perturbations at a Single-Cell Resolution*, NeurIPS 2022.
- **Chemical Checker**: M. Duran-Frigola et al., *Extending the small-molecule similarity principle to all levels of biology with the Chemical Checker*, Nature Biotechnology, 2020.
- **GDSC**: Genomics of Drug Sensitivity in Cancer, <https://www.cancerrxgene.org>

## License

This repository does not have a licence of its own yet. Third-party parts keep their original licences:
- `chemcpa/`: the ChemCPA code is MIT-licensed (see `chemcpa/LICENSE`).
- `synergy/`: the code is derived from CCSynergy, and its `Data/` is CCSynergy's data. Both are under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
