# ChemCPA Reproduction (fine-tuning runs)

This folder contains the subset of my `chemCPA-reproduction` working copy that is needed to run three ChemCPA fine-tuning sweeps on a SLURM cluster. The ChemCPA code itself comes from [theislab/chemCPA](https://github.com/theislab/chemCPA) (MIT licence, see `LICENSE`); the run scripts, `misc.py` and `make_morgan_embedding.py` are my additions for plugging in different drug representations.

Large data files, the container image (its definition files are included) and one config file are **not** included. They are listed under [What is missing](#what-is-missing) below.

## The three runs

All three scripts submit a 10-task SLURM array (one task per hyperparameter combination, `--array=0-9`), train from scratch (`model.load_pretrained=false`) and log to the W&B project `chemCPA-Paper`. They differ only in the drug embedding that ChemCPA receives:

| Script | Drug representation | How the embedding parquet is built | Hydra config |
|---|---|---|---|
| `run-own.sh` | Morgan fingerprints (radius 2, 1024 bits) | `make_morgan_embedding.py`, from a SMILES CSV | `finetune_morgan` |
| `run-ohe.sh` | One-hot encoding (one column per drug) | inline Python in the script, from the drug index of the rdkit parquet | `finetune_morgan` (datapath overridden) |
| `run-drugzip1.sh` | DrugZip latent vectors (128-d) | `misc.py`, from a 1.2M-row latent CSV | `finetune_mana_vae` |

Each script builds its embedding parquet only if it does not exist yet, so the first array task creates it and the others reuse it.

## Folder layout

```
chemcpa/
├── run-own.sh, run-ohe.sh, run-drugzip1.sh     SLURM array scripts
├── generate_finetune_hparam_combinations.py   writes finetune_hparam_combinations.json
├── make_morgan_embedding.py                   Morgan fingerprint parquet (run-own.sh)
├── misc.py                                    latent-CSV parquet (run-drugzip1.sh) + helper functions
├── all_drugs_inchikeys.csv                    row order of the 1.2M-drug files (one InChIKey per row)
├── chemCPA.def, environment.yml               Apptainer definition of the base image
├── update_wandb.def                           builds updated_chemCPA.sif (the image the scripts use) from chemCPA.sif
├── chemCPA/                                   model, data loading and train_hydra.py entry point
├── config/                                    Hydra configs used by the two fine-tuning setups
└── embeddings/rdkit/chemCPA-rdkit_embeddings/
    └── sciplex_complete_middle_subset_lincs_genes_v2_rdkit2D_embedding.parquet
                                               drug index (canonical SMILES) that every embedding is aligned to
```

## What is missing

### 1. `config/wandb/default.yaml` (required)

Every top-level config loads `wandb: default`, so Hydra stops with a "could not find wandb/default" error until this file exists. It was excluded by a `wandb/` rule in the original `.gitignore`. Its keys are passed directly to Lightning's `WandbLogger`, so a minimal version looks like this:

```yaml
project: chemCPA-Paper
entity: <your-wandb-entity>
```

The run scripts override `wandb.name` and `wandb.project` on the command line.

### 2. The sci-Plex 3 dataset (`.h5ad`, ~1.4 GB)

`config/dataset/sciplex.yaml` expects:

```
<PROJECT_ROOT>/datasets/mana_sciplex_complete_middle_subset_lincs_genes_v2.h5ad
```

This is ChemCPA's preprocessed sci-Plex 3 subset (`sciplex_complete_middle_subset_lincs_genes_v2.h5ad`, obtainable through the download scripts in `raw_data/` of the upstream ChemCPA repository). `misc.remove_forbidden_drugs()` is the helper I used to filter drugs out of the original file. Whatever file you use must contain the fields the config refers to:

- `obs`: `condition`, `cov_drug_dose_name`, `dose_val`, `cell_type`, `SMILES`, `split_ood_finetuning`
- `uns`: `lincs_DEGs`

### 3. The Apptainer image (`$BULK/containers/updated_chemCPA.sif`)

All Python runs inside this image. The image itself is not included (several GB), but you can rebuild it from the two definition files in this folder:

| File | Builds |
|---|---|
| `chemCPA.def` (+ `environment.yml`) | `chemCPA.sif`: CUDA 11.3 / Ubuntu 20.04 base image with the `chemCPA` conda environment (Python 3.9, PyTorch 1.12.1, Lightning, Hydra, RDKit, scanpy, wandb, …) |
| `update_wandb.def` | `updated_chemCPA.sif`: starts from `./chemCPA.sif` and pins `wandb==0.22.3`, `numpy<2.0` and `pandas<2.0.0`. This is the image the run scripts use. |

From this folder:

```bash
apptainer build chemCPA.sif chemCPA.def
apptainer build updated_chemCPA.sif update_wandb.def
mkdir -p "$BULK/containers" && mv updated_chemCPA.sif "$BULK/containers/"
```

Depending on your cluster, `apptainer build` may need `--fakeroot`. The base image is large (~17 GB), so build it on a node with enough disk space.

### 4. Drug input files (per run)

| Run | File the script reads | Format |
|---|---|---|
| `run-own.sh` | `$BULK/Sensitivity/dataset/Drugs/SMILES/smiles_no_key_duplicates.csv` | no header, two columns `InChIKey,SMILES`, same row order as `all_drugs_inchikeys.csv` (checked by the script) |
| `run-drugzip1.sh` | `$BULK/CCSynergy/drugzip-25in1-full-1.0-b256.csv` | DrugZip latent vectors: one row per drug in the order of `all_drugs_inchikeys.csv`, `FEATURE_SIZE` (128) columns |
| `run-ohe.sh` | nothing extra | built from the included rdkit parquet |

### 5. Files created at run time (nothing to supply)

`finetune_hparam_combinations.json`, `embeddings/<tag>/*.parquet`, `logs/`, `training_output/` and `wandb/` are all generated by the scripts.

## Placeholders to fill in

Personal and cluster-specific values have been replaced with placeholders:

| File | Placeholder | What to put there |
|---|---|---|
| `run-*.sh` | `#SBATCH --account=<your-slurm-account>` | your SLURM account |
| `run-*.sh` | `#SBATCH --partition=<your-partition>` | partition(s) to submit to |
| `run-*.sh` | `#SBATCH --qos=<your-qos>` | QOS (must allow the 24 h time limit) |
| `run-*.sh` | `WANDB_API_KEY="${WANDB_API_KEY:-<your-wandb-api-key>}"` | leave as is and put `WANDB_API_KEY=...` in `<PROJECT_ROOT>/.env` (the scripts load it automatically), or replace the placeholder locally. Never commit a real key or the `.env` file. |
| `run-*.sh` | `module use /path/to/your/modulefiles` | your cluster's module path, or remove the line |
| `config/dataset/sciplex.yaml` | `/path/to/chemCPA-reproduction/...` | absolute path to the `.h5ad` dataset |
| `config/model/embedding/sciplex_*.yaml` | `/path/to/chemCPA-reproduction/...` | only used when calling `train_hydra.py` directly: the scripts override it with `model.embedding.datapath` |

The scripts also assume that the environment variable `$BULK` points to your storage root and that this folder lives at `$BULK/chemCPA-reproduction` (`PROJECT_ROOT`). Either copy it there under that name or change `PROJECT_ROOT` and `SCRIPT` at the top of each script.

## Running

From `PROJECT_ROOT`:

```bash
mkdir -p logs
sbatch run-own.sh
```

The same goes for `run-ohe.sh` and `run-drugzip1.sh`. Note that:

- `logs/` must exist **before** submitting, because SLURM opens `logs/finetune_%A_%a.out` before the script's own `mkdir -p logs` runs.
- `run-own.sh` currently has Windows (CRLF) line endings, which `sbatch` rejects. Convert it first with `dos2unix run-own.sh`.
- Results are written to `logs/<run name>/` and `training_output/<W&B run id>/`, and metrics go to W&B.
