# Example: DrugZip and the AE baseline on 200 molecules

A small, self-contained run of `drugzip.py` (DrugZip, the VAE) and `ae.py` (the plain autoencoder baseline), to check that the code works before running it on the full Chemical Checker set (1,230,665 molecules).

## Example data

`data/A1.csv` … `data/E5.csv` contain the first 200 molecules (sorted by InChIKey) of each of the 25 Chemical Checker signature files (type 3, downloaded from <https://chemicalchecker.com/downloads/signature3> and converted with `csv_signature_generator.py`). The rows are unchanged from the full files:
- a header (`"","V1",…,"V128"`)
- then one row per molecule: the InChIKey followed by its 128 signature values

All 25 files list the same molecules in the same order, which is what the scripts expect. Together they are 14 MB.

## Running it

From the repository root, with the environment from the main README:

```bash
bash examples/drugzip/run_example.sh
```

or inside the container built from `synergy/synergy.def`:

```bash
PYTHON="apptainer exec $PWD/synergy/synergy.sif python" bash examples/drugzip/run_example.sh
```

The script:
1. copies `data/` to `output/data/`, so the cached `input_matrix.npy` is not written into `data/`
2. points both scripts at that copy (`DRUGZIP_DATA_DIR`)
3. disables W&B logging (set `WANDB_MODE=online` to log)
4. runs `drugzip.py` and then `ae.py` from `output/`

Each script took about 1 minute on CPU in the container. Training stops early on the 20% validation split, so the run time can vary a little.

## Expected output

In `examples/drugzip/output/` (ignored by git):

| File | Content |
|---|---|
| `drugzip-25in1-full-0.0-b256.csv` | the DrugZip vectors: a header row, then 200 rows × 128 values, in the same molecule order as `data/` |
| `drugzip-reconstructed-full-0.0-b256.csv` | the VAE's reconstruction of the 3,200 input values per molecule |
| `encoder-full-0.0-b256.{keras,h5}`, `vae-full-0.0-b256.{keras,h5}` | the trained encoder and VAE |
| `ae-25in1-full-b256.csv` | the AE baseline's 128-value vectors for the same 200 molecules |
| `ae_encoder-do.keras`, `ae_full.keras` | the trained AE encoder and autoencoder |

The model files are large (about 0.5 GB for each full model), so expect roughly 1.8 GB in `output/`. With only 200 molecules the vectors are not meaningful; the run checks the pipeline, not the results.

## Running on the full data

1. Download the 25 signature files from the Chemical Checker and convert them with `csv_signature_generator.py` (see the main README). Name them `A1.csv` … `E5.csv`.
2. Point the scripts at that folder: `export DRUGZIP_DATA_DIR=/path/to/signatures/` (with a trailing `/`), or edit `directory_path` in the script.
3. Run `python drugzip.py` and/or `python ae.py`.

The full input matrix is 1,230,665 × 3,200 values in float64 (about 31.5 GB). It is built once and cached as `input_matrix.npy` in the data folder, so plan memory and disk space accordingly.
