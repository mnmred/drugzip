"""Build a ChemCPA drug-embedding parquet from Morgan fingerprints."""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from misc import SCIPLEX_ROW_INDICES

DEFAULT_PARQUET_IN = "embeddings/rdkit/chemCPA-rdkit_embeddings/sciplex_complete_middle_subset_lincs_genes_v2_rdkit2D_embedding.parquet"


def smiles_to_morgan(smiles_list, nbits=1024, radius=2):
    """Row k = Morgan bit vector of smiles_list[k]; zero vector for missing/invalid SMILES."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fps = np.zeros((len(smiles_list), nbits), dtype=np.float32)
    for k, smi in enumerate(smiles_list):
        mol = None if smi is None or pd.isna(smi) else Chem.MolFromSmiles(smi)
        if mol is None:
            print(f"Warning: invalid/missing SMILES at row {k}, using zero vector")
            continue
        # GetOnBits works on every RDKit version (the container has 2022.09)
        fps[k, list(gen.GetFingerprint(mol).GetOnBits())] = 1.0
    print("zero fingerprints:", int((fps.sum(axis=1) == 0).sum()), "of", len(fps))
    return fps


def select_smiles(smiles_csv, indices, inchikeys_csv="all_drugs_inchikeys.csv"):
    """Pick SMILES from the 1.2M-row (key, smiles) file by row position; None for -1."""
    df = pd.read_csv(smiles_csv, header=None, names=["key", "smiles"])
    all_keys = pd.read_csv(inchikeys_csv, header=None).iloc[:, 0]

    # Safety checks: the file must share the row order the indices were built on
    if len(df) != len(all_keys):
        raise ValueError(f"{smiles_csv} has {len(df)} rows but {inchikeys_csv} has {len(all_keys)}.")
    valid = [i for i in indices if i != -1]
    mismatched = (df["key"].iloc[valid].values != all_keys.iloc[valid].values).sum()
    if mismatched:
        raise ValueError(f"{mismatched} selected rows have a different InChIKey than {inchikeys_csv}.")

    return [None if i == -1 else df["smiles"].iloc[i] for i in indices]


def build_morgan_parquet(parquet_in, parquet_out, smiles_csv=None, nbits=1024, radius=2):
    index = pd.read_parquet(parquet_in).index  # canonical SMILES that chemCPA matches on

    if smiles_csv is not None:
        if len(SCIPLEX_ROW_INDICES) != len(index):
            raise ValueError(f"{len(SCIPLEX_ROW_INDICES)} indices but {parquet_in} has {len(index)} rows.")
        smiles = select_smiles(smiles_csv, SCIPLEX_ROW_INDICES)
    else:
        smiles = list(index)  # fingerprint the chemCPA SMILES directly (no -1 gaps)

    fps = smiles_to_morgan(smiles, nbits=nbits, radius=radius)
    df = pd.DataFrame(fps, index=index, columns=[f"latent_{i+1}" for i in range(nbits)])

    os.makedirs(os.path.dirname(parquet_out), exist_ok=True)
    df.to_parquet(parquet_out)
    print(f"Saved {df.shape} to {parquet_out}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the ChemCPA drug-embedding parquet from Morgan fingerprints")
    parser.add_argument("--smiles-csv", default=None,
                        help="1.2M-row (key, smiles) file; omit to fingerprint the parquet's own SMILES index")
    parser.add_argument("--nbits", type=int, default=1024)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--parquet-out", required=True)
    parser.add_argument("--parquet-in", default=DEFAULT_PARQUET_IN)
    args = parser.parse_args()

    build_morgan_parquet(args.parquet_in, args.parquet_out, args.smiles_csv, args.nbits, args.radius)