import pandas as pd
import anndata
import os

SCIPLEX_ROW_INDICES = [-1, 73768, 434518, 479115, 276925, 1183005, 1087973, 1085936, 218704, 70726, 970934, 1123906, 254287, 
               507047, 197492, 289164, 866945, 968070, 809899, 1152730, 828279, 529849, 645301, 1209478, 397841, 321903, 
               410618, 429319, 547639, 1079681, 849680, 43140, 895584, 930537, 267537, 37603, 914366, 941157, 879351, 
               934575, 611998, 255833, 1069813, 1008042, 233453, 717587, 205012, 46405, 988314, 987584, 163583, 1224528, 
               19074, 540645, 157608, 535984, 518389, 764001, 27712, 594730, 1021165, 182412, 978754, 726507, 871291, 
               481893, 1046972, 547646, 508781, 1221445, 419959, 313648, 223286, 130677, 334738, 339621, 431488, 1155236, 
               508739, 604891, 104230, -1, 999574, 258357, 1176891, 199881, 807279, 421133, 440916, 315251, -1, 663232, 
               -1, 922080, 929262, 282668, 9170, 63496, 11293, 772564, 1215804, 102794, 519218, 104825, 614924, 229161, 
               320080, 602212, 732897, 357423, 589194, 447518, 912882, 526811, 672435, 1158722, 513439, 132278, 109956, 
               41213, 463896, 313622, -1, 1162815, 646743, 685397, 971740, 21132, 633967, 839258, 286823, 820208, 428545, 
               338861, 207381, 1106243, 289092, 565342, 910691, 560005, 32132, 48635, 318497, 524698, 211246, 134002, -1, 
               397883, 613911, 1171991, 966773, 131916, 1220865, 382443, 1201817, 798269, 244059, 315250, 728756, 520506, 
               179507, 973698, 1098341, 335369, 798338, 1036908, 850250, 861063, 494590, 1119272, 1106306, 396645, -1, 
               348676, 289866, 573791, 6403, -1, 435094, 398665, 94193, 230775, 823659, 88616, 284494, 903022, 189396, 
               1043627]

def parquet_read(parquet_file):
    # Read the parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)
    #show dataframe 
    print(df.head())
    print(f"DataFrame shape: {df.shape}")

    # csv = df.to_csv(csv_file, index=False)

# parquet_read(
#     "embeddings/rdkit/chemCPA-rdkit_embeddings/sciplex_complete_middle_subset_lincs_genes_v2_rdkit2D_embedding.parquet"
# )

def find_smiles_indices(parquet_file, smiles_mapping):
    df = pd.read_parquet(parquet_file)
    # the rows indices are the smiles strings
    smiles_strings = df.index.tolist()
    indices = []
    not_found = 0

    # for smiles in smiles_strings:
    #     if smiles in smiles_mapping:
    #         indices.append(smiles_mapping[smiles])
    #     else:
    #         print(f"SMILES string not found in mapping: {smiles}")
    #         not_found += 1

    smiles_mapping = pd.read_csv(smiles_mapping)
    smiles_mapping_only = smiles_mapping['smiles']
    
    for smiles in smiles_strings:
        # check in the smiles column of the mapping file
        if smiles in smiles_mapping_only.values:
            index = smiles_mapping_only[smiles_mapping_only == smiles].index[0]
            indices.append(index)
        else:
            new_smiles = smiles.replace("/", "")
            # the \ sign too can be removed
            new_smiles = new_smiles.replace("\\", "")

            print(f"SMILES string not found in mapping: {smiles}")
            if new_smiles in smiles_mapping_only.values:
                index = smiles_mapping_only[smiles_mapping_only == new_smiles].index[0]
                indices.append(index)
            else:
                print(f"Also not found after removing slashes: {new_smiles}")
                new_smiles = new_smiles.replace("[O]", "O")
                if new_smiles in smiles_mapping_only.values:
                    index = smiles_mapping_only[smiles_mapping_only == new_smiles].index[0]
                    indices.append(index)
                else:
                    print(f"Also not found after replacing [O] with O: {new_smiles}")
                    not_found += 1
    

    print(f"Total SMILES strings not found: {not_found}")
    # print(indices)
    #but i don't want it to print np.int64(40) but just 40, so convert to int
    indices = [int(i) for i in indices]
    print(indices)

    #check the numbers missing 0-188 included in the indices list
    missing_indices = [i for i in range(189) if i not in indices]
    print(f"Missing indices: {missing_indices}")
    
    return indices

# find_smiles_indices(
#     "embeddings/rdkit/chemCPA-rdkit_embeddings/sciplex_complete_middle_subset_lincs_genes_v2_rdkit2D_embedding.parquet",
#     "sciplex3_drug_smiles_mapping.csv"
# )   

# mapping = pd.read_csv("sciplex3_drug_smiles_mapping.csv")
# print(mapping.head())


def find_inchi_key():

    indices = [46, 28, 40, 117, 120, 152, 97, 20, 180, 134, 98, 167, 84, 139, 125, 
               90, 67, 177, 147, 182, 49, 106, 103, 78, 1, 50, 53, 110, 101, 107, 
               113, 82, 72, 69, 66, 145, 92, 119, 17, 9, 176, 132, 160, 6, 34, 13, 
               10, 0, 70, 29, 35, 118, 185, 108, 178, 57, 141, 181, 174, 4, 146, 96, 
               89, 165, 153, 144, 154, 169, 171, 61, 112, 14, 179, 138, 114, 170, 32, 
               52, 75, 71, 51, 173, 143, 31, 95, 142, 80, 60, 133, 55, 30, 188, 22, 151, 
               25, 3, 100, 175, 85, 136, 87, 24, 68, 23, 166, 26, 126, 150, 73, 2, 8, 122, 
               184, 163, 105, 63, 99, 48, 44, 157, 156, 186, 19, 64, 123, 115, 94, 27, 62, 
               124, 127, 121, 128, 111, 129, 155, 86, 183, 45, 81, 116, 172, 5, 41, 16, 109, 
               91, 137, 65, 164, 93, 18, 76, 131, 74, 158, 83, 43, 59, 54, 11, 58, 140, 88, 
               159, 47, 130, 7, 33, 38, 39, 36, 21, 12, 37, 168, 162, 149, 148, 77, 79, 187, 
               15, 42, 56, 161, 102, 135]
    
    mapping = pd.read_csv("sciplex3_drug_smiles_mapping.csv")
    # inchi_keys = mapping['inchi_key'].iloc[indices].tolist()
    #but this won't work as it doesn't have that column
    
    # retrieve the mapping rows in the indices order
    mapping_rows = mapping.iloc[indices]
    print(mapping_rows)
    mapping_names = mapping_rows['drug_name']
    print(mapping_names)

    # put in a csv file the drug names and the corresponding indices
    output_df = pd.DataFrame({'drug_name': mapping_names})
    output_df.to_csv("drug_names_in_order.csv", index=False)

def retrieve_key_position():
    # in drugs_ordered, the relevant keys are found
    # in all_drugs_inchikeys, all inchikeys are found
    # we want to find the position of the relevant keys in the all inchikeys list
    drugs_ordered = pd.read_csv("drugs_ordered.csv", header=None)
    all_drugs_inchikeys = pd.read_csv("all_drugs_inchikeys.csv", header=None)
    # the keys are in the second column of the all_drugs_inchikeys file, unnamed column
    drug_keys = drugs_ordered.iloc[:, 1].tolist()    
    all_inchikeys = all_drugs_inchikeys.iloc[:, 0].tolist()   
    positions = []
    for drug in drug_keys:
        if drug in all_inchikeys:
            position = all_inchikeys.index(drug)
            positions.append(position)
        else:
            print(f"Drug name not found in all inchikeys: {drug}")
            positions.append(-1)  # Append -1 for not found drugs
    print(positions)
    print(f"Total drugs not found: {positions.count(-1)}")
    print(f"Total drugs found: {len(positions) - positions.count(-1)}")


def make_csv_25in1(latent_csv, out_csv, feature_size):
    indices = SCIPLEX_ROW_INDICES
    # retrieve from 25in1-full-b256.csv the correct vectors
    # full_df = pd.read_csv("25in1-full-b256.csv", header=None)
    # selected_vectors = []
    # for index in indices:
    #     if index != -1:
    #         vector = full_df.iloc[index].tolist()
    #         selected_vectors.append(vector)
    #     else:
    #         print("Index -1 found, adding dummy vector.")
    #         selected_vectors.append([0] * 128)  # Append a dummy vector with 128 zeros
    # # save the selected vectors in a new csv file
    # output_df = pd.DataFrame(selected_vectors)  
    # output_df.to_csv("selected_vae_vectors.csv", index=False, header=False)
    with open(latent_csv) as f:
        first_line = f.readline().strip().split(",")
    has_header = all(c.strip().isdigit() for c in first_line)
    full_df = pd.read_csv(latent_csv, header=0 if has_header else None)

    # Safety checks: one row per InChIKey and the expected feature size
    with open("all_drugs_inchikeys.csv") as f:
        n_keys = sum(1 for _ in f)
    if len(full_df) != n_keys:
        raise ValueError(f"{latent_csv} has {len(full_df)} rows but all_drugs_inchikeys.csv has {n_keys}.")
    if full_df.shape[1] != feature_size:
        raise ValueError(f"{latent_csv} has {full_df.shape[1]} features, expected {feature_size}.")

    selected_vectors = []
    for index in indices:
        if index != -1:
            vector = full_df.iloc[index].tolist()
            selected_vectors.append(vector)
        else:
            print("Index -1 found, adding dummy vector.")
            selected_vectors.append([0] * feature_size)
    # save the selected vectors in a new csv file
    output_df = pd.DataFrame(selected_vectors)
    output_df.to_csv(out_csv, index=False, header=False)

def h5ad_to_csv(h5ad_file, csv_file):
    import anndata
    # Load the h5ad file
    adata = anndata.read_h5ad(h5ad_file)    # Convert the data to a DataFrame
    if hasattr(adata.X, 'toarray'):
        data = adata.X.toarray()
    else:
        data = adata.X
    # df = pd.DataFrame(data, index=adata.obs_names, columns=adata.var_names)
    # Save the DataFrame to a CSV file
    # df.to_csv(csv_file)
    print(adata.obs_names)
    
    print(f"\n" + "=" * 80)
    print(f"ALL {len(adata.obs['condition'].unique())} UNIQUE DRUGS IN DATASET")
    print("=" * 80)
    all_drugs = sorted(adata.obs['condition'].unique())
    for i, drug in enumerate(all_drugs, 1):
        count = (adata.obs['condition'] == drug).sum()
        print(f"{i}. {drug} ({count} cells)")

        # test should be in the config split

# h5ad_to_csv("sciplex_complete_middle_subset_lincs_genes_v2.h5ad", "sciplex_complete_middle_subset_lincs_genes_v2.csv")


def remove_forbidden_drugs(input_h5ad, output_h5ad, forbidden_drugs):
    print(f"Loading {input_h5ad}...")
    # Load the original h5ad file
    adata = anndata.read_h5ad(input_h5ad)
    
    initial_cell_count = adata.n_obs
    print(f"Initial number of cells: {initial_cell_count}")
    
    # Create a boolean mask: True for cells we want to keep, False for forbidden drugs
    # The '~' symbol negates the .isin() condition
    mask = ~adata.obs['condition'].isin(forbidden_drugs)
    
    # Apply the mask to filter the AnnData object
    # We use .copy() to ensure we have an actual new object in memory, not just a "view" of the old one
    filtered_adata = adata[mask].copy()
    
    final_cell_count = filtered_adata.n_obs
    removed_cells = initial_cell_count - final_cell_count
    
    print(f"Removed {removed_cells} cells treated with forbidden drugs.")
    print(f"Remaining number of cells: {final_cell_count}")
    
    # Save the filtered AnnData object to a new h5ad file
    print(f"Saving filtered dataset to {output_h5ad}...")
    filtered_adata.write_h5ad(output_h5ad)
    print("Done!")

# forbidden_list = [
#     "Hesperadin",
#     "MK-0752",
#     "NVP-BSK805",
#     "MC1568",
#     "SRT3025",
#     "FLLL32",
#     "Quisinostat",
#     "SRT1720"
# ]

# input_file = "original_sciplex_complete_middle_subset_lincs_genes_v2.h5ad"
# output_file = "sciplex_complete_middle_subset_lincs_genes_v2.h5ad"

# remove_forbidden_drugs(input_file, output_file, forbidden_list)


def analyze_drug_splits(h5ad_file, condition_col='condition', split_col='split_random'):
    """
    Analyzes an h5ad file to show the distribution of drugs across train/test splits.
    
    Parameters:
    - h5ad_file: Path to the .h5ad file
    - condition_col: The name of the column containing the drug names
    - split_col: The name of the column containing the train/test labels
    """
    print(f"Loading {h5ad_file}...")
    adata = anndata.read_h5ad(h5ad_file)
    
    # 1. Verify that the necessary columns exist in the dataset
    if condition_col not in adata.obs.columns:
        print(f"❌ Error: Column '{condition_col}' not found in adata.obs.")
        return
    if split_col not in adata.obs.columns:
        print(f"Error: Column '{split_col}' not found in adata.obs.")
        print(f"Available metadata columns are: {list(adata.obs.columns)}")
        print(f"Please update the 'split_col' parameter to match your dataset.")
        return

    # 2. Identify the unique splits (e.g., 'train', 'test', 'validation')
    splits = adata.obs[split_col].unique()
    print(f"\n" + "=" * 80)
    print(f"DATASET SPLITS FOUND: {list(splits)}")
    print("=" * 80)
    
    split_drugs_dict = {}

    # 3. Iterate through each split and print the drug information
    for split_name in sorted(splits):
        # Subset the metadata for only the current split
        split_data = adata.obs[adata.obs[split_col] == split_name]
        
        # Get unique drugs in this split
        drugs_in_split = sorted(split_data[condition_col].unique())
        split_drugs_dict[split_name] = set(drugs_in_split)
        
        print(f"\n--- SPLIT: {split_name.upper()} ({len(split_data)} total cells) ---")
        print(f"Total unique drugs: {len(drugs_in_split)}")
        
        for i, drug in enumerate(drugs_in_split, 1):
            count = (split_data[condition_col] == drug).sum()
            print(f"  {i}. {drug} ({count} cells)")

    # 4. (Optional but helpful) Check for drug overlaps between splits
    print("\n" + "=" * 80)
    print("SPLIT OVERLAP ANALYSIS")
    print("=" * 80)
    
    # If standard train/test setup exists, let's see if any drugs are in both
    if 'train' in split_drugs_dict and 'test' in split_drugs_dict:
        overlapping_drugs = split_drugs_dict['train'].intersection(split_drugs_dict['test'])
        if overlapping_drugs:
            print(f"Found {len(overlapping_drugs)} drugs present in BOTH train and test splits:")
            for drug in sorted(overlapping_drugs):
                print(f"  - {drug}")
        else:
            print("Perfect separation: No drugs overlap between train and test splits.")


# --- How to use it ---
# Note: If your train/test labels are stored in a column named something else 
# (like 'Dataset' or 'group'), change the `split_col` parameter below!

# file_path = "mana_sciplex_complete_middle_subset_lincs_genes_v2.h5ad"
# analyze_drug_splits(file_path, condition_col='condition', split_col='split_ood_finetuning')

def replace_embeddings(parquet_file, csv_file, output_file):
    print(f"Loading original parquet: {parquet_file}")
    df_parquet = pd.read_parquet(parquet_file)
    
    print(f"Loading new VAE vectors CSV: {csv_file}")
    # Read the new data. Note: If your CSV has a column specifically for SMILES, 
    # you might need to use index_col=0 inside read_csv().
    df_csv = pd.read_csv(csv_file, header=None) 
    
    print("\n" + "="*20)
    print("PERFORMING SAFETY CHECKS")
    print("="*20)
    
    # 1. Double-check row counts
    if df_parquet.shape[0] != df_csv.shape[0]:
        raise ValueError(f"Row count mismatch! Parquet has {df_parquet.shape[0]} rows, but CSV has {df_csv.shape[0]} rows.")
    else:
        print(f"Row counts match perfectly: Both files have {df_parquet.shape[0]} drugs.")
        
    print(f"Old column count (to be removed): {df_parquet.shape[1]}")
    print(f"New column count (to be plugged in): {df_csv.shape[1]}")
    
    # 2. The Delicate Part: Swapping data while keeping the index
    # We create a brand new DataFrame. We feed it the data and columns from the CSV, 
    # but we force it to use the exact index (the SMILES) from the old parquet file.
    df_new = pd.DataFrame(
        data=df_csv.values, 
        index=df_parquet.index
    )

    df_new.columns = [f"latent_{i+1}" for i in range(df_new.shape[1])]
    
    # 3. Save the result
    print(f"\nSaving updated data to: {output_file}")
    df_new.to_parquet(output_file)
    
    print("\nDone! Here is a preview of your new DataFrame:")
    print(df_new.head())
    print(f"New DataFrame shape: {df_new.shape}")
    
    return df_new

# --- How to use it ---
# parquet_in = "embeddings/rdkit/chemCPA-rdkit_embeddings/sciplex_complete_middle_subset_lincs_genes_v2_rdkit2D_embedding.parquet"
# csv_in = "selected_vae_vectors.csv"
# parquet_out = "embeddings/mana_vae/sciplex_complete_middle_subset_lincs_genes_v2_mana_vae_embedding.parquet"

# replace_embeddings(parquet_in, csv_in, parquet_out)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build the ChemCPA drug-embedding parquet from a latent CSV")
    parser.add_argument("--latent-csv", required=True)
    parser.add_argument("--feature-size", type=int, required=True)
    parser.add_argument("--selected-csv", required=True)
    parser.add_argument("--parquet-out", required=True)
    parser.add_argument("--parquet-in", default="embeddings/rdkit/chemCPA-rdkit_embeddings/sciplex_complete_middle_subset_lincs_genes_v2_rdkit2D_embedding.parquet")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.parquet_out), exist_ok=True)
    make_csv_25in1(args.latent_csv, args.selected_csv, args.feature_size)
    replace_embeddings(args.parquet_in, args.selected_csv, args.parquet_out)