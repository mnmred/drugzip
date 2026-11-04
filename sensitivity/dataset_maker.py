import pandas as pd

def xlsx_to_csv(xlsx_file, csv_file):
    df = pd.read_excel(xlsx_file)
    df.to_csv(csv_file, index=False)

def csv_curation(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Only select rows where the 'DATASET' column equals 'GDSC2'
    condition = df['DATASET'] == 'GDSC2'
    df_cd = df[condition]


    #selecting specific columns from the dataframe
    column1 = 'CELL_LINE_NAME'  
    column2 = 'DRUG_ID'      
    column3 = 'DRUG_NAME'  
    column4 = 'CANCER_TYPE'
    column5 = 'LN_IC50'
    df_new = df_cd[[column1, column2, column3, column4, column5]]
    df_new.to_csv(output_csv, index=False)

def dataset_cell_check(csv_file, cell_file):
    df = pd.read_csv(csv_file)
    cell_df = pd.read_csv(cell_file)
    print("Number of duplicated rows:", df.duplicated().sum())
    #check cell names
    cells_seen = df["CELL_LINE_NAME"].unique()
    # print("Unique cell lines in dataset:", cells_seen)
    print("Number of unique cell lines:", len(cells_seen))

    # cell names are in header of cell_df
    cell_names_in_cell_df = cell_df.columns[1:]  # The first column is not a cell name
    cells_unfound = set(cells_seen) - set(cell_names_in_cell_df)
    # print("Cell lines in dataset that are not in cell_df:", cells_unfound)
    print("Number of cell lines in dataset that are not in cell_df:", len(cells_unfound))
    cells_overlap = set(cells_seen) & set(cell_names_in_cell_df)
    print("Number of cell lines in dataset that are also in cell_df:", len(cells_overlap))

    # select only the rows where the 'CELL_LINE_NAME' column is in the cell names of cell_df
    df_filtered = df[df['CELL_LINE_NAME'].isin(cell_names_in_cell_df)]
    df_filtered.to_csv("dataset/GDSC2_cell1_overlap.csv", index=False)

def check_duplicate_combinations(csv_file):
    df = pd.read_csv(csv_file)
    # Check for duplicate combinations of 'CELL_LINE_NAME' and 'DRUG_ID'
    duplicate_combinations = df.duplicated(subset=['CELL_LINE_NAME', 'DRUG_ID'], keep=False)
    num_duplicates = duplicate_combinations.sum()
    print("Number of duplicate cell line and drug ID combinations:", num_duplicates)

def distribution_of_ln_ic50(csv_file):
    df = pd.read_csv(csv_file)
    # Calculate the distribution of 'LN_IC50' values
    distribution = df['LN_IC50'].describe() 
    print("Distribution of LN_IC50 values:")
    print(distribution)

def dataset_drug_check(csv_file, drug_file):
    df = pd.read_csv(csv_file)
    drug_df = pd.read_csv(drug_file, header=None)
    drugs_seen = df["DRUG_NAME"].unique() 
    print("Number of unique drugs:", len(drugs_seen))

    drug_names_in_drug_df = drug_df[0] #select the whole first column of drug_df which contains drug names
    drugs_unfound = set(drugs_seen) - set(drug_names_in_drug_df)
    print("Number of drugs in dataset that are not in drug_df:", len(drugs_unfound))
    drugs_overlap = set(drugs_seen) & set(drug_names_in_drug_df)
    print("Number of drugs in dataset that are also in drug_df:", len(drugs_overlap))
    df_filtered = df[df['DRUG_NAME'].isin(drug_names_in_drug_df)]
    df_filtered.to_csv("dataset/GDSC2_cnd_overlap.csv", index=False)


# concatenate the three drug name files into one file
def concatenate_drug_files(file1, file2, file3, output_file):
    df1 = pd.read_csv(file1, header=None)
    df2 = pd.read_csv(file2, header=None)
    df3 = pd.read_csv(file3, header=None)
    concatenated_df = pd.concat([df1, df2, df3], ignore_index=True)
    concatenated_df.to_csv(output_file, index=False, header=False)
    
def txt_to_csv(txt_file, csv_file):
    df = pd.read_csv(txt_file, sep="\t", header=None)
    df.to_csv(csv_file, index=False, header=False)

def select_drugs_from_csv():
    df = pd.read_csv("dataset/GDSC2_cnd_overlap.csv")
    drug_names = df["DRUG_NAME"].unique()
    df_mapping = pd.read_csv("dataset/all.csv", header=None)
    df_mapping = df_mapping[1]
    #get index of drug names in drug_names_df that are in df_mapping in order
    drug_names_in_mapping = df_mapping[df_mapping.isin(drug_names)]
    drug_names_in_mapping = drug_names_in_mapping.reset_index()
    drug_names_in_mapping.to_csv("dataset/drug_names_in_mapping.csv", index=False, header=False)
    indices = drug_names_in_mapping.index.tolist()

    return indices

def drug_position_mapping():
    df_mapping = pd.read_csv("dataset/Drugs/InChiKeys/all_drugs_inchikeys.csv", header=None) #it is 0 indexed
    # iterate through gdsc2 drug names and find their position in df_mapping
    # add position column to GDSC2_cnd_overlap.csv
    df_gdsc2 = pd.read_csv("dataset/GDSC2_cnd_overlap.csv")
    df_name_to_key = pd.read_csv("dataset/Drugs/InChiKeys/syn-keys.csv", header=None)

    drug_names = df_gdsc2["DRUG_NAME"].unique()
    
    # Extract inchikeys column from mapping (column 0)
    inchikeys_mapping = df_mapping[0]
    
    drug_positions = []
    for drug in drug_names:
        # Step 1: Find the inchikey for this drug name in df_name_to_key
        # Assuming column 0 has drug names and column 1 has inchikeys
        drug_rows = df_name_to_key[df_name_to_key[0] == drug]
        
        if len(drug_rows) > 0:
            inchikey = drug_rows.iloc[0][1]
            
            # Step 2: Find the inchikey in df_mapping and get its position
            if inchikey in inchikeys_mapping.values:
                position = inchikeys_mapping[inchikeys_mapping == inchikey].index[0]
                drug_positions.append(position)
            else:
                drug_positions.append(-1)  # inchikey not found in mapping
        else:
            drug_positions.append(-1)  # drug name not found in name_to_key mapping
    
    df_gdsc2["DRUG_POSITION"] = df_gdsc2["DRUG_NAME"].apply(lambda x: drug_positions[drug_names.tolist().index(x)])
    df_gdsc2.to_csv("dataset/GDSC2_cnd_overlap_with_positions.csv", index=False)

# same as drug_position_mapping but for cell lines
def cell_position_mapping():        
    df_mapping = pd.read_csv("dataset/Cells/cell_lines.csv", header=None)
    # iterate through gdsc2 cell line names and find their position in df_mapping
    # add position column to GDSC2_cnd_overlap.csv
    df_gdsc2 = pd.read_csv("dataset/GDSC2_cnd_overlap_with_positions.csv")
    cell_names = df_gdsc2["CELL_LINE_NAME"].unique()         
    df_mapping = df_mapping[0]
    cell_positions = []
    for cell in cell_names:
        if cell in df_mapping.values:
            position = df_mapping[df_mapping == cell].index[0]
            cell_positions.append(position)
        else:
            cell_positions.append(-1) # if cell is not found, add -1 to the list
    df_gdsc2["CELL_POSITION"] = df_gdsc2["CELL_LINE_NAME"].apply(lambda x: cell_positions[cell_names.tolist().index(x)])
    df_gdsc2.to_csv("dataset/GDSC2_cnd_overlap_with_positions.csv", index=False)

def remove_unmapped_entries(csv_file, output_csv):
    df = pd.read_csv(csv_file)
    # Remove rows where DRUG_POSITION or CELL_POSITION is -1
    df_filtered = df[(df['DRUG_POSITION'] != -1)]
    df_filtered.to_csv(output_csv, index=False)

def check_duplicates_smiles():
    df = pd.read_csv("dataset/Drugs/SMILES/smiles.csv", header=None)
    # Check for duplicate SMILES in the dataframe
    duplicate_keys = df.duplicated(subset=[0], keep=False)
    num_duplicates = duplicate_keys.sum()
    print("Number of duplicate InChIKeys:", num_duplicates)
    print("Number of unique InChIKeys:", df[0].nunique())

    duplicate_smiles = df.duplicated(subset=[1], keep=False)
    num_duplicates = duplicate_smiles.sum()
    print("Number of duplicate SMILES:", num_duplicates)
    print("Number of unique SMILES:", df[1].nunique())

    print("Example of duplicate InChIKeys:")
    print(df[duplicate_keys])

def remove_key_duplicates_smiles(input_csv, output_csv):
    df = pd.read_csv(input_csv, header=None)
    # Remove duplicate InChIKeys, keeping the first occurrence
    df_unique = df.drop_duplicates(subset=[0], keep='first')
    df_unique.to_csv(output_csv, index=False, header=False)
    print("Size after removing duplicate InChIKeys:", df_unique.shape[0])

def return_unique_cancer_types(csv_file):
    df = pd.read_csv(csv_file)
    unique_cancer_types = df["CANCER_TYPE"].unique()
    print("Unique cancer types in the dataset:", unique_cancer_types)
    print("Number of unique cancer types:", len(unique_cancer_types))

# return_unique_cancer_types("dataset/GDSC2_cnd_overlap_with_positions.csv")
# df = pd.read_csv("dataset/GDSC2_cnd_overlap_with_positions.csv")
# print("Number of unique drugs:", len(df["DRUG_NAME"].unique()))
# print("Number of unique cell lines:", len(df["CELL_LINE_NAME"].unique()))

# df_vae = pd.read_csv("dataset/Drugs/25in1-full-b256.csv")
# print("Size of VAE features:", df_vae.shape[0])
# df_all = pd.read_csv("dataset/Drugs/InChiKeys/all_drugs_inchikeys.csv", header=None)
# print("Size of all drug mapping:", df_all.shape[0])
# df_smiles = pd.read_csv("dataset/Drugs/SMILES/smiles.csv", header=None)
# print("Size of SMILES features:", df_smiles.shape[0])
# df_smiles1 = pd.read_csv("dataset/Drugs/SMILES/smiles1.csv", header=None)
# print("Size of SMILES features part 1:", df_smiles1.shape[0])
# df_smiles2 = pd.read_csv("dataset/Drugs/SMILES/smiles2.csv", header=None)
# print("Size of SMILES features part 2:", df_smiles2.shape[0])
# df_smiles3 = pd.read_csv("dataset/Drugs/SMILES/smiles3.csv", header=None)
# print("Size of SMILES features part 3:", df_smiles3.shape[0])
# print non empty rows for second column of smiles_no_key_duplicates.csv
# df_smiles_no_key_duplicates = pd.read_csv("dataset/Drugs/SMILES/smiles_no_key_duplicates.csv", header=None)
# non_empty_smiles = df_smiles_no_key_duplicates[df_smiles_no_key_duplicates[1].notna()]
# print("Number of non-empty SMILES after removing key duplicates:", non_empty_smiles.shape[0]) 

df = pd.read_csv("dataset/GDSC2_cnd_overlap_with_positions.csv")
print("size of drug positions:", len(df["DRUG_POSITION"].unique()))
