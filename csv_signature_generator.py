import h5py
import pandas as pd
import os

def convert_h5_to_csv(input_file, output_file):
    with h5py.File(input_file, "r") as hf:
        # 'V' contains the 128-sized floating point vectors
        vectors = hf['V'][:]
        
        # 'keys' contains the InChIKeys (stored as bytes)
        # Decoding them to standard strings
        inchikeys = [k.decode('utf-8') for k in hf['keys'][:]]

    # Create column headers V1 through V128
    cols = [f"V{i}" for i in range(1, 129)]

    # Create the DataFrame
    df = pd.DataFrame(vectors, columns=cols)

    # Insert the InChIKey as the first column with an empty header name
    # This allows you to track the drug identity across all CC files
    df.insert(0, "", inchikeys)

    # Export to CSV with quoting to match your requested style
    df.to_csv(output_file, index=False, quoting=1)

    print(f"Success! {len(df)} drugs exported to {output_file}")

# Function to convert all .h5 files in a folder to .csv
# Retrieve the files from chemicalchecker.com/downloads/signature3
def convert_folder_h5_to_csv(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".h5"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename.replace(".h5", ".csv"))
            convert_h5_to_csv(input_file, output_file)

