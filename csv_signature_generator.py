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
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        if filename.endswith(".h5"):
            input_file = os.path.join(input_folder, filename)
            # A1_sign3.h5 -> A1.csv (ae.py / drugzip.py expect A1.csv..E5.csv)
            stem = filename[:-3]  # drop .h5
            if stem.endswith("_sign3"):
                stem = stem[: -len("_sign3")]
            output_file = os.path.join(output_folder, f"{stem}.csv")
            convert_h5_to_csv(input_file, output_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Chemical Checker signature3 .h5 files to DrugZip CSVs."
    )
    parser.add_argument("input_folder", help="Folder containing A1.h5..E5.h5")
    parser.add_argument(
        "output_folder",
        nargs="?",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Data",
            "ChemicalChecker_Full",
        ),
        help="Destination for A1.csv..E5.csv (default: Data/ChemicalChecker_Full)",
    )
    args = parser.parse_args()
    convert_folder_h5_to_csv(args.input_folder, args.output_folder)

