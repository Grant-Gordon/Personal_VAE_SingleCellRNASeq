import pickle

# Paths to PKL files (modify these as needed)
pkl_files = [
    "../../../july2024_census_data/full/human_metadata_1.pkl",
    "../../../july2024_census_data/full/human_metadata_2.pkl"
]

for file_path in pkl_files:
    print(f"Loading {file_path}")

    # Load the pickle file
    with open(file_path, "rb") as f:
        metadata = pickle.load(f)

    # Check the type of metadata
    if isinstance(metadata, dict):
        print(f"Metadata keys: {list(metadata.keys())[:5]}")  # Print first few keys

        # Preview first few values for first two keys
        for key in list(metadata.keys())[:2]:
            values = metadata[key]
            print(f"First 5 values for {key}: {values[:5] if isinstance(values, list) else values}")

    elif isinstance(metadata, list):  # If it's a list of dictionaries
        print(f"Metadata contains {len(metadata)} entries.")
        print("First entry:", metadata[0])

    else:
        print(f"Metadata is of type {type(metadata)}, previewing raw content:")
        print(metadata)

    print("\n" + "-"*50 + "\n")
