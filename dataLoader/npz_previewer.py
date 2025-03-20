import numpy as np
import scipy.sparse as sp

# Paths to NPZ files (modify these as needed)
npz_files = [
    "../../../july2024_census_data/full/human_counts_1.npz",
    "../../../july2024_census_data/full/human_counts_2.npz"
]

for file_path in npz_files:
    print(f"Loading {file_path}")

    # Load CSR matrix
    data = sp.load_npz(file_path)

    # Convert first few rows to dense to inspect
    num_rows, num_cols = data.shape
    preview_cols = min(5, num_cols)  # First 5 columns or fewer if not available
    preview_rows = min(5, num_rows)  # First 5 rows or fewer if not available

    dense_preview = data[:preview_rows, :preview_cols].toarray()

    print(f"Shape: {data.shape}")
    print(f"First {preview_rows} rows, {preview_cols} columns:")
    print(dense_preview)

    print("\n" + "-"*50 + "\n")
