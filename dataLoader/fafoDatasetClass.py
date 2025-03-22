import os  
import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
import pickle
import numpy as np
import random

SEED = 1
random.seed(SEED)

class SuperBatch:
    """
    Class designed to load as many batches into RAM of bigmem node as possible to minimize number of times npz files are loaded and parsed.
    """
    sample_length = 1  # TODO: Implement correctly

    @staticmethod
    def chunk(lst, size):
        """
        Helper function to split all indices of full_dataset into sub-arrays of length = batchsize.
        """
        for i in range(0, len(lst), size):
            yield lst[i:i + size]

    def getMaxBatchesPerSuperBatch(self):
        """
        Calculate the number of batches that will fit in the bigmem RAM partition (need to look into HPC specs).
        """
        BATCH_SIZE_BYTES = 100  # TODO: Find the actual batch size in bytes
        AVAILABLE_MEM_IN_BIGMEM = 1000  # TODO: Fetch actual RAM size

        return int(AVAILABLE_MEM_IN_BIGMEM / BATCH_SIZE_BYTES)

    def populateSuperBatch(self):
        """
        Parse the count.npz files and load relevant data into the superbatch.
        """
        for count_file, start_idx in self.file_row_offsets:
            count_path = os.path.join(self.data_dir, count_file)
            print(f"Loading {count_path} into memory...")

            sparse_matrix = sp.load_npz(count_path)  # Load sparse matrix


            #TODO validate GPT section is correct. Not sure its doing what i want it to
            for batch in self.batches_as_sample_ID:
                for sample_ID in batch:
                    relative_idx = sample_ID - start_idx
                    if 0 <= relative_idx < sparse_matrix.shape[0]:  # Ensure index is valid
                        self.superbatch_data[sample_ID] = sparse_matrix[relative_idx, :].toarray()

    def __init__(self, batch_size, data_dir, species="human"):
        """
        Constructor for the SuperBatch.
        """
        self.data_dir = data_dir
        self.species = species
        self.batch_size = batch_size

        # Count number of data files for species
        print("Counting count/metadata files...")
        self.full_dataset_count_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"{species}_counts_")])
        self.full_dataset_metadata_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"{species}_metadata_")])
        print(f"Count files: {len(self.full_dataset_count_files)}, Metadata files: {len(self.full_dataset_metadata_files)}\n\n")

        assert len(self.full_dataset_count_files) == len(self.full_dataset_metadata_files), "Error: Mismatch between count and metadata files!"

        # Get row count from metadata
        total_rows = 0
        self.file_row_offsets = []  # (count.npz, start_idx_in_full_dataset)

        for meta_file in self.full_dataset_metadata_files:
            meta_path = os.path.join(data_dir, meta_file)
            print("Opening:", meta_file)
            with open(meta_path, "rb") as f:
                matrix_metadata = pickle.load(f)

            num_rows = matrix_metadata.shape[0]
            corresponding_counts_file = meta_file.replace('metadata', 'counts').replace('.pkl', '.npz')
            self.file_row_offsets.append((corresponding_counts_file, total_rows))
            print(f"({corresponding_counts_file}, {total_rows}) appended to self.file_row_offsets")
            total_rows += num_rows

        self.full_dataset_total_rows = total_rows
        print(f"\nTotal rows:\t{self.full_dataset_total_rows}")

        self.num_batches = self.getMaxBatchesPerSuperBatch()

        # Shuffle sample indices and split into subarrays of batch size
        sample_ID_array = list(range(self.full_dataset_total_rows))
        random.shuffle(sample_ID_array)
        batch_IDxsample_ID = list(SuperBatch.chunk(sample_ID_array, batch_size))

        # Fit as many batches into superbatch and record those left behind
        self.batches_as_sample_ID = batch_IDxsample_ID[:self.num_batches]  
        self.excluded_batches_as_sample_ID = batch_IDxsample_ID[self.num_batches:]

        # Preallocate space for the superbatch
        self.superbatch_data = np.zeros((self.batch_size * self.num_batches, SuperBatch.sample_length), dtype=np.float32)

        self.populateSuperBatch()

    def __len__(self):
        return self.num_batches

    def samplesInSuperBatch(self):
        return self.num_batches * self.batch_size

    def __getitem__(self, batch_idx):
        batch = self.superbatch_data[batch_idx * self.batch_size: (batch_idx + 1) * self.batch_size]
        return torch.tensor(batch, dtype=torch.float32)
