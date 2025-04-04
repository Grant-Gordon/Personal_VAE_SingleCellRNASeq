import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
import pickle
import numpy as np
import os
import random 



class ChunksDataset:
    def __init__(self, data_dir_path, target_species = "human"):
        self.data_dir_path = data_dir_path
        self.target_species = target_species
        self.shuffled_chunk_list = self._get_shuffled_chunk_list()
        self.current_chunk_idx = -1 

    #returns a shuffled list of tuples(count, metadata) file names
    def _get_shuffled_chunk_list(self):
        chunk_list = []

        files = set(os.listdir(self.data_dir_path))

        for file in files:
            if file.startswith(f"{self.target_species}_counts"):
                metadata_file = file.replace("counts", "metadata")
                metadata_file = metadata_file.replace("npz", "pkl")
                if metadata_file in files: 
                    chunk_list.append((file,metadata_file))
        random.shuffle(chunk_list)
        return chunk_list
    
    def __getitem__(self, index):
        return self.shuffled_chunk_list[index]
    
    def __len__(self):
        return len(self.shuffled_chunk_list)



class SingleChunkDataset(Dataset):
    
    def __init__(self, counts_path, metadata_path):
        self.counts_path = counts_path
        self.metadata_path = metadata_path

        with open(self.counts_path, "rb") as f:
            self.counts_csr = sp.load_npz(f)
        self.samples_in_chunk = self.counts_csr.shape[0]


        with open(self.metadata_path, "rb") as f:
            self.metadata = pickle.load(f)


    def __getitem__(self, index):
        return torch.tensor(self.counts_csr[index].toarray().flatten(), dtype=torch.float32)  # Convert row to dense
    

    def __len__(self):
        return self.samples_in_chunk
