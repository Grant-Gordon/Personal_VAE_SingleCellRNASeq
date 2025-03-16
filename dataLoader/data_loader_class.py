"""
Data_Loader_class

This script provides functionality for loading in the July2024 census data fullset (human/mouse_counts_x.npz & huma/mouse_metadata_x.pkl). 
Loads from sparse CSR formate into CSR tensor format 

Methods: 
    -collate_csr_tensor(batch): collates a batch of csr tensors and returns a single sparse CSR tensor representing the batch and a list of the metadata 

"""

import torch
from torch.utils.data import Dataset
import scipy.sparse as sp
import pickle
import numpy as np
import os 

def collate_csr_tensors(batch):
    """Collate a batch of sparse CSR tensors.
    Args:
        batch(lsit): A list of tuples with sparse csr tensors and metadata 
    Returns:
       tuple:  A single sparse CSR tensor representing the batch and a list of the metadata 
    """
    crow_indices_list = []
    col_indices_list = []
    values_list=[]
    metadata_batch = []

    #track row offset when stacking multiple  csr matrices 
    row_offset = 0

    for counts, metadata in batch:
        #unpack csr tensor components 
        crow_indices = counts.crow_indices().cpu()
        col_indices = counts.col_indices().cpu()
        values = counts.values().cpu()

        #adjust crow_indeces for batch offset
        adjuscted_crow = crow_indices + row_offset


        #add components to the batch
        crow_indices_list.append(adjuscted_crow[:-1]) # drop last index to avoid overlap, see python-list-slicing 
        col_indices_list.append(col_indices)
        values_list.append(values)

        #add final row pointer offset 
        row_offset += counts.shape[0]

        #store metadata
        metadata_batch.append(metadata)

    #concat components
    crow_indices = torch.cat(crow_indices_list + [torch.tensor([row_offset])])
    col_indices = torch.cat(col_indices_list)
    values = torch.cat(values_list)

    #create batch csr tensor
    batch_csr_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size = (row_offset, counts.shape[1]))
    return batch_csr_tensor, metadata_batch

class SingleCellDataset(Dataset):    
    """
    Custom Pytorch Dataset for july2024_census_data/full 

    Args: 
        data_dir (str): Path to data directory
        species(str): Either "human" or mouse". NOTE future implementation of multispcies handling  will be needed

    Methods: 
        __len__: Returns number of files in Dataset
        __getitem__: Loads a sparce count matrix and metadata for a given index
    """
    #@TODO add num files counter? 
    def __init__(self,data_dir, species="human"):
        self.data_dir = data_dir
        self.species = species

        #count number of data files for speices 
        print("counting count/metadata files...")
        self.count_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"{species}_counts_")])
        self.metadata_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"{species}_metadata_")])
        print(f"count file: {len(self.count_files)}, metadataFiles: {len(self.metadata_files)}")

        #Error checking
        assert len(self.count_files)  == len(self.metadata_files)

    def __len__(self):
        return len(self.count_files)
    
    def __getitem__(self, idx):
        print(f"in __getitem__ at idx: {idx}")
        #load sparse matrix
        #TODO Confirm july2024_censusdata/name/of/files
        count_path= os.path.join(self.data_dir,self.count_files[idx])
        metadata_path= os.path.join(self.data_dir, self.metadata_files[idx])

        print("loading counts_as_sparse_matrix")
        counts_as_scipy_csr = sp.load_npz(count_path) #load sparse matrix 
        print("loaded npz as sparse matrix ")
        
        print("loading counts_as_csr_tensor")
        counts_as_csr_tensor = torch.sparse_csr_tensor(
            torch.tensor(counts_as_scipy_csr.indptr),
            torch.tensor(counts_as_scipy_csr.indices),
            torch.tensor(counts_as_scipy_csr.data),
            size=counts_as_scipy_csr.shape
        )
        print("loaded sparse matrix into csr tensor")
        print(counts_as_csr_tensor)

        # load Metadata
        print("opening metadata (.pkl)")
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        print("retuning")
        return counts_as_csr_tensor, metadata
