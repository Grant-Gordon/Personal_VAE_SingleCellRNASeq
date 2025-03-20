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
    num_cols = batch[0][0].shape[1] # number of columns in matrix
    
    for counts, metadata in batch:
        #unpack csr tensor components 
        crow_indices = counts.crow_indices().cpu()
        col_indices = counts.col_indices().cpu()
        values = counts.values().cpu()

        #adjust crow_indices for batch offset
        adjuscted_crow = crow_indices + row_offset


        #add components to the batch
        crow_indices_list.append(adjuscted_crow[:-1]) # drop last index to avoid overlap, see python-list-slicing 
        col_indices_list.append(col_indices)
        values_list.append(values)

        #add 1 row  
        row_offset += 1

        #store metadata
        metadata_batch.append(metadata)

    #concat components
    crow_indices = torch.cat(crow_indices_list + [torch.tensor([row_offset])])
    col_indices = torch.cat(col_indices_list)
    values = torch.cat(values_list)

    #create batch csr tensor
    batch_csr_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size = (row_offset, num_cols))
   
    return batch_csr_tensor, metadata_batch

class SingleCellDataset(Dataset):    
    """
    Custom Pytorch Dataset for july2024_census_data/full 

    Args: 
        data_dir (str): Path to data directory
        species(str): Either "human" or mouse". NOTE future implementation of multispcies handling  will be needed

    Methods: 
        __len__: Returns number of total samples in the Dataset i.e. summed row count of all npz files in Dataset
        __getitem__: Retunrns tuple (sample_as_csr_row_tensor, metadata_row(pandas-df)). Returns a the (counts,metadata) of a sample[idx] where 0 < idx < __len__()
    
    Attributes:
        data_dir(string): path to full dataset e.g "../../../july2024_census/full/"
        species(string): species of the dataSet. currently only supports "human", "mouse"
        count_files(string[]): list contains name of all count npz files for the species
        metadata_files(string[]): list containing name of all metadata.pkl files for the speceis
        file_row_offset( (string, int): tuple containing the list of count.npz files and the row offset the csr matrix starts at
        total_rows(int): the total number of samples in the entire dataset i.e. summed row count of all npz files in the Dataset
    """
    #@TODO add num files counter? 
    def __init__(self,data_dir, species="human"):
        self.data_dir = data_dir
        self.species = species

        #count number of data files for speices 
        print("counting count/metadata files...")
        self.count_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"{species}_counts_")])
        self.metadata_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"{species}_metadata_")])
        print(f"count file: {len(self.count_files)}, metadataFiles: {len(self.metadata_files)}\n\n")

        #Error checking
        assert len(self.count_files)  == len(self.metadata_files), "Assert Error: Mismatch between Count and metadata files!"

        #Get row count from metadata (should be same as npz thus cheaper. 120 .pkl * 50 MB = ~6GB laoded in and out)
        self.file_row_offsets = []
        total_rows = 0

        for meta_file in self.metadata_files:
            meta_path = os.path.join(data_dir, meta_file)
            print("Opening: ", meta_file)
            with open(meta_path, "rb") as f:
                matrix_metadata = pickle.load(f)
            num_rows = matrix_metadata.shape[0]
            corresponding_counts_file = meta_file.replace('metadata', 'counts')
            corresponding_counts_file = corresponding_counts_file.replace('.pkl','.npz')
            self.file_row_offsets.append((corresponding_counts_file, total_rows))
            print(f"({corresponding_counts_file}, {total_rows}) Appended to self.file_row_offsets")
            total_rows += num_rows
        self.total_rows = total_rows
        print(f"\nTotal_rows: \t{self.total_rows}")

    def __len__(self):
        return self.total_rows
    
    def __getitem__(self, idx):
    
        #Find file with requested index
        file_idx = next(i for i, (_, start_row) in enumerate(self.file_row_offsets) if start_row > idx) -1
        file_name, start_row = self.file_row_offsets[file_idx]

        count_path = os.path.join(self.data_dir, file_name)
        metadata_path = os.path.join(self.data_dir, self.metadata_files[file_idx])


        #Only npz containing file
        count_as_scipy_csr=sp.load_npz(count_path)
        row_idx = idx - start_row #local row index within file
        single_row_csr = count_as_scipy_csr.getrow(row_idx)
        print(f"Sample {row_idx} fetched from {count_path} as \n\t{single_row_csr}")

        #convert row to sparse tensor
        sample_as_csr_row_tensor = torch.sparse_csr_tensor(
            torch.tensor(single_row_csr.indptr),
            torch.tensor(single_row_csr.indices),
            torch.tensor(single_row_csr.data),
            size = single_row_csr.shape
        )
        #TODO fix [romnt statement
        print(f"converted sample to csr_row_tensor with size:\t{sample_as_csr_row_tensor.size()}")


        #TODO load corresp row not full matrix??
        #Load corresponding metadata row into tensor
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        metadata_row = metadata.iloc[row_idx]
        
        print(f"loaded {metadata_path} and fetched sample at {row_idx}")

        return sample_as_csr_row_tensor, metadata_row
