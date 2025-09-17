#dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
from scipy import sparse
import pickle
import os
import random
from typing import Dict, Tuple, List

def create_dataloader(
    data_dir:str,
    counts_file:str,
    metadata_file:str,
    batch_size:int,
    meta_category_ID_map_path:str, 
    field_specs_path:str
    ):
    with open(meta_category_ID_map_path, 'rb') as f:
        category_to_id:Dict[str, Dict[str,int]] = pickle.load(f)

    with open(field_specs_path, 'rb') as f:
        field_specs: Dict[str, Dict[str, int]] = pickle.load(f)


    dataset = SingleChunkDataset(
        counts_path=os.path.join(data_dir, counts_file),
        metadata_path=os.path.join(data_dir, metadata_file),
        category_to_id=category_to_id,
        field_specs=field_specs
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, #TODO make configurable
        pin_memory=True
    )

class SingleChunkDataset(Dataset):
    def __init__(self,
        counts_path:str,
        metadata_path:str,
        category_to_id: Dict[str, Dict[str, int]],
        field_specs:Dict[str, Dict[str, int]]
        ):
        self.category_to_id = category_to_id
        self.field_specs = field_specs
       
        with open(counts_path, 'rb') as f:
            self.counts_csr = sparse.load_npz(f)
        self.samples_in_chunk = self.counts_csr.shape[0]

        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        self.used_fields: List[str] = [f for f, spec in self.field_specs.items() if spec.get("USING", False)]

    def __getitem__(self, index: int):
        # Expression row -> dense 1D FloatTensor [G]
        row = self.counts_csr[index]
        expr = torch.tensor(row.toarray().ravel(), dtype=torch.float32)

        # Metadata IDs per used field
        meta_ids: Dict[str, torch.Tensor] = {}
        meta_row = self.metadata.iloc[index]
        for field in self.used_fields:
            cat_map = self.category_to_id.get(field, {})
            raw_val = meta_row.get(field, "__UNK__")
            id_val = cat_map.get(raw_val, cat_map.get("__UNK__", 0))
            meta_ids[field] = torch.tensor(id_val, dtype=torch.long)

        return {"expr": expr, "metadata": meta_ids}

    
    def __len__(self):
        return self.samples_in_chunk
    

class ChunksDataset:
#TODO Add Warning about hardcoding for structure of files in data_dir_path 

    def __init__(self, data_dir_path:str, target_species:str = "human"):
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
    