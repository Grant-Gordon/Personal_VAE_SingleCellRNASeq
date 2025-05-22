import torch
from torch.utils.data import Dataset
from scipy import sparse
import pickle


class SingeleChunkDataset(Dataset):
    def __init__(self, counts_path, metadata_path, vocab_dict=None):
        self.vocab_dict = vocab_dict or {}

        with open(counts_path, 'rb') as f:
            self.counts_csr = sparse.load_npz(f)
        self.samples_in_chunk = self.counts_csr.shape[0]


        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)


    def __getitem__(self, index):
        expr = torch.tensor(self.count_csr[index].toarray().flatten(), dtype=torch.float32)

        meta_row = self.metadata.iloc[index]
        meta_encoded = {}
        for k in meta_row.index: #k=column name
            v = meta_row[k]
            if k in self.vocab_dict:
                meta_encoded[k] = torch.tensor(self.vocab_dict[k].get(v,0), dtype=torch.long)
            elif isinstance(v, (int,float)):
                meta_encoded[k] = torch.tensor(v, dtype=torch.float32)
            else:
                continue
    
        return{
            "expr":expr,
            "metadata":meta_encoded
        }
    
    def __len__(self):
        return self.samples_in_chunk