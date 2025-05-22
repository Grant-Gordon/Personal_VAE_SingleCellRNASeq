import torch
from torch.utils.data import Dataset
from scipy import sparse
import pickle


class SingleChunkDataset(Dataset):
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

        for field, spec in self.metadata_fields.items():
                if spec["type"] == "IGNORE":
                    continue

                value = meta_row[field]

                if spec["type"] == "embedding" or spec["type"] == "onehot":
                    vocab = self.vocab_dict.get(field, {})
                    idx = vocab.get(value, 0)
                    meta_encoded[field] = torch.tensor(idx, dtype=torch.long)

                elif spec["type"] == "continuous":
                    meta_encoded[field] = torch.tensor(float(value), dtype=torch.float32)



        return{
            "expr":expr,
            "metadata":meta_encoded
        }
    
    def __len__(self):
        return self.samples_in_chunk