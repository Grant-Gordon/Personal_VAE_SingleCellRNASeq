#single_chunk_loader.py
from scipy import sparse
import numpy as np
import pickle #TODO

def load_csr_pointers(counts_path):
    with open(counts_path, 'rb') as f:
        counts_csr = sparse.load_npz(f)
    return {
        "data":counts_csr.data,
        "indices":counts_csr.indices,
        "indptr":counts_csr.indptr,
        "shape":counts_csr.shape,
        "nnz":counts_csr.nnz
        }



