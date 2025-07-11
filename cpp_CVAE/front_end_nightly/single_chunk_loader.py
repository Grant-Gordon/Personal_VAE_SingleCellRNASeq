#single_chunk_loader.py
from scipy import sparse
import numpy as np
import pickle #TODO


def load_csr_pointers(counts_path):

    with open(counts_path, 'rb') as f:
        counts_csr = sparse.load_npz(f)


    return (counts_csr.data, counts_csr.indices, counts_csr.indptr, counts_csr.shape, counts_csr.nnz)



