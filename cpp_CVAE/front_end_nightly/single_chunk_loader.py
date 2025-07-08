#single_chunk_loader.py
from scipy import sparse
import numpy as np
import pickle #TODO


def load_csr_pointers(counts_path):

    with open(counts_path, 'rb') as f:
        counts_csr = sparse.load_npz(f)
    #samples_in_chunk = counts_csr.shape[0]
    #non_zeroes = counts_csr.nnz #DEBUG

    return (counts_csr.data, counts_csr.indices, counts_csr.indptr)



