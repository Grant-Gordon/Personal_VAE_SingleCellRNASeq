import os
import pickle
from torch.utils.data import DataLoader
from dataloader.chunks_dataset import ChunksDataset
from dataloader.single_chunk_dataset import SingleChunkDataset


def create_chunks_dataset(data_dir, species):
    return ChunksDataset(data_dir, species)

def create_dataloader(data_dir, counts_file, metadata_file, batch_size, config):
    with open(config["metadata_vocab"], 'rb') as f:
        vocab_dict = pickle.load(f)

    dataset = SingleChunkDataset(
        counts_path=os.path.join(data_dir, counts_file),
        metadata_path=os.path.join(data_dir, metadata_file),
        vocab_dict=vocab_dict,
        metadata_fields=config["metadata_fields"]
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, #TODO make configurable
        pin_memory=True
    )