import torch
from torch.utils.data import DataLoader
from single_chunk_dataset import ChunksDataset, SingleChunkDataset

data_dir = "/path/to/counts/and/metadata/"
chunks_dataset = ChunksDataset(data_dir_path=data_dir, target_species="human")

epochs = 10
batch_size = 128
device = torch.device("cuda")

for epoch in range(epochs):
    for chunk in range(len(chunks_dataset)):
        single_chunk_dataset = SingleChunkDataset(counts_path=chunk[0], metadata_path=chunk[1])
        dataloader = DataLoader(single_chunk_dataset, batch_size=batch_size, shuffle=True)

        for batch in dataloader:
            #train
