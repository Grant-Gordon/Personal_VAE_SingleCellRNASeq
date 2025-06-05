import os
import random


class ChunksDataset:
#TODO Add Warning about hardcoding for structure of files in data_dir_path 

    def __init__(self, data_dir_path, target_species = "human"):
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