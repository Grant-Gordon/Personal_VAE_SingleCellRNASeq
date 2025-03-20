"""
test_dataloader
Simple scrupt to test the functionality of the datalaoder 

Modules: 
    -SingelCellDataset: Custom pytorch Dataset class for loading sparse count/metadata
    -collate_csr_tensor: custome collate_fn function for batching sparse CSR into tensors. **NOTE: This functions correctness has not been fully tested**

"""

from torch.utils.data import DataLoader
from data_loader_class import SingleCellDataset
from data_loader_class import collate_csr_tensors


data_dir = '../../../july2024_census_data/full/'

print("creating human_dataset")
human_dataset = SingleCellDataset(data_dir, species="human")
print("creating mouse_dataset")
mouse_dataset = SingleCellDataset(data_dir, species="mouse")

print("creating human_loader")
human_loader = DataLoader(human_dataset, batch_size=5, shuffle=True, collate_fn=collate_csr_tensors)
print("creating mosue_loader")
mouse_loader = DataLoader(mouse_dataset, batch_size=5, shuffle=True, collate_fn=collate_csr_tensors)

def load_first_batch():
    
    """
    Load and print first batch of humanDataset 
    """
    print("Called load_first_batch()")
    for counts, metadata in human_loader:
        print("Counts Shape: ", counts.shape)
        print("Metadata: ", metadata)
        break




if __name__=="__main__":
    load_first_batch()


