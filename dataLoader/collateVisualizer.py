import torch
import matplotlib.pyplot as plt

def visualize_sparse_csr(csr_tensor, title="Sparse CSR Tensor", save_path="sparse_visualization.jpg"):
    if not csr_tensor.is_sparse_csr:
        raise ValueError("Input must be a sparse CSR tensor.")

    # Extract row and column indices from the sparse tensor
    rows = csr_tensor.crow_indices().cpu().numpy()
    cols = csr_tensor.col_indices().cpu().numpy()

    # Convert compressed row pointers to actual row indices
    row_indices = []
    for i in range(len(rows) - 1):
        start = rows[i]
        end = rows[i + 1]
        row_indices.extend([i] * (end - start))

    plt.scatter(cols[:100000], row_indices[:100000], s=0.05, alpha=0.5)
    plt.title(title)
    plt.savefig(save_path, dpi=300)  # Save to file
    print(f"Saved sparse visualization to {save_path}")
if __name__ == "__main__":
    from data_loader_class import SingleCellDataset, collate_csr_tensors


    data_dir = '../../../july2024_census_data/full/'
    human_dataset = SingleCellDataset(data_dir, species="human")

    # Load batched tensor from your DataLoader    
    dataset = human_dataset
    batch = [dataset[44], dataset[6]]  # Example batch
    batched_tensor, metadata = collate_csr_tensors(batch)
    
    visualize_sparse_csr(batched_tensor, title="Batched Sparse Tensor", save_path="batch_collate_viz.jpg")
