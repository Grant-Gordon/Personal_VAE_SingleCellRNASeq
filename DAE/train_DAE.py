import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
import csv

from dataLoader.single_chunk_dataset import SingleChunkDataset, ChunksDataset
from DAE.dae import DeepAutoEncoder

def main():

    parser = argparse.ArgumentParser(description="Train Deep AutoEncoder on single-cell RNA chunks.")
    parser.add_argument("--data_dir", type=str, default="../../july2024_census_data/full/", help="Path to dataset directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save trained model.")
    parser.add_argument("--species",type=str, default="human",  help = "the target species that the the dataset will be comprised of NOTE file name convention is hardcoded")
   
    args = parser.parse_args()
   
    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    output_dir = args.output_dir

    # Initialize dataset
    chunks_dataset = ChunksDataset(data_dir_path=data_dir, target_species="human")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #TODO is esle neccessary? dont want to run on cpu anyway 

    # Extract input size dynamically from the first chunk
    first_counts_file, first_metadata_file = chunks_dataset[0]

    first_counts_path = os.path.join(data_dir, first_counts_file)
    first_metadata_path = os.path.join(data_dir, first_metadata_file)

    first_chunk_dataset = SingleChunkDataset(counts_path=first_counts_path, metadata_path=first_metadata_path)
    input_size = first_chunk_dataset[0].shape[0]  # Assuming each sample is 1D (genes/features)

    # Initialize model, optimizer, and loss function
    model = DeepAutoEncoder(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()


    #prepare logging
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard_logs"))
    timing_log_path = os.path.join(output_dir, "dataloader_timing_log.csv")
    log_file = open(timing_log_path, 'w', newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["epoch", "chunk", "batch", "chunk_load_time", "batch_train_time"])


    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for chunk_idx, (counts_file, metadata_file) in enumerate(chunks_dataset):  # Iterate over each chunk (lazy loading)
            chunk_start_time = time.time()
            single_chunk_dataset = SingleChunkDataset(counts_path=os.path.join(data_dir, counts_file), metadata_path=os.path.join(data_dir, metadata_file))
            dataloader = DataLoader(single_chunk_dataset, batch_size=batch_size, shuffle=True)
            chunk_load_time = time.time() - chunk_start_time

            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(device)

                batch_start_time = time.time()

                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                
                batch_train_time = time.time() = batch_start_time

                csv_writer.writerow([
                    epoch,chunk_idx,
                    batch_idx,
                    f"{chunk_load_time:.6f}",  
                    f"{batch_train_time:.6f}"
                    ])

                epoch_loss += loss.item()
                num_batches += 1  # Track number of batches for averaging

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

    # Save the model
    model_save_path = os.path.join(output_dir, "dae_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    writer.close()
    log_file.close()

if __name__ == "__main__":
    main()
