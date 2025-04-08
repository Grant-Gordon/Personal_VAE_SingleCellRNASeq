import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
import csv
import threading
from queue import Queue

from dataLoader.single_chunk_dataset import SingleChunkDataset, ChunksDataset
from DAE.dae import DeepAutoEncoder

def create_dataloader(data_dir, counts_file, metadata_file, batch_size):
    dataset = SingleChunkDataset(
        counts_path=os.path.join(data_dir, counts_file),
        metadata_path=os.path.join(data_dir, metadata_file)
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,      #TODO tweak to optimize performacne
        pin_memory=True     #avoid memory duplication when mulitprocessing on GPU TODO(look into docs)
    )   

    return dataloader
    


def main():
    main_start_time = time.time()
    parser = argparse.ArgumentParser(description="Train Deep AutoEncoder on single-cell RNA chunks.")
    parser.add_argument("--data_dir", type=str, default="../../july2024_census_data/full/", help="Path to dataset directory.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--output_dir", type=str, default="./", help="Directory to save trained model.")
    parser.add_argument("--species",type=str, default="human",  help = "the target species that the the dataset will be comprised of NOTE file name convention is hardcoded")
    parser.add_argument("--chunks_preloaded", type=int, default=1,help="The number of chunks the dataloader will attempt to pre-load into RAM via background threads (preload_queue.size =chunks_prelaoded)")
    parser.add_argument("--num_preloader_threads", type=int, default=1, help="The number of threads that will be attemping to preload chunks into RAM:")
    args = parser.parse_args()
   
    data_dir = args.data_dir
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    output_dir = args.output_dir
    species = args.species
    chunks_preloaded = args.chunks_preloaded
    num_preloader_threads = args.num_preloader_threads

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #TODO is esle neccessary? dont want to run on cpu anyway 
    # Initialize dataset
    chunks_dataset = ChunksDataset(data_dir_path=data_dir, target_species=species)

    # Extract input size from  the first chunk
    first_counts_file, first_metadata_file = chunks_dataset[0]
    input_size = create_dataloader(data_dir, first_counts_file, first_metadata_file, 1).dataset[0].shape[0] #createdatalaoder = dl, dl.dataset = SingleChunk, SingelChunk[0] = counts[0].array.flatten tensor
  
    # Initialize model, optimizer, and loss function
    model = DeepAutoEncoder(input_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()


    #prepare logging
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard_logs"))
    timing_log_path = os.path.join(output_dir, "dataloader_timing_log.csv")
    log_file = open(timing_log_path, 'w', newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "epoch",
        "chunk",
        "batch",
        "chunk_load_time",
        "batch_creation_time" 
        "batch_train_time"])

    #prepare for multithreaded pre-loading of chunks
    preload_queue = Queue(maxsize=chunks_preloaded)
    chunk_queue= Queue()
    def preload_worker():
        while True:
            try:
                chunk_idx = chunk_queue.get(timeout=3) #Wait up to 3 seconds for work
            except:
                break # exit thread if no more work 
            
            chunk_start_time = time.time()
            counts_file, metadata_file = chunks_dataset[chunk_idx]
            dataloader = create_dataloader(data_dir=data_dir, counts_file=counts_file, metadata_file=metadata_file, batch_size=batch_size)
            chunk_load_time = time.time() - chunk_start_time
            
            preload_queue.put((chunk_idx, dataloader, chunk_load_time)) # blocks if Q full 
            chunk_queue.task_done()

    #populate Q with chunks
    for i in range(len(chunks_dataset)):
        chunk_queue.put(i)

    for _ in range(num_preloader_threads):
        t = threading.Thread(target= preload_worker)
        t.daemon=True
        t.start()


    print(f"Time until First Epoch reached: {time.time() - main_start_time:.2f}s")
    for epoch in range(epochs):
        epoch_start_time = time.time()

        epoch_loss = 0.0
        num_batches = 0

        for chunk_idx in range(len(chunks_dataset)):
            loaded_chunk_idx, current_loader, chunk_load_time = preload_queue.get()

            assert loaded_chunk_idx == chunk_idx,  f"Expected chunk {chunk_idx}, got {loaded_chunk_idx}"


            #Training step 
            for batch_idx, batch in enumerate(current_loader):
                batch_creation_start_time = time.time()
                batch = batch.to(device)
                batch_creation_time  = time.time() - batch_creation_start_time

                batch_start_time = time.time()
                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                batch_train_time = time.time() - batch_start_time

                csv_writer.writerow([
                    epoch,
                    chunk_idx,
                    batch_idx,
                    f"{chunk_load_time:.6f}",  
                    f"{batch_creation_time:.6f}",
                    f"{batch_train_time}"
                    ])

                epoch_loss += loss.item()
                num_batches += 1  # Track number of batches for averaging
            preload_queue.task_done() #optional???

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Epoch Time: {epoch_time:.2f}s")

    # Save the model
    model_save_path = os.path.join(output_dir, "dae_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    writer.close()
    log_file.close()

if __name__ == "__main__":
    main()
