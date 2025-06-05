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
import numpy as np

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
   
   #instantiate args
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
    tensorboard_writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard_logs"))
    timing_log_path = os.path.join(output_dir, "dataloader_timing_log.csv")
    chunks_log_file = open(timing_log_path, 'w', newline="")
    csv_writer = csv.writer(chunks_log_file)
    csv_writer.writerow([
        "epoch",
        "chunk_idx",
        "chunk_load_time",
        "chunk_train_time",
        "chunk_loss",
        "avg_batch_train_time",
        "max_batch_train_time",
        "min_batch_train_time",
        "avg_batch_creation_time"
    
    ])

    #prepare for multithreaded pre-loading of chunks
    preload_queue = Queue(maxsize=chunks_preloaded)
    chunk_queue= Queue()
    def preload_worker():
        while True:
            chunk_idx = chunk_queue.get()

            chunk_start_time = time.time()
            counts_file, metadata_file = chunks_dataset[chunk_idx]
            dataloader = create_dataloader(data_dir=data_dir, counts_file=counts_file, metadata_file=metadata_file, batch_size=batch_size)
            chunk_load_time = time.time() - chunk_start_time
            
            preload_queue.put((chunk_idx, dataloader, chunk_load_time)) # blocks if Q full 
            chunk_queue.task_done()


    for _ in range(num_preloader_threads):
        t = threading.Thread(target= preload_worker)
        t.daemon=True
        t.start()


    print(f"Time until First Epoch reached: {time.time() - main_start_time:.2f}s")
    for epoch in range(epochs):
        epoch_start_time = time.time()

        #populate Q with chunks
        for i in range(len(chunks_dataset)):
            chunk_queue.put(i)

        epoch_total_loss = 0.0
        num_batches_in_epoch = 0
        chunk_load_times=[]

        for chunk_idx in range(len(chunks_dataset)):
            chunk_train_time_start = time.time()
            loaded_chunk_idx, current_loader, chunk_load_time = preload_queue.get()
            chunk_load_times.append(chunk_load_time)

            assert loaded_chunk_idx == chunk_idx,  f"Expected chunk {chunk_idx}, got {loaded_chunk_idx}"

            #chunk's batches logging vars
            batch_train_times = []
            batch_creation_times = []
            chunk_total_loss = 0.0
            num_batches_in_chunk=0

            #Training step 
            batch_creation_start_time = time.time()
            for batch_idx, batch in enumerate(current_loader):
                batch_creation_time  = time.time() - batch_creation_start_time
                batch = batch.to(device)

                batch_start_time = time.time()
                optimizer.zero_grad()
                outputs = model(batch)
                batch_loss = criterion(outputs, batch)
                batch_loss.backward()
                optimizer.step()
                batch_train_time = time.time() - batch_start_time

                #summate batch metrics
                chunk_total_loss += batch_loss.item()
                batch_train_times.append(batch_train_time)
                batch_creation_times.append(batch_creation_time)
                num_batches_in_chunk +=1
                batch_creation_start_time = time.time()

            #chunk stats
            chunk_train_time = time.time() - chunk_train_time_start
            avg_batch_train_time = np.mean(batch_train_times)
            max_batch_train_time = np.max(batch_train_times)
            min_batch_train_time = np.min(batch_train_times)
            avg_batch_creation_time = np.mean(batch_creation_times)
            chunk_avg_loss = chunk_total_loss / num_batches_in_chunk

            csv_writer.writerow([
                epoch,
                chunk_idx,
                chunk_load_time,
                chunk_train_time,
                chunk_avg_loss,
                avg_batch_train_time,
                max_batch_train_time,
                min_batch_train_time,
                avg_batch_creation_time
               
            ])

            tensorboard_writer.add_scalars(f"ChunkStats/Epoch_{epoch}", {
                "ChunkLoss": chunk_avg_loss,
                "AvgBatchTrainTime": avg_batch_train_time,
                "AvgBatchCreationTime": avg_batch_creation_time,
            }, global_step=chunk_idx + epoch * len(chunks_dataset))


            epoch_total_loss += chunk_total_loss
            num_batches_in_epoch += num_batches_in_chunk

        #epoch stats
        epoch_time = time.time() - epoch_start_time
        epoch_avg_loss = epoch_total_loss / num_batches_in_epoch
        
        #Tensorboard Epoch Summary
        tensorboard_writer.add_scalars("EpochSummary", {
            "EpochLoss": epoch_avg_loss,
            "EpochTime": epoch_time,
            "MinChunkLoadTime": min(chunk_load_times),
            "MaxChunkLoadTime": max(chunk_load_times),
            "AvgChunkLoadTime": sum(chunk_load_times) / len(chunk_load_times),
        }, epoch)
        tensorboard_writer.add_histogram("BatchTrainTimes", torch.tensor(batch_train_times), epoch)
        tensorboard_writer.add_histogram("LastChunk/BatchCreationTimes", torch.tensor(batch_creation_times), epoch)

        # TensorBoard: text summary (replaces print)
        summary_text = (
            f"Epoch {epoch+1}/{epochs}\n"
            f"Avg Epoch Loss: {epoch_avg_loss:.6f}\n"
            f"Epoch Time: {epoch_time:.2f}s\n"
            f"Chunk Load Times: min={min(chunk_load_times):.3f}s, "
            f"max={max(chunk_load_times):.3f}s, avg={sum(chunk_load_times)/len(chunk_load_times):.3f}s\n"
        )
        tensorboard_writer.add_text("EpochLogs", summary_text, epoch)

    # Save the model
    model_save_path = os.path.join(output_dir, "dae_model.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    tensorboard_writer.close()
    chunks_log_file.close()

if __name__ == "__main__":
    main()
