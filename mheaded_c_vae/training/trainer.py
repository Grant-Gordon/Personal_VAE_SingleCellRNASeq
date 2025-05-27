import os
import time
import torch
import numpy as np
import pickle
from queue import Queue
from dataloader.preloader import start_preload_workers
import utils.logging_helpers as logging
from dataloader.loader import create_chunks_dataset, create_dataloader
from models.conditional_vae import ConditionalVAE
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, config):
        self.config = config
        self.current_epoch = -1
        self.global_timestep = -1

        self.device = torch.device(config['training'].get('device','cuda'))

        #init dataset and preload queeus
        self.chunks_dataset = create_chunks_dataset(
            config["data"]["data_dir"],
            config["data"]["species"]
        )
        self.preload_queue = Queue(maxsize=config["data"]["chunks_preloaded"])
        self.chunk_queue=Queue()

        start_preload_workers(
            self.chunks_dataset,
            config["data"]["data_dir"],
            config['training']['batch_size'],
            config['data']['num_preloader_threads'],
            self.preload_queue,
            self.chunk_queue,
            config # pass to workers to provide vocab to datasets 
        )

        #logging
        self.tbWriter, self.log_file, self.csv_writer = logging.init_logging(config["training"]["output_dir"])


        #dynamically get input size from first chunk
        counts_file, metadta_file = self.chunks_dataset[0]
        first_loader = create_dataloader(  
            config["data"]["data_dir"],
            counts_file,
            metadta_file,
            batch_size = 1,
            config=config
        )
        expr_dim = first_loader.dataset[0]['expr'].shape[0]


        #init model, optimizer, loss
        with open(config["metadata_vocab"], 'rb') as f:
            vocab_dict= pickle.load(f)

        self.model = ConditionalVAE(
            input_dim=expr_dim,
            latent_dim=config["model"]["latent_dim"],
            metadata_fields_dict=config["metadata_fields"],
            vocab_dict=vocab_dict,
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr = config["training"]["lr"])


    def train(self):
        num_epochs = self.config['training']['epochs']
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.current_epoch = epoch
            #queue chunk idxs for the epoch
            for idx in range(len(self.chunks_dataset)):
                self.chunk_queue.put(idx)
            
            epoch_loss, epoch_kl, epoch_recon, batch_times, creation_times, chunk_load_times = [],[],[],[],[],[]

            for chunk_idx in range(len(self.chunks_dataset)):
                loaded_idx, loader, chunk_load_time = self.preload_queue.get()
                assert loaded_idx == chunk_idx, "loaded chunk and expected chunk idx do not match"
                chunk_load_times.append(chunk_load_time)
                
                self.global_timestep +=1
                chunk_loss, chunk_kl, chunk_recon, bt, ct = self.train_chunk(loader)
                epoch_loss.append(chunk_loss)
                epoch_kl.append(chunk_kl)
                epoch_recon.append(chunk_recon)
                batch_times.extend(bt)
                creation_times.extend(ct)

                self.log_chunk_stats(epoch, chunk_idx, chunk_loss, chunk_kl, chunk_recon, bt, ct, chunk_load_time)

            self.log_epoch_stats(epoch, epoch_loss, epoch_kl, epoch_recon, batch_times, creation_times, chunk_load_times, epoch_start)

        self.save_model()
        self.tbWriter.close()
        self.log_file.close()

    def train_chunk(self, loader):
        self.model.train()
        total_loss, total_kl, total_recon = 0,0,0
        batch_times, creation_times = [],[]
        start = time.time()

        for batch in loader:
            creation_times.append(time.time() - start)
            expr_batch = batch["expr"].to(self.device)
            metadata_batch = {k: v.to(self.device) for k, v in batch["metadata"].items()}

            batch_start_time = time.time()
            self.optimizer.zero_grad()
            reconstructed, mu, logvar, offsets_dict, z_star = self.model(expr_batch, metadata_batch)
            model_out = self.model(expr_batch, metadata_batch)
            reconstructed = model_out["recon"]
            mu = model_out["mu"]
            logvar = model_out["logvar"]
            offsets_dict = model_out["offsets"]
            z_star = model_out["z_star"]
            z = model_out["z"]


            loss, recon_loss, kl = self.model.vae_loss(reconstructed, expr_batch, mu, logvar)
            loss.backward()
            self.optimizer.step()
            batch_times.append(time.time() - batch_start_time)

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl.item()
            start = time.time()
        
        logging.log_latent_distributions(writer=self.tbWriter, mu=mu, logvar=logvar, global_timestep=self.global_timestep)
        logging.log_metadata_head_offsets_norms(writer=self.tbWriter, metadata_offsets_dict=offsets_dict, z_star=z_star,  global_timestep=self.global_timestep)
        logging.log_z_baseline_norm(writer=self.tbWriter, z=z, global_timesetp=self.global_timestep)
        logging.log_z_shift_from_metadata(writer=self.tbWriter,z=z, z_star=z_star, global_timestep=self.global_timestep)
        n = len(loader)
        return total_loss/n, total_kl/n, total_recon/n, batch_times, creation_times
    

    def log_chunk_stats(self, epoch, chunk_idx, loss, kl, recon, bt, ct, load_time):
        self.csv_writer.writerow([
            epoch, chunk_idx, load_time,
            sum(bt) + sum(ct), loss, recon, kl,
            np.mean(bt), np.max(bt), np.min(bt), np.mean(ct)
        ])
        self.tbWriter.add_scalars(f"Chunk/Epoch_{epoch}", {
            "Loss": loss, "KL": kl, "Recon": recon,
            "Avg_batch_time": np.mean(bt),
            "Avg_creation_time": np.mean(ct)
        }, global_step=chunk_idx)

    def log_epoch_stats(self, epoch, losses, kls, recons, bt, ct, chunk_times, start):
        self.tbWriter.add_scalars("Epoch", {
            "Loss": np.mean(losses),
            "KL": np.mean(kls),
            "Recon": np.mean(recons),
            "Time": time.time() - start,
            "Chunk_load_min": np.min(chunk_times),
            "Chunk_load_max": np.max(chunk_times),
            "Chunk_load_avg": np.mean(chunk_times)
        }, epoch)

        self.tbWriter.add_histogram("Batch_times", torch.tensor(bt), epoch)
        self.tbWriter.add_histogram("Creation_times", torch.tensor(ct), epoch)

    def save_model(self):
        path = os.path.join(self.config['training']['output_dir'], "vae_model.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
