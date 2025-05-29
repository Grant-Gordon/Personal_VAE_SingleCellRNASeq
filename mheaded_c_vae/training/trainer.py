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
import threading


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
        self.preload_buffer= {}
        self.buffer_lock = threading.Lock()
        self.chunk_queue=Queue()

        start_preload_workers(
            dataset=self.chunks_dataset,
            data_dir=config["data"]["data_dir"],
            batch_size=config['training']['batch_size'],
            num_threads=config['data']['num_preloader_threads'],
            preload_buffer=self.preload_buffer,
            buffer_lock=self.buffer_lock,
            chunk_queue=self.chunk_queue,
            config=self.config # pass to workers to provide vocab to datasets 
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
        total_train_time_start = time.time()
        num_epochs = self.config['training']['epochs']
       
        for epoch in range(num_epochs):
            epoch_start = time.time()
            self.current_epoch = epoch
           
            #queue chunk idxs for the epoch
            for idx in range(len(self.chunks_dataset)):
                self.chunk_queue.put(idx)
            
            epoch_loss, epoch_kl, epoch_recon, batch_times_by_chunk, creation_times_by_chunk, chunk_load_times = [],[],[],[],[],[]

            for chunk_idx in range(len(self.chunks_dataset)):
                start_time_waiting_for_chunk = time.time()
                while True:
                    with self.buffer_lock:
                        if chunk_idx in self.preload_buffer:
                            loader, chunk_load_time = self.preload_buffer.pop(chunk_idx)
                            break
                    if time.time() - start_time_waiting_for_chunk > 60:
                        raise TimeoutError(f"Chunk {chunk_idx} not found in buffer ater 60 seconds.")
                    time.sleep(0.1)
                chunk_load_times.append(chunk_load_time)
                
                self.global_timestep +=1
                chunk_loss, chunk_kl, chunk_recon, bt, ct, model_out = self.train_chunk(loader)
                epoch_loss.append(chunk_loss)
                epoch_kl.append(chunk_kl)
                epoch_recon.append(chunk_recon)
                batch_times_by_chunk.append(bt)
                creation_times_by_chunk.append(ct)


                logging.log_chunk_loss_per_epoch(
                    writer=self.tbWriter,
                    epoch=self.current_epoch,
                    chunk_index=self.global_timestep,
                    loss_total=chunk_loss,
                    loss_recon=chunk_recon,
                    loss_kl=chunk_kl
                    )

            epoch_duration= time.time() - epoch_start

            logging.log_epoch_timing_text(
                writer=self.tbWriter,
                epoch=self.current_epoch,
                epoch_duration=epoch_duration,
                chunk_train_durations=[
                    sum(bt) + sum(ct) for bt, ct in zip(batch_times_by_chunk, creation_times_by_chunk) #TODO?
                ],
                chunk_load_durations=chunk_load_times
            )
            per_chunk_stats = []
            num_chunks = len(batch_times_by_chunk)
            for i in range(num_chunks):
                bt = batch_times_by_chunk[i]
                ct = creation_times_by_chunk[i]

                per_chunk_stats.append({
                    "batch_creation_min": min(ct),
                    "batch_creation_avg": sum(ct) / len(ct),
                    "batch_creation_max": max(ct),
                    "batch_train_min": min(bt),
                    "batch_train_avg": sum(bt) / len(bt),
                    "batch_train_max": max(bt),
                    "chunk_load_time": chunk_load_times[i],
                    "chunk_train_duration": sum(bt) + sum(ct),
                    "chunk_idx": chunk_idx
                })

            logging.log_chunk_timing_text_per_epoch(
                writer=self.tbWriter,
                epoch=self.current_epoch,
                per_chunk_stats=per_chunk_stats
            )

            logging.log_epoch_loss_summary(
                writer=self.tbWriter,
                epoch=self.current_epoch,
                loss_total=np.mean(epoch_loss),
                loss_recon=np.mean(epoch_recon),
                loss_kl=np.mean(epoch_kl)
            )
        total_train_time = time.time() - total_train_time_start

        logging.log_final_training_summary(
             writer=self.tbWriter,
            total_wall_time=total_train_time,
            final_loss_total=np.mean(epoch_loss[-3:]),
            final_loss_recon=np.mean(epoch_recon[-3:]),
            final_loss_kl=np.mean(epoch_kl[-3:]),
            final_metadata_offsets=model_out["offsets"], 
            final_z_star=model_out["z_star_raw"],
            global_step=self.global_timestep
            )

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
            model_out = self.model(expr_batch, metadata_batch)
            reconstructed = model_out["recon"]
            mu = model_out["mu"]
            logvar = model_out["logvar"]
            offsets_dict = model_out["offsets"]
            z_star = model_out["z_star_raw"]
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
        logging.log_z_baseline_norm(writer=self.tbWriter, z=z, global_timestep=self.global_timestep)
        logging.log_z_shift_from_metadata(writer=self.tbWriter,z=z, z_star=z_star, global_timestep=self.global_timestep)
        n = len(loader)
        return total_loss/n, total_kl/n, total_recon/n, batch_times, creation_times, model_out
    


    def save_model(self):
        path = os.path.join(self.config['training']['output_dir'], "vae_model.pth")
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
