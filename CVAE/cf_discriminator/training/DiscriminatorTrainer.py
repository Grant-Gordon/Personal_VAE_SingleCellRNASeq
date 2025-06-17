import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from cf_discriminator.utils.trans_protocol import generate_trans, track_trans
from cf_discriminator.model.discriminator_model import Discriminator
class DiscriminatorTrainer:
    def __init__(
        self, 
        config,
        live_cvae: Optional[nn.Module] = None,
    ) -> None:
        """
        Trainer for the discriminator model. Supports training alongside a live CVAE
        or using a frozen pretrained CVAE from disk.
        
        Args:
          config: Discriminator config dict loaded from yaml
        """
        self.config = config
        self.device = config["training"]["device"]
        self.freeze_cvae = config["training"]["freeze_cvae"]
        
        #CVAE live or pretrained
        if config["source_cvae"]["train_with_pretrained_cvae"]:
            self.cvae = torch.load(config["source_cvae"]["cvae_checkpoint_path"], map_location=self.device)
        else:
            self.cvae = live_cvae

        #Freeze weights
        if self.freeze_cvae:
            print("[DiscriminatorTrainer] Freezing CVAE weights.")
            for param in self.cvae.parameters():
                param.requires_grad = False


        self.expr_dim = self.cvae.input_dim
        self.vocab_dict = self.cvae.vocab_dict
        self.metadata_fields_dict = self.cvae.metadata_fields_dict

        self.metadata_classes_per_field = {field: len(v) for field, v in self.vocab_dict.items()}        
        self.fields_to_trans = config["fields_to_change"]
        self.track_transitions_enabled = config["training"]["track_transitions"]

        self.discr_model = Discriminator(
            config=self.config,
            metadata_classes_per_field=self.metadata_classes_per_field,
            expr_dim=self.expr_dim
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.discr_model.parameters(), lr=self.config["training"]["learning_rate"])

    
    def train_on_batch(self, expr:torch.Tensor, metadata:dict[str, torch.Tensor]) -> dict:
        """
        Performs one discriminator training step on a batch of cis and trans samples
        
        Returns optional metadata like loss and transition stats for logging
        """

        cvae_batch_size = expr.size(0)
        expr = expr.to(self.device)
        metadata = {k: v.to(self.device) for k,v in metadata.items()}


        #######FORWARD#########
        with (torch.no_grad() if self.freeze_cvae else torch.enable_grad()):
            gx_hat_cis = self.cvae(expr,metadata)

        ######GEN TRANS #########
        metadata_trans = generate_trans(metadata, self.fields_to_trans, self.vocab_dict)

        with torch.no_grad():   
            gx_hat_trans = self.cvae(expr, metadata_trans).detach()

        ##############PREP LABELS##############
        gx_combined = torch.cat([gx_hat_cis, gx_hat_trans], dim = 0)
        labels_is_cis = torch.cat([torch.ones(cvae_batch_size, 1, device=self.device), torch.zeros(cvae_batch_size, 1, device=self.device)], dim=0)

        metadata_combined = {field: torch.cat([metadata[field],metadata_trans[field]], dim=0) for field in metadata}
        sample_origin_idx = torch.cat([torch.arange(cvae_batch_size), torch.arange(cvae_batch_size)], dim=0)

        #####SHUFFLE BATCH#############
        indices = torch.randperm(2 * cvae_batch_size, device=self.device)
        gx_combined = gx_combined[indices] #'advanced indexing' pytorch specific thing v useful
        labels_is_cis = labels_is_cis[indices]
        metadata_combined = {k : v[indices] for k,v in metadata_combined.items()}
        sample_origin_idx = sample_origin_idx[indices]

        ######Forward through discr##########

        disc_out = self.discr_model(gx_combined)

        total_loss = 0.0
        losses_by_field = {}


        for field in self.metadata_classes_per_field:
            val_logits = disc_out[field]["value"]
            cis_logits = disc_out[field]["cis"]
            
            val_targets = metadata_combined[field]
            cis_targets = labels_is_cis

            val_loss = F.cross_entropy(val_logits, val_targets)
            cis_loss = F.binary_cross_entropy_with_logits(cis_logits, cis_targets)


            total_loss += val_loss + cis_loss

            losses_by_field[field]={
                "value_loss":val_loss.item(),
                "cis_loss": cis_loss.item(),
            }

        ####optimizer step######
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        ####track trans############
        transition_stats = {}
        if self.track_transitions_enabled:
            transition_stats = track_trans(metadata, metadata_trans, labels_is_cis, origin_indices=sample_origin_idx)
    
        return {
            "total_loss": total_loss.item(),
            "losses_by_field": losses_by_field,
            "transitions": transition_stats
        }
            
    def train_using_pretrained_cvae(self) -> None:
        """
        Trains the discriminator using a pretrained (frozen) CVAE.
        Loads expression and metadata from disk chunk-by-chunk and calls train_on_batch.
        """
        from dataloader.loader import create_chunks_dataset
        from dataloader.preloader import start_preload_workers
        from queue import Queue
        import threading
        import time
        from utils.config_parser import parse_config

        cvae_config = parse_config(self.config["source_cvae"]["cvae_config_path"])

        self.chunks_dataset = create_chunks_dataset(
            cvae_config["data"]["data_dir"],
            cvae_config["data"]["species"]
        )
        self.preload_buffer = {}
        self.buffer_lock = threading.Lock()
        self.chunk_queue = Queue()

        start_preload_workers(
            dataset=self.chunks_dataset,
            data_dir=cvae_config["data"]["data_dir"],
            batch_size=cvae_config["training"]["batch_size"],
            num_threads=cvae_config["data"]["num_preloader_threads"],
            preload_buffer=self.preload_buffer,
            buffer_lock=self.buffer_lock,
            chunk_queue=self.chunk_queue,
            config=cvae_config
        )

        num_epochs = self.config["training"]["train_last_n_epochs"]

        for epoch in range(num_epochs):
            print(f"\n[DiscriminatorTrainer] Epoch {epoch + 1}/{num_epochs}")

            for chunk_idx in range(len(self.chunks_dataset)):
                start_time_waiting_for_chunk = time.time()
                while True:
                    with self.buffer_lock:
                        if chunk_idx in self.preload_buffer:
                            loader, chunk_load_time = self.preload_buffer.pop(chunk_idx)
                            break
                    if time.time() - start_time_waiting_for_chunk > 60:
                        raise TimeoutError(f"Chunk {chunk_idx} not found in buffer after 60 seconds.")
                    time.sleep(0.1)

                print(f"  [Chunk {chunk_idx}] Loaded in {chunk_load_time:.2f}s")

                for batch_idx, batch in enumerate(loader):
                    expr = batch["expr"]
                    metadata = batch["metadata"]

                    log_dict = self.train_on_batch(expr=expr, metadata=metadata)

                    total_loss = log_dict["total_loss"]
                    print(f"    [Epoch {epoch+1}, Chunk {chunk_idx}, Batch {batch_idx}] Total Loss: {total_loss:.4f}")
