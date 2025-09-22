#trainer.py 
import torch
import torch.optim as optim
from torch import Tensor
import os
from torch.utils.data import DataLoader
from CAE import CAE
from iny_outy_dataloader import SingleChunkDataset, ChunksDataset
from typing import Dict, Tuple, Any
import random
import json


class Trainer():
    def __init__(self,
                 data_dir="/mnt/projects/debruinz_project/july2024_census_data/subset",
                 expr_glob="human_counts_?.npz", #NOTE: Glob uses ? not * for hyphenated training
                 meta_glob="human_metadata_?.pkl", #NOTE: Glob uses ? not * for hyphenated training
                 field_specs_path="./metadata_vocab.json" ,
                 meta_fields_vocabs_path="./metadata_field_specs.json",
                 learning_rate=0.001,
                 batch_size = 128,
                 latent_dim = 128,
                 batch_workers = 0,
                 batch_prefetch_factor=0
                 ):
        self.data_dir= data_dir
        self.expr_glob = expr_glob
        self.meta_glob = meta_glob
        self.field_specs_path = field_specs_path
        self.metadata_fields_vocabs_path = meta_fields_vocabs_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_dim = latent_dim    
        self.batch_workers = batch_workers
        self.batch_prefetch_factor = batch_prefetch_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Load in JSONs
        with open (self.field_specs_path) as f:
            self.field_specs_dict = json.load(f)
        with open(meta_fields_vocabs_path) as f:
            self.metadata_fields_vocabs = json.load(f)
        print(f"Successfully loaded metadata JSON files - Inside Trainer.__init__()")

        #Dataloader for Chunks 
        chunks_dataset = ChunksDataset(self.data_dir, meta_glob_pattern=self.meta_glob, gene_expr_glob_pattern=self.expr_glob)
        assert len(chunks_dataset) > 0, "No chunks found, Dataset empty"
        first_csr, _ = chunks_dataset[0]
        input_dim = int(first_csr.shape[1])


        #Model and Optimizer
        self.model = CAE(input_dim, self.latent_dim, self.field_specs_dict).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)


        self.outer_loader = DataLoader(
            chunks_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=False,
            pin_memory=(self.device == "cuda"),
            collate_fn=lambda batch: batch[0],  #unwraps List: [(csr, meta)] into Tuple: (csr, meta)
            )
        print("Succesfully created model, optim, and outer_laoder - Inside Trainer.__init__()")

    
    def train(self, num_epochs:int):
        self.model.train()
        self.epoch_loss = 0.0
        self.chunks_trained_on = 0
        for epoch in range(num_epochs):
            print(f"Beggining epoch: {epoch}")
            #loop chunks
            self.chunk_num_in_epoch = 0
            for expr_csr_chunk, meta_chunk in self.outer_loader:
                self.chunk_num_in_epoch+=1
                self.chunks_trained_on+=1
                inner_dataset =  SingleChunkDataset((expr_csr_chunk, meta_chunk), field_specs=self.field_specs_dict, field_value_map=self.metadata_fields_vocabs)
                inner_loader = DataLoader(
                    dataset=inner_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.batch_workers,
                    prefetch_factor=self.batch_prefetch_factor,
                    pin_memory=(self.device.type == "cuda"),
                    drop_last=False
                )
                
                self.chunk_loss = 0.0
                #loop batches 
                for expr_batch, meta_batches in inner_loader:
                    expr_batch = expr_batch.to(self.device, non_blocking=True)
                    meta_batches = {k: v.to(self.device, non_blocking=True)for k,v in meta_batches.items()}
                    
                    self.train_on_batch(expr_batch, meta_batches)
                print(f"Chunk Loss on chunk {self.chunk_num_in_epoch}: {self.chunk_loss}")


    
        

    def train_on_batch(self, expr_batch: Tensor, meta_batches: Dict[str, Tensor]):
        self.optimizer.zero_grad()
        batch_s_context = {f: meta_batches[f] for f in self.model.used_fields}

        batch_t_context = self.trans_gen_protocol(batch_s_context)

        #first_cycle
        expr_hat_s_to_t, integration_term_1  = self.model(expr_batch, batch_s_context, batch_t_context)
        
        #second_cycle
        expr_hat_t_to_s, integration_term_2 = self.model(expr_hat_s_to_t, batch_t_context, batch_s_context)

        adversarial_loss = self.get_adversarial_loss() #TODO figure out how to implement GAN
        
        recon_loss_final = torch.nn.functional.mse_loss(expr_batch, expr_hat_t_to_s, reduction="mean")
        integration_loss = torch.nn.functional.mse_loss(integration_term_1, integration_term_2, reduction="mean")

        aggregate_loss = adversarial_loss + recon_loss_final + integration_loss
        self.chunk_loss+= aggregate_loss
        self.epoch_loss+= aggregate_loss
        
        aggregate_loss.backward()
        self.optimizer.step()




    #TODO psuedo code for generating trans samples
    def trans_gen_protocol(self, source_context):
        # Pick a field to change
        field = random.choice(self.model.used_fields)
        device = source_context[field].device

        # Clone all fields to avoid in-place edits on caller's tensors
        target_context: Dict[str, Tensor] = {k: v.clone() for k, v in source_context.items()}

        # Cardinality and batch size
        num_classes = int(self.field_specs_dict[field].get("cardinality", 0))
        assert num_classes > 0, f"Field '{field}' must have positive cardinality"

        batch_size = target_context[field].shape[0]

        # Current indices via argmax (works for valid one-hots; all-zero rows map to 0)
        old_idx = target_context[field].argmax(dim=1)

        # Sample a shift in [1, num_classes-1] so new != old, then wrap
        shift = torch.randint(1, num_classes, (batch_size,), device=device)
        new_idx = (old_idx + shift) % num_classes  # guaranteed different from old_idx

        # One-hot -> float32
        target_context[field] = torch.nn.functional.one_hot(new_idx, num_classes=num_classes).to(torch.float32)

        return target_context
        
    #TODO: psuedo code for getting GAN loss
    def get_adversarial_loss(self):
        return 0
    