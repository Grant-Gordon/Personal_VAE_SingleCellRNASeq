#trainer.py 
import torch
import torch.optim as optim
import os
import dataloader
from CAE import CAE

#############################
DATA_DIR=""
#Preprocessed metadata 
META_FIELDS_VOCABS=""    # { field_name: { value: idx, ... }, ... }
FIELD_SPECS=""           #[ FieldSpec(field=..., cardinality=..., using=..., non_null_fraction=...), ... ]
#Training
LEARNING_RATE=0.001
BATCH_SIZE=128
#Model
LATENT_DIM=128
##############################

class Trainer():
    def __init__(self):
        self.chunks_dataset = dataloader.ChunksDataset(DATA_DIR)
        #dynamically get input size from first chunk
        counts_file, metadata_file = self.chunks_dataset[0]
        first_loader = dataloader.create_dataloader( DATA_DIR, counts_file, metadata_file, META_FIELDS_VOCABS, FIELD_SPECS, batch_size=1)
        expr_dim = first_loader.dataset[0]['expr'].shape[0]

        self.model = CAE(expr_dim, LATENT_DIM, FIELD_SPECS, META_FIELDS_VOCABS)
        self.optimizer = optim.Adam(self.model.parameters(), lr = LEARNING_RATE)
    
    
    def train(self, num_epochs:int):
        self.model.train()
    
        for epoch in range(num_epochs):
            #dataloader stuff 
            for chunk_idx in range(len(self.chunks_dataset)):
                expr_file, metadata_file = self.chunks_dataset[chunk_idx]
                chunk_loader = dataloader.create_dataloader(DATA_DIR, expr_file, metadata_file, BATCH_SIZE, FIELD_SPECS)
                self.train_on_chunk(chunk_loader)
    
    def train_on_chunk(self, chunk_loader):
        for batch in chunk_loader:
            self.train_on_batch(batch)

    def train_on_batch(self, batch):
        self.optimizer.zero_grad()
        batch_expr = batch["expr"]
        batch_s_context = {f: batch["metadata"][f] for f in used_fields}

        batch_t_context = self.trans_gen_protocol(batch_s_context)

        #first_cycle
        expr_hat_s_to_t, integration_term_1  = self.model(batch_expr, batch_s_context, batch_t_context)
        
        #second_cycle
        expr_hat_t_to_s, integration_term_2 = self.model(expr_hat_s_to_t, batch_t_context, batch_s_context)

        adversarial_loss = self.get_adversarial_loss() #TODO figure out how to implement GAN
        
        recon_loss_final = torch.nn.functional.mse_loss(batch_expr, expr_hat_t_to_s, reduction="mean")
        integration_loss = torch.nn.functional.mse_loss(integration_term_1, integration_term_2, reduction="mean")

        aggregate_loss = adversarial_loss + recon_loss_final + integration_loss

        
        aggregate_loss.backward()
        self.optimizer.step()

    #TODO psuedo code for generating trans samples
    def trans_gen_protocol(source_context):
        field_changed = rand(num_fields)
        new_val = rand(field_cardinalites[field_changed])
        target_context = source_context
        target_context[field_changed] = META_FIELDS_VOCABS[new_val]
        return target_context
        