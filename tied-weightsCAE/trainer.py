#trainer.py 
import torch
import torch.optim as optim
from torch import Tensor
import time
from torch.utils.data import DataLoader
import torch.nn as nn
from CAE import CAE
from context_classifier import ContextClassifier
from iny_outy_dataloader import SingleChunkDataset, ChunksDataset
from typing import Dict, Tuple, List, Any
import random
from collections import defaultdict
import json
import logging_helpers as log


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
                classifier_latent_dim = 128,
                batch_workers = 1,
                batch_prefetch_factor=1
                ):
        t0_init = time.time()
        self.data_dir= data_dir
        self.expr_glob = expr_glob
        self.meta_glob = meta_glob
        self.field_specs_path = field_specs_path
        self.metadata_fields_vocabs_path = meta_fields_vocabs_path
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.latent_dim = latent_dim  
        self.classifier_latent_dim = classifier_latent_dim  
        self.batch_workers = batch_workers
        self.batch_prefetch_factor = batch_prefetch_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tbwriter = log.init_logging()
        
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
        self.model = CAE(input_dim, self.latent_dim, self.field_specs_dict).to(self.device, dtype=torch.float32)
        self.classifier = ContextClassifier(input_dim, self.classifier_latent_dim, self.field_specs_dict).to(self.device)
        self.generator_optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.classifier_optimizer = optim.Adam(self.classifier.parameters(), lr = self.learning_rate)

          
        self.outer_loader = DataLoader(
            chunks_dataset,
            batch_size=1,
            shuffle=True,
            num_workers=1,
            prefetch_factor=1,
            persistent_workers=True,
            pin_memory=(torch.cuda.is_available()),
            collate_fn=lambda batch: batch[0],  #unwraps List: [(csr, meta)] into Tuple: (csr, meta)
            )
        print("Succesfully created model, optim, and outer_laoder - Inside Trainer.__init__()")
        t1_init = time.time() - t0_init
        print(f"[Time Initialing]: {t1_init}, [Current Time]: {time.time()}")
    
    def train(self, num_epochs:int): #TODO: SOmething not right with GPU, check dcgm. 
        t0_train = time.time() #TODO add timing summaries in logging_helpers
        self.model.train()

        #Establish Global Logging
        self.chunks_trained_on = 0 

        for epoch in range(num_epochs):
            print(f"Beggining epoch: {epoch}")
            #Establish Epoch Logging
            self.sum_chunk_train_times=0.0
            self.epoch_loss_terms = defaultdict(float)
            self.epoch_classifier_loss = 0.0
            self.chunk_num_in_epoch = 0
            t0_epoch = time.time()  ###RESET LOGS
            
            #loop chunks
            for expr_csr_chunk, meta_chunk in self.outer_loader:
                #Establish Chunk Logging 
                self.chunk_num_in_epoch+=1 ###ITERATE LOGS
                self.chunks_trained_on+=1
                self.batch_num_in_chunk=0 ###RESET LOGs
                self.sum_batch_train_times=0.0
                self.chunk_loss_terms = defaultdict(float)
                t0_chunk = time.time()

                #Create Datalaoder to prelaod batches from Chunk
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
                #loop batches 
                for expr_batch, meta_batches in inner_loader:
                    t0_batch = time.time()
                    #Move to GPU (supposedly#TODO)
                    expr_batch = expr_batch.to(self.device, dtype=torch.float32, non_blocking=True)
                    meta_batches = {k: v.to(self.device, dtype=torch.float32, non_blocking=True)for k,v in meta_batches.items()}
                    
                    #Train Batch 
                    batch_loss_terms = self.train_on_batch(expr_batch, meta_batches)
                    #Compute Logs
                    self.sum_batch_train_times += time.time() - t0_batch
                    self.batch_num_in_chunk+=1
                    for k,v in batch_loss_terms.items():
                        self.chunk_loss_terms[k] += float(v) if torch.is_tensor(v) else float(v)
                    #IN SCOPE BATCH
                #IN SCOPE CHUNK
                log.per_chunk_loss(self.tbwriter, self.chunks_trained_on, dict(self.chunk_loss_terms))
                if "classif" in self.chunk_loss_terms:
                    log.per_chunk_classifier_loss(self.tbwriter,self.chunks_trained_on, float(self.chunk_loss_terms["classif"]))
                for k,v in self.chunk_loss_terms.items():
                    self.epoch_loss_terms[k] += float(v)
                self.sum_chunk_train_times += time.time() - t0_chunk
            #IN SCOPE EPOCH
            log.per_epoch_loss(self.tbwriter, epoch, dict(self.epoch_loss_terms))
            if "classif" in self.epoch_loss_terms:
                log.per_epoch_classifier_loss(self.tbwriter, epoch, float(self.epoch_loss_terms["classif"]))
        self.tbwriter.close()
    

    def train_on_batch(self, expr_batch: Tensor, meta_batches: Dict[str, Tensor]):
        #current_batchs_size = expr_batch.size()[0] #TODO: Should be 0 or 1???

        #Establish Metadata Contexts
        batch_s_context = {f: meta_batches[f] for f in self.model.used_fields}
        batch_t_context, changed_fields, t_as_idxs = self.trans_gen_protocol(batch_s_context)

        #Protect against Frozen Classifier
        self.classifier.eval()
        for p in self.classifier.parameters():
            p.requires_grad_(False)
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(True)

        #first_cycle
        expr_hat_s_to_t, integration_term_1  = self.model(expr_batch, batch_s_context, batch_t_context)
        
        #second_cycle
        expr_hat_t_to_s, integration_term_2 = self.model(expr_hat_s_to_t, batch_t_context, batch_s_context)

        #Gather Loss Terms 
        loss_terms = {   
            "recon": (recon_loss_final := nn.functional.mse_loss(expr_batch, expr_hat_t_to_s, reduction="mean")),
            "integ": (integration_loss := nn.functional.mse_loss(integration_term_1, integration_term_2, reduction="mean")),
            "adv": (adversarial_loss := self.get_adversarial_loss(expr_hat_s_to_t, t_as_idxs, changed_fields)),
            "aggreg": (aggregate_loss := adversarial_loss + recon_loss_final + integration_loss), #TODO add an orthogonality term?
            "classif": 0.0
        }

        #Generator Step
        self.generator_optimizer.zero_grad(set_to_none=True)
        aggregate_loss.backward()
        self.generator_optimizer.step()
        #self.nonneg_projecction_() #clamp? TODO:?
       
       #TODO: determin ideal strat for Classifier training. E.g. warmup + while <accuracy_thresh? every batch? etc. 
        #Classifier Step (supervised on Real Data)
        if self.should_train_classifier():
            self.classifier.train()
            for p in self.classifier.parameters():
                p.requires_grad_(True)

            logits_real = self.classifier(expr_batch.detach())
            s_as_idx = {f: torch.argmax(batch_s_context[f], dim=1).to(self.device, non_blocking=True).long() for f in self.model.used_fields} #conver onehot to int-idx for cross-entropy [0,0,1,0] -> 2, CE(logits, 2)
            classif_loss_term = [nn.functional.cross_entropy(logits_real[f], s_as_idx[f]) for f in self.model.used_fields]
            loss_c = torch.stack(classif_loss_term).mean()

            
            self.classifier_optimizer.zero_grad(set_to_none=True)
            loss_c.backward()
            self.classifier_optimizer.step()

            loss_terms["classif"] = float(loss_c.item())
        return loss_terms

    def trans_gen_protocol(self, source_context):
        # Pick a field to change
        field = random.choice(self.model.used_fields)
        device = source_context[field].device

        # Clone all fields to avoid in-place edits on caller's tensors
        target_context: Dict[str, Tensor] = {k: v.clone() for k, v in source_context.items()}

        t_as_idxs = {k: torch.argmax(target_context[k], dim =1)for k in self.model.used_fields} #Convert onehot to int-id's

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
        target_context[field] = nn.functional.one_hot(new_idx, num_classes=num_classes).to(torch.float32)
        t_as_idxs[field] = new_idx
        changed_fields = [field]

        return target_context, changed_fields, t_as_idxs
        
    def get_adversarial_loss(self, x_st:Tensor,  t_as_idxs:Dict[str,Tensor], changed_fields =List[str])-> Tensor:
            self.classifier.to(self.device)

            with torch.no_grad():
                for p in self.classifier.parameters():
                    p.requires_grad_(False)
            self.classifier.eval()
            logits_trans = self.classifier(x_st)
            adv_terms = []
            for f in changed_fields:
                adv_terms.append(nn.functional.cross_entropy(logits_trans[f], t_as_idxs[f])) # CE adds -log to stop gradient explosion slightly better than SM(1-P(t))
            adv_loss = torch.stack(adv_terms).mean() if adv_terms else x_st.new_zeros(())
            return adv_loss


    def should_train_classifier(self):
        return True
    
    #TODO: slight pretrain of classifier?
    def classifier_warmup():
        return 
        #
        # for 1 epoch
        #   CE(real_logits, unchanged meta)#