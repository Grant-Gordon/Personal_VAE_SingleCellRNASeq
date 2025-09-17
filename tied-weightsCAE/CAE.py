#CAE.py
import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import scipy.sparse as sp
from typing import Dict, Tuple

class CAE(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int, field_specs:Dict[Dict[str, int]]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.field_specs = field_specs
        #Encoder
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)

        #Metadata per-field-head
        self.meta_heads = nn.ModuleDict({field: nn.Linear(field["cardinality"], latent_dim) for field in field_specs}) #TODO: Should be using one hot for all fields regardless of cardinality. Should not be using embeddings. 
      
    
    def forward(self, expr:Tensor, source_context, target_context) -> Tuple[Tensor, Tensor]:
        h = self.encoder(expr)   #h = X W^TS


        os_list = []
        ot_list = []
        for field in self.field_specs.keys():
            os_list.append(self.meta_heads[field](source_context[field])) 
            ot_list.append(self.meta_heads[field](target_context[field])) 
        os = torch.stack(os_list, dim=0).sum(dim=0) if os_list else torch.zeros_like(h)
        ot = torch.stack(ot_list, dim=0).sum(dim=0) if ot_list else torch.zeros_like(h)
        h_tilde = torch.relu(h - os) # integration loss = first_cycle_(hg-os) - second_cycle(hg-os). Need to cache h - os
        z = h_tilde + ot  # z = h - os + ot #TODO add RELU clipping to ensure nonneg

        expr_hat = torch.matmul(z, self.encoder.weight) # X_hat = (h - os + ot)W
        recon_loss = nn.functional.mse_loss(expr_hat, expr, reduction="mean")


        return expr_hat, h_tilde 


    


