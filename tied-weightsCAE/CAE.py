#CAE.py
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, Tuple

class CAE(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int, field_specs:Dict[str, Dict[str, int]]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.field_specs = field_specs
        #Encoder
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)

        self.used_fields = [f for f, spec in field_specs.items() if spec.get("using", False)]

        #Metadata per-field-head
        self.meta_heads = nn.ModuleDict()
        for field in self.used_fields:
            card = int(field_specs[field].get("cardinality", 0))
            assert card> 0, f"field: {field} must have cardinality greater than 0"
            self.meta_heads[field] = nn.Linear(card, latent_dim, bias=False) #TODO: bias on metaheads?
            print(f"Instantiated meta_head for field: {field}.")
      
    
    def forward(self, expr:Tensor, source_context: Dict[str, Tensor], target_context:Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        h = self.encoder(expr)   #h = X W^TS


        # Sum per-field offsets
        if self.used_fields:
            os_list = [self.meta_heads[f](source_context[f]) for f in self.used_fields]  # each [B, latent_dim]
            ot_list = [self.meta_heads[f](target_context[f]) for f in self.used_fields]
            os = torch.stack(os_list, dim=0).sum(dim=0)  # [B, latent_dim]
            ot = torch.stack(ot_list, dim=0).sum(dim=0)  # [B, latent_dim]
        else:
            os = torch.zeros_like(h)
            ot = torch.zeros_like(h)

        h_tilde = torch.relu(h - os) # integration loss = first_cycle_(hg-os) - second_cycle(hg-os). Need to cache h - os
        z = h_tilde + ot  # z = h - os + ot #TODO add RELU clipping to ensure nonneg

        expr_hat = torch.matmul(z, self.encoder.weight) # X_hat = (h - os + ot)W
        #recon_loss = nn.functional.mse_loss(expr_hat, expr, reduction="mean")


        return expr_hat, h_tilde, os_list, ot_list


    


