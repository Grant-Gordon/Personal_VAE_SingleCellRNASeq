#refactor_core.py
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
        #Encoder- tied-weihghts for dec
        self.base_encoder = nn.Linear(input_dim, latent_dim, bias=False)
        #TODO: implement Moore-Penrose iterative updates of a He initialization 
        self.used_fields = [f for f, spec in field_specs.items() if spec.get("using", False)]

        #Field Shared Enc/Dex + per-context Heads
        self.shared_meta_encoders = nn.ModuleDict()
        self.shared_meta_decoders = nn.ModuleDict()
        self.field_context_head_pool = nn.ModuleDict()
        for field in self.used_fields:
            #shraed Enc/Dec's
            card = int(field_specs[field].get("cardinality", 0))
            assert card> 0, f"field: {field} must have cardinality greater than 0"
            self.shared_meta_encoders[field] = nn.Linear(input_dim, latent_dim, bias=False) 
            self.shared_meta_decoders[field] = nn.Linear(latent_dim, input_dim, bias=False) 
           
            #Per-context FFN head.
            self.field_context_head_pool[field] = nn.ModuleDict()
            for context in range(card):
                self.field_context_head_pool[field][str(context)] = nn.Linear(latent_dim, latent_dim, bias=False)
        
        #init weights for all heads
        self._weight_init(self.base_encoder)
        for head in self.shared_meta_encoders.values():
            self._weight_init(head)
        for head in self.shared_meta_decoders.values():
            self._weight_init(head)
        for field_pool in self.field_context_head_pool.values():
            for context_head in field_pool.values():
                self._weight_init(context_head)



    
    def forward(self, expr:Tensor, source_context: Dict[str, Tensor], target_context:Dict[str, Tensor]) -> Dict[str, Tensor]:
        hidden_encodings = {} # Cache these for integration loss terms later 
        head_logits = {} # Used to track the influence of heads Base & all fields 

        #Base encoder. Simple X W^TW
        hidden_encodings["base"] = self.base_encoder(expr)   #hx = X W^T
        head_logits["base"] = torch.matmul(hidden_encodings["base"], self.base_encoder.weight) # (XW^T)W

        for field in self.used_fields:
            #Enc With Source Metadata
            shared_encoding = self.shared_meta_encoders[field](expr)
            hidden_encodings[field] = self.field_context_head_pool[field][str(source_context[field])](shared_encoding) #h_field = X * C_shared_enc * Cs_FFN

            #Dec with Target Metadata
            decoded_context =  self.field_context_head_pool[field][str(target_context[field])](hidden_encodings[field])
            head_logits[field] = self.shared_meta_decoders[field](decoded_context) 
        
        #Combine Head Outputs and return 
        X_st = torch.stack(list(head_logits.values()), dim=0,).sum(dim=0)
        return {
            "X_st": X_st,   #Trans genreated transcriptome from contest s->t
            "hidden_encodings": hidden_encodings, #Cache hidden layers for integration loss later
            "head_logits": head_logits #Cache Base + Metadata logits for tracking of metadata influence on final epoch
        }


    def _weight_init(module: nn.Module ):
        with torch.no_grad():
            #set weights to uniform dist
            W = module.weight
            W.uniform_(0.0, 1.0)
            #Sum rows (for denominator) calmped for divide-by-zero gaurd
            row_norms = torch.linalg.vector_norm(W, dim=1, keepdim=True).clamp_min(1e-12) #Dim = 1 for rows dim=0 for cols 
            #normalize Row In place by dividing by row_sums
            W.div_(row_norms)
            #DEBUG "Do all row L2s look like 1?"
            assert torch.allclose(torch.linalg.vector_norm(W, dim=1), torch.ones(W.size(0)), atol=1e-6)

