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
        self.field_context_head_pool: Dict[nn.ModuelDict]
        for field in self.used_fields:
            #shraed Enc/Dec's
            card = int(field_specs[field].get("cardinality", 0))
            assert card> 0, f"field: {field} must have cardinality greater than 0"
            self.shared_meta_encoders[field] = nn.Linear(input_dim, latent_dim, bias=False) 
            self.shared_meta_decoders[field] = nn.Linear(latent_dim, input_dim, bias=False) 
            #Per-context FFN head.
            for context in range(card):
                self.field_context_head_pool[field][context] = nn.Linear(latent_dim, latent_dim, bias=False)
            



    
    def forward(self, expr:Tensor, source_context: Dict[str, Tensor], target_context:Dict[str, Tensor]) -> Dict[str, Tensor]:
        hidden_encodings = {} # Cache these for integration loss terms later 
        head_logits = {} # Used to track the influence of heads Base & all fields 

        #Base encoder. Simple X W^TW
        hidden_encodings["base"] = self.base_encoder(expr)   #hx = X W^T
        head_logits["base"] = torch.matmul(hidden_encodings["base"], self.encoder.weights) # (XW^T)W

        for field in self.used_fields:
            #Enc With Source Metadata
            shared_encoding = self.shared_meta_encoders[field](expr)
            hidden_encodings[field] = self.field_context_head_pool[source_context[field]](shared_encoding) #h_field = X * C_shared_enc * Cs_FFN

            #Dec with Target Metadata
            decoded_context =  self.field_context_head_pool[target_context[field]](hidden_encodings[field])
            head_logits[field] = self.shared_meta_decoders[field](decoded_context) 
        
        #Combine Head Outputs and return 
        X_st = sum(head_logits.values())
        return {
            "X_st": X_st,   #Trans genreated transcriptome from contest s->t
            "hidden_encodings": hidden_encodings, #Cache hidden layers for integration loss later
            "metadata_logits": head_logits #Cache Base + Metadata logits for tracking of metadata influence on final epoch
        }


    


