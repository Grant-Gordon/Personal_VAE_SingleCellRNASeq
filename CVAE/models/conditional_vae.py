import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple
import utils.loss_helpers as Loss

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, metadata_fields_dict, vocab_dict, lambda_l2_penalty):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.metadata_fields_dict = metadata_fields_dict
        self.vocab_dict=vocab_dict
        self.lambda_l2_penalty = lambda_l2_penalty



        #Gene Expression Encoder (VAE)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)


        #Metadata Heads
        self.metadata_heads = nn.ModuleDict()
        for field, strategy in metadata_fields_dict.items():
            if strategy["type"] == "embedding":
                embedding_dim = strategy.get("embedding_dim", self.latent_dim )
                if embedding_dim != self.latent_dim:
                    print(f"WARNING: Embeding dim for {field} ({embedding_dim}) != latent_dim ({self.latent_dim}). Check config files")
                num_classes = len(vocab_dict[field])
                self.metadata_heads[field] = nn.Sequential(
                    nn.Embedding(num_classes, embedding_dim),
                    nn.Linear(embedding_dim, latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, latent_dim)
                )
            elif strategy["type"] == "onehot":
                num_classes = len(vocab_dict[field]) #TODO
                self.metadata_heads[field] = nn.Sequential(
                    nn.Linear(num_classes, latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, latent_dim)
                )
            elif strategy["type"] == "continuous":
                self.metadata_heads[field] = nn.Sequential(
                    nn.Linear(1, latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, latent_dim)
                )

        #Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        self.norm = nn.LayerNorm(latent_dim)

    def encode(self,x):
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu +eps * std
    
    def forward(self, expr:Tensor , metadata: Dict[str, Tensor]) -> Dict[str, Tensor] :
        """
        Args:
            expr: Tensor of shape [B, input_dim]
            metadata: Dict[str, Tensor] for each metadata field

        Returns:
            recon: reconstructed input
            mu, logvar: parameters of approximate posterior
            offsets_dict: per-field metadata offsets (B, latent_dim)
            z_star: metadata-conditioned latent vector
        """

        mu, logvar = self.encode(expr)
        z_expr = self.reparameterize(mu, logvar)

        #add metadata additive effects
        offsets = []
        offsets_dict={}

        for field, head in self.metadata_heads.items():
            value = metadata[field]
            strategy = self.metadata_fields_dict[field]["type"]
            
            if strategy == "embedding":
                offset = head(value)

            elif strategy == "onehot":
                onehot = F.one_hot(value, num_classes=len(self.vocab_dict[field])).float()
                offset = head(onehot)

            elif strategy == "continuous":
                offset = head(value.unsqueeze(1))
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            offsets.append(offset)
            offsets_dict[field] = offset
        
        z_star_raw = z_expr + sum(offsets) if offsets else z_expr
        z_star_normed = self.norm(z_star_raw)

        recon = self.decoder(z_star_normed)

        return{
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "offsets": offsets_dict,
            "z_expr": z_expr,
            "z_star_raw": z_star_raw

        }

    
    def compute_total_loss(
            self,
            recon:torch.Tensor,
            target: torch.Tensor,
            mu: torch.Tensor,
            logvar: torch.Tensor,
            metadata_offsets: Dict[str, torch.Tensor] = None,
            lambda_l2: float = 0.0,
    ) ->Dict[str, torch.Tensor]:
        recon_loss = Loss.mse_recon_loss(recon=recon, target=target)
        kl_loss = Loss.kl_divergence(mu=mu, logvar=logvar)
        l2_penalty = Loss.l2_metadata_penalty(metadata_offsets=metadata_offsets, lambda_l2=lambda_l2)

        total = recon_loss + kl_loss + l2_penalty
        return {
            "recon":recon_loss,
            "kl": kl_loss,
            "l2": l2_penalty,
            "total":total
        }

