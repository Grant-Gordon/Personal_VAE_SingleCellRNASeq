import torch
import torch.nn.functional as F
from typing import Dict


def mse_recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target, reduction="mean")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)


def l2_metadata_penalty(metadata_offsets: Dict[str, torch.Tensor], lambda_l2: float)-> torch.Tensor:
    if not metadata_offsets or lambda_l2 ==0:
        return torch.tensor(0.0, device=next(iter(metadata_offsets.values())).device)
    
    penalty = sum(torch.norm(offset, p=2, dim=1).pow(2).mean() for offset in metadata_offsets.values())
    return lambda_l2 * penalty

