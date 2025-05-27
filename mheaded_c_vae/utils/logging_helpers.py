import os
import csv
import torch
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
def init_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard_logs"))
    log_path = os.path.join(output_dir, "dataloader_timing_log.csv")

    log_file = open(log_path, 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "epoch",
        "chunk_idx",
        "chunk_load_time",
        "chunk_train_time",
        "chunk_loss(AVG)",
        "chunk_recon_loss(AVG)",
        "chunk_kl_loss(AVG)",
        "avg_batch_train_time",
        "max_batch_train_time",
        "min_batch_train_time",
        "avg_batch_creation_time"
    ])

    return writer, log_file, csv_writer
def log_latent_distributions(writer, mu, logvar, global_timestep):
    writer.add_scalar("latent/mu_mean", mu.mean().item(), global_timestep)
    writer.add_scalar("latent/mu_std", mu.std().item(), global_timestep)
    writer.add_scalar("latent/logvar_mean", logvar.mean().item(), global_timestep)
    writer.add_scalar("latent/logvar_std", logvar.std().item(), global_timestep)

def log_metadata_head_offsets_norms(
        writer: SummaryWriter, 
        metadata_offsets_dict: Dict[str, Tensor],
        z_star: Tensor,
        global_timestep: int
) -> None:
    z_star_norm = torch.norm(z_star.detach(), dim=1).mean().item()

    field_norms: Dict[str, float]= {}

    for field, offset in metadata_offsets_dict.items():
        norms = torch.norm(offset.detach(), dim=1)
        mean_norm = norms.mean().item()
        field_norms[field] = mean_norm
    
    total_offset_norm = sum(field_norms.values())
    for field, mean_norm in field_norms.items():
        rel_to_z = mean_norm / z_star_norm if z_star_norm != 0 else float('nan')
        rel_to_offsets = mean_norm/ total_offset_norm if total_offset_norm != 0 else float('nan')

        writer.add_scalar(f"metadata_head_norm/{field}_abs", mean_norm, global_timestep)
        writer.add_scalar(f"metadata_head_norm/{field}_rel_to_z", rel_to_z, global_timestep)
        writer.add_scalar(f"metadata_head_norm/{field}_rel_to_offsets", rel_to_offsets, global_timestep)

def log_z_baseline_norm(writer: SummaryWriter, z:Tensor, global_timestep:int)->None:
    """
    Logs the mean L2 norm of the baseline latent vector z (before metadata).
    """
    z_norm = torch.norm(z.detach(), dim=1).mean().item()
    writer.add_scalar("latent/z_norm_baseline", z_norm, global_timestep)

def log_z_shift_from_metadata(writer: SummaryWriter, z:Tensor, z_star: Tensor, global_timestep:int) -> None:
    """
    Logs the relative L2 shift from z to z_star: ‖z_star - z‖ / ‖z‖
    """
    shift = torch.norm((z_star - z).detach(), dim=1)
    base = torch.norm(z.detach(), dim=1)
    rel_shift = (shift / (base +1e-8)).mean().item()
    writer.add_scaler("latent/z)relative_shift", rel_shift, global_timestep)

