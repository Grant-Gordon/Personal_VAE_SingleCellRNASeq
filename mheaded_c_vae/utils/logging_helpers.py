import os
import csv
import torch
from torch.utils.tensorboard import SummaryWriter

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

def log_metadata_head_offsets_norms(writer, metadata_offsets_dict, global_timestep):
  
  for field, offset in metadata_offsets_dict.items():
        if not offset.is_floating_point():
            print(f"WARNING: Skipping norm logging for field '{field}' — not a float tensor.") #TODO fix
            continue

        norms = torch.norm(offset.detach(), dim=1)  # shape: [batch_size]
        mean_norm = norms.mean().item()
        writer.add_scalar(f"metadata_head_norm/{field}", mean_norm, global_timestep)




