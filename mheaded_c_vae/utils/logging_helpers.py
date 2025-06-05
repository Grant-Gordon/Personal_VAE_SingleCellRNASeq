import os
import csv
import torch
from typing import Dict
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

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
    writer.add_scalar("latent/logvar_mean", logvar.mean().item(), global_timestep)


def log_expr_z_norm(writer: SummaryWriter, z_expr:Tensor, global_timestep:int)->None:
    """
    Logs the mean L2 norm of the baseline latent vector z (before metadata).
    """
    z_norm = torch.norm(z_expr.detach(), dim=1).mean().item()
    writer.add_scalar("norms/z/expr_z_l2", z_norm, global_timestep)

def log_z_shift_from_metadata(writer: SummaryWriter, z_expr:Tensor, z_star: Tensor, global_timestep:int) -> None:
    """
    Logs the relative L2 shift from z to z_star: ‖z_star - z‖ / ‖z‖
    """
    shift = torch.norm((z_star - z_expr).detach(), dim=1)
    base = torch.norm(z_expr.detach(), dim=1)
    rel_shift = (shift / (base +1e-8)).mean().item()
    writer.add_scalar("norms/z/relative_shift", rel_shift, global_timestep)




def log_epoch_loss_summary(
        writer: SummaryWriter,
        epoch: int,
        loss_total: float,
        loss_recon: float,
        loss_kl: float,
        l2_penalty:float
    ) -> None:
    """
    Logs a summary of total, reconstruction, and KL losses and metadata L2 penalty, at the end of each epoch.

    """
    writer.add_scalars(f"loss/epoch_level", {
        'total': loss_total,
        'recon': loss_recon,
        'kl': loss_kl,
        'l2_penalty': l2_penalty
    }, epoch)

def log_global_chunk_loss_summary(
        writer: SummaryWriter,
        global_timestep: int,
        loss_total: float,
        loss_recon: float,
        loss_kl: float,
        l2_penalty:float
    ) -> None:
    """
    Logs a summary of total, reconstruction, and KL losses and metadata L2 penalty, at the end of each chunk.

    """
    writer.add_scalars(f"loss/global_chunk_level", {
        'total': loss_total,
        'recon': loss_recon,
        'kl': loss_kl,
        'l2_penalty': l2_penalty
    }, global_timestep)


def log_chunk_loss_per_epoch(
    writer: SummaryWriter,
    epoch: int,
    chunk_index: int,
    loss_total: float,
    loss_recon: float,
    loss_kl: float
) -> None:
    """
    Logs per-chunk losses under a per-epoch TensorBoard group.

    This creates a separate graph per epoch, x-axis = chunk index (usually ~15 points).
    """
    writer.add_scalar(f"loss/per_epoch/epoch_{epoch}/total", loss_total, chunk_index)
    writer.add_scalar(f"loss/per_epoch/epoch_{epoch}/recon", loss_recon, chunk_index)
    writer.add_scalar(f"loss/per_epoch/epoch_{epoch}/kl", loss_kl, chunk_index)

def log_epoch_timing_text(
    writer: SummaryWriter,
    epoch: int,
    epoch_duration: float,
    chunk_train_durations: list[float],
    chunk_load_durations: list[float]
) -> None:
    """
    Logs a plain-text summary of timing stats for the epoch.
    Appears under TensorBoard's 'Text' tab.
    """
    summary = (
        f"Epoch {epoch} Timing Summary:\n"
        f"- Epoch Duration: {epoch_duration:.2f} sec\n"
        f"- Chunk Train Time: min={min(chunk_train_durations):.2f}, "
        f"avg={sum(chunk_train_durations)/len(chunk_train_durations):.2f}, "
        f"max={max(chunk_train_durations):.2f} sec\n"
        f"- Chunk Load Time: min={min(chunk_load_durations):.2f}, "
        f"avg={sum(chunk_load_durations)/len(chunk_load_durations):.2f}, "
        f"max={max(chunk_load_durations):.2f} sec"
    )

    print(summary)
    writer.add_text("timing/epoch_summary", summary, global_step=epoch)

def log_chunk_timing_text_per_epoch(
    writer: SummaryWriter,
    epoch: int,
    per_chunk_stats: list[dict[str, float]]
) -> None:
    """
    Logs a per-epoch text summary of per-chunk timing stats.
    Each dict should contain the following float entries for a single chunk:
      - batch_creation_min, batch_creation_avg, batch_creation_max
      - batch_train_min, batch_train_avg, batch_train_max
      - chunk_load_time
      - chunk_train_duration
    """
    lines = [f"Timing Summary for Epoch {epoch} (per chunk):"]
    for i, stats in enumerate(per_chunk_stats):
        lines.append(
            f"  Chunk {stats['chunk_idx']:02d} | "
            f"Create: {stats['batch_creation_avg']:.2f}s "
            f"Create Batch: min {stats['batch_creation_min']:.4f}s, ave {stats['batch_creation_avg']:.3f}s,  max {stats['batch_creation_max']:.3f}s | "
            f"Train Batch: min {stats['batch_train_min']:.4f}, ave {stats['batch_train_avg']:.3f}s, max {stats['batch_train_max']:.3f}s | "
            f"Load Chunk: {stats['chunk_load_time']:.2f}s | "
            f"Total: {stats['chunk_train_duration']:.2f}s"
        )

    summary = "\n".join(lines)
    print(summary)
    writer.add_text(f"timing/per_epoch/epoch_{epoch}", summary, global_step=epoch)


def log_metadata_head_offsets_norms(
    writer: SummaryWriter,
    metadata_offsets_dict: dict[str, torch.Tensor],
    z_expr: torch.Tensor,
    global_timestep: int
) -> None:
    """
    Logs absolute and relative L2 norms of each metadata head output.
    Appears under:
      - metadata_v_ossets/{field}
      - metadata_v_z_expr/{field}
    """
    z_expr_norm = torch.norm(z_expr.detach(), dim=1).mean().item()
    field_norms: dict[str, float] = {}

    for field, offset in metadata_offsets_dict.items():
        
        assert offset.dtype.is_floating_point, f"Assert Error, field '{field}' is not a float tensor"
      
        field_norm = torch.norm(offset.detach(), dim=1).mean().item()
        field_norms[field] = field_norm

    total_offset_norm = sum(field_norms.values())

    for field, norm in field_norms.items():
        rel_to_z_star = norm / z_expr_norm if z_expr_norm > 0 else float('nan')
        rel_to_offsets = norm / total_offset_norm if total_offset_norm > 0 else float('nan')

        writer.add_scalars(f"metadata/{field}",{
            'rel_z_star' : rel_to_z_star,
            'rel_offsets': rel_to_offsets

        }, global_timestep)

def log_final_training_summary(
    writer: SummaryWriter,
    total_wall_time: float,
    final_loss_total: float,
    final_loss_recon: float,
    final_loss_kl: float,
    final_l2_penalty: float,
    final_metadata_offsets: dict[str, torch.Tensor],
    final_z_expr: torch.Tensor,
    global_step: int
) -> None:
    """
    Logs a final training summary to TensorBoard (Text tab), including:
    - Total training wall time
    - Final loss values
    - Metadata field influence rankings (relative to z_star and to total offset magnitude)

    `total_wall_time` is the full duration (in seconds) from the start to the end of training.
    This includes all chunk loading, training, and overhead between epochs.
    """
    summary_lines = [
        f"Final Training Summary:",
        f"- Total Wall Time: {total_wall_time:.2f} sec",
        f"- Final Total Loss: {final_loss_total:.4f}",
        f"- Final Recon Loss: {final_loss_recon:.4f}",
        f"- Final KL Loss: {final_loss_kl:.4f}",
        f"- Final l2-Penalty:{final_l2_penalty:.4f}"
        ""
    ]
    z_expr_sq = (final_z_expr.detach() **2).sum(dim=1).mean().item()
    field_sq_norms: dict[str, float] = {
        field: (offset.detach() **2).sum(dim=1).mean().item()
        for field, offset in final_metadata_offsets.items()
        if offset.dtype.is_floating_point
    }
    total_offset_sq = sum(field_sq_norms.values())



    # Sort by contribution
    sorted_fields = sorted(field_sq_norms.items(), key=lambda x: x[1], reverse=True)

    summary_lines.append("- Metadata Influence Rankings (squared percent contributions):")
    for field, norm in sorted_fields:
        rel_to_z_expr = 100.0 * norm / z_expr_sq if z_expr_sq > 0 else float('nan')
        rel_to_offsets = 100.0 * norm / total_offset_sq if total_offset_sq > 0 else float('nan')
        summary_lines.append(
            f"  {field:32} | % of z*: {rel_to_z_expr:6.2f}% | % of total offset: {rel_to_offsets:6.2f}%"
        )

    summary = "\n".join(summary_lines)
    print(summary)
    writer.add_text("training/final_summary", summary, global_step=global_step)


def log_l2_penalty(
    writer: SummaryWriter,
    l2_penalty: torch.Tensor,
    total_loss: torch.Tensor,
    global_step: int,
) -> None:
    """
    Logs the L2 penalty and its fractional contribution to total loss.

    Args:
        writer: TensorBoard SummaryWriter instance.
        l2_penalty: Scalar tensor representing the metadata L2 penalty.
        total_loss: Scalar tensor for total loss (including recon, KL, etc.).
        global_step: Current global training step.
    """
    writer.add_scalar("loss/l2_penalty", l2_penalty.item(), global_step)

    # Avoid divide-by-zero just in case
    if total_loss.item() > 0:
        frac = l2_penalty.item() / total_loss.item()
        writer.add_scalar("loss/l2_frac_of_total", frac, global_step)

# def log_metadata_field_sparsity(
#     writer: SummaryWriter,
#     avg_abs_offsets_dict: dict[str, torch.Tensor],
#     global_timestep: int
# ) -> None:
#     """
#     Logs per-field sparsity histograms showing how strongly each metadata field
#     affects each latent dimension. Assumes input has been averaged over the chunk.

#     Each histogram:
#       - x-axis: latent dimension index (1 to D)
#       - y-axis: mean absolute offset magnitude

#     Args:
#         writer (SummaryWriter): TensorBoard writer
#         avg_abs_offsets_dict (dict): {field_name: (D,) tensor of averaged |offset|}
#         global_timestep (int): Global step (typically end of last chunk in epoch)
#     """
#     for field, avg_abs in avg_abs_offsets_dict.items():
#         assert isinstance(avg_abs, torch.Tensor) or avg_abs.ndim != 1, "Assert Error: Expected avg_abs in avg_abs_offset_dict to be type torch.Tensor"

#         writer.add_histogram(
#             tag=f"metadata_sparsity_latent/{field}",
#             values=avg_abs,
#             global_step=global_timestep
#         )




def log_metadata_field_sparsity(
    writer: SummaryWriter,
    avg_abs_offsets_dict: dict[str, torch.Tensor],
    global_timestep: int
) -> None:
    """
    Logs a per-field bar chart of average absolute offsets per latent dimension.

    Each plot:
        - x-axis: latent dimension index (1 to D)
        - y-axis: avg absolute offset magnitude

    Args:
        writer (SummaryWriter): TensorBoard writer
        avg_abs_offsets_dict (dict): {field_name: (D,) tensor of averaged |offset|}
        global_timestep (int): Logging step (e.g., end of final chunk)
    """
    for field, avg_abs in avg_abs_offsets_dict.items():
        assert isinstance(avg_abs, torch.Tensor) and avg_abs.ndim == 1, \
            f"Expected 1D torch.Tensor for field '{field}'"

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(1, len(avg_abs) + 1), avg_abs.cpu().numpy())
        ax.set_title(f"Sparsity of '{field}' metadata offsets")
        ax.set_xlabel("Latent Dimension")
        ax.set_ylabel("Mean |offset| of Chunk")
        ax.set_xticks(range(0, len(avg_abs) + 1, max(1, len(avg_abs) // 8)))  # tick every ~16 dims

        writer.add_figure(f"metadata_sparsity_latent/{field}", fig, global_step=global_timestep)
        plt.close(fig)  # cleanup to avoid memory leaks
