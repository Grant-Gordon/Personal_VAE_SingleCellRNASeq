import os
import csv
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
