import os
import torch
from typing import Dict, Tuple, List, Any
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt


TB_OUTPUT_PATH = "/mnt/projects/debruinz_project/grant_gordon/Personal_VAE_SingleCellRNASeq/tied-weightsCAE/job-outputs"

def init_logging():
    os.makedirs(TB_OUTPUT_PATH, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(TB_OUTPUT_PATH, "tensorboard_logs"))
    return writer 

def per_chunk_loss(
        writer: SummaryWriter,
        chunks_trained_on: int,
        chunk_loss_terms: Dict[str, float]
    ) -> None:
    """
    Logs a summary of total, reconstruction, integration, and Adverserial loss at the end of each chunk.
    """
    writer.add_scalars(f"loss/chunk_loss", {
        'total': chunk_loss_terms["aggreg"],
        'recon': chunk_loss_terms["recon"],
        'integ': chunk_loss_terms["integ "],
        'adv': chunk_loss_terms["adv"] 
    }, chunks_trained_on)


def per_epoch_loss(
        writer: SummaryWriter,
        epoch: int,
        epoch_loss_terms: Dict[str, float]
    ) -> None:
    """
    Logs a summary of total, reconstruction, integration, and Adverserial loss at the end of each chunk.
    """
    writer.add_scalars(f"loss/epoch_loss", {
        'total': epoch_loss_terms["aggreg"],
        'recon': epoch_loss_terms["recon"],
        'integ': epoch_loss_terms["integ "],
        'adv': epoch_loss_terms["adv"] 
    }, epoch)

def per_chunk_classifier_loss(
        writer: SummaryWriter,
        chunks_trained_on: int,
        chunk_classifier_loss: Dict[str, float]
    )->None:
    writer.add_scalars("loss/chunk_classifier_loss", chunk_classifier_loss, chunks_trained_on)

def per_epoch_classifier_loss(
        writer: SummaryWriter,
        epoch: int,
        epoch_classifier_loss: Dict[str, float]
    )->None:
    writer.add_scalars("loss/epoch_classifier_loss", epoch_classifier_loss, epoch)


    