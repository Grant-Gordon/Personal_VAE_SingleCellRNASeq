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
##################LOSS LOGS############################
def per_chunk_raw_loss(
        writer: SummaryWriter,
        chunks_trained_on: int,
        chunk_loss_terms: Dict[str, float]
    ) -> None:
    """
    Logs a summary of total, reconstruction, integration, and Adverserial loss at the end of each chunk.
    """
    writer.add_scalars(f"loss_raw/chunk_raw_loss", {
        'total': chunk_loss_terms["aggreg"],
        'recon': chunk_loss_terms["recon"],
        'integ': chunk_loss_terms["integ"],
        'adv': chunk_loss_terms["adv"] 
    }, chunks_trained_on)


def per_chunk_normed_loss(
        writer: SummaryWriter,
        chunks_trained_on: int,
        chunk_loss_terms: Dict[str, float]
    ) -> None:
    """
    Logs a summary of total, reconstruction, integration, and Adverserial loss at the end of each chunk.
    """
    writer.add_scalars(f"loss_normed/chunk_raw_loss", {
        'total': chunk_loss_terms["aggreg"],
        'recon': chunk_loss_terms["recon"],
        'integ': chunk_loss_terms["integ"],
        'adv': chunk_loss_terms["adv"] 
    }, chunks_trained_on)


def per_epoch_raw_loss(
        writer: SummaryWriter,
        epoch: int,
        epoch_loss_terms: Dict[str, float]
    ) -> None:
    """
    Logs a summary of total, reconstruction, integration, and Adverserial loss at the end of each chunk.
    """
    writer.add_scalars(f"loss_raw/epoch_loss", {
        'total': epoch_loss_terms["aggreg"],
        'recon': epoch_loss_terms["recon"],
        'integ': epoch_loss_terms["integ"],
        'adv': epoch_loss_terms["adv"] 
    }, epoch)

def per_epoch_normed_loss(
        writer: SummaryWriter,
        epoch: int,
        epoch_loss_terms: Dict[str, float]
    ) -> None:
    """
    Logs a summary of total, reconstruction, integration, and Adverserial loss at the end of each chunk.
    """
    writer.add_scalars(f"loss_normed/epoch_loss", {
        'total': epoch_loss_terms["aggreg"],
        'recon': epoch_loss_terms["recon"],
        'integ': epoch_loss_terms["integ"],
        'adv': epoch_loss_terms["adv"] 
    }, epoch)

def per_chunk_classifier_loss(
        writer: SummaryWriter,
        chunks_trained_on: int,
        chunk_classifier_loss: float
    )->None:
    writer.add_scalar("loss_raw/chunk_classifier_loss", chunk_classifier_loss, chunks_trained_on)

def per_epoch_classifier_loss(
        writer: SummaryWriter,
        epoch: int,
        epoch_classifier_loss: float
    )->None:
    writer.add_scalar("loss_raw/epoch_classifier_loss", epoch_classifier_loss, epoch)

###################Gradient Logs #############################
def per_chunk_grad_norms(writer, chunks_trained_on, grad_ems):
    writer.add_scalars(f"grad/chunk_gradient_norms", {
            'mean': grad_ems.get_Mean(),
            'min': grad_ems.get_Min(),
            'max': grad_ems.get_Max(),
        }, chunks_trained_on)
    
#TODO: Make this a pyplot. 
###################Gradient Logs #############################
def log_metadata_influence(writer, metadata_ems):
    for field, ems in metadata_ems:
        writer.add_scalars(f"metadata_influence/{field}",{
            'mean': metadata_ems.get_Mean(),
            'min': metadata_ems.get_Min(),
            'max': metadata_ems.get_Max()  
            })
        

class Ems:
    def __init__(self, use_max=True, use_min=True, use_mean=True, use_mode=False, use_median=False):
        self.use_max = use_max
        self.use_min = use_min
        self.use_mean = use_mean
        self.use_mode = use_mode
        self.use_median = use_median 
        if use_max: self.Max = None
        if use_min: self.Min = None
        if use_mean: 
            self.Mean = None
            self.sum = None
            self.count =None
        if use_mode: 
            self.Mode = None
            self.occurances = {}
        if use_median: 
            self.Median=None
            self.elements = []

    def update(self, val):
        if val ==  None:
            return
        
        if self.use_max and (self.Max == None or val > self.Max):
            self.Max =val

        if self.use_min and (self.Min == None or val < self.Min):
            self.Min = val
        
        if self.use_mean:
            if self.sum is not None:
                self.sum+=val
            else:
                self.sum = val
            if self.count is not None:
                self.count+=1
            else:
                self.count=1

        if self.use_mode:
            if val in self.occurances:
                self.occurances[val]+=1
            else:
                self.occurances[val] = 1

        if self.use_median:
            self.elements.append(val)

    def get_Max(self):
        return self.Max
    def set_Max(self, max):
        self.Max = max
    def get_Min(self):
        return self.Min
    def set_Min(self, min):
        self.Min = min
    def get_Mean(self):
        if self.count != 0:
            return self.sum / self.count
        else: 
            return None

    def set_Mean(self, mean, sum, count):
        self.Mean = mean
        self.sum = sum
        self.count = count
    def get_Mode(self):
        key = max(self.occurances, key=self.occurances.get)
        return key, self.occurances[key]
    def set_Mode(self, mode, occurances):
        self.Mode = mode
        self.occurances = occurances
