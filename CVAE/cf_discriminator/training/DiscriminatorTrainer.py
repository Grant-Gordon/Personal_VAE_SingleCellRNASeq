import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os
from cf_discriminator.utils.trans_protocol import generate_trans, track_trans
from cf_discriminator.model.discriminator_model import Discriminator

class DiscriminatorTrainer:
    def __init__(
        self, 
        config,
        live_cvae: Optional[nn.Module] = None,
        cvae_checkpoint_path: Optional[str] = None
    ) -> None:
        """
        Trainer for the discriminator model. Supports training alongside a live CVAE
        or using a frozen pretrained CVAE from disk.
        
        Args:
          config: Discriminator config dict loaded from yaml
        """


        #TODO
        self.config = config
        self.discr_model = config[""]
        self.vocab = config[""]
        self.metadata_classes = config[""]
        self.device = config[""]
        #load and configure model

        #if train on live CVAE
        if live_cvae is not None: 
            self.cvae= live_cvae.to(self.device)
            self.freeze_cvae = False
        #if training on a pretrained CVAE
        elif cvae_checkpoint_path:
            if not os.path.exists(cvae_checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {cvae_checkpoint_path}")
            checkpoint = torch.load(cvae_checkpoint_path, map_location=self.device)
            self.cvae = checkpoint["model"] #Assumes full model is saved under 'model' #TODO
            self.cvae.eval()

            for p in self.cvae.parameters():
                p.requires_grad = False
            self.freeze_cvae = True
        else:
            raise ValueError("Must provide either a live CVAE or a checkpoint path")
        
        self.fields_to_trans = config.get("fields_to_trans", list(self.metadata_classes.keys()))
        self.track_transitions_enabled = config.get("track_transitions", False)

    
    def train_on_batch(self, expr:torch.Tensor, metadata:dict[str, torch.Tensor]) -> dict:
        """
        Performs one discriminato training step on  abatch of cis and srans samples
        
        Returns optional metadata like loss and stransition stats for logging
        """

        cvae_batch_size = expr.size(0)
        expr = expr.to(self.device)
        metadata = {k: v.to(self.device) for k,v in metadata.items()}


        #######FORWARD#########
        with torch.no_grad if self.freeze_cvae else torch.enable_grad():
            gx_hat_cis = self.cvae(expr,metadata)

        ######GEN TRANS #########
        metadata_trans = generate_trans(metadata, self.fields_to_trans, self.vocab)

        with torch.no_grad():   
            gx_hat_trans = self.cvae(expr, metadata_trans).detach()

        ##############PREP LABELS##############
        gx_combined = torch.cat([gx_hat_cis, gx_hat_trans], dim = 0)
        labels_is_cis = torch.cat([torch.ones(cvae_batch_size, 1, device=self.device), torch.zeros(cvae_batch_size, 1, device=self.device)], dim=0)

        metadata_combined = {field: torch.cat([metadata[field],metadata_trans[field]], dim=0) for field in metadata}
        sample_origin_idx = torch.cat([torch.arange(cvae_batch_size), torch.arange(cvae_batch_size)], dim=0)

        #####SHUFFLE BATCH#############
        indices = torch.randperm(2 * cvae_batch_size, device=self.device)
        gx_combined = gx_combined[indices] #'advanced indexing' pytorch specific thing v useful
        labels_is_cis = labels_is_cis[indices]
        metadata_combined = {k : v[indices] for k,v in metadata_combined.items()}
        sample_origin_idx - sample_origin_idx[indices]

        ######Forward through discr##########

        disc_out = self.discr_model(gx_combined)

        total_loss = 0.0
        losses_by_field = {}


        for field in self.metadata_classes:
            val_logits = disc_out[field]["value"]
            cis_logits = disc_out[field]["cis"]
            
            val_targets = metadata_combined[field]
            cis_targets = labels_is_cis

            val_loss = F.cross_entropy(val_logits, val_targets)
            cis_loss = F.binary_cross_entropy_with_logits(cis_logits, cis_targets)


            total_loss += val_loss + cis_loss

            losses_by_field[field]={
                "value_loss":val_loss.item(),
                "cis_loss": cis_loss.item(),
            }

        ####optimizer step######
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        ####track trans############
        transition_stats = {}
        if self.track_transitions_enabled:
            transition_stats = track_trans(metadata, metadata_trans, labels_is_cis, origin_indices=sample_origin_idx)
    
        return {
            "total_loss": total_loss,
            "losses_by_field": losses_by_field,
            "transitions": transition_stats
        }
        
