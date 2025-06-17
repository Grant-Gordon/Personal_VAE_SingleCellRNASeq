# discriminator/model/discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class Discriminator(nn.Module):
    def __init__(
        self,
        config: dict, #TODO match config 
        metadata_classes_per_field: Dict[str, int],
        expr_dim: int
    ) -> None:
        """
        Discriminator model for evaluating CVAE-generated samples.

        Args:
            config (dict): Model hyperparameters loaded from a YAML file.
            metadata_classes_per_field (dict): Mapping from metadata field names
                                               to number of unique classes.
        """
        super().__init__()

        input_dim = expr_dim
        hidden_dim = config["discriminator_architecture"]["hidden_dim"]
        dropout = config.get("discriminator_architecture", {}).get("dropout", 0.0)
        activation_fn = self._get_activation(config.get("discriminator_architecture", {}).get("activation", "relu"))

        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation_fn,
            nn.Dropout(dropout)
        )

        self.metadata_value_heads = nn.ModuleDict()
        self.metadata_cis_heads = nn.ModuleDict()

        for field_name, num_classes in metadata_classes_per_field.items():
            self.metadata_value_heads[field_name] = nn.Sequential(
                nn.Linear(hidden_dim, num_classes)
            )
            self.metadata_cis_heads[field_name] = nn.Sequential(
                nn.Linear(hidden_dim, 1)  # Binary classifier for cis/trans
            )

    def forward(self, gx_hat: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass.

        Args:
            gx_hat (torch.Tensor): Generated gene expression sample (B, input_dim)

        Returns:
            dict[str, dict[str, torch.Tensor]]: Nested dict with outputs for each metadata field.
                Outer keys are field names.
                Inner keys are 'value' and 'cis' corresponding to the classifier outputs.
        """
        features = self.shared_encoder(gx_hat)
        output = {}

        for field_name in self.metadata_value_heads:
            logits_value = self.metadata_value_heads[field_name](features)
            logits_cis = self.metadata_cis_heads[field_name](features)

            output[field_name] = {
                "value": logits_value,  # shape (B, num_classes)
                "cis": logits_cis       # shape (B, 1)
            }

        return output

    def _get_activation(self, name: str):
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        elif name == "gelu":
            return nn.GELU()
        elif name == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {name}")
