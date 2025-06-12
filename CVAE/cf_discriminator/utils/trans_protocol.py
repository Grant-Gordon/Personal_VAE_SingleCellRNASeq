import torch
import random
from typing import Dict, List, Tuple


def generate_trans(
        metadata: Dict[str, torch.Tensor],
        fields_to_change: List[str],
        vocab: Dict[str, List[str]],
)->Dict[str,torch.Tensor]:
    """
    Changes a subset of metadata fields per sample to simulate trans conditions.

    Args:
        metadata: Original metadata batch (field -> (B,) tensor)
        fields_to_change: Which metadata fields are eligible for mutation
        vocab: Full vocab list per field
        num_changed_fields: How many fi samples to mutate per field

    Returns:
        New metadata dict with the same shape as input, with values altered in specifc fields
    """
    changed_metadata = {}

    batch_size = next(iter(metadata.values())).size(0)
    
    for field in metadata:
        values = metadata[field].clone()

        if field in fields_to_change:
            vocab_size = len(vocab[field])
            for i in range(batch_size):
                current_val = values[i].item()
                candidates = [v for v in range(vocab_size) if v != current_val]
                values[i] = random.choice(candidates)


        changed_metadata[field] = values
    return changed_metadata


def track_trans(
        metadata: Dict[str, torch.Tensor],
        metadata_trans: Dict[str, torch.Tensor],
        is_cis: torch.Tensor,
        origin_indices: torch.Tensor
)-> Dict[str, List[Tuple[int, int]]]:
    """
    For trans samples (is_cis == 0), returns list of (original, mutated) transitions per field.

    Args:
        metadata: Original (cis) metadata
        metadata_trans: Mutated (trans) metadata
        is_cis: Tensor of shape (2B,) with 1 = cis, 0 = trans

    Returns:
        dict mapping each metadata field to a list of (original, mutated) tuples
    """

    batch_size = is_cis.size(0) // 2
    transitions = {field: [] for field in metadata}

    for i in range( 2 * batch_size):
        if is_cis[i].item() ==0:
            idx=origin_indices[i].itme()
            for field in metadata:
                original = metadata[field][idx].item()
                changed = metadata_trans[field][idx].item()
                transitions[field].append((original, changed))
    return transitions