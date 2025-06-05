from collections import defaultdict
import torch

class ChunkSparsityAccumulator:
    """
    Accumulates mean absolute offset magnitudes per metadata field across
    batches in a training chunk. Intended for logging sparsity histograms
    at chunk boundaries.
    """
     
    def __init__(self) -> None:
        
        self._sum_by_field: dict[str, torch.Tensor] = {}
        self._sample_count: int = 0

    def update(self, offsets_dict: dict[str, torch.Tensor]) -> None:
        """
        Update the accumulator with a new batch of metadata offsets.

        Args:
            offsets_dict (dict): {field_name: offset tensor (B x D)}
        """
        batch_size = next(iter(offsets_dict.values())).shape[0]
        self._sample_count += batch_size

        for field, offset in offsets_dict.items():
            assert offset.dtype.is_floating_point, "Assert Error: expected floating point"
            abs_sum = offset.detach().abs().sum(dim=0) # (D,)
            if field not in self._sum_by_field:
                self._sum_by_field[field] = abs_sum
            else:
                self._sum_by_field[field] += abs_sum
        
    
    def finalize(self) -> dict[str, torch.Tensor]:
        """
        Finalize and return the mean absolute offset per latent dimension
        for each metadata field.

        Returns:
            dict: {field_name: avg_abs_offset (D,)}
        """
        assert self._sample_count !=0, "No samples were added to the accumulator"

        return{field: total/ self._sample_count for field, total in self._sum_by_field.items()}

        
    
    def reset(self) -> None:
        """Clears all internal state."""
        self._sum_by_field.clear()
        self._sample_count = 0