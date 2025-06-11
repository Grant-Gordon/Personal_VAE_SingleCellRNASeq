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
        self._gene_expr_sum: torch.Tensor = None

    def update(self, offsets_dict: dict[str, torch.Tensor], gene_expr_z: torch.Tensor) -> None:
        """
        Update the accumulator with a new batch of metadata offsets.

        Args:
            offsets_dict (dict): {field_name: offset tensor (B x D)}
        """
        batch_size = next(iter(offsets_dict.values())).shape[0]
        self._sample_count += batch_size

        for field, offset in offsets_dict.items():
            assert offset.dtype.is_floating_point, "Assert Error: expected floating point"
            abs_sum_meta = offset.detach().abs().sum(dim=0) # (D,)
            if field not in self._sum_by_field:
                self._sum_by_field[field] = abs_sum_meta
            else:
                self._sum_by_field[field] += abs_sum_meta
        
        assert gene_expr_z.dtype.is_floating_point, "Assert Error: expected floating point"
        abs_sum_gx = gene_expr_z.detach().abs().sum(dim=0) #(D,)
        
        if self._gene_expr_sum == None: #TODO confirm logic 
            self._gene_expr_sum = abs_sum_gx
        else:
            self._gene_expr_sum += abs_sum_gx
    
    def finalize(self) -> dict[str, torch.Tensor]:
        """
        Finalize and return the mean absolute offset per latent dimension
        for each metadata field.

        Returns:
            dict: {field_name: avg_abs_offset (D,)}
        """
        assert self._sample_count !=0, "No samples were added to the accumulator"

        return{field: total/ self._sample_count for field, total in self._sum_by_field.items()},  self._gene_expr_sum / self._sample_count

        
    
    def reset(self) -> None:
        """Clears all internal state."""
        self._sum_by_field.clear()
        self._sample_count = 0
        self._gene_expr_sum =None #TODO this might not be right