import torch
import numpy as np
from typing import Tuple, Iterator


class CustomDataLoader:
    def __init__(
        self,
        *tensors: torch.Tensor,
        batch_size: int = 32,
        drop_last: bool = False,
        shuffle: bool = False
    ):
        if tensors:
            assert all(t.size(0) == tensors[0].size(0) for t in tensors), "All tensors must have same length"
            self.tensors = tensors
            self.length = tensors[0].size(0)
        else:
            self.tensors = None
            self.length = 0
            
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, ...]]:
        indices = np.arange(self.length)

        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.length, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.length)
            
            if self.drop_last and end_idx - start_idx < self.batch_size:
                continue
                
            batch_indices = indices[start_idx:end_idx]
            batch = tuple(tensor[batch_indices] for tensor in self.tensors)
                
            yield batch
            
    def __len__(self) -> int:
        if self.drop_last:
            return self.length // self.batch_size
        else:
            return (self.length + self.batch_size - 1) // self.batch_size