"""
Custom samplers for efficient resume training.
"""
import torch
from torch.utils.data import Sampler
from typing import Iterator, Optional


class ResumableSampler(Sampler):
    """
    A sampler wrapper that can skip already-processed samples when resuming training.

    This sampler wraps another sampler (e.g., DistributedSampler or RandomSampler)
    and efficiently skips the first `start_index` samples without loading them.

    Args:
        base_sampler: The underlying sampler to wrap
        start_index: The index to start sampling from (in terms of samples, not batches)
    """

    def __init__(self, base_sampler: Sampler, start_index: int = 0):
        self.base_sampler = base_sampler
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        """Iterate over indices starting from start_index."""
        iterator = iter(self.base_sampler)

        # Skip the first start_index samples efficiently
        # Using itertools.islice would still create the skipped items
        # So we manually advance the iterator
        for _ in range(self.start_index):
            try:
                next(iterator)
            except StopIteration:
                # If start_index exceeds the sampler length, just return empty
                return

        # Yield remaining items
        for idx in iterator:
            yield idx

    def __len__(self) -> int:
        """Return the number of samples after skipping."""
        total_len = len(self.base_sampler)
        return max(0, total_len - self.start_index)

    def set_epoch(self, epoch: int):
        """Forward set_epoch call to base sampler if it has this method (e.g., DistributedSampler)."""
        if hasattr(self.base_sampler, 'set_epoch'):
            self.base_sampler.set_epoch(epoch)


class ResumableDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Extension of DistributedSampler that supports resuming from a specific sample index.

    This is more efficient than ResumableSampler wrapper for distributed training
    because it computes the indices to skip directly without iterating.

    Args:
        dataset: Dataset to sample from
        num_replicas: Number of processes participating in distributed training
        rank: Rank of the current process
        shuffle: Whether to shuffle the dataset
        seed: Random seed for shuffling
        start_index: Global sample index to start from (will be divided among ranks)
    """

    def __init__(self, dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = True,
                 seed: int = 0, start_index: int = 0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                        shuffle=shuffle, seed=seed)
        self.start_index = start_index

    def __iter__(self) -> Iterator[int]:
        """Iterate starting from start_index."""
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible (standard DDP behavior)
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * ((padding_size // len(indices)) + 1))[:padding_size]
        else:
            indices = indices[:self.total_size]

        assert len(indices) == self.total_size

        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # Skip already processed indices
        # start_index is global, so we need to convert to per-rank index
        per_rank_start = self.start_index // self.num_replicas
        if self.rank < (self.start_index % self.num_replicas):
            per_rank_start += 1

        indices = indices[per_rank_start:]

        return iter(indices)

    def __len__(self) -> int:
        """Return number of samples for this rank after skipping."""
        per_rank_start = self.start_index // self.num_replicas
        if self.rank < (self.start_index % self.num_replicas):
            per_rank_start += 1
        return max(0, self.num_samples - per_rank_start)
