from __future__ import annotations

from torch.utils.data import Dataset


class BanglaWordDataset(Dataset):
    """Dataset wrapper for a curated collection of word samples."""

    def __init__(self, samples):
        self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
