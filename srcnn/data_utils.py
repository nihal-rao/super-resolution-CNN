import h5py
import numpy as np
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    """
    Holds the training data
    """
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['data'][idx], f['target'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])


class EvalDataset(Dataset):
    """
    Holds evaluation data
    """
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return f['data'][idx], f['target'][idx]

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['data'])