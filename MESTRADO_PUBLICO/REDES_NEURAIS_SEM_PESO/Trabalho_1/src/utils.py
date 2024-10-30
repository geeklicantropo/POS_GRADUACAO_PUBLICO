import logging
import h5py
import torch
from torch.utils.data import Dataset

def setup_logging(log_file):
    """
    Sets up logging configuration.
    """
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

class HDF5Dataset(Dataset):
    """
    Dataset for loading data from HDF5 files.
    """
    def __init__(self, h5_file_path):
        self.h5_file = h5py.File(h5_file_path, 'r')
        self.data = self.h5_file['data']
        self.labels = self.h5_file['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        binarized_data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = int(self.labels[idx])
        return binarized_data, label

    def __del__(self):
        self.h5_file.close()
