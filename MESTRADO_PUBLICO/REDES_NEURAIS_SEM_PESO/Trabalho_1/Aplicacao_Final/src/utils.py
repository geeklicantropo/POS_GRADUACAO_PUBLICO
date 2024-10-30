import logging
import sys
from torch.utils.data import Dataset
import h5py
import numpy as np

def setup_logging(log_file):
    """
    Sets up logging to output messages to both the console and a log file.

    Parameters:
        log_file (str): Path to the log file.
    """
    #Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    #Set the logging level and format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  #Log to file
            logging.StreamHandler(sys.stdout)  #Log to console
        ]
    )

class HDF5Dataset(Dataset):
    """
    Custom Dataset for loading data from an HDF5 file.

    This dataset loads data lazily from an HDF5 file, which is efficient for large datasets.

    Parameters:
        h5_file_path (str): Path to the HDF5 file.
    """
    def __init__(self, h5_file_path):
        #Open the HDF5 file
        self.h5_file = h5py.File(h5_file_path, 'r')
        #Load data and labels
        self.data = self.h5_file['data']
        self.labels = self.h5_file['labels']

    def __len__(self):
        #Return the number of samples
        return self.labels.shape[0]

    def __getitem__(self, idx):
        #Retrieve the sample at the given index
        data = self.data[idx]
        label = self.labels[idx]
        #Convert data and label to numpy arrays. Important to handle big loads
        data = np.array(data, dtype=np.float32)
        label = np.array(label, dtype=np.int64)
        return data, label
