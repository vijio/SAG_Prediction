import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PHMDataLoader(Dataset):
    def __init__(self, root_dir, split='train', window_size=1000):
        self.root_dir = root_dir
        self.window_size = window_size
        self.metadata = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
        self.file_list = self._get_file_list(split)
        
    def _get_file_list(self, split):
        split_condition = self.metadata['split'] == split
        return self.metadata[split_condition]['file_id'].tolist()
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, f"{self.file_list[idx]}.csv")
        data = pd.read_csv(file_path).values
        return self._extract_features(data)
    
    def _extract_features(self, signal):
        features = [
            np.max(signal),
            np.sqrt(np.mean(np.square(signal))),
            np.mean(np.power(signal, 3)),
            np.mean(np.power(signal, 4)),
            np.mean(np.abs(signal)),
            np.max(signal) / np.mean(np.abs(signal)),
            np.mean(np.power(signal, 4)) / np.power(np.sqrt(np.mean(np.square(signal))), 4),
            np.max(signal) / np.sqrt(np.mean(np.square(signal))),
            np.max(signal) / np.square(np.mean(np.sqrt(np.abs(signal)))),
            np.mean(np.power(signal, 4)) / np.power(np.sqrt(np.mean(np.square(signal))), 4)
        ]
        return np.array(features, dtype=np.float32)