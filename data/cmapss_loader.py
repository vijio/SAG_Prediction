import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class CMAPSSDataLoader(Dataset):
    def __init__(self, file_path, sequence_length=50, split='train'):
        self.data = self._load_and_preprocess(file_path)
        self.sequence_length = sequence_length
        self.sensor_cols = [f'sensor_{i+1}' for i in range(21)]
        
    def _load_and_preprocess(self, file_path):
        raw_data = pd.read_csv(file_path, sep=' ', header=None)
        raw_data.drop(columns=[26, 27], inplace=True)  # Remove NaN columns
        columns = ['engine', 'cycle'] + [f'setting{i+1}' for i in range(3)] + self.sensor_cols
        raw_data.columns = columns
        return raw_data.groupby('engine').apply(self._add_rul)
    
    def _add_rul(self, group):
        group['RUL'] = group['cycle'].max() - group['cycle']
        return group
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        engine_data = self.data.iloc[idx]
        sequences = []
        for i in range(0, len(engine_data) - self.sequence_length + 1):
            seq = engine_data.iloc[i:i+self.sequence_length][self.sensor_cols]
            sequences.append(seq.values)
        return np.array(sequences, dtype=np.float32)