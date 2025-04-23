import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class XJTU_SYDataLoader(Dataset):
    """
    XJTU-SY Bearing Dataset Loader for run-to-failure vibration data
    Ref: 
    """
    
    def __init__(self, root_dir, bearing_id, axis='both', window_size=2560, stride=128):
        """
        Args:
            root_dir (str): Path to dataset directory
            bearing_id (str): Bearing identifier (e.g., '1_1', '2_3')
            axis (str): 'horizontal', 'vertical' or 'both'
            window_size (int): Sample window length (default 2560 ≈ 100ms @25.6kHz)
            stride (int): Sliding window step size
        """
        self.window_size = window_size
        self.stride = stride
        self.files = self._load_file_paths(root_dir, bearing_id)
        self.axis = axis
        self.samples = self._preprocess()

    def _load_file_paths(self, root_dir, bearing_id):
        bearing_path = os.path.join(root_dir, f"Bearing{bearing_id}")
        return sorted([os.path.join(bearing_path, f) 
                      for f in os.listdir(bearing_path) if f.endswith('.csv')],
                      key=lambda x: int(os.path.basename(x).split('.')[0]))

    def _preprocess(self):
        samples = []
        for file_path in self.files:
            data = pd.read_csv(file_path, header=None).values
            if self.axis == 'horizontal':
                signal = data[:, 0]
            elif self.axis == 'vertical':
                signal = data[:, 1]
            else:  # both axes
                signal = data.flatten()
            
            # Apply sliding window
            for i in range(0, len(signal)-self.window_size+1, self.stride):
                window = signal[i:i+self.window_size]
                samples.append(window)
        return np.array(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @staticmethod
    def get_default_split(root_dir, test_bearings=['1_3', '2_4', '3_3']):
        """
        Create train/test split based on 
        Default test set: 1 bearing from each operating condition
        """
        train_bearings = [f"{cond}_{i}" 
                         for cond in [1,2,3] 
                         for i in [1,2,3,4,5] 
                         if f"{cond}_{i}" not in test_bearings]
        
        return {
            'train': [XJTU_SYDataLoader(root_dir, b) for b in train_bearings],
            'test': [XJTU_SYDataLoader(root_dir, b) for b in test_bearings]
        }