from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class DataPreprocessor:
    def __init__(self, scaler=StandardScaler()):
        self.scaler = scaler
        
    def sliding_window(self, data, window_size, step=1):
        windows = []
        for i in range(0, len(data) - window_size + 1, step):
            windows.append(data[i:i+window_size])
        return np.array(windows)
    
    def normalize(self, data):
        if len(data.shape) == 3:
            original_shape = data.shape
            data = data.reshape(-1, original_shape[-1])
            data = self.scaler.fit_transform(data)
            return data.reshape(original_shape)
        return self.scaler.fit_transform(data)
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        return train_test_split(X, y, test_size=test_size, random_state=random_state)