import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
class DataPipeline:
    def __init__(self, window_size: int, horizon: int):
        self.window_size = window_size
        self.horizon = horizon
        self.min_val = None
        self.max_val = None
    def apply_cubic_spline(self, data: np.ndarray) -> np.ndarray:
        cleaned_data = data.copy()
        for col in range(cleaned_data.shape[1]):
            y = cleaned_data[:, col]
            x = np.arange(len(y))
            nan_mask = np.isnan(y)            
            if np.any(nan_mask):
                if np.all(nan_mask):
                    cleaned_data[:, col] = 0.5 
                else:
                    x_valid = x[~nan_mask]
                    y_valid = y[~nan_mask]                    
                    if len(x_valid) > 3:
                        cs = CubicSpline(x_valid, y_valid, bc_type='not-a-knot')
                        cleaned_data[nan_mask, col] = cs(x[nan_mask])
                    else:
                        df_temp = pd.DataFrame(y)
                        df_temp = df_temp.ffill().bfill() 
                        cleaned_data[:, col] = df_temp.to_numpy().flatten()                        
        return cleaned_data
    def fit_transform_minmax(self, data: np.ndarray) -> np.ndarray:
        self.min_val = np.min(data, axis=0)
        self.max_val = np.max(data, axis=0)        
        denom = np.where((self.max_val - self.min_val) == 0, 1.0, self.max_val - self.min_val)
        return (data - self.min_val) / denom
    def inverse_transform_minmax(self, normalized_data: np.ndarray) -> np.ndarray:
        if self.min_val is None or self.max_val is None:
            raise ValueError("executar fit_transform_minmax.")
        return normalized_data * (self.max_val - self.min_val) + self.min_val
    def construct_sliding_windows(self, data: np.ndarray):
        X, y = [], []
        for i in range(len(data) - self.window_size - self.horizon + 1):
            X.append(data[i : i + self.window_size])
            y.append(data[i + self.window_size + self.horizon - 1])
        return np.array(X), np.array(y)