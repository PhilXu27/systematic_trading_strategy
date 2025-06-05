import numpy as np
import pandas as pd

class Standardizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """
        Compute mean and standard deviation in training sample.
        """
        self.mean = X.mean()
        self.std = X.std(ddof=0)

    def transform(self, X):
        """
        Standardize sample using computed mean and standard deviation
        """
        if self.mean is None or self.std is None:
            raise ValueError("Must fit the scaler before calling transform.")
        return (X - self.mean) / self.std
    
    def standardize(self, X, is_train=False):
        if is_train:
            self.fit(X)
        return self.transform(X)