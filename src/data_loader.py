"""
Data loading module for heart disease dataset
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import print_section


class DataLoader:
    """Data loader for UCI Heart Disease dataset"""
    
    def __init__(self, data_path='data/raw/heart.csv'):
        """
        Initialize DataLoader
        
        Args:
            data_path (str): Path to the data file
        """
        self.data_path = data_path
        self.data = None
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 
            'slope', 'ca', 'thal'
        ]
        self.target_name = 'target'
    
    def load(self):
        """
        Load data from CSV file
        
        Returns:
            tuple: (X, y) feature matrix and target vector
        """
        print_section("Loading Data")
        
        # Check if data file exists
        if not os.path.exists(self.data_path):
            print(f"Data file not found at {self.data_path}")
            print("Attempting to download from UCI repository...")
            self._download_data()
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        
        # If no column names, add them
        if self.data.shape[1] == 14:
            self.data.columns = self.feature_names + [self.target_name]
        
        print(f"✓ Data loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"\nFirst few rows:")
        print(self.data.head())
        
        # Separate features and target
        if self.target_name in self.data.columns:
            X = self.data.drop(self.target_name, axis=1)
            y = self.data[self.target_name]
        else:
            # Assume last column is target
            X = self.data.iloc[:, :-1]
            y = self.data.iloc[:, -1]
        
        # Convert binary target to 0 and 1
        y = (y > 0).astype(int)
        
        print(f"\nTarget distribution:")
        print(f"  Negative (0 - No Disease): {(y == 0).sum()}")
        print(f"  Positive (1 - Disease): {(y == 1).sum()}")
        
        return X, y
    
    def _download_data(self):
        """
        Download heart disease dataset from UCI repository
        """
        import urllib.request
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        
        # Create directory if not exists
        Path(self.data_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            print(f"Downloading from {url}")
            urllib.request.urlretrieve(url, self.data_path)
            print("✓ Data downloaded successfully!")
        except Exception as e:
            print(f"✗ Failed to download data: {e}")
            print("\nPlease download manually from:")
            print("https://archive.ics.uci.edu/ml/datasets/Heart+Disease")
            raise
    
    def get_info(self):
        """
        Get information about the dataset
        """
        if self.data is None:
            self.load()
        
        print_section("Dataset Information")
        print(self.data.info())
        print("\nStatistical Summary:")
        print(self.data.describe())


def load_data(data_path='data/raw/heart.csv'):
    """
    Simple function to load data
    
    Args:
        data_path (str): Path to data file
        
    Returns:
        tuple: (X, y) feature matrix and target vector
    """
    loader = DataLoader(data_path)
    return loader.load()