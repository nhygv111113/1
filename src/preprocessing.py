"""
Data preprocessing module
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.utils import print_section


class DataPreprocessor:
    """Handle data preprocessing"""
    
    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize preprocessor
        
        Args:
            test_size (float): Test set size
            random_state (int): Random seed
        """
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def preprocess(self, X, y):
        """
        Execute complete preprocessing pipeline
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print_section("Data Preprocessing")
        
        # Handle missing values
        X = self._handle_missing_values(X)
        
        # Handle outliers
        X = self._handle_outliers(X)
        
        # Split data
        self._split_data(X, y)
        
        # Scale features
        self._scale_features()
        
        print("✓ Preprocessing completed!")
        print(f"\nTraining set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _handle_missing_values(self, X):
        """
        Handle missing values
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values")
            # Fill missing values with mean
            X = X.fillna(X.mean())
            print("✓ Missing values filled with mean")
        else:
            print("✓ No missing values found")
        
        return X
    
    def _handle_outliers(self, X):
        """
        Handle outliers using IQR method
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            pd.DataFrame: Data with outliers handled
        """
        print("Handling outliers...")
        
        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Cap outliers
        X = X.clip(lower=lower_bound, upper=upper_bound)
        
        print("✓ Outliers handled (capped at IQR bounds)")
        
        return X
    
    def _split_data(self, X, y):
        """
        Split data into train and test sets
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        print(f"✓ Data split: {100*(1-self.test_size):.0f}% train, {100*self.test_size:.0f}% test")
    
    def _scale_features(self):
        """Scale features using StandardScaler"""
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print("✓ Features scaled using StandardScaler")


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Simple function to preprocess data
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Test set size
        random_state (int): Random seed
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    preprocessor = DataPreprocessor(test_size=test_size, random_state=random_state)
    return preprocessor.preprocess(X, y)