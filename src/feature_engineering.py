"""
Feature engineering module
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from src.utils import print_section


class FeatureEngineer:
    """Handle feature engineering"""
    
    def __init__(self, n_features=10, scaling_method='standard'):
        """
        Initialize FeatureEngineer
        
        Args:
            n_features (int): Number of top features to select
            scaling_method (str): 'standard' or 'minmax'
        """
        self.n_features = n_features
        self.scaling_method = scaling_method
        self.scaler = None
        self.selector = None
        self.feature_names = None
    
    def fit_transform(self, X_train, X_test, y_train):
        """
        Fit and transform features
        
        Args:
            X_train (array-like): Training features
            X_test (array-like): Test features
            y_train (array-like): Training target
            
        Returns:
            tuple: (X_train_transformed, X_test_transformed)
        """
        print_section("Feature Engineering")
        
        # Create feature names if DataFrame
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
            X_train = X_train.values
            X_test = X_test.values
        
        # Feature selection
        X_train, X_test = self._select_features(X_train, X_test, y_train)
        
        # Scaling
        X_train, X_test = self._scale_features(X_train, X_test)
        
        return X_train, X_test
    
    def _select_features(self, X_train, X_test, y_train):
        """
        Select top K features based on SelectKBest
        
        Args:
            X_train (array): Training features
            X_test (array): Test features
            y_train (array): Training target
            
        Returns:
            tuple: (X_train_selected, X_test_selected)
        """
        print(f"Selecting top {self.n_features} features...")
        
        # Use SelectKBest with f_classif
        self.selector = SelectKBest(f_classif, k=min(self.n_features, X_train.shape[1]))
        X_train = self.selector.fit_transform(X_train, y_train)
        X_test = self.selector.transform(X_test)
        
        indices = self.selector.get_support(indices=True)
        
        print(f"✓ Selected {X_train.shape[1]} features")
        
        if self.feature_names:
            selected_names = [self.feature_names[i] for i in indices]
            print(f"Selected features: {', '.join(selected_names)}")
        
        return X_train, X_test
    
    def _scale_features(self, X_train, X_test):
        """
        Scale features
        
        Args:
            X_train (array): Training features
            X_test (array): Test features
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        print(f"✓ Features scaled using {self.scaling_method} scaler")
        
        return X_train, X_test