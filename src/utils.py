"""
Utility functions for the project
"""

import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_config(config_path='config.yaml'):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(paths):
    """
    Create directories if they don't exist
    
    Args:
        paths (list): List of directory paths to create
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_model(model, path):
    """
    Save model to disk
    
    Args:
        model: Model object
        path (str): Path to save the model
    """
    import joblib
    joblib.dump(model, path)
    print(f"✓ Model saved to {path}")


def load_model(path):
    """
    Load model from disk
    
    Args:
        path (str): Path to model file
        
    Returns:
        Model object
    """
    import joblib
    return joblib.load(path)


def print_section(title):
    """
    Print formatted section title
    
    Args:
        title (str): Section title
    """
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def check_missing_values(df):
    """
    Check for missing values in dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Missing values report
    """
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_percent
    })
    
    return missing_df[missing_df['Missing Count'] > 0].sort_values(
        'Missing Count', ascending=False
    )


def get_data_statistics(df):
    """
    Get statistical summary of dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Statistics dictionary
    """
    stats = {
        'shape': df.shape,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
    }
    return stats