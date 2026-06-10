"""
Main entry point for heart disease classification project
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, create_directories, print_section
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import train_models
from src.evaluate import evaluate_models
from src.visualize import create_visualizations


def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print(" "*20 + "HEART DISEASE CLASSIFICATION PROJECT")
    print("="*80 + "\n")
    
    # Load configuration
    print("Loading configuration...")
    config = load_config('config.yaml')
    
    # Create necessary directories
    create_directories([
        'data/raw',
        'data/processed',
        'models',
        'plots',
        'reports'
    ])
    
    # ========== STEP 1: Load Data ==========
    print("\nSTEP 1: Loading Data")
    print("-" * 80)
    
    data_path = config['data']['raw_path']
    X, y = load_data(data_path)
    
    # ========== STEP 2: Preprocess Data ==========
    print("\n\nSTEP 2: Preprocessing Data")
    print("-" * 80)
    
    test_size = config['data']['test_size']
    random_state = config['data']['random_state']
    X_train, X_test, y_train, y_test = preprocess_data(X, y, test_size=test_size, random_state=random_state)
    
    # ========== STEP 3: Train Models ==========
    print("\n\nSTEP 3: Training Models")
    print("-" * 80)
    
    results = train_models(X_train, X_test, y_train, y_test, config=config)
    
    # ========== STEP 4: Evaluate Models ==========
    print("\n\nSTEP 4: Evaluating Models")
    print("-" * 80)
    
    metrics_df = evaluate_models(results, y_test)
    
    # ========== STEP 5: Visualize Results ==========
    print("\n\nSTEP 5: Visualizing Results")
    print("-" * 80)
    
    create_visualizations(results, y_test, metrics_df, output_path='plots/')
    
    # ========== SUMMARY ==========
    print_section("PROJECT SUMMARY")
    
    print("✓ Data Loading: Completed")
    print("✓ Data Preprocessing: Completed")
    print("✓ Model Training: Completed (4 models trained)")
    print("✓ Model Evaluation: Completed")
    print("✓ Visualization: Completed")
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print("\n" + metrics_df.to_string(index=False))
    
    best_model = metrics_df.iloc[0]['Model']
    best_score = metrics_df.iloc[0]['F1-Score']
    
    print(f"\n{'✓ Best Model:':<30} {best_model}")
    print(f"{'  F1-Score:':<30} {best_score:.4f}")
    
    print("\nOutput Files:")
    print("  - plots/accuracy_comparison.png")
    print("  - plots/metrics_comparison.png")
    print("  - plots/roc_curves.png")
    print("  - plots/confusion_matrices.png")
    
    print("\n" + "="*80)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()