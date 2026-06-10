"""
Standalone training script for heart disease classification
Run this script to train models without the full pipeline
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, create_directories
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.model import ModelTrainer
from src.evaluate import ModelEvaluator


def main():
    """Run training script"""
    
    print("\n" + "="*80)
    print("HEART DISEASE CLASSIFICATION - TRAINING SCRIPT")
    print("="*80 + "\n")
    
    # Load configuration
    config = load_config('config.yaml')
    create_directories(['models/', 'data/raw', 'data/processed'])
    
    # Load and preprocess data
    print("Loading data...")
    X, y = load_data(config['data']['raw_path'])
    
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(
        X, y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    # Train models
    print("\nTraining models...")
    trainer = ModelTrainer(config=config)
    results = trainer.train(X_train, X_test, y_train, y_test)
    
    # Save models
    print("\nSaving models...")
    trainer.save_models('models/')
    
    # Evaluate models
    print("\nEvaluating models...")
    evaluator = ModelEvaluator()
    metrics_df = evaluator.evaluate(results, y_test)
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80 + "\n")
    
    # Print results
    print(metrics_df.to_string(index=False))
    
    print("\n✓ Models saved to: models/")


if __name__ == '__main__':
    main()