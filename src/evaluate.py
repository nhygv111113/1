"""
Model evaluation module
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from src.utils import print_section


class ModelEvaluator:
    """Evaluate and compare model performance"""
    
    def __init__(self):
        """Initialize ModelEvaluator"""
        self.results = {}
        self.metrics_df = None
    
    def evaluate(self, results, y_test):
        """
        Evaluate all models
        
        Args:
            results (dict): Results from model training
            y_test (array): Test target
            
        Returns:
            pd.DataFrame: Metrics comparison
        """
        print_section("Model Evaluation")
        
        metrics_list = []
        
        for model_name, result in results.items():
            y_pred = result['y_pred']
            y_pred_proba = result['y_pred_proba'][:, 1]
            
            # Calculate metrics
            metrics = {
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
            }
            
            metrics_list.append(metrics)
            
            # Store confusion matrix and classification report
            self.results[model_name] = {
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'metrics': metrics
            }
        
        # Create metrics dataframe
        self.metrics_df = pd.DataFrame(metrics_list).sort_values('F1-Score', ascending=False)
        
        # Print results
        self._print_results(y_test)
        
        return self.metrics_df
    
    def _print_results(self, y_test):
        """Print evaluation results"""
        print("\n" + "="*80)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        # Print metrics table
        print("\n" + self.metrics_df.to_string(index=False))
        
        # Print detailed results for each model
        print("\n" + "="*80)
        print("DETAILED EVALUATION RESULTS")
        print("="*80)
        
        for model_name in self.results.keys():
            print(f"\n{'─'*80}")
            print(f"Model: {model_name}")
            print(f"{'─'*80}")
            
            # Confusion Matrix
            cm = self.results[model_name]['confusion_matrix']
            print(f"\nConfusion Matrix:")
            print(f"  True Negatives:  {cm[0,0]}")
            print(f"  False Positives: {cm[0,1]}")
            print(f"  False Negatives: {cm[1,0]}")
            print(f"  True Positives:  {cm[1,1]}")
            
            # Classification Report
            print(f"\nClassification Report:")
            print(self.results[model_name]['classification_report'])
    
    def get_best_model(self):
        """
        Get best model based on F1-score
        
        Returns:
            str: Name of best model
        """
        if self.metrics_df is None:
            return None
        return self.metrics_df.iloc[0]['Model']
    
    def get_metrics_summary(self):
        """
        Get summary of metrics
        
        Returns:
            pd.DataFrame: Metrics summary
        """
        return self.metrics_df


def evaluate_models(results, y_test):
    """
    Simple function to evaluate models
    
    Args:
        results (dict): Results from model training
        y_test (array): Test target
        
    Returns:
        pd.DataFrame: Metrics comparison
    """
    evaluator = ModelEvaluator()
    return evaluator.evaluate(results, y_test)