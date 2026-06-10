"""
Visualization module
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os
from pathlib import Path


class ModelVisualizer:
    """Visualize model results"""
    
    def __init__(self, output_path='plots/'):
        """
        Initialize ModelVisualizer
        
        Args:
            output_path (str): Path to save plots
        """
        self.output_path = output_path
        Path(output_path).mkdir(parents=True, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 5)
    
    def plot_accuracy_comparison(self, metrics_df):
        """
        Plot accuracy comparison across models
        
        Args:
            metrics_df (pd.DataFrame): Metrics dataframe
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = metrics_df['Model']
        accuracy = metrics_df['Accuracy']
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, accuracy, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_path, 'accuracy_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_metrics_comparison(self, metrics_df):
        """
        Plot multiple metrics comparison
        
        Args:
            metrics_df (pd.DataFrame): Metrics dataframe
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(metrics_df))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            offset = (i - 2) * width
            ax.bar(x + offset, metrics_df[metric], width, label=metric, alpha=0.8)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Model'], rotation=45, ha='right')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = os.path.join(self.output_path, 'metrics_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_roc_curves(self, results, y_test):
        """
        Plot ROC curves for all models
        
        Args:
            results (dict): Results from model training
            y_test (array): Test target
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
        
        for (model_name, result), color in zip(results.items(), colors):
            y_pred_proba = result['y_pred_proba'][:, 1]
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot random classifier
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
        
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(self.output_path, 'roc_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()
    
    def plot_confusion_matrices(self, results, y_test):
        """
        Plot confusion matrices for all models
        
        Args:
            results (dict): Results from model training
            y_test (array): Test target
        """
        n_models = len(results)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(results.items()):
            y_pred = result['y_pred']
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=True, annot_kws={'size': 14})
            axes[idx].set_title(f'{model_name}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
            axes[idx].set_xticklabels(['No Disease', 'Disease'])
            axes[idx].set_yticklabels(['No Disease', 'Disease'])
        
        # Hide extra subplots
        for idx in range(n_models, 4):
            axes[idx].set_visible(False)
        
        plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        save_path = os.path.join(self.output_path, 'confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {save_path}")
        plt.close()


def create_visualizations(results, y_test, metrics_df, output_path='plots/'):
    """
    Create all visualizations
    
    Args:
        results (dict): Results from model training
        y_test (array): Test target
        metrics_df (pd.DataFrame): Metrics dataframe
        output_path (str): Path to save plots
    """
    visualizer = ModelVisualizer(output_path=output_path)
    
    print("\nGenerating visualizations...")
    visualizer.plot_accuracy_comparison(metrics_df)
    visualizer.plot_metrics_comparison(metrics_df)
    visualizer.plot_roc_curves(results, y_test)
    visualizer.plot_confusion_matrices(results, y_test)
    print("✓ Visualizations complete!")