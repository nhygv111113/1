"""
Model training module
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from src.utils import print_section, save_model
import os


class ModelTrainer:
    """Train multiple models and compare performance"""
    
    def __init__(self, config=None):
        """
        Initialize ModelTrainer
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or {}
        self.models = {}
        self.results = {}
    
    def train(self, X_train, X_test, y_train, y_test):
        """
        Train multiple models
        
        Args:
            X_train (array): Training features
            X_test (array): Test features
            y_train (array): Training target
            y_test (array): Test target
            
        Returns:
            dict: Results with trained models and predictions
        """
        print_section("Model Training")
        
        # Train Logistic Regression
        self._train_logistic_regression(X_train, X_test, y_train, y_test)
        
        # Train Random Forest
        self._train_random_forest(X_train, X_test, y_train, y_test)
        
        # Train SVM
        self._train_svm(X_train, X_test, y_train, y_test)
        
        # Train XGBoost
        self._train_xgboost(X_train, X_test, y_train, y_test)
        
        return self.results
    
    def _train_logistic_regression(self, X_train, X_test, y_train, y_test):
        """Train Logistic Regression"""
        print("\n[1/4] Training Logistic Regression...")
        
        params = self.config.get('models', {}).get('logistic_regression', {}).get('hyperparameters', {})
        
        model = LogisticRegression(
            C=params.get('C', 1.0),
            max_iter=params.get('max_iter', 1000),
            solver=params.get('solver', 'lbfgs'),
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        self.models['Logistic Regression'] = model
        self.results['Logistic Regression'] = {
            'model': model,
            'y_pred': model.predict(X_test),
            'y_pred_proba': model.predict_proba(X_test),
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"✓ Logistic Regression trained")
        print(f"  Train Accuracy: {self.results['Logistic Regression']['train_score']:.4f}")
        print(f"  Test Accuracy: {self.results['Logistic Regression']['test_score']:.4f}")
        print(f"  CV Score: {self.results['Logistic Regression']['cv_mean']:.4f} (+/- {self.results['Logistic Regression']['cv_std']:.4f})")
    
    def _train_random_forest(self, X_train, X_test, y_train, y_test):
        """Train Random Forest"""
        print("\n[2/4] Training Random Forest...")
        
        params = self.config.get('models', {}).get('random_forest', {}).get('hyperparameters', {})
        
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 5),
            random_state=params.get('random_state', 42),
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        self.models['Random Forest'] = model
        self.results['Random Forest'] = {
            'model': model,
            'y_pred': model.predict(X_test),
            'y_pred_proba': model.predict_proba(X_test),
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"✓ Random Forest trained")
        print(f"  Train Accuracy: {self.results['Random Forest']['train_score']:.4f}")
        print(f"  Test Accuracy: {self.results['Random Forest']['test_score']:.4f}")
        print(f"  CV Score: {self.results['Random Forest']['cv_mean']:.4f} (+/- {self.results['Random Forest']['cv_std']:.4f})")
    
    def _train_svm(self, X_train, X_test, y_train, y_test):
        """Train Support Vector Machine"""
        print("\n[3/4] Training Support Vector Machine...")
        
        params = self.config.get('models', {}).get('svm', {}).get('hyperparameters', {})
        
        model = SVC(
            kernel=params.get('kernel', 'rbf'),
            C=params.get('C', 1.0),
            gamma=params.get('gamma', 'scale'),
            probability=True,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        self.models['SVM'] = model
        self.results['SVM'] = {
            'model': model,
            'y_pred': model.predict(X_test),
            'y_pred_proba': model.predict_proba(X_test),
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"✓ SVM trained")
        print(f"  Train Accuracy: {self.results['SVM']['train_score']:.4f}")
        print(f"  Test Accuracy: {self.results['SVM']['test_score']:.4f}")
        print(f"  CV Score: {self.results['SVM']['cv_mean']:.4f} (+/- {self.results['SVM']['cv_std']:.4f})")
    
    def _train_xgboost(self, X_train, X_test, y_train, y_test):
        """Train XGBoost"""
        print("\n[4/4] Training XGBoost...")
        
        params = self.config.get('models', {}).get('xgboost', {}).get('hyperparameters', {})
        
        model = xgb.XGBClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 5),
            learning_rate=params.get('learning_rate', 0.1),
            random_state=params.get('random_state', 42),
            tree_method='hist',
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        self.models['XGBoost'] = model
        self.results['XGBoost'] = {
            'model': model,
            'y_pred': model.predict(X_test),
            'y_pred_proba': model.predict_proba(X_test),
            'train_score': model.score(X_train, y_train),
            'test_score': model.score(X_test, y_test),
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"✓ XGBoost trained")
        print(f"  Train Accuracy: {self.results['XGBoost']['train_score']:.4f}")
        print(f"  Test Accuracy: {self.results['XGBoost']['test_score']:.4f}")
        print(f"  CV Score: {self.results['XGBoost']['cv_mean']:.4f} (+/- {self.results['XGBoost']['cv_std']:.4f})")
    
    def save_models(self, path='models/'):
        """Save all trained models"""
        os.makedirs(path, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(path, f"{model_name.lower().replace(' ', '_')}.pkl")
            save_model(model, model_path)


def train_models(X_train, X_test, y_train, y_test, config=None):
    """
    Simple function to train models
    
    Args:
        X_train (array): Training features
        X_test (array): Test features
        y_train (array): Training target
        y_test (array): Test target
        config (dict): Configuration dictionary
        
    Returns:
        dict: Results with trained models
    """
    trainer = ModelTrainer(config=config)
    return trainer.train(X_train, X_test, y_train, y_test)