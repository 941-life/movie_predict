import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, 
                           confusion_matrix, mean_absolute_error, mean_squared_error, r2_score,
                           precision_score, recall_score, f1_score, explained_variance_score)
import pandas as pd

class ModelTrainer:
    def __init__(self, n_splits=5):
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        
    def prepare_classification_data(self, X, y):
        """Prepare data for classification with scaling"""
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    def prepare_regression_data(self, X, y):
        """Prepare data for regression with scaling"""
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    def train_classification_models(self, X, y):
        """Train multiple classification models with cross-validation"""
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0)
        }
        
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        results = {}
        
        for name, model in models.items():
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
                
                scores = {
                    'fold': fold,
                    'accuracy': accuracy_score(y_val, y_pred),
                    'precision': precision_score(y_val, y_pred, zero_division=0),
                    'recall': recall_score(y_val, y_pred, zero_division=0),
                    'f1': f1_score(y_val, y_pred, zero_division=0)
                }
                
                if y_proba is not None:
                    scores['roc_auc'] = roc_auc_score(y_val, y_proba)
                
                fold_scores.append(scores)
            
            results[name] = {
                'model': model,
                'scores': fold_scores,
                'mean_accuracy': np.mean([s['accuracy'] for s in fold_scores]),
                'std_accuracy': np.std([s['accuracy'] for s in fold_scores]),
                'mean_precision': np.mean([s['precision'] for s in fold_scores]),
                'std_precision': np.std([s['precision'] for s in fold_scores]),
                'mean_recall': np.mean([s['recall'] for s in fold_scores]),
                'std_recall': np.std([s['recall'] for s in fold_scores]),
                'mean_f1': np.mean([s['f1'] for s in fold_scores]),
                'std_f1': np.std([s['f1'] for s in fold_scores])
            }
            
            if 'roc_auc' in fold_scores[0]:
                results[name].update({
                    'mean_roc_auc': np.mean([s['roc_auc'] for s in fold_scores]),
                    'std_roc_auc': np.std([s['roc_auc'] for s in fold_scores])
                })
        
        return results
    
    def train_regression_models(self, X, y):
        """Train multiple regression models with cross-validation"""
        models = {
            'RandomForest': RandomForestRegressor(random_state=42),
            'SVR': SVR(),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=1.0),
            'GradientBoosting': GradientBoostingRegressor(random_state=42)
        }
        
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        results = {}
        
        for name, model in models.items():
            fold_scores = []
            for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                scores = {
                    'fold': fold,
                    'mae': mean_absolute_error(y_val, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                    'r2': r2_score(y_val, y_pred),
                    'explained_variance': explained_variance_score(y_val, y_pred)
                }
                
                fold_scores.append(scores)
            
            results[name] = {
                'model': model,
                'scores': fold_scores,
                'mean_mae': np.mean([s['mae'] for s in fold_scores]),
                'std_mae': np.std([s['mae'] for s in fold_scores]),
                'mean_rmse': np.mean([s['rmse'] for s in fold_scores]),
                'std_rmse': np.std([s['rmse'] for s in fold_scores]),
                'mean_r2': np.mean([s['r2'] for s in fold_scores]),
                'std_r2': np.std([s['r2'] for s in fold_scores]),
                'mean_explained_variance': np.mean([s['explained_variance'] for s in fold_scores]),
                'std_explained_variance': np.std([s['explained_variance'] for s in fold_scores])
            }
        
        return results
    
    def tune_hyperparameters(self, model, param_grid, X, y, cv=5, method='grid'):
        """Tune hyperparameters using GridSearchCV or RandomizedSearchCV"""
        if method == 'grid':
            search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=cv,
                scoring='accuracy' if isinstance(model, RandomForestClassifier) else 'r2',
                n_jobs=-1
            )
        else:  # method == 'random'
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=20,  # number of parameter settings to try
                cv=cv,
                scoring='accuracy' if isinstance(model, RandomForestClassifier) else 'r2',
                n_jobs=-1,
                random_state=42
            )
        
        search.fit(X, y)
        return search.best_params_, search.best_score_
    
    def evaluate_classification_model(self, model, X_test, y_test):
        """Evaluate classification model performance"""
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return results
    
    def evaluate_regression_model(self, model, X_test, y_test):
        """Evaluate regression model performance"""
        y_pred = model.predict(X_test)
        
        results = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        return results 