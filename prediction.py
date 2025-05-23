import pandas as pd
import numpy as np
from data_processor import (load_and_preprocess_data, process_genres, 
                          process_languages, create_success_label, 
                          generate_data_visualizations, get_feature_sets,
                          generate_model_performance_plots, analyze_feature_importance,
                          generate_prediction_analysis, generate_residual_plot)
from model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split

def main():
    # Load and preprocess data
    df = load_and_preprocess_data('./movies_metadata.csv')
    
    # Process genres and languages
    df, genres_encoded, mlb = process_genres(df)
    lang_encoded, lang_encoder = process_languages(df)
    
    # Generate basic visualizations
    generate_data_visualizations(df)
    
    # Get feature sets
    final_features_cls, features_reg = get_feature_sets(df, genres_encoded, lang_encoded)
    
    # Prepare classification data
    X_cls = pd.concat([df[final_features_cls], genres_encoded, lang_encoded], axis=1)
    y_cls = create_success_label(df)
    
    # Prepare regression data
    X_reg = df[features_reg]
    y_reg = df['revenue']
    
    # Initialize model trainer
    trainer = ModelTrainer(n_splits=5)
    
    # Train and evaluate classification models
    print("\n--- Classification Models ---")
    X_cls_scaled, y_cls = trainer.prepare_classification_data(X_cls, y_cls)
    cls_results = trainer.train_classification_models(X_cls_scaled, y_cls)
    
    # Generate classification performance plots
    generate_model_performance_plots(cls_results)
    
    # Print classification results
    for model_name, result in cls_results.items():
        print(f"\n{model_name} Results:")
        print(f"Mean Accuracy: {result['mean_accuracy']:.4f} (±{result['std_accuracy']:.4f})")
        print(f"Mean Precision: {result['mean_precision']:.4f} (±{result['std_precision']:.4f})")
        print(f"Mean Recall: {result['mean_recall']:.4f} (±{result['std_recall']:.4f})")
        print(f"Mean F1-Score: {result['mean_f1']:.4f} (±{result['std_f1']:.4f})")
        if 'mean_roc_auc' in result:
            print(f"Mean ROC-AUC: {result['mean_roc_auc']:.4f} (±{result['std_roc_auc']:.4f})")
    
    # Train and evaluate regression models
    print("\n--- Regression Models ---")
    X_reg_scaled, y_reg = trainer.prepare_regression_data(X_reg, y_reg)
    reg_results = trainer.train_regression_models(X_reg_scaled, y_reg)
    
    # Generate regression performance plots
    generate_model_performance_plots(reg_results)
    
    # Print regression results
    for model_name, result in reg_results.items():
        print(f"\n{model_name} Results:")
        print(f"Mean MAE: {result['mean_mae']:,.2f} (±{result['std_mae']:,.2f})")
        print(f"Mean RMSE: {result['mean_rmse']:,.2f} (±{result['std_rmse']:,.2f})")
        print(f"Mean R²: {result['mean_r2']:.4f} (±{result['std_r2']:.4f})")
        print(f"Mean Explained Variance: {result['mean_explained_variance']:.4f} (±{result['std_explained_variance']:.4f})")
    
    # Hyperparameter tuning for best classification model
    best_cls_model = max(cls_results.items(), key=lambda x: x[1]['mean_accuracy'])[0]
    print(f"\nTuning hyperparameters for {best_cls_model}...")
    
    param_grid = {
        'RandomForest': {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 20, 30, 40, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_samples_split': [2, 5, 10]
        },
        'Ridge': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
        },
        'Lasso': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
            'selection': ['cyclic', 'random']
        }
    }
    
    # Try both grid search and random search
    best_params_grid, best_score_grid = trainer.tune_hyperparameters(
        cls_results[best_cls_model]['model'],
        param_grid[best_cls_model],
        X_cls_scaled,
        y_cls,
        method='grid'
    )
    
    best_params_random, best_score_random = trainer.tune_hyperparameters(
        cls_results[best_cls_model]['model'],
        param_grid[best_cls_model],
        X_cls_scaled,
        y_cls,
        method='random'
    )
    
    print("\nGrid Search Results:")
    print(f"Best parameters: {best_params_grid}")
    print(f"Best score: {best_score_grid:.4f}")
    
    print("\nRandom Search Results:")
    print(f"Best parameters: {best_params_random}")
    print(f"Best score: {best_score_random:.4f}")
    
    # Analyze feature importance for the best model
    best_model = cls_results[best_cls_model]['model']
    feature_names = X_cls.columns
    important_features = analyze_feature_importance(best_model, feature_names)
    
    if important_features:
        print("\nTop 10 Important Features:")
        for feature, importance in important_features:
            print(f"{feature}: {importance:.4f}")
    
    # Generate prediction analysis plots for regression
    best_reg_model = max(reg_results.items(), key=lambda x: x[1]['mean_r2'])[0]
    y_pred_reg = reg_results[best_reg_model]['model'].predict(X_reg_scaled)
    generate_prediction_analysis(y_reg, y_pred_reg)
    generate_residual_plot(y_reg, y_pred_reg)

if __name__ == "__main__":
    main()