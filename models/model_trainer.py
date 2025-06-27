"""
Model training utilities for customer churn prediction.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import logging
from pathlib import Path

from .base_model import BaseModel
from .churn_predictor import ChurnPredictor
from .feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Model training class for customer churn prediction.
    
    This class handles the complete training pipeline including data preparation,
    feature engineering, model training, and validation.
    """
    
    def __init__(self, models_dir: str = "saved_models"):
        """
        Initialize the model trainer.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.feature_engineer = FeatureEngineer()
        self.trained_models: Dict[str, BaseModel] = {}
        self.training_history: List[Dict[str, Any]] = []
    
    def prepare_data(self, 
                    df: pd.DataFrame, 
                    target_column: str = 'churn',
                    test_size: float = 0.2,
                    random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training by splitting and preprocessing.
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            test_size: Proportion of data for testing
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Preparing data: {len(df)} samples, {len(df.columns)} features")
        
        # Create engineered features
        X, y = self.feature_engineer.prepare_features(df, target_column)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Fit and transform features
        X_train_processed = self.feature_engineer.fit_transform(X_train)
        X_test_processed = self.feature_engineer.transform(X_test)
        
        logger.info(f"Data prepared: Train={len(X_train_processed)}, Test={len(X_test_processed)}")
        logger.info(f"Features after processing: {len(X_train_processed.columns)}")
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def train_model(self,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   algorithm: str = "random_forest",
                   model_params: Optional[Dict[str, Any]] = None,
                   cv_folds: int = 5) -> ChurnPredictor:
        """
        Train a single model with the specified algorithm.
        
        Args:
            X_train: Training features
            y_train: Training targets
            algorithm: Algorithm to use
            model_params: Model hyperparameters
            cv_folds: Number of cross-validation folds
            
        Returns:
            Trained model
        """
        logger.info(f"Training {algorithm} model")
        
        # Initialize model
        if model_params is None:
            model_params = {}
        
        model = ChurnPredictor(
            algorithm=algorithm,
            model_name=f"churn_predictor_{algorithm}",
            **model_params
        )
        
        # Train the model
        model.train(X_train, y_train)
        
        # Perform cross-validation
        cv_scores = self._cross_validate(model, X_train, y_train, cv_folds)
        
        # Store training information
        training_info = {
            'algorithm': algorithm,
            'model_params': model_params,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'training_samples': len(X_train),
            'feature_count': len(X_train.columns)
        }
        
        model.metadata.update(training_info)
        self.training_history.append(training_info)
        
        # Store the model
        model_key = f"{algorithm}_{len(self.trained_models)}"
        self.trained_models[model_key] = model
        
        logger.info(f"Model trained successfully. CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return model
    
    def train_multiple_models(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            algorithms: List[str] = None,
                            model_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, ChurnPredictor]:
        """
        Train multiple models with different algorithms.
        
        Args:
            X_train: Training features
            y_train: Training targets
            algorithms: List of algorithms to train
            model_configs: Configuration for each algorithm
            
        Returns:
            Dictionary of trained models
        """
        if algorithms is None:
            algorithms = ["random_forest", "gradient_boosting", "logistic_regression"]
        
        if model_configs is None:
            model_configs = {}
        
        trained_models = {}
        
        for algorithm in algorithms:
            try:
                config = model_configs.get(algorithm, {})
                model = self.train_model(X_train, y_train, algorithm, config)
                trained_models[algorithm] = model
            except Exception as e:
                logger.error(f"Failed to train {algorithm}: {e}")
        
        return trained_models
    
    def hyperparameter_tuning(self,
                            X_train: pd.DataFrame,
                            y_train: pd.Series,
                            algorithm: str = "random_forest",
                            param_grid: Optional[Dict[str, List]] = None,
                            cv_folds: int = 5,
                            scoring: str = "roc_auc") -> ChurnPredictor:
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X_train: Training features
            y_train: Training targets
            algorithm: Algorithm to tune
            param_grid: Parameter grid for search
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric for optimization
            
        Returns:
            Best model found
        """
        logger.info(f"Starting hyperparameter tuning for {algorithm}")
        
        # Default parameter grids
        default_grids = {
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "gradient_boosting": {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            "logistic_regression": {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        if param_grid is None:
            param_grid = default_grids.get(algorithm, {})
        
        # Create base model
        base_model = ChurnPredictor(algorithm=algorithm)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model.model,
            param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Create optimized model
        optimized_model = ChurnPredictor(
            algorithm=algorithm,
            model_name=f"optimized_{algorithm}",
            **grid_search.best_params_
        )
        
        optimized_model.train(X_train, y_train)
        
        # Store tuning results
        tuning_info = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
        
        optimized_model.metadata.update(tuning_info)
        
        logger.info(f"Hyperparameter tuning completed. Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return optimized_model
    
    def _cross_validate(self, model: BaseModel, X: pd.DataFrame, y: pd.Series, cv_folds: int) -> np.ndarray:
        """Perform cross-validation on the model."""
        return cross_val_score(model.model, X, y, cv=cv_folds, scoring='roc_auc')
    
    def save_model(self, model: BaseModel, model_name: str) -> Path:
        """
        Save a trained model to disk.
        
        Args:
            model: Model to save
            model_name: Name for the saved model file
            
        Returns:
            Path to the saved model
        """
        filepath = self.models_dir / f"{model_name}.pkl"
        model.save_model(filepath)
        return filepath
    
    def save_feature_engineer(self, filename: str = "feature_preprocessor.pkl") -> Path:
        """
        Save the feature engineer to disk.
        
        Args:
            filename: Name for the saved preprocessor file
            
        Returns:
            Path to the saved preprocessor
        """
        filepath = self.models_dir / filename
        self.feature_engineer.save_preprocessor(str(filepath))
        return filepath
    
    def load_model(self, model_path: Path) -> BaseModel:
        """
        Load a saved model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model
        """
        model = ChurnPredictor()
        model.load_model(model_path)
        return model
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all training activities.
        
        Returns:
            Dictionary containing training summary
        """
        return {
            'total_models_trained': len(self.trained_models),
            'algorithms_used': list(set([info['algorithm'] for info in self.training_history])),
            'training_history': self.training_history,
            'feature_engineer_info': self.feature_engineer.get_feature_info()
        }
