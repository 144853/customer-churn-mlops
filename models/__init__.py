"""
Models package for customer churn prediction.

This package contains all model-related modules including:
- Base model classes
- Churn prediction models
- Feature engineering utilities
- Model training and evaluation
- Ensemble methods
"""

from .base_model import BaseModel
from .churn_predictor import ChurnPredictor
from .feature_engineering import FeatureEngineer
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .ensemble_model import EnsembleModel

__all__ = [
    'BaseModel',
    'ChurnPredictor', 
    'FeatureEngineer',
    'ModelTrainer',
    'ModelEvaluator',
    'EnsembleModel'
]

__version__ = '1.0.0'
