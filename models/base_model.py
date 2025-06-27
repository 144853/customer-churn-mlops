"""
Base model class for all machine learning models in the churn prediction system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    
    This class defines the common interface and functionality that all
    models in the churn prediction system should implement.
    """
    
    def __init__(self, model_name: str, version: str = "1.0.0"):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
            version: Version of the model
        """
        self.model_name = model_name
        self.version = version
        self.model: Optional[BaseEstimator] = None
        self.is_trained = False
        self.feature_names: Optional[List[str]] = None
        self.metadata: Dict[str, Any] = {}
        
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on the provided data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions on the provided data.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of prediction probabilities
        """
        pass
    
    def save_model(self, filepath: Path) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path where to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
            
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'version': self.version,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: Path) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data['model_name']
        self.version = model_data['version']
        self.feature_names = model_data['feature_names']
        self.metadata = model_data['metadata']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
            
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def validate_input(self, X: pd.DataFrame) -> None:
        """
        Validate input data format and features.
        
        Args:
            X: Input features to validate
            
        Raises:
            ValueError: If input validation fails
        """
        if self.feature_names is None:
            raise ValueError("Model not trained - feature names not available")
            
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
            
        extra_features = set(X.columns) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Extra features will be ignored: {extra_features}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_name': self.model_name,
            'version': self.version,
            'is_trained': self.is_trained,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'metadata': self.metadata
        }
