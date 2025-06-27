"""
Churn prediction model implementation.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ChurnPredictor(BaseModel):
    """
    Customer churn prediction model.
    
    This class implements various algorithms for predicting customer churn
    including Random Forest, Gradient Boosting, and Logistic Regression.
    """
    
    def __init__(self, 
                 algorithm: str = "random_forest",
                 model_name: str = "churn_predictor",
                 version: str = "1.0.0",
                 **model_params):
        """
        Initialize the churn predictor.
        
        Args:
            algorithm: Algorithm to use ('random_forest', 'gradient_boosting', 'logistic_regression')
            model_name: Name of the model
            version: Version of the model
            **model_params: Additional parameters for the underlying model
        """
        super().__init__(model_name, version)
        self.algorithm = algorithm
        self.model_params = model_params
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the underlying model based on the algorithm."""
        if self.algorithm == "random_forest":
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            default_params.update(self.model_params)
            self.model = RandomForestClassifier(**default_params)
            
        elif self.algorithm == "gradient_boosting":
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            }
            default_params.update(self.model_params)
            self.model = GradientBoostingClassifier(**default_params)
            
        elif self.algorithm == "logistic_regression":
            default_params = {
                'random_state': 42,
                'max_iter': 1000
            }
            default_params.update(self.model_params)
            self.model = LogisticRegression(**default_params)
            
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the churn prediction model.
        
        Args:
            X: Training features
            y: Training targets (churn labels)
            **kwargs: Additional training parameters
        """
        logger.info(f"Training {self.algorithm} model on {len(X)} samples")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Train the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Store metadata
        self.metadata.update({
            'algorithm': self.algorithm,
            'training_samples': len(X),
            'feature_count': len(X.columns),
            'positive_class_ratio': y.mean(),
            'model_parameters': self.model.get_params()
        })
        
        logger.info(f"Model training completed successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn for the given customers.
        
        Args:
            X: Customer features
            
        Returns:
            Array of churn predictions (0: no churn, 1: churn)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.validate_input(X)
        
        # Select only the features used during training
        X_selected = X[self.feature_names]
        predictions = self.model.predict(X_selected)
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict churn probabilities for the given customers.
        
        Args:
            X: Customer features
            
        Returns:
            Array of churn probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.validate_input(X)
        
        # Select only the features used during training
        X_selected = X[self.feature_names]
        probabilities = self.model.predict_proba(X_selected)
        
        return probabilities
    
    def get_churn_probability(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get the probability of churn (positive class).
        
        Args:
            X: Customer features
            
        Returns:
            Array of churn probabilities
        """
        probabilities = self.predict_proba(X)
        return probabilities[:, 1]  # Return probability of positive class (churn)
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.get_churn_probability(X_test)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        return metrics
    
    def get_top_features(self, top_n: int = 10) -> Optional[pd.DataFrame]:
        """
        Get the top N most important features for churn prediction.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with top features and their importance scores
        """
        importance_df = self.get_feature_importance()
        if importance_df is not None:
            return importance_df.head(top_n)
        return None
    
    def predict_customer_risk(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict customer churn risk with risk categories.
        
        Args:
            X: Customer features
            
        Returns:
            DataFrame with customer predictions and risk categories
        """
        probabilities = self.get_churn_probability(X)
        predictions = self.predict(X)
        
        # Define risk categories
        def categorize_risk(prob):
            if prob < 0.3:
                return "Low"
            elif prob < 0.7:
                return "Medium"
            else:
                return "High"
        
        results = pd.DataFrame({
            'churn_prediction': predictions,
            'churn_probability': probabilities,
            'risk_category': [categorize_risk(p) for p in probabilities]
        })
        
        return results
