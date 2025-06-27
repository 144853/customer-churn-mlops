"""
Ensemble model implementation for customer churn prediction.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import logging

from .base_model import BaseModel
from .churn_predictor import ChurnPredictor

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines multiple churn prediction models.
    
    This class implements various ensemble methods including voting,
    stacking, and weighted averaging for improved prediction performance.
    """
    
    def __init__(self, 
                 models: List[BaseModel],
                 ensemble_method: str = "voting",
                 voting_type: str = "soft",
                 model_name: str = "ensemble_churn_predictor",
                 version: str = "1.0.0"):
        """
        Initialize the ensemble model.
        
        Args:
            models: List of base models to ensemble
            ensemble_method: Method to use ('voting', 'weighted', 'stacking')
            voting_type: Type of voting ('hard' or 'soft')
            model_name: Name of the ensemble model
            version: Version of the ensemble model
        """
        super().__init__(model_name, version)
        
        self.base_models = models
        self.ensemble_method = ensemble_method
        self.voting_type = voting_type
        self.weights: Optional[List[float]] = None
        self.meta_model: Optional[BaseModel] = None
        
        # Validate inputs
        if not models:
            raise ValueError("At least one base model is required")
        
        for model in models:
            if not model.is_trained:
                raise ValueError("All base models must be trained")
        
        self._initialize_ensemble()
    
    def _initialize_ensemble(self) -> None:
        """Initialize the ensemble based on the method."""
        if self.ensemble_method == "voting":
            self._setup_voting_ensemble()
        elif self.ensemble_method == "weighted":
            self._setup_weighted_ensemble()
        elif self.ensemble_method == "stacking":
            self._setup_stacking_ensemble()
        else:
            raise ValueError(f"Unsupported ensemble method: {self.ensemble_method}")
    
    def _setup_voting_ensemble(self) -> None:
        """Setup voting ensemble."""
        # Create estimator list for VotingClassifier
        estimators = []
        for i, model in enumerate(self.base_models):
            estimator_name = f"model_{i}_{getattr(model, 'algorithm', 'unknown')}"
            estimators.append((estimator_name, model.model))
        
        self.model = VotingClassifier(
            estimators=estimators,
            voting=self.voting_type
        )
        
        logger.info(f"Setup voting ensemble with {len(estimators)} models")
    
    def _setup_weighted_ensemble(self) -> None:
        """Setup weighted ensemble (weights will be set during training)."""
        if self.weights is None:
            # Initialize with equal weights
            self.weights = [1.0 / len(self.base_models)] * len(self.base_models)
        
        logger.info(f"Setup weighted ensemble with weights: {self.weights}")
    
    def _setup_stacking_ensemble(self) -> None:
        """Setup stacking ensemble."""
        # Meta-model will be trained on base model predictions
        self.meta_model = ChurnPredictor(
            algorithm="logistic_regression",
            model_name="meta_model"
        )
        
        logger.info("Setup stacking ensemble with logistic regression meta-model")
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> None:
        """
        Train the ensemble model.
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional training parameters
        """
        logger.info(f"Training ensemble model using {self.ensemble_method} method")
        
        # Store feature names from the first model
        self.feature_names = self.base_models[0].feature_names
        
        if self.ensemble_method == "voting":
            # Voting ensemble is already configured, just mark as trained
            self.is_trained = True
            
        elif self.ensemble_method == "weighted":
            # Optimize weights using validation performance
            self._optimize_weights(X, y)
            self.is_trained = True
            
        elif self.ensemble_method == "stacking":
            # Train meta-model on base model predictions
            self._train_stacking(X, y)
            self.is_trained = True
        
        # Store metadata
        self.metadata.update({
            'ensemble_method': self.ensemble_method,
            'num_base_models': len(self.base_models),
            'base_model_algorithms': [getattr(m, 'algorithm', 'unknown') for m in self.base_models],
            'training_samples': len(X)
        })
        
        logger.info("Ensemble model training completed")
    
    def _optimize_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Optimize ensemble weights using validation performance."""
        from sklearn.model_selection import cross_val_score
        
        # Get cross-validation predictions from each model
        base_predictions = []
        base_scores = []
        
        for model in self.base_models:
            # Get predictions
            pred_proba = model.predict_proba(X)[:, 1]
            base_predictions.append(pred_proba)
            
            # Get CV score
            cv_score = cross_val_score(model.model, X, y, cv=5, scoring='roc_auc').mean()
            base_scores.append(cv_score)
        
        # Normalize scores to create weights
        total_score = sum(base_scores)
        self.weights = [score / total_score for score in base_scores]
        
        logger.info(f"Optimized weights: {self.weights}")
    
    def _train_stacking(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train stacking ensemble meta-model."""
        # Generate meta-features using cross-validation
        from sklearn.model_selection import cross_val_predict
        
        meta_features = []
        
        for model in self.base_models:
            # Get out-of-fold predictions
            pred_proba = cross_val_predict(
                model.model, X, y, cv=5, method='predict_proba'
            )[:, 1]
            meta_features.append(pred_proba)
        
        # Create meta-feature DataFrame
        meta_X = pd.DataFrame(
            np.column_stack(meta_features),
            columns=[f'model_{i}_pred' for i in range(len(self.base_models))]
        )
        
        # Train meta-model
        self.meta_model.train(meta_X, y)
        
        logger.info("Stacking meta-model trained successfully")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        self.validate_input(X)
        
        if self.ensemble_method == "voting":
            return self.model.predict(X[self.feature_names])
            
        elif self.ensemble_method == "weighted":
            return self._predict_weighted(X)
            
        elif self.ensemble_method == "stacking":
            return self._predict_stacking(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions using the ensemble.
        
        Args:
            X: Features for prediction
            
        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        self.validate_input(X)
        
        if self.ensemble_method == "voting":
            return self.model.predict_proba(X[self.feature_names])
            
        elif self.ensemble_method == "weighted":
            return self._predict_proba_weighted(X)
            
        elif self.ensemble_method == "stacking":
            return self._predict_proba_stacking(X)
    
    def _predict_weighted(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted predictions."""
        # Get predictions from all models
        predictions = []
        for model in self.base_models:
            pred = model.predict(X[self.feature_names])
            predictions.append(pred)
        
        # Weighted average
        weighted_pred = np.average(predictions, axis=0, weights=self.weights)
        
        # Convert to binary predictions
        return (weighted_pred > 0.5).astype(int)
    
    def _predict_proba_weighted(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted probability predictions."""
        # Get probability predictions from all models
        probabilities = []
        for model in self.base_models:
            prob = model.predict_proba(X[self.feature_names])
            probabilities.append(prob)
        
        # Weighted average
        weighted_prob = np.average(probabilities, axis=0, weights=self.weights)
        
        return weighted_prob
    
    def _predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
        """Make stacking predictions."""
        # Get base model predictions
        meta_features = []
        for model in self.base_models:
            pred_proba = model.predict_proba(X[self.feature_names])[:, 1]
            meta_features.append(pred_proba)
        
        # Create meta-feature DataFrame
        meta_X = pd.DataFrame(
            np.column_stack(meta_features),
            columns=[f'model_{i}_pred' for i in range(len(self.base_models))]
        )
        
        # Predict using meta-model
        return self.meta_model.predict(meta_X)
    
    def _predict_proba_stacking(self, X: pd.DataFrame) -> np.ndarray:
        """Make stacking probability predictions."""
        # Get base model predictions
        meta_features = []
        for model in self.base_models:
            pred_proba = model.predict_proba(X[self.feature_names])[:, 1]
            meta_features.append(pred_proba)
        
        # Create meta-feature DataFrame
        meta_X = pd.DataFrame(
            np.column_stack(meta_features),
            columns=[f'model_{i}_pred' for i in range(len(self.base_models))]
        )
        
        # Predict using meta-model
        return self.meta_model.predict_proba(meta_X)
    
    def get_model_contributions(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Get individual model contributions to the ensemble prediction.
        
        Args:
            X: Features for prediction
            
        Returns:
            DataFrame with individual model predictions and contributions
        """
        contributions = []
        
        for i, model in enumerate(self.base_models):
            pred_proba = model.predict_proba(X[self.feature_names])[:, 1]
            
            contribution_data = {
                'model_index': i,
                'algorithm': getattr(model, 'algorithm', 'unknown'),
                'predictions': pred_proba.tolist()
            }
            
            if self.ensemble_method == "weighted":
                contribution_data['weight'] = self.weights[i]
                contribution_data['weighted_contribution'] = (pred_proba * self.weights[i]).tolist()
            
            contributions.append(contribution_data)
        
        return pd.DataFrame(contributions)
    
    def evaluate_base_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate individual base models and the ensemble.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        # Evaluate base models
        for i, model in enumerate(self.base_models):
            pred = model.predict(X_test[self.feature_names])
            pred_proba = model.predict_proba(X_test[self.feature_names])[:, 1]
            
            accuracy = accuracy_score(y_test, pred)
            auc = roc_auc_score(y_test, pred_proba)
            
            results.append({
                'model_type': 'base',
                'model_name': f"model_{i}_{getattr(model, 'algorithm', 'unknown')}",
                'accuracy': accuracy,
                'roc_auc': auc
            })
        
        # Evaluate ensemble
        ensemble_pred = self.predict(X_test)
        ensemble_pred_proba = self.predict_proba(X_test)[:, 1]
        
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        ensemble_auc = roc_auc_score(y_test, ensemble_pred_proba)
        
        results.append({
            'model_type': 'ensemble',
            'model_name': f"ensemble_{self.ensemble_method}",
            'accuracy': ensemble_accuracy,
            'roc_auc': ensemble_auc
        })
        
        return pd.DataFrame(results)
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble configuration.
        
        Returns:
            Dictionary with ensemble information
        """
        info = {
            'ensemble_method': self.ensemble_method,
            'num_base_models': len(self.base_models),
            'base_models': []
        }
        
        for i, model in enumerate(self.base_models):
            model_info = {
                'index': i,
                'algorithm': getattr(model, 'algorithm', 'unknown'),
                'model_name': model.model_name,
                'version': model.version
            }
            
            if self.ensemble_method == "weighted" and self.weights:
                model_info['weight'] = self.weights[i]
            
            info['base_models'].append(model_info)
        
        if self.ensemble_method == "stacking" and self.meta_model:
            info['meta_model'] = {
                'algorithm': getattr(self.meta_model, 'algorithm', 'unknown'),
                'is_trained': self.meta_model.is_trained
            }
        
        return info
