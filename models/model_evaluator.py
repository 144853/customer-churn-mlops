"""
Model evaluation utilities for customer churn prediction.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)
import logging

from .base_model import BaseModel

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Model evaluation class for customer churn prediction.
    
    This class provides comprehensive evaluation metrics and visualizations
    for churn prediction models.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results: Dict[str, Dict[str, Any]] = {}
    
    def evaluate_model(self, 
                      model: BaseModel,
                      X_test: pd.DataFrame,
                      y_test: pd.Series,
                      model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model to evaluate
            X_test: Test features
            y_test: Test targets
            model_name: Name for the model (for tracking)
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        if model_name is None:
            model_name = model.model_name
        
        logger.info(f"Evaluating model: {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
        
        # Calculate basic metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'matrix': cm.tolist(),
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        }
        
        # Classification report
        metrics['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
        
        # Business metrics
        metrics['business_metrics'] = self._calculate_business_metrics(
            y_test, y_pred, y_pred_proba
        )
        
        # Feature importance (if available)
        feature_importance = model.get_feature_importance()
        if feature_importance is not None:
            metrics['feature_importance'] = feature_importance.to_dict('records')
        
        # Store results
        self.evaluation_results[model_name] = metrics
        
        logger.info(f"Model evaluation completed. AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def compare_models(self, 
                      models: Dict[str, BaseModel],
                      X_test: pd.DataFrame,
                      y_test: pd.Series) -> pd.DataFrame:
        """
        Compare multiple models side by side.
        
        Args:
            models: Dictionary of models to compare
            X_test: Test features
            y_test: Test targets
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for model_name, model in models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            comparison_row = {
                'model_name': model_name,
                'algorithm': getattr(model, 'algorithm', 'Unknown'),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'precision_at_10': metrics['business_metrics']['precision_at_10'],
                'recall_at_10': metrics['business_metrics']['recall_at_10']
            }
            
            comparison_data.append(comparison_row)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('roc_auc', ascending=False)
        
        return comparison_df
    
    def _calculate_business_metrics(self, 
                                  y_true: pd.Series,
                                  y_pred: np.ndarray,
                                  y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate business-relevant metrics."""
        
        # Sort by probability descending
        sorted_indices = np.argsort(y_pred_proba)[::-1]
        y_true_sorted = y_true.iloc[sorted_indices]
        
        # Calculate precision and recall at different percentiles
        total_samples = len(y_true)
        percentiles = [5, 10, 20, 30]
        business_metrics = {}
        
        for percentile in percentiles:
            top_n = int(total_samples * percentile / 100)
            if top_n > 0:
                top_predictions = y_true_sorted.iloc[:top_n]
                precision_at_n = top_predictions.sum() / top_n
                recall_at_n = top_predictions.sum() / y_true.sum()
                
                business_metrics[f'precision_at_{percentile}'] = precision_at_n
                business_metrics[f'recall_at_{percentile}'] = recall_at_n
        
        # Calculate cost savings (assuming intervention cost and churn cost)
        intervention_cost = 50  # Cost to intervene per customer
        churn_cost = 500       # Cost when customer churns
        
        # True positives (correctly identified churners)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        # False positives (incorrectly identified churners)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        # False negatives (missed churners)
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        # Calculate costs
        intervention_costs = (tp + fp) * intervention_cost
        prevented_churn_savings = tp * churn_cost
        missed_churn_costs = fn * churn_cost
        
        net_savings = prevented_churn_savings - intervention_costs - missed_churn_costs
        
        business_metrics.update({
            'intervention_cost_per_customer': intervention_cost,
            'churn_cost_per_customer': churn_cost,
            'total_intervention_costs': intervention_costs,
            'prevented_churn_savings': prevented_churn_savings,
            'missed_churn_costs': missed_churn_costs,
            'net_savings': net_savings,
            'roi': (net_savings / intervention_costs) if intervention_costs > 0 else 0
        })
        
        return business_metrics
    
    def plot_confusion_matrix(self, 
                            model_name: str,
                            figsize: Tuple[int, int] = (8, 6)) -> plt.Figure:
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name: Name of the model to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        cm_data = self.evaluation_results[model_name]['confusion_matrix']
        cm = np.array(cm_data['matrix'])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        
        ax.set_title(f'Confusion Matrix - {model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curves(self, 
                       models: Dict[str, BaseModel],
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot ROC curves for multiple models.
        
        Args:
            models: Dictionary of models
            X_test: Test features
            y_test: Test targets
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curves(self, 
                                    models: Dict[str, BaseModel],
                                    X_test: pd.DataFrame,
                                    y_test: pd.Series,
                                    figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot precision-recall curves for multiple models.
        
        Args:
            models: Dictionary of models
            X_test: Test features
            y_test: Test targets
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for model_name, model in models.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            
            ax.plot(recall, precision, label=f'{model_name}')
        
        # Baseline (random classifier)
        baseline = y_test.mean()
        ax.axhline(y=baseline, color='k', linestyle='--', label=f'Random Classifier ({baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves Comparison')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, 
                              model_name: str,
                              top_n: int = 15,
                              figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot feature importance for a model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to show
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        feature_importance = self.evaluation_results[model_name].get('feature_importance')
        if feature_importance is None:
            raise ValueError(f"Feature importance not available for {model_name}")
        
        # Convert to DataFrame and get top features
        importance_df = pd.DataFrame(feature_importance)
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bars = ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
        ax.invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def generate_evaluation_report(self, model_name: str) -> str:
        """
        Generate a comprehensive evaluation report for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Formatted evaluation report
        """
        if model_name not in self.evaluation_results:
            raise ValueError(f"Model {model_name} not found in evaluation results")
        
        results = self.evaluation_results[model_name]
        
        report = f"""
=== Model Evaluation Report: {model_name} ===

Performance Metrics:
- Accuracy: {results['accuracy']:.4f}
- Precision: {results['precision']:.4f}
- Recall: {results['recall']:.4f}
- F1-Score: {results['f1_score']:.4f}
- ROC-AUC: {results['roc_auc']:.4f}

Confusion Matrix:
- True Negatives: {results['confusion_matrix']['true_negatives']}
- False Positives: {results['confusion_matrix']['false_positives']}
- False Negatives: {results['confusion_matrix']['false_negatives']}
- True Positives: {results['confusion_matrix']['true_positives']}

Business Metrics:
- Precision at 10%: {results['business_metrics']['precision_at_10']:.4f}
- Recall at 10%: {results['business_metrics']['recall_at_10']:.4f}
- Net Savings: ${results['business_metrics']['net_savings']:,.2f}
- ROI: {results['business_metrics']['roi']:.2f}

Model effectively identifies {results['recall']:.1%} of churning customers
with {results['precision']:.1%} precision.
        """
        
        return report.strip()
