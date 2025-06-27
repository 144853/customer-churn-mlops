"""
Feature engineering utilities for customer churn prediction.
"""

from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering class for customer churn prediction.
    
    This class handles data preprocessing, feature creation, and transformation
    for the churn prediction model.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.scaler: Optional[StandardScaler] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.one_hot_encoder: Optional[OneHotEncoder] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names: Optional[List[str]] = None
        self.is_fitted = False
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw customer data.
        
        Args:
            df: Raw customer dataframe
            
        Returns:
            DataFrame with engineered features
        """
        df_engineered = df.copy()
        
        # Tenure-based features
        if 'tenure_months' in df.columns:
            df_engineered['tenure_years'] = df_engineered['tenure_months'] / 12
            df_engineered['is_new_customer'] = (df_engineered['tenure_months'] <= 6).astype(int)
            df_engineered['is_long_term_customer'] = (df_engineered['tenure_months'] >= 24).astype(int)
        
        # Charges-based features
        if 'monthly_charges' in df.columns and 'total_charges' in df.columns:
            # Convert total_charges to numeric (handle potential string values)
            df_engineered['total_charges'] = pd.to_numeric(
                df_engineered['total_charges'], errors='coerce'
            )
            
            # Average monthly charges over tenure
            df_engineered['avg_monthly_charges'] = (
                df_engineered['total_charges'] / df_engineered['tenure_months']
            ).fillna(df_engineered['monthly_charges'])
            
            # Charges ratio
            df_engineered['charges_ratio'] = (
                df_engineered['monthly_charges'] / df_engineered['avg_monthly_charges']
            ).fillna(1.0)
            
            # High value customer flag
            monthly_charges_median = df_engineered['monthly_charges'].median()
            df_engineered['is_high_value'] = (
                df_engineered['monthly_charges'] > monthly_charges_median
            ).astype(int)
        
        # Service-based features
        service_columns = [
            'online_security', 'online_backup', 'device_protection',
            'tech_support', 'streaming_tv', 'streaming_movies'
        ]
        
        available_services = [col for col in service_columns if col in df.columns]
        if available_services:
            # Count of additional services
            df_engineered['total_services'] = 0
            for col in available_services:
                df_engineered['total_services'] += (df_engineered[col] == 'Yes').astype(int)
            
            # Service adoption rate
            df_engineered['service_adoption_rate'] = (
                df_engineered['total_services'] / len(available_services)
            )
        
        # Contract and payment features
        if 'contract_type' in df.columns:
            df_engineered['is_month_to_month'] = (
                df_engineered['contract_type'] == 'Month-to-month'
            ).astype(int)
        
        if 'payment_method' in df.columns:
            df_engineered['is_automatic_payment'] = (
                df_engineered['payment_method'].str.contains('automatic|bank', case=False, na=False)
            ).astype(int)
        
        # Customer demographics
        if 'age' in df.columns:
            df_engineered['age_group'] = pd.cut(
                df_engineered['age'], 
                bins=[0, 30, 50, 65, 100], 
                labels=['Young', 'Middle', 'Senior', 'Elder']
            )
        
        # Internet usage features
        if 'internet_service' in df.columns:
            df_engineered['has_internet'] = (
                df_engineered['internet_service'] != 'No'
            ).astype(int)
            
            df_engineered['has_fiber'] = (
                df_engineered['internet_service'] == 'Fiber optic'
            ).astype(int)
        
        logger.info(f"Created {len(df_engineered.columns) - len(df.columns)} new features")
        return df_engineered
    
    def prepare_features(self, df: pd.DataFrame, target_column: str = 'churn') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for modeling by separating features and target.
        
        Args:
            df: Input dataframe
            target_column: Name of the target column
            
        Returns:
            Tuple of (features_df, target_series)
        """
        # Create engineered features
        df_processed = self.create_features(df)
        
        # Separate features and target
        if target_column in df_processed.columns:
            X = df_processed.drop(columns=[target_column])
            y = df_processed[target_column]
        else:
            X = df_processed
            y = None
        
        return X, y
    
    def fit_preprocessor(self, X: pd.DataFrame) -> None:
        """
        Fit preprocessors on the training data.
        
        Args:
            X: Training features
        """
        # Identify column types
        numeric_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove columns with too many unique values (potential IDs)
        categorical_columns = [
            col for col in categorical_columns 
            if X[col].nunique() <= 50  # Reasonable threshold for categorical
        ]
        
        # Create preprocessor
        preprocessor_steps = []
        
        if numeric_columns:
            preprocessor_steps.append(
                ('num', StandardScaler(), numeric_columns)
            )
        
        if categorical_columns:
            preprocessor_steps.append(
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_columns)
            )
        
        if preprocessor_steps:
            self.preprocessor = ColumnTransformer(
                transformers=preprocessor_steps,
                remainder='drop'
            )
            
            # Fit the preprocessor
            self.preprocessor.fit(X)
            
            # Store feature names
            self._create_feature_names(X, numeric_columns, categorical_columns)
            self.is_fitted = True
            
            logger.info(f"Fitted preprocessor with {len(self.feature_names)} features")
        else:
            raise ValueError("No suitable columns found for preprocessing")
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted preprocessors.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transformation")
        
        # Transform the data
        X_transformed = self.preprocessor.transform(X)
        
        # Convert to DataFrame with proper column names
        X_transformed_df = pd.DataFrame(
            X_transformed, 
            columns=self.feature_names,
            index=X.index
        )
        
        return X_transformed_df
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessors and transform features in one step.
        
        Args:
            X: Features to fit and transform
            
        Returns:
            Transformed features
        """
        self.fit_preprocessor(X)
        return self.transform(X)
    
    def _create_feature_names(self, X: pd.DataFrame, numeric_columns: List[str], categorical_columns: List[str]) -> None:
        """Create feature names for the transformed dataset."""
        feature_names = []
        
        # Add numeric column names
        feature_names.extend(numeric_columns)
        
        # Add one-hot encoded categorical column names
        if categorical_columns:
            # Get the one-hot encoder from the preprocessor
            cat_transformer = self.preprocessor.named_transformers_['cat']
            cat_feature_names = cat_transformer.get_feature_names_out(categorical_columns)
            feature_names.extend(cat_feature_names)
        
        self.feature_names = feature_names
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Get information about the feature engineering process.
        
        Returns:
            Dictionary with feature engineering information
        """
        return {
            'is_fitted': self.is_fitted,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'has_scaler': self.preprocessor is not None,
            'has_encoders': len(self.label_encoders) > 0
        }
    
    def save_preprocessor(self, filepath: str) -> None:
        """
        Save the fitted preprocessor to disk.
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before saving")
        
        import joblib
        
        preprocessor_data = {
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'label_encoders': self.label_encoders
        }
        
        joblib.dump(preprocessor_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str) -> None:
        """
        Load a fitted preprocessor from disk.
        
        Args:
            filepath: Path to load the preprocessor from
        """
        import joblib
        
        preprocessor_data = joblib.load(filepath)
        
        self.preprocessor = preprocessor_data['preprocessor']
        self.feature_names = preprocessor_data['feature_names']
        self.label_encoders = preprocessor_data['label_encoders']
        self.is_fitted = True
        
        logger.info(f"Preprocessor loaded from {filepath}")
