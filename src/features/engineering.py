"""Feature engineering module for customer churn prediction."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

from ..config.settings import settings


class FeatureEngineer:
    """Feature engineering pipeline for customer churn prediction."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.text_vectorizer = None
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal/rolling aggregation features."""
        logger.info("Creating temporal features")
        
        df = df.copy()
        
        # Ensure datetime columns
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['account_age_days'] = (pd.Timestamp.now() - df['created_at']).dt.days
        
        # Monthly charges trends (if usage data available)
        if 'monthly_charges' in df.columns:
            df['charges_per_tenure'] = df['monthly_charges'] / (df['tenure_months'] + 1)
            df['total_charges_estimated'] = df['monthly_charges'] * df['tenure_months']
        
        # Contract duration features
        if 'contract_type' in df.columns:
            contract_duration_map = {
                'Month-to-month': 1,
                'One year': 12,
                'Two year': 24
            }
            df['contract_duration_months'] = df['contract_type'].map(contract_duration_map)
            df['tenure_contract_ratio'] = df['tenure_months'] / (df['contract_duration_months'] + 1)
        
        return df
    
    def create_usage_features(self, df: pd.DataFrame, usage_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create usage-based features."""
        logger.info("Creating usage features")
        
        df = df.copy()
        
        if usage_df is not None and not usage_df.empty:
            # Calculate rolling aggregates (30, 60, 90 days)
            usage_df['date'] = pd.to_datetime(usage_df['date'])
            current_date = usage_df['date'].max()
            
            for window in [30, 60, 90]:
                window_start = current_date - pd.Timedelta(days=window)
                window_data = usage_df[usage_df['date'] >= window_start]
                
                # Aggregate by customer
                agg_features = window_data.groupby('customer_id').agg({
                    'data_usage_gb': ['mean', 'sum', 'std'],
                    'call_minutes': ['mean', 'sum', 'std'],
                    'sms_count': ['mean', 'sum', 'std']
                }).round(2)
                
                # Flatten column names
                agg_features.columns = [f'{col[0]}_{col[1]}_{window}d' for col in agg_features.columns]
                agg_features = agg_features.reset_index()
                
                # Merge with main dataframe
                df = df.merge(agg_features, on='customer_id', how='left')
        
        # Service usage features
        service_columns = ['internet_service', 'phone_service']
        for col in service_columns:
            if col in df.columns:
                df[f'{col}_flag'] = (df[col] == 'Yes').astype(int)
        
        return df
    
    def create_billing_features(self, df: pd.DataFrame, billing_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create billing-related features."""
        logger.info("Creating billing features")
        
        df = df.copy()
        
        if billing_df is not None and not billing_df.empty:
            # Payment behavior features
            payment_stats = billing_df.groupby('customer_id').agg({
                'amount': ['mean', 'sum', 'std'],
                'payment_status': lambda x: (x == 'Late').sum(),
                'billing_date': 'count'
            }).round(2)
            
            payment_stats.columns = ['avg_bill_amount', 'total_billed', 'bill_amount_std', 
                                   'late_payments', 'total_bills']
            payment_stats = payment_stats.reset_index()
            
            # Calculate late payment rate
            payment_stats['late_payment_rate'] = (
                payment_stats['late_payments'] / payment_stats['total_bills']
            ).fillna(0)
            
            # Merge with main dataframe
            df = df.merge(payment_stats, on='customer_id', how='left')
        
        # Payment method encoding
        if 'payment_method' in df.columns:
            # Create binary features for payment methods
            payment_methods = df['payment_method'].unique()
            for method in payment_methods:
                if pd.notna(method):
                    df[f'payment_{method.lower().replace(" ", "_")}'] = (
                        df['payment_method'] == method
                    ).astype(int)
        
        return df
    
    def create_support_features(self, df: pd.DataFrame, tickets_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create support ticket features."""
        logger.info("Creating support features")
        
        df = df.copy()
        
        if tickets_df is not None and not tickets_df.empty:
            # Support ticket statistics
            ticket_stats = tickets_df.groupby('customer_id').agg({
                'ticket_id': 'count',
                'priority': lambda x: (x == 'High').sum(),
                'status': lambda x: (x == 'Open').sum(),
                'category': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
            })
            
            ticket_stats.columns = ['total_tickets', 'high_priority_tickets', 
                                  'open_tickets', 'most_common_category']
            ticket_stats = ticket_stats.reset_index()
            
            # Calculate ticket rates
            ticket_stats['high_priority_rate'] = (
                ticket_stats['high_priority_tickets'] / ticket_stats['total_tickets']
            ).fillna(0)
            
            # Merge with main dataframe
            df = df.merge(ticket_stats, on='customer_id', how='left')
            
            # Text features from ticket descriptions
            if 'description' in tickets_df.columns:
                self._create_text_features(df, tickets_df)
        
        return df
    
    def _create_text_features(self, df: pd.DataFrame, tickets_df: pd.DataFrame) -> pd.DataFrame:
        """Create text features from support ticket descriptions."""
        logger.info("Creating text features from support tickets")
        
        # Aggregate text by customer
        customer_text = tickets_df.groupby('customer_id')['description'].apply(
            lambda x: ' '.join(x.fillna(''))
        ).reset_index()
        
        # Initialize TF-IDF vectorizer
        if self.text_vectorizer is None:
            self.text_vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
        
        # Create TF-IDF features
        if not customer_text.empty:
            tfidf_matrix = self.text_vectorizer.fit_transform(customer_text['description'])
            
            # Create feature names
            feature_names = [f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
            
            # Convert to dataframe
            tfidf_df = pd.DataFrame(
                tfidf_matrix.toarray(),
                columns=feature_names
            )
            tfidf_df['customer_id'] = customer_text['customer_id'].values
            
            # Merge with main dataframe
            df = df.merge(tfidf_df, on='customer_id', how='left')
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info("Encoding categorical features")
        
        df = df.copy()
        
        # Define categorical columns
        categorical_columns = [
            'gender', 'contract_type', 'internet_service', 
            'payment_method', 'most_common_category'
        ]
        
        # Filter existing columns
        categorical_columns = [col for col in categorical_columns if col in df.columns]
        
        for col in categorical_columns:
            if fit:
                # Use LabelEncoder for binary/ordinal categories
                if col in ['gender']:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(
                        df[col].fillna('Unknown')
                    )
                # Use OneHotEncoder for nominal categories
                else:
                    if col not in self.encoders:
                        self.encoders[col] = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    
                    encoded_features = self.encoders[col].fit_transform(
                        df[[col]].fillna('Unknown')
                    )
                    
                    # Create feature names
                    feature_names = [f'{col}_{cat}' for cat in self.encoders[col].categories_[0]]
                    
                    # Add encoded features to dataframe
                    encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
                    df = pd.concat([df, encoded_df], axis=1)
            else:
                # Transform using fitted encoders
                if col in self.encoders:
                    if isinstance(self.encoders[col], LabelEncoder):
                        df[f'{col}_encoded'] = self.encoders[col].transform(
                            df[col].fillna('Unknown')
                        )
                    else:
                        encoded_features = self.encoders[col].transform(
                            df[[col]].fillna('Unknown')
                        )
                        feature_names = [f'{col}_{cat}' for cat in self.encoders[col].categories_[0]]
                        encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)
                        df = pd.concat([df, encoded_df], axis=1)
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling numerical features")
        
        df = df.copy()
        
        # Define numerical columns to scale
        numerical_columns = [
            'age', 'tenure_months', 'monthly_charges', 'total_charges',
            'account_age_days', 'charges_per_tenure', 'total_charges_estimated',
            'avg_bill_amount', 'total_billed', 'late_payment_rate'
        ]
        
        # Add usage features if they exist
        usage_features = [col for col in df.columns if any(x in col for x in ['_30d', '_60d', '_90d'])]
        numerical_columns.extend(usage_features)
        
        # Add TF-IDF features if they exist
        tfidf_features = [col for col in df.columns if col.startswith('tfidf_')]
        numerical_columns.extend(tfidf_features)
        
        # Filter existing columns
        numerical_columns = [col for col in numerical_columns if col in df.columns]
        
        if numerical_columns:
            if fit:
                self.scalers['numerical'] = StandardScaler()
                df[numerical_columns] = self.scalers['numerical'].fit_transform(
                    df[numerical_columns].fillna(0)
                )
            else:
                if 'numerical' in self.scalers:
                    df[numerical_columns] = self.scalers['numerical'].transform(
                        df[numerical_columns].fillna(0)
                    )
        
        return df
    
    def create_feature_pipeline(self, 
                              customer_df: pd.DataFrame,
                              usage_df: Optional[pd.DataFrame] = None,
                              billing_df: Optional[pd.DataFrame] = None,
                              tickets_df: Optional[pd.DataFrame] = None,
                              fit: bool = True) -> pd.DataFrame:
        """Run complete feature engineering pipeline."""
        logger.info("Running feature engineering pipeline")
        
        # Start with customer data
        df = customer_df.copy()
        
        # Create features
        df = self.create_temporal_features(df)
        df = self.create_usage_features(df, usage_df)
        df = self.create_billing_features(df, billing_df)
        df = self.create_support_features(df, tickets_df)
        
        # Encode and scale features
        df = self.encode_categorical_features(df, fit=fit)
        df = self.scale_numerical_features(df, fit=fit)
        
        # Store feature names for later use
        if fit:
            self.feature_names = [col for col in df.columns 
                                if col not in ['customer_id', 'created_at', 'updated_at']]
        
        logger.info(f"Feature engineering completed. Total features: {len(self.feature_names)}")
        
        return df
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.feature_names
    
    def save_feature_engineering_artifacts(self, path: str) -> None:
        """Save feature engineering artifacts."""
        import joblib
        
        artifacts = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_names': self.feature_names,
            'text_vectorizer': self.text_vectorizer
        }
        
        joblib.dump(artifacts, path)
        logger.info(f"Feature engineering artifacts saved to: {path}")
    
    def load_feature_engineering_artifacts(self, path: str) -> None:
        """Load feature engineering artifacts."""
        import joblib
        
        artifacts = joblib.load(path)
        self.scalers = artifacts['scalers']
        self.encoders = artifacts['encoders']
        self.feature_names = artifacts['feature_names']
        self.text_vectorizer = artifacts.get('text_vectorizer')
        
        logger.info(f"Feature engineering artifacts loaded from: {path}")
