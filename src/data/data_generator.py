"""Data generation and preprocessing module for churn prediction."""

import logging
import random
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnDataGenerator:
    """Generate synthetic churn prediction dataset."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize the data generator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Set random seeds for reproducibility
        np.random.seed(self.config['reproducibility']['numpy_seed'])
        random.seed(self.config['reproducibility']['random_seed'])
        
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def generate_synthetic_data(self, n_samples: int = None) -> pd.DataFrame:
        """Generate synthetic customer churn dataset.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic customer data
        """
        if n_samples is None:
            n_samples = self.config['data']['synthetic_size']
            
        logger.info(f"Generating {n_samples} synthetic customer records")
        
        # Customer demographics
        data = {
            'customerID': [f'CUST_{i:06d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        }
        
        # Service information
        data.update({
            'tenure': np.random.gamma(2, 15, n_samples).astype(int).clip(1, 72),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.5, 0.1]),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.3, 0.4, 0.3]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        })
        
        # Contract and billing
        data.update({
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        })
        
        # Generate charges based on services and tenure
        base_monthly = np.random.normal(60, 20, n_samples)
        internet_boost = np.where(data['InternetService'] == 'Fiber optic', 20, 
                                 np.where(data['InternetService'] == 'DSL', 10, 0))
        contract_discount = np.where(data['Contract'] == 'Two year', -10,
                                   np.where(data['Contract'] == 'One year', -5, 0))
        
        data['MonthlyCharges'] = np.clip(base_monthly + internet_boost + contract_discount, 20, 120)
        data['TotalCharges'] = data['MonthlyCharges'] * data['tenure'] + np.random.normal(0, 50, n_samples)
        data['TotalCharges'] = np.clip(data['TotalCharges'], 0, None)
        
        # Generate churn based on realistic patterns
        churn_prob = self._calculate_churn_probability(data)
        data['Churn'] = np.random.binomial(1, churn_prob, n_samples)
        
        df = pd.DataFrame(data)
        
        # Convert TotalCharges to numeric, handling any string values
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['MonthlyCharges'] * df['tenure'])
        
        logger.info(f"Generated dataset with {len(df)} records, {df['Churn'].sum()} churned customers ({df['Churn'].mean():.2%})")
        
        return df
    
    def _calculate_churn_probability(self, data: Dict[str, List]) -> np.ndarray:
        """Calculate realistic churn probabilities based on customer characteristics.
        
        Args:
            data: Dictionary containing customer data
            
        Returns:
            Array of churn probabilities
        """
        n_samples = len(data['customerID'])
        base_prob = 0.2  # Base churn rate
        
        # Factors that increase churn probability
        tenure_factor = np.where(np.array(data['tenure']) < 12, 0.3, 
                               np.where(np.array(data['tenure']) < 24, 0.1, -0.1))
        
        contract_factor = np.where(np.array(data['Contract']) == 'Month-to-month', 0.25, 
                                 np.where(np.array(data['Contract']) == 'One year', 0.05, -0.1))
        
        payment_factor = np.where(np.array(data['PaymentMethod']) == 'Electronic check', 0.15, 0)
        
        internet_factor = np.where(np.array(data['InternetService']) == 'Fiber optic', 0.1, 0)
        
        charges_factor = np.where(np.array(data['MonthlyCharges']) > 80, 0.1, 0)
        
        # Combine factors
        churn_prob = base_prob + tenure_factor + contract_factor + payment_factor + internet_factor + charges_factor
        
        # Ensure probabilities are between 0 and 1
        churn_prob = np.clip(churn_prob, 0.01, 0.95)
        
        return churn_prob
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Preprocess the dataset for modeling.
        
        Args:
            df: Raw dataset
            
        Returns:
            Tuple of (features_df, target_series, customer_ids)
        """
        logger.info("Preprocessing data for modeling")
        
        # Separate features and target
        customer_ids = df['customerID'].copy()
        target = df[self.config['features']['target_column']].copy()
        
        # Remove non-feature columns
        feature_cols = (self.config['features']['categorical_features'] + 
                       self.config['features']['numerical_features'])
        features_df = df[feature_cols].copy()
        
        # Handle missing values
        features_df = self._handle_missing_values(features_df)
        
        # Encode categorical variables
        features_df = self._encode_categorical_features(features_df)
        
        # Create additional features
        features_df = self._create_additional_features(features_df, df)
        
        logger.info(f"Preprocessed data shape: {features_df.shape}")
        logger.info(f"Features: {list(features_df.columns)}")
        
        return features_df, target, customer_ids
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with missing values handled
        """
        # For numerical features, fill with median
        numerical_features = self.config['features']['numerical_features']
        for col in numerical_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # For categorical features, fill with mode
        categorical_features = self.config['features']['categorical_features']
        for col in categorical_features:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
        
        return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with encoded categorical features
        """
        categorical_features = self.config['features']['categorical_features']
        
        for col in categorical_features:
            if col in df.columns:
                # Use label encoding for simplicity
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def _create_additional_features(self, features_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features.
        
        Args:
            features_df: Current features dataframe
            original_df: Original dataframe with all columns
            
        Returns:
            Dataframe with additional features
        """
        # Average monthly charges per tenure
        features_df['avg_monthly_charges'] = original_df['TotalCharges'] / (original_df['tenure'] + 1)
        
        # High value customer indicator
        features_df['high_value_customer'] = (features_df['avg_monthly_charges'] > features_df['avg_monthly_charges'].quantile(0.75)).astype(int)
        
        # Long tenure customer
        features_df['long_tenure_customer'] = (features_df['tenure'] > features_df['tenure'].quantile(0.75)).astype(int)
        
        # Senior citizen with high charges
        if 'SeniorCitizen' in features_df.columns:
            features_df['senior_high_charges'] = (features_df['SeniorCitizen'] * features_df['avg_monthly_charges'] > features_df['avg_monthly_charges'].quantile(0.8)).astype(int)
        
        return features_df
    
    def split_data(self, features_df: pd.DataFrame, target: pd.Series, 
                   customer_ids: pd.Series = None) -> Dict[str, Any]:
        """Split data into train, validation, and test sets.
        
        Args:
            features_df: Features dataframe
            target: Target series
            customer_ids: Customer ID series
            
        Returns:
            Dictionary containing split datasets
        """
        logger.info("Splitting data into train/validation/test sets")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features_df, target, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_seed'],
            stratify=target
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=self.config['data']['validation_size'] / (1 - self.config['data']['test_size']),
            random_state=self.config['data']['random_seed'],
            stratify=y_temp
        )
        
        # Split customer IDs if provided
        if customer_ids is not None:
            _, _, _, customer_ids_test = train_test_split(
                features_df, customer_ids,
                test_size=self.config['data']['test_size'],
                random_state=self.config['data']['random_seed'],
                stratify=target
            )
            _, _, _, customer_ids_val = train_test_split(
                X_temp, customer_ids.iloc[X_temp.index],
                test_size=self.config['data']['validation_size'] / (1 - self.config['data']['test_size']),
                random_state=self.config['data']['random_seed'],
                stratify=y_temp
            )
            customer_ids_train = customer_ids.iloc[X_train.index]
        else:
            customer_ids_train = customer_ids_val = customer_ids_test = None
        
        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info(f"Train set: {X_train_scaled.shape[0]} samples")
        logger.info(f"Validation set: {X_val_scaled.shape[0]} samples")
        logger.info(f"Test set: {X_test_scaled.shape[0]} samples")
        
        return {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'customer_ids_train': customer_ids_train,
            'customer_ids_val': customer_ids_val,
            'customer_ids_test': customer_ids_test,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }


def main():
    """Main function to generate and save synthetic data."""
    generator = ChurnDataGenerator()
    
    # Generate synthetic data
    df = generator.generate_synthetic_data()
    
    # Save raw data
    df.to_csv('data/raw/synthetic_churn_data.csv', index=False)
    logger.info("Saved raw synthetic data to data/raw/synthetic_churn_data.csv")
    
    # Preprocess data
    features_df, target, customer_ids = generator.preprocess_data(df)
    
    # Split data
    data_splits = generator.split_data(features_df, target, customer_ids)
    
    # Save processed data
    import joblib
    joblib.dump(data_splits, 'data/processed/churn_data_splits.joblib')
    logger.info("Saved processed data splits to data/processed/churn_data_splits.joblib")
    
    # Save feature names
    import json
    with open('data/processed/feature_names.json', 'w') as f:
        json.dump(list(features_df.columns), f, indent=2)
    logger.info("Saved feature names to data/processed/feature_names.json")


if __name__ == "__main__":
    main()
