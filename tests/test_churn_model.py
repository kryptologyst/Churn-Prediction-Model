"""Tests for the churn prediction model."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

# Import modules to test
from src.data.data_generator import ChurnDataGenerator
from src.models.churn_models import ModelFactory, LogisticRegressionModel, RandomForestModel
from src.eval.evaluator import ChurnEvaluator
from src.utils.utils import (
    set_random_seeds, load_config, save_config, create_directories,
    calculate_business_impact, format_currency, format_percentage,
    get_feature_categories, validate_data_quality
)


class TestDataGenerator:
    """Test cases for ChurnDataGenerator."""
    
    def test_data_generator_initialization(self):
        """Test data generator initialization."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config = {
                'data': {'synthetic_size': 1000, 'random_seed': 42},
                'features': {
                    'categorical_features': ['Contract'],
                    'numerical_features': ['tenure'],
                    'target_column': 'Churn'
                },
                'reproducibility': {'numpy_seed': 42, 'random_seed': 42}
            }
            import yaml
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            generator = ChurnDataGenerator(config_path)
            assert generator.config['data']['synthetic_size'] == 1000
        finally:
            os.unlink(config_path)
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        config = {
            'data': {'synthetic_size': 100},
            'features': {
                'categorical_features': ['Contract'],
                'numerical_features': ['tenure'],
                'target_column': 'Churn'
            },
            'reproducibility': {'numpy_seed': 42, 'random_seed': 42}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            generator = ChurnDataGenerator(config_path)
            df = generator.generate_synthetic_data(50)
            
            assert len(df) == 50
            assert 'Churn' in df.columns
            assert 'tenure' in df.columns
            assert 'Contract' in df.columns
            assert df['Churn'].isin([0, 1]).all()
        finally:
            os.unlink(config_path)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        config = {
            'data': {'synthetic_size': 100},
            'features': {
                'categorical_features': ['Contract'],
                'numerical_features': ['tenure'],
                'target_column': 'Churn'
            },
            'reproducibility': {'numpy_seed': 42, 'random_seed': 42}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            generator = ChurnDataGenerator(config_path)
            df = generator.generate_synthetic_data(50)
            features_df, target, customer_ids = generator.preprocess_data(df)
            
            assert len(features_df) == len(target)
            assert len(features_df) == len(customer_ids)
            assert target.name == 'Churn'
        finally:
            os.unlink(config_path)


class TestModels:
    """Test cases for model classes."""
    
    def test_model_factory(self):
        """Test model factory creation."""
        config = {'random_state': 42}
        
        # Test logistic regression
        lr_model = ModelFactory.create_model('logistic_regression', config)
        assert isinstance(lr_model, LogisticRegressionModel)
        
        # Test random forest
        rf_model = ModelFactory.create_model('random_forest', config)
        assert isinstance(rf_model, RandomForestModel)
        
        # Test invalid model
        with pytest.raises(ValueError):
            ModelFactory.create_model('invalid_model', config)
    
    def test_logistic_regression_model(self):
        """Test logistic regression model."""
        config = {'random_state': 42, 'max_iter': 100}
        model = LogisticRegressionModel(config)
        
        # Create dummy data
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        # Test training
        model.fit(X_train, y_train)
        assert model.is_trained
        
        # Test prediction
        X_test = pd.DataFrame({
            'feature1': np.random.randn(10),
            'feature2': np.random.randn(10)
        })
        
        proba = model.predict_proba(X_test)
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        
        pred = model.predict(X_test)
        assert len(pred) == 10
        assert np.all(np.isin(pred, [0, 1]))
    
    def test_random_forest_model(self):
        """Test random forest model."""
        config = {'random_state': 42, 'n_estimators': 10}
        model = RandomForestModel(config)
        
        # Create dummy data
        X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        # Test training
        model.fit(X_train, y_train)
        assert model.is_trained
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert len(importance) == 2
        assert importance.sum() > 0


class TestEvaluator:
    """Test cases for ChurnEvaluator."""
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        config = {
            'business_metrics': {
                'churn_cost': 100,
                'retention_cost': 20,
                'false_positive_cost': 5
            }
        }
        evaluator = ChurnEvaluator(config)
        assert evaluator.business_metrics['churn_cost'] == 100
    
    def test_calculate_business_metrics(self):
        """Test business metrics calculation."""
        config = {
            'business_metrics': {
                'churn_cost': 100,
                'retention_cost': 20,
                'false_positive_cost': 5
            }
        }
        evaluator = ChurnEvaluator(config)
        
        # Create dummy predictions
        y_true = pd.Series([1, 0, 1, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])  # One false negative
        y_pred_proba = np.array([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.8, 0.2], [0.1, 0.9]])
        
        metrics = evaluator._calculate_business_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'cost_savings' in metrics
        assert 'roi_percent' in metrics
        assert metrics['missed_churn_cost'] > 0  # One false negative


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_random_seeds(self):
        """Test random seed setting."""
        set_random_seeds(42)
        # This is hard to test directly, but we can check it doesn't raise an error
        assert True
    
    def test_format_currency(self):
        """Test currency formatting."""
        assert format_currency(1000) == "$1.0K"
        assert format_currency(1000000) == "$1.0M"
        assert format_currency(500) == "$500.00"
    
    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(0.1234) == "12.3%"
        assert format_percentage(0.1234, 2) == "12.34%"
        assert format_percentage(0.5) == "50.0%"
    
    def test_get_feature_categories(self):
        """Test feature categorization."""
        features = ['gender', 'tenure', 'MonthlyCharges', 'Contract', 'avg_monthly_charges']
        categories = get_feature_categories(features)
        
        assert 'gender' in categories['demographics']
        assert 'tenure' in categories['contract']
        assert 'MonthlyCharges' in categories['billing']
        assert 'avg_monthly_charges' in categories['engineered']
    
    def test_validate_data_quality(self):
        """Test data quality validation."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, np.nan, 5],
            'feature2': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        quality_report = validate_data_quality(df, 'target')
        
        assert quality_report['total_rows'] == 5
        assert quality_report['total_columns'] == 3
        assert quality_report['missing_values'] == 1
        assert quality_report['target_distribution'] is not None


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline."""
        # Create temporary config
        config = {
            'data': {'synthetic_size': 100, 'random_seed': 42},
            'features': {
                'categorical_features': ['Contract'],
                'numerical_features': ['tenure'],
                'target_column': 'Churn'
            },
            'models': {
                'logistic_regression': {'random_state': 42}
            },
            'business_metrics': {
                'churn_cost': 100,
                'retention_cost': 20,
                'false_positive_cost': 5
            },
            'reproducibility': {'numpy_seed': 42, 'random_seed': 42}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            import yaml
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            # Generate data
            generator = ChurnDataGenerator(config_path)
            df = generator.generate_synthetic_data(100)
            features_df, target, customer_ids = generator.preprocess_data(df)
            data_splits = generator.split_data(features_df, target, customer_ids)
            
            # Train model
            model = ModelFactory.create_model('logistic_regression', config['models']['logistic_regression'])
            model.fit(data_splits['X_train'], data_splits['y_train'])
            
            # Evaluate model
            evaluator = ChurnEvaluator(config)
            results = evaluator.evaluate_model(
                model, data_splits['X_test'], data_splits['y_test']
            )
            
            # Check results
            assert 'ml_metrics' in results
            assert 'business_metrics' in results
            assert results['ml_metrics']['roc_auc'] > 0
            assert results['business_metrics']['cost_savings'] is not None
            
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])
