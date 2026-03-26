"""Utility functions for the churn prediction project."""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import joblib
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    
    try:
        import random
        random.seed(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def create_directories(directories: List[str]) -> None:
    """Create directories if they don't exist.
    
    Args:
        directories: List of directory paths to create
    """
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")


def save_model_artifacts(model, model_name: str, output_dir: str = "models") -> None:
    """Save model and related artifacts.
    
    Args:
        model: Trained model
        model_name: Name of the model
        output_dir: Output directory
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save model
    model_path = f"{output_dir}/{model_name}_model.joblib"
    model.save_model(model_path)
    
    # Save feature importance if available
    try:
        importance = model.get_feature_importance()
        importance_path = f"{output_dir}/{model_name}_feature_importance.joblib"
        joblib.dump(importance, importance_path)
    except Exception as e:
        logger.warning(f"Could not save feature importance for {model_name}: {e}")


def load_model_artifacts(model_name: str, model_dir: str = "models") -> Optional[Any]:
    """Load model and related artifacts.
    
    Args:
        model_name: Name of the model
        model_dir: Model directory
        
    Returns:
        Loaded model or None if not found
    """
    model_path = f"{model_dir}/{model_name}_model.joblib"
    
    if not Path(model_path).exists():
        logger.warning(f"Model not found: {model_path}")
        return None
    
    try:
        from src.models.churn_models import ModelFactory
        model = ModelFactory.create_model(model_name, {})
        model.load_model(model_path)
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        return None


def calculate_business_impact(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_pred_proba: np.ndarray,
                           churn_cost: float = 100,
                           retention_cost: float = 20,
                           false_positive_cost: float = 5) -> Dict[str, float]:
    """Calculate business impact metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        churn_cost: Cost of losing a customer
        retention_cost: Cost of retention campaign
        false_positive_cost: Cost of unnecessary retention effort
        
    Returns:
        Dictionary of business metrics
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate costs
    missed_churn_cost = fn * churn_cost
    unnecessary_retention_cost = fp * false_positive_cost
    successful_retention_cost = tp * retention_cost
    total_cost = missed_churn_cost + unnecessary_retention_cost + successful_retention_cost
    
    # Calculate savings
    no_intervention_cost = (tp + fn) * churn_cost
    cost_savings = no_intervention_cost - total_cost
    
    # Calculate ROI
    roi = (cost_savings / total_cost) * 100 if total_cost > 0 else 0
    
    return {
        'missed_churn_cost': missed_churn_cost,
        'unnecessary_retention_cost': unnecessary_retention_cost,
        'successful_retention_cost': successful_retention_cost,
        'total_cost': total_cost,
        'cost_savings': cost_savings,
        'roi_percent': roi,
        'value_protected': tp * churn_cost,
        'value_lost': fn * churn_cost,
        'net_value_impact': tp * churn_cost - fn * churn_cost
    }


def format_currency(amount: float) -> str:
    """Format currency amount for display.
    
    Args:
        amount: Currency amount
        
    Returns:
        Formatted currency string
    """
    if amount >= 1000000:
        return f"${amount/1000000:.1f}M"
    elif amount >= 1000:
        return f"${amount/1000:.1f}K"
    else:
        return f"${amount:.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format percentage for display.
    
    Args:
        value: Percentage value (0-1)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value*100:.{decimals}f}%"


def get_feature_categories(feature_names: List[str]) -> Dict[str, List[str]]:
    """Categorize features into business-relevant groups.
    
    Args:
        feature_names: List of feature names
        
    Returns:
        Dictionary mapping categories to feature lists
    """
    categories = {
        'demographics': [],
        'service_usage': [],
        'billing': [],
        'contract': [],
        'engagement': [],
        'engineered': [],
        'other': []
    }
    
    for feature in feature_names:
        feature_lower = feature.lower()
        
        if any(keyword in feature_lower for keyword in ['gender', 'senior', 'partner', 'dependent']):
            categories['demographics'].append(feature)
        elif any(keyword in feature_lower for keyword in ['phone', 'internet', 'online', 'streaming', 'tech']):
            categories['service_usage'].append(feature)
        elif any(keyword in feature_lower for keyword in ['charge', 'billing', 'payment']):
            categories['billing'].append(feature)
        elif any(keyword in feature_lower for keyword in ['contract', 'tenure']):
            categories['contract'].append(feature)
        elif any(keyword in feature_lower for keyword in ['paperless', 'multiple']):
            categories['engagement'].append(feature)
        elif any(keyword in feature_lower for keyword in ['avg', 'high', 'long', 'senior_high']):
            categories['engineered'].append(feature)
        else:
            categories['other'].append(feature)
    
    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def validate_data_quality(df: pd.DataFrame, target_column: str = 'Churn') -> Dict[str, Any]:
    """Validate data quality and return summary statistics.
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        
    Returns:
        Dictionary with data quality metrics
    """
    quality_report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'data_types': df.dtypes.to_dict(),
        'target_distribution': None,
        'numeric_summary': None,
        'categorical_summary': None
    }
    
    # Target distribution
    if target_column in df.columns:
        quality_report['target_distribution'] = df[target_column].value_counts().to_dict()
    
    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        quality_report['numeric_summary'] = df[numeric_cols].describe().to_dict()
    
    # Categorical summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        categorical_summary = {}
        for col in categorical_cols:
            categorical_summary[col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'frequency': df[col].value_counts().head().to_dict()
            }
        quality_report['categorical_summary'] = categorical_summary
    
    return quality_report


def log_model_performance(model_name: str, metrics: Dict[str, float], 
                         log_file: str = "logs/model_performance.log") -> None:
    """Log model performance metrics.
    
    Args:
        model_name: Name of the model
        metrics: Dictionary of performance metrics
        log_file: Path to log file
    """
    Path("logs").mkdir(exist_ok=True)
    
    with open(log_file, 'a') as f:
        f.write(f"\n{model_name} Performance:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("-" * 50 + "\n")


def create_model_summary(models: Dict[str, Any]) -> pd.DataFrame:
    """Create a summary table of all models.
    
    Args:
        models: Dictionary of trained models
        
    Returns:
        DataFrame with model summary
    """
    summary_data = []
    
    for model_name, model in models.items():
        try:
            summary_data.append({
                'Model': model_name,
                'Type': type(model).__name__,
                'Trained': model.is_trained,
                'Features': len(model.get_feature_importance()) if hasattr(model, 'get_feature_importance') else 'N/A'
            })
        except Exception as e:
            logger.warning(f"Could not get summary for {model_name}: {e}")
            summary_data.append({
                'Model': model_name,
                'Type': type(model).__name__,
                'Trained': 'Unknown',
                'Features': 'N/A'
            })
    
    return pd.DataFrame(summary_data)


def export_results_to_excel(results: Dict[str, Any], output_path: str = "assets/results.xlsx") -> None:
    """Export results to Excel file with multiple sheets.
    
    Args:
        results: Dictionary containing results
        output_path: Path to save Excel file
    """
    Path("assets").mkdir(exist_ok=True)
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Model performance summary
        if 'leaderboard' in results:
            results['leaderboard'].to_excel(writer, sheet_name='Model Performance', index=False)
        
        # Detailed results for each model
        for model_name, model_results in results.items():
            if model_name == 'leaderboard':
                continue
            
            # Create sheets for different result types
            if 'ml_metrics' in model_results:
                ml_df = pd.DataFrame([model_results['ml_metrics']])
                ml_df.to_excel(writer, sheet_name=f'{model_name}_ML_Metrics', index=False)
            
            if 'business_metrics' in model_results:
                business_df = pd.DataFrame([model_results['business_metrics']])
                business_df.to_excel(writer, sheet_name=f'{model_name}_Business_Metrics', index=False)
    
    logger.info(f"Results exported to {output_path}")


def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements and installed packages.
    
    Returns:
        Dictionary with system information
    """
    import sys
    import platform
    
    system_info = {
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'processor': platform.processor(),
        'memory': 'N/A'  # Would need psutil for memory info
    }
    
    # Check key packages
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm',
        'matplotlib', 'seaborn', 'plotly', 'streamlit', 'shap'
    ]
    
    package_versions = {}
    for package in required_packages:
        try:
            module = __import__(package)
            package_versions[package] = getattr(module, '__version__', 'Unknown')
        except ImportError:
            package_versions[package] = 'Not Installed'
    
    system_info['packages'] = package_versions
    
    return system_info
