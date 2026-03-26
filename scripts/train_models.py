"""Main training script for churn prediction models."""

import logging
import yaml
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import argparse
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import our modules
from src.data.data_generator import ChurnDataGenerator
from src.models.churn_models import ModelFactory, optimize_xgboost_hyperparameters, optimize_lightgbm_hyperparameters
from src.eval.evaluator import ChurnEvaluator
from src.viz.visualizer import ChurnVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/churn_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def setup_reproducibility(config: Dict[str, Any]) -> None:
    """Set up reproducibility settings.
    
    Args:
        config: Configuration dictionary
    """
    reproducibility_config = config.get('reproducibility', {})
    
    # Set numpy seed
    np.random.seed(reproducibility_config.get('numpy_seed', 42))
    
    # Set random seed
    import random
    random.seed(reproducibility_config.get('random_seed', 42))
    
    # Set other seeds if needed
    if reproducibility_config.get('torch_seed'):
        import torch
        torch.manual_seed(reproducibility_config['torch_seed'])
    
    logger.info("Reproducibility settings configured")


def prepare_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare data for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary containing data splits
    """
    logger.info("Preparing data")
    
    # Check if processed data exists
    processed_data_path = "data/processed/churn_data_splits.joblib"
    
    if Path(processed_data_path).exists():
        logger.info("Loading existing processed data")
        data_splits = joblib.load(processed_data_path)
    else:
        logger.info("Generating new synthetic data")
        generator = ChurnDataGenerator("configs/config.yaml")
        
        # Generate synthetic data
        df = generator.generate_synthetic_data()
        
        # Save raw data
        df.to_csv('data/raw/synthetic_churn_data.csv', index=False)
        logger.info("Saved raw synthetic data")
        
        # Preprocess and split data
        features_df, target, customer_ids = generator.preprocess_data(df)
        data_splits = generator.split_data(features_df, target, customer_ids)
        
        # Save processed data
        joblib.dump(data_splits, processed_data_path)
        logger.info("Saved processed data splits")
    
    logger.info(f"Data prepared: {data_splits['X_train'].shape[0]} train, "
               f"{data_splits['X_val'].shape[0]} val, {data_splits['X_test'].shape[0]} test samples")
    
    return data_splits


def train_models(config: Dict[str, Any], data_splits: Dict[str, Any]) -> Dict[str, Any]:
    """Train all configured models.
    
    Args:
        config: Configuration dictionary
        data_splits: Dictionary containing data splits
        
    Returns:
        Dictionary containing trained models
    """
    logger.info("Training models")
    
    models = {}
    models_config = config.get('models', {})
    
    # Extract data
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    
    # Train each model
    for model_name, model_config in models_config.items():
        logger.info(f"Training {model_name}")
        
        try:
            # Create model
            model = ModelFactory.create_model(model_name, model_config)
            
            # Train model
            model.fit(X_train, y_train, X_val, y_val)
            
            # Save model
            model_path = f"models/{model_name}_model.joblib"
            Path("models").mkdir(exist_ok=True)
            model.save_model(model_path)
            
            models[model_name] = model
            logger.info(f"Successfully trained and saved {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            continue
    
    logger.info(f"Trained {len(models)} models successfully")
    return models


def optimize_hyperparameters(config: Dict[str, Any], data_splits: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize hyperparameters for advanced models.
    
    Args:
        config: Configuration dictionary
        data_splits: Dictionary containing data splits
        
    Returns:
        Dictionary containing optimized hyperparameters
    """
    logger.info("Optimizing hyperparameters")
    
    optimization_config = config.get('optimization', {})
    n_trials = optimization_config.get('n_trials', 50)
    
    # Extract data
    X_train = data_splits['X_train']
    y_train = data_splits['y_train']
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    
    optimized_params = {}
    
    # Optimize XGBoost if configured
    if 'xgboost' in config.get('models', {}):
        logger.info("Optimizing XGBoost hyperparameters")
        try:
            xgb_params = optimize_xgboost_hyperparameters(X_train, y_train, X_val, y_val, n_trials)
            optimized_params['xgboost'] = xgb_params
            logger.info("XGBoost hyperparameter optimization completed")
        except Exception as e:
            logger.error(f"XGBoost optimization failed: {e}")
    
    # Optimize LightGBM if configured
    if 'lightgbm' in config.get('models', {}):
        logger.info("Optimizing LightGBM hyperparameters")
        try:
            lgb_params = optimize_lightgbm_hyperparameters(X_train, y_train, X_val, y_val, n_trials)
            optimized_params['lightgbm'] = lgb_params
            logger.info("LightGBM hyperparameter optimization completed")
        except Exception as e:
            logger.error(f"LightGBM optimization failed: {e}")
    
    # Save optimized parameters
    if optimized_params:
        joblib.dump(optimized_params, "models/optimized_hyperparameters.joblib")
        logger.info("Saved optimized hyperparameters")
    
    return optimized_params


def evaluate_models(models: Dict[str, Any], data_splits: Dict[str, Any], 
                   config: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all trained models.
    
    Args:
        models: Dictionary containing trained models
        data_splits: Dictionary containing data splits
        config: Configuration dictionary
        
    Returns:
        Dictionary containing evaluation results
    """
    logger.info("Evaluating models")
    
    # Create evaluator
    evaluator = ChurnEvaluator(config)
    
    # Extract test data
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    customer_ids_test = data_splits['customer_ids_test']
    
    # Evaluate each model
    evaluation_results = {}
    
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}")
        
        try:
            results = evaluator.evaluate_model(
                model, X_test, y_test, 
                customer_ids=customer_ids_test
            )
            evaluation_results[model_name] = results
            logger.info(f"Successfully evaluated {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {model_name}: {e}")
            continue
    
    # Generate evaluation report
    evaluator.generate_evaluation_report("assets")
    
    # Create leaderboard
    leaderboard = evaluator.create_model_leaderboard()
    leaderboard.to_csv("assets/model_leaderboard.csv", index=False)
    logger.info("Saved model leaderboard")
    
    logger.info(f"Evaluated {len(evaluation_results)} models")
    return evaluation_results


def create_visualizations(df: pd.DataFrame, evaluation_results: Dict[str, Any], 
                          config: Dict[str, Any]) -> None:
    """Create visualizations for the analysis.
    
    Args:
        df: Original dataframe
        evaluation_results: Dictionary containing evaluation results
        config: Configuration dictionary
    """
    logger.info("Creating visualizations")
    
    # Create visualizer
    visualizer = ChurnVisualizer(config)
    
    # Generate all plots
    visualizer.generate_all_plots(df, evaluation_results)
    
    logger.info("Visualizations created successfully")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train churn prediction models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--optimize', action='store_true',
                       help='Run hyperparameter optimization')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training and only evaluate existing models')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set up reproducibility
    setup_reproducibility(config)
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("assets").mkdir(exist_ok=True)
    Path("assets/plots").mkdir(exist_ok=True)
    
    # Prepare data
    data_splits = prepare_data(config)
    
    # Load original dataframe for visualizations
    df = pd.read_csv('data/raw/synthetic_churn_data.csv')
    
    if not args.skip_training:
        # Optimize hyperparameters if requested
        if args.optimize:
            optimized_params = optimize_hyperparameters(config, data_splits)
            # Update config with optimized parameters
            for model_name, params in optimized_params.items():
                if model_name in config.get('models', {}):
                    config['models'][model_name].update(params)
        
        # Train models
        models = train_models(config, data_splits)
    else:
        # Load existing models
        logger.info("Loading existing models")
        models = {}
        models_dir = Path("models")
        
        for model_file in models_dir.glob("*_model.joblib"):
            model_name = model_file.stem.replace("_model", "")
            try:
                model = ModelFactory.create_model(model_name, {})
                model.load_model(str(model_file))
                models[model_name] = model
                logger.info(f"Loaded {model_name} model")
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
    
    if not models:
        logger.error("No models available for evaluation")
        return
    
    # Evaluate models
    evaluation_results = evaluate_models(models, data_splits, config)
    
    # Create visualizations
    create_visualizations(df, evaluation_results, config)
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Models trained: {len(models)}")
    print(f"Models evaluated: {len(evaluation_results)}")
    
    if evaluation_results:
        print("\nModel Performance (ROC AUC):")
        for model_name, results in evaluation_results.items():
            roc_auc = results['ml_metrics']['roc_auc']
            cost_savings = results['business_metrics']['cost_savings']
            print(f"  {model_name}: ROC AUC = {roc_auc:.3f}, Cost Savings = ${cost_savings:.2f}")
    
    print(f"\nResults saved to:")
    print(f"  - Models: models/")
    print(f"  - Evaluation: assets/")
    print(f"  - Visualizations: assets/plots/")
    print(f"  - Logs: logs/churn_prediction.log")
    
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
