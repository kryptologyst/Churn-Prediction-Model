"""Comprehensive evaluation module for churn prediction models."""

import logging
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, log_loss,
    confusion_matrix, classification_report, precision_recall_curve, roc_curve
)
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

logger = logging.getLogger(__name__)


class ChurnEvaluator:
    """Comprehensive evaluator for churn prediction models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.business_metrics = config.get('business_metrics', {})
        self.evaluation_results = {}
        
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      X_train: pd.DataFrame = None, y_train: pd.Series = None,
                      customer_ids: pd.Series = None) -> Dict[str, Any]:
        """Evaluate a churn prediction model comprehensively.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            X_train: Training features (optional)
            y_train: Training target (optional)
            customer_ids: Customer IDs (optional)
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info(f"Evaluating {model.model_name} model")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        
        # Calculate standard ML metrics
        ml_metrics = self._calculate_ml_metrics(y_test, y_pred, y_pred_proba)
        
        # Calculate business metrics
        business_metrics = self._calculate_business_metrics(y_test, y_pred, y_pred_proba)
        
        # Calculate calibration metrics
        calibration_metrics = self._calculate_calibration_metrics(y_test, y_pred_proba)
        
        # Calculate threshold optimization
        threshold_analysis = self._optimize_threshold(y_test, y_pred_proba)
        
        # Calculate feature importance analysis
        feature_analysis = self._analyze_feature_importance(model, X_test.columns)
        
        # Calculate SHAP analysis if available
        shap_analysis = self._analyze_shap_values(model, X_test)
        
        # Compile results
        results = {
            'model_name': model.model_name,
            'ml_metrics': ml_metrics,
            'business_metrics': business_metrics,
            'calibration_metrics': calibration_metrics,
            'threshold_analysis': threshold_analysis,
            'feature_analysis': feature_analysis,
            'shap_analysis': shap_analysis,
            'predictions': {
                'y_true': y_test.values,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba[:, 1],
                'customer_ids': customer_ids.values if customer_ids is not None else None
            }
        }
        
        self.evaluation_results[model.model_name] = results
        logger.info(f"Evaluation completed for {model.model_name}")
        
        return results
    
    def _calculate_ml_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                            y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate standard machine learning metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of ML metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1]),
            'pr_auc': average_precision_score(y_true, y_pred_proba[:, 1]),
            'brier_score': brier_score_loss(y_true, y_pred_proba[:, 1]),
            'log_loss': log_loss(y_true, y_pred_proba[:, 1])
        }
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()
        
        # Calculate additional metrics
        metrics['specificity'] = metrics['tn'] / (metrics['tn'] + metrics['fp']) if (metrics['tn'] + metrics['fp']) > 0 else 0
        metrics['sensitivity'] = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
        
        return metrics
    
    def _calculate_business_metrics(self, y_true: pd.Series, y_pred: np.ndarray,
                                 y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate business-relevant metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of business metrics
        """
        churn_cost = self.business_metrics.get('churn_cost', 100)
        retention_cost = self.business_metrics.get('retention_cost', 20)
        false_positive_cost = self.business_metrics.get('false_positive_cost', 5)
        
        # Calculate costs
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Cost of missed churners (false negatives)
        missed_churn_cost = fn * churn_cost
        
        # Cost of unnecessary retention campaigns (false positives)
        unnecessary_retention_cost = fp * false_positive_cost
        
        # Cost of successful retention campaigns (true positives)
        successful_retention_cost = tp * retention_cost
        
        # Total cost
        total_cost = missed_churn_cost + unnecessary_retention_cost + successful_retention_cost
        
        # Cost savings (compared to no intervention)
        no_intervention_cost = (tp + fn) * churn_cost
        cost_savings = no_intervention_cost - total_cost
        
        # ROI
        roi = (cost_savings / total_cost) * 100 if total_cost > 0 else 0
        
        # Customer lifetime value impact
        avg_customer_value = churn_cost  # Assuming churn cost represents customer value
        value_protected = tp * avg_customer_value
        value_lost = fn * avg_customer_value
        
        return {
            'missed_churn_cost': missed_churn_cost,
            'unnecessary_retention_cost': unnecessary_retention_cost,
            'successful_retention_cost': successful_retention_cost,
            'total_cost': total_cost,
            'cost_savings': cost_savings,
            'roi_percent': roi,
            'value_protected': value_protected,
            'value_lost': value_lost,
            'net_value_impact': value_protected - value_lost
        }
    
    def _calculate_calibration_metrics(self, y_true: pd.Series, 
                                     y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate calibration metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of calibration metrics
        """
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba[:, 1] > bin_lower) & (y_pred_proba[:, 1] <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin, 1].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_pred_proba[:, 1] > bin_lower) & (y_pred_proba[:, 1] <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred_proba[in_bin, 1].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score_loss(y_true, y_pred_proba[:, 1])
        }
    
    def _optimize_threshold(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Optimize classification threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary with threshold optimization results
        """
        # Calculate metrics for different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        threshold_metrics = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba[:, 1] >= threshold).astype(int)
            
            metrics = {
                'threshold': threshold,
                'precision': precision_score(y_true, y_pred_thresh, zero_division=0),
                'recall': recall_score(y_true, y_pred_thresh, zero_division=0),
                'f1_score': f1_score(y_true, y_pred_thresh, zero_division=0),
                'accuracy': accuracy_score(y_true, y_pred_thresh)
            }
            
            # Calculate business metrics for this threshold
            cm = confusion_matrix(y_true, y_pred_thresh)
            tn, fp, fn, tp = cm.ravel()
            
            churn_cost = self.business_metrics.get('churn_cost', 100)
            retention_cost = self.business_metrics.get('retention_cost', 20)
            false_positive_cost = self.business_metrics.get('false_positive_cost', 5)
            
            total_cost = fn * churn_cost + fp * false_positive_cost + tp * retention_cost
            no_intervention_cost = (tp + fn) * churn_cost
            cost_savings = no_intervention_cost - total_cost
            
            metrics.update({
                'total_cost': total_cost,
                'cost_savings': cost_savings,
                'roi': (cost_savings / total_cost) * 100 if total_cost > 0 else 0
            })
            
            threshold_metrics.append(metrics)
        
        threshold_df = pd.DataFrame(threshold_metrics)
        
        # Find optimal threshold based on different criteria
        optimal_thresholds = {
            'f1_score': threshold_df.loc[threshold_df['f1_score'].idxmax(), 'threshold'],
            'cost_savings': threshold_df.loc[threshold_df['cost_savings'].idxmax(), 'threshold'],
            'roi': threshold_df.loc[threshold_df['roi'].idxmax(), 'threshold']
        }
        
        return {
            'threshold_metrics': threshold_df.to_dict('records'),
            'optimal_thresholds': optimal_thresholds,
            'recommended_threshold': optimal_thresholds['cost_savings']  # Business-focused
        }
    
    def _analyze_feature_importance(self, model, feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            
        Returns:
            Dictionary with feature importance analysis
        """
        try:
            importance = model.get_feature_importance()
            
            # Top features
            top_features = importance.head(10).to_dict()
            
            # Feature categories (if we can infer them)
            feature_categories = self._categorize_features(feature_names)
            
            # Category importance
            category_importance = {}
            for category, features in feature_categories.items():
                category_score = importance[importance.index.isin(features)].sum()
                category_importance[category] = category_score
            
            return {
                'top_features': top_features,
                'feature_categories': feature_categories,
                'category_importance': category_importance,
                'total_features': len(feature_names)
            }
        except Exception as e:
            logger.warning(f"Could not analyze feature importance: {e}")
            return {}
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[str, List[str]]:
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
            else:
                categories['other'].append(feature)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _analyze_shap_values(self, model, X_test: pd.DataFrame) -> Dict[str, Any]:
        """Analyze SHAP values for model interpretability.
        
        Args:
            model: Trained model
            X_test: Test features
            
        Returns:
            Dictionary with SHAP analysis
        """
        try:
            shap_values = model.get_shap_values(X_test)
            
            # Calculate mean absolute SHAP values
            if shap_values.ndim == 2:
                mean_shap_values = np.abs(shap_values).mean(axis=0)
            else:
                # Handle multi-dimensional SHAP values
                mean_shap_values = np.abs(shap_values).mean(axis=(0, 1)) if shap_values.ndim > 2 else np.abs(shap_values).mean(axis=0)
            
            feature_names = X_test.columns
            
            # Top features by SHAP importance
            shap_importance = pd.Series(mean_shap_values, index=feature_names).sort_values(ascending=False)
            top_shap_features = shap_importance.head(10).to_dict()
            
            return {
                'top_shap_features': top_shap_features,
                'shap_values_available': True,
                'mean_shap_values': mean_shap_values.tolist()
            }
        except Exception as e:
            logger.warning(f"Could not analyze SHAP values: {e}")
            return {'shap_values_available': False}
    
    def create_model_leaderboard(self) -> pd.DataFrame:
        """Create a leaderboard comparing all evaluated models.
        
        Returns:
            DataFrame with model comparison
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available")
        
        leaderboard_data = []
        
        for model_name, results in self.evaluation_results.items():
            ml_metrics = results['ml_metrics']
            business_metrics = results['business_metrics']
            
            leaderboard_data.append({
                'Model': model_name,
                'ROC AUC': ml_metrics['roc_auc'],
                'PR AUC': ml_metrics['pr_auc'],
                'F1 Score': ml_metrics['f1_score'],
                'Precision': ml_metrics['precision'],
                'Recall': ml_metrics['recall'],
                'Cost Savings': business_metrics['cost_savings'],
                'ROI (%)': business_metrics['roi_percent'],
                'ECE': results['calibration_metrics']['ece'],
                'Brier Score': ml_metrics['brier_score']
            })
        
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values('ROC AUC', ascending=False)
        
        return leaderboard
    
    def save_evaluation_results(self, filepath: str) -> None:
        """Save evaluation results to file.
        
        Args:
            filepath: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        for model_name, results in self.evaluation_results.items():
            serializable_results[model_name] = convert_to_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {filepath}")
    
    def load_evaluation_results(self, filepath: str) -> None:
        """Load evaluation results from file.
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            self.evaluation_results = json.load(f)
        
        logger.info(f"Loaded evaluation results from {filepath}")
    
    def generate_evaluation_report(self, output_dir: str = "assets") -> None:
        """Generate comprehensive evaluation report.
        
        Args:
            output_dir: Directory to save report files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create leaderboard
        leaderboard = self.create_model_leaderboard()
        leaderboard.to_csv(f"{output_dir}/model_leaderboard.csv", index=False)
        
        # Save detailed results
        self.save_evaluation_results(f"{output_dir}/evaluation_results.json")
        
        # Generate visualizations
        self._create_evaluation_plots(output_dir)
        
        logger.info(f"Evaluation report generated in {output_dir}")
    
    def _create_evaluation_plots(self, output_dir: str) -> None:
        """Create evaluation plots.
        
        Args:
            output_dir: Directory to save plots
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.style.use('seaborn-v0_8')
        
        # Model comparison plot
        if len(self.evaluation_results) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # ROC AUC comparison
            models = list(self.evaluation_results.keys())
            roc_aucs = [self.evaluation_results[model]['ml_metrics']['roc_auc'] for model in models]
            
            axes[0, 0].bar(models, roc_aucs)
            axes[0, 0].set_title('ROC AUC Comparison')
            axes[0, 0].set_ylabel('ROC AUC')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Cost savings comparison
            cost_savings = [self.evaluation_results[model]['business_metrics']['cost_savings'] for model in models]
            
            axes[0, 1].bar(models, cost_savings)
            axes[0, 1].set_title('Cost Savings Comparison')
            axes[0, 1].set_ylabel('Cost Savings ($)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Calibration comparison
            eces = [self.evaluation_results[model]['calibration_metrics']['ece'] for model in models]
            
            axes[1, 0].bar(models, eces)
            axes[1, 0].set_title('Expected Calibration Error')
            axes[1, 0].set_ylabel('ECE')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Feature importance comparison (if available)
            axes[1, 1].text(0.5, 0.5, 'Feature Importance\nComparison', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Feature Importance')
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # Individual model plots
        for model_name, results in self.evaluation_results.items():
            self._create_model_specific_plots(model_name, results, output_dir)
    
    def _create_model_specific_plots(self, model_name: str, results: Dict[str, Any], 
                                   output_dir: str) -> None:
        """Create model-specific plots.
        
        Args:
            model_name: Name of the model
            results: Model evaluation results
            output_dir: Directory to save plots
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # ROC Curve
        y_true = np.array(results['predictions']['y_true'])
        y_pred_proba = np.array(results['predictions']['y_pred_proba'])
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = results['ml_metrics']['roc_auc']
        
        axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title(f'{model_name} - ROC Curve')
        axes[0, 0].legend()
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = results['ml_metrics']['pr_auc']
        
        axes[0, 1].plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title(f'{model_name} - Precision-Recall Curve')
        axes[0, 1].legend()
        
        # Feature Importance
        if 'feature_analysis' in results and results['feature_analysis']:
            top_features = results['feature_analysis'].get('top_features', {})
            if top_features:
                features = list(top_features.keys())[:10]
                importances = list(top_features.values())[:10]
                
                axes[1, 0].barh(features, importances)
                axes[1, 0].set_xlabel('Importance')
                axes[1, 0].set_title(f'{model_name} - Top 10 Features')
        
        # Calibration Plot
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        axes[1, 1].plot(mean_predicted_value, fraction_of_positives, 'o-', label=model_name)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        axes[1, 1].set_xlabel('Mean Predicted Probability')
        axes[1, 1].set_ylabel('Fraction of Positives')
        axes[1, 1].set_title(f'{model_name} - Calibration Plot')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name.lower()}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
