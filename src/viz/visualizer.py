"""Visualization module for churn prediction analysis."""

import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import shap
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ChurnVisualizer:
    """Comprehensive visualizer for churn prediction analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.plot_style = config.get('visualization', {}).get('plot_style', 'seaborn-v0_8')
        self.figure_size = config.get('visualization', {}).get('figure_size', [10, 6])
        self.dpi = config.get('visualization', {}).get('dpi', 300)
        self.save_plots = config.get('visualization', {}).get('save_plots', True)
        self.plots_path = config.get('visualization', {}).get('plots_path', 'assets/plots')
        
        # Set plotting style
        plt.style.use(self.plot_style)
        
        # Create plots directory
        import os
        os.makedirs(self.plots_path, exist_ok=True)
    
    def plot_data_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot data distribution and basic statistics.
        
        Args:
            df: Input dataframe
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Churn distribution
        churn_counts = df['Churn'].value_counts()
        axes[0, 0].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%')
        axes[0, 0].set_title('Churn Distribution')
        
        # Tenure distribution
        axes[0, 1].hist(df['tenure'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Tenure Distribution')
        axes[0, 1].set_xlabel('Tenure (months)')
        axes[0, 1].set_ylabel('Frequency')
        
        # Monthly charges distribution
        axes[0, 2].hist(df['MonthlyCharges'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Monthly Charges Distribution')
        axes[0, 2].set_xlabel('Monthly Charges ($)')
        axes[0, 2].set_ylabel('Frequency')
        
        # Contract type distribution
        contract_counts = df['Contract'].value_counts()
        axes[1, 0].bar(contract_counts.index, contract_counts.values)
        axes[1, 0].set_title('Contract Type Distribution')
        axes[1, 0].set_xlabel('Contract Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Internet service distribution
        internet_counts = df['InternetService'].value_counts()
        axes[1, 1].bar(internet_counts.index, internet_counts.values)
        axes[1, 1].set_title('Internet Service Distribution')
        axes[1, 1].set_xlabel('Internet Service')
        axes[1, 1].set_ylabel('Count')
        
        # Payment method distribution
        payment_counts = df['PaymentMethod'].value_counts()
        axes[1, 2].bar(payment_counts.index, payment_counts.values)
        axes[1, 2].set_title('Payment Method Distribution')
        axes[1, 2].set_xlabel('Payment Method')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = save_path or f"{self.plots_path}/data_distribution.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved data distribution plot to {save_path}")
        
        plt.show()
    
    def plot_churn_by_features(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot churn rates by different features.
        
        Args:
            df: Input dataframe
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Churn by tenure
        tenure_churn = df.groupby(pd.cut(df['tenure'], bins=6))['Churn'].mean()
        axes[0, 0].bar(range(len(tenure_churn)), tenure_churn.values)
        axes[0, 0].set_title('Churn Rate by Tenure')
        axes[0, 0].set_xlabel('Tenure (months)')
        axes[0, 0].set_ylabel('Churn Rate')
        axes[0, 0].set_xticks(range(len(tenure_churn)))
        axes[0, 0].set_xticklabels([f"{int(interval.left)}-{int(interval.right)}" for interval in tenure_churn.index], rotation=45)
        
        # Churn by monthly charges
        charges_churn = df.groupby(pd.cut(df['MonthlyCharges'], bins=6))['Churn'].mean()
        axes[0, 1].bar(range(len(charges_churn)), charges_churn.values)
        axes[0, 1].set_title('Churn Rate by Monthly Charges')
        axes[0, 1].set_xlabel('Monthly Charges ($)')
        axes[0, 1].set_ylabel('Churn Rate')
        axes[0, 1].set_xticks(range(len(charges_churn)))
        axes[0, 1].set_xticklabels([f"{int(interval.left)}-{int(interval.right)}" for interval in charges_churn.index], rotation=45)
        
        # Churn by contract type
        contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
        axes[0, 2].bar(contract_churn.index, contract_churn.values)
        axes[0, 2].set_title('Churn Rate by Contract Type')
        axes[0, 2].set_xlabel('Contract Type')
        axes[0, 2].set_ylabel('Churn Rate')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Churn by internet service
        internet_churn = df.groupby('InternetService')['Churn'].mean().sort_values(ascending=False)
        axes[1, 0].bar(internet_churn.index, internet_churn.values)
        axes[1, 0].set_title('Churn Rate by Internet Service')
        axes[1, 0].set_xlabel('Internet Service')
        axes[1, 0].set_ylabel('Churn Rate')
        
        # Churn by payment method
        payment_churn = df.groupby('PaymentMethod')['Churn'].mean().sort_values(ascending=False)
        axes[1, 1].bar(payment_churn.index, payment_churn.values)
        axes[1, 1].set_title('Churn Rate by Payment Method')
        axes[1, 1].set_xlabel('Payment Method')
        axes[1, 1].set_ylabel('Churn Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Churn by senior citizen status
        senior_churn = df.groupby('SeniorCitizen')['Churn'].mean()
        axes[1, 2].bar(['Not Senior', 'Senior'], senior_churn.values)
        axes[1, 2].set_title('Churn Rate by Senior Citizen Status')
        axes[1, 2].set_xlabel('Senior Citizen')
        axes[1, 2].set_ylabel('Churn Rate')
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = save_path or f"{self.plots_path}/churn_by_features.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved churn by features plot to {save_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Plot correlation heatmap of numerical features.
        
        Args:
            df: Input dataframe
            save_path: Optional path to save the plot
        """
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numerical_cols].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Heatmap')
        
        if self.save_plots:
            save_path = save_path or f"{self.plots_path}/correlation_heatmap.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved correlation heatmap to {save_path}")
        
        plt.show()
    
    def plot_model_performance(self, evaluation_results: Dict[str, Any], 
                              save_path: Optional[str] = None) -> None:
        """Plot model performance comparison.
        
        Args:
            evaluation_results: Dictionary containing evaluation results for multiple models
            save_path: Optional path to save the plot
        """
        models = list(evaluation_results.keys())
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # ROC AUC comparison
        roc_aucs = [evaluation_results[model]['ml_metrics']['roc_auc'] for model in models]
        axes[0, 0].bar(models, roc_aucs)
        axes[0, 0].set_title('ROC AUC Comparison')
        axes[0, 0].set_ylabel('ROC AUC')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Precision comparison
        precisions = [evaluation_results[model]['ml_metrics']['precision'] for model in models]
        axes[0, 1].bar(models, precisions)
        axes[0, 1].set_title('Precision Comparison')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Recall comparison
        recalls = [evaluation_results[model]['ml_metrics']['recall'] for model in models]
        axes[0, 2].bar(models, recalls)
        axes[0, 2].set_title('Recall Comparison')
        axes[0, 2].set_ylabel('Recall')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # F1 Score comparison
        f1_scores = [evaluation_results[model]['ml_metrics']['f1_score'] for model in models]
        axes[1, 0].bar(models, f1_scores)
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Cost savings comparison
        cost_savings = [evaluation_results[model]['business_metrics']['cost_savings'] for model in models]
        axes[1, 1].bar(models, cost_savings)
        axes[1, 1].set_title('Cost Savings Comparison')
        axes[1, 1].set_ylabel('Cost Savings ($)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Calibration error comparison
        eces = [evaluation_results[model]['calibration_metrics']['ece'] for model in models]
        axes[1, 2].bar(models, eces)
        axes[1, 2].set_title('Expected Calibration Error')
        axes[1, 2].set_ylabel('ECE')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = save_path or f"{self.plots_path}/model_performance.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved model performance plot to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, evaluation_results: Dict[str, Any], 
                       save_path: Optional[str] = None) -> None:
        """Plot ROC curves for all models.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, results in evaluation_results.items():
            y_true = np.array(results['predictions']['y_true'])
            y_pred_proba = np.array(results['predictions']['y_pred_proba'])
            roc_auc = results['ml_metrics']['roc_auc']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            save_path = save_path or f"{self.plots_path}/roc_curves.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved ROC curves plot to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, evaluation_results: Dict[str, Any], 
                                    save_path: Optional[str] = None) -> None:
        """Plot precision-recall curves for all models.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, results in evaluation_results.items():
            y_true = np.array(results['predictions']['y_true'])
            y_pred_proba = np.array(results['predictions']['y_pred_proba'])
            pr_auc = results['ml_metrics']['pr_auc']
            
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            plt.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            save_path = save_path or f"{self.plots_path}/precision_recall_curves.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved precision-recall curves plot to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model_name: str, feature_importance: pd.Series, 
                               top_n: int = 15, save_path: Optional[str] = None) -> None:
        """Plot feature importance for a model.
        
        Args:
            model_name: Name of the model
            feature_importance: Series with feature importance scores
            top_n: Number of top features to show
            save_path: Optional path to save the plot
        """
        top_features = feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        top_features.plot(kind='barh')
        plt.title(f'{model_name} - Top {top_n} Feature Importance')
        plt.xlabel('Importance Score')
        plt.gca().invert_yaxis()
        
        if self.save_plots:
            save_path = save_path or f"{self.plots_path}/{model_name.lower()}_feature_importance.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")
        
        plt.show()
    
    def plot_shap_summary(self, model, X_test: pd.DataFrame, 
                         save_path: Optional[str] = None) -> None:
        """Plot SHAP summary plot.
        
        Args:
            model: Trained model with SHAP explainer
            X_test: Test features
            save_path: Optional path to save the plot
        """
        try:
            shap_values = model.get_shap_values(X_test)
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.title(f'{model.model_name} - SHAP Summary Plot')
            
            if self.save_plots:
                save_path = save_path or f"{self.plots_path}/{model.model_name.lower()}_shap_summary.png"
                plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved SHAP summary plot to {save_path}")
            
            plt.show()
        except Exception as e:
            logger.warning(f"Could not create SHAP summary plot: {e}")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'],
                   yticklabels=['No Churn', 'Churn'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if self.save_plots:
            save_path = save_path or f"{self.plots_path}/{model_name.lower()}_confusion_matrix.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved confusion matrix plot to {save_path}")
        
        plt.show()
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                               model_name: str, save_path: Optional[str] = None) -> None:
        """Plot calibration curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save_path: Optional path to save the plot
        """
        from sklearn.calibration import calibration_curve
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, 'o-', label=model_name)
        plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'{model_name} - Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            save_path = save_path or f"{self.plots_path}/{model_name.lower()}_calibration.png"
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved calibration curve plot to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, evaluation_results: Dict[str, Any], 
                                   df: pd.DataFrame) -> None:
        """Create an interactive dashboard using Plotly.
        
        Args:
            evaluation_results: Dictionary containing evaluation results
            df: Original dataframe
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Model Performance', 'Churn Distribution', 
                          'ROC Curves', 'Feature Importance',
                          'Cost Analysis', 'Calibration'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        models = list(evaluation_results.keys())
        
        # Model performance comparison
        roc_aucs = [evaluation_results[model]['ml_metrics']['roc_auc'] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=roc_aucs, name='ROC AUC'),
            row=1, col=1
        )
        
        # Churn distribution
        churn_counts = df['Churn'].value_counts()
        fig.add_trace(
            go.Pie(labels=['No Churn', 'Churn'], values=churn_counts.values, name='Churn Distribution'),
            row=1, col=2
        )
        
        # ROC curves
        for model_name, results in evaluation_results.items():
            y_true = np.array(results['predictions']['y_true'])
            y_pred_proba = np.array(results['predictions']['y_pred_proba'])
            roc_auc = results['ml_metrics']['roc_auc']
            
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            fig.add_trace(
                go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{model_name} (AUC={roc_auc:.3f})'),
                row=2, col=1
            )
        
        # Add diagonal line for ROC
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'),
            row=2, col=1
        )
        
        # Feature importance (using first model)
        if models:
            first_model = models[0]
            if 'feature_analysis' in evaluation_results[first_model]:
                top_features = evaluation_results[first_model]['feature_analysis'].get('top_features', {})
                if top_features:
                    features = list(top_features.keys())[:10]
                    importances = list(top_features.values())[:10]
                    fig.add_trace(
                        go.Bar(x=importances, y=features, orientation='h', name='Feature Importance'),
                        row=2, col=2
                    )
        
        # Cost analysis
        cost_savings = [evaluation_results[model]['business_metrics']['cost_savings'] for model in models]
        fig.add_trace(
            go.Bar(x=models, y=cost_savings, name='Cost Savings'),
            row=3, col=1
        )
        
        # Calibration curve (using first model)
        if models:
            first_model = models[0]
            y_true = np.array(evaluation_results[first_model]['predictions']['y_true'])
            y_pred_proba = np.array(evaluation_results[first_model]['predictions']['y_pred_proba'])
            
            from sklearn.calibration import calibration_curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            
            fig.add_trace(
                go.Scatter(x=mean_predicted_value, y=fraction_of_positives, 
                          mode='markers+lines', name=f'{first_model} Calibration'),
                row=3, col=2
            )
            
            # Add perfect calibration line
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                          line=dict(dash='dash'), name='Perfect Calibration'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Churn Prediction Analysis Dashboard",
            showlegend=True
        )
        
        # Save interactive plot
        if self.save_plots:
            save_path = f"{self.plots_path}/interactive_dashboard.html"
            fig.write_html(save_path)
            logger.info(f"Saved interactive dashboard to {save_path}")
        
        fig.show()
    
    def generate_all_plots(self, df: pd.DataFrame, evaluation_results: Dict[str, Any]) -> None:
        """Generate all standard plots.
        
        Args:
            df: Original dataframe
            evaluation_results: Dictionary containing evaluation results
        """
        logger.info("Generating all visualization plots")
        
        # Data analysis plots
        self.plot_data_distribution(df)
        self.plot_churn_by_features(df)
        self.plot_correlation_heatmap(df)
        
        # Model performance plots
        if evaluation_results:
            self.plot_model_performance(evaluation_results)
            self.plot_roc_curves(evaluation_results)
            self.plot_precision_recall_curves(evaluation_results)
            
            # Individual model plots
            for model_name, results in evaluation_results.items():
                # Feature importance
                if 'feature_analysis' in results and results['feature_analysis']:
                    top_features = results['feature_analysis'].get('top_features', {})
                    if top_features:
                        feature_importance = pd.Series(top_features)
                        self.plot_feature_importance(model_name, feature_importance)
                
                # Confusion matrix
                y_true = np.array(results['predictions']['y_true'])
                y_pred = np.array(results['predictions']['y_pred'])
                self.plot_confusion_matrix(y_true, y_pred, model_name)
                
                # Calibration curve
                y_pred_proba = np.array(results['predictions']['y_pred_proba'])
                self.plot_calibration_curve(y_true, y_pred_proba, model_name)
        
        # Interactive dashboard
        if evaluation_results:
            self.create_interactive_dashboard(evaluation_results, df)
        
        logger.info("All visualization plots generated successfully")
