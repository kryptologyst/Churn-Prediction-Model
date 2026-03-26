"""Streamlit demo for churn prediction model."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import joblib
import yaml
from pathlib import Path
import logging

# Import our modules
from src.data.data_generator import ChurnDataGenerator
from src.models.churn_models import ModelFactory
from src.eval.evaluator import ChurnEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is an experimental research and educational tool.</strong></p>
    <ul>
        <li>This model is for demonstration and learning purposes only</li>
        <li>Do not use predictions for automated business decisions without human review</li>
        <li>Model performance may vary significantly on real-world data</li>
        <li>Always validate predictions with domain experts and additional data</li>
        <li>Consider ethical implications and potential biases in predictions</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">📊 Churn Prediction Model Demo</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Model Overview", "Data Analysis", "Model Performance", "Predictions", "Feature Analysis"]
)

# Load configuration
@st.cache_data
def load_config():
    """Load configuration."""
    try:
        with open('configs/config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("Configuration file not found. Please run the training script first.")
        return None

config = load_config()

# Load models
@st.cache_data
def load_models():
    """Load trained models."""
    models = {}
    models_dir = Path("models")
    
    if not models_dir.exists():
        return models
    
    for model_file in models_dir.glob("*_model.joblib"):
        model_name = model_file.stem.replace("_model", "")
        try:
            model = ModelFactory.create_model(model_name, {})
            model.load_model(str(model_file))
            models[model_name] = model
        except Exception as e:
            st.error(f"Failed to load {model_name}: {e}")
    
    return models

models = load_models()

# Load evaluation results
@st.cache_data
def load_evaluation_results():
    """Load evaluation results."""
    try:
        import json
        with open('assets/evaluation_results.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

evaluation_results = load_evaluation_results()

# Load data
@st.cache_data
def load_data():
    """Load the dataset."""
    try:
        return pd.read_csv('data/raw/synthetic_churn_data.csv')
    except FileNotFoundError:
        return None

df = load_data()

# Model Overview Page
if page == "Model Overview":
    st.header("Model Overview")
    
    if not models:
        st.error("No trained models found. Please run the training script first.")
        st.code("python scripts/train_models.py")
    else:
        st.success(f"Found {len(models)} trained models")
        
        # Model information
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Available Models")
            for model_name in models.keys():
                st.write(f"✅ {model_name}")
        
        with col2:
            st.subheader("Model Types")
            st.write("• **Logistic Regression**: Linear baseline model")
            st.write("• **Random Forest**: Ensemble tree-based model")
            st.write("• **XGBoost**: Gradient boosting model")
            st.write("• **LightGBM**: Light gradient boosting model")
        
        # Configuration summary
        if config:
            st.subheader("Configuration Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Dataset Size", f"{config['data']['synthetic_size']:,}")
            
            with col2:
                st.metric("Test Size", f"{config['data']['test_size']*100:.0f}%")
            
            with col3:
                st.metric("Random Seed", config['data']['random_seed'])

# Data Analysis Page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    if df is None:
        st.error("No data found. Please run the training script first.")
    else:
        # Data overview
        st.subheader("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        
        with col2:
            churn_rate = df['Churn'].mean()
            st.metric("Churn Rate", f"{churn_rate:.1%}")
        
        with col3:
            avg_tenure = df['tenure'].mean()
            st.metric("Avg Tenure", f"{avg_tenure:.1f} months")
        
        with col4:
            avg_charges = df['MonthlyCharges'].mean()
            st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")
        
        # Churn distribution
        st.subheader("Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        
        fig = px.pie(
            values=churn_counts.values,
            names=['No Churn', 'Churn'],
            title="Customer Churn Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        # Select features to analyze
        numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = ['Contract', 'InternetService', 'PaymentMethod']
        
        feature_type = st.selectbox("Select feature type", ["Numerical", "Categorical"])
        
        if feature_type == "Numerical":
            selected_feature = st.selectbox("Select numerical feature", numerical_features)
            
            fig = px.histogram(
                df, x=selected_feature, color='Churn',
                title=f"Distribution of {selected_feature} by Churn Status",
                barmode='overlay'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            selected_feature = st.selectbox("Select categorical feature", categorical_features)
            
            # Churn rate by category
            churn_by_category = df.groupby(selected_feature)['Churn'].agg(['count', 'sum', 'mean']).reset_index()
            churn_by_category.columns = [selected_feature, 'Total', 'Churned', 'Churn_Rate']
            
            fig = px.bar(
                churn_by_category, x=selected_feature, y='Churn_Rate',
                title=f"Churn Rate by {selected_feature}",
                text='Churn_Rate'
            )
            fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlation")
        numerical_df = df[numerical_features + ['Churn']]
        correlation_matrix = numerical_df.corr()
        
        fig = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Feature Correlation Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

# Model Performance Page
elif page == "Model Performance":
    st.header("Model Performance")
    
    if not evaluation_results:
        st.error("No evaluation results found. Please run the training script first.")
    else:
        # Performance metrics
        st.subheader("Performance Metrics")
        
        # Create performance dataframe
        performance_data = []
        for model_name, results in evaluation_results.items():
            ml_metrics = results['ml_metrics']
            business_metrics = results['business_metrics']
            
            performance_data.append({
                'Model': model_name,
                'ROC AUC': ml_metrics['roc_auc'],
                'Precision': ml_metrics['precision'],
                'Recall': ml_metrics['recall'],
                'F1 Score': ml_metrics['f1_score'],
                'Cost Savings ($)': business_metrics['cost_savings'],
                'ROI (%)': business_metrics['roi_percent']
            })
        
        performance_df = pd.DataFrame(performance_data)
        performance_df = performance_df.sort_values('ROC AUC', ascending=False)
        
        # Display metrics table
        st.dataframe(performance_df, use_container_width=True)
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC AUC Comparison")
            fig = px.bar(
                performance_df, x='Model', y='ROC AUC',
                title="Model ROC AUC Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cost Savings Comparison")
            fig = px.bar(
                performance_df, x='Model', y='Cost Savings ($)',
                title="Model Cost Savings Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # ROC curves
        st.subheader("ROC Curves")
        fig = go.Figure()
        
        for model_name, results in evaluation_results.items():
            y_true = np.array(results['predictions']['y_true'])
            y_pred_proba = np.array(results['predictions']['y_pred_proba'])
            roc_auc = results['ml_metrics']['roc_auc']
            
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'{model_name} (AUC = {roc_auc:.3f})'
            ))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Random'
        ))
        
        fig.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calibration curves
        st.subheader("Calibration Curves")
        fig = go.Figure()
        
        for model_name, results in evaluation_results.items():
            y_true = np.array(results['predictions']['y_true'])
            y_pred_proba = np.array(results['predictions']['y_pred_proba'])
            
            from sklearn.calibration import calibration_curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            
            fig.add_trace(go.Scatter(
                x=mean_predicted_value, y=fraction_of_positives,
                mode='markers+lines',
                name=model_name
            ))
        
        # Add perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            name='Perfect Calibration'
        ))
        
        fig.update_layout(
            title="Calibration Curves",
            xaxis_title="Mean Predicted Probability",
            yaxis_title="Fraction of Positives"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Predictions Page
elif page == "Predictions":
    st.header("Customer Churn Predictions")
    
    if not models:
        st.error("No trained models found. Please run the training script first.")
    else:
        # Model selection
        selected_model = st.selectbox("Select model for predictions", list(models.keys()))
        model = models[selected_model]
        
        st.subheader("Customer Information")
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.slider("Tenure (months)", 1, 72, 24)
            monthly_charges = st.slider("Monthly Charges ($)", 20, 120, 60)
            total_charges = st.slider("Total Charges ($)", 0, 10000, 1000)
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        
        with col2:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            payment_method = st.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        
        # Additional features
        st.subheader("Additional Services")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        
        with col2:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        
        with col3:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
        
        # Create customer data
        customer_data = {
            'tenure': tenure,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'SeniorCitizen': senior_citizen,
            'Contract': contract,
            'InternetService': internet_service,
            'PaymentMethod': payment_method,
            'PhoneService': phone_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'PaperlessBilling': paperless_billing,
            'Partner': partner,
            'Dependents': dependents,
            'gender': 'Male',  # Default value
            'MultipleLines': 'No'  # Default value
        }
        
        # Convert to DataFrame
        customer_df = pd.DataFrame([customer_data])
        
        # Preprocess data (simplified version)
        try:
            # Load preprocessor
            data_splits = joblib.load('data/processed/churn_data_splits.joblib')
            scaler = data_splits['scaler']
            label_encoders = data_splits['label_encoders']
            
            # Encode categorical variables
            for col, encoder in label_encoders.items():
                if col in customer_df.columns:
                    customer_df[col] = encoder.transform(customer_df[col].astype(str))
            
            # Create additional features
            customer_df['avg_monthly_charges'] = customer_df['TotalCharges'] / (customer_df['tenure'] + 1)
            customer_df['high_value_customer'] = (customer_df['avg_monthly_charges'] > 70).astype(int)
            customer_df['long_tenure_customer'] = (customer_df['tenure'] > 30).astype(int)
            
            # Scale features
            feature_names = scaler.feature_names_in_
            customer_scaled = scaler.transform(customer_df[feature_names])
            customer_scaled_df = pd.DataFrame(customer_scaled, columns=feature_names)
            
            # Make prediction
            if st.button("Predict Churn Risk"):
                with st.spinner("Making prediction..."):
                    # Get prediction probability
                    churn_probability = model.predict_proba(customer_scaled_df)[0, 1]
                    
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Churn Probability", f"{churn_probability:.1%}")
                    
                    with col2:
                        risk_level = "High" if churn_probability > 0.7 else "Medium" if churn_probability > 0.3 else "Low"
                        st.metric("Risk Level", risk_level)
                    
                    with col3:
                        recommendation = "Immediate Action" if churn_probability > 0.7 else "Monitor Closely" if churn_probability > 0.3 else "Low Priority"
                        st.metric("Recommendation", recommendation)
                    
                    # Risk visualization
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = churn_probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Churn Risk Score"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Action recommendations
                    st.subheader("Recommended Actions")
                    
                    if churn_probability > 0.7:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>High Risk Customer - Immediate Action Required</h4>
                            <ul>
                                <li>Contact customer immediately to understand concerns</li>
                                <li>Offer personalized retention incentives</li>
                                <li>Assign dedicated account manager</li>
                                <li>Schedule follow-up calls within 48 hours</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    elif churn_probability > 0.3:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>Medium Risk Customer - Monitor Closely</h4>
                            <ul>
                                <li>Send targeted retention offers</li>
                                <li>Increase engagement through preferred channels</li>
                                <li>Monitor usage patterns weekly</li>
                                <li>Proactive outreach within 1 week</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box">
                            <h4>Low Risk Customer - Maintain Service</h4>
                            <ul>
                                <li>Continue current service level</li>
                                <li>Regular check-ins for satisfaction</li>
                                <li>Offer upsell opportunities</li>
                                <li>Monitor monthly for changes</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Feature Analysis Page
elif page == "Feature Analysis":
    st.header("Feature Analysis")
    
    if not evaluation_results:
        st.error("No evaluation results found. Please run the training script first.")
    else:
        # Model selection for feature analysis
        selected_model = st.selectbox("Select model for feature analysis", list(evaluation_results.keys()))
        
        results = evaluation_results[selected_model]
        
        if 'feature_analysis' in results and results['feature_analysis']:
            # Feature importance
            st.subheader("Feature Importance")
            
            top_features = results['feature_analysis'].get('top_features', {})
            if top_features:
                # Create feature importance plot
                features = list(top_features.keys())[:15]
                importances = list(top_features.values())[:15]
                
                fig = px.bar(
                    x=importances, y=features,
                    orientation='h',
                    title=f"Top 15 Features - {selected_model}",
                    labels={'x': 'Importance Score', 'y': 'Features'}
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Feature categories
                st.subheader("Feature Categories")
                category_importance = results['feature_analysis'].get('category_importance', {})
                
                if category_importance:
                    fig = px.pie(
                        values=list(category_importance.values()),
                        names=list(category_importance.keys()),
                        title="Feature Importance by Category"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            # SHAP analysis
            if 'shap_analysis' in results and results['shap_analysis'].get('shap_values_available'):
                st.subheader("SHAP Analysis")
                
                top_shap_features = results['shap_analysis'].get('top_shap_features', {})
                if top_shap_features:
                    features = list(top_shap_features.keys())[:15]
                    shap_values = list(top_shap_features.values())[:15]
                    
                    fig = px.bar(
                        x=shap_values, y=features,
                        orientation='h',
                        title=f"Top 15 SHAP Features - {selected_model}",
                        labels={'x': 'Mean |SHAP Value|', 'y': 'Features'}
                    )
                    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature analysis not available for this model.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Churn Prediction Model Demo | Educational and Research Purpose Only</p>
    <p>⚠️ Do not use for automated business decisions without human review</p>
</div>
""", unsafe_allow_html=True)
