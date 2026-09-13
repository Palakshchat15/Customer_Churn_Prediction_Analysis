import shap
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np

def generate_shap_explanations(model, model_name, X_train, X_test):
    print("\nGenerating SHAP Explanations...")
    
    # We use a sample of the training data to initialize the explainer to speed things up
    X_train_sample = shap.utils.sample(X_train, 100)
    
    # Sample test set to max 500 rows — SHAP on full RF test set (10k rows) takes hours
    X_test_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)
    
    if "XGBoost" in model_name:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_sample, check_additivity=False)
    else:
        # Random Forest / other tree models
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_test_sample, check_additivity=False)

    # Output path for the plot
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / "outputs"
    artifacts_dir.mkdir(exist_ok=True)
    plot_path = artifacts_dir / "shap_summary.png"
    
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    
    print(f"SHAP summary plot saved to {plot_path}")
    
    return explainer, shap_values, X_test_sample

def get_local_explanation(explainer, shap_values, X_test, customer_index, top_n=3):
    """
    Extract the top positive contributing features for a specific customer.
    These are the reasons pushing the model toward predicting 'Churn'.
    """
    # For xgboost, shap_values.values might be 2D.
    # For binary classification, we want the values pushing towards class 1 (churn)
    if len(shap_values.values.shape) == 3: # Some models return 3D arrays (N, features, classes)
        customer_shap = shap_values[customer_index, :, 1]
    else:
        customer_shap = shap_values[customer_index]
        
    feature_names = X_test.columns
    shap_vals = customer_shap.values
    feature_vals = customer_shap.data
    
    # Create a DataFrame of feature contributions
    contributions = pd.DataFrame({
        'Feature': feature_names,
        'Value': feature_vals,
        'SHAP_Value': shap_vals
    })
    
    # Sort by SHAP_Value descending to get the features pushing most towards churn
    contributions = contributions.sort_values(by='SHAP_Value', ascending=False)
    
    top_churn_drivers = contributions.head(top_n)
    
    reasons = []
    for _, row in top_churn_drivers.iterrows():
        reasons.append(f"{row['Feature']} (value: {row['Value']:.2f})")
        
    return reasons
