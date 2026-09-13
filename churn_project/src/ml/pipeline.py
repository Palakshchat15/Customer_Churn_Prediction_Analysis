import pandas as pd
from pathlib import Path

from src.ml.feature_engineering import engineer_features
from src.ml.model_training import train_and_compare
from src.ml.explainability import generate_shap_explanations, get_local_explanation
from src.ml.retention_llm import generate_retention_message

def run_pipeline():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "processed" / "cell2cell_engineered.csv"
    
    if not data_path.exists():
        print(f"ERROR: Could not find dataset at {data_path}")
        return
        
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 1. Feature Engineering
    X_train, X_test, y_train, y_test = engineer_features(df)
    
    # 2. Model Training & Comparison
    print("\n=============================================")
    print("STEP 1: Model Comparison (SMOTE + Training)")
    print("=============================================")
    best_model, best_name = train_and_compare(X_train, X_test, y_train, y_test)
    
    # 3. SHAP Explainability
    print("\n=============================================")
    print("STEP 2: SHAP Explainability")
    print("=============================================")
    explainer, shap_values, X_test_sample = generate_shap_explanations(best_model, best_name, X_train, X_test)
    
    # Find a sample customer who actually churned (or predicted to churn)
    # For demonstration, we just pick the first customer in the test set with y_test == 1
    high_risk_indices = y_test[y_test == 1].index
    if len(high_risk_indices) > 0:
        # Get the iloc index corresponding to the test set DataFrame
        customer_index = 0 # taking the first one in the X_test array
        
        # Get the local SHAP explanation
        churn_reasons = get_local_explanation(explainer, shap_values, X_test_sample, customer_index, top_n=3)
        
        print(f"\nTop churn drivers for Customer #{customer_index}:")
        for reason in churn_reasons:
            print(f" - {reason}")
            
        # 4. LLM Integration
        print("\n=============================================")
        print("STEP 3: LLM Retention Message Generation")
        print("=============================================")
        print("Generating retention message based on the customer's top churn reasons...")
        
        message = generate_retention_message(churn_reasons)
        print("\n--- Generated Message ---")
        print(message)
        print("-------------------------\n")
    else:
        print("No high-risk churn customers found in the test set to demonstrate LLM message.")
        
    print("\n=============================================")
    print("Pipeline Execution Completed.")
    print("=============================================")

if __name__ == "__main__":
    run_pipeline()
