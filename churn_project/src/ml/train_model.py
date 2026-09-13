import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib
from pathlib import Path
from feature_engineering import engineer_features

def train_and_evaluate():
    """
    Trains an XGBoost model and evaluates its performance on churn prediction.
    """
    print("="*50)
    print("  TRAINING MACHINE LEARNING MODEL (XGBoost)")
    print("="*50)
    
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / "data" / "processed" / "cell2cell_engineered.csv"
    
    if not data_path.exists():
        print(f"ERROR: Cannot find {data_path}. Please run ETL first.")
        return
        
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = engineer_features(df)
    
    print("\nTraining XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    print("\nEvaluating Model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save the model
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "xgboost_churn_model.pkl"
    joblib.dump(model, model_path)
    
    print(f"\nModel saved successfully to: {model_path}")

if __name__ == "__main__":
    train_and_evaluate()
