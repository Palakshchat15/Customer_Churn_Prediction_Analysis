import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from pathlib import Path
import joblib


def engineer_features(df, k_best: int = 40):
    """
    Prepares raw/cleaned data for Machine Learning.

    Improvements over v1:
    - OrdinalEncoder instead of LabelEncoder (avoids silent ordinal assumption).
    - Mutual-information feature selection — keeps the top-k most informative
      features and discards noisy ones (was 71, now capped at k_best=40).
    - NaN imputation uses column median (numeric) / mode (categorical).

    Returns X_train, X_test, y_train, y_test.
    """
    print("\nEngineering features for Machine Learning…")
    df = df.copy()

    # ── 1. Drop ID-like columns ───────────────────────────────
    id_cols = [c for c in df.columns if c.lower() in ('customerid', 'customer_id', 'id')]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
        print(f"  Dropped ID columns: {id_cols}")

    # ── 2. Encode target ──────────────────────────────────────
    if df['Churn'].dtype == object:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0, 1: 1, 0: 0})
    df = df.dropna(subset=['Churn'])
    df['Churn'] = df['Churn'].astype(int)

    # ── 3. Separate features / target ────────────────────────
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # ── 4. Impute missing values ──────────────────────────────
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "Unknown")

    # ── 5. Encode categoricals with OrdinalEncoder ───────────
    ordinal_encoders = {}
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols].astype(str))
        ordinal_encoders['ordinal'] = enc
    print(f"  Encoded {len(cat_cols)} categorical column(s).")

    # ── 6. Train-Test split (stratified) ─────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── 7. Scale ──────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # ── 8. Mutual-information feature selection ───────────────
    n_features = X_train_scaled.shape[1]
    k = min(k_best, n_features)
    print(f"  Selecting top {k} features (from {n_features}) via mutual information…")
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel  = selector.transform(X_test_scaled)

    selected_cols = X.columns[selector.get_support()].tolist()
    print(f"  Selected features: {selected_cols[:10]}{'…' if len(selected_cols) > 10 else ''}")

    # Rebuild DataFrames with column names
    X_train_final = pd.DataFrame(X_train_sel, columns=selected_cols)
    X_test_final  = pd.DataFrame(X_test_sel,  columns=selected_cols)

    # ── 9. Persist artefacts ──────────────────────────────────
    artifacts_dir = Path(__file__).resolve().parent.parent.parent / "models"
    artifacts_dir.mkdir(exist_ok=True)

    joblib.dump(scaler,            artifacts_dir / "scaler.pkl")
    joblib.dump(ordinal_encoders,  artifacts_dir / "label_encoders.pkl")
    joblib.dump(selector,          artifacts_dir / "feature_selector.pkl")
    joblib.dump(selected_cols,     artifacts_dir / "selected_features.pkl")

    print(f"\nFeature engineering complete.")
    print(f"  Train: {X_train_final.shape}  |  Test: {X_test_final.shape}")
    churn_rate = y_train.mean()
    print(f"  Train churn rate: {churn_rate:.1%}")

    return X_train_final, X_test_final, y_train, y_test


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    input_path = project_root / "data" / "processed" / "cell2cell_engineered.csv"

    if input_path.exists():
        df = pd.read_csv(input_path)
        X_train, X_test, y_train, y_test = engineer_features(df)
    else:
        print(f"Please place data at {input_path}")
