import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, precision_recall_curve
)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def find_best_threshold(model, X_test, y_test,
                         mode: str = 'balanced',
                         min_precision: float = 0.40,
                         min_recall:    float = 0.50):
    """
    Find the optimal classification threshold on the PR curve.

    Modes
    -----
    'balanced'  : Pick the threshold where |precision - recall| is minimised,
                  subject to both being above their floor values.
                  → Targets ~equal precision & recall (e.g. 0.52 / 0.55).

    'recall'    : Maximise recall while keeping precision >= min_precision.
                  → Catches as many churners as possible.

    'f1'        : Maximise F1 (previous behaviour).
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

    # Slice to the threshold range (last element of precisions/recalls has no threshold)
    prec = precisions[:-1]
    rec  = recalls[:-1]
    f1   = 2 * (prec * rec) / (prec + rec + 1e-8)

    # Build a validity mask: both precision and recall must meet their floors
    valid = (prec >= min_precision) & (rec >= min_recall)

    if not valid.any():
        print(f"  [Threshold] No point satisfies precision>={min_precision} & recall>={min_recall}. "
              f"Falling back to best F1.")
        best_idx = np.argmax(f1)
    elif mode == 'balanced':
        # Among valid points, minimise |precision - recall|
        gap = np.abs(prec - rec)
        masked_gap = np.where(valid, gap, np.inf)
        best_idx = np.argmin(masked_gap)
        print(f"  [Threshold] Balanced mode → precision={prec[best_idx]:.3f}, "
              f"recall={rec[best_idx]:.3f}, gap={gap[best_idx]:.3f}")
    elif mode == 'recall':
        # Among valid points, maximise recall
        masked_rec = np.where(valid, rec, -1)
        best_idx = np.argmax(masked_rec)
    else:  # 'f1'
        masked_f1 = np.where(valid, f1, 0)
        best_idx = np.argmax(masked_f1)

    best_thresh = thresholds[best_idx]
    return best_thresh, f1[best_idx]


def evaluate_model(model, X_test, y_test, model_name, threshold=None):
    y_prob = model.predict_proba(X_test)[:, 1]

    if threshold is None:
        threshold, _ = find_best_threshold(model, X_test, y_test)

    y_pred = (y_prob >= threshold).astype(int)

    auc_roc    = roc_auc_score(y_test, y_prob)
    f1         = f1_score(y_test, y_pred, zero_division=0)
    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred, zero_division=0)

    metrics = {
        'AUC-ROC':   auc_roc,
        'F1-Score':  f1,
        'Precision': precision,
        'Recall':    recall,
        'Threshold': threshold,
    }

    print(f"\n--- {model_name} Performance (threshold={threshold:.3f}) ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    return metrics, threshold


# ──────────────────────────────────────────────
# Model definitions with tuned hyper-parameters
# ──────────────────────────────────────────────

def _build_xgb(n_churn, n_no_churn):
    # scale_pos_weight tells XGBoost: "missing a churner is this many times
    # worse than a false alarm" — directly fights the majority-class bias.
    scale_pos = n_no_churn / max(n_churn, 1)
    print(f"  XGB scale_pos_weight = {scale_pos:.2f}  (higher → more recall)")
    param_dist = {
        'n_estimators':     [200, 300, 400],
        'max_depth':        [4, 5, 6, 7],
        'learning_rate':    [0.03, 0.05, 0.1],
        'subsample':        [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [3, 5, 7],
        'gamma':            [0, 0.1, 0.2],
    }
    base = XGBClassifier(
        scale_pos_weight=scale_pos,
        eval_metric='aucpr',   # area under PR curve — better than logloss for imbalanced data
        random_state=42,
        n_jobs=-1,
    )
    return base, param_dist


def _build_rf():
    param_dist = {
        'n_estimators': [200, 300],
        'max_depth':    [8, 12, None],
        'max_features': ['sqrt', 'log2'],
        'min_samples_leaf': [2, 4, 6],
        'class_weight': ['balanced', 'balanced_subsample'],
    }
    base = RandomForestClassifier(random_state=42, n_jobs=-1)
    return base, param_dist


# ──────────────────────────────────────────────
# Main training function
# ──────────────────────────────────────────────

def train_and_compare(X_train, X_test, y_train, y_test):
    print("\n" + "="*55)
    print("  OPTIMIZED MODEL TRAINING")
    print("="*55)

    # ── Class imbalance ──────────────────────────────────────
    n_churn    = int(y_train.sum())
    n_no_churn = int(len(y_train) - n_churn)
    churn_rate = n_churn / len(y_train)
    print(f"\nClass distribution — Churn: {n_churn} ({churn_rate:.1%})  |  No-Churn: {n_no_churn}")

    # Apply SMOTE only when minority class < 40 %
    # sampling_strategy=0.6 → churners become 60% of non-churners in training
    # (deliberately avoids 50/50 to prevent over-aggressive recall at expense of precision)
    if churn_rate < 0.40:
        print("Applying SMOTE (minority class under-represented)…")
        smote = SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.6)
        X_train_r, y_train_r = smote.fit_resample(X_train, y_train)
        new_rate = y_train_r.mean()
        print(f"  Resampled → {X_train_r.shape[0]} rows  |  churn rate: {new_rate:.1%}")
    else:
        X_train_r, y_train_r = X_train, y_train
        print("SMOTE skipped — classes already balanced.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results   = {}
    thresholds = {}

    # ── 1. XGBoost ──────────────────────────────────────────
    # 'average_precision' = area under the PR curve.
    # Optimizing this directly maximises precision+recall balance
    # instead of optimising for AUC-ROC which ignores TN/FP ratio.
    CV_SCORING = 'average_precision'

    print("\n[1/2] Tuning XGBoost with RandomizedSearchCV…")
    xgb_base, xgb_params = _build_xgb(n_churn, n_no_churn)
    xgb_search = RandomizedSearchCV(
        xgb_base, xgb_params,
        n_iter=20, scoring=CV_SCORING, cv=cv,
        random_state=42, n_jobs=-1, verbose=0,
    )
    xgb_search.fit(X_train_r, y_train_r)
    best_xgb = xgb_search.best_estimator_
    print(f"  Best XGB params: {xgb_search.best_params_}")
    xgb_metrics, xgb_thresh = evaluate_model(best_xgb, X_test, y_test, "XGBoost (Tuned)")
    results['XGBoost']   = xgb_metrics
    thresholds['XGBoost'] = xgb_thresh

    # ── 2. Random Forest ────────────────────────────────────
    print("\n[2/2] Tuning Random Forest with RandomizedSearchCV…")
    rf_base, rf_params = _build_rf()
    rf_search = RandomizedSearchCV(
        rf_base, rf_params,
        n_iter=15, scoring=CV_SCORING, cv=cv,
        random_state=42, n_jobs=-1, verbose=0,
    )
    rf_search.fit(X_train_r, y_train_r)
    best_rf = rf_search.best_estimator_
    print(f"  Best RF params: {rf_search.best_params_}")
    rf_metrics, rf_thresh = evaluate_model(best_rf, X_test, y_test, "Random Forest (Tuned)")
    results['Random Forest']   = rf_metrics
    thresholds['Random Forest'] = rf_thresh

    # ── Model selection ──────────────────────────────────────
    print("\n--- Model Comparison ---")
    comparison = pd.DataFrame(results).T
    print(comparison.to_string())

    if results['XGBoost']['AUC-ROC'] >= results['Random Forest']['AUC-ROC']:
        best_model = best_xgb
        best_name  = "XGBoost"
        print(f"\n✔ Selected: XGBoost  (AUC-ROC = {results['XGBoost']['AUC-ROC']:.4f})")
    else:
        best_model = best_rf
        best_name  = "Random Forest"
        print(f"\n✔ Selected: Random Forest  (AUC-ROC = {results['Random Forest']['AUC-ROC']:.4f})")

    # ── Persist artefacts ────────────────────────────────────
    models_dir = Path(__file__).resolve().parent.parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "best_churn_model.pkl"
    joblib.dump(best_model, model_path)
    joblib.dump(thresholds[best_name], models_dir / "best_threshold.pkl")
    print(f"\nModel saved → {model_path}")
    print(f"Threshold saved → {thresholds[best_name]:.4f}")

    return best_model, best_name
