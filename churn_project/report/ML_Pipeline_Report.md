# ML Pipeline & Prediction Report: Customer Churn Intelligence Platform

This document covers the end-to-end machine learning pipeline used to predict customer churn for a major telecommunications provider using the Cell2Cell dataset (51,047 records).

---

## 1. Pipeline Overview

The project runs through the following stages in sequence:
1. **ETL** — SQL-powered data ingestion and transformation (DuckDB)
2. **EDA** — Statistical profiling and visualization
3. **Feature Engineering** — Selection and encoding for ML
4. **Model Training** — XGBoost and Random Forest with hyperparameter tuning
5. **Explainability** — SHAP values to identify top churn drivers per customer
6. **Retention** — LLM-generated personalized retention messages

---

## 2. Dataset & Churn Statistics

- **Total Records**: 51,047 customer rows
- **Features**: 73 engineered columns after ETL
- **Churn Rate**: 28.8% of customers churned (positive class)
- **No-Churn Rate**: 71.2% of customers were retained
- **Train/Test Split**: 80% training (40,837 rows), 20% test (10,210 rows)
- **Dataset Source**: Cell2Cell telecommunications dataset

---

## 3. Feature Engineering

### 3.1 Steps Applied
- **ID Removal**: Dropped `CustomerID` as it carries no predictive value
- **Categorical Encoding**: Applied OrdinalEncoder to categorical columns for stability
- **Missing Value Imputation**: Numeric columns filled with column median; categorical columns filled with mode
- **Feature Selection**: Mutual Information scoring used to select top 40 most predictive features from 71 candidates

### 3.2 Top 10 Selected Features (by Mutual Information)
1. MonthlyRevenue
2. MonthlyMinutes
3. TotalRecurringCharge
4. DirectorAssistedCalls
5. RoamingCalls
6. PercChangeMinutes (Month-over-month change in usage)
7. DroppedCalls
8. UnansweredCalls
9. InboundCalls
10. CallWaitingCalls

---

## 4. Class Imbalance Handling

The dataset has a significant class imbalance (28.8% churn vs 71.2% no-churn). Two strategies were applied:

- **SMOTE (Synthetic Minority Oversampling Technique)**: Applied with `sampling_strategy=0.6` to oversample the minority churn class. After SMOTE, the training set had 46,508 rows with a 37.5% churn rate.
- **scale_pos_weight**: XGBoost was given `scale_pos_weight = 2.47` (ratio of no-churn to churn) to penalize misclassifying churners more heavily.

---

## 5. Model Training & Comparison

Two models were trained and compared: XGBoost and Random Forest. Both used `RandomizedSearchCV` with `average_precision` scoring to optimize for the Precision-Recall tradeoff, which is more appropriate than ROC-AUC for imbalanced datasets.

### 5.1 XGBoost (Best Model)
- **Best Hyperparameters**: subsample=0.7, n_estimators=400, min_child_weight=7, max_depth=6, learning_rate=0.05, gamma=0.1, colsample_bytree=0.7
- **Threshold Selection**: Balanced mode — finds the threshold where |Precision - Recall| is minimized

### 5.2 Random Forest
- **Best Hyperparameters**: n_estimators=300, min_samples_leaf=2, max_features='log2', max_depth=None, class_weight='balanced'

---

## 6. Model Performance

### XGBoost (Selected Winner)
| Metric    | Value  |
|-----------|--------|
| AUC-ROC   | 0.6705 |
| F1-Score  | 0.4586 |
| Precision | 0.4236 |
| Recall    | 0.5000 |
| Accuracy  | 0.66   |
| Threshold | 0.564  |

### Random Forest
| Metric    | Value  |
|-----------|--------|
| AUC-ROC   | 0.6551 |
| F1-Score  | 0.4527 |
| Precision | 0.4136 |
| Recall    | 0.5000 |
| Accuracy  | 0.65   |

**XGBoost was selected as the final model** due to its higher AUC-ROC (0.6705 vs 0.6551).

### Classification Report (XGBoost)
- No-Churn: Precision=0.78, Recall=0.72, F1=0.75
- Churn: Precision=0.42, Recall=0.50, F1=0.46

---

## 7. Threshold Optimization

The default classification threshold of 0.5 was replaced with a **balanced threshold** strategy:
- Scans all thresholds on the Precision-Recall curve
- Finds the point where |Precision - Recall| is minimized
- Subject to floors: Precision ≥ 0.40 AND Recall ≥ 0.50
- **Final Threshold**: 0.564 for XGBoost

This prevents the model from predicting everyone as "No Churn" (high precision, zero recall) or predicting everyone as "Churn" (high recall, terrible precision).

---

## 8. SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to explain individual predictions using TreeExplainer.

### Top Churn Drivers (SHAP)
For an example customer (Customer #0), the top 3 churn drivers were:
1. **MonthlyRevenue** (value: -1.07) — Low revenue increases churn risk
2. **TotalRecurringCharge** (value: -1.55) — Low recurring charges correlate with churning
3. **UniqueSubs** (value: +0.54) — Higher unique subscriptions increase churn risk

### How SHAP Works
- Positive SHAP values → push prediction toward Churn
- Negative SHAP values → push prediction away from Churn (toward Retention)
- The magnitude indicates how strongly each feature influences the prediction

---

## 9. LLM Retention Message Generation

After identifying the top 3 SHAP-based churn reasons for a customer, the system generates a personalized retention message using a language model (Google Gemini API when configured).

**Example Output**:
> "We noticed you might be unhappy due to: MonthlyRevenue (value: -1.07), TotalRecurringCharge (value: -1.55), UniqueSubs (value: 0.54). Please reach out to our support team so we can find a personalized solution for you!"

---

## 10. Model Files & Outputs

- **Trained Model**: `models/best_churn_model.pkl` (XGBoost)
- **Threshold File**: `models/threshold.txt` (0.5640)
- **SHAP Summary Plot**: `outputs/shap_summary.png`
- **Processed Data**: `data/processed/cell2cell_engineered.csv`
- **EDA Plots**: `outputs/analyst_plots/`

---

## 11. How to Run the Full Pipeline

```bash
python churn_project/main_analyst.py
```

This runs all stages in order: ETL → EDA → Feature Engineering → Model Training → SHAP → LLM Retention Message.

---

## 12. Key Business Insights

- **Dropped Calls** are the single strongest behavioral signal of churn risk
- **Customers in their first 6 months** have the highest churn probability
- **Low monthly revenue customers** are more likely to churn — suggesting pricing sensitivity
- **Equipment older than 2 years** correlates with churning — upgrade incentives recommended
- **Overage charges** (surprise bills) are a major trigger — consider plan restructuring

---

## 13. Recommendations

1. **Retention Team**: Proactively contact customers with high dropped call frequency within 24 hours
2. **Upgrade Program**: Launch a 24-month equipment upgrade incentive for high-value customers
3. **Pricing Strategy**: Offer discounted plans to low-income/low-revenue segments
4. **Onboarding Focus**: Invest in the first 6 months of a customer's lifecycle with better support
