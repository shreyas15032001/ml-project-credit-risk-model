# Credit Risk Modeling

A default-risk classification model for personal loans, built on merged customer,
loan, and credit-bureau data. Given an applicant's profile, the model estimates
their probability of default and converts it into a 300–900 credit score band
(Poor / Average / Good / Excellent), served through a small Streamlit app.

## Problem

Lenders need to estimate, at the point of application, how likely a borrower is
to default — and they need that estimate to be interpretable, not just accurate,
since credit decisions have to be explainable to regulators and applicants alike.
This project builds that estimator end-to-end: from raw records to a scored,
deployable prediction.

## Data

Three sources, joined on customer/loan IDs:
- **Customer data** — demographics, income, residence type
- **Loan data** — loan amount, tenure, purpose, type, disbursal details
- **Bureau data** — credit utilization, delinquency history, open/closed accounts

## Approach

- **Cleaning** — fixed inconsistent categorical labels, removed rows with
  data-entry errors (e.g. processing fee exceeding loan amount), handled
  missing values — all done *after* the train/test split to avoid leakage.
- **Feature engineering** — derived loan-to-income ratio, delinquent-months-to-
  loan-tenure ratio, and average days-past-due per delinquency, since raw
  bureau fields were less predictive than these ratios.
- **Feature selection** — used **Variance Inflation Factor (VIF)** to drop
  multicollinear numeric features, and **Weight of Evidence / Information
  Value (WOE/IV)** — the standard credit-risk technique for scoring how much a
  categorical (or binned numeric) feature separates good vs. bad borrowers —
  to select categorical predictors.
- **Modeling** — compared Logistic Regression, Random Forest, and XGBoost.
  Tuned hyperparameters with **Optuna** (Bayesian search) and
  **RandomizedSearchCV**, optimizing for **macro F1 / recall** rather than
  accuracy, since missing an actual defaulter is far costlier than a false
  alarm on a good borrower.
- **Class imbalance** — defaults were a small minority class; addressed via
  random undersampling and **SMOTE-Tomek**, which improved recall on the
  default class substantially over the naive baseline.
- **Final model** — Logistic Regression (post SMOTE-Tomek), chosen over
  XGBoost despite a marginally lower F1, because its coefficients stay
  directly interpretable — a real constraint in credit risk, not just a
  nice-to-have.

## Evaluation

- **ROC-AUC** for overall discrimination.
- **Rank-ordering by decile** — bucketing applicants by predicted default
  probability and checking that observed default rates decrease monotonically
  down the deciles, which confirms the model ranks risk correctly, not just
  classifies it.
- **KS statistic** — measures the maximum separation between the cumulative
  distributions of defaulters and non-defaulters. This model reached a KS of
  **86.4, concentrated in the top 3 deciles** (a KS above ~40 is generally
  considered a strong model in credit scoring) — meaning the riskiest ~30% of
  applicants account for the large majority of actual defaults.

## App

A Streamlit interface (`app/main.py`) takes applicant inputs, calls the saved
model via `app/prediction_helper.py`, and returns a default probability, a
300–900 credit score, and a risk rating.

## Tech stack

Python, pandas, numpy, scikit-learn, XGBoost, imbalanced-learn (SMOTE-Tomek),
Optuna, statsmodels (VIF), Streamlit, joblib.

## Project structure

```
credit-risk-modeling/
├── notebooks/
│   └── credit_risk_model.ipynb   # EDA, feature engineering, modeling, evaluation
├── app/
│   ├── main.py                   # Streamlit UI
│   └── prediction_helper.py      # loads model, scores new applicants
├── artifacts/
│   └── model_data.joblib         # trained model + scaler + feature list
├── data/
│   ├── customers.csv
│   └── bureau_data.csv
└── requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
streamlit run app/main.py
```

## Limitations & next steps

- Trained on a single snapshot of data — no time-based validation, so it's
  untested against portfolio drift over time.
- Class imbalance was handled via resampling rather than cost-sensitive
  learning; a threshold analysis tied to actual lending cost/loss figures
  would make the recall-vs-precision tradeoff more concrete.
- No SHAP/feature-importance breakdown yet beyond raw logistic coefficients —
  would help validate that the model's reasoning matches domain intuition.
