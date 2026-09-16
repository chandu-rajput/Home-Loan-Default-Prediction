# Home Credit Loan Default Prediction

An end-to-end machine learning project for predicting loan default risk using customer application data and historical credit behavior.

## Project Overview

This project focuses on predicting whether a customer is likely to default on a loan. The analysis combines the main application dataset with historical bureau records, previous applications, POS/cash loan history, credit-card behavior, and installment payment history.

The workflow includes data cleaning, historical data aggregation, feature selection, model comparison, hyperparameter tuning, and threshold optimization.

## Project Workflow

```text
Application Data
       +
Bureau
       +
Bureau Balance
       +
Previous Applications
       +
POS/Cash Balance
       +
Credit Card Balance
       +
Installment Payments
       ↓
Customer-Level Feature Aggregation
       ↓
Final Modeling Dataset
       ↓
Feature Selection
       ↓
Model Comparison
       ↓
Hyperparameter Tuning
       ↓
Threshold Optimization
       ↓
Final LightGBM Model
```

## Models

The following models were evaluated:

* Decision Tree
* Random Forest
* XGBoost
* LightGBM

LightGBM achieved the best validation performance and was selected for final tuning.

## Feature Selection

Feature selection was performed using:

* LightGBM Gain importance
* Correlation-based redundancy analysis
* Business interpretation of credit-risk features

## Final Model

The final model uses LightGBM with an optimized classification threshold of **0.15**.

### Final Test Results

| Metric    |  Score |
| --------- | -----: |
| ROC-AUC   | 0.7866 |
| Recall    | 46.75% |
| Precision | 26.54% |
| F1-Score  | 33.86% |

The threshold was optimized to improve the model's ability to identify potential loan defaults.

## Repository Structure

```text
home-credit-default-prediction/
│
├── notebooks/
│   ├── 01_application_eda.ipynb
│   ├── 02_bureau.ipynb
│   ├── 03_bureau_balance.ipynb
│   ├── 04_previous_application.ipynb
│   ├── 05_pos_cash_balance.ipynb
│   ├── 06_credit_card_balance.ipynb
│   ├── 07_installments_payments.ipynb
│   └── 08_final_modeling.ipynb
│
├── models/
│   └── home_credit_final_model.pkl
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

The raw Home Credit datasets are not included in this repository because of their large size.

Download the dataset separately and place the required files in the local `data/` directory.

## Tools and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* LightGBM
* Jupyter
* Joblib
