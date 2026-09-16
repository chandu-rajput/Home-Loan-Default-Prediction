# Home Credit Loan Default Prediction

An end-to-end machine learning project for predicting loan default risk using customer application data and historical credit behavior.

## Project Overview

This project aims to predict whether a customer is likely to default on a loan using the Home Credit dataset.

The project combines the main customer application data with historical financial information from:

* Bureau records
* Bureau balance history
* Previous applications
* POS/Cash loan history
* Credit card history
* Installment payment history

The historical datasets are aggregated at the customer level and merged with the main application data to create a final modeling dataset.

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
Data Validation
       ↓
Feature Selection
       ↓
Baseline Model Comparison
       ↓
Hyperparameter Tuning
       ↓
Threshold Optimization
       ↓
Final LightGBM Model
```

## Models Evaluated

The following classification models were evaluated:

* Decision Tree
* Random Forest
* XGBoost
* LightGBM

LightGBM achieved the strongest validation performance and was selected for hyperparameter tuning and final evaluation.

## Feature Selection

Feature selection was performed using:

* LightGBM Gain importance
* Correlation-based redundancy analysis
* Business interpretation of credit-risk features

This helped reduce unnecessary and highly redundant features while maintaining model performance.

## Final Model

The final model is a tuned **LightGBM classifier** with a classification threshold of **0.15**.

The threshold was optimized on the validation set to improve the detection of potential loan defaults.

### Final Test Results

| Metric    |      Score |
| --------- | ---------: |
| ROC-AUC   | **0.7866** |
| Recall    | **46.75%** |
| Precision | **26.54%** |
| F1-Score  | **33.86%** |

### Confusion Matrix

```text
                 Predicted
                 0       1
Actual  0      37584   4819
        1       1983   1741
```

The model identifies a substantial portion of actual defaults while prioritizing Recall through threshold optimization.

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

The raw datasets are not included in this repository because of their large size.

Download the required Home Credit datasets separately and place them in the local `data/` directory before running the notebooks.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* LightGBM
* Jupyter Notebook
* Joblib

## Key Learning Outcomes

This project demonstrates an end-to-end credit-risk machine learning workflow, including:

* Working with multiple related financial datasets
* Customer-level feature aggregation
* Feature selection and redundancy reduction
* Imbalanced classification
* Model comparison
* Hyperparameter tuning
* Classification threshold optimization
* Final model evaluation using ROC-AUC, Recall, Precision, and F1-Score


## 👤 Author

**Chandrapal Rajput**

* Linkedin : https://www.linkedin.com/in/chandrapal-deora/

---
