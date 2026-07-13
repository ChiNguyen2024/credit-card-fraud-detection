# Credit Card Fraud Detection

![Python](https://img.shields.io/badge/Python-3.12-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green)
![Optuna](https://img.shields.io/badge/Optuna-4.9-purple)
![SHAP](https://img.shields.io/badge/SHAP-0.52-orange)

A binary classification project that detects fraudulent credit card transactions on a severely imbalanced dataset (0.17% fraud). The focus is on the business tradeoff at the heart of fraud detection: catching fraud (avoiding financial loss) while minimizing false positives (avoiding customer friction).

## Business Problem

Credit card fraud is costly, and the challenge is asymmetric: a missed fraud (false negative) means direct financial loss, while a wrongly blocked legitimate transaction (false positive) erodes customer trust. This project builds a model that balances both — maximizing fraud capture while keeping legitimate customers unaffected.

## Data

The [ULB Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) contains 284,807 real European cardholder transactions from September 2013, of which only 492 (0.17%) are fraudulent.

- `V1`–`V28` — anonymized features from a PCA transformation (for privacy)
- `Time`, `Amount` — the only non-transformed features
- `Class` — target (0 = legitimate, 1 = fraud)

> The dataset is not included in this repository due to size. Download it from Kaggle and place `creditcard.csv` in the project root.

## Workflow

1. **EDA** — inspected distributions, confirmed no missing values, identified severe class imbalance and a right-skewed `Amount` distribution
2. **Phik correlation** — measured feature association with the target (robust for numeric–binary relationships), identifying V10–V12, V14, V17, V18 as most correlated with fraud
3. **Preprocessing** — stratified train/test split; `RobustScaler` on `Amount` and `Time` (chosen for resilience to outliers given the skew), PCA features passed through unchanged
4. **Modeling** — LightGBM pipeline with `class_weight='balanced'`, tuned via `OptunaSearchCV` (scoring on PR-AUC, the appropriate metric for imbalanced data)
5. **Evaluation** — PR-AUC, F1, and confusion matrix analysis with business interpretation
6. **Interpretability** — SHAP analysis to identify the strongest fraud drivers

## Key Findings

- **PR-AUC = 0.88** and **F1 = 0.85** on the held-out test set — in the top range of published results on this benchmark
- The model correctly identified **91 fraudulent transactions**, missed **22**, and wrongly flagged only **10** legitimate transactions (false positive rate 0.014%)
- **SHAP** identified `V4` and `V14` as the dominant fraud indicators; `Amount` and `Time` also contributed meaningfully despite being the only non-anonymized features
- Accuracy was deliberately avoided as a metric — with 99.8% legitimate transactions it would be misleading

## Tech Stack

`Python` · `pandas` · `scikit-learn` · `LightGBM` · `Optuna` · `SHAP` · `phik` · `matplotlib` · `seaborn`
 
## Repository Structure
credit-card-fraud-detection/
├── fraud_detection.ipynb   # main notebook (EDA → modeling → SHAP)
├── requirements.txt
├── README.md
└── .gitignore

## How to Run
```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle into the project root, then run the notebook top to bottom.
