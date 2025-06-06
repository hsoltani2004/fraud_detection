
# Fraud Detection Project - Full Summary Report

## 1. Introduction
Fraud detection is a critical aspect of the financial ecosystem to minimize financial losses. This project presents a machine learning pipeline to detect fraudulent transactions with high precision and recall, tailored for highly imbalanced datasets

---

## 2. Dataset Overview
The dataset contains **99,615** transactions with the following attributes:
- **Originator**: Sender bank code
- **Beneficiary**: Receiver bank code
- **Date and Time**: When the transaction occurred
- **Type**: Transaction type (e.g., MT103, MT202)
- **Currency**: Currency type (all values normalized to USD)
- **Value**: Transaction amount
- **Aggregate Value**: Sum of transactions
- **Aggregate Volume**: Number of transactions
- **Flag**: Fraud label (0: Normal, 1: Fraudulent)

The dataset is highly imbalanced, with less than 2% fraud transactions

---

## 3. Data Pre-processing

### 3.1 Data Cleaning
- **Extracted Country Codes**: From Originator and Beneficiary bank codes
- **Extracted Transaction Hour**: From time field
- **Dropped Redundant Fields**: Originator, Beneficiary, raw Date and Time fields
- **Handled Missing Values**: Confirmed no missing values post-cleaning

### 3.2 Feature Engineering
- **Log Transformation**: Applied to skewed fields (`value`, `aggregate_value`, `aggregate_volume`).
- **Label Encoding**: Applied to categorical features (`originator_country`, `beneficiary_country`, `currency`, `type`)

### 3.3 Outlier Treatment
- **IQR Method**: Capped extreme values in log-transformed fields

### 3.4 Data Scaling
- **StandardScaler**: Used for numerical features to normalize feature distributions

### 3.5 Feature Selection
- **ANOVA F-test**: Selected features based on their discriminative power relative to fraud labels

---

## 4. Model Development

### 4.1 Handling Imbalanced Dataset
- **SMOTE**: Synthetic Minority Oversampling Technique to balance classes in the training set

### 4.2 Models Trained
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble of decision trees with bagging
- **XGBoost**: Gradient Boosting with regularization
- **LightGBM**: Fast and scalable gradient boosting

### 4.3 Hyperparameter Tuning
- **GridSearchCV**: Performed for each model using 3-fold cross-validation
- **Scoring Metric**: ROC AUC Score

---

## 5. Evaluation Metrics
- **Accuracy**: General correctness.
- **Precision**: Focused on reducing false positives
- **Recall (Sensitivity)**: Critical in detecting as many frauds as possible
- **F1-Score**: Balance between Precision and Recall
- **ROC AUC Score**: Overall model performance; higher the better
- **Confusion Matrix**: True Positives (TP), False Positives (FP), True Negatives (TN), False Negatives (FN)
- **Precision-Recall Curve**: Emphasizes performance on imbalanced data

---

## 6. Results
| Model               | ROC AUC Score | Precision | Recall | F1-Score |
|---------------------|---------------|-----------|--------|----------|
| Logistic Regression  | 0.923        | 0.87      | 0.82   | 0.85      |
| Random Forest        | 0.998        | 0.98      | 0.99   | 0.98     |
| XGBoost              | 0.999          | 1.00      | 0.99   | 0.99     |
| LightGBM             | 0.998          | 0.99      | 0.98   | 0.98     |

**Best Model**: **XGBoost** (ROC AUC = 0.999)

---

## 7. Deployment

### Streamlit Standalone App
- **Framework**: Streamlit for frontend interaction
- **Backend**: Model is loaded inside Streamlit
- **User Input**: Form fields for transaction features
- **Output**: Fraud prediction and fraud probability

---

## 8. Conclusion
Fraud detection models must prioritize **recall** to catch as many frauds as possible, without sacrificing too much **precision** to avoid unnecessary customer inconvenience. The ensemble models (XGBoost and LightGBM) provided excellent performance and would be strong candidates for production deployment.

---


