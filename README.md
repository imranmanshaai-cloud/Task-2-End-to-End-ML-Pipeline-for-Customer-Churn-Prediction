# End-to-End Machine Learning Pipeline for Customer Churn Prediction

## Objective
The objective of this project is to build a reusable and production-ready machine learning pipeline for predicting customer churn using structured customer data. The project demonstrates best practices in data preprocessing, model training, hyperparameter tuning, and pipeline export using Scikit-learn.

---

## Dataset
- Dataset Structure: Telco Customer Churn–style dataset
- Target Variable: **Churn** (Yes / No)
- Features include:
  - Demographic information
  - Service usage details
  - Billing and contract information

> Note: Due to dataset accessibility constraints, a synthetic dataset was created that closely mirrors the structure and characteristics of the original Telco Customer Churn dataset. This approach preserves the validity of the machine learning pipeline and modeling workflow.

---

## Methodology

### 1. Data Preparation
- Converted `TotalCharges` to numerical format
- Removed non-predictive identifiers
- Performed stratified train–test split

### 2. Preprocessing Pipeline
- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- Combined preprocessing using `ColumnTransformer`

### 3. Model Development
- Baseline Model: Logistic Regression
- Advanced Model: Random Forest Classifier
- Both models integrated into a unified pipeline

### 4. Hyperparameter Tuning
- Performed using `GridSearchCV`
- Cross-validation applied on the full pipeline
- Optimized Random Forest parameters

### 5. Model Export
- Exported the complete trained pipeline using `joblib`
- Reloaded pipeline to verify reusability

---

## Model Evaluation
Models were evaluated using:
- Accuracy
- F1-score
- Classification Report

Random Forest demonstrated the ability to capture non-linear patterns in the data, while Logistic Regression served as a strong baseline.

---

## Key Skills Gained
- Scikit-learn Pipeline API
- ColumnTransformer for preprocessing
- Hyperparameter tuning with GridSearchCV
- Model export and reuse with joblib
- Production-ready ML practices

---

## How to Run
1. Open the notebook in Google Colab or Jupyter Notebook
2. Run all cells sequentially
3. The trained pipeline will be saved as:

