# Kraljic Portfolio Matrix Machine Learning Project

## Project Overview

This project applies machine learning algorithms to classify products based on the Kraljic portfolio model into four strategic purchasing categories: Strategic, Bottleneck, Leverage, and Non-Critical. The dataset includes product, supplier, and risk-related features.

## Dataset Description

- **Total Records:** 1,000
- **Features:** 11, including product information, supplier location, lead times, order volumes, cost, and risk scores.
- **Target Variable:** `Kraljic_Category` with 4 classes.

## Exploratory Data Analysis (EDA)

- Visualizations of class distributions, correlations, and feature distributions.
- Summary statistics and data integrity checks.

## Data Preprocessing

- Encoding categorical variables (`LabelEncoder`).
- Feature scaling with `StandardScaler`.
- Train-test split with stratification.

## Machine Learning Models

The following classification models have been implemented and evaluated:

- Logistic Regression
- Random Forest (Bagging)
- XGBoost (Boosting)
- Support Vector Machine (SVM)
- Gradient Boosting Classifier
- K-Nearest Neighbors (KNN)
- Naive Bayes (GaussianNB)
- Decision Tree

## Model Evaluation

Metrics used include:

- Accuracy
- Precision
- Recall
- F1 Score

Confusion matrices and classification reports are provided for each model.

## Summary of Results

| Model                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression   | 0.985    | 0.9858    | 0.985  | 0.985    |
| Random Forest        | 1.000    | 1.0000    | 1.000  | 1.000    |
| XGBoost              | 0.975    | 0.9758    | 0.975  | 0.975    |
| SVM                  | 0.790    | 0.8312    | 0.790  | 0.783    |
| Gradient Boosting    | 1.000    | 1.0000    | 1.000  | 1.000    |
| K-Nearest Neighbors  | 0.860    | 0.8636    | 0.860  | 0.860    |
| Naive Bayes          | 1.000    | 1.0000    | 1.000  | 1.000    |
| Decision Tree        | 0.990    | 0.9904    | 0.990  | 0.990    |

*(Values as computed during evaluation)*

## Visualizations

- Confusion matrices per model.
- Performance comparison bar chart with unique colors for accuracy.





