# Kraljic Portfolio Matrix Machine Learning Project

## Project Overview

This project applies machine learning algorithms to classify products based on the Kraljic portfolio model into four strategic purchasing categories: Strategic, Bottleneck, Leverage, and Non-Critical. The dataset includes product, supplier, and risk-related features.

## Dataset Description

- **Total Records:** 1,000
- **Features:** 11, including product information, supplier location, lead times, order volumes, cost, and risk scores.
- **Target Variable:** `Kraljic_Category` with 4 classes.
- Product_ID: Unique product identifier
- Product_Name: Name/type of the product (e.g., Semiconductors, Pharma APIs)
- Supplier_Region: Region of the supplier (Asia, Europe, Global, etc.)
- Lead_Time_Days: Number of days from order to receipt
- Order_Volume_Units: Units ordered
- Cost_per_Unit: Price per unit
- Supply_Risk_Score: Risk score for supply chain vulnerabilities
- Profit_Impact_Score: Impact on profit if supply is disrupted
- Environmental_Impact: Environmental impact score
- Single_Source_Risk: Binary feature if single sourcing is a risk (Yes/No)
- Kraljic_Category: Target variable with categories (Strategic, Bottleneck, Leverage, Non-Critical)

## Exploratory Data Analysis (EDA)

- Visualizations of class distributions, correlations, and feature distributions.
- Class distribution visualization showing balanced categories.
- Correlation heatmaps to understand relationships among numeric variables.
- Boxplots for understanding how cost and other features vary by category.
- Inspection of missing values and data types.
- Summary statistics and data integrity checks.

## Data Preprocessing

- Encoding categorical variables (`LabelEncoder`).
- Feature scaling with `StandardScaler`.
- Train-test split with stratification.

## Machine Learning Models

The following classification models have been implemented and evaluated:

- Logistic Regression: Baseline linear model for classification.
- Random Forest: Bagging ensemble of decision trees to reduce variance.
- XGBoost: Gradient boosting model for improved accuracy on structured data.
- Support Vector Machine (SVM): Kernel-based model effective for complex boundaries.
- Gradient Boosting Classifier: Another boosting model for comparison.
- K-Nearest Neighbors (KNN): Instance-based learning using neighbors.
- Naive Bayes (GaussianNB): Probabilistic model assuming feature independence.
- Decision Tree: Simple interpretable tree-based classifier.

Each model is trained on the training data, then evaluated on the test set.

## Model Evaluation

For each model, the following evaluation metrics are computed and compared:

- Accuracy: Overall correctness of classification.
- Precision: Proportion of positive identifications that were actually correct.
- Recall: Proportion of actual positives that were identified correctly.
- F1 Score: Harmonic mean of precision and recall; balances the two.

Confusion matrices and detailed classification reports are generated for each model, helping to understand strengths and weaknesses by class.

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





