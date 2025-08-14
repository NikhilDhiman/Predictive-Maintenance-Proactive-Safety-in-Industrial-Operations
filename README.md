# Predictive Maintenance: Proactive Safety in Industrial Operations

This project implements a **machine learning pipeline** for **predictive maintenance** in industrial settings.  
It focuses on **proactive safety** by predicting machine failures before they occur, reducing downtime, preventing serious incidents, and protecting the environment.

---

## 📌 Why It Matters
- **SIF Prevention**: Detect hazards early to avoid serious incidents.
- **Environmental Protection**: Prevent leaks, emissions, and hazardous releases.
- **Cost Reduction**: Minimize unplanned maintenance and operational downtime.

---

## 📊 Dataset
- **Source**: [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv)
- **Description**: Synthetic dataset simulating sensor readings and failure records for industrial equipment.

---

## ⚙️ Features
- **Exploratory Data Analysis (EDA)**:
  - Data overview, distribution plots, and correlation analysis.
  - Feature engineering: temperature difference, categorical encoding.
- **Modeling**:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost (baseline and tuned with RandomizedSearchCV)
- **Class Imbalance Handling**:
  - `class_weight='balanced'` for tree and logistic models.
  - `scale_pos_weight` for XGBoost.
- **Evaluation Metrics**:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC
  - PR-AUC
- **Visualization**:
  - Metric comparison bar charts.
  - Pairplots and heatmaps for relationships.

---

## 📦 Installation

```bash
pip install pandas matplotlib seaborn scikit-learn shap lime imbalanced-learn xgboost
