# 🏭 Predictive Maintenance & Equipment Failure Prediction

A machine learning-based predictive maintenance system designed to identify equipment that may be at risk of failure using historical machine and sensor data.

The project uses machine learning classification models to analyze equipment characteristics and predict whether a machine is likely to experience failure. A Streamlit dashboard provides an interactive interface for exploring the data and generating predictions.

---

## 📌 Project Overview

Unexpected equipment failures can cause:

- Production downtime
- Maintenance costs
- Equipment damage
- Reduced productivity
- Operational delays

Predictive maintenance uses historical equipment and sensor data to identify patterns associated with machine failures.

This project applies machine learning to predict equipment failure and support proactive maintenance decisions.

---

## 🎯 Objectives

The main objectives of this project are:

1. Analyze industrial equipment and sensor data.
2. Perform data preprocessing and cleaning.
3. Identify patterns related to equipment failure.
4. Train machine learning classification models.
5. Compare model performance using evaluation metrics.
6. Predict whether equipment is at risk of failure.
7. Provide an interactive Streamlit dashboard.
8. Support proactive maintenance planning.

---

## ✨ Key Features

### 🔹 Data Preprocessing

- Missing-value handling
- Numerical feature processing
- Categorical feature encoding
- Feature preparation
- Train-test splitting

### 🔹 Exploratory Data Analysis

The project analyzes equipment-related characteristics such as:

- Temperature
- Rotational speed
- Torque
- Tool wear
- Machine type
- Other available equipment parameters

### 🔹 Machine Learning

The project can use classification models such as:

- Logistic Regression
- Random Forest
- XGBoost

Models are evaluated and compared using standard classification metrics.

### 🔹 Failure Prediction

The system predicts whether an equipment instance belongs to the failure or non-failure class based on the available input features.

### 🔹 Interactive Dashboard

The Streamlit application provides:

- Dataset overview
- Equipment analysis
- Visualizations
- Model information
- Failure prediction
- Prediction results

---

## 📊 Dataset

This project is designed for industrial equipment failure prediction using machine and sensor data.

The dataset contains equipment-related features that can include:

| Feature | Description |
|---|---|
| Type | Type/category of equipment |
| Air Temperature | Ambient air temperature |
| Process Temperature | Equipment process temperature |
| Rotational Speed | Machine rotational speed |
| Torque | Machine torque |
| Tool Wear | Tool usage/wear information |
| Failure | Equipment failure target |

> **Note:** The exact features and target values depend on the dataset used in the project.

---

## 🔄 Machine Learning Workflow

```text
Industrial Equipment Dataset
            ↓
       Data Cleaning
            ↓
     Data Preprocessing
            ↓
 Exploratory Data Analysis
            ↓
     Feature Preparation
            ↓
       Train/Test Split
            ↓
      Model Training
            ↓
    Model Evaluation
            ↓
     Model Selection
            ↓
   Equipment Prediction
            ↓
   Streamlit Dashboard