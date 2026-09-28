# 🏭 Predictive Maintenance & Equipment Failure Prediction System
### *An End-to-End Industrial Machine Learning & Decision Support System*

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App%20Live-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML%20Pipelines-F7931E.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EB5424.svg)](https://xgboost.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Project Overview

Industrial manufacturing systems and CNC milling machines are subject to heavy continuous mechanical and thermal stress. Unexpected equipment failures cause expensive production downtime, safety hazards, and reactive repair costs.

The **Predictive Maintenance & Equipment Failure Prediction System** is a complete, production-ready Machine Learning system that:
- Ingests telemetry from machine sensors (temperature, rotational speed, torque, tool wear).
- Computes physics-derived operational variables (mechanical power, thermal dissipation, mechanical overstrain).
- Predicts breakdown probabilities with calibrated ML classifiers.
- Classifies machine health into **Low Risk (Normal)**, **Medium Risk (Watch)**, and **High Risk (Critical)**.
- Explains contributing operational factors using transparent Explainable AI principles.
- Provides actionable, advisory maintenance recommendations through an industrial Streamlit dashboard.

---

## 🎯 2. Problem Statement & Objectives

### Problem Statement
In traditional industrial setups, equipment maintenance follows either **reactive maintenance** (run-to-failure, resulting in catastrophic downtime) or **calendar-based maintenance** (costly over-servicing). Predictive maintenance leverages sensor telemetry to forecast impending failure states before mechanical breakdown occurs.

### Objectives
1. **Sensor Ingestion & Telemetry Monitoring:** Ingest multi-sensor streams (thermal, rotational, torque, and tooling wear).
2. **Physics-Informed Feature Engineering:** Formulate domain-specific features reflecting known failure physics (Heat Dissipation, Power Overload, Tool Overstrain).
3. **Imbalance-Aware Classification:** Handle significant class imbalance (~3.4% failure rate) using cost-sensitive learning (`scale_pos_weight`, balanced class weights) and threshold tuning.
4. **Transparent Evaluation:** Benchmark Logistic Regression, Random Forest, and XGBoost using Precision, Recall, F1-Score, and ROC-AUC.
5. **Real-Time Interactive Dashboard:** Provide an industrial-grade Streamlit web interface for fleet health monitoring, interactive simulation, and decision support.

---

## 📊 3. Dataset Description

The system uses the benchmark **AI4I 2020 Predictive Maintenance Dataset** (hosted on the UC Irvine Machine Learning Repository):
- **Total Records:** 10,000 synthetic instances reflecting real milling machine operational records.
- **Features:** 14 columns (sensor measurements, identifiers, failure targets, and failure modes).
- **Target Variable:** `Machine failure` (Binary: 0 = Normal, 1 = Failure; 339 failure cases = 3.39%).

### Sensor Variables
| Variable | Unit | Description |
| :--- | :--- | :--- |
| `Type` | Categorical | Quality variant: **L** (Low, 50%), **M** (Medium, 30%), **H** (High, 20%) |
| `Air temperature [K]` | Kelvin | Ambient factory temperature (~300 K) |
| `Process temperature [K]` | Kelvin | Machine process temperature (~310 K) |
| `Rotational speed [rpm]` | RPM | Spindle speed (1168 to 2886 rpm) |
| `Torque [Nm]` | Newton-meters | Spindle resistance torque (3.8 to 76.6 Nm) |
| `Tool wear [min]` | Minutes | Cumulative tool usage duration (0 to 253 min) |

### Failure Modes Recorded
- **TWF (Tool Wear Failure):** Occurs when tool wear reaches critical threshold (200–240 min).
- **HDF (Heat Dissipation Failure):** Occurs when $\Delta T < 8.6\text{ K}$ and speed $< 1380\text{ rpm}$.
- **PWF (Power Failure):** Occurs when mechanical power $P < 3.5\text{ kW}$ or $P > 9.0\text{ kW}$.
- **OSF (Overstrain Failure):** Occurs when product of Tool Wear $\times$ Torque exceeds structural limits.
- **RNF (Random Failure):** Uncorrelated random failure mode (~0.1% chance).

---

## ⚙️ 4. Feature Engineering & Preprocessing Pipeline

To empower the ML models with physical domain knowledge, four domain variables are engineered:

$$1.\quad \text{Temp Difference } [K] = \text{Process Temperature} - \text{Air Temperature}$$
$$2.\quad \text{Mechanical Power } [kW] = \text{Rotational Speed } (rpm) \times \text{Torque } (Nm) \times \left(\frac{2\pi}{60}\right) \times 10^{-3}$$
$$3.\quad \text{Overstrain Factor } [min \cdot Nm] = \text{Tool Wear } (min) \times \text{Torque } (Nm)$$
$$4.\quad \text{Thermal Ratio} = \frac{\text{Process Temperature}}{\text{Air Temperature}}$$

### Data Preprocessing Architecture
- **Categorical Handling:** `OneHotEncoder(drop='first', handle_unknown='ignore')` applied to `Type`.
- **Numerical Scaling:** `StandardScaler()` applied to sensor and engineered numerical features.
- **Pipeline Bundling:** Encapsulated in a `ColumnTransformer` inside an end-to-end `sklearn.pipeline.Pipeline` to guarantee zero data leakage between training and testing folds.

---

## 🧠 5. Machine Learning Models & Evaluation Benchmark

All models are trained with **stratified 80/20 train-test splits** and evaluated on unseen test data (2,000 samples).

### Model Comparison Table
| Metric | Baseline: Logistic Regression | Random Forest Classifier | Production Model: XGBoost |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 85.85% | 98.30% | **98.75%** |
| **Precision (Failure Class)** | 17.91% | 71.25% | **82.09%** |
| **Recall (Failure Class)** | **88.24%** | 83.82% | **80.88%** |
| **F1-Score (Failure Class)** | 0.2978 | 0.7703 | **0.8148** |
| **ROC-AUC** | 0.9385 | 0.9813 | **0.9826** |
| **Training Time** | ~0.08s | ~0.55s | ~0.14s |

> **Selection Rationale:** In predictive maintenance, high precision and recall on the minority failure class are paramount. While Logistic Regression yields high recall, its false positive rate is high (low precision). **XGBoost** provides the highest balance with an **F1-Score of 0.8148** and **ROC-AUC of 0.9826**.

---

## 🖥️ 6. Streamlit Dashboard Features

The application is structured into 5 interactive modules:

1. **📊 1. System Overview:**
   - Real-time fleet KPI metrics (Total Equipment, Historical Failures, Fleet Failure Rate, High-Risk Fleet count).
   - Donut chart of fleet risk distribution and breakdown of failure types.
   - Raw dataset telemetry explorer.

2. **🔍 2. Equipment Monitoring:**
   - Searchable and filterable fleet table (Filter by Variant L/M/H, Risk Category, Temperature, Torque, and Tool Wear ranges).
   - Real-time failure probability indicators and risk badges.
   - One-click CSV export of filtered equipment data.

3. **📉 3. Failure Analytics:**
   - Multi-tab visual analytics powered by interactive Plotly charts.
   - Torque vs. Rotational Speed scatter plot classified by failure modes.
   - Thermal gradient distributions (Process Temp vs. Air Temp).
   - Sensor histograms and correlation heatmaps.

4. **⚙️ 4. Predict Machine Failure:**
   - Single-unit real-time risk prediction interface with scenario quick-fill presets.
   - Animated failure probability gauge (0–100%).
   - Derived physics parameters breakdown.
   - **Explainable AI (XAI)** contributing factors list with severity tags.
   - **Actionable Maintenance Recommendations** prioritized by urgency.

5. **📈 5. Model Performance:**
   - Comprehensive model benchmark table with non-fabricated metrics.
   - Interactive Confusion Matrix visualizer and ROC Curves for all candidate models.
   - Global Feature Importance bar chart.
   - Technical methodology and imbalance mitigation details.

---

## 📁 7. Project Structure

```text
customer-predictive-maintenance/
│
├── app.py                            # Streamlit multi-page dashboard
├── requirements.txt                  # Python dependencies
├── README.md                         # Comprehensive documentation
│
├── data/
│   └── predictive_maintenance.csv    # AI4I 2020 dataset (10,000 records)
│
├── models/
│   ├── predictive_maintenance_model.pkl # Serialized best model & pipeline bundle
│   └── model_metrics.json            # Model evaluation summary JSON
│
├── src/
│   ├── __init__.py                   # Package initialization
│   ├── preprocessing.py              # Data loading, cleaning & feature engineering
│   ├── train_model.py                # Multi-model training and evaluation script
│   ├── prediction.py                 # Inference, risk scoring, XAI & recommendations
│   └── evaluation.py                 # Metrics calculation & ROC/PR curve generators
│
├── notebooks/
│   └── EDA_and_Modeling.ipynb        # Step-by-step Jupyter notebook
│
└── assets/
    └── style.css                     # Custom industrial dashboard styling
```

---

## 🚀 8. Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12 installed on your system.

### Step 1: Clone or Navigate to Project Directory
```bash
cd "c:\Users\riya5\Downloads\predictive model"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train / Verify ML Models (Optional - Pre-trained model included)
```bash
python src/train_model.py
```

### Step 4: Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

The application will launch in your default web browser at `http://localhost:8501`.

---

## 🔮 9. Future Scope & Roadmap

- [ ] **Live IoT Streaming:** Ingest sensor MQTT/OPC-UA telemetry streams from factory PLCs.
- [ ] **Remaining Useful Life (RUL):** Implement regression/survival models (Weibull, Cox Proportional Hazards) for time-to-failure forecasting.
- [ ] **Digital Twin Integration:** 3D CAD visualization of milling spindles showing localized thermal stress.
- [ ] **Automated Work Orders:** Automated webhook alerts to Enterprise Asset Management (EAM/SAP PM) systems.
- [ ] **Continuous Learning:** Scheduled automated model retraining pipelines upon receiving new verified breakdown logs.

---

## 📜 10. License & Acknowledgements

- **Dataset Source:** AI4I 2020 Predictive Maintenance Dataset, UC Irvine Machine Learning Repository (CC BY 4.0).
- **Academic Context:** Developed as a 5th-Semester B.Tech Minor Project in Artificial Intelligence & Machine Learning.
