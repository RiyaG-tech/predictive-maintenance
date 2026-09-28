"""
Predictive Maintenance & Equipment Failure Prediction System
Streamlit Multi-Page Interactive Dashboard
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Set project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import load_data, engineer_features
from src.prediction import (
    load_model_bundle,
    predict_single_instance,
    predict_batch_dataset,
    classify_risk,
)
from src.train_model import train_and_evaluate_all

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_file = PROJECT_ROOT / "assets" / "style.css"
if css_file.exists():
    with open(css_file, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Helper Functions
@st.cache_data
def get_dataset(data_path="data/predictive_maintenance.csv"):
    if not os.path.exists(data_path):
        return None
    return load_data(data_path)


@st.cache_resource
def get_model(model_path="models/predictive_maintenance_model.pkl"):
    if not os.path.exists(model_path):
        return None
    return load_model_bundle(model_path)


# Sidebar Configuration & Navigation
st.sidebar.markdown("## 🏭 Industrial System Menu")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Go to Navigation:",
    [
        "📊 1. System Overview",
        "🔍 2. Equipment Monitoring",
        "📉 3. Failure Analytics",
        "⚙️ 4. Predict Machine Failure",
        "📈 5. Model Performance",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Risk Thresholds")
st.sidebar.caption("Configurable application-defined risk criteria:")

col_th1, col_th2 = st.sidebar.columns(2)
with col_th1:
    low_thresh = st.number_input("Low Max (%)", min_value=10.0, max_value=50.0, value=30.0, step=5.0) / 100.0
with col_th2:
    high_thresh = st.number_input("High Min (%)", min_value=50.0, max_value=95.0, value=70.0, step=5.0) / 100.0

st.sidebar.markdown("---")

# Quick dataset and model health check in sidebar
df_raw = get_dataset()
model_bundle = get_model()

if model_bundle is not None:
    st.sidebar.success(f"Active Model: **{model_bundle['best_model_name']}**")
else:
    st.sidebar.warning("Model not found on disk.")
    if st.sidebar.button("Train Model Now"):
        with st.spinner("Training models..."):
            train_and_evaluate_all("data/predictive_maintenance.csv")
            st.cache_resource.clear()
            st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("📌 **Project**: B.Tech Minor Project")
st.sidebar.caption("📚 **Dataset**: AI4I 2020 Predictive Maintenance")
st.sidebar.caption("🛡️ **Stack**: Python, Scikit-learn, XGBoost, Streamlit")


# Header Component
def render_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="main-header">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# PAGE 1: SYSTEM OVERVIEW
# ==============================================================================
if nav_choice == "📊 1. System Overview":
    render_header(
        "Predictive Maintenance & Equipment Failure System",
        "Real-time industrial sensor analytics, failure risk prediction, and proactive maintenance decision support.",
    )

    if df_raw is None:
        st.error("Dataset not found at `data/predictive_maintenance.csv`. Please verify the dataset location.")
        st.stop()

    # Calculate batch predictions for overview
    if model_bundle is not None:
        df_scored = predict_batch_dataset(model_bundle, df_raw, low_thresh, high_thresh)
    else:
        df_scored = df_raw.copy()
        df_scored["Risk_Level"] = "Unknown"
        df_scored["Failure_Probability_Pct"] = 0.0

    total_machines = len(df_raw)
    actual_failures = int(df_raw["Machine failure"].sum())
    overall_fail_rate = (actual_failures / total_machines) * 100.0
    high_risk_count = int((df_scored["Risk_Level"] == "High Risk").sum())
    med_risk_count = int((df_scored["Risk_Level"] == "Medium Risk").sum())
    low_risk_count = int((df_scored["Risk_Level"] == "Low Risk").sum())

    # KPI Summary Cards Row
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Total Equipment</div>
                <div class="kpi-value">{total_machines:,}</div>
                <div class="kpi-subtitle">Monitored units</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Historical Failures</div>
                <div class="kpi-value">{actual_failures:,}</div>
                <div class="kpi-subtitle">Recorded breakdowns</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Failure Rate</div>
                <div class="kpi-value">{overall_fail_rate:.2f}%</div>
                <div class="kpi-subtitle">Baseline imbalance</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">High Risk Fleet</div>
                <div class="kpi-value" style="color: #ef4444;">{high_risk_count:,}</div>
                <div class="kpi-subtitle">Prob &gt; {int(high_thresh*100)}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k5:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">Active AI Model</div>
                <div class="kpi-value" style="font-size: 1.25rem; color: #3b82f6;">{model_bundle['best_model_name'] if model_bundle else 'None'}</div>
                <div class="kpi-subtitle">Production Pipeline</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("🛡️ Fleet Risk Classification Breakdown")
        risk_counts = pd.DataFrame({
            "Risk Level": ["Low Risk (Normal)", "Medium Risk (Watch)", "High Risk (Critical)"],
            "Count": [low_risk_count, med_risk_count, high_risk_count],
            "Color": ["#22c55e", "#eab308", "#ef4444"]
        })
        fig_risk = px.pie(
            risk_counts,
            names="Risk Level",
            values="Count",
            hole=0.55,
            color="Risk Level",
            color_discrete_map={
                "Low Risk (Normal)": "#22c55e",
                "Medium Risk (Watch)": "#eab308",
                "High Risk (Critical)": "#ef4444",
            },
        )
        fig_risk.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_risk, use_container_width=True)

    with c2:
        st.subheader("⚠️ Failure Modes in Dataset")
        failure_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
        failure_labels = {
            "TWF": "Tool Wear Failure (TWF)",
            "HDF": "Heat Dissipation Failure (HDF)",
            "PWF": "Power Failure (PWF)",
            "OSF": "Overstrain Failure (OSF)",
            "RNF": "Random Failure (RNF)",
        }
        mode_counts = {failure_labels[col]: int(df_raw[col].sum()) for col in failure_cols}
        df_modes = pd.DataFrame(list(mode_counts.items()), columns=["Failure Mode", "Occurrences"]).sort_values(by="Occurrences", ascending=True)

        fig_modes = px.bar(
            df_modes,
            x="Occurrences",
            y="Failure Mode",
            orientation="h",
            color="Occurrences",
            color_continuous_scale="Blues",
            text="Occurrences",
        )
        fig_modes.update_layout(
            margin=dict(t=20, b=20, l=20, r=20),
            height=320,
            coloraxis_showscale=False,
            xaxis_title="Count of Machines",
            yaxis_title=""
        )
        fig_modes.update_traces(textposition="outside")
        st.plotly_chart(fig_modes, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Dataset Snapshot & Sensor Specifications")
    st.dataframe(
        df_raw.head(8),
        use_container_width=True,
    )


# ==============================================================================
# PAGE 2: EQUIPMENT MONITORING
# ==============================================================================
elif nav_choice == "🔍 2. Equipment Monitoring":
    render_header(
        "Equipment Health Monitoring & Fleet Explorer",
        "Filter, search, and monitor machine sensor readings, predicted failure probability, and assigned risk status.",
    )

    if df_raw is None or model_bundle is None:
        st.error("Dataset or trained model is unavailable.")
        st.stop()

    df_scored = predict_batch_dataset(model_bundle, df_raw, low_thresh, high_thresh)

    # Filter Controls Expandable
    with st.expander("🛠️ Interactive Fleet Filters & Range Selectors", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            type_filter = st.multiselect(
                "Machine Variant (Type):",
                options=["L", "M", "H"],
                default=["L", "M", "H"],
            )
            search_query = st.text_input("Search Product ID or UDI:", placeholder="e.g. M14860 or 105")

        with f2:
            risk_filter = st.multiselect(
                "Risk Classification:",
                options=["Low Risk", "Medium Risk", "High Risk"],
                default=["Low Risk", "Medium Risk", "High Risk"],
            )
            temp_range = st.slider(
                "Air Temperature Range [K]:",
                float(df_raw["Air temperature [K]"].min()),
                float(df_raw["Air temperature [K]"].max()),
                (float(df_raw["Air temperature [K]"].min()), float(df_raw["Air temperature [K]"].max())),
            )

        with f3:
            torque_range = st.slider(
                "Torque Range [Nm]:",
                float(df_raw["Torque [Nm]"].min()),
                float(df_raw["Torque [Nm]"].max()),
                (float(df_raw["Torque [Nm]"].min()), float(df_raw["Torque [Nm]"].max())),
            )
            wear_range = st.slider(
                "Tool Wear Range [min]:",
                float(df_raw["Tool wear [min]"].min()),
                float(df_raw["Tool wear [min]"].max()),
                (float(df_raw["Tool wear [min]"].min()), float(df_raw["Tool wear [min]"].max())),
            )

    # Apply Filters
    df_filtered = df_scored[
        (df_scored["Type"].isin(type_filter))
        & (df_scored["Risk_Level"].isin(risk_filter))
        & (df_scored["Air temperature [K]"] >= temp_range[0])
        & (df_scored["Air temperature [K]"] <= temp_range[1])
        & (df_scored["Torque [Nm]"] >= torque_range[0])
        & (df_scored["Torque [Nm]"] <= torque_range[1])
        & (df_scored["Tool wear [min]"] >= wear_range[0])
        & (df_scored["Tool wear [min]"] <= wear_range[1])
    ]

    if search_query.strip():
        q = search_query.strip().lower()
        df_filtered = df_filtered[
            df_filtered["Product ID"].str.lower().str.contains(q)
            | df_filtered["UDI"].astype(str).str.contains(q)
        ]

    # Metrics on filtered subset
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matching Equipment", f"{len(df_filtered):,}")
    m2.metric("High Risk in Selection", f"{(df_filtered['Risk_Level'] == 'High Risk').sum():,}")
    m3.metric("Medium Risk in Selection", f"{(df_filtered['Risk_Level'] == 'Medium Risk').sum():,}")
    m4.metric("Average Failure Prob", f"{df_filtered['Failure_Probability_Pct'].mean():.2f}%" if len(df_filtered) > 0 else "0%")

    st.markdown("---")

    # Display columns
    display_cols = [
        "UDI",
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Failure_Probability_Pct",
        "Risk_Level",
        "Predicted_Status",
        "Machine failure",
    ]

    st.dataframe(
        df_filtered[display_cols].rename(columns={
            "Failure_Probability_Pct": "Failure Prob (%)",
            "Machine failure": "Actual Failure",
            "Risk_Level": "Risk Category",
            "Predicted_Status": "AI Prediction"
        }),
        use_container_width=True,
        height=450,
    )

    # CSV Download
    csv_data = df_filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Filtered Fleet Data (CSV)",
        data=csv_data,
        file_name="monitored_equipment_export.csv",
        mime="text/csv",
    )


# ==============================================================================
# PAGE 3: FAILURE ANALYTICS
# ==============================================================================
elif nav_choice == "📉 3. Failure Analytics":
    render_header(
        "Exploratory Sensor & Failure Analytics",
        "Investigate relationships between industrial operating variables, sensor thresholds, and machine failure occurrences.",
    )

    if df_raw is None:
        st.error("Dataset is not loaded.")
        st.stop()

    df_eng = engineer_features(df_raw)

    tab1, tab2, tab3 = st.tabs(["🔥 Sensor Interactions", "📊 Distributions & Variants", "🔗 Feature Correlations"])

    with tab1:
        st.subheader("Torque vs. Rotational Speed by Failure Mode")
        st.caption("Inspect mechanical operating states leading to Power (PWF) and Overstrain (OSF) failures.")

        df_eng["Failure_Category"] = "Normal Operation"
        df_eng.loc[df_eng["TWF"] == 1, "Failure_Category"] = "Tool Wear (TWF)"
        df_eng.loc[df_eng["HDF"] == 1, "Failure_Category"] = "Heat Dissipation (HDF)"
        df_eng.loc[df_eng["PWF"] == 1, "Failure_Category"] = "Power Failure (PWF)"
        df_eng.loc[df_eng["OSF"] == 1, "Failure_Category"] = "Overstrain (OSF)"
        df_eng.loc[df_eng["RNF"] == 1, "Failure_Category"] = "Random Failure (RNF)"

        fig_scatter = px.scatter(
            df_eng,
            x="Rotational speed [rpm]",
            y="Torque [Nm]",
            color="Failure_Category",
            color_discrete_map={
                "Normal Operation": "#94a3b8",
                "Tool Wear (TWF)": "#eab308",
                "Heat Dissipation (HDF)": "#ef4444",
                "Power Failure (PWF)": "#8b5cf6",
                "Overstrain (OSF)": "#f97316",
                "Random Failure (RNF)": "#06b6d4",
            },
            opacity=0.7,
            hover_data=["UDI", "Product ID", "Tool wear [min]", "Power [kW]"],
        )
        fig_scatter.update_layout(height=450, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Thermal gradient scatter
        st.subheader("Process Temp vs. Air Temp by Failure State")
        fig_temp = px.scatter(
            df_eng,
            x="Air temperature [K]",
            y="Process temperature [K]",
            color=df_eng["Machine failure"].map({0: "Normal", 1: "Failure"}),
            color_discrete_map={"Normal": "#3b82f6", "Failure": "#ef4444"},
            opacity=0.65,
            marginal_x="box",
            marginal_y="box",
        )
        fig_temp.update_layout(height=450, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_temp, use_container_width=True)

    with tab2:
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.subheader("Failure Rate by Product Quality Variant")
            type_fail = df_raw.groupby("Type")["Machine failure"].agg(
                Total="count",
                Failures="sum",
                Failure_Rate=lambda x: (x.sum() / x.count()) * 100
            ).reset_index()

            fig_type = px.bar(
                type_fail,
                x="Type",
                y="Failure_Rate",
                text="Failure_Rate",
                color="Type",
                color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b"],
            )
            fig_type.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig_type.update_layout(
                yaxis_title="Failure Rate (%)",
                xaxis_title="Machine Type (L = Low, M = Medium, H = High Quality)",
                height=350,
            )
            st.plotly_chart(fig_type, use_container_width=True)

        with col_v2:
            st.subheader("Tool Wear Distribution (Normal vs. Failure)")
            fig_box_tw = px.box(
                df_raw,
                x="Machine failure",
                y="Tool wear [min]",
                color="Machine failure",
                color_discrete_map={0: "#10b981", 1: "#ef4444"},
            )
            fig_box_tw.update_layout(
                xaxis=dict(tickmode="array", tickvals=[0, 1], ticktext=["Normal (0)", "Failure (1)"]),
                height=350,
                showlegend=False,
            )
            st.plotly_chart(fig_box_tw, use_container_width=True)

        # Histograms of all sensor readings
        st.subheader("Sensor Distributions Across Fleet")
        sensor_to_plot = st.selectbox(
            "Select Sensor Variable to Inspect:",
            [
                "Torque [Nm]",
                "Tool wear [min]",
                "Rotational speed [rpm]",
                "Air temperature [K]",
                "Process temperature [K]",
                "Power [kW]",
                "Temp difference [K]",
                "Overstrain factor [min x Nm]",
            ]
        )
        fig_hist = px.histogram(
            df_eng,
            x=sensor_to_plot,
            color="Machine failure",
            color_discrete_map={0: "#3b82f6", 1: "#ef4444"},
            barmode="overlay",
            marginal="rug",
            opacity=0.6,
        )
        fig_hist.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Correlation Heatmap of Sensors & Engineered Features")
        numeric_cols = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
            "Temp difference [K]",
            "Power [kW]",
            "Overstrain factor [min x Nm]",
            "Machine failure",
        ]
        corr_matrix = df_eng[numeric_cols].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
        fig_corr.update_layout(height=520, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_corr, use_container_width=True)


# ==============================================================================
# PAGE 4: PREDICT MACHINE FAILURE
# ==============================================================================
elif nav_choice == "⚙️ 4. Predict Machine Failure":
    render_header(
        "Interactive Machine Failure Prediction & Risk Assessment",
        "Input real-time sensor parameters, compute failure probability, analyze key contributing factors, and receive maintenance recommendations.",
    )

    if model_bundle is None:
        st.error("Trained model is not available. Please train the model from the overview page.")
        st.stop()

    st.subheader("1. Select Operating Preset or Enter Sensor Readings")

    preset = st.selectbox(
        "Quick Scenario Presets (Pre-fill with realistic operational states):",
        [
            "Custom Manual Input",
            "🟢 Standard Normal Operation (Optimal conditions)",
            "🔴 High Tool Wear Failure Scenario (TWF)",
            "🔴 Heat Dissipation Breakdown Scenario (HDF)",
            "🔴 Power Overload Scenario (PWF)",
            "🔴 Severe Mechanical Overstrain Scenario (OSF)",
        ],
        index=0,
    )

    # Set default values based on preset
    if preset == "🟢 Standard Normal Operation (Optimal conditions)":
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "M", 298.2, 308.7, 1500, 40.0, 30
    elif preset == "🔴 High Tool Wear Failure Scenario (TWF)":
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "L", 300.5, 310.2, 1550, 42.0, 235
    elif preset == "🔴 Heat Dissipation Breakdown Scenario (HDF)":
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "L", 303.8, 311.2, 1340, 52.0, 110
    elif preset == "🔴 Power Overload Scenario (PWF)":
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "H", 301.0, 310.5, 2600, 70.0, 80
    elif preset == "🔴 Severe Mechanical Overstrain Scenario (OSF)":
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "L", 299.0, 309.0, 1380, 68.0, 215
    else:
        d_type, d_air, d_proc, d_speed, d_torque, d_wear = "M", 300.0, 310.0, 1538, 40.0, 108

    with st.form("prediction_form"):
        col_in1, col_in2, col_in3 = st.columns(3)

        with col_in1:
            in_type = st.selectbox(
                "Machine Variant / Type:",
                options=["L", "M", "H"],
                index=["L", "M", "H"].index(d_type),
                help="L = Low variant (50%), M = Medium variant (30%), H = High variant (20%)",
            )
            in_air_temp = st.number_input(
                "Air Temperature [K]:",
                min_value=290.0,
                max_value=315.0,
                value=float(d_air),
                step=0.1,
                help=f"Equivalent to {d_air - 273.15:.1f} °C",
            )

        with col_in2:
            in_proc_temp = st.number_input(
                "Process Temperature [K]:",
                min_value=300.0,
                max_value=320.0,
                value=float(d_proc),
                step=0.1,
                help=f"Equivalent to {d_proc - 273.15:.1f} °C",
            )
            in_rot_speed = st.number_input(
                "Rotational Speed [rpm]:",
                min_value=1000,
                max_value=3000,
                value=int(d_speed),
                step=10,
            )

        with col_in3:
            in_torque = st.number_input(
                "Torque [Nm]:",
                min_value=3.0,
                max_value=90.0,
                value=float(d_torque),
                step=0.5,
            )
            in_tool_wear = st.number_input(
                "Tool Wear [min]:",
                min_value=0,
                max_value=300,
                value=int(d_wear),
                step=1,
                help="Tool usage in minutes before replacement",
            )

        submit_btn = st.form_submit_button("⚡ Predict Failure Risk", use_container_width=True)

    if submit_btn:
        input_payload = {
            "Type": in_type,
            "Air temperature [K]": in_air_temp,
            "Process temperature [K]": in_proc_temp,
            "Rotational speed [rpm]": in_rot_speed,
            "Torque [Nm]": in_torque,
            "Tool wear [min]": in_tool_wear,
        }

        with st.spinner("Analyzing machine physics & evaluating ML pipeline..."):
            result = predict_single_instance(
                model_bundle,
                input_payload,
                low_thresh=low_thresh,
                high_thresh=high_thresh,
            )

        st.markdown("---")
        st.subheader("2. Prediction & Risk Assessment Outcome")

        # Top Result Summary Card
        res_c1, res_c2, res_c3 = st.columns([1.2, 1.5, 1.3])

        with res_c1:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left: 6px solid {result['risk_color']};">
                    <div class="kpi-title">Assigned Risk Category</div>
                    <div class="kpi-value" style="color: {result['risk_color']};">{result['risk_badge']}</div>
                    <div class="kpi-subtitle">Model: {model_bundle['best_model_name']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with res_c2:
            prob_val = result["failure_probability_pct"]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_val,
                title={"text": "Predicted Failure Probability (%)", "font": {"size": 14}},
                number={"suffix": "%", "font": {"size": 24, "color": result["risk_color"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": result["risk_color"]},
                    "steps": [
                        {"range": [0, low_thresh * 100], "color": "#f0fdf4"},
                        {"range": [low_thresh * 100, high_thresh * 100], "color": "#fefce8"},
                        {"range": [high_thresh * 100, 100], "color": "#fef2f2"},
                    ],
                    "threshold": {
                        "line": {"color": "#b91c1c", "width": 3},
                        "thickness": 0.75,
                        "value": high_thresh * 100,
                    },
                },
            ))
            fig_gauge.update_layout(height=180, margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

        with res_c3:
            eng_vals = result["engineered_values"]
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-title">Physics & Derived Variables</div>
                    <div style="font-size: 0.85rem; color: #334155; line-height: 1.6;">
                        • <b>Thermal Difference:</b> {eng_vals['Temp difference [K]']} K<br>
                        • <b>Mechanical Power:</b> {eng_vals['Power [kW]']} kW<br>
                        • <b>Overstrain Factor:</b> {eng_vals['Overstrain factor [min x Nm]']} min*Nm<br>
                        • <b>Thermal Ratio:</b> {eng_vals['Temp ratio']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Explainable AI & Maintenance Recommendations
        e_col1, e_col2 = st.columns(2)

        with e_col1:
            st.subheader("🔍 Explainable AI: Contributing Factors")
            st.caption("Key operational factors associated with the model's prediction:")

            for factor in result["contributing_factors"]:
                sev_color = "#ef4444" if factor["severity"] == "High" else ("#eab308" if factor["severity"] == "Medium" else "#22c55e")
                st.markdown(
                    f"""
                    <div class="action-card" style="border-left-color: {sev_color};">
                        <div class="action-title">{factor['factor']} — <span style="color: {sev_color};">{factor['value']}</span></div>
                        <div class="action-desc">{factor['detail']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.info("ℹ️ **Note**: These features contributed strongly to the model's prediction based on historical machine failure correlations.")

        with e_col2:
            st.subheader("🛠️ Suggested Maintenance Actions")
            st.caption("Advisory preventive maintenance protocols for shop-floor operators:")

            for rec in result["recommendations"]:
                st.markdown(
                    f"""
                    <div class="action-card">
                        <div class="action-title">{rec['icon']} {rec['system']} ({rec['urgency']})</div>
                        <div class="action-desc">{rec['action']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption("⚠️ Maintenance recommendations represent decision-support guidance, not automated mechanical diagnostics.")


# ==============================================================================
# PAGE 5: MODEL PERFORMANCE
# ==============================================================================
elif nav_choice == "📈 5. Model Performance":
    render_header(
        "Machine Learning Model Evaluation & Benchmark",
        "Transparent, non-fabricated metrics across Logistic Regression, Random Forest, and XGBoost classifiers.",
    )

    if model_bundle is None:
        st.error("No trained models found.")
        st.stop()

    all_metrics = model_bundle.get("all_metrics", {})
    best_name = model_bundle.get("best_model_name", "Unknown")

    st.success(f"🏆 **Selected Production Model: {best_name}** (Selected based on balanced F1-Score & ROC-AUC under class imbalance)")

    # Performance Comparison Table
    st.subheader("1. Multi-Model Benchmark Comparison")

    comparison_data = []
    for m_name, m_info in all_metrics.items():
        comparison_data.append({
            "Model": m_name,
            "Accuracy": f"{m_info['accuracy'] * 100:.2f}%",
            "Precision (Class 1)": f"{m_info['precision'] * 100:.2f}%",
            "Recall (Class 1)": f"{m_info['recall'] * 100:.2f}%",
            "F1-Score": f"{m_info['f1']:.4f}",
            "ROC-AUC": f"{m_info['roc_auc']:.4f}",
            "PR-AUC": f"{m_info['pr_auc']:.4f}",
            "Training Time": f"{m_info.get('training_time_sec', 0):.2f}s",
        })

    df_comp = pd.DataFrame(comparison_data)
    st.table(df_comp)

    st.markdown("---")

    # Confusion Matrices and ROC Curves
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("2. Confusion Matrix Comparison")
        selected_cm_model = st.selectbox("Select Model for Confusion Matrix:", list(all_metrics.keys()))
        cm_data = all_metrics[selected_cm_model]["confusion_matrix"]

        cm_df = pd.DataFrame(
            cm_data,
            index=["Actual: Normal (0)", "Actual: Failure (1)"],
            columns=["Pred: Normal (0)", "Pred: Failure (1)"],
        )

        fig_cm = px.imshow(
            cm_df,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Class", color="Count"),
        )
        fig_cm.update_layout(height=350, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report summary
        cr = all_metrics[selected_cm_model]["classification_report"]
        st.caption(f"**Detailed Metrics for {selected_cm_model}:**")
        st.write({
            "Normal (Class 0) Precision": round(cr["0"]["precision"], 4),
            "Normal (Class 0) Recall": round(cr["0"]["recall"], 4),
            "Failure (Class 1) Precision": round(cr["1"]["precision"], 4),
            "Failure (Class 1) Recall": round(cr["1"]["recall"], 4),
        })

    with c2:
        st.subheader("3. ROC Curves (Receiver Operating Characteristic)")

        fig_roc = go.Figure()
        colors = {"Logistic Regression": "#94a3b8", "Random Forest": "#10b981", "XGBoost": "#3b82f6"}

        for m_name, m_info in all_metrics.items():
            roc_dict = m_info["roc_curve"]
            fig_roc.add_trace(go.Scatter(
                x=roc_dict["fpr"],
                y=roc_dict["tpr"],
                mode="lines",
                name=f"{m_name} (AUC = {m_info['roc_auc']:.3f})",
                line=dict(color=colors.get(m_name, "#3b82f6"), width=2),
            ))

        # Baseline diagonal
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guess (AUC = 0.500)",
            line=dict(dash="dash", color="#cbd5e1"),
        ))

        fig_roc.update_layout(
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Sensitivity / Recall)",
            height=350,
            margin=dict(t=20, b=20, l=20, r=20),
            legend=dict(yanchor="bottom", y=0.02, xanchor="right", x=0.98),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    st.markdown("---")

    # Feature Importance Section
    st.subheader("4. Feature Importance Analysis")
    st.caption("Relative importance of original sensor readings and engineered physics variables in the top model:")

    top_features = all_metrics[best_name].get("top_features", [])
    if top_features:
        df_fi = pd.DataFrame(top_features).head(10).sort_values(by="Importance", ascending=True)

        fig_fi = px.bar(
            df_fi,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
        )
        fig_fi.update_layout(
            height=400,
            margin=dict(t=20, b=20, l=20, r=20),
            coloraxis_showscale=False,
            xaxis_title="Relative Feature Importance Score",
            yaxis_title="Engineered / Sensor Feature",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Technical Methodology Note
    st.markdown("---")
    st.subheader("5. Technical Methodology & Class Imbalance Handling")
    st.markdown(
        """
        - **Imbalance Ratio**: The AI4I 2020 dataset possesses a ~3.4% failure class prevalence (339 positive cases out of 10,000 samples).
        - **Handling Strategies**:
            - Evaluated models use **cost-sensitive learning** (`class_weight='balanced'` in Logistic Regression and Random Forest, and `scale_pos_weight` in XGBoost).
            - Train/test splitting utilizes **stratified partitioning** to preserve exact class frequencies in both subsets.
            - Model selection prioritizes **F1-Score and ROC-AUC** over raw Accuracy to prevent accuracy paradox on minority failure events.
        - **Target Limitation Note**: The target indicates occurrence of machine failure during operating runs. Future time-window intervals (e.g. 7-day windows) are not recorded in this sensor dataset and are therefore not fabricated.
        """
    )
