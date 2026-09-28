"""
Predictive Maintenance & Equipment Failure Prediction System
Streamlit Multi-Page Interactive Dashboard

NEW: Users can upload their own CSV (Page 0), map their columns to the model's
expected sensor columns, and run the whole dashboard on their own data.
"""

import io
import os
import re
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


# ==============================================================================
# CONSTANTS FOR CUSTOM DATA UPLOAD
# ==============================================================================
REQUIRED_COLS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
NUMERIC_COLS = [c for c in REQUIRED_COLS if c != "Type"]
TEMP_COLS = ["Air temperature [K]", "Process temperature [K]"]
MODE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
OPTIONAL_COLS = ["UDI", "Product ID", "Machine failure"] + MODE_COLS

# Common alternative header names (compared after lowercasing & stripping symbols)
ALIASES = {
    "Type": ["type", "producttype", "variant", "quality", "qualityvariant", "machinetype"],
    "Air temperature [K]": ["airtemperaturek", "airtemperature", "airtemp", "airtempk", "ambienttemperature", "airtemperaturec", "airtempc"],
    "Process temperature [K]": ["processtemperaturek", "processtemperature", "processtemp", "proctemp", "processtempk", "processtemperaturec", "processtempc"],
    "Rotational speed [rpm]": ["rotationalspeedrpm", "rotationalspeed", "rpm", "speed", "rotspeed"],
    "Torque [Nm]": ["torquenm", "torque"],
    "Tool wear [min]": ["toolwearmin", "toolwear", "wear", "toolwearminutes"],
    "UDI": ["udi", "id", "index", "machineid", "rowid"],
    "Product ID": ["productid", "product", "machine", "equipmentid", "assetid", "serial"],
    "Machine failure": ["machinefailure", "failure", "target", "label", "fail", "breakdown"],
    "TWF": ["twf"],
    "HDF": ["hdf"],
    "PWF": ["pwf"],
    "OSF": ["osf"],
    "RNF": ["rnf"],
}

# Rough sanity ranges (wider than AI4I training data) to warn about unit mistakes
SANITY_RANGES = {
    "Air temperature [K]": (280.0, 330.0),
    "Process temperature [K]": (290.0, 340.0),
    "Rotational speed [rpm]": (500.0, 5000.0),
    "Torque [Nm]": (0.0, 150.0),
    "Tool wear [min]": (0.0, 500.0),
}

NONE_OPTION = "— not in my file —"


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


@st.cache_data(show_spinner=False)
def read_uploaded_csv(file_bytes: bytes) -> pd.DataFrame:
    """Read an uploaded CSV, auto-detecting delimiter and trying common encodings."""
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise ValueError(f"Could not read the file as CSV: {last_err}")


@st.cache_data(show_spinner=False)
def score_dataset(_bundle, df: pd.DataFrame, low: float, high: float, model_name: str):
    """Cached batch scoring (model_name is part of the cache key)."""
    return predict_batch_dataset(_bundle, df, low, high)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def auto_detect_mapping(columns):
    """Guess which uploaded column matches each canonical column."""
    normalized = {c: _norm(c) for c in columns}
    mapping = {}
    used = set()
    for canon in REQUIRED_COLS + OPTIONAL_COLS:
        mapping[canon] = None
        # exact canonical match first
        for col in columns:
            if col == canon and col not in used:
                mapping[canon] = col
                used.add(col)
                break
        if mapping[canon] is not None:
            continue
        for alias in ALIASES.get(canon, []):
            hit = next((c for c, n in normalized.items() if n == alias and c not in used), None)
            if hit is not None:
                mapping[canon] = hit
                used.add(hit)
                break
    return mapping


def build_dataset(raw: pd.DataFrame, mapping: dict, temp_unit: str, default_type: str):
    """
    Convert an arbitrary uploaded dataframe into the schema the ML pipeline expects.
    Returns (dataframe, has_labels, notes, errors).
    """
    notes, errors = [], []
    out = pd.DataFrame(index=raw.index)

    # ---- Required columns ----
    for canon in REQUIRED_COLS:
        src = mapping.get(canon)
        if src:
            out[canon] = raw[src]
        elif canon == "Type":
            out["Type"] = default_type
            notes.append(f"`Type` not provided — every machine assigned variant **{default_type}**.")
        else:
            errors.append(f"Required column **{canon}** is not mapped.")

    if errors:
        return None, False, notes, errors

    # ---- Numeric coercion ----
    for c in NUMERIC_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if temp_unit == "Celsius (°C)":
        for c in TEMP_COLS:
            out[c] = out[c] + 273.15
        notes.append("Temperature columns converted from °C to Kelvin (+273.15).")

    # ---- Type normalisation (Low/Medium/High -> L/M/H) ----
    out["Type"] = out["Type"].astype(str).str.strip().str.upper().str[:1]
    bad_types = sorted(set(out["Type"].unique()) - {"L", "M", "H"})
    if bad_types:
        errors.append(
            f"`Type` must be L / M / H (or Low / Medium / High). Unrecognised values: {bad_types[:8]}"
        )
        return None, False, notes, errors

    # ---- Drop incomplete rows ----
    before = len(out)
    valid_mask = out[NUMERIC_COLS].notna().all(axis=1)
    raw_valid = raw.loc[valid_mask]
    out = out.loc[valid_mask].copy()
    dropped = before - len(out)
    if dropped:
        notes.append(f"Dropped **{dropped:,}** row(s) with missing / non-numeric sensor values.")
    if out.empty:
        errors.append("No valid rows remain after cleaning. Check your column mapping.")
        return None, False, notes, errors

    out = out.reset_index(drop=True)
    raw_valid = raw_valid.reset_index(drop=True)

    # ---- Optional identifiers ----
    if mapping.get("UDI"):
        out.insert(0, "UDI", raw_valid[mapping["UDI"]].values)
    else:
        out.insert(0, "UDI", np.arange(1, len(out) + 1))
    if mapping.get("Product ID"):
        out.insert(1, "Product ID", raw_valid[mapping["Product ID"]].astype(str).values)
    else:
        out.insert(1, "Product ID", [f"{t}{i:05d}" for i, t in zip(out["UDI"], out["Type"])])

    # ---- Optional labels ----
    has_labels = False
    if mapping.get("Machine failure"):
        lab = pd.to_numeric(raw_valid[mapping["Machine failure"]], errors="coerce").fillna(0)
        out["Machine failure"] = (lab > 0).astype(int).values
        has_labels = True
        notes.append("Failure labels detected — actual-vs-predicted analytics are enabled.")
    else:
        out["Machine failure"] = 0
        notes.append("No failure-label column mapped — running in **prediction-only** mode.")

    for m in MODE_COLS:
        if mapping.get(m):
            out[m] = (pd.to_numeric(raw_valid[mapping[m]], errors="coerce").fillna(0) > 0).astype(int).values
        else:
            out[m] = 0

    return out, has_labels, notes, errors


def sanity_warnings(df: pd.DataFrame):
    warns = []
    for col, (lo, hi) in SANITY_RANGES.items():
        share = ((df[col] < lo) | (df[col] > hi)).mean() * 100
        if share > 5:
            warns.append(
                f"{share:.0f}% of `{col}` values fall outside the plausible range "
                f"[{lo:g}, {hi:g}]. Check units (e.g. °C vs K)."
            )
    return warns


def template_csv() -> bytes:
    tmpl = pd.DataFrame({
        "Product ID": ["M14860", "L47181", "L47182", "H29424", "M14863"],
        "Type": ["M", "L", "L", "H", "M"],
        "Air temperature [K]": [298.1, 298.2, 298.1, 298.4, 298.2],
        "Process temperature [K]": [308.6, 308.7, 308.5, 308.9, 308.7],
        "Rotational speed [rpm]": [1551, 1408, 1498, 1433, 1408],
        "Torque [Nm]": [42.8, 46.3, 49.4, 39.5, 40.0],
        "Tool wear [min]": [0, 3, 5, 7, 9],
        "Machine failure (optional, 0/1)": [0, 0, 0, 0, 0],
    })
    return tmpl.to_csv(index=False).encode("utf-8")


# ==============================================================================
# SESSION STATE FOR CUSTOM DATASET
# ==============================================================================
st.session_state.setdefault("custom_df", None)
st.session_state.setdefault("custom_name", None)
st.session_state.setdefault("custom_has_labels", False)


# Sidebar Configuration & Navigation
st.sidebar.markdown("## 🏭 Industrial System Menu")
st.sidebar.markdown("---")

nav_choice = st.sidebar.radio(
    "Go to Navigation:",
    [
        "📂 0. Upload Your Data",
        "📊 1. System Overview",
        "🔍 2. Equipment Monitoring",
        "📉 3. Failure Analytics",
        "⚙️ 4. Predict Machine Failure",
        "📈 5. Model Performance",
    ],
    index=1,
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

# ---- Active dataset selection ----
model_bundle = get_model()
default_df = get_dataset()

using_custom = st.session_state["custom_df"] is not None
if using_custom:
    df_raw = st.session_state["custom_df"]
    has_labels = st.session_state["custom_has_labels"]
    dataset_label = st.session_state["custom_name"]
else:
    df_raw = default_df
    has_labels = df_raw is not None
    dataset_label = "AI4I 2020 (built-in)"

st.sidebar.markdown("### 🗂️ Active Dataset")
if df_raw is not None:
    if using_custom:
        st.sidebar.success(f"**{dataset_label}**\n\n{len(df_raw):,} rows · custom upload")
        if st.sidebar.button("↩️ Switch back to built-in dataset"):
            st.session_state["custom_df"] = None
            st.session_state["custom_name"] = None
            st.session_state["custom_has_labels"] = False
            st.rerun()
    else:
        st.sidebar.info(f"**{dataset_label}**\n\n{len(df_raw):,} rows")
else:
    st.sidebar.warning("No dataset loaded. Upload a CSV on page 0.")

st.sidebar.markdown("---")

if model_bundle is not None:
    st.sidebar.success(f"Active Model: **{model_bundle['best_model_name']}**")
else:
    st.sidebar.warning("Model not found on disk.")
    if st.sidebar.button("Train Model Now"):
        with st.spinner("Training models..."):
            train_and_evaluate_all("data/predictive_maintenance.csv")
            st.cache_resource.clear()
            st.cache_data.clear()
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
# PAGE 0: UPLOAD YOUR DATA
# ==============================================================================
if nav_choice == "📂 0. Upload Your Data":
    render_header(
        "Bring Your Own Data",
        "Upload your machine sensor CSV, map its columns, and run predictive maintenance on your own equipment fleet.",
    )

    left, right = st.columns([2, 1])

    with right:
        st.subheader("📑 What your file needs")
        st.markdown(
            """
            **Required columns** (any header names — you can map them next):
            - Machine variant / type (`L`, `M`, `H`) — *optional, can use a default*
            - Air temperature
            - Process temperature
            - Rotational speed (rpm)
            - Torque (Nm)
            - Tool wear (min)

            **Optional columns**
            - Product ID / UDI (machine identifiers)
            - `Machine failure` (0/1) — unlocks actual-vs-predicted analytics
            - `TWF, HDF, PWF, OSF, RNF` failure-mode flags
            """
        )
        st.download_button(
            "📥 Download CSV Template",
            data=template_csv(),
            file_name="predictive_maintenance_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with left:
        st.subheader("1. Upload CSV")
        uploaded = st.file_uploader("Choose a CSV file", type=["csv", "txt"])

        if uploaded is None:
            st.info("⬆️ Upload a CSV to get started. Until then, the built-in AI4I 2020 dataset is used.")
        else:
            try:
                raw_df = read_uploaded_csv(uploaded.getvalue())
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                st.stop()

            file_key = f"{uploaded.name}_{uploaded.size}"
            st.success(f"Loaded **{uploaded.name}** — {len(raw_df):,} rows × {raw_df.shape[1]} columns")
            with st.expander("Preview uploaded data", expanded=False):
                st.dataframe(raw_df.head(10), use_container_width=True)

            st.subheader("2. Map Your Columns")
            st.caption("Columns were auto-matched by name. Adjust anything that looks wrong.")

            detected = auto_detect_mapping(list(raw_df.columns))
            options = [NONE_OPTION] + list(raw_df.columns)

            def _select(canon, label, required):
                default = detected.get(canon)
                idx = options.index(default) if default in options else 0
                choice = st.selectbox(
                    label + (" *" if required else ""),
                    options,
                    index=idx,
                    key=f"map_{canon}_{file_key}",
                )
                return None if choice == NONE_OPTION else choice

            mapping = {}
            m1, m2, m3 = st.columns(3)
            with m1:
                mapping["Type"] = _select("Type", "Machine Type (L/M/H)", False)
                mapping["Air temperature [K]"] = _select("Air temperature [K]", "Air Temperature", True)
            with m2:
                mapping["Process temperature [K]"] = _select("Process temperature [K]", "Process Temperature", True)
                mapping["Rotational speed [rpm]"] = _select("Rotational speed [rpm]", "Rotational Speed (rpm)", True)
            with m3:
                mapping["Torque [Nm]"] = _select("Torque [Nm]", "Torque (Nm)", True)
                mapping["Tool wear [min]"] = _select("Tool wear [min]", "Tool Wear (min)", True)

            with st.expander("Optional columns (IDs, failure labels, failure modes)", expanded=False):
                o1, o2 = st.columns(2)
                with o1:
                    mapping["UDI"] = _select("UDI", "Row / Unit ID (UDI)", False)
                    mapping["Product ID"] = _select("Product ID", "Product / Machine ID", False)
                    mapping["Machine failure"] = _select("Machine failure", "Failure Label (0/1)", False)
                with o2:
                    for m in MODE_COLS:
                        mapping[m] = _select(m, f"{m} flag", False)

            st.subheader("3. Units & Defaults")
            u1, u2 = st.columns(2)
            with u1:
                temp_unit = st.radio(
                    "Temperature unit in your file:",
                    ["Kelvin (K)", "Celsius (°C)"],
                    horizontal=True,
                )
            with u2:
                default_type = st.selectbox(
                    "Default machine variant (used only if Type is not mapped):",
                    ["L", "M", "H"],
                    index=1,
                )

            if st.button("✅ Validate & Use This Dataset", type="primary", use_container_width=True):
                new_df, labels_found, notes, errors = build_dataset(raw_df, mapping, temp_unit, default_type)

                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    for n in notes:
                        st.info(n)
                    for w in sanity_warnings(new_df):
                        st.warning(w)

                    st.session_state["custom_df"] = new_df
                    st.session_state["custom_name"] = uploaded.name
                    st.session_state["custom_has_labels"] = labels_found
                    st.success(
                        f"Dataset ready: **{len(new_df):,} machines**. "
                        "Open any other page from the sidebar to see results on your data."
                    )
                    st.rerun()

    # ---- Quick results on the active custom dataset ----
    if using_custom and model_bundle is not None:
        st.markdown("---")
        st.subheader("📊 Quick Batch Prediction Results (your data)")
        with st.spinner("Scoring your equipment..."):
            scored = score_dataset(model_bundle, df_raw, low_thresh, high_thresh, model_bundle["best_model_name"])

        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Machines Scored", f"{len(scored):,}")
        q2.metric("High Risk", f"{(scored['Risk_Level'] == 'High Risk').sum():,}")
        q3.metric("Medium Risk", f"{(scored['Risk_Level'] == 'Medium Risk').sum():,}")
        q4.metric("Avg Failure Prob", f"{scored['Failure_Probability_Pct'].mean():.2f}%")

        if has_labels:
            actual_pos = scored["Machine failure"] == 1
            if actual_pos.sum() > 0:
                caught = (scored.loc[actual_pos, "Risk_Level"] != "Low Risk").mean() * 100
                st.caption(
                    f"Of the **{int(actual_pos.sum())}** actual failures in your file, "
                    f"**{caught:.1f}%** were flagged Medium or High risk by the model."
                )

        top_risk = scored.sort_values("Failure_Probability_Pct", ascending=False).head(15)
        show_cols = [c for c in ["UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
                                 "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
                                 "Failure_Probability_Pct", "Risk_Level"] if c in top_risk.columns]
        st.markdown("**Top 15 highest-risk machines**")
        st.dataframe(top_risk[show_cols], use_container_width=True)

        st.download_button(
            "📥 Download Full Scored Dataset (CSV)",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="scored_equipment_predictions.csv",
            mime="text/csv",
        )
    elif using_custom and model_bundle is None:
        st.warning("Dataset loaded, but no trained model is available to score it. Train the model from the sidebar.")


# ==============================================================================
# PAGE 1: SYSTEM OVERVIEW
# ==============================================================================
elif nav_choice == "📊 1. System Overview":
    render_header(
        "Predictive Maintenance & Equipment Failure System",
        "Real-time industrial sensor analytics, failure risk prediction, and proactive maintenance decision support.",
    )

    if df_raw is None:
        st.error("Dataset not found at `data/predictive_maintenance.csv`. Upload your own CSV on page 0 or verify the dataset location.")
        st.stop()

    if using_custom:
        st.info(f"📂 Showing your uploaded dataset: **{dataset_label}**")

    # Calculate batch predictions for overview
    if model_bundle is not None:
        df_scored = score_dataset(model_bundle, df_raw, low_thresh, high_thresh, model_bundle["best_model_name"])
    else:
        df_scored = df_raw.copy()
        df_scored["Risk_Level"] = "Unknown"
        df_scored["Failure_Probability_Pct"] = 0.0

    total_machines = len(df_raw)
    high_risk_count = int((df_scored["Risk_Level"] == "High Risk").sum())
    med_risk_count = int((df_scored["Risk_Level"] == "Medium Risk").sum())
    low_risk_count = int((df_scored["Risk_Level"] == "Low Risk").sum())

    if has_labels:
        actual_failures = int(df_raw["Machine failure"].sum())
        overall_fail_rate = (actual_failures / total_machines) * 100.0
        fail_val, fail_sub = f"{actual_failures:,}", "Recorded breakdowns"
        rate_val, rate_sub = f"{overall_fail_rate:.2f}%", "Baseline imbalance"
    else:
        fail_val, fail_sub = "N/A", "No labels in uploaded data"
        rate_val = f"{(high_risk_count / total_machines) * 100:.2f}%"
        rate_sub = "Share flagged High Risk"

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
                <div class="kpi-value">{fail_val}</div>
                <div class="kpi-subtitle">{fail_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-title">{'Failure Rate' if has_labels else 'High-Risk Rate'}</div>
                <div class="kpi-value">{rate_val}</div>
                <div class="kpi-subtitle">{rate_sub}</div>
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
        failure_cols = MODE_COLS
        has_modes = has_labels and int(df_raw[failure_cols].sum().sum()) > 0

        if has_modes:
            st.subheader("⚠️ Failure Modes in Dataset")
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
        else:
            st.subheader("📈 Predicted Failure Probability Distribution")
            fig_pd = px.histogram(
                df_scored,
                x="Failure_Probability_Pct",
                nbins=40,
                color_discrete_sequence=["#3b82f6"],
            )
            fig_pd.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                height=320,
                xaxis_title="Predicted Failure Probability (%)",
                yaxis_title="Machines",
            )
            st.plotly_chart(fig_pd, use_container_width=True)

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

    df_scored = score_dataset(model_bundle, df_raw, low_thresh, high_thresh, model_bundle["best_model_name"])

    # Filter Controls Expandable
    with st.expander("🛠️ Interactive Fleet Filters & Range Selectors", expanded=True):
        f1, f2, f3 = st.columns(3)

        def _range(col):
            lo, hi = float(df_raw[col].min()), float(df_raw[col].max())
            if lo == hi:
                hi = lo + 1e-6
            return lo, hi

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
            a_lo, a_hi = _range("Air temperature [K]")
            temp_range = st.slider("Air Temperature Range [K]:", a_lo, a_hi, (a_lo, a_hi))

        with f3:
            t_lo, t_hi = _range("Torque [Nm]")
            torque_range = st.slider("Torque Range [Nm]:", t_lo, t_hi, (t_lo, t_hi))
            w_lo, w_hi = _range("Tool wear [min]")
            wear_range = st.slider("Tool Wear Range [min]:", w_lo, w_hi, (w_lo, w_hi))

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
            df_filtered["Product ID"].astype(str).str.lower().str.contains(q, regex=False)
            | df_filtered["UDI"].astype(str).str.contains(q, regex=False)
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
    ]
    if has_labels:
        display_cols.append("Machine failure")
    display_cols = [c for c in display_cols if c in df_filtered.columns]

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

    df_eng = engineer_features(df_raw).copy()

    STATE_COLORS = {
        "Normal": "#3b82f6",
        "Failure": "#ef4444",
        "Low Risk": "#22c55e",
        "Medium Risk": "#eab308",
        "High Risk": "#ef4444",
    }

    # Decide what the "state" of each machine is: real label or model-predicted risk
    if has_labels:
        df_eng["State"] = df_eng["Machine failure"].map({0: "Normal", 1: "Failure"})
        df_eng["Rate"] = df_eng["Machine failure"] * 100.0
        corr_target = "Machine failure"
        rate_title = "Failure Rate (%)"
        state_word = "Failure State"
    else:
        if model_bundle is None:
            st.error("Your file has no failure labels, and no trained model is available to estimate risk.")
            st.stop()
        st.info(
            "📂 Your dataset has no failure labels, so analytics use the **model-predicted risk level** "
            "instead of actual failures."
        )
        scored_tmp = score_dataset(model_bundle, df_raw, low_thresh, high_thresh, model_bundle["best_model_name"])
        df_eng["State"] = scored_tmp["Risk_Level"].values
        df_eng["Failure_Probability_Pct"] = scored_tmp["Failure_Probability_Pct"].values
        df_eng["Rate"] = (df_eng["State"] == "High Risk").astype(float) * 100.0
        corr_target = "Failure_Probability_Pct"
        rate_title = "High-Risk Share (%)"
        state_word = "Predicted Risk"

    has_modes = has_labels and int(df_raw[MODE_COLS].sum().sum()) > 0

    tab1, tab2, tab3 = st.tabs(["🔥 Sensor Interactions", "📊 Distributions & Variants", "🔗 Feature Correlations"])

    with tab1:
        hover_cols = [c for c in ["UDI", "Product ID", "Tool wear [min]", "Power [kW]"] if c in df_eng.columns]

        if has_modes:
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
                hover_data=hover_cols,
            )
        else:
            st.subheader(f"Torque vs. Rotational Speed by {state_word}")
            fig_scatter = px.scatter(
                df_eng,
                x="Rotational speed [rpm]",
                y="Torque [Nm]",
                color="State",
                color_discrete_map=STATE_COLORS,
                opacity=0.7,
                hover_data=hover_cols,
            )
        fig_scatter.update_layout(height=450, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Thermal gradient scatter
        st.subheader(f"Process Temp vs. Air Temp by {state_word}")
        fig_temp = px.scatter(
            df_eng,
            x="Air temperature [K]",
            y="Process temperature [K]",
            color="State",
            color_discrete_map=STATE_COLORS,
            opacity=0.65,
            marginal_x="box",
            marginal_y="box",
        )
        fig_temp.update_layout(height=450, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_temp, use_container_width=True)

    with tab2:
        col_v1, col_v2 = st.columns(2)

        with col_v1:
            st.subheader(f"{rate_title.replace(' (%)', '')} by Product Quality Variant")
            type_fail = df_eng.groupby("Type")["Rate"].mean().reset_index(name="Failure_Rate")

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
                yaxis_title=rate_title,
                xaxis_title="Machine Type (L = Low, M = Medium, H = High Quality)",
                height=350,
            )
            st.plotly_chart(fig_type, use_container_width=True)

        with col_v2:
            st.subheader(f"Tool Wear Distribution by {state_word}")
            fig_box_tw = px.box(
                df_eng,
                x="State",
                y="Tool wear [min]",
                color="State",
                color_discrete_map=STATE_COLORS,
            )
            fig_box_tw.update_layout(height=350, showlegend=False, xaxis_title="")
            st.plotly_chart(fig_box_tw, use_container_width=True)

        # Histograms of all sensor readings
        st.subheader("Sensor Distributions Across Fleet")
        sensor_options = [
            s for s in [
                "Torque [Nm]",
                "Tool wear [min]",
                "Rotational speed [rpm]",
                "Air temperature [K]",
                "Process temperature [K]",
                "Power [kW]",
                "Temp difference [K]",
                "Overstrain factor [min x Nm]",
            ] if s in df_eng.columns
        ]
        sensor_to_plot = st.selectbox("Select Sensor Variable to Inspect:", sensor_options)
        fig_hist = px.histogram(
            df_eng,
            x=sensor_to_plot,
            color="State",
            color_discrete_map=STATE_COLORS,
            barmode="overlay",
            marginal="rug",
            opacity=0.6,
        )
        fig_hist.update_layout(height=380, margin=dict(t=20, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab3:
        st.subheader("Correlation Heatmap of Sensors & Engineered Features")
        if not has_labels:
            st.caption("No labels available — correlations are shown against the model's predicted failure probability.")
        numeric_cols = [
            c for c in [
                "Air temperature [K]",
                "Process temperature [K]",
                "Rotational speed [rpm]",
                "Torque [Nm]",
                "Tool wear [min]",
                "Temp difference [K]",
                "Power [kW]",
                "Overstrain factor [min x Nm]",
            ] if c in df_eng.columns
        ] + [corr_target]
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

    st.info("ℹ️ These metrics come from the model's original held-out test split on the AI4I 2020 dataset. They do not change when you upload your own data.")
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
        - **Custom Data Note**: Uploaded datasets are scored by the model trained on AI4I 2020. Predictions are most reliable when your sensor ranges and machine types resemble that training data.
        """
    )