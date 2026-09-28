"""
Prediction and Decision Support Module for Predictive Maintenance.
Handles real-time inference, risk categorization, feature contribution analysis,
and actionable maintenance recommendations.
"""

from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import engineer_features


# Default baseline reference statistics from dataset for explainability
REFERENCE_STATS = {
    "Air temperature [K]": {"mean": 300.0, "std": 2.0, "warn_high": 304.0},
    "Process temperature [K]": {"mean": 310.0, "std": 1.48, "warn_high": 313.0},
    "Rotational speed [rpm]": {"mean": 1538.0, "std": 179.0, "warn_low": 1380.0, "warn_high": 2200.0},
    "Torque [Nm]": {"mean": 40.0, "std": 10.0, "warn_high": 60.0},
    "Tool wear [min]": {"mean": 108.0, "std": 63.6, "warn_high": 200.0},
    "Temp difference [K]": {"mean": 10.0, "std": 1.0, "warn_low": 8.6},
    "Power [kW]": {"mean": 6.28, "std": 1.0, "warn_low": 3.5, "warn_high": 9.0},
    "Overstrain factor [min x Nm]": {"mean": 4300.0, "std": 2800.0, "warn_high": 11000.0},
}


def load_model_bundle(model_path: str = "models/predictive_maintenance_model.pkl") -> Dict[str, Any]:
    """Load trained pipeline and metadata bundle."""
    try:
        bundle = joblib.load(model_path)
        return bundle
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load trained model from {model_path}. Please train the model first. Error: {e}"
        )


def classify_risk(probability: float, low_threshold: float = 0.30, high_threshold: float = 0.70) -> Tuple[str, str, str]:
    """
    Classify failure probability into application-defined risk categories.
    Returns: (Risk Level, Color Code, Status Badge)
    """
    if probability < low_threshold:
        return "Low Risk", "#22c55e", "🟢 NORMAL / LOW RISK"
    elif probability <= high_threshold:
        return "Medium Risk", "#eab308", "🟡 MODERATE / MEDIUM RISK"
    else:
        return "High Risk", "#ef4444", "🔴 CRITICAL / HIGH RISK"


def analyze_contributing_factors(input_dict: Dict[str, Any], engineered_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Identify machine parameters and operating conditions that contributed strongly
    to the failure risk based on domain physics thresholds and deviation from normal baseline.
    """
    contributions = []

    # 1. Tool wear inspection
    tool_wear = float(input_dict.get("Tool wear [min]", 0))
    if tool_wear >= 200:
        contributions.append({
            "factor": "Critical Tool Wear",
            "feature": "Tool wear [min]",
            "value": f"{tool_wear:.1f} min",
            "severity": "High",
            "detail": f"Tool usage ({tool_wear:.0f} min) exceeds critical threshold (>200 min), leading to cutting edge degradation."
        })
    elif tool_wear >= 160:
        contributions.append({
            "factor": "Elevated Tool Wear",
            "feature": "Tool wear [min]",
            "value": f"{tool_wear:.1f} min",
            "severity": "Medium",
            "detail": f"Tool wear ({tool_wear:.0f} min) is in warning territory (160-200 min)."
        })

    # 2. Overstrain inspection (Torque * Tool wear)
    overstrain = float(engineered_dict.get("Overstrain factor [min x Nm]", 0))
    m_type = input_dict.get("Type", "L")
    os_thresh = 11000 if m_type == "L" else (12000 if m_type == "M" else 13000)
    if overstrain >= os_thresh:
        contributions.append({
            "factor": "Mechanical Overstrain",
            "feature": "Overstrain factor [min x Nm]",
            "value": f"{overstrain:.0f} min*Nm",
            "severity": "High",
            "detail": f"Product of torque and tool wear ({overstrain:.0f}) exceeds material strain limit ({os_thresh})."
        })

    # 3. Heat Dissipation & Thermal gradient
    temp_diff = float(engineered_dict.get("Temp difference [K]", 10.0))
    rot_speed = float(input_dict.get("Rotational speed [rpm]", 1500))
    if temp_diff < 8.6 and rot_speed < 1380:
        contributions.append({
            "factor": "Heat Dissipation Anomaly",
            "feature": "Temp difference [K]",
            "value": f"{temp_diff:.2f} K",
            "severity": "High",
            "detail": f"Low thermal gradient ({temp_diff:.2f} K < 8.6 K) with sub-nominal speed ({rot_speed:.0f} rpm) indicates cooling breakdown."
        })
    elif temp_diff < 8.6:
        contributions.append({
            "factor": "Reduced Thermal Gradient",
            "feature": "Temp difference [K]",
            "value": f"{temp_diff:.2f} K",
            "severity": "Medium",
            "detail": f"Temperature differential ({temp_diff:.2f} K) is below standard operating threshold (8.6 K)."
        })

    # 4. Mechanical Power Dissipation
    power_kw = float(engineered_dict.get("Power [kW]", 6.0))
    if power_kw < 3.5 or power_kw > 9.0:
        contributions.append({
            "factor": "Power Envelope Deviation",
            "feature": "Power [kW]",
            "value": f"{power_kw:.2f} kW",
            "severity": "High",
            "detail": f"Calculated mechanical power ({power_kw:.2f} kW) is outside safe operational boundaries (3.5 - 9.0 kW)."
        })

    # 5. Torque Stress
    torque = float(input_dict.get("Torque [Nm]", 40.0))
    if torque > 60.0:
        contributions.append({
            "factor": "High Torque Stress",
            "feature": "Torque [Nm]",
            "value": f"{torque:.1f} Nm",
            "severity": "High",
            "detail": f"Operating torque ({torque:.1f} Nm) is significantly higher than standard rating (40 Nm)."
        })
    elif torque < 15.0:
        contributions.append({
            "factor": "Abnormally Low Torque",
            "feature": "Torque [Nm]",
            "value": f"{torque:.1f} Nm",
            "severity": "Medium",
            "detail": f"Operating torque ({torque:.1f} Nm) indicates potential slip or disengaged load."
        })

    # 6. Rotational Speed
    if rot_speed > 2500:
        contributions.append({
            "factor": "Overspeed Operating Condition",
            "feature": "Rotational speed [rpm]",
            "value": f"{rot_speed:.0f} rpm",
            "severity": "Medium",
            "detail": f"Rotational speed ({rot_speed:.0f} rpm) exceeds typical operational envelope."
        })
    elif rot_speed < 1200:
        contributions.append({
            "factor": "Underspeed / Stalling Risk",
            "feature": "Rotational speed [rpm]",
            "value": f"{rot_speed:.0f} rpm",
            "severity": "Medium",
            "detail": f"Rotational speed ({rot_speed:.0f} rpm) indicates heavy resistance or motor drag."
        })

    if not contributions:
        contributions.append({
            "factor": "Nominal Operating Parameters",
            "feature": "All Sensors",
            "value": "Within Safe Tolerances",
            "severity": "Low",
            "detail": "All monitored sensor variables are within acceptable standard operating ranges."
        })

    return contributions


def generate_maintenance_recommendations(risk_level: str, contributing_factors: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Generate actionable, domain-specific possible maintenance actions.
    Note: These are advisory maintenance guidelines to support decision-making.
    """
    recommendations = []

    factor_names = [f["factor"] for f in contributing_factors]

    # Tool wear advice
    if any("Tool Wear" in f for f in factor_names):
        recommendations.append({
            "system": "Tooling & Spindle Assembly",
            "action": "Inspect cutting insert wear pattern and replace tool before next production batch.",
            "urgency": "Immediate" if risk_level == "High Risk" else "Scheduled",
            "icon": "🔧"
        })

    # Overstrain advice
    if any("Overstrain" in f for f in factor_names):
        recommendations.append({
            "system": "Drive Train & Mechanical Load",
            "action": "Reduce feed rate or workpiece load to relieve torque overstrain; check tool sharpness.",
            "urgency": "Immediate",
            "icon": "⚙️"
        })

    # Heat Dissipation advice
    if any("Heat" in f or "Thermal" in f for f in factor_names):
        recommendations.append({
            "system": "Thermal & Cooling Subsystem",
            "action": "Check coolant fluid levels, pump pressure, and radiator airflow for heat dissipation bottlenecks.",
            "urgency": "High",
            "icon": "❄️"
        })

    # Power failure advice
    if any("Power" in f for f in factor_names):
        recommendations.append({
            "system": "Electrical & Inverter Drive",
            "action": "Inspect motor drive electrical terminals, check current draw stability and inverter diagnostics.",
            "urgency": "High",
            "icon": "⚡"
        })

    # Torque advice
    if any("Torque" in f for f in factor_names):
        recommendations.append({
            "system": "Mechanical Transmission",
            "action": "Check spindle bearing lubrication and gearbox alignment for friction buildup.",
            "urgency": "Scheduled",
            "icon": "🛢️"
        })

    # General High Risk baseline
    if risk_level == "High Risk" and not recommendations:
        recommendations.append({
            "system": "General Equipment Inspection",
            "action": "Schedule comprehensive preventive diagnostic check before continuing heavy duty operations.",
            "urgency": "Immediate",
            "icon": "🚨"
        })

    # Normal / Low risk baseline
    if risk_level == "Low Risk" and not recommendations:
        recommendations.append({
            "system": "Routine Maintenance",
            "action": "Continue normal operation under standard sensor monitoring schedule. Log routine inspection at next shift.",
            "urgency": "Routine",
            "icon": "✅"
        })

    return recommendations


def predict_single_instance(
    model_bundle: Dict[str, Any],
    input_data: Dict[str, Any],
    low_thresh: float = 0.30,
    high_thresh: float = 0.70,
) -> Dict[str, Any]:
    """
    Perform single machine prediction with full explainability and recommendations.
    """
    pipeline = model_bundle["pipeline"]

    # Build single-row DataFrame
    df_single = pd.DataFrame([input_data])

    # Engineer features
    df_feat = engineer_features(df_single)

    # Predict probability
    prob_failure = float(pipeline.predict_proba(df_feat)[0, 1])
    pred_class = int(prob_failure >= 0.5)

    risk_level, color, badge = classify_risk(prob_failure, low_thresh, high_thresh)

    # Extract engineered feature values for analysis
    engineered_dict = df_feat.iloc[0].to_dict()

    contributing_factors = analyze_contributing_factors(input_data, engineered_dict)
    recommendations = generate_maintenance_recommendations(risk_level, contributing_factors)

    return {
        "prediction_class": pred_class,
        "prediction_label": "Failure Likely" if pred_class == 1 else "Normal Operation",
        "failure_probability": prob_failure,
        "failure_probability_pct": round(prob_failure * 100, 2),
        "risk_level": risk_level,
        "risk_color": color,
        "risk_badge": badge,
        "contributing_factors": contributing_factors,
        "recommendations": recommendations,
        "engineered_values": {
            "Temp difference [K]": round(float(df_feat["Temp difference [K]"].iloc[0]), 2),
            "Power [kW]": round(float(df_feat["Power [kW]"].iloc[0]), 2),
            "Overstrain factor [min x Nm]": round(float(df_feat["Overstrain factor [min x Nm]"].iloc[0]), 1),
            "Temp ratio": round(float(df_feat["Temp ratio"].iloc[0]), 4),
        }
    }


def predict_batch_dataset(
    model_bundle: Dict[str, Any],
    df: pd.DataFrame,
    low_thresh: float = 0.30,
    high_thresh: float = 0.70,
) -> pd.DataFrame:
    """
    Run batch predictions across an entire fleet DataFrame.
    """
    pipeline = model_bundle["pipeline"]
    df_feat = engineer_features(df)

    probs = pipeline.predict_proba(df_feat)[:, 1]

    df_result = df.copy()
    df_result["Failure_Probability"] = np.round(probs, 4)
    df_result["Failure_Probability_Pct"] = np.round(probs * 100, 2)
    df_result["Predicted_Status"] = np.where(probs >= 0.5, "Failure Risk", "Normal")

    def get_risk(p):
        if p < low_thresh:
            return "Low Risk"
        elif p <= high_thresh:
            return "Medium Risk"
        else:
            return "High Risk"

    df_result["Risk_Level"] = [get_risk(p) for p in probs]

    return df_result
