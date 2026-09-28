"""
Evaluation module for Predictive Maintenance System.
Calculates performance metrics, confusion matrices, ROC/PR curves, and feature importance.
"""

from typing import Any, Dict
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluate a fitted pipeline/model on test data and return comprehensive metrics.
    """
    y_pred = model.predict(X_test)

    # Probabilities for class 1 (Machine failure)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
    else:
        y_proba = y_pred.astype(float)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_test, y_proba))
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Calculate ROC curve coordinates
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)

    # Precision-Recall curve
    precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_test, y_proba)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm,
        "classification_report": report,
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        },
        "pr_curve": {
            "precision": precision_vals.tolist(),
            "recall": recall_vals.tolist(),
            "thresholds": pr_thresholds.tolist(),
        },
    }


def get_feature_names_from_preprocessor(preprocessor) -> list:
    """Extract transformed feature names from ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback manual reconstruction
        return [
            "num__Air temp",
            "num__Process temp",
            "num__Rot speed",
            "num__Torque",
            "num__Tool wear",
            "num__Temp diff",
            "num__Power",
            "num__Overstrain",
            "num__Temp ratio",
            "cat__Type_L",
            "cat__Type_M",
        ]


def extract_feature_importance(pipeline, feature_names: list) -> pd.DataFrame:
    """
    Extract model-specific feature importances or linear coefficients.
    """
    model = pipeline.named_steps["classifier"]
    clean_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.zeros(len(feature_names))

    df_imp = pd.DataFrame({
        "Feature": clean_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    return df_imp
