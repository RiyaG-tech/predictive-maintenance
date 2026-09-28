"""
Model training module for Predictive Maintenance System.
Trains Logistic Regression, Random Forest, and XGBoost classifiers,
evaluates them, selects the best model, and saves the trained pipeline.
"""

import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import xgboost as xgb

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    load_data,
    prepare_train_test_data,
    get_preprocessor,
)
from src.evaluation import (
    evaluate_model,
    get_feature_names_from_preprocessor,
    extract_feature_importance,
)


def train_and_evaluate_all(data_path: str, models_dir: str = "models"):
    """
    Train baseline and tree-based models, compare performance, and save the top model.
    """
    os.makedirs(models_dir, exist_ok=True)

    print("Loading data from:", data_path)
    df = load_data(data_path)

    X_train, X_test, y_train, y_test, raw_features = prepare_train_test_data(df)

    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / max(pos_count, 1)

    print(f"Training set size: {len(X_train)} (Failures: {pos_count}, Normal: {neg_count})")
    print(f"Testing set size: {len(X_test)}")

    preprocessor = get_preprocessor()

    # Define Candidate Models
    candidate_configs = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    trained_pipelines = {}

    for model_name, clf in candidate_configs.items():
        print(f"\n--- Training {model_name} ---")
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = round(time.time() - start_time, 3)

        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["training_time_sec"] = train_time

        # Feature importances
        feature_names = get_feature_names_from_preprocessor(pipeline.named_steps["preprocessor"])
        fi_df = extract_feature_importance(pipeline, feature_names)
        metrics["top_features"] = fi_df.to_dict(orient="records")

        results[model_name] = metrics
        trained_pipelines[model_name] = pipeline

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

    # Select best model primarily by F1-Score (given high class imbalance)
    best_model_name = max(results.keys(), key=lambda m: (results[m]["f1"], results[m]["roc_auc"]))
    best_pipeline = trained_pipelines[best_model_name]

    print(f"\n==========================================")
    print(f"Selected Best Model: {best_model_name}")
    print(f"F1-Score: {results[best_model_name]['f1']:.4f} | ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"==========================================")

    # Save artifacts
    model_save_path = os.path.join(models_dir, "predictive_maintenance_model.pkl")
    metrics_save_path = os.path.join(models_dir, "model_metrics.json")

    artifact_bundle = {
        "pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "all_metrics": results,
        "raw_features": raw_features,
        "feature_names": get_feature_names_from_preprocessor(best_pipeline.named_steps["preprocessor"]),
        "class_labels": ["No Failure (Normal)", "Machine Failure"],
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(artifact_bundle, model_save_path)
    print(f"Saved complete model bundle to {model_save_path}")

    # Save metrics JSON for quick inspection
    with open(metrics_save_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "trained_timestamp": artifact_bundle["trained_timestamp"],
            "models": results
        }, f, indent=2)
    print(f"Saved metrics summary to {metrics_save_path}")

    return artifact_bundle


if __name__ == "__main__":
    train_and_evaluate_all("data/predictive_maintenance.csv")
