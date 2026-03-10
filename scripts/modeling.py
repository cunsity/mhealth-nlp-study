"""Modeling functions for regression and classification tasks."""

import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score


def run_linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "model": model,
        "mse": mean_squared_error(y_test, preds),
        "r2": r2_score(y_test, preds),
        "predictions": preds,
    }


def run_logistic_regression(X_train, X_test, y_train, y_test, random_state: int = 42):
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "model": model,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "predictions": preds,
    }


def run_random_forest(X_train, X_test, y_train, y_test, random_state: int = 42):
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "model": model,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "predictions": preds,
    }


def compare_classifiers(results: dict) -> pd.DataFrame:
    rows = []
    for name, metrics in results.items():
        rows.append({
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1"],
        })
    return pd.DataFrame(rows)
