"""Utility helpers for evaluation and display."""

from sklearn.metrics import classification_report, confusion_matrix


def print_classification_summary(y_true, y_pred, title="Model"):
    print(f"\n=== {title} ===")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
