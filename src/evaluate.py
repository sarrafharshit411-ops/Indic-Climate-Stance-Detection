"""Evaluate a CSV containing true label and prediction columns."""
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
LABELS = ["Against", "Favour", "None"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--label-column", default="label")
    p.add_argument("--prediction-column", default="prediction")
    p.add_argument("--output", type=Path, default=Path("results/evaluation.json"))
    a = p.parse_args()
    df = pd.read_csv(a.input)
    for c in (a.label_column, a.prediction_column):
        if c not in df: raise ValueError(f"Missing column: {c}")
    y, pred = df[a.label_column].astype(str), df[a.prediction_column].astype(str)
    invalid = (set(y) | set(pred)) - set(LABELS)
    if invalid: raise ValueError(f"Unexpected labels: {sorted(invalid)}")
    result = {"examples": len(df), "accuracy": float(accuracy_score(y, pred)),
              "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)),
              "classification_report": classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0),
              "confusion_matrix_label_order": LABELS,
              "confusion_matrix": confusion_matrix(y, pred, labels=LABELS).tolist()}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
