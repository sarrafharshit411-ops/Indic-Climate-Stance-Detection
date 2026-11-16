"""TF-IDF + Logistic Regression / Linear SVM baseline.
CSV columns: text,label,split. Labels: Against,Favour,None.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

LABELS = ["Against", "Favour", "None"]

def load_data(path: Path):
    df = pd.read_csv(path)
    required = {"text", "label", "split"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    df = df.dropna(subset=["text", "label", "split"]).copy()
    df["text"] = df["text"].astype(str)
    if not set(df.label.unique()).issubset(LABELS):
        raise ValueError(f"Labels must be {LABELS}")
    if not set(df.split.unique()).issubset({"train", "validation", "test"}):
        raise ValueError("split must be train, validation, or test")
    parts = {s: df[df.split == s] for s in ("train", "validation", "test")}
    if parts["train"].empty or parts["test"].empty:
        raise ValueError("Train and test splits must be non-empty")
    return parts

def metrics(model, frame):
    pred = model.predict(frame.text)
    return {"accuracy": float(accuracy_score(frame.label, pred)),
            "macro_f1": float(f1_score(frame.label, pred, labels=LABELS, average="macro", zero_division=0)),
            "report": classification_report(frame.label, pred, labels=LABELS, output_dict=True, zero_division=0)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("results/baseline"))
    p.add_argument("--model", choices=["logistic_regression", "linear_svm"], default="linear_svm")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    args = p.parse_args()
    parts = load_data(args.data)
    weight = None if args.class_weight == "none" else "balanced"
    clf = LogisticRegression(max_iter=2000, class_weight=weight) if args.model == "logistic_regression" else LinearSVC(class_weight=weight)
    model = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200000, sublinear_tf=True)), ("classifier", clf)])
    model.fit(parts["train"].text, parts["train"].label)
    args.output.mkdir(parents=True, exist_ok=True)
    result = {"model": args.model, "class_weight": args.class_weight,
              "train_examples": len(parts["train"]),
              "validation": metrics(model, parts["validation"]) if len(parts["validation"]) else None,
              "test": metrics(model, parts["test"]),
              "note": "Use validation for model selection; keep test metrics for final reporting."}
    dest = args.output / f"{args.model}_{args.class_weight}.json"
    dest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {dest}")

if __name__ == "__main__":
    main()
