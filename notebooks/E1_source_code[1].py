# E1 — TF-IDF + Logistic Regression
# Generated from the Colab notebook. Run in an environment with the listed dependencies.

import os
import io
import json
import zipfile
import requests
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# Existing public project repository
GITHUB_OWNER = "sarrafharshit411-ops"
GITHUB_REPO = "Indic-Climate-Stance-Detection"
GITHUB_BRANCH = "main"

REPO_API = (
    f"https://api.github.com/repos/{GITHUB_OWNER}/"
    f"{GITHUB_REPO}/git/trees/{GITHUB_BRANCH}?recursive=1"
)
RAW_BASE = (
    f"https://raw.githubusercontent.com/{GITHUB_OWNER}/"
    f"{GITHUB_REPO}/{GITHUB_BRANCH}/"
)

WORK_DIR = Path("/content/E1_TFIDF_Logistic_Regression")
DATA_DIR = WORK_DIR / "data"
OUTPUT_DIR = WORK_DIR / "outputs"
MODEL_DIR = WORK_DIR / "models"

for folder in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = {
    0: "Against",
    1: "Favour",
    2: "None"
}

LANGUAGES = ["hindi", "bengali"]
SPLITS = ["train", "val", "test"]

print("Project:", f"{GITHUB_OWNER}/{GITHUB_REPO}")
print("Fetching dataset file list from GitHub...")


# Locate the six expected CSV files in the repository.
response = requests.get(REPO_API, timeout=60)
response.raise_for_status()
tree_data = response.json()

if tree_data.get("truncated"):
    raise RuntimeError(
        "GitHub returned a truncated repository file list. "
        "Please check the repository structure."
    )

all_paths = [
    item["path"] for item in tree_data.get("tree", [])
    if item.get("type") == "blob"
]

required_files = [
    f"{language}_{split}.csv"
    for language in LANGUAGES
    for split in SPLITS
]

found_paths = {}
for filename in required_files:
    matches = [
        path for path in all_paths
        if path.split("/")[-1].lower() == filename.lower()
    ]
    if matches:
        # Prefer a copy inside the repository's data/ directory.
        matches.sort(key=lambda p: (not p.lower().startswith("data/"), len(p)))
        found_paths[filename] = matches[0]

missing_files = [name for name in required_files if name not in found_paths]

if not missing_files:
    print("Found all six CSV files:")
    for filename, repo_path in found_paths.items():
        print(" ", repo_path)
        file_url = RAW_BASE + requests.utils.quote(repo_path, safe="/")
        file_response = requests.get(file_url, timeout=120)
        file_response.raise_for_status()
        (DATA_DIR / filename).write_bytes(file_response.content)

else:
    # Fallback: look for the cleaned dataset ZIP in the repository.
    zip_matches = [
        path for path in all_paths
        if path.lower().endswith(".zip")
        and ("indic_climate_stage1_cleaned" in path.lower()
             or path.lower().startswith("data/"))
    ]
    if not zip_matches:
        raise FileNotFoundError(
            "Could not find all six cleaned CSV files or a dataset ZIP in "
            f"{GITHUB_OWNER}/{GITHUB_REPO}. Missing: {missing_files}. "
            "Upload the extracted six CSVs or the cleaned ZIP into the "
            "repository's data/ folder, commit the changes, and rerun."
        )

    zip_matches.sort(
        key=lambda p: (
            "indic_climate_stage1_cleaned" not in p.lower(),
            not p.lower().startswith("data/"),
            len(p)
        )
    )
    zip_path = zip_matches[0]
    zip_url = RAW_BASE + requests.utils.quote(zip_path, safe="/")
    zip_response = requests.get(zip_url, timeout=180)
    zip_response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(zip_response.content)) as zf:
        for filename in required_files:
            candidates = [
                name for name in zf.namelist()
                if name.split("/")[-1].lower() == filename.lower()
            ]
            if not candidates:
                raise FileNotFoundError(
                    f"{filename} was not found inside {zip_path}"
                )
            (DATA_DIR / filename).write_bytes(zf.read(candidates[0]))

    print("Downloaded and extracted:", zip_path)

print("Dataset download complete.")


# Load each split. Only the training data is used to fit TF-IDF.
def load_split(language, split):
    csv_path = DATA_DIR / f"{language}_{split}.csv"
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")

    required_columns = {"text", "label"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{csv_path.name} is missing columns: {missing_columns}"
        )

    frame["text"] = frame["text"].fillna("").astype(str)
    frame["label"] = pd.to_numeric(
        frame["label"], errors="raise"
    ).astype(int)

    invalid_labels = set(frame["label"].unique()) - {0, 1, 2}
    if invalid_labels:
        raise ValueError(
            f"Unexpected labels in {csv_path.name}: {invalid_labels}"
        )

    return frame


metrics_rows = []
prediction_frames = []

for language in LANGUAGES:
    print("\\n" + "=" * 65)
    print("E1 —", language.upper())
    print("=" * 65)

    train_df = load_split(language, "train")
    val_df = load_split(language, "val")
    test_df = load_split(language, "test")

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))
    print("Training label distribution:")
    print(train_df["label"].value_counts().sort_index())

    # Word-level TF-IDF using unigrams and bigrams.
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=100000,
        sublinear_tf=True,
        strip_accents=None
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])

    # E1 baseline: Logistic Regression without class weighting.
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=1.0,
        random_state=42
    )
    model.fit(X_train, train_df["label"])

    # Save the trained model and fitted vectorizer for later reuse.
    model_path = MODEL_DIR / f"{language}_E1_tfidf_logistic.joblib"
    joblib.dump(
        {"vectorizer": vectorizer, "model": model},
        model_path
    )

    split_data = [
        ("validation", val_df, X_val),
        ("test", test_df, X_test)
    ]

    for split_name, frame, features in split_data:
        predictions = model.predict(features)

        accuracy = accuracy_score(frame["label"], predictions)
        macro_f1 = f1_score(
            frame["label"], predictions,
            average="macro", zero_division=0
        )
        weighted_f1 = f1_score(
            frame["label"], predictions,
            average="weighted", zero_division=0
        )

        metrics_rows.append({
            "experiment": "E1",
            "model": "TF-IDF + Logistic Regression",
            "language": language,
            "split": split_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "train_samples": len(train_df),
            "evaluated_samples": len(frame)
        })

        report = classification_report(
            frame["label"],
            predictions,
            labels=[0, 1, 2],
            target_names=[
                LABEL_NAMES[0],
                LABEL_NAMES[1],
                LABEL_NAMES[2]
            ],
            output_dict=True,
            zero_division=0
        )
        pd.DataFrame(report).transpose().to_csv(
            OUTPUT_DIR / f"{language}_{split_name}_classification_report.csv"
        )

        cm = confusion_matrix(
            frame["label"], predictions, labels=[0, 1, 2]
        )
        cm_df = pd.DataFrame(
            cm,
            index=["actual_Against", "actual_Favour", "actual_None"],
            columns=["pred_Against", "pred_Favour", "pred_None"]
        )
        cm_df.to_csv(
            OUTPUT_DIR / f"{language}_{split_name}_confusion_matrix.csv"
        )

        prediction_df = frame.copy()
        prediction_df["predicted_label"] = predictions
        prediction_df["actual_class"] = prediction_df["label"].map(LABEL_NAMES)
        prediction_df["predicted_class"] = [
            LABEL_NAMES[int(value)] for value in predictions
        ]
        prediction_df["experiment"] = "E1"
        prediction_df["evaluation_split"] = split_name
        prediction_frames.append(prediction_df)

        print(
            f"{split_name.title()} — "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f} | "
            f"Weighted F1: {weighted_f1:.4f}"
        )

print("\\nE1 training and evaluation completed.")


# Save consolidated metrics and predictions.
metrics_df = pd.DataFrame(metrics_rows)
metrics_path = OUTPUT_DIR / "E1_metrics_summary.csv"
metrics_df.to_csv(metrics_path, index=False)

predictions_df = pd.concat(prediction_frames, ignore_index=True)
predictions_path = OUTPUT_DIR / "E1_all_predictions.csv"
predictions_df.to_csv(predictions_path, index=False)

print("E1 METRICS SUMMARY")
display(metrics_df)

print("\\nSaved output files:")
for file_path in sorted(OUTPUT_DIR.iterdir()):
    print("-", file_path.name)


# Create a downloadable ZIP containing results, trained models and source code.
import shutil
from google.colab import files

source_code = r"""
# E1: TF-IDF + Logistic Regression
# This notebook downloads the cleaned dataset from the project's public GitHub repository.
# See the notebook cells for the complete reproducible workflow.
"""

( WORK_DIR / "E1_source_code.py" ).write_text(
    source_code, encoding="utf-8"
)

readme_text = """
E1 — TF-IDF + Logistic Regression
=================================

Repository:
https://github.com/sarrafharshit411-ops/Indic-Climate-Stance-Detection

Experiment:
- Word-level TF-IDF
- Unigrams and bigrams (1, 2)
- Logistic Regression
- No class weighting (traditional baseline)
- Separate models for Hindi and Bengali

Data:
- Dataset is downloaded automatically from the repository.
- Expected files: hindi_train.csv, hindi_val.csv, hindi_test.csv,
  bengali_train.csv, bengali_val.csv, bengali_test.csv.
- The downloader searches the repository recursively.
- Labels: 0=Against, 1=Favour, 2=None.

Evaluation:
- Validation and test accuracy
- Macro F1 and weighted F1
- Per-class precision, recall and F1
- Confusion matrices
- Prediction CSV

Important:
TF-IDF is fitted only on training text. Validation and test data are
transformed using the training-fitted vectorizer.
"""

(OUTPUT_DIR / "README_E1.txt").write_text(
    readme_text, encoding="utf-8"
)

zip_base = "/content/E1_TFIDF_Logistic_Regression_outputs"
shutil.make_archive(zip_base, "zip", root_dir=WORK_DIR)

zip_file = zip_base + ".zip"
print("Created:", zip_file)
files.download(zip_file)


