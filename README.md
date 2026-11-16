# Indic Climate Stance Detection

Multilingual stance classification for the claim: **“Climate change and global warming are a serious concern.”** The three labels are **Against**, **Favour**, and **None**.

## Dataset snapshot

The team's working files contain 3,580 examples: 2,871 train, 354 validation, and 355 test. Overall labels: Against 387 (10.81%), Favour 2,863 (79.97%), None 330 (9.22%). The exact source provenance, official download URL, and redistribution licence still require verification.

## Research question

How does machine translation of the English stance dataset into Hindi and Bengali affect multilingual stance classification, especially minority-class recognition?

Hindi and Bengali examples will be synthetic machine translations, not naturally collected Indic-language posts. IndicTrans2 is the planned translation candidate.

## Planned methodology

1. Audit text, labels, duplicates, and split integrity.
2. Preserve source IDs and train/validation/test assignments across translations.
3. Establish TF-IDF Logistic Regression and Linear SVM baselines.
4. Fine-tune XLM-R with standard and class-weighted cross-entropy.
5. Consider MuRIL as a comparison if compute/time permit.
6. Report accuracy, macro-F1, per-class precision/recall/F1, and confusion matrices.

## Preliminary English-only results

These figures are recorded in project notes; they have not yet been reproduced from code in this repository and are not Hindi/Bengali results.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 80.23% | 32.13% |
| Linear SVM | 81.07% | 42.89% |
| Weighted Linear SVM | 79.94% | 46.97% |
| Word + character TF-IDF SVM | 79.66% | 46.24% |
| XLM-R, weighted CE | 73.16% | 52.97% |
| MuRIL | 78.81% | 40.18% |

## Equal team contribution target

| Member | Target | Workstream |
|---|---:|---|
| Harshit Raj | 25% | XLM-R, experiment integration, reproducibility |
| Kavvya Kumar Yadav | 25% | Dataset audit, EDA, TF-IDF/traditional ML |
| Shwet Shivang | 25% | Transformer comparison, evaluation, error analysis |
| Sanskar Garg | 25% | IndicTrans2, Hindi/Bengali translation and quality checks |

These are agreed planning targets, not claims of completed work. Track actual work with commits, pull requests, reviews, and experiment evidence in [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md).

## Repository guide

- `src/`: baseline training and evaluation scripts
- `data/`: dataset placement and handling rules
- `docs/`: dataset protocol, methodology, and contribution plan
- `notebooks/`: exploratory analysis and experiments
- `results/`: compact reproducible metrics and plots

## Setup

Python 3.10+. Install dependencies:

```bash
pip install -r requirements.txt
```

Run a baseline:

```bash
python -m src.train_baseline --data data/processed/dataset.csv --model linear_svm --class-weight balanced
```

Expected CSV columns: `text,label,split`; labels: Against/Favour/None; split values: train/validation/test.

## Data integrity

Do not commit raw data until its licence and redistribution permission are verified. Never allow translated copies of one source example to cross dataset splits. Fit preprocessing on training data only; use validation for model selection and reserve test for final evaluation.
