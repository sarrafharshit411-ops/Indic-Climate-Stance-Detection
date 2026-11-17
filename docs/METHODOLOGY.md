# Methodology

## Problem
Three-class stance classification for the claim: “Climate change and global warming are a serious concern.”

Labels: `0=Against`, `1=Favour`, `2=None`.

## Data pipeline
1. Start from supplied labelled English data (3,580 examples).
2. Preserve train (2,871), validation (354), and test (355) assignments.
3. Translate into Hindi and Bengali using AI4Bharat IndicTrans2.
4. Retain source IDs and labels across languages.
5. Run structural checks and conservative whitespace normalization.
6. Treat translation quality as a limitation requiring further semantic review.

## Model families planned / under investigation
- Traditional baselines: TF-IDF + Logistic Regression, Linear SVM, Naive Bayes.
- Multilingual transformer: XLM-RoBERTa.
- Indic-focused transformer: MuRIL.
- IndicBERT v2 and Bengali-specific models may be evaluated after checkpoint suitability and compute-budget checks.

Listing a model does not mean an experiment is complete. Add results only after reproducible training.

## Class imbalance
Training counts: Against 308, Favour 2,294, None 269. The planned study compares standard and class-weighted cross-entropy. Weights are computed from training labels only; validation/test remain unchanged.

## Evaluation
Primary metric: macro-F1. Also report accuracy, per-class precision/recall/F1 and confusion matrices. Compare languages/models using consistent protocols where feasible.

## Run metadata
Record language, dataset version, checkpoint, seed, package versions, max sequence length, batch size, epochs, learning rate, loss configuration, selected validation epoch, test metrics, and limitations.
