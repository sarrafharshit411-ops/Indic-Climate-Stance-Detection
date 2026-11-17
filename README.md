# 🌍 Indic Climate Stance Detection

**Three-class stance classification** for the claim: “Climate change and global warming are a serious concern.”

Labels: `Against` · `Favour` · `None`

## Project status

This repository contains the project's initial notebook, data protocol, methodology notes, and experiment scaffolding. Model comparisons are ongoing; the presence of an experiment plan does not mean training has completed.

## Dataset

The supplied labelled English dataset contains 3,580 source examples:

| Split | Rows |
|---|---:|
| Train | 2,871 |
| Validation | 354 |
| Test | 355 |

Hindi and Bengali versions were generated using AI4Bharat IndicTrans2. These are **machine-translated examples**, not naturally collected Hindi/Bengali social-media posts.

Training distribution:

| Label | Count | Share |
|---|---:|---:|
| Against (0) | 308 | 10.73% |
| Favour (1) | 2,294 | 79.90% |
| None (2) | 269 | 9.37% |

The upstream source provenance and redistribution licence have not yet been independently verified. The CSV files are therefore not included in this repository update.

## Methodology

1. Preserve source IDs, labels, and train/validation/test split assignments across translations.
2. Perform structural validation and conservative whitespace normalization.
3. Establish traditional baselines (TF-IDF + Logistic Regression, Linear SVM, Naive Bayes).
4. Fine-tune multilingual/Indic transformers (XLM-R and MuRIL; other candidates subject to feasibility).
5. Compare standard and class-weighted cross-entropy using training-derived weights only.
6. Evaluate on untouched validation/test splits using macro-F1, accuracy, per-class precision/recall/F1, and confusion matrices.

See [dataset protocol](docs/DATASET_PROTOCOL.md) and [methodology](docs/METHODOLOGY.md).

## Repository structure

```
data/       Dataset instructions and protocol (data files excluded pending licence review)
docs/       Dataset protocol, methodology, contribution evidence
notebooks/  Experiment notebooks and notebook guide
src/        Reusable training/evaluation scripts
results/    Reproducible metrics and result guides
```

## Setup

```bash
git clone https://github.com/sarrafharshit411-ops/Indic-Climate-Stance-Detection.git
cd Indic-Climate-Stance-Detection
pip install -r requirements.txt
```

The existing baseline notebook and scripts are retained. Refer to their individual instructions before running.

## Contribution tracking

Contributions are recorded with dated, verifiable commit/PR/artifact evidence in [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md). Planned assignments and percentage targets are not presented as completed work.

## Limitations

- Translation quality may affect stance meaning; structural cleaning does not establish semantic correctness.
- Source dataset provenance and redistribution permission require verification.
- Results must only be reported after the corresponding run is complete and reproducible.
