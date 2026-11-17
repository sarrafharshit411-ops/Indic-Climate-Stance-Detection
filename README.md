# 🌍 Indic Climate Stance Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![NLP](https://img.shields.io/badge/NLP-Multilingual-green?style=for-the-badge&logo=huggingface)
![XLM-R](https://img.shields.io/badge/Model-XLM--R%20%7C%20MuRIL-orange?style=for-the-badge)
![Languages](https://img.shields.io/badge/Languages-English%20%7C%20Hindi%20%7C%20Bengali-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active%20Research-red?style=for-the-badge)

**Multilingual stance classification for the claim:**
> *"Climate change and global warming are a serious concern."*

**Labels:** `Against` · `Favour` · `None`

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Research Question](#-research-question)
- [Methodology](#-methodology)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)
- [Data Integrity](#-data-integrity)
- [Contributors](#-contributors)
- [Citation](#-citation)

---

## 🔭 Overview

This project addresses **multilingual stance detection** in the context of climate change discourse across Indian languages. The core challenge: given a text in English, Hindi, or Bengali, classify its stance towards the target claim into one of three categories — **Against**, **Favour**, or **None**.

The pipeline leverages:
- **Machine translation** (IndicTrans2) to synthesize Hindi and Bengali training data from an English stance dataset
- **Traditional ML baselines** (TF-IDF + Logistic Regression / SVM)
- **State-of-the-art multilingual transformers** (XLM-R, MuRIL)
- Rigorous **class-imbalance handling** via weighted loss functions

This work directly contributes to the growing body of research on **low-resource NLP for Indic languages** and **computational climate communication**.

---

## 📊 Dataset

| Split      | Examples |
|------------|----------|
| Train      | 2,871    |
| Validation | 354      |
| Test       | 355      |
| **Total**  | **3,580**|

### Label Distribution

| Label   | Count | Percentage |
|---------|-------|------------|
| Favour  | 2,863 | 79.97%     |
| Against | 387   | 10.81%     |
| None    | 330   | 9.22%      |

> ⚠️ **Note:** The dataset is heavily imbalanced with `Favour` as the dominant class. Class-weighted training strategies are employed to address minority-class recognition for `Against` and `None`.

> ⚠️ The exact source provenance, official download URL, and redistribution licence still require verification. **Do not commit raw data to this repository until confirmed.**

**Expected CSV format:**
```
text,label,split
"The earth is getting hotter...",Favour,train
"This is exaggerated...",Against,validation
```
Columns: `text`, `label` (Against/Favour/None), `split` (train/validation/test)

---

## ❓ Research Question

> *How does machine translation of an English stance dataset into Hindi and Bengali affect multilingual stance classification — particularly minority-class recognition?*

Hindi and Bengali examples are **synthetic machine translations** (using [IndicTrans2](https://github.com/AI4Bharat/IndicTrans2)), not naturally collected Indic-language social media posts. This introduces a controlled setting for studying cross-lingual transfer while acknowledging translation noise as a confounding factor.

---

## 🧪 Methodology

### Phase 1 — Data Preparation
1. Audit raw text for duplicates, encoding issues, and label inconsistencies
2. Preserve source IDs and train/validation/test assignments across all translated versions
3. Translate English training set to Hindi and Bengali using **IndicTrans2**
4. Validate split integrity: no translated copy of a training example may appear in validation or test

### Phase 2 — Baseline Models
3. **TF-IDF + Logistic Regression** — word unigram/bigram features
4. **Linear SVM** — standard and class-weighted variants
5. **TF-IDF SVM** — combined word + character n-gram features

### Phase 3 — Transformer Fine-tuning
4. **XLM-RoBERTa (XLM-R)** — standard and class-weighted cross-entropy
5. **MuRIL** — Google's multilingual model pretrained on Indic scripts (compute-permitting)

### Phase 4 — Evaluation
6. Report: **Accuracy**, **Macro-F1**, per-class Precision / Recall / F1, and **Confusion Matrices**

---

## 📈 Results

> **Note:** These are preliminary English-only results recorded in project notes. They have not yet been reproduced end-to-end from code in this repository. Hindi/Bengali multilingual results are pending.

| Model                        | Accuracy | Macro-F1 |
|------------------------------|:--------:|:--------:|
| Logistic Regression          |  80.23%  |  32.13%  |
| Linear SVM                   |  81.07%  |  42.89%  |
| Weighted Linear SVM          |  79.94%  |  46.97%  |
| Word + Character TF-IDF SVM  |  79.66%  |  46.24%  |
| **XLM-R (Weighted CE)**      |**73.16%**|**52.97%**|
| MuRIL                        |  78.81%  |  40.18%  |

> XLM-R achieves the best Macro-F1 by better handling minority classes, despite slightly lower overall accuracy — consistent with class-imbalance behaviour.

---

## 🗂️ Project Structure

```
Indic-Climate-Stance-Detection/
│
├── data/                        # Dataset placement (see Data Integrity)
│   ├── raw/                     # Original source data (do NOT commit without licence)
│   └── processed/               # Cleaned, translated, and split datasets
│
├── docs/                        # Documentation and protocols
│   ├── CONTRIBUTIONS.md         # Actual contribution evidence (commits, PRs, results)
│   ├── DATASET_PROTOCOL.md      # Data handling and integrity rules
│   └── METHODOLOGY.md           # Detailed experimental design
│
├── notebooks/                   # Exploratory analysis and experiments
│   └── Indic_Climate_Stance_Challenge_v01 (1).ipynb
│
├── results/                     # Reproducible metrics and plots
│   ├── confusion_matrices/
│   └── metrics.csv
│
├── src/                         # Training and evaluation scripts
│   └── train_baseline.py        # Baseline model training
│
├── .gitignore
├── CONTRIBUTING.md              # Contribution workflow guide
├── README.md                    # This file
└── requirements.txt             # Python dependencies
```

---

## ⚙️ Setup & Usage

### Prerequisites
- Python **3.10+**
- pip

### Installation

```bash
git clone https://github.com/sarrafharshit411-ops/Indic-Climate-Stance-Detection.git
cd Indic-Climate-Stance-Detection
pip install -r requirements.txt
```

### Run a Baseline

```bash
python -m src.train_baseline \
    --data data/processed/dataset.csv \
    --model linear_svm \
    --class-weight balanced
```

### Available Model Options
| `--model` flag     | Description                        |
|--------------------|------------------------------------|
| `logistic`         | Logistic Regression + TF-IDF       |
| `linear_svm`       | Linear SVM                         |
| `char_svm`         | Word + Character n-gram TF-IDF SVM |
| `xlmr`             | XLM-RoBERTa (fine-tuned)           |

---

## 🔒 Data Integrity

> These rules are **mandatory** for all contributors.

- ❌ Do **not** commit raw data until its licence and redistribution permission are verified
- ❌ Never allow translated copies of one source example to cross dataset splits (train/val/test contamination)
- ✅ Fit all preprocessing (vectorizers, scalers) on **training data only**
- ✅ Use **validation** for model selection; reserve **test** for final one-time evaluation
- ❌ Do not commit model checkpoints or generated datasets (see `.gitignore`)
- ✅ All results must record dataset version, random seed, and configuration

---

## 👥 Contributors

This is a collaborative team project with equal contribution targets across four workstreams:

| Contributor | GitHub | Workstream |
|---|---|---|
| **Harshit Raj** | [@sarrafharshit411-ops](https://github.com/sarrafharshit411-ops) | XLM-R fine-tuning, experiment integration, reproducibility |
| **Kavvya Kumar Yadav** | — | Dataset audit, EDA, TF-IDF / traditional ML |
| **Shwet Shivang** | [@ShwetShivang](https://github.com/ShwetShivang) | Transformer model comparison, evaluation framework, error analysis |
| **Sanskar Garg** | — | IndicTrans2 integration, Hindi/Bengali translation pipeline & quality checks |

> Contribution targets are planning goals. Actual contributions are tracked via commits, pull requests, reviews, and experiment evidence in [docs/CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md).

### Shwet Shivang's Contributions
- **Transformer Comparison**: Benchmarking XLM-R against MuRIL across all language settings
- **Evaluation Framework**: Implementing per-class precision/recall/F1 reporting and confusion matrix visualization
- **Error Analysis**: Investigating failure modes on minority classes (`Against`, `None`) and cross-lingual transfer errors
- **Code Review**: Reviewing baseline implementation, data pipeline, and translation quality scripts

---

## 📄 Citation

If you use this codebase or methodology in your research, please cite:

```bibtex
@misc{indicclimancestance2024,
  title        = {Indic Climate Stance Detection: Multilingual Stance Classification for Climate Change Claims},
  author       = {Harshit Raj and Kavvya Kumar Yadav and Shwet Shivang and Sanskar Garg},
  year         = {2024},
  howpublished = {\url{https://github.com/sarrafharshit411-ops/Indic-Climate-Stance-Detection}},
  note         = {Team project}
}
```

---

## 📚 References

- [IndicTrans2 — AI4Bharat](https://github.com/AI4Bharat/IndicTrans2) — Neural machine translation for Indian languages
- [XLM-RoBERTa — Conneau et al. (2020)](https://arxiv.org/abs/1911.02116) — Unsupervised cross-lingual representation learning
- [MuRIL — Google Research](https://huggingface.co/google/muril-base-cased) — Multilingual representations for Indian languages
- [SemEval-2016 Task 6](https://arxiv.org/abs/1606.05943) — Detecting Stance in Tweets (original stance detection benchmark)

---

<div align="center">

Made with ❤️ by the Indic Climate Stance Detection Team

[![GitHub](https://img.shields.io/badge/GitHub-sarrafharshit411--ops-black?style=flat-square&logo=github)](https://github.com/sarrafharshit411-ops/Indic-Climate-Stance-Detection)

</div>
