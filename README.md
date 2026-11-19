# 🌍 Indic Climate Stance Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![NLP](https://img.shields.io/badge/Task-Multilingual%20Stance%20Detection-green?style=for-the-badge)
![Models](https://img.shields.io/badge/Models-TF--IDF%20%7C%20XLM--R%20%7C%20MuRIL-orange?style=for-the-badge)
![Languages](https://img.shields.io/badge/Languages-English%20%7C%20Hindi%20%7C%20Bengali-purple?style=for-the-badge)
![Status](https://img.shields.io/badge/Stage-Milestone%202%20Evaluation-blueviolet?style=for-the-badge)

**Claim-specific stance classification for climate-change-related social-media text**

> “Climate change and global warming is a serious concern.”

**Labels:** `Against` · `Favour` · `None`

**CSET346 — Natural Language Processing | Academic Year 2026–27**

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Task Definition](#task-definition)
- [Dataset](#dataset)
- [Multilingual Translation](#multilingual-translation)
- [Methodology and Experiments](#methodology-and-experiments)
- [Experimental Results](#experimental-results)
- [Class-wise Evaluation](#class-wise-evaluation)
- [Repository Structure](#repository-structure)
- [Environment and Usage](#environment-and-usage)
- [Limitations and Reproducibility Notes](#limitations-and-reproducibility-notes)
- [Team Members](#team-members)
- [References](#references)

---

## Project Overview

This project develops and evaluates a three-class stance-detection workflow for social-media posts discussing climate change. For a fixed claim, the system predicts whether a post expresses **Against**, **Favour**, or **None**.

The project compares traditional TF-IDF-based machine-learning classifiers with multilingual transformer models. The experimental workflow includes English validation experiments and separate Hindi and Bengali test experiments using translated versions of the working English dataset.

The central research focus is **minority-class recognition under substantial class imbalance**, not simply maximising overall accuracy.

## Task Definition

| Label | Meaning |
|---|---|
| `Against` (0) | The post opposes or disputes the target claim |
| `Favour` (1) | The post supports the target claim |
| `None` (2) | The post expresses no clear stance toward the target claim |

This is **stance detection**, not topic detection or general sentiment analysis. A post can mention climate change without supporting or opposing the specified claim.

## Dataset

The working dataset contains **3,580 labelled examples**.

### Split statistics

| Split | Against | Favour | None | Total |
|---|---:|---:|---:|---:|
| Train | 308 | 2,294 | 269 | 2,871 |
| Validation | 44 | 284 | 26 | 354 |
| Test | 35 | 285 | 35 | 355 |
| **Overall** | **387** | **2,863** | **330** | **3,580** |

### Overall class distribution

| Class | Examples | Share |
|---|---:|---:|
| Favour | 2,863 | 79.97% |
| Against | 387 | 10.81% |
| None | 330 | 9.22% |

The dataset is strongly imbalanced. A classifier that predicts only the majority class can obtain approximately 80.28% test accuracy while failing to identify either minority class. For this reason, **Macro-F1 is the primary model-comparison metric**, alongside accuracy, Weighted-F1, per-class precision/recall/F1, and confusion matrices.

> **Dataset provenance:** The current working set contains 3,580 examples. Project notes report 3,777 examples in the referenced ClimateStance publication; the 197-example difference has not been resolved. The exact working-file source identity, official download URL, version, and redistribution licence require confirmation. Do not treat the working files as an independently verified official course dataset.

## Multilingual Translation

The working English examples were translated into Hindi and Bengali using AI4Bharat IndicTrans2.

**Translation checkpoint:** [ai4bharat/indictrans2-en-indic-dist-200M](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M)  
**Toolkit:** [IndicTransToolkit](https://github.com/AI4Bharat/IndicTrans2)

| Setting | Value |
|---|---|
| Source language | `eng_Latn` |
| Hindi target | `hin_Deva` |
| Bengali target | `ben_Beng` |
| Maximum sequence length | 256 |
| Beam size | 4 |
| Batch size | 4 |
| Intermediate checkpoint frequency | Every 10 batches |

The original train/validation/test membership and labels were preserved during translation.

**Important interpretation:** Hindi and Bengali examples are machine translations of English posts and inherit their English labels. They are not naturally authored Indic-language posts and were not independently annotated by Hindi/Bengali stance annotators. Structural matching checks do not prove that stance meaning was preserved in every translation.

The supplied preprocessing records describe different details for different stages: the English experimental-analysis notes report URL removal and whitespace normalisation, while the Stage 1 translation/audit notes report whitespace trimming/collapse with URLs, hashtags, mentions, and punctuation preserved in the translated data. These descriptions should not be conflated as one identical preprocessing pipeline.

## Methodology and Experiments

Six experimental groups were evaluated:

| ID | Model | Purpose |
|---|---|---|
| E1 | TF-IDF + Logistic Regression | Traditional baseline |
| E2 | Word TF-IDF + Linear SVM | Unweighted margin-based baseline |
| E3 | Class-weighted word TF-IDF + Linear SVM | Address class imbalance |
| E4 | Word + character TF-IDF + class-weighted Linear SVM | Combine lexical and character features |
| E5 | XLM-RoBERTa-base | Multilingual contextual transformer |
| E6 | MuRIL-base-cased | Transformer for Indian-language use cases |

### Classical ML workflow

- TF-IDF vectorisation was fitted on training data and applied to validation/test data.
- Logistic Regression and Linear SVM were used as traditional baselines.
- Class weighting was applied in E3 and E4.
- E4 combines word-level and character-level TF-IDF features.

### Transformer workflow

The supplied Hindi/Bengali XLM-R experiments used the following recorded configuration:

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Maximum sequence length | 160 |
| Batch size | 8 |
| Encoder learning rate | 1e-5 |
| Classifier-head learning rate | 3e-5 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Random seed | 42 |
| Scheduler | Linear warmup/decay |
| Model selection | Validation Macro-F1; validation loss as tie-breaker |

Standard and class-weighted cross-entropy variants were compared where the corresponding experiment was run. Class weights were calculated from training labels only.

### Why class-weighted loss?

The dominant Favour class represents 79.97% of the working dataset. Standard cross-entropy can result in majority-class collapse. Weighted cross-entropy increases the loss contribution of minority-class errors, encouraging the model to learn Against and None as well.

## Experimental Results

**Evaluation splits are different:** English metrics below are validation results; Hindi and Bengali metrics are test results. They must not be interpreted as a direct like-for-like comparison across languages.

### English — validation results

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|
| E1 Logistic Regression | 80.23% | 32.13% | 71.94% |
| E2 Linear SVM | 81.07% | 42.89% | 75.35% |
| E3 Weighted Linear SVM | 79.94% | 46.97% | 76.80% |
| E4 Word + Character Weighted SVM | 79.66% | 46.24% | 76.04% |
| E5 XLM-R Weighted CE | 73.16% | **52.97%** | 75.15% |
| E6 MuRIL Weighted CE | 78.81% | 40.18% | 73.69% |

Among the supplied English validation results, weighted XLM-R achieved the highest Macro-F1 (52.97%). Linear SVM achieved the highest accuracy (81.07%), but its Against F1 was only 0.076.

**English standard-cross-entropy XLM-R and MuRIL metrics were not present in the supplied English result files.** They are unavailable, not zero, and are not inferred from Indic experiments.

### Hindi and Bengali — test results

| Model / Variant | Hindi Accuracy | Hindi Macro-F1 | Bengali Accuracy | Bengali Macro-F1 |
|---|---:|---:|---:|---:|
| E1 Logistic Regression | 80.28% | 29.69% | 80.28% | 29.73% |
| E2 Linear SVM | 79.72% | 36.19% | 80.00% | 33.16% |
| E3 Weighted SVM | 79.72% | 43.39% | 79.15% | 38.55% |
| E4 Word + Character Weighted SVM | 81.69% | 50.40% | 79.72% | 40.17% |
| E5 XLM-R Standard CE | 80.28% | 29.69% | 80.28% | 29.69% |
| **E5 XLM-R Weighted CE** | **73.80%** | **52.01%** | **70.14%** | **43.95%** |
| E6 MuRIL Standard CE | 80.28% | 29.69% | 80.28% | 29.69% |
| E6 MuRIL Weighted CE | 78.03% | 42.45% | 78.03% | 39.88% |

Additional Weighted-F1 results:

| Model | Hindi Weighted-F1 | Bengali Weighted-F1 |
|---|---:|---:|
| E1 Logistic Regression | 71.50% | 71.61% |
| E2 Linear SVM | 73.08% | 72.56% |
| E3 Weighted SVM | 75.25% | 73.77% |
| E4 Word + Character Weighted SVM | 77.89% | 74.48% |
| E5 XLM-R Weighted CE | 75.38% | 71.26% |
| E6 MuRIL Weighted CE | 74.66% | 73.93% |

### Main findings

- Weighted XLM-R achieved the highest reported Indic test Macro-F1 in both languages: **52.01% Hindi** and **43.95% Bengali**.
- It exceeded the E4 word + character weighted SVM by 1.61 percentage points in Hindi and 3.78 points in Bengali.
- Standard XLM-R and standard MuRIL predicted every Indic test example as Favour. Their 80.28% accuracy reflects the majority-class share and not balanced stance recognition.
- Weighted training improved minority-class recognition, while overall accuracy decreased.
- E4 remains a strong classical baseline, especially for Hindi (50.40% Macro-F1).

## Class-wise Evaluation

### Weighted XLM-R — Hindi

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Against | 34.29% | 34.29% | 34.29% |
| Favour | 88.93% | 81.75% | 85.19% |
| None | 29.31% | 48.57% | 36.56% |
| **Macro-F1** | | | **52.01%** |

Confusion matrix (actual rows, predicted columns; order: Against, Favour, None):

```text
              Predicted
             A    F    N
Actual A    12   17    6
       F    17  233   35
       N     6   12   17
```

### Weighted XLM-R — Bengali

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Against | 24.24% | 22.86% | 23.53% |
| Favour | 84.87% | 80.70% | 82.73% |
| None | 21.57% | 31.43% | 25.58% |
| **Macro-F1** | | | **43.95%** |

Confusion matrix (actual rows, predicted columns; order: Against, Favour, None):

```text
              Predicted
             A    F    N
Actual A     8   21    6
       F    21  230   34
       N     4   20   11
```

The model performs substantially better on Favour than on either minority class. Each minority class has only 35 test examples per language, so individual predictions can noticeably affect the reported scores.

## Repository Structure

The repository includes experiment notebooks, baseline/evaluation scripts, documentation, and result artifacts. The existing top-level structure is:

```text
Indic-Climate-Stance-Detection/
├── data/
├── docs/
│   ├── CONTRIBUTIONS.md
│   ├── DATASET_AND_PROTOCOL.md
│   └── METHODOLOGY.md
├── notebooks/
│   ├── Hindi and Bengali XLM-R experiments
│   ├── TF-IDF baseline experiments
│   ├── MuRIL experiments
│   └── compact result notebooks
├── results/
│   ├── E2_small_results/
│   ├── E3_small_results/
│   ├── language-specific result archives
│   └── experimental analysis document
├── src/
│   ├── train_baseline.py
│   └── evaluate.py
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

See the repository's `notebooks/`, `results/`, and `docs/` directories for the actual experiment artifacts and supporting documentation.

## Environment and Usage

### Requirements

- Python 3.10+
- pip
- For transformer fine-tuning: a compatible PyTorch/Transformers environment and sufficient GPU memory are recommended.
- Classical baselines can be run on CPU.

### Clone the repository

```bash
git clone https://github.com/sarrafharshit411-ops/Indic-Climate-Stance-Detection.git
cd Indic-Climate-Stance-Detection
```

### Install the repository's listed dependencies

```bash
pip install -r requirements.txt
```

The current `requirements.txt` lists the lightweight baseline dependencies:

```text
pandas
scikit-learn
numpy
matplotlib
```

Transformer notebooks may require additional packages (for example, PyTorch, Transformers, Datasets, SentencePiece, and IndicTransToolkit). Install the versions required by the individual notebook/runtime. The current requirements file is not a complete pinned environment for every transformer or translation notebook.

### Run the baseline script

The repository contains `src/train_baseline.py`. Check its command-line options before running:

```bash
python src/train_baseline.py --help
```

For transformer experiments, open the relevant notebook under `notebooks/` in Google Colab or a compatible Jupyter environment, select the appropriate GPU runtime, and follow the cells in order. Dataset files and model checkpoints may need to be supplied separately; they are not assumed to be included in the repository.

> The README documents the measured experiments. It does not claim that every experiment can be reproduced from a single command with only the current lightweight requirements file.

## Limitations and Reproducibility Notes

1. **Machine-translated Indic data:** Hindi and Bengali texts are translated from English and inherit English labels. Translation may introduce semantic drift, unnatural phrasing, or stance-label noise.
2. **No independent Indic annotation:** Independent human stance annotations for Hindi and Bengali were not supplied.
3. **Dataset provenance discrepancy:** The working set contains 3,580 examples, while project notes attribute 3,777 examples to the referenced ClimateStance publication. The 197-record difference and exact provenance remain unresolved.
4. **Single-seed experiments:** The reported transformer results are single runs with seed 42. No mean, standard deviation, or statistical-significance claim is available.
5. **Small minority test support:** Each Hindi/Bengali test split contains only 35 Against and 35 None examples.
6. **English standard-CE result gap:** English standard-cross-entropy XLM-R and MuRIL numeric results were not found in the supplied English artifacts.
7. **Preprocessing notes:** The English experimental-analysis notes and Stage 1 translation audit describe different URL-handling details. They should be documented as separate pipeline descriptions.

These results should be treated as experimental findings on translated working data, not as a production-ready classifier or a validated measurement of naturally authored Indic public opinion.

## Team Members

**Course:** CSET346 — Natural Language Processing  
**Institution:** Bennett University  
**Academic Year:** 2026–27

| Team Member | University ID | GitHub Profile |
|---|---|---|
| **Harshit Raj** | `s24cseu0733` | [@sarrafharshit411-ops](https://github.com/sarrafharshit411-ops) |
| **Kavvya Kumar Yadav** | `s24cseu0651` | [@K-Kumaleo](https://github.com/K-Kumaleo) |
| **Shwet Shivang** | `s24csu0537` | [@ShwetShivang](https://github.com/ShwetShivang) |
| **Sanskar Garg** | `s24cseu0397` | [@sanskar171](https://github.com/sanskar171) |

### Contribution tracking

The repository's [contribution register](docs/CONTRIBUTIONS.md) records the team's planned workstreams and an evidence-based completion log. Workstream descriptions are responsibility areas, not independently audited proof of completed individual work. Commit, pull-request, review, and experiment links should be added as work is delivered.

All team members are expected to participate in experiment documentation, review, error analysis, and final reporting. Individual contribution claims should be supported by repository history and project artifacts.

## References

1. Mohammad, S. M., et al. (2016). [SemEval-2016 Task 6: Detecting Stance in Tweets](https://aclanthology.org/S16-1003/).
2. Vaid, R., Pant, K., & Shrivastava, M. (2022). [Towards Fine-grained Classification of Climate Change Related Social Media Text](https://aclanthology.org/2022.acl-srw.35/).
3. Conneau, A., et al. (2020). [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/).
4. Khanuja, S., et al. (2021). [MuRIL: Multilingual Representations for Indian Languages](https://arxiv.org/abs/2103.10730).
5. AI4Bharat. [IndicTrans2 model checkpoint](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M) and [IndicTrans2 toolkit](https://github.com/AI4Bharat/IndicTrans2).
6. Luo, et al. (2020). [Global Warming Stance Detection (GWSD)](https://aclanthology.org/2020.findings-emnlp.296/).
7. Shiwakoti, et al. (2024). [ClimaConvo](https://aclanthology.org/2024.lrec-main.88/).

---

<div align="center">

**Made by the Indic Climate Stance Detection Team**

[Repository](https://github.com/sarrafharshit411-ops/Indic-Climate-Stance-Detection) · [Issues](https://github.com/sarrafharshit411-ops/Indic-Climate-Stance-Detection/issues)

</div>
