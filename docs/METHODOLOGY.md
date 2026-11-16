# Methodology

## Task

Classify stance toward the claim: “Climate change and global warming are a serious concern.” Labels: Against, Favour, None.

## Planned workflow

1. Dataset audit, label validation, duplicate checks, and split verification.
2. Traditional baselines: TF-IDF with Logistic Regression and Linear SVM; consider word and character features.
3. Translate English examples into Hindi and Bengali using the planned IndicTrans2 workflow. Preserve source IDs and split assignments.
4. Use XLM-R as the primary multilingual transformer baseline. Compare standard cross-entropy with class-weighted cross-entropy.
5. Consider MuRIL as a comparative model if compute/time permit.
6. Evaluate accuracy, macro-F1, per-class precision/recall/F1, and confusion matrices.

## Preliminary English-only results

These values were recorded in project notes. They have not been reproduced from code in this repository and are not Hindi/Bengali results.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Logistic Regression | 80.23% | 32.13% |
| Linear SVM | 81.07% | 42.89% |
| Weighted Linear SVM | 79.94% | 46.97% |
| Word + character TF-IDF SVM | 79.66% | 46.24% |
| XLM-R, weighted CE | 73.16% | 52.97% |
| MuRIL | 78.81% | 40.18% |

## Research gap

Study how machine-translated Hindi and Bengali versions of an English climate-stance dataset affect multilingual stance classification, particularly minority-class performance. Conclusions must be limited to translated benchmark data; this project does not by itself represent naturally authored Indic climate discourse.
