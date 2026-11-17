# Dataset and integrity protocol

## Task
Predict stance toward: “Climate change and global warming are a serious concern.”

Labels: `0 = Against`, `1 = Favour`, `2 = None`.

## Splits per language
| Split | Rows |
|---|---:|
| Train | 2,871 |
| Validation | 354 |
| Test | 355 |
| Total | 3,580 |

Hindi and Bengali retain matching source IDs, original texts, labels, and split assignments.

## Class distribution
| Split | Against (0) | Favour (1) | None (2) |
|---|---:|---:|---:|
| Train | 308 | 2,294 | 269 |
| Validation | 44 | 284 | 26 |
| Test | 35 | 285 | 35 |

## Translation and limitations
Hindi and Bengali are machine-translated from English using IndicTrans2. They are synthetic translated data, not independently collected natural Hindi/Bengali social-media posts. Translation artifacts and stance drift may remain. Structural cleaning is not semantic translation validation.

The exact upstream source and redistribution licence remain unverified. Do not publicly distribute the CSVs until verified.

## Structural cleaning
- Collapsed repeated whitespace and trimmed leading/trailing whitespace.
- Preserved wording, punctuation, URLs, hashtags, mentions, labels, IDs, language, and split.
- No rows intentionally deleted, labels changed, or translations manually corrected.
- Audit covered six files / 7,160 language-specific rows: no missing values, empty translations, duplicate IDs, or duplicate full rows.
- Repeated translated strings were retained; identical translated text alone does not prove duplicate source examples.

## Experimental rules
1. Preserve IDs and original split assignments.
2. Fit learned preprocessing on training data only.
3. Calculate class weights from training labels only.
4. Never rebalance validation/test.
5. Use validation for model selection and test only for final evaluation.
6. Report accuracy, macro-F1, per-class precision/recall/F1, confusion matrices.
7. Record model ID, seed, package versions, max length, batch size, epochs, learning rate, and loss.
