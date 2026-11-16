# Dataset and experimental protocol

## Working dataset statistics

| Split | Examples |
|---|---:|
| Train | 2,871 |
| Validation | 354 |
| Test | 355 |
| Total | 3,580 |

| Label | Count | Share |
|---|---:|---:|
| Against | 387 | 10.81% |
| Favour | 2,863 | 79.97% |
| None | 330 | 9.22% |

Recorded split-level counts:
- Train: Against 308, Favour 2,294, None 269
- Validation: Against 44, Favour 284, None 26
- Test: Against 35, Favour 285, None 35

The supplied files have been described by the team as ClimateStance-based. Exact source identity, official URL, version, and licence are not yet independently confirmed.

## Translation and leakage control

Hindi and Bengali are planned machine translations of the English examples (IndicTrans2 is the planned candidate). They are synthetic translated examples, not naturally collected Indic-language social posts. Assign stable source IDs before translation and preserve each original train/validation/test assignment across all languages. Never let translations of one source item cross split boundaries.

## Protocol

1. Audit missing text, labels, duplicates, and split counts.
2. Verify source provenance and redistribution rights.
3. Translate while preserving source ID and split.
4. Inspect a stratified sample for negation, stance preservation, and translation errors.
5. Fit vectorizers and model parameters on training data only.
6. Select hyperparameters using validation data only.
7. Compute class weights from training labels only.
8. Freeze decisions before final test evaluation.

Report accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix. Macro-F1 is essential because Favour is about 80% of the dataset.
