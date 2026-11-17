# Source code

Place reusable scripts here rather than duplicating notebook logic.

Recommended modules:
- `data_validation.py`: schema and split integrity checks
- `train_tfidf.py`: traditional ML baselines
- `train_transformer.py`: configurable transformer training
- `evaluate.py`: shared metrics and confusion matrix export

Scripts should accept explicit file paths/configuration. Fit learned preprocessing on training data only.
