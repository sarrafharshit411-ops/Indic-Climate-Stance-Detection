# Results directory

Store completed experiments by model and language.

Suggested layout:
```
results/
  tfidf/{hindi,bengali}/
  xlmr/{hindi,bengali}/
  muril/{hindi,bengali}/
  indicbert/{hindi,bengali}/
```

Each completed run should include metrics (accuracy, macro-F1, per-class precision/recall/F1), confusion matrix, experiment configuration, training history/logs, and a run note with the dataset version.

Do not commit placeholder/unverified scores. Avoid large checkpoints by default; use approved model hosting or Git LFS only when agreed by the team.
