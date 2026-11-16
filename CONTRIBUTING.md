# Contributing

1. Pull the latest `main` branch.
2. Create a focused branch, e.g. `feature/translation-pipeline` or `experiment/xlmr-weighted-loss`.
3. Make one logical change at a time.
4. Run relevant checks and document commands/results.
5. Push the branch and open a pull request to `main`.
6. Request at least one teammate review.
7. Merge after review and update `docs/CONTRIBUTIONS.md` with evidence.

## Pull request checklist

- [ ] Purpose and scope are clear.
- [ ] Dataset split integrity is preserved.
- [ ] No restricted raw data, credentials, or personal data are included.
- [ ] Results record dataset version, seed, and configuration.
- [ ] Documentation is updated.
- [ ] A teammate has reviewed the change.

Use Python 3.10+. Keep preprocessing, training, and evaluation separate. Fit preprocessing on training data only. Do not commit checkpoints or generated datasets.
