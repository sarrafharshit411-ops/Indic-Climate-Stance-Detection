# Dataset

Place the supplied CSV locally in `data/raw/` and cleaned files in `data/processed/`. These folders are ignored by Git to avoid publishing data before its licence and redistribution permissions are confirmed.

## Expected columns

- `text`: input text
- `label`: Against, Favour, or None
- `split`: train, validation, or test

Keep a stable `example_id` in the source data. Hindi/Bengali translations must retain the same ID and original split.

## Current dataset snapshot

3,580 examples: train 2,871; validation 354; test 355. Overall labels: Against 387 (10.81%), Favour 2,863 (79.97%), None 330 (9.22%). The exact provenance and official download URL still require verification. Do not describe the working files as identical to a public release until verified.
