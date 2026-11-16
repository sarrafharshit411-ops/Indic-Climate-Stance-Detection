# Initial dataset package

The project’s initial dataset has been supplied as three CSV split files. The uploaded files were inspected and their structure and counts verified.

| File | Rows | Columns |
|---|---:|---|
| train.csv | 2,871 | text, label |
| val.csv | 354 | text, label |
| test.csv | 355 | text, label |
| **Total** | **3,580** | |

## Label distribution

The numeric labels in the supplied files have these counts:

| Label code | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| 0 | 308 | 44 | 35 | 387 |
| 1 | 2,294 | 284 | 285 | 2,863 |
| 2 | 269 | 26 | 35 | 330 |

These totals match the project's previously recorded class distribution. The working label interpretation is 0 = Against, 1 = Favour, and 2 = None; verify this mapping against the original dataset documentation before final publication.

## File integrity checks

- All three files contain exactly the columns `text` and `label`.
- No missing values were found in either column.
- Split row counts and label counts match the project’s recorded statistics.

## Repository placement

Place the supplied CSV files at:
- `data/raw/train.csv`
- `data/raw/val.csv`
- `data/raw/test.csv`

Preserve the supplied train/validation/test split. Do not merge and randomly re-split them. Add a stable source-row ID before translation, and keep all translations of a source row in its original split.

## Provenance and licensing

The exact source URL, dataset version, and redistribution licence still need to be verified. Do not publish the raw files in this public repository until redistribution permission is confirmed. Record the official source, version, licence, and checksums here once verified.
