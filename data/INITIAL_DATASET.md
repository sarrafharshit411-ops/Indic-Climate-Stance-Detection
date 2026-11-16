# Initial dataset package

The initial dataset was supplied to the project as `Dataset — ClimateStance Files.zip`. It contains three CSV files:

| File | Rows | Columns |
|---|---:|---|
| train.csv | 2,871 | text, label |
| val.csv | 354 | text, label |
| test.csv | 355 | text, label |

Combined: 3,580 examples. Label codes in the supplied files are numeric:
- `0`: Against (counts match the team's recorded Against totals)
- `1`: Favour
- `2`: None

Combined class counts: 0 = 387, 1 = 2,863, 2 = 330.

## Handling

The supplied CSVs are the original initial split files. Keep the train/validation/test split unchanged. Do not merge and randomly re-split them. Add a stable source-row ID before translation, and keep all translations of a source row in its original split.

## Repository upload status

The project documentation and scripts are in this repository. The actual CSV payloads have not yet been committed. The connected GitHub file-writing capability accepts UTF-8 text files only and cannot transfer the uploaded ZIP/binary dataset directly. The dataset source licence and redistribution permission should also be checked before publishing the CSVs in this public repository.

To add the data manually, extract the supplied ZIP and upload:
- `train.csv` to `data/raw/train.csv`
- `val.csv` to `data/raw/val.csv`
- `test.csv` to `data/raw/test.csv`

Once the licence is confirmed and files are uploaded, update this page with the source URL, version, licence, and checksum.
