# Indic Climate Classification — E1 and E4

This repository contains two independent, Google Colab-ready experiments:

- **E1:** TF-IDF + Logistic Regression (traditional baseline)
- **E4:** Word + Character TF-IDF + class-weighted Linear SVM

Both notebooks contain the cleaned dataset archive internally, so they do not ask
the user to upload a dataset or manually provide a path. They train separate models
for Hindi and Bengali using the supplied train/validation/test splits.

## Run
1. Upload this repository/folders to GitHub.
2. Open either `.ipynb` file in Google Colab.
3. Select **Runtime → Run all**.
4. The last cell downloads a ZIP with outputs and code.

Dataset labels: 0 = Against, 1 = Favour, 2 = None.

The source dataset README notes that cleaning was structural/text-format cleaning
only and does not certify translation quality. Translation review should be
considered before interpreting results.
