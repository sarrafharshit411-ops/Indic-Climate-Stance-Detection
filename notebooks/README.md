# Word + Character TF-IDF Class-Weighted SVM

This notebook trains separate Hindi and Bengali stance detection models.

- Word TF-IDF: unigrams and bigrams
- Character TF-IDF: character n-grams (2–5)
- Combined features: word + character sparse features
- Classifier: LinearSVC with `class_weight="balanced"`
- Labels: 0 = Against, 1 = Favour, 2 = None

## Dataset
Place the following CSVs in `/content/data/`:
- hindi_train.csv, hindi_val.csv, hindi_test.csv
- bengali_train.csv, bengali_val.csv, bengali_test.csv

The notebook exports metrics, classification reports, confusion matrices, and predictions to `/content/results/Word_Character_TFIDF_Class_Weighted_SVM/`.
