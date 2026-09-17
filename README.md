# OSCC-MS Subtype Classifier

A machine learning project that predicts the **Mesenchymal (MS)** molecular subtype of Oral/Head and Neck Squamous Cell Carcinoma (OSCC/HNSC) from a patient's gene expression profile, using a Random Forest classifier.

This was built for my Fundamentals of AI/ML course project. The idea was simple: pick something from the ML Basics module (classification, feature importance, bias-variance, model evaluation) and apply it to a real problem instead of a toy dataset. I went with cancer subtype classification because it's a genuinely useful application of supervised learning and it let me work with real patient data instead of something synthetic.

## Overview

Head and neck cancer isn't one disease — researchers have identified four molecular subtypes based on gene expression patterns: Basal, Mesenchymal, Atypical, and Classical. The Mesenchymal subtype is of particular clinical interest because it's associated with a more aggressive, invasive tumor phenotype (driven by an EMT-like — epithelial to mesenchymal transition — expression signature).

This project trains a Random Forest to answer one question from a patient's gene expression data: **is this tumor Mesenchymal (MS), or something else (Other)?**

Given a CSV of expression values for a 35-gene panel, the trained model returns:
- the predicted subtype (MS / Other)
- a confidence score for each class
- the specific genes that most influenced that particular prediction

## Features

- Real dataset — 279 actual TCGA-HNSC patients, not synthetic data
- End-to-end CLI pipeline: raw data → clean dataset → trained model → evaluation → prediction, all runnable with one command each
- 5-fold stratified cross-validation during training, not just a single train/test split
- Evaluation outputs: accuracy, precision, recall, F1, ROC-AUC, confusion matrix plot, ROC curve plot, feature importance plot
- Per-patient explainability — every prediction comes with the top genes that drove it, not just a bare label
- Schema validation on both the training dataset and any new patient CSV, so bad input fails with a clear error instead of a silent wrong prediction
- Logging to both console and a log file for every pipeline run
- Unit tests covering all five modules

## Technologies / Tools Used

- **Python 3.10+**
- **scikit-learn** — Random Forest classifier, cross-validation, train/test split, evaluation metrics
- **pandas / numpy** — data loading, merging, and manipulation
- **matplotlib / seaborn** — confusion matrix, ROC curve, and feature importance plots
- **openpyxl** — reading the raw Excel-format subtype label file
- **joblib** — model persistence
- **pytest** — unit testing
- **Git / GitHub** — version control

## Project Structure

```
oscc-ms-real/
├── main.py                     # single CLI entry point
├── requirements.txt
├── statement.md                 # problem statement, scope, target users
├── src/
│   ├── config.py                # paths, gene panel, hyperparameters
│   ├── logger.py                # shared logging setup
│   ├── data_module.py           # Module 1: raw data ingestion & merge
│   ├── preprocessing_module.py  # Module 2: validation & train/test split
│   ├── training_module.py       # Module 3: Random Forest + CV training
│   ├── evaluation_module.py     # Module 4: metrics & plots
│   └── prediction_module.py     # Module 5: single-patient prediction
├── tests/                       # pytest unit tests for every module
├── data/
│   ├── raw/                     # raw source files (see raw/README.md)
│   ├── oscc_gene_expression.csv # processed real dataset (committed)
│   └── sample_patient.csv       # example patient for prediction demo
├── models/                      # trained model + metadata (generated)
├── outputs/                     # metrics.json + evaluation plots (generated)
├── logs/                        # app.log (generated)
└── docs/, reports/               # design documentation and the project report
```

## Requirements

- Python 3.10 or later
- pip

## Setup / Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# 2. Create a virtual environment (recommended, keeps things isolated)
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

No API keys or external services needed. The processed dataset (`data/oscc_gene_expression.csv`) is already committed to the repo, so training and prediction work right after cloning — you don't have to re-download the raw TCGA files unless you specifically want to rebuild the dataset from scratch.

## Running the Project

Everything is driven through `main.py` from the terminal — no GUI required.

```bash
# (Optional) rebuild the dataset from the raw TCGA source files
# Not required to run the project — see data/raw/README.md if you want to do this
python main.py build-dataset

# Train the Random Forest with 5-fold cross-validation
python main.py train

# Evaluate the trained model on the held-out test set
python main.py evaluate

# Predict the subtype for a new patient
python main.py predict --input data/sample_patient.csv
```

### Example prediction output

```json
[
  {
    "patient_id": "TCGA-CR-7383-01",
    "predicted_subtype": "MS",
    "probabilities": { "MS": 0.913, "Other": 0.087 },
    "top_contributing_genes": ["VIM", "MMP2", "COL1A2", "ZEB2", "ZEB1"]
  }
]
```

To run a prediction on your own patient data, put it in a CSV with the same 35 gene columns as `data/sample_patient.csv` — the column names need to match `src/config.py::ALL_GENES` exactly, or the pipeline will reject it with a validation error telling you what's missing.

## Testing

```bash
pytest tests/ -v
```

The test suite covers:
- schema validation (missing columns, null values, invalid labels all get caught)
- correctness of the binary label derivation from the original 4-class subtype
- the train/test split staying stratified and non-overlapping
- the trained model beating a majority-class baseline and scoring above-chance ROC-AUC
- prediction output structure and probability sanity checks

## Dataset

- **Expression data**: TCGA-HNSC IlluminaHiSeq RNASeqV2 gene expression matrix, downloaded from the UCSC Xena Browser (566 samples x 20,530 genes)
- **Subtype labels**: TCGA HNSC molecular subtype calls from Supplementary Table S7.2 of the original TCGA Nature 2015 paper on head and neck cancer
- **Final merged dataset**: 279 patients where both files overlap, subset to a 35-gene literature panel (EMT/mesenchymal markers like VIM, ZEB1/2, SNAI1/2; epithelial markers like CDH1, EPCAM, keratins; housekeeping genes as controls). Binary label: MS = 75 patients, Other = 204 patients.
- Full source links and re-download steps: `data/raw/README.md`

## Results

| Metric | Value |
|---|---|
| 5-fold CV ROC-AUC | 0.957 |
| Test accuracy | 0.90 |
| Test precision (MS) | 0.88 |
| Test recall (MS) | 0.74 |
| Test ROC-AUC | 0.964 |

The top predictive genes turned out to be VIM, ZEB2, ZEB1, MMP2, and FAP — which lines up with what's expected biologically, since these are all well-known EMT/mesenchymal markers. That was a good sign the model was picking up real signal and not just memorizing noise.

## Screenshots

See `outputs/confusion_matrix.png`, `outputs/roc_curve.png`, and `outputs/feature_importance.png` after running `python main.py evaluate`.
