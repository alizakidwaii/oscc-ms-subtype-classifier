 # OSCC-MS Subtype Classifier

A machine learning project that predicts whether a tumor belongs to the **Mesenchymal (MS)** molecular subtype of Oral/Head and Neck Squamous Cell Carcinoma (OSCC/HNSC) using gene expression data and a Random Forest classifier.

## About the Project

This project was developed as part of my **Fundamentals of AI/ML** course. I wanted to take the concepts we learned in the ML Basics module, such as classification, feature importance, model evaluation, and bias-variance, and apply them to a real-world problem instead of using a simple toy dataset.

I chose cancer subtype classification because it is a practical application of supervised machine learning and allows us to work with real patient gene-expression data.

## Overview

Head and neck cancer can be divided into different molecular subtypes based on gene expression patterns. The four major subtypes are **Basal, Mesenchymal, Atypical, and Classical**.

This project focuses on the **Mesenchymal (MS)** subtype, which is associated with an aggressive and invasive tumor phenotype and has an **EMT-like (epithelial-to-mesenchymal transition)** expression pattern.

The main question the model tries to answer is:

**Given a patient's gene expression profile, is the tumor Mesenchymal (MS) or Other?**

The model uses expression values from a panel of **35 genes**. For a given patient, it provides:

* Predicted subtype: **MS / Other**
* Probability for each class
* The genes that contributed most to the prediction

## Main Features

* Uses a real dataset containing **279 TCGA-HNSC patients**
* Complete pipeline from raw data processing to prediction
* Uses **5-fold stratified cross-validation** during training
* Evaluates the model using accuracy, precision, recall, F1-score, and ROC-AUC
* Generates confusion matrix, ROC curve, and feature-importance plots
* Provides gene-level information to help understand individual predictions
* Checks the input data before training or prediction
* Gives clear errors when required columns or values are missing
* Maintains logs for pipeline runs
* Includes unit tests for the five main modules

## Technologies Used

* **Python 3.10+**
* **scikit-learn** – Random Forest, cross-validation, data splitting, and evaluation metrics
* **pandas / numpy** – data processing and manipulation
* **matplotlib / seaborn** – evaluation and visualization plots
* **openpyxl** – reading the Excel subtype-label file
* **joblib** – saving and loading the trained model
* **pytest** – unit testing
* **Git / GitHub** – version control

## Project Structure

```text
oscc-ms-real/
├── main.py                     # Main command-line entry point
├── requirements.txt
├── statement.md                # Problem statement and project scope
├── src/
│   ├── config.py               # Paths, genes and model settings
│   ├── logger.py               # Logging setup
│   ├── data_module.py          # Raw data loading and merging
│   ├── preprocessing_module.py # Data validation and train/test split
│   ├── training_module.py      # Random Forest training
│   ├── evaluation_module.py    # Metrics and plots
│   └── prediction_module.py    # Prediction for a new patient
├── tests/                      # Unit tests
├── data/
│   ├── raw/                    # Original source files
│   ├── oscc_gene_expression.csv
│   └── sample_patient.csv
├── models/                     # Saved trained model
├── outputs/                    # Metrics and evaluation plots
├── logs/                       # Log files
└── docs/, reports/             # Documentation and project report
```

## Requirements

* Python **3.10 or later**
* pip

## Installation

First, clone the repository:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

Creating a virtual environment is recommended:

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

Then install the required packages:

```bash
pip install -r requirements.txt
```

No API keys or external services are required.

The processed dataset is already included in the project, so the model can be trained without downloading the original TCGA files again. The raw files only need to be downloaded if you want to rebuild the dataset yourself.

## Running the Project

The project can be run directly from the terminal using `main.py`.

### 1. Build the dataset

This step is optional because the processed dataset is already provided.

```bash
python main.py build-dataset
```

### 2. Train the model

```bash
python main.py train
```

The Random Forest is trained using **5-fold stratified cross-validation**.

### 3. Evaluate the model

```bash
python main.py evaluate
```

This produces the evaluation metrics and plots.

### 4. Predict a new patient

```bash
python main.py predict --input data/sample_patient.csv
```

## Example Prediction

A prediction looks like this:

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

This means the model classified the sample as **MS**, with a higher predicted probability for MS than Other. It also lists the genes that contributed most to that prediction.

For a new patient, the input CSV must contain the same **35 gene columns** used by the model. The column names must exactly match the gene names defined in `src/config.py`.

## Testing

The project includes unit tests for the different modules.

Run them using:

```bash
pytest tests/ -v
```

The tests check things such as:

* Missing or invalid columns
* Null values
* Incorrect subtype labels
* Correct conversion of the original four-class subtype into the MS/Other classification
* Proper stratified train/test splitting
* Model performance compared with a majority-class baseline
* Prediction format and probability values

## Dataset

The project uses data from **TCGA-HNSC**.

* **Expression data:** TCGA-HNSC IlluminaHiSeq RNASeqV2 gene-expression matrix from the UCSC Xena Browser
* Original expression dataset: **566 samples × 20,530 genes**
* **Subtype labels:** TCGA HNSC molecular subtype information from Supplementary Table S7.2 of the TCGA Nature 2015 head and neck cancer study
* **Final dataset:** 279 patients present in both datasets
* The final data was reduced to a **35-gene panel** containing EMT/mesenchymal markers, epithelial markers, and housekeeping genes
* Final labels:

  * **MS:** 75 patients
  * **Other:** 204 patients

The original data sources and instructions for rebuilding the dataset are available in:

```text
data/raw/README.md
```

## Results

The current model produced the following results:

| Metric              | Value |
| ------------------- | ----: |
| 5-fold CV ROC-AUC   | 0.957 |
| Test Accuracy       |  0.90 |
| Test Precision (MS) |  0.88 |
| Test Recall (MS)    |  0.74 |
| Test ROC-AUC        | 0.964 |

The genes that appeared among the most important predictors were **VIM, ZEB2, ZEB1, MMP2, and FAP**. These genes are associated with EMT and mesenchymal characteristics, which is consistent with the biological focus of the project.

## Output Plots

After running:

```bash
python main.py evaluate
```

the following plots can be found in the `outputs/` folder:

* `confusion_matrix.png`
* `roc_curve.png`
* `feature_importance.png`

## Project Goal

The overall goal of this project is to show how a basic machine learning approach can be applied to a real biological dataset.

Instead of only predicting **MS or Other**, the project also tries to make the prediction easier to understand by showing which genes were important for the model's decision.
