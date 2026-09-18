# Problem Statement

## Problem Statement

Oral/Head and Neck Squamous Cell Carcinoma (OSCC/HNSC) is not the same at the molecular level in every patient. TCGA research identified four main subtypes based on their RNA expression patterns: **Basal, Mesenchymal, Atypical, and Classical**. Among these, the **Mesenchymal (MS) subtype** is commonly associated with a more invasive and EMT-related nature and has been linked with poorer outcomes in previous studies.

Identifying the molecular subtype using gene expression data can be difficult because it requires knowledge of many marker genes and how their expression patterns relate to each other. Therefore, this project focuses on a simpler and more specific problem: **Can a machine learning model use the expression values of selected marker genes to classify a tumor as Mesenchymal or Other?** The project also aims to understand which genes contribute most to the model's classification, making the prediction easier to interpret.

## Scope of the Project

* The project performs **binary classification**: Mesenchymal (MS) vs Other, where Other includes Basal, Atypical, and Classical subtypes. It does not perform complete four-class classification.
* A selected **35-gene panel** based on existing literature is used instead of the complete ~20,000-gene expression dataset. This keeps the model simpler and more interpretable for the available **279-patient dataset**.
* The project uses **real, publicly available TCGA-HNSC data**, including the UCSC Xena gene expression data and subtype labels from the original TCGA study.
* The project is implemented as a **command-line pipeline** covering data input, preprocessing, training, evaluation, and prediction. This keeps the system reproducible and suitable for the course requirement of having an executable project.
* This is an **academic proof-of-concept** and does not include clinical deployment, hospital-system integration, or a clinical decision-support interface. It is not intended to be used as a diagnostic tool.

## Target Users

* **Primary users:** The course evaluator or grader, who will assess how well the project applies supervised machine learning concepts such as classification, cross-validation, feature importance, and evaluation metrics.
* **Possible real-world users:** Bioinformatics researchers or laboratory technicians who have gene expression data and want a quick, explainable first-level prediction of the molecular subtype. The results could help in deciding which samples may need further research or clinical investigation.

## High-Level Features

* Combines **TCGA gene expression data and subtype labels** into a clean and validated dataset.
* Trains a **Random Forest classifier** using stratified 5-fold cross-validation.
* Evaluates the model using **accuracy, precision, recall, F1-score, and ROC-AUC**, along with a confusion matrix, ROC curve, and feature-importance visualization.
* Accepts a new patient's gene expression data in CSV format and predicts whether the sample belongs to the **Mesenchymal or Other** category.
* Provides **class probabilities and important genes** that contributed to the prediction.
* Checks the input files against the expected format and gives an error if the data is incorrect.
* Maintains a **log file** for tracking each pipeline run.
