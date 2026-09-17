# Problem Statement

## Problem Statement

Oral/Head and Neck Squamous Cell Carcinoma (OSCC/HNSC) is not molecularly uniform — TCGA's genomic analysis identified four distinct subtypes based on RNA expression patterns: Basal, Mesenchymal, Atypical, and Classical. Among these, the Mesenchymal (MS) subtype is associated with a more invasive, EMT-driven phenotype and is generally linked to worse outcomes in the literature. Identifying a patient's molecular subtype from gene expression data is something that, done manually, requires specialist knowledge of dozens of marker genes and how their expression patterns interact. This project asks a narrower, answerable version of that problem: given a patient's expression values for a defined panel of literature-backed marker genes, can a machine learning model reliably classify the tumor as Mesenchymal versus Other, and can it do so in a way that's explainable rather than a black box?

## Scope of the Project

- Binary classification only: Mesenchymal (MS) vs Other (Basal + Atypical + Classical combined). The project does not attempt full 4-class subtyping.
- Restricted to a curated 35-gene panel grounded in existing literature (EMT/mesenchymal markers, epithelial markers, housekeeping controls) rather than the full ~20,000-gene expression matrix, to keep the model interpretable and avoid the curse of dimensionality on a relatively small (279-patient) real-world cohort.
- Uses real, publicly available TCGA-HNSC data (UCSC Xena expression matrix + the original TCGA Nature 2015 paper's subtype labels), not synthetic or simulated data.
- Delivered as a command-line pipeline (data ingestion → preprocessing → training → evaluation → prediction), not a web app or GUI — this was a deliberate choice to keep the project fully scriptable and reproducible from the terminal, per the course's executability requirement.
- Does not cover model deployment, a clinical decision-support interface, or integration with real hospital systems — this is an academic exercise in applying supervised learning to a real biomedical dataset, not a production diagnostic tool.

## Target Users

- **Primary (for this course submission):** the course evaluator/grader, assessing whether the project correctly applies supervised ML concepts (classification, cross-validation, feature importance, evaluation metrics) to a real dataset.
- **Illustrative real-world users (the scenario the project is modelled on):** bioinformatics researchers or lab technicians who have gene expression data for a patient sample and want a fast, explainable first-pass classification of molecular subtype, to help prioritize samples for further clinical or research follow-up. This project is a proof of concept for that kind of tool — not a validated clinical instrument.

## High-Level Features

- Ingests and merges two real public data sources (TCGA gene expression + TCGA subtype labels) into one clean, validated dataset
- Trains a Random Forest classifier with stratified 5-fold cross-validation
- Evaluates the model on a held-out test set with standard classification metrics (accuracy, precision, recall, F1, ROC-AUC) plus visual outputs (confusion matrix, ROC curve, feature importance)
- Predicts the subtype of a new patient from a CSV of gene expression values, returning the predicted class, class probabilities, and the top genes driving that individual prediction
- Validates all input data (training dataset and new patient CSVs) against an expected schema, so malformed input is rejected with a clear error rather than silently producing a wrong prediction
- Logs every pipeline run to a log file for traceability
