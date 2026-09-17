"""
data_module.py
---------------
Functional Module 1: Data Ingestion.

Loads the two real, publicly-available source files:
  1. TCGA-HNSC gene expression matrix (UCSC Xena, IlluminaHiSeq RNASeqV2,
     n=566 samples x 20,530 genes, log2(RSEM+1) normalized)
  2. TCGA-HNSC RNA molecular subtype labels (Supplementary Table S7.2,
     Walter et al./TCGA Nature 2015 paper: Basal / Mesenchymal /
     Atypical / Classical, per patient barcode)

...and merges them into a single, clean modelling-ready CSV:
data/oscc_gene_expression.csv, with one row per patient, the 35-gene
literature panel as features, plus the 4-class and binary (MS/Other)
subtype labels.

This module is intentionally decoupled from training/evaluation/
prediction: swapping in a different cohort or a larger gene panel only
requires changes here (Maintainability & Scalability).
"""

import gzip
import os
import sys

import pandas as pd
import openpyxl

from src import config
from src.logger import get_logger

logger = get_logger(__name__)


class DataIngestionError(Exception):
    """Raised when the raw source files are missing or malformed."""


def _check_raw_files_exist() -> None:
    missing = [p for p in (config.RAW_EXPRESSION_PATH, config.RAW_SUBTYPE_PATH)
               if not os.path.exists(p)]
    if missing:
        raise DataIngestionError(
            "Missing raw source file(s): " + ", ".join(missing) +
            "\nSee data/raw/README.md for download instructions."
        )


def _load_subtype_labels() -> dict:
    """Parse Supplementary Table S7.2 -> {patient_barcode: RNA subtype}."""
    logger.info("Loading subtype labels from %s", config.RAW_SUBTYPE_PATH)
    wb = openpyxl.load_workbook(config.RAW_SUBTYPE_PATH, data_only=True)
    ws = wb["Platform_Class_Labels"]

    subtype_map = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        barcode, rna_subtype = row[0], row[1]
        if barcode is None or rna_subtype is None:
            continue
        patient_barcode = barcode.replace(".", "-")  # TCGA.BA.4074 -> TCGA-BA-4074
        subtype_map[patient_barcode] = rna_subtype

    logger.info("Loaded %d labelled patients", len(subtype_map))
    return subtype_map


def _load_expression_matrix() -> pd.DataFrame:
    """Load the gene x sample matrix, subset to the gene panel, transpose
    to sample x gene (the orientation the rest of the pipeline expects)."""
    logger.info("Loading gene expression matrix from %s", config.RAW_EXPRESSION_PATH)

    opener = gzip.open if config.RAW_EXPRESSION_PATH.endswith(".gz") else open
    with opener(config.RAW_EXPRESSION_PATH, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)

    logger.info("Full matrix shape: %d genes x %d samples", df.shape[0], df.shape[1])

    missing_genes = [g for g in config.ALL_GENES if g not in df.index]
    if missing_genes:
        raise DataIngestionError(f"Gene panel genes missing from expression matrix: {missing_genes}")

    panel = df.loc[config.ALL_GENES]
    panel_t = panel.transpose()  # samples as rows, genes as columns
    panel_t.index.name = "sample_barcode"
    return panel_t


def _patient_barcode(sample_barcode: str) -> str:
    """TCGA-CR-7383-01 -> TCGA-CR-7383 (drop the sample/vial suffix)."""
    return "-".join(sample_barcode.split("-")[:3])


def build_dataset(save: bool = True) -> pd.DataFrame:
    """Merge expression + subtype files into the final modelling dataset."""
    _check_raw_files_exist()

    subtype_map = _load_subtype_labels()
    expression = _load_expression_matrix()

    # Keep only primary tumor samples (suffix "-01"), matched to a labelled patient
    tumor_samples = [s for s in expression.index if s.endswith(config.TUMOR_SAMPLE_SUFFIX)]
    logger.info("Tumor samples in expression matrix: %d", len(tumor_samples))

    rows = []
    for sample_barcode in tumor_samples:
        patient_barcode = _patient_barcode(sample_barcode)
        if patient_barcode not in subtype_map:
            continue
        row = expression.loc[sample_barcode].to_dict()
        row[config.ID_COLUMN] = sample_barcode
        row[config.FULL_SUBTYPE_COLUMN] = subtype_map[patient_barcode]
        row[config.LABEL_COLUMN] = (
            config.POSITIVE_CLASS if subtype_map[patient_barcode] == "Mesenchymal"
            else config.NEGATIVE_CLASS
        )
        rows.append(row)

    if not rows:
        raise DataIngestionError("No overlapping patients found between the two source files.")

    dataset = pd.DataFrame(rows)
    ordered_cols = [config.ID_COLUMN] + config.ALL_GENES + [config.FULL_SUBTYPE_COLUMN, config.LABEL_COLUMN]
    dataset = dataset[ordered_cols]

    logger.info("Final dataset: %d patients x %d gene features", dataset.shape[0], len(config.ALL_GENES))
    logger.info("Binary label counts:\n%s", dataset[config.LABEL_COLUMN].value_counts().to_string())
    logger.info("Full subtype counts:\n%s", dataset[config.FULL_SUBTYPE_COLUMN].value_counts().to_string())

    if save:
        dataset.to_csv(config.DEFAULT_DATASET_PATH, index=False)
        logger.info("Saved dataset to %s", config.DEFAULT_DATASET_PATH)

        # Also emit one held-out patient as a ready-to-use prediction example
        sample_row = dataset.drop(columns=[config.FULL_SUBTYPE_COLUMN, config.LABEL_COLUMN]).iloc[[0]]
        sample_row.to_csv(config.DEFAULT_SAMPLE_PATIENT_PATH, index=False)
        logger.info("Saved example patient for prediction to %s", config.DEFAULT_SAMPLE_PATIENT_PATH)

    return dataset


if __name__ == "__main__":
    try:
        build_dataset()
    except DataIngestionError as exc:
        logger.error(str(exc))
        sys.exit(1)
