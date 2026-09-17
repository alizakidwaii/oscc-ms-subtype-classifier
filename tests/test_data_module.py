"""
tests/test_data_module.py
---------------------------
Validates the merged dataset produced by data_module.py: correct shape,
no missing values, labels within the expected set, and the binary
label correctly derives from the 4-class RNA subtype.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config


@pytest.fixture(scope="module")
def dataset():
    if not os.path.exists(config.DEFAULT_DATASET_PATH):
        pytest.skip("Dataset not built yet -- run 'python main.py build-dataset' first.")
    return pd.read_csv(config.DEFAULT_DATASET_PATH)


def test_dataset_not_empty(dataset):
    assert len(dataset) > 0


def test_required_columns_present(dataset):
    required = [config.ID_COLUMN] + config.ALL_GENES + [config.LABEL_COLUMN, config.FULL_SUBTYPE_COLUMN]
    for col in required:
        assert col in dataset.columns, f"Missing column: {col}"


def test_no_missing_values(dataset):
    gene_and_label_cols = config.ALL_GENES + [config.LABEL_COLUMN]
    assert dataset[gene_and_label_cols].isnull().sum().sum() == 0


def test_binary_labels_valid(dataset):
    assert set(dataset[config.LABEL_COLUMN].unique()) <= {config.POSITIVE_CLASS, config.NEGATIVE_CLASS}


def test_binary_label_matches_full_subtype(dataset):
    """Every row labelled 'MS' must have subtype_full == 'Mesenchymal', and vice versa."""
    ms_rows = dataset[dataset[config.LABEL_COLUMN] == config.POSITIVE_CLASS]
    assert (ms_rows[config.FULL_SUBTYPE_COLUMN] == "Mesenchymal").all()

    other_rows = dataset[dataset[config.LABEL_COLUMN] == config.NEGATIVE_CLASS]
    assert (other_rows[config.FULL_SUBTYPE_COLUMN] != "Mesenchymal").all()


def test_gene_columns_numeric(dataset):
    for gene in config.ALL_GENES:
        assert pd.api.types.is_numeric_dtype(dataset[gene]), f"{gene} is not numeric"


def test_sample_ids_unique(dataset):
    assert dataset[config.ID_COLUMN].is_unique
