"""
tests/test_preprocessing_module.py
------------------------------------
Validates schema-validation logic and the stratified train/test split.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.preprocessing_module import validate_schema, split_dataset, SchemaValidationError


@pytest.fixture(scope="module")
def dataset():
    if not os.path.exists(config.DEFAULT_DATASET_PATH):
        pytest.skip("Dataset not built yet -- run 'python main.py build-dataset' first.")
    return pd.read_csv(config.DEFAULT_DATASET_PATH)


def test_valid_schema_passes(dataset):
    validate_schema(dataset)  # should not raise


def test_missing_column_raises(dataset):
    broken = dataset.drop(columns=[config.ALL_GENES[0]])
    with pytest.raises(SchemaValidationError):
        validate_schema(broken)


def test_null_values_raise(dataset):
    broken = dataset.copy()
    broken.loc[0, config.ALL_GENES[0]] = None
    with pytest.raises(SchemaValidationError):
        validate_schema(broken)


def test_unexpected_label_raises(dataset):
    broken = dataset.copy()
    broken.loc[0, config.LABEL_COLUMN] = "Unknown"
    with pytest.raises(SchemaValidationError):
        validate_schema(broken)


def test_split_is_stratified_and_disjoint(dataset):
    X_train, X_test, y_train, y_test = split_dataset(dataset)

    assert len(X_train) + len(X_test) == len(dataset)
    assert set(X_train.index).isdisjoint(set(X_test.index))

    train_ratio = (y_train == config.POSITIVE_CLASS).mean()
    test_ratio = (y_test == config.POSITIVE_CLASS).mean()
    full_ratio = (dataset[config.LABEL_COLUMN] == config.POSITIVE_CLASS).mean()

    # Stratified split should keep the MS proportion close to the full-dataset proportion
    assert abs(train_ratio - full_ratio) < 0.05
    assert abs(test_ratio - full_ratio) < 0.05
