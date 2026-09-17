"""
tests/test_training_and_evaluation.py
----------------------------------------
Trains a small model on the real dataset and checks that the pipeline
produces a valid, reasonably-performing classifier and correctly
structured evaluation outputs.
"""

import os
import sys

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.preprocessing_module import split_dataset
from src.evaluation_module import compute_metrics


@pytest.fixture(scope="module")
def dataset():
    if not os.path.exists(config.DEFAULT_DATASET_PATH):
        pytest.skip("Dataset not built yet -- run 'python main.py build-dataset' first.")
    return pd.read_csv(config.DEFAULT_DATASET_PATH)


@pytest.fixture(scope="module")
def trained_model(dataset):
    X_train, X_test, y_train, y_test = split_dataset(dataset)
    model = RandomForestClassifier(**config.RF_PARAMS)
    model.fit(X_train, y_train)
    return model, X_test, y_test


def test_model_predicts_known_classes(trained_model):
    model, X_test, y_test = trained_model
    preds = model.predict(X_test)
    assert set(preds) <= {config.POSITIVE_CLASS, config.NEGATIVE_CLASS}


def test_model_beats_majority_baseline(trained_model, dataset):
    """A useful classifier should outperform always predicting the majority
    class ('Other'), which would score ~73% accuracy on this dataset."""
    model, X_test, y_test = trained_model
    metrics, _, _ = compute_metrics(model, X_test, y_test)

    majority_baseline = (dataset[config.LABEL_COLUMN] == config.NEGATIVE_CLASS).mean()
    assert metrics["accuracy"] > majority_baseline


def test_roc_auc_above_chance(trained_model):
    model, X_test, y_test = trained_model
    metrics, _, _ = compute_metrics(model, X_test, y_test)
    assert metrics["roc_auc"] > 0.5


def test_metrics_within_valid_range(trained_model):
    model, X_test, y_test = trained_model
    metrics, _, _ = compute_metrics(model, X_test, y_test)
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        assert 0.0 <= metrics[key] <= 1.0
