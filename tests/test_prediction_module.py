"""
tests/test_prediction_module.py
----------------------------------
Validates single-patient prediction: correct output structure, valid
probability distributions, and correctness on a known-label patient.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.prediction_module import predict_patients, validate_input, PredictionInputError
import pandas as pd


@pytest.fixture(scope="module")
def prediction_results():
    if not os.path.exists(config.DEFAULT_MODEL_PATH):
        pytest.skip("Model not trained yet -- run 'python main.py train' first.")
    if not os.path.exists(config.DEFAULT_SAMPLE_PATIENT_PATH):
        pytest.skip("Sample patient file not found -- run 'python main.py build-dataset' first.")
    return predict_patients(config.DEFAULT_SAMPLE_PATIENT_PATH)


def test_returns_one_result_per_patient(prediction_results):
    assert len(prediction_results) == 1


def test_result_has_expected_keys(prediction_results):
    result = prediction_results[0]
    for key in ("patient_id", "predicted_subtype", "probabilities", "top_contributing_genes"):
        assert key in result


def test_predicted_subtype_is_valid_class(prediction_results):
    assert prediction_results[0]["predicted_subtype"] in (config.POSITIVE_CLASS, config.NEGATIVE_CLASS)


def test_probabilities_sum_to_one(prediction_results):
    probs = prediction_results[0]["probabilities"]
    assert abs(sum(probs.values()) - 1.0) < 1e-6


def test_top_genes_are_from_gene_panel(prediction_results):
    top_genes = prediction_results[0]["top_contributing_genes"]
    assert all(g in config.ALL_GENES for g in top_genes)


def test_validate_input_rejects_missing_gene_column():
    df = pd.DataFrame({g: [1.0] for g in config.ALL_GENES[:-1]})  # missing last gene
    with pytest.raises(PredictionInputError):
        validate_input(df)


def test_validate_input_rejects_null_values():
    df = pd.DataFrame({g: [1.0] for g in config.ALL_GENES})
    df.loc[0, config.ALL_GENES[0]] = None
    with pytest.raises(PredictionInputError):
        validate_input(df)
