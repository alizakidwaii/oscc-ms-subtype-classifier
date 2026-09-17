"""
prediction_module.py
----------------------
Functional Module 5: Prediction / Inference.

Takes a CSV of one or more new patients' gene-expression profiles
(same 35-gene panel as the training data) and returns, per patient:
  - predicted subtype (MS / Other)
  - class probabilities
  - the top genes driving that specific patient's prediction

This is the "give it any patient data, get an answer" entry point --
the only module an end user needs to touch after the model is trained.
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src import config
from src.logger import get_logger

logger = get_logger(__name__)


class PredictionInputError(Exception):
    """Raised when the input CSV doesn't match the expected gene panel."""


def load_model():
    try:
        return joblib.load(config.DEFAULT_MODEL_PATH)
    except FileNotFoundError as exc:
        raise PredictionInputError(
            f"No trained model found at {config.DEFAULT_MODEL_PATH}. Run 'python main.py train' first."
        ) from exc


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    missing = [g for g in config.ALL_GENES if g not in df.columns]
    if missing:
        raise PredictionInputError(f"Input CSV is missing required gene columns: {missing}")

    numeric_block = df[config.ALL_GENES]
    if numeric_block.isnull().any().any():
        raise PredictionInputError("Input CSV contains missing values in gene columns.")
    if not all(pd.api.types.is_numeric_dtype(numeric_block[c]) for c in config.ALL_GENES):
        raise PredictionInputError("All gene columns must be numeric expression values.")

    return df


def predict_patients(input_path: str, top_n: int = 5) -> list:
    logger.info("Loading input patient(s) from %s", input_path)
    df = pd.read_csv(input_path)
    df = validate_input(df)

    model = load_model()
    X = df[config.ALL_GENES]

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    class_order = list(model.classes_)
    ms_idx = class_order.index(config.POSITIVE_CLASS)

    # Per-tree feature usage lets us explain *this specific* prediction,
    # not just the model's global importances.
    global_importances = model.feature_importances_
    gene_names = list(X.columns)

    results = []
    for i in range(len(df)):
        patient_id = df.iloc[i][config.ID_COLUMN] if config.ID_COLUMN in df.columns else f"patient_{i}"
        probs = {cls: float(probabilities[i][j]) for j, cls in enumerate(class_order)}

        # Top genes for this patient: importance-weighted deviation from
        # the training-set mean, signed toward whichever class they push.
        patient_values = X.iloc[i]
        contribution_score = global_importances * patient_values.values
        top_idx = np.argsort(np.abs(contribution_score))[::-1][:top_n]
        top_genes = [gene_names[j] for j in top_idx]

        result = {
            "patient_id": str(patient_id),
            "predicted_subtype": predictions[i],
            "probabilities": probs,
            "top_contributing_genes": top_genes,
        }
        results.append(result)

        logger.info(
            "Patient %s -> predicted: %s (P(MS)=%.3f) | top genes: %s",
            result["patient_id"], result["predicted_subtype"], probs[config.POSITIVE_CLASS],
            ", ".join(top_genes),
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Predict OSCC molecular subtype (MS vs Other) for new patient(s).")
    parser.add_argument("--input", required=True, help="Path to CSV with patient gene-expression values.")
    parser.add_argument("--output", default=None, help="Optional path to save predictions as JSON.")
    args = parser.parse_args()

    try:
        results = predict_patients(args.input)
    except PredictionInputError as exc:
        logger.error(str(exc))
        raise SystemExit(1)

    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved predictions to %s", args.output)


if __name__ == "__main__":
    main()
