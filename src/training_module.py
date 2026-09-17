"""
training_module.py
--------------------
Functional Module 3: Model Training.

Trains a class-balanced Random Forest classifier on the training split,
reports 5-fold stratified cross-validation performance (so accuracy
isn't judged on a single lucky split), fits the final model on the full
training set, and persists it (+ metadata) to disk.
"""

import json
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src import config
from src.logger import get_logger

logger = get_logger(__name__)


def cross_validate(X_train, y_train) -> dict:
    model = RandomForestClassifier(**config.RF_PARAMS)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)

    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    logger.info(
        "%d-fold CV ROC-AUC: mean=%.4f, std=%.4f, folds=%s",
        config.CV_FOLDS, scores.mean(), scores.std(), np.round(scores, 4).tolist(),
    )
    return {"cv_roc_auc_mean": float(scores.mean()), "cv_roc_auc_std": float(scores.std()),
            "cv_roc_auc_folds": scores.tolist()}


def train_model(X_train, y_train) -> RandomForestClassifier:
    logger.info("Training RandomForestClassifier with params: %s", config.RF_PARAMS)
    model = RandomForestClassifier(**config.RF_PARAMS)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    logger.info("Training complete in %.2fs", elapsed)
    return model


def save_model(model, cv_results: dict, feature_names: list):
    joblib.dump(model, config.DEFAULT_MODEL_PATH)
    logger.info("Saved trained model to %s", config.DEFAULT_MODEL_PATH)

    metadata = {
        "model_type": "RandomForestClassifier",
        "params": config.RF_PARAMS,
        "features": feature_names,
        "positive_class": config.POSITIVE_CLASS,
        "negative_class": config.NEGATIVE_CLASS,
        "cv_results": cv_results,
        "random_seed": config.RANDOM_SEED,
    }
    with open(config.DEFAULT_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Saved model metadata to %s", config.DEFAULT_METADATA_PATH)


def run_training():
    from src.preprocessing_module import run_preprocessing
    X_train, X_test, y_train, y_test = run_preprocessing()

    cv_results = cross_validate(X_train, y_train)
    model = train_model(X_train, y_train)
    save_model(model, cv_results, list(X_train.columns))

    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    run_training()
