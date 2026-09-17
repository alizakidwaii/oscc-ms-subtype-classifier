"""
preprocessing_module.py
------------------------
Functional Module 2: Preprocessing & Validation.

Validates the merged dataset against the expected schema (all gene
columns present, no unexpected nulls, labels within the known set),
then produces a stratified train/test split. Keeping validation as its
own step -- rather than silently trusting the CSV -- is this project's
error-handling / reliability strategy: a malformed dataset fails fast
with a clear message instead of producing a silently-wrong model.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config
from src.logger import get_logger

logger = get_logger(__name__)


class SchemaValidationError(Exception):
    """Raised when the dataset does not match the expected schema."""


def load_dataset(path: str = None) -> pd.DataFrame:
    path = path or config.DEFAULT_DATASET_PATH
    logger.info("Loading dataset from %s", path)
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SchemaValidationError(
            f"Dataset not found at {path}. Run 'python main.py build-dataset' first."
        ) from exc
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise SchemaValidationError with a clear message on any mismatch."""
    required_cols = [config.ID_COLUMN] + config.ALL_GENES + [config.LABEL_COLUMN]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise SchemaValidationError(f"Dataset is missing required columns: {missing_cols}")

    null_counts = df[config.ALL_GENES + [config.LABEL_COLUMN]].isnull().sum()
    if null_counts.sum() > 0:
        raise SchemaValidationError(f"Dataset contains null values:\n{null_counts[null_counts > 0]}")

    unexpected_labels = set(df[config.LABEL_COLUMN].unique()) - {config.POSITIVE_CLASS, config.NEGATIVE_CLASS}
    if unexpected_labels:
        raise SchemaValidationError(f"Unexpected label values found: {unexpected_labels}")

    non_numeric = [g for g in config.ALL_GENES if not pd.api.types.is_numeric_dtype(df[g])]
    if non_numeric:
        raise SchemaValidationError(f"Gene columns are not numeric: {non_numeric}")

    logger.info("Schema validation passed: %d rows, %d gene features", len(df), len(config.ALL_GENES))


def summarize_class_balance(df: pd.DataFrame) -> pd.Series:
    counts = df[config.LABEL_COLUMN].value_counts()
    logger.info("Class balance:\n%s", counts.to_string())
    return counts


def split_dataset(df: pd.DataFrame):
    """Stratified split so the (imbalanced) MS/Other ratio is preserved
    in both train and test partitions."""
    X = df[config.ALL_GENES]
    y = df[config.LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_SEED,
    )

    logger.info(
        "Split dataset: %d train / %d test (test_size=%.2f, stratified on '%s')",
        len(X_train), len(X_test), config.TEST_SIZE, config.LABEL_COLUMN,
    )
    return X_train, X_test, y_train, y_test


def run_preprocessing(path: str = None):
    df = load_dataset(path)
    validate_schema(df)
    summarize_class_balance(df)
    return split_dataset(df)


if __name__ == "__main__":
    run_preprocessing()
