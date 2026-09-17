"""
main.py
-------
Single command-line entry point for the OSCC-MS Subtype Classifier
pipeline. Every stage of the project -- from raw-data ingestion to a
single-patient prediction -- runs from here, satisfying the
"fully executable via command line" requirement.

Usage:
    python main.py build-dataset
    python main.py train
    python main.py evaluate
    python main.py predict --input data/sample_patient.csv
"""

import argparse
import sys

from src.logger import get_logger

logger = get_logger("main")


def cmd_build_dataset(args):
    from src.data_module import build_dataset, DataIngestionError
    try:
        build_dataset()
    except DataIngestionError as exc:
        logger.error(str(exc))
        sys.exit(1)


def cmd_train(args):
    from src.training_module import run_training
    from src.preprocessing_module import SchemaValidationError
    try:
        run_training()
    except SchemaValidationError as exc:
        logger.error(str(exc))
        sys.exit(1)


def cmd_evaluate(args):
    from src.evaluation_module import run_evaluation
    run_evaluation()


def cmd_predict(args):
    from src.prediction_module import predict_patients, PredictionInputError
    import json
    try:
        results = predict_patients(args.input)
    except PredictionInputError as exc:
        logger.error(str(exc))
        sys.exit(1)
    print(json.dumps(results, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved predictions to %s", args.output)


def build_parser():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="OSCC-MS Subtype Classifier -- gene-expression based ML classification "
                     "of Mesenchymal vs Other oral squamous cell carcinoma subtypes.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-dataset", help="Merge raw TCGA files into the modelling dataset.")
    sub.add_parser("train", help="Train the Random Forest classifier with cross-validation.")
    sub.add_parser("evaluate", help="Evaluate the trained model on the held-out test set.")

    predict_parser = sub.add_parser("predict", help="Predict subtype for new patient(s).")
    predict_parser.add_argument("--input", required=True, help="CSV file with patient gene-expression values.")
    predict_parser.add_argument("--output", default=None, help="Optional path to save predictions as JSON.")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "build-dataset": cmd_build_dataset,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "predict": cmd_predict,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
