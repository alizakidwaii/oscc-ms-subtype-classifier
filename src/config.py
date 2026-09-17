"""
config.py
---------
Central configuration for the OSCC-MS Subtype Classifier.
All constants (paths, gene panel, hyperparameters) live here so the
rest of the codebase stays modular and maintainable -- changing a
hyperparameter or a file path never requires touching module logic.
"""

import os

# ---------------------------------------------------------------- #
# Paths
# ---------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Raw source files (downloaded manually -- see data/raw/README.md)
RAW_EXPRESSION_PATH = os.path.join(RAW_DATA_DIR, "TCGA_HNSC_HiSeqV2.gz")
RAW_SUBTYPE_PATH = os.path.join(RAW_DATA_DIR, "TCGA_HNSC_subtypes_TableS7.2.xlsx")

# Processed dataset built by data_module.py from the two raw files above
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "oscc_gene_expression.csv")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
DEFAULT_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
DEFAULT_SAMPLE_PATIENT_PATH = os.path.join(DATA_DIR, "sample_patient.csv")
LOG_FILE = os.path.join(LOGS_DIR, "app.log")

# ---------------------------------------------------------------- #
# Gene panel (biologically-grounded marker genes; see README/report
# for literature references -- Walter et al. 2013; Mayhew et al. 2022)
# All 35 genes below are real HGNC gene symbols confirmed present in
# the TCGA-HNSC IlluminaHiSeq RNASeqV2 expression matrix (UCSC Xena).
# ---------------------------------------------------------------- #
MESENCHYMAL_UP_GENES = ["VIM", "CDH2", "SNAI1", "SNAI2", "ZEB1", "ZEB2", "FN1",
                          "TWIST1", "MMP2", "MMP9", "TGFB1", "COL1A1", "COL1A2",
                          "ACTA2", "FAP", "PDGFRB"]

EPITHELIAL_DOWN_GENES = ["CDH1", "EPCAM", "KRT5", "KRT14", "KRT8", "KRT18",
                           "DSP", "CLDN1", "CLDN4", "OCLN", "TJP1"]

HOUSEKEEPING_GENES = ["ACTB", "GAPDH", "TUBB", "RPL13A", "B2M", "HPRT1", "PPIA", "YWHAZ"]

ALL_GENES = MESENCHYMAL_UP_GENES + EPITHELIAL_DOWN_GENES + HOUSEKEEPING_GENES

LABEL_COLUMN = "subtype"            # binary label used for modelling: MS / Other
FULL_SUBTYPE_COLUMN = "subtype_full"  # original 4-class TCGA RNA subtype (reference only)
ID_COLUMN = "sample_id"
POSITIVE_CLASS = "MS"    # the class of clinical interest (Mesenchymal)
NEGATIVE_CLASS = "Other"

# Patient/tumor-sample barcode filter: TCGA sample codes ending "-01"
# denote primary solid tumor; "-11" denotes adjacent normal tissue.
TUMOR_SAMPLE_SUFFIX = "-01"

# ---------------------------------------------------------------- #
# Model hyperparameters
# ---------------------------------------------------------------- #
RANDOM_SEED = 42

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

TEST_SIZE = 0.25
CV_FOLDS = 5
TOP_N_GENES = 10


# ---------------------------------------------------------------- #
# Directory bootstrap
# ---------------------------------------------------------------- #
def ensure_directories() -> None:
    """Create the runtime output directories if they don't already exist.

    Git does not track empty directories, so on a fresh `git clone` the
    models/, outputs/ and logs/ folders are absent. Creating them on
    import means the pipeline works immediately after cloning, with no
    manual `mkdir` step for the user.
    """
    for directory in (DATA_DIR, RAW_DATA_DIR, MODELS_DIR, OUTPUTS_DIR, LOGS_DIR):
        os.makedirs(directory, exist_ok=True)


ensure_directories()
