"""
logger.py
---------
Shared logger configuration. Every module logs to both the console and
a rotating-free log file (logs/app.log), which is how this project
satisfies the "Logging / Monitoring" non-functional requirement --
every pipeline run leaves an auditable trail of what happened.
"""

import logging
from src import config


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (avoids duplicate handlers on repeated imports)
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(config.LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
