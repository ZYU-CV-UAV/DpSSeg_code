import logging
import os


def setup_logger(log_file):

    logger = logging.getLogger("DpSSeg")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)

    # console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger