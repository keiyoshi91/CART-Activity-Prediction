import logging

import settings


def get_logger(file_path: str):
    logger = logging.getLogger(__name__)

    # Handler
    f_handler = logging.FileHandler(file_path)
    f_handler.setFormatter(logging.Formatter(settings.FORMATTER))
    s_handler = logging.StreamHandler()
    s_handler.setFormatter(logging.Formatter(settings.FORMATTER))
    logger.addHandler(f_handler)
    logger.addHandler(s_handler)

    return logger
