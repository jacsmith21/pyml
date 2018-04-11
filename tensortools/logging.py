import logging
import time


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s/%(asctime)s/%(module)s:%(lineno)s] - %(message)s')
    formatter.converter = time.gmtime
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
