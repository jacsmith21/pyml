import logging
import time

import tensorflow as tf


def get_logger(log_name, logging_level=logging.DEBUG):
    logger = logging.getLogger(log_name)
    formatter = logging.Formatter('[%(levelname)s/%(asctime)s/%(module)s:%(lineno)s] - %(message)s')
    formatter.converter = time.gmtime

    logger.setLevel(logging_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def make_summaries(*scalars):
    for key in scalars:
        tf.summary.scalar(key, scalars[key])


