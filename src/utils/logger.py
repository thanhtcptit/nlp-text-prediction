import os
import sys
import logging


def get_file_logger(log_path=None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO, handlers=handlers)
    return logging.getLogger()


class Logger:
    def __init__(self, log_path=None):
        self.logger = get_file_logger(log_path)

    def log(self, msg):
        self.logger.log(logging.INFO, msg)