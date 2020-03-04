"""Utility function for control logging
"""

from time import time
import logging as lg
import pandas as pd


class UnknownError(Exception):
    """Undefined error type

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class logging:
    """Collection of functions over `logging` module """

    default_formatter = lg.Formatter(fmt='%(asctime)s %(message)s',
                                     datefmt='%m/%d/%Y %I:%M:%S %p')

    @staticmethod
    def info(msg, silent=False):
        """Wrapper over `logging.info`"""
        if not silent:
            lg.info(msg)

    @staticmethod
    def warning(msg):
        """Wrapper over `logging.warning`"""
        lg.warning(msg)

    @staticmethod
    def error(msg, error_type=UnknownError, ignore_error=False):
        """Wrapper over `logging.error` and raise an Error"""
        lg.error(msg)
        if not ignore_error:
            raise error_type(msg)

    @staticmethod
    def debug(msg):
        """Wrapper over `logging.debug`"""
        lg.debug(msg)

    @staticmethod
    def check_handler_name(name, logger=None):
        """Check if handler with name is attached to the logger (default root logger)"""
        if logger is None:
            logger = lg.getLogger()
        return any([handler.get_name() == name for handler in logger.handlers])

    @staticmethod
    def add_console_handler(logger=None, name='console', formatter=default_formatter):
        """Add console handler (write to stdout/stderr) to the logger (default root logger)"""
        import sys

        if logger is None:
            logger = lg.getLogger()
        if not logging.check_handler_name(name, logger=logger):
            console_handler = lg.StreamHandler(sys.stdout)
            console_handler.set_name(name)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    @staticmethod
    def add_file_handler(file_path, logger=None, name='logfile', formatter=default_formatter):
        """Add file handler (write to log file) to the logger (default root logger)"""

        if logger is None:
            logger = lg.getLogger()
        if not logging.check_handler_name(name, logger=logger):
            file_handler = lg.FileHandler(file_path)
            file_handler.set_name(name)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)


class Logger:
    """Logger associate with some object to record processing steps"""

    def __init__(self, silent=False):
        self.log = pd.DataFrame(columns=['time', 'message']).set_index('time')
        self.silent = silent

    def info(self, msg, silent=False):
        from datetime import datetime
        self.log.loc[datetime.now()] = msg
        if not (self.silent and silent):
            logging.info(msg)


class Timer:
    """Time the process within the scope"""

    def __init__(self, message=None):
        self.message = message if message else 'Time cost: {elapsed_time:.2f} {unit}.'

    def __enter__(self):
        self.start = time()
        return None

    def __exit__(self, type, value, traceback):
        elapsed_time = time() - self.start
        if elapsed_time < 60:
            unit = 'seconds'
        elif elapsed_time < 3600:
            unit = 'minutes'
            elapsed_time /= 60.0
        else:
            unit = 'hours'
            elapsed_time /= 3600.0
        logging.info('-' * 50)
        logging.info(self.message.format(elapsed_time=elapsed_time, unit=unit))

