import sys
from time import time
import logging

class Logger:
    """A simple logger to log data"""

    def __init__(self, log=None, silent=False):
        import pandas as pd
        if log is not None:
            self.log = log
        else:
            self.log = pd.DataFrame(columns=['Timestamp', 'Message'])
            self.log.set_index(['Timestamp'], inplace=True)
        self._silent = silent

    def __repr__(self):
        return self.log

    def add(self, message, show=False):
        from datetime import datetime
        self.log.loc[datetime.now()] = [message]
        if (not self._silent) or show:
            print(message)

    def merge_from(self, logger_list):
        if isinstance(logger_list, Logger):
            logger_list = [logger_list]
        for logger in logger_list:
            self.log = self.log.append(logger.log)
        self.log.sort_index(inplace=True)

    def to_json(self):
        return self.log.to_json(orient='index')

    def to_tsv(self, file_path, sep='\t'):
        self.log.to_csv(path_or_buf=file_path, sep=sep)

    @classmethod
    def from_json(cls, path_o_str, silent=False):
        import pandas as pd
        log = pd.read_json(path_o_str, orient='index')
        cls(log=log, silent=silent)


class Timer:
    def __init__(self, message=None, save_to=None):
        if message:
            self.message = message
        else:
            self.message = 'It took {elapsed_time:.2f} {unit}.'
        self.save_to = save_to

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
        if self.save_to is not None:
            with open(self.save_to, 'w') as f:
                f.write(self.message.format(elapsed_time=elapsed_time, unit=unit))


class FileLogger(object):
    """Log standard output to a file"""

    def __init__(self, file_path):
        if not '.log' in file_path:
            self.stdout = file_path + '.log'
        else:
            self.stdout = file_path

    def __enter__(self):
        self.sys_stdout = sys.stdout
        self.sys_stderr = sys.stderr

        sys.stdout = open(self.stdout, 'w')
        sys.stderr = sys.stdout

    def __exit__(self, type, value, traceback):
        sys.stdout.close()
        sys.stdout = self.sys_stdout
        sys.stderr = self.sys_stderr


