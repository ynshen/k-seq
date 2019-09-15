
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

    def to_json(self):
        return self.log.to_json(orient='index')

    def to_tsv(self, file_path, sep='\t'):
        self.log.to_csv(path_or_buf=file_path, sep=sep)

    @classmethod
    def from_json(cls, path_o_str, silent=False):
        import pandas as pd
        log = pd.read_json(path_o_str, orient='index')
        cls(log=log, silent=silent)



