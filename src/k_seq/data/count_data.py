import pandas as pd
import numpy as np


class CountData:
    """Class to store count data info for estimators

    Attributes:
        count (pd.DataFrame): dataframe contains the count info
        ctrl_vars (pd.DataFrame): dataframe stores controlled variable, e.g. BYO concentration
        input_pools (list): a list of sample name of input pools
        truth (pd.DataFrame): optional, store the true values of parameters for sequences in count table
        note (str): optional, any note about the dataset

    """

    def __repr__(self):
        return self.note

    def __init__(self, count, ctrl_vars, input_pools, truth=None, note=None):

        self.count = count
        self.ctrl_vars = ctrl_vars
        self.input_pools = input_pools
        self.truth = truth
        self.note = note

    @classmethod
    def from_simu_path(cls, path, input_pools=None, note=None):
        """Load data from a path of simulated data"""

        count = pd.read_csv(f'{path}/Y.csv', index_col='seq')
        ctrl_vars = pd.read_csv(f'{path}/x.csv', index_col=0)
        try:
            truth = pd.read_csv(f'{path}/truth.csv', index_col=0)
        except:
            truth = None
        if input_pools is None:
            input_pools = [sample for sample in ctrl_vars.columns
                           if np.isnan(ctrl_vars.loc['c', sample]) or ctrl_vars.loc['c', sample] < 0]
        return cls(count, ctrl_vars, input_pools, truth, note)

    @classmethod
    def from_SeqTable(cls, seq_table, table_name=None, input_pools=None, note=None):
        """Load data from a SeqTable object"""
        import pickle

        if isinstance(seq_table, str):
            with open(seq_table, 'rb') as handle:
                seq_table = pickle.load(handle)

        if table_name is None:
            table_name = 'table'
        count = getattr(seq_table, table_name)
        ctrl_vars = pd.DataFrame({'c': seq_table.x_values, 'n': count[seq_table.x_values].sum(axis=0)})
        ctrl_vars.index.name = 'params'
        if input_pools is None:
            input_pools = seq_table.grouper.input.group
        try:
            truth = getattr(seq_table, 'truth')
        except:
            truth = None

        return cls(count, ctrl_vars, input_pools, truth, note)

    def select(self, n, shuffle=False):
        from copy import deepcopy

        new_data = deepcopy(self)
        idx = np.arange(self.count.shape[0])
        if shuffle:
            np.random.shuffle(idx)
        idx = idx[:n]
        new_data.count = self.count.iloc[idx, :]
        if hasattr(self, 'truth'):
            new_data.truth = self.truth.iloc[idx, :]
        new_data.note = f'{self.note} (select n={n})'

        return new_data

    @property
    def count_input(self):
        return self.count[self.input_pools]

    @property
    def ctrl_vars_input(self):
        return self.ctrl_vars[self.input_pools]

    @property
    def count_reacted(self):
        reacted_samples = [sample for sample in self.ctrl_vars.columns if sample not in self.input_pools]
        return self.count[reacted_samples]

    @property
    def ctrl_vars_reacted(self):
        reacted_samples = [sample for sample in self.ctrl_vars.columns if sample not in self.input_pools]
        return self.ctrl_vars[reacted_samples]
