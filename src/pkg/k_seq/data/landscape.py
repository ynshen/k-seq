
import pandas as pd
import numpy as np


class Peaks(object):

    def __init__(self, target, center_seq, name=None, radius=None):
        self.target = target
        self.center_seq = center_seq
        self.radius = radius
        self.name = name
        if isinstance(self.target, pd.DataFrame):
            self.dist_to_center = self.target.apply(self._edit_dist_to_seq, axis=1)
        elif hasattr(self.target, 'table'):
            self.dist_to_center = self.target.table.apply(self._edit_dist_to_seq, axis=1)

    def _edit_dist_to_seq(self, seq):
        from Levenshtein import distance
        if isinstance(seq, pd.Series):
            seq = seq.name
        return distance(seq, self.center_seq)

    def peak_coverage(self, max_radius):
        """survey the coverage of peak: detected seqs vs all possible seqs
        all possible sequences in k edit distance is an hard question,
        NOTE: For now only calculate on hamming distance, with only substitution as possible edits
        """

        def seq_counter(dist):
            return np.sum(self.dist_to_center == dist)

        dist_counter = pd.Series(data =[seq_counter(dist) for dist in np.arange(max_radius + 1)],
                                 index = np.arange(max_radius + 1))

        seq_len = len(self.center_seq)

        poss_seqs = None
        for dist in np.arange(max_radius + 1):
            if dist == 0:
                poss_seqs = [1]
            else:
                poss_seqs.append(poss_seqs[-1] * 3 * (seq_len - dist + 1))

        poss_seqs = pd.Series(poss_seqs, index=np.arange(max_radius + 1))
        return dist_counter/poss_seqs

    def peak_abun(self, max_radius, rel_abun_table=None):
        """report the relative abundance of peaks in total"""

        dist_list = pd.Series(data=np.arange(max_radius + 1), index=np.arange(max_radius + 1))
        if rel_abun_table is None:
            if isinstance(self.target, pd.DataFrame):
                rel_abun_table = self.target.divide(self.target.sum(axis=0), axis=1)
            else:
                rel_abun_table = self.target.table.divide(self.target.tabl.sum(axis=0), axis=1)

        def get_rel_abun(dist):
            seqs = self.dist_to_center[self.dist_to_center <= dist].index.values
            return rel_abun_table.loc[seqs].sum(axis=0)

        peak_abun = dist_list.apply(get_rel_abun)
        return peak_abun

