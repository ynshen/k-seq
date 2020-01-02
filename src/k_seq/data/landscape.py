"""A simple landscape module to define sequence peaks on edit distance"""
import pandas as pd
import numpy as np


class Peaks(object):
    """A sequence peak defined on edit distance"""

    def __init__(self, target, center_seq, name=None, radius=None):
        self.target = target
        self.center_seq = center_seq
        self.radius = radius
        self.name = name
        if isinstance(self.target, pd.DataFrame):
            # index contains sequence
            self.dist_to_center = self.target.index.to_series().map(self._edit_dist_to_seq)
            self.rel_abun_table = self.target.divide(self.target.sum(axis=0), axis=1)
        elif hasattr(self.target, 'table'):
            # if have 'table' dataframe attributes
            self.dist_to_center = self.target.table.index.to_series().map(self._edit_dist_to_seq)
            self.rel_abun_table = self.target.table.divide(self.target.table.sum(axis=0), axis=1)

    def _edit_dist_to_seq(self, seq):
        from Levenshtein import distance
        if isinstance(seq, pd.Series):
            seq = seq.name
        return distance(seq, self.center_seq)

    def peak_coverage(self, max_radius):
        """survey the coverage of peak: detected seqs vs all possible seqs
        NOTICE:
            all possible sequences in k edit distance is an hard question,
        NOTE: For now only calculate on hamming distance, with only substitution as possible edits
        """

        def seq_counter(dist):
            return np.sum(self.dist_to_center == dist)

        dist_counter = pd.Series(data=[seq_counter(dist) for dist in np.arange(max_radius + 1)],
                                 index=np.arange(max_radius + 1))
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
        """report the relative abundance and number of unique sequences of peaks in total reads
        Returns:
            pd.DataFrame contains rel_abun
            pd.DataFrame contains uniq_seq
        """

        dist_list = pd.Series(data=np.arange(max_radius + 1), index=np.arange(max_radius + 1))
        if rel_abun_table is None:
            rel_abun_table = self.rel_abun_table

        def get_rel_abun(dist):
            seqs = self.dist_to_center[self.dist_to_center <= dist].index.values
            return rel_abun_table.loc[seqs].sum(axis=0)

        def get_uniq_seq(dist):
            seqs = self.dist_to_center[self.dist_to_center <= dist].index.values
            return len(seqs)

        peak_abun = dist_list.apply(get_rel_abun)
        peak_uniq_seq = dist_list.apply(get_uniq_seq)
        return peak_abun, peak_uniq_seq

    @staticmethod
    def from_peak_list(peak_list):
        """Create a mega peak of a list of peaks, distance to peak center will be the minimal distance to either
        of the center"""

        import copy

        mega_peak = copy.deepcopy(peak_list[0])
        mega_peak.name = f"Merged peak ({','.join([peak.name for peak in peak_list])})"
        mega_peak.center_seq = {peak.name: peak.center_seq for peak in peak_list}
        mega_peak.dist_to_center = pd.DataFrame({peak.name: peak.dist_to_center for peak in peak_list}).min(axis=1)
        mega_peak.peak_coverage = None

        return mega_peak



