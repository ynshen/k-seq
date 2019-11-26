
from .seq_table import slice_table


class FilterBase(object):
    """Base type for filters"""

    def __init__(self, target, axis=0):

        self.target = target
        self.axis = axis

    @staticmethod
    def func(**kwargs):
        """Standalone static methods that return a boolean pd.Series as mask"""
        pass

    def apply(self, **kwargs):
        """Wrapper over func that can run with class/optional info"""
        self.func(**kwargs)

    def get_passed_item(self, target=None, **kwargs):
        if target is None:
            target = self.target
        mask = self.apply(target)
        if self.axis == 1:
            return target.columns[mask]
        else:
            return target.index[mask]

    def summary(self, target=None, **kwargs):
        import pandas as pd
        if target is None:
            target = self.target
        mask = self.apply(target, **kwargs)
        if self.axis == 1:
            summary = pd.DataFrame(index=target.columns)
            summary['unique'] = (target > 0).sum(0)
            summary['unique_passed'] = (target.loc[mask] > 0).sum(0)
            summary['total'] = target.sum(0)
            summary['total_passed'] = target.loc[mask].sum(0)
        else:
            summary = pd.DataFrame(index=target.index,
                                   columns=['unique', 'unique_passed', 'total', 'total_passed'])
            summary['unique'] = (target > 0).sum(1)
            summary['unique_passed'] = (target[mask] > 0).sum(1)
            summary['total'] = target.sum(1)
            summary['total_passed'] = target[mask].sum(1)
        return summary

    def get_filtered_table(self, target=None, remove_zero=False):
        """Return the table with only True mask"""
        if target is None:
            target = self.target
        if hasattr(target, 'table'):
            target = target.table
        return slice_table(table=target,
                           keys=self.get_passed_item(target=target),
                           axis=self.axis,
                           remove_zero=remove_zero)

    @classmethod
    def from_func(cls, func, target=None, axis=0):
        inst = cls(target=target, axis=axis)
        inst.filter_fn = func
        return inst


class FilterCollection(object):
    """Applies a collection of filters to the object in sequence, with sev"""
    pass


class SpikeInFilter(FilterBase):

    def __init__(self, target, center_seq=None, radius=None, reverse=True, axis=0):
        super().__init__(target, axis)
        import pandas as pd

        if isinstance(target, pd.DataFrame):
            self.target = target
            if center_seq is None:
                raise ValueError('center_seq is None')
            else:
                self.center_seq = center_seq
            if radius is None:
                raise ValueError('radius is None')
            else:
                self.radius = radius
            self.dist_to_center = target.index.to_series().apply(self._edit_dist)
        elif hasattr(target, 'table'):
            self.target = target.table
            if center_seq is None:
                try:
                    self.center_seq = target.spike_in.spike_in_seq
                    self.dist_to_center = target.spike_in.dist_to_center
                    if radius is None:
                        self.radius = target.spike_in.radius
                    else:
                        self.radius = radius
                except:
                    raise ValueError('No spike-in information found')
            else:
                if hasattr(target, 'spike_in'):
                    if center_seq == target.spike_in.spike_in_seq:
                        self.center_seq = center_seq
                        self.dist_to_center = target.spike_in.dist_to_center
                        if radius is None:
                            self.radius = target.spike_in.radius
                        else:
                            self.radius = radius
                    else:
                        self.center_seq = center_seq
                        self.dist_to_center = target.index.to_series().apply(self._edit_dist)
                        self.radius = radius
                else:
                    self.center_seq = center_seq
                    self.dist_to_center = target.index.to_series().apply(self._edit_dist)
                    self.radius = radius
        self.reverse = reverse

    def _edit_dist(self, seq):
        from Levenshtein import distance
        return distance(seq, self.center_seq)

    @staticmethod
    def func(target, center_seq, radius, dist_to_center=None, reverse=True):
        from Levenshtein import distance

        def within_peak(seq):
            return distance(center_seq, seq) <= radius

        if dist_to_center is None:
            mask = target.index.to_series().apply(within_peak)
        else:
            mask = dist_to_center <= radius
        if reverse:
            return ~mask
        else:
            return mask

    def apply(self, target=None, center_seq=None, radius=None, dist_to_center=None, reverse=True):
        if target is None:
            target = self.target
        if center_seq is None:
            center_seq = self.center_seq
        if radius is None:
            radius = self.radius
        if dist_to_center is None:
            dist_to_center = self.dist_to_center
        return self.func(target=target, center_seq=center_seq, dist_to_center=dist_to_center,
                         radius=radius, reverse=reverse)


class SeqLengthFilter(FilterBase):

    def __init__(self, target, min_len=None, max_len=None, axis=0):
        import pandas as pd

        super().__init__(target)
        if isinstance(target, pd.DataFrame):
            self.target = target
        else:
            self.target = target.table
        self.min_len = min_len
        self.max_len = max_len
        self.axis = axis

    @staticmethod
    def func(target, min_len, max_len):
        import numpy as np
        seq_len = target.index.to_series().apply(len)
        mask = np.repeat(True, len(seq_len))
        if min_len is not None:
            mask = mask & (seq_len >= min_len)
        if max_len is not None:
            mask = mask & (seq_len <= max_len)
        return mask

    def apply(self, target=None, min_len=None, max_len=None):
        if target is None:
            target = self.target
        if min_len is None:
            min_len = self.min_len
        if max_len is None:
            max_len = self.max_len
        return self.func(target=target, min_len=min_len, max_len=max_len)


class SingletonFilter(FilterBase):

    def __init__(self, target, axis=0, reverse=True):
        import pandas as pd

        super().__init__(target)
        if isinstance(target, pd.DataFrame):
            self.target = target
        else:
            self.target = target.table
        self.reverse = reverse
        self.axis = axis

    @staticmethod
    def func(target, reverse=True):
        mask = target.sum(axis=1) == 1
        if reverse:
            return ~mask
        else:
            return mask

    def apply(self, target=None, reverse=None):
        if target is None:
            target = self.target
        if reverse is None:
            reverse = self.reverse
        return self.func(target, reverse)


class DetectedTimesFilter(FilterBase):

    def __init__(self, target, min_detected_times=6, axis=0, reverse=False):
        import pandas as pd

        super().__init__(target)
        if isinstance(target, pd.DataFrame):
            self.target = target
        else:
            self.target = target.table
        self.min_detected_times = min_detected_times
        self.reverse = reverse
        self.axis = axis

    @staticmethod
    def func(target, min_detected_times, reverse=True):
        mask = (target > 0).sum(axis=1) >= min_detected_times
        if reverse:
            return ~mask
        else:
            return mask

    def apply(self, target=None, min_detected_times=None, reverse=None):
        if target is None:
            target = self.target
        if min_detected_times is None:
            min_detected_times = self.min_detected_times
        if reverse is None:
            reverse = self.reverse
        return self.func(target=target, min_detected_times=min_detected_times, reverse=reverse)


class SampleFilter(FilterBase):

    def __init__(self, target, sample_to_keep=None, sample_to_remove=None):
        import pandas as pd

        super().__init__(target)
        if isinstance(target, pd.DataFrame):
            self.target = target
        else:
            self.target = target.table

        self.sample_to_keep = sample_to_keep
        self.sample_to_remove = sample_to_remove
        self.reverse = False
        self.axis = 1

    @staticmethod
    def func(target, sample_to_keep, reverse=False):
        mask = target.columns.isin(sample_to_keep)
        if reverse:
            return ~mask
        else:
            return mask

    def apply(self, target=None, sample_to_keep=None, sample_to_remove=None):
        if target is None:
            target = self.target
        if sample_to_keep is None:
            sample_to_keep = self.sample_to_keep
        if sample_to_remove is None:
            sample_to_remove = self.sample_to_remove

        if sample_to_keep is None:
            sample_to_keep = list(self.target.columns)
        if sample_to_remove is not None:
            sample_to_keep = [sample for sample in sample_to_keep if sample not in sample_to_remove]

        return self.func(target=target, sample_to_keep=sample_to_keep, reverse=False)

# class SeqFilter:
#
#     class Filter:
#         def __init__(self, func, value):
#             self.func = func
#             self.value = value
#
#     def __init__(self, seq_table, seq_length_range=None, max_edit_dist_to_seqs=None,
#                  min_occur_input=None, min_occur_reacted=None,
#                  min_counts_input=None, min_counts_reacted=None,
#                  min_rel_abun_input=None, min_rel_abun_reacted=None):
#         """
#         Filter object with some built-in filter options
#
#         Use `SeqFilter.filter_fn` to get the `callable` function
#
#         Use `SeqFilter.seq_to_keep` to get a list of sequences passed the filters
#
#         Args:
#             seq_table (`SeqTable`): the `SeqTable` instance to apply filters on
#             seq_length_range ([min, max]): only keep sequences within range [min, max]
#             max_edit_dist_to_seqs (`int`):
#             min_counts_input (`int`):
#             min_counts_reacted (`int`):
#             min_rel_abun_input (`float`): relative abundance is only based on valid sequences
#             min_rel_abun_reacted (`float`): relative abundance is only based on valid sequences
#         """
#
#         import numpy as np
#         import pandas as pd
#
#         self.seq_table = seq_table
#
#         if seq_length_range is not None:
#             self.seq_length_range = self.Filter(
#                 func=lambda seq: seq_length_range[0] <= len(seq) <= seq_length_range[1],
#                 value = seq_length_range)
#
#         if max_edit_dist_to_seqs is not None:
#             if isinstance(max_edit_dist_to_seqs, list) or isinstance(max_edit_dist_to_seqs, tuple):
#                 max_edit_dist_to_seqs = {seq[0]: int(seq[1]) for seq in max_edit_dist_to_seqs}
#
#             def edit_dist_filter_fn(seq):
#                 import Levenshtein
#                 flag = True
#                 for target, max_dist in max_edit_dist_to_seqs.items():
#                     flag = flag and Levenshtein.distance(seq, target) <= max_dist
#                 return flag
#
#             self.max_edit_dist_to_seqs = self.Filter(
#                 func=edit_dist_filter_fn,
#                 value=max_edit_dist_to_seqs
#             )
#
#         if min_occur_input is not None:
#             self.min_occur_input = self.Filter(
#                 func=lambda seq: np.sum(self.seq_table.count_table_input.loc[seq] > 0) >= min_occur_input,
#                 value=min_occur_input
#             )
#         if min_occur_reacted is not None:
#             self.min_occur_reacted = self.Filter(
#                 func=lambda seq: np.sum(self.seq_table.count_table_reacted.loc[seq] > 0) >= min_occur_reacted,
#                 value=min_occur_reacted
#             )
#
#
#         if min_counts_input is not None:
#             self.min_counts_input = self.Filter(
#                 func=lambda seq: self.seq_table.count_table_input.loc[seq].mean() >= min_counts_input,
#                 value=min_counts_input
#             )
#
#         if min_counts_reacted is not None:
#             self.min_counts_reacted = self.Filter(
#                 func=lambda seq: self.seq_table.count_table_reacted.loc[seq].mean() >= min_counts_reacted,
#                 value=min_counts_reacted
#             )
#
#         if min_rel_abun_input is not None:
#             self.min_rel_abun_input = self.Filter(
#                 func=lambda seq: (self.seq_table.count_table_input.loc[seq]/self.seq_table.count_table_input.sum(axis=0)).mean() >= min_rel_abun_input,
#                 value=min_rel_abun_input
#             )
#
#         if min_rel_abun_reacted is not None:
#             self.min_rel_abun_reacted = self.Filter(
#                 func=lambda seq: (self.seq_table.count_table_reacted.loc[seq]/self.seq_table.count_table_reacted.sum(axis=0)).mean() >= min_rel_abun_reacted,
#                 value=min_rel_abun_reacted
#             )
#
#     def print_filters(self):
#         print('Following filter added:')
#         for filter,content in self.__dict__.items():
#             if isinstance(content, self.Filter):
#                 print('\t{}:{}'.format(filter, content.value))
#
#     def apply_filters(self):
#         seq_to_keep = self.seq_table.count_table_reacted.index
#         self.seq_to_keep = seq_to_keep[seq_to_keep.map(self.filter_fn)]
#
#     @property
#     def filter_fn(self):
#         def _filter_fn(seq):
#             import numpy as np
#             flags = [filter.func(seq) for filter in self.__dict__.values()
#                      if isinstance(filter, self.Filter)]
#             return np.all(flags)
#         return _filter_fn