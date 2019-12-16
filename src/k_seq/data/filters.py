"""This module contains filters to apply on SeqTable
"""

from .seq_table import slice_table
import pandas as pd


class FilterBase(object):
    """Abstract template for constructing filters"""

    def __init__(self, target=None, axis=0, *args, **kwargs):
        """A filter should contains at least target, axis, and reverse info for conducting filtering
        They could be assigned during instantiation or later
        """
        self.target = target
        self.axis = axis

    @staticmethod
    def func(target, *args, **kwargs):
        """Core method for filtering
        Returns a boolean pd.Series with same shape and order of the given axis in the target
        True for passed item, False for filtered item
        *reverse should apply outside this method
        """
        pass

    def mask(self, *args, **kwargs):
        """Return the the boolean mask for given target table
        Wrapper over func to for formatting and preprocessing over `func`
        """
        return self.func(*args, **kwargs)

    def get_passed_item(self, target=None, reverse=None, *args, **kwargs):
        """Return the items that pass the filter when reverse is False
        """
        if target is None:
            target = self.target
        mask = self.mask(target)
        if reverse:
            mask = ~mask
        if self.axis == 1:
            return list(target.columns[mask])
        else:
            return list(target.index[mask])

    def __call__(self, *args, **kwargs):
        """Directly call on the filter returns a filtered table"""
        return self.get_filtered_table(*args, **kwargs)

    def get_filtered_table(self, target=None, remove_zero=True, reverse=False, axis=None, *args, **kwargs):
        """Return a filtered table

        Args:
            target (pd.DataFrame): target table to filter
            remove_zero (bool): if remove all-zero items from another axis after filtering. Default True.
            reverse (bool): if return filtered items instead of items passed the filter. Default False.
            axis (0 or 1): if apply filter on index (0) or columns (1)
        """
        if target is None:
            target = self.target
        if hasattr(target, 'table'):
            target = target.table
        axis = self.axis if axis is None else axis
        return slice_table(table=target,
                           keys=self.get_passed_item(target=target, reverse=reverse, *args, **kwargs),
                           axis=axis,
                           remove_zero=remove_zero)

    def summary(self, target=None, **kwargs):
        """Returns a pd.DataFrame as the summary"""
        import pandas as pd
        if target is None:
            target = self.target
        mask = self.mask(target, **kwargs)
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

    @classmethod
    def from_func(cls, func, target=None, axis=0):
        inst = cls(target=target, axis=axis)
        inst.filter_fn = func
        return inst


class FilterCollection(object):
    """Applies a collection of filters to the object in sequence, Not implemented yet
    todo: add a pipeline for filters
    """
    pass


class SampleFilter(FilterBase):

    def __init__(self, target=None, samples_to_keep=None, samples_to_remove=None, axis=1):
        super().__init__(target, axis)
        if target is not None:
            if isinstance(target, pd.DataFrame):
                self.target = target
            else:
                self.target = getattr(target, 'table')
        self.samples_to_keep = samples_to_keep
        self.samples_to_remove = samples_to_remove
        self.axis = axis

    @staticmethod
    def func(target, samples_to_keep, axis=1):
        if axis == 0:
            return target.index.isin(samples_to_keep)
        else:
            return target.columns.isin(samples_to_keep)

    def mask(self, target=None, samples_to_keep=None, samples_to_remove=None, axis=None):
        if target is None:
            target = self.target
        elif isinstance(target, pd.DataFrame):
            pass
        else:
            target = getattr(target, 'table')
        if samples_to_keep is None:
            samples_to_keep = self.samples_to_keep
        if samples_to_remove is None:
            samples_to_remove = self.samples_to_remove
        if axis is None:
            axis = self.axis
        sample_list = target.columns if axis == 1 else target.index
        if samples_to_keep is not None:
            sample_list = [sample for sample in sample_list if sample in samples_to_keep]
        if samples_to_remove is not None:
            sample_list = [sample for sample in sample_list if sample not in samples_to_remove]
        return self.func(target, samples_to_keep=sample_list)


class SpikeInFilter(FilterBase):
    """Filter out seqs that are spike-in
    """

    def __init__(self, target=None, center_seq=None, radius=None, reverse=False, axis=0):
        super().__init__(target, axis)

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
                    # infer from SeqTable object
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
    def func(target, center_seq, radius, dist_to_center=None):
        from Levenshtein import distance

        def within_peak(seq):
            return distance(center_seq, seq)

        if dist_to_center is None:
            dist_to_center = target.index.to_series().apply(within_peak)
        non_peak_seq = dist_to_center[dist_to_center > radius]
        return target.index.isin(non_peak_seq.index)

    def mask(self, target=None, center_seq=None, radius=None, dist_to_center=None, reverse=True):
        if target is None:
            target = self.target
        if center_seq is None:
            center_seq = self.center_seq
        if radius is None:
            radius = self.radius
        if dist_to_center is None:
            dist_to_center = self.dist_to_center
        return self.func(target=target, center_seq=center_seq, dist_to_center=dist_to_center,
                         radius=radius)


class SeqLengthFilter(FilterBase):

    def __init__(self, target, min_len=None, max_len=None, axis=0):

        super().__init__(target)
        if isinstance(target, pd.DataFrame):
            self.target = target
        else:
            self.target = target.table
        self.min_len = min_len
        self.max_len = max_len
        self.axis = axis

    @staticmethod
    def func(target, min_len=None, max_len=None):
        import numpy as np
        seq_len = target.index.to_series().apply(len)
        mask = np.repeat(True, len(seq_len))
        if min_len is not None:
            mask = mask & (seq_len >= min_len)
        if max_len is not None:
            mask = mask & (seq_len <= max_len)
        return mask

    def mask(self, target=None, min_len=None, max_len=None):
        if target is None:
            target = self.target
        if min_len is None:
            min_len = self.min_len
        if max_len is None:
            max_len = self.max_len
        return self.func(target=target, min_len=min_len, max_len=max_len)


class SingletonFilter(FilterBase):

    def __init__(self, target, axis=0):

        super().__init__(target)
        if isinstance(target, pd.DataFrame):
            self.target = target
        else:
            self.target = target.table
        self.axis = axis

    @staticmethod
    def func(target, axis=1):
        mask = target.sum(axis=axis) == 1
        return ~mask

    def mask(self, target=None):
        if target is None:
            target = self.target
        return self.func(target)


class DetectedTimesFilter(FilterBase):

    def __init__(self, target=None, min_detected_times=6, axis=0):

        super().__init__(target)
        if target is not None:
            if isinstance(target, pd.DataFrame):
                self.target = target
            else:
                self.target = target.table
        self.min_detected_times = min_detected_times
        self.axis = axis

    @staticmethod
    def func(target, min_detected_times):
        return (target > 0).sum(axis=1) >= min_detected_times

    def mask(self, target=None, min_detected_times=None):
        if target is None:
            target = self.target
        if min_detected_times is None:
            min_detected_times = self.min_detected_times
        return self.func(target=target, min_detected_times=min_detected_times)

