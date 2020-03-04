"""This module contains filters to apply on SeqData
TODO: write test cases for filters
"""

from .seq_data import slice_table
import pandas as pd
from abc import ABC, abstractmethod
from yutility import logging
from ..utility.func_tools import update_none


class Filter(ABC):
    """Template class for filters

    ``Filters`` filter a ``target`` table by index or column to another table, similar to ``Transformer``, but by
        design do not change the content in the table. To write a filter, one should:

        - indicate default ``axis`` in ``__init__`` and store as an attribute, indicating which axis you are filtering,
        0 for index (row), 1 for columns

        - implement staticmethod ``_get_mask`` with input of target table and return a bool mask along the axis it
        filters. True for pass, False for not pass.
        classmethod ``filter`` should then apply to a target pd.DataFrame table and return a filtered table

        - implement method ``get_mask`` wrapper over ``_get_mask`` to assign stored attributes as argument. By design,
        it should only need to take a ``target`` argument, or without argument, returns ``self.mask`` corresponding to
        ``self.target``

    """

    def __init__(self, target=None, axis=0):
        """A filter should contains at least axis info for filtering
        """
        self.axis = axis
        self.mask = None
        self.target = target

    @staticmethod
    @abstractmethod
    def _get_mask(target, axis, **kwargs):
        """Implementation required
        Returns a boolean pd.Series with same shape and order of the axis to filter
        True for passed item, False for filtered item
        *reverse should apply outside this method
        """
        pass

    @abstractmethod
    def get_mask(self, target=None, axis=None, **kwargs):
        """Implementation required
        Return the the boolean mask for given table
        Wrapper over _get_mask to for preprocessing arguments
        This content is a minimal example
        """
        if target is None:
            return self.mask
        axis = update_none(axis, self.axis)
        return self._get_mask(target, axis, **kwargs)

    @classmethod
    def filter(cls, target, axis=None, remove_empty=False, reverse=False, **kwargs):
        """Classmethod to directly apply filters"""

        mask = cls._get_mask(target=target, axis=axis, **kwargs)
        if reverse:
            mask = ~mask
        if axis is None:
            logging.error('Please indicate axis to filter', error_type=ValueError)
        if axis == 0:
            return slice_table(table=target,
                               keys=target.index[mask],
                               axis=0,
                               remove_empty=remove_empty)
        else:
            return slice_table(table=target,
                               keys=target.columns[mask],
                               axis=1,
                               remove_empty=remove_empty)

    @property
    def target(self):
        if hasattr(self, '_target') and self._target is not None:
            return self._target
        else:
            return None

    @target.setter
    def target(self, value):
        """You update the target, you update the mask"""
        if value is None:
            self._target = None
        elif isinstance(value, pd.DataFrame):
            # otherwise do the filtering the update the mask
            self._target = value
            self.mask = self.get_mask(target=value)

    def get_passed_item(self, target=None, axis=None, reverse=False):
        """Return the items that pass the filter, pre-saved mask could be provided
        if ``reverse`` is false, get filtered items
        """
        if target is None:
            target = self.target
            mask = self.mask
        else:
            mask = self.get_mask(target)
        if reverse:
            mask = ~mask
        axis = update_none(axis, self.axis)
        if axis == 1:
            return list(target.columns[mask])
        else:
            return list(target.index[mask])

    def __call__(self, target=None, remove_empty=False, reverse=False, axis=None):
        """Directly call on the filter returns a filtered table"""
        return self.get_filtered_table(target=target, remove_empty=remove_empty, reverse=reverse, axis=axis)

    def get_filtered_table(self, target=None, remove_empty=False, reverse=False, axis=None):
        """Return a filtered table

        Args:
            target (pd.DataFrame): table to filter
            remove_empty (bool): if remove all-zero items from another axis after filtering. Default False.
            reverse (bool): if return filtered items instead of items passed the filter. Default False.
            axis (0 or 1): if apply filter on index/row (0) or columns (1)
        """

        if target is None:
            target = self.target
            keys = self.get_passed_item(reverse=reverse)
        else:
            keys = self.get_passed_item(target, reverse=reverse)
        axis = update_none(axis, self.axis)
        from .seq_data import SeqTable
        if isinstance(target, SeqTable):
            unit = target.unit
        else:
            unit = None
        return SeqTable(slice_table(table=target,
                                    keys=keys,
                                    axis=axis,
                                    remove_empty=remove_empty), unit=unit)

    def summary(self, target=None, axis=None, **kwargs):
        """Returns a pd.DataFrame as the summary"""
        if target is None:
            target = self.target
            filtered = self.get_filtered_table()
        else:
            filtered = self.get_filtered_table(target, **kwargs)

        axis = update_none(axis, self.axis)

        if axis == 0:
            summary = pd.DataFrame(index=target.columns)
        else:
            summary = pd.DataFrame(index=target.index,
                                   columns=['unique', 'unique_passed', 'total', 'total_passed'])
        summary['unique'] = (target > 0).sum(axis)
        summary['unique_passed'] = (filtered > 0).sum(axis)
        summary['total'] = target.sum(axis)
        summary['total_passed'] = filtered.sum(axis)
        return summary


class CustomizedFilter(Filter):

    def __init__(self, mask_func, target=None, axis=0):
        """Create a Filter object from a given filter function
        filter function should take a pd.DataFrame and return a boolean mask (True for pass, False to non pass)
        """
        super().__init__(target=target, axis=axis)
        self._get_mask = mask_func

    @staticmethod
    def _get_mask(**kwargs):
        pass

    def get_mask(self, target=None, **kwargs):
        return self._get_mask(target, **kwargs)


class FilterPipe(Filter):
    """Applies a collection of filters to the object in sequence, Not implemented yet
    """

    def __init__(self, filters, target=None, axis=0):
        if not isinstance(filters, (list, tuple)):
            logging.error("`filters` should be a list of Filter instance")
        self.filters = filters
        super().__init__(target=target, axis=axis)

    def _get_mask_piped(self, target, axis=None):
        import numpy as np

        axis = update_none(axis, self.axis)
        mask = pd.Series(np.repeat(True, target.shape[axis]), index=target.index if axis == 0 else target.column)
        for filter in self.filters:
            mask = mask & filter.get_mask(target=target, axis=axis)
        return mask

    @staticmethod
    def _get_mask(target, axis, **kwargs):
        logging.error('not `_get_mask` but `_get_mask_piped` is implemented for FilterPipe',
                      error_type=NotImplementedError)

    def get_mask(self, target=None, axis=None, *args, **kwargs):
        if target is None:
            return self.mask
        else:
            axis = update_none(axis, self.axis)
            return self._get_mask_piped(target=target, axis=axis)


class SampleFilter(Filter):
    """Filter samples based on index name"""

    def __init__(self, target=None, samples_to_keep=None, samples_to_remove=None, axis=1):
        super().__init__(target, axis)
        self.samples_to_keep = samples_to_keep
        self.samples_to_remove = samples_to_remove

    @staticmethod
    def _get_mask(target, samples_to_keep=None, samples_to_remove=None, axis=1):
        sample_list = target.columns if axis == 1 else target.index
        if samples_to_keep is not None:
            sample_list = [sample for sample in sample_list if sample in samples_to_keep]
        if samples_to_remove is not None:
            sample_list = [sample for sample in sample_list if sample not in samples_to_remove]
        if axis == 0:
            return target.index.isin(sample_list)
        else:
            return target.columns.isin(sample_list)

    def get_mask(self, target=None, samples_to_keep=None, samples_to_remove=None, axis=None):
        target = update_none(target, self.target)
        samples_to_keep = update_none(samples_to_keep, self.samples_to_keep)
        samples_to_remove = update_none(samples_to_remove, self.samples_to_remove)
        axis = update_none(axis, self.axis)
        return self._get_mask(target, samples_to_keep=samples_to_keep, samples_to_remove=samples_to_remove, axis=axis)


class SpikeInFilter(Filter):
    """Filter out seqs that are spike-in

    Attributes:
        peak (landscape.Peak): the spike-in peak instance
    """

    def __init__(self, target, center_seq=None, radius=None, dist_type=None, reverse=False, axis=0):
        """
        Args:
            target (pd.DataFrame or table.SeqData): needed to calculate distance of sequences to center seq
                If target is pd.DataFrame, center, radius, and dist_type must
                provide. If target is SeqData, it must infer from ``SeqData.spike_in`` accessor if applicable for
                consistency
            center_seq (str): center sequence of added spike-in sequence
            radius (int): radius of spike-in peak. Seqs with distance ≤ radius will be filtered
        """
        from ..data.seq_data import SeqData

        super().__init__(target, axis)

        if isinstance(target, pd.DataFrame) or (isinstance(target, SeqData) and not hasattr(target, 'spike_in')):
            # if not spike-in info could infer
            if center_seq is None:
                logging.error('center_seq is None', error_type=ValueError)
            else:
                self.center_seq = center_seq
            if radius is None:
                logging.error('radius is None', error_type=ValueError)
            else:
                self.radius = radius
            if dist_type is None:
                logging.error('dist_type is None', error_type=ValueError)
            else:
                self.dist_type = dist_type
            from landscape import Peak
            self.peak = Peak(center_seqs=center_seq, seqs=target, radius=radius, dist_type=dist_type, name='spike-in')
            self.target = target if isinstance(target, pd.DataFrame) else target.table.original
        elif isinstance(target, SeqData):
            if ((center_seq is not None) and (center_seq != target.spike_in.peak.center_seq)) or \
                    ((radius is not None) and (radius != target.spike_in.peak.radius)) or \
                    ((dist_type is not None) and (dist_type != target.spike_in.peak.dist_type)):
                logging.error("spike-in info found in target and does not match argument", error_type=ValueError)
            else:
                self.peak = target.spike_in.peak
                self.target = target.table.original
        else:
            logging.error("Unknown target type", error_type=TypeError)
        self.reverse = reverse

    @staticmethod
    def _get_mask(target, radius, dist_to_center, axis):
        non_peak_seq = dist_to_center[dist_to_center > radius]
        if axis == 0:
            return target.index.isin(non_peak_seq.index)
        else:
            return target.column.isin(non_peak_seq.index)

    def get_mask(self, target=None, axis=None):
        """Apply to a target table"""
        target = update_none(target, self.target)
        axis = update_none(axis, self.axis)
        return self._get_mask(target=target, dist_to_center=self.peak.dist_to_center,
                              radius=self.peak.radius, axis=axis)


class SeqLengthFilter(Filter):
    """Filter out sequences with length < min_len or > max_len"""

    def __init__(self, target=None, min_len=None, max_len=None, axis=0):

        super().__init__(target)
        self.min_len = min_len
        self.max_len = max_len
        self.axis = axis
        if isinstance(target, pd.DataFrame):
            self.target = target
        else:
            self.target = target.table.original

    @staticmethod
    def _get_mask(target, min_len=None, max_len=None, axis=0):
        import numpy as np
        if axis == 0:
            seq_len = target.index.to_series().apply(len)
        else:
            seq_len = target.columns.to_series().apply(len)
        mask = np.repeat(True, len(seq_len))
        if min_len is not None:
            mask = mask & (seq_len >= min_len)
        if max_len is not None:
            mask = mask & (seq_len <= max_len)
        return mask

    def get_mask(self, target=None, min_len=None, max_len=None, axis=None):
        target = update_none(target, self.target)
        min_len = update_none(min_len, self.min_len)
        max_len = update_none(max_len, self.max_len)
        axis = update_none(axis, self.axis)
        return self._get_mask(target=target, min_len=min_len, max_len=max_len, axis=axis)


class SingletonFilter(Filter):
    """Filter out sequence has only been detected 1 count across all samples"""

    def __init__(self, target, axis=0):

        super().__init__(target)
        self.axis = axis
        if isinstance(target, pd.DataFrame):
            self.target = target
        else:
            self.target = target.table.original

    @staticmethod
    def _get_mask(target, axis=0):
        mask = target.sum(axis=1 - axis) == 1
        return ~mask

    def get_mask(self, target=None, axis=None, **kwargs):
        target = update_none(target, self.target)
        axis = update_none(axis, self.axis)
        return self._get_mask(target, axis=axis)


class DetectedTimesFilter(Filter):
    """Filter sequences by the minimal times being detected (count > 0)"""

    def __init__(self, target=None, min_detected_times=6, axis=0):
        """Filter"""

        super().__init__(target)
        self.min_detected_times = min_detected_times
        self.axis = axis
        if target is not None:
            if isinstance(target, pd.DataFrame):
                self.target = target
            else:
                self.target = target.table.original

    @staticmethod
    def _get_mask(target, min_detected_times, axis=0):
        return (target > 0).sum(axis=1 - axis) >= min_detected_times

    def get_mask(self, target=None, min_detected_times=None, axis=None):
        target = update_none(target, self.target)
        min_detected_times = update_none(min_detected_times, self.min_detected_times)
        axis = update_none(axis, self.axis)
        return self._get_mask(target=target, min_detected_times=min_detected_times, axis=axis)
