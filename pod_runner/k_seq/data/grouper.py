from .seq_table import slice_table


class Group(object):
    # todo: bug: self.target did not change if the linked table change (SeqTable.table)

    def __repr__(self):
        if self._type == 0:
            return f'Group: {self.group}'
        else:
            return f'Groups on:\n' + '\n'.join(f'{key}: {list(value)}' for key,value in self.group.items())

    def __getitem__(self, item):
        return self.group[item]

    def __iter__(self):
        if self._type == 0:
            return self.group.__iter__()
        else:
            return ((key, value) for key,value in self.group.items())

    def __init__(self, group, target=None, axis=1):
        import numpy as np
        import pandas as pd

        if isinstance(group, (list, np.ndarray, pd.Series)):
            self._type = 0
            self.group = list(group)
        elif isinstance(group, dict):
            self._type = 1
            self.group = group
        else:
            TypeError('Unaccepted group type')
        self.target = target
        self.axis = axis

    def get_table(self, target=None, remove_zero=False):
        if target is None:
            target = self.target
        if self._type == 0:
            return slice_table(table=target, keys=self.group, axis=self.axis, remove_zero=remove_zero)
        else:
            return ((key, slice_table(table=target, keys=members, axis=self.axis, remove_zero=remove_zero))
                    for key, members in self.group.items())


class Grouper(object):

    def __repr__(self):
        groupings = [grouping for grouping in self.__dict__.keys() if grouping != '_target']
        return f"Grouper contains {len(groupings)} groupings:\n" + '\n'.join(groupings)

    def __init__(self, groupers, target):
        self._target = target
        self.add(groupers, target)

    def add(self, groupers, target=None):
        if target is None:
            target = self._target
        for key, members in groupers.items():
            if isinstance(members, dict):
                if 'axis' in members.keys():
                    if len(members) !=2:
                        raise TypeError(f'Wrong dictionary format for group {key}')
                    else:
                        self.__setattr__(key, Group(members['group'], axis=members['axis'], target=target))
                else:
                    self.__setattr__(key, Group(members, axis=1, target=target))
            elif isinstance(members, list):
                self.__setattr__(key, Group(members, axis=1, target=target))