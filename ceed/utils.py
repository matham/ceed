"""Utilities
===================

Various tools used in :mod:`ceed`.
"""
import re
import pathlib
from collections import deque
from typing import List, Tuple, Any, Union, Dict

__all__ = (
    'update_key_if_other_key', 'collapse_list_to_counts', 'get_plugin_modules',
    'CeedWithID', 'UniqueNames',
)

_name_pat = re.compile('^(.+)-([0-9]+)$')


def update_key_if_other_key(items, key, value, other_key, key_map):
    """Given a dict, or list/tuple of dicts (recursively), it goes through all
    the dicts and updates the keys who match.

    Specifically, if a key matches ``key``, its value matches ``value``,
    there's another key named ``other_key`` in the dict, and the ``value`` of
    ``other_key`` is in ``key_map``, then the value of ``other_key`` is
    updated to that value from ``key_map``.
    """
    for item in items:
        if isinstance(item, dict):
            if key in item and item[key] == value and other_key in item:
                item[other_key] = key_map.get(item[other_key], item[other_key])
            update_key_if_other_key(
                item.values(), key, value, other_key, key_map)
        elif isinstance(item, (list, tuple)):
            update_key_if_other_key(item, key, value, other_key, key_map)


def collapse_list_to_counts(values: list) -> List[Tuple[Any, int]]:
    """Converts a sequence of items to tuples of the item and count of
    sequential items.

    E.g.::

        >>> collapse_list_to_counts([1, 1, 2, 3, 1, 1, 1, 3,])
        [(1, 2), (2, 1), (3, 1), (1, 3), (3, 1)]
    """
    counter = None
    last_item = object()
    res = []

    for value in values:
        if value != last_item:
            if counter is not None:
                res.append((last_item, counter))

            last_item = value
            counter = 1
        else:
            counter += 1

    if counter is not None:
        # we saw some items at least, the last was not added
        res.append((last_item, counter))

    return res


def get_plugin_modules(
        base_package: str, root: Union[str, pathlib.Path]
) -> Tuple[List[str], List[Tuple[Tuple[str], bytes]]]:
    """Takes a package name and it's corresponding root path and returns a list
    of the modules recursively within this package, as well as the source files
    in bytes.

    Only ``*.py`` files are considered, and although included with the source
    bytes, the ``packages`` list skips any files that start with a underscore
    (except ``__init__.py`` of course).
    """
    packages = []
    files = []

    fifo = deque([pathlib.Path(root)])
    while fifo:
        directory = fifo.popleft()
        relative_dir = directory.relative_to(root)
        directory_mod = '.'.join((base_package,) + relative_dir.parts)
        for item in directory.iterdir():
            if item.is_dir():
                if not item.name == '__pycache__':
                    fifo.append(item)
                continue

            if not item.is_file() or not item.name.endswith(('.py', '.pyo')):
                continue

            # only pick one of pyo/py
            if item.suffix == '.pyo' and item.with_suffix('.py').exists():
                continue

            files.append(
                (relative_dir.parts + (item.name, ), item.read_bytes()))
            if item.name.startswith('_') and item.name != '__init__.py' \
                    and item.name != '__init__.pyo':
                continue

            name = item.name[:-3]
            if name == '__init__':
                package = directory_mod
            else:
                package = f'{directory_mod}.{name}'

            packages.append(package)

    return packages, files


class CeedWithID:
    """Adds :attr:`ceed_id` to the class so that any inheriting class instance
    can be associated with a unique integer ID for logging purposes.

    The ID is not automatically set for every object, it is manually set
    when :meth:`set_ceed_id` is called. See stage/function for when it's called.
    """

    ceed_id: int = 0
    """The integer id of the object.
    """

    def set_ceed_id(self, min_available: int) -> int:
        """Sets the ID of this and any sub objects, each to a number equal or
        greater than ``min_available`` and returns the next minimum number
        available to be used.

        See :attr:`~ceed.analysis.CeedDataReader.event_data` for more details.

        :param min_available: The minimum number available to be used for the
            ID so it is unique.
        :return: The next minimum available number that can be used. Any number
            larger or equal to it is free to be used.
        """
        self.ceed_id = min_available
        return min_available + 1


class UniqueNames(set):
    r"""A set of names, that helps ensure no names in the set is duplicated.

    It provides a :meth:`fix_name` method that fixes the name, given some input
    such that the returned name will not be in the set.

    E.g.:

    .. code-block:: python

        >>> names = UniqueNames()
        >>> names.fix_name('floor')
        'floor'
        >>> names.add('floor')
        >>> names.fix_name('floor')
        'floor-1'
        >>> names.add('floor-1')
        >>> names.fix_name('floor')
        'floor-2'
        >>> names.fix_name('floor-1')
        'floor-2'
        >>> names.fix_name('floor-0')
        'floor-0'
        >>> names.remove('floor')
        >>> names.fix_name('floor')
        'floor'
        >>> names.fix_name('floor-1')
        'floor-2'
        >>> names.add('floor-1')
        Traceback (most recent call last):
          File "<ipython-input-14-b43fff249a6b>", line 1, in <module>
            names.add('floor-1')
          File "G:\Python\libs\ceed\ceed\utils.py", line 202, in add
            >>> names.add('floor')
        ValueError: Tried to add floor-1, but it is already in the set
    """

    prefix_count: Dict[str, List[int]] = {}

    name_num_pat = re.compile('^(.*?)(-[0-9]+)?$')

    def __init__(self, *args, **kwargs):
        set.__init__(self, *args, **kwargs)
        prefix_count = self.prefix_count = {}

        pat = self.name_num_pat
        for item in self:
            base, n = re.match(pat, item).groups()
            if n is None:
                n = 0
            else:
                n = int(n[1:])

            prefix_count[base] = [1, n + 1]

    def add(self, element: str) -> None:
        if element in self:
            raise ValueError(
                f'Tried to add {element}, but it is already in the set')
        set.add(self, element)

        prefix_count = self.prefix_count
        base, n = re.match(self.name_num_pat, element).groups()
        if n is None:
            n = 0
        else:
            n = int(n[1:])

        if base not in prefix_count:
            prefix_count[base] = [1, n + 1]
        else:
            count = prefix_count[base]
            count[0] += 1
            count[1] = max(count[1], n + 1)

    def fix_name(self, name: str) -> str:
        """If the name is already in the set, it returns a new name so that is
        not in the set. Otherwise, it returns the original name.

        :param name: The name to check if it already exists in the set.
        :return: The original or fixed name, such that it is not in the set.
        """
        if name not in self:
            return name

        base, _ = re.match(self.name_num_pat, name).groups()
        return f'{base}-{self.prefix_count[base][1]}'

    def remove(self, element: str) -> None:
        set.remove(self, element)

        base, _ = re.match(self.name_num_pat, element).groups()
        count = self.prefix_count[base]
        count[0] -= 1
        if not count[0]:
            del self.prefix_count[base]

    def clear(self) -> None:
        set.clear(self)

        self.prefix_count = {}
