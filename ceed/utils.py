'''Utilities
===================

Utilities used in :mod:`ceed`.
'''
import re
import pathlib
from collections import deque
from typing import List, Tuple, Any, Union

__all__ = (
    'fix_name', 'update_key_if_other_key', 'collapse_list_to_counts',
    'get_plugin_modules',
)

_name_pat = re.compile('^(.+)-([0-9]+)$')


def fix_name(name, *names):
    '''Fixes the name so that it is unique among the names in ``names``.

    :Params:

        `name`: str
            A name of something
        `*names`: iterables of strings
            Positional argument, where each is a iterable of strings among
            which we ensure that the returned name is unique.

    :returns:

        A string that is unique among all the ``names``, but is similar to
        ``name``. We append a integer to make it unique.

    E.g.::

        >>> fix_name('troll', ['toll', 'foll'], ['bole', 'cole'])
        'troll'
        >>> fix_name('troll', ['troll', 'toll', 'foll'], ['bole', 'cole'])
        'troll-2'
        >>> fix_name('troll', ['troll-2', 'toll', 'foll'], ['bole', 'cole'])
        'troll'
        >>> fix_name('troll', ['troll', 'troll-2', 'toll', 'foll'], \
['bole', 'cole'])
        'troll-3'
    '''
    if not any((name in n for n in names)):
        return name

    m = re.match(_name_pat, name)
    i = 2
    if m is not None:
        name, i = m.groups()
        i = int(i)

    new_name = '{}-{}'.format(name, i)
    while any((new_name in n for n in names)):
        i += 1
        new_name = '{}-{}'.format(name, i)
    return new_name


def update_key_if_other_key(items, key, value, other_key, key_map):
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
                fifo.append(item)
                continue

            if not item.is_file() or not item.name.endswith('.py'):
                continue

            files.append(
                (relative_dir.parts + (item.name, ), item.read_bytes()))
            if item.name.startswith('_') and item.name != '__init__.py':
                continue

            name = item.name[:-3]
            if name == '__init__':
                package = directory_mod
            else:
                package = f'{directory_mod}.{name}'

            packages.append(package)

    return packages, files
