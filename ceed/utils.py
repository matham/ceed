'''Utilities
===================

Utilities used in :mod:`ceed`.
'''
import re

__all__ = ('fix_name', 'update_key_if_other_key')

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
