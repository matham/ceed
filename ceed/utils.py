'''Utilities
===================

Utilities used in :mod:`ceed`.
'''
import re

__all__ = ('fix_name', )

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
