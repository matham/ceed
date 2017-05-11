import re

_name_pat = re.compile('^(.+)-([0-9]+)$')


def fix_name(name, *names):
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
