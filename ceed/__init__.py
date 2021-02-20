'''Ceed
=================

Ceed is an in vitro experiment that stimulates brain slices with a projector
and records their activity.
'''

import sys
import os
import pathlib

__all__ = ('__version__', 'is_view_inst', 'get_pyinstaller_datas')

__version__ = '1.0.0.dev1'

is_view_inst = 'CEED_IS_VIEW' in os.environ
'''Whether ceed is imported from the client code that displays the projector
output as in :mod:`ceed.view.main`.
'''


def get_pyinstaller_datas():
    """Returns the ``datas`` list required by PyInstaller to be able to package
    :mod:`base_kivy_app` in a application.

    """
    root = pathlib.Path(os.path.dirname(sys.modules[__name__].__file__))
    datas = []
    for pat in ('**/*.kv', '*.kv'):
        for f in root.glob(pat):
            datas.append((str(f), str(f.relative_to(root.parent).parent)))

    return datas
