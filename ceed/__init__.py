"""Ceed
=======

Ceed is a Python based application to help run in-vitro experiments that
optically stimulates brain slices and records their activity.

Ceed is the user interface for describing and running the experiments.
The overall system is described in :ref:`ceed-blueprint`.
See the :ref:`ceed-guide` for the Ceed specific guide.
"""

import sys
import os
import pathlib

__all__ = ('__version__', 'is_view_inst', 'get_pyinstaller_datas')

__version__ = '1.0.0.dev2'

is_view_inst = 'CEED_IS_VIEW' in os.environ
'''As described in :mod:`~ceed.view.controller`, Ceed will start a second
process to run a "real" experiment in full-screen.

:attr:`is_view_inst` is automatically set to True when Ceed is running in
this second process.
'''


def get_pyinstaller_datas():
    """Returns the ``datas`` list required by PyInstaller to be able to package
    :mod:`ceed` in a executable application.

    """
    root = pathlib.Path(os.path.dirname(sys.modules[__name__].__file__))
    datas = []
    for pat in ('**/*.kv', '*.kv'):
        for f in root.glob(pat):
            datas.append((str(f), str(f.relative_to(root.parent).parent)))

    return datas
