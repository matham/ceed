'''Ceed
=================

Ceed is an in vitro experiment that stimulates brain slices with a projector
and records their activity.
'''

import os

__all__ = ('__version__', 'is_view_inst')

__version__ = '1.0.0-dev'

is_view_inst = 'CEED_IS_VIEW' in os.environ
'''Whether ceed is imported from the client code that displays the projector
output as in :mod:`ceed.view.main`.
'''
