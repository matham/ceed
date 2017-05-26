'''Ceed
=================

Ceed is an in vitro experiment that stimulates brain slices with a projector
and records their activity.
'''

import os

__all__ = ('__version__', 'has_gui_control', 'is_view_inst')

__version__ = '0.1-dev'

has_gui_control = False
'''whether ceed is running from a GUI or if it has been imported as a library.
If run as a GUI, all the widgets are displayed.
'''
is_view_inst = 'CEED_IS_VIEW' in os.environ
'''Whether ceed is imported from the client code that displays the projector
output as in :mod:`ceed.view.main`.
'''
