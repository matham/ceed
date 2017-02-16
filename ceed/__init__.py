'''Ceed
=================

Ceed is an in vitro experiment that stimulates brain slices and records
their activity.
'''

import os

__version__ = '0.1-dev'

has_gui_control = False
is_view_inst = 'CEED_IS_VIEW' in os.environ
