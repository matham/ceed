'''Ceed App
=====================

The main module that runs the GUI.
'''

from functools import partial
from os.path import join, dirname, isdir

from cplcom.app import CPLComApp, run_app as run_cpl_app

from kivy.properties import ObjectProperty
from kivy.resources import resource_add_path
from kivy.uix.behaviors.knspace import knspace
from kivy.garden.filebrowser import FileBrowser
from kivy.lang import Builder

import ceed.function.plugin
import ceed.shape

__all__ = ('CeedApp', 'run_app')


class CeedApp(CPLComApp):
    '''The app which runs the GUI.
    '''

    def __init__(self, **kwargs):
        self.data_directory = join(dirname(__file__), 'data')
        super(CeedApp, self).__init__(**kwargs)

run_app = partial(run_cpl_app, CeedApp)
'''The function that starts the GUI and the entry point for
the main script.
'''

if __name__ == '__main__':
    run_app()
