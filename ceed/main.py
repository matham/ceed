'''Ceed App
=====================

The main module that runs the GUI.
'''

from functools import partial
from os.path import join, dirname, isdir

from cplcom.app import CPLComApp, run_app as run_cpl_app
from cplcom.config import populate_dump_config

from kivy.properties import ObjectProperty
from kivy.resources import resource_add_path
from kivy.uix.behaviors.knspace import knspace
from kivy.garden.filebrowser import FileBrowser
from kivy.lang import Builder

import ceed
import ceed.function.plugin
import ceed.shape

__all__ = ('CeedApp', 'run_app')


class CeedApp(CPLComApp):
    '''The app which runs the GUI.
    '''

    @classmethod
    def get_config_classes(cls):
        d = super(CeedApp, cls).get_config_classes()
        return d

    def __init__(self, **kwargs):
        super(CeedApp, self).__init__(**kwargs)
        settings = self.app_settings = populate_dump_config(
            self.ensure_config_file(self.json_config_path),
            self.get_config_classes())

        for k, v in settings['app'].items():
            setattr(self, k, v)

    def build(self):
        # Builder.load_file(join(dirname(__file__), 'record.kv'))
        return super(CeedApp, self).build()

    def on_start(self):
        self.set_tittle()
        Clock.schedule_interval(self.set_tittle, 1)

    def set_tittle(self, *largs):
        ''' Sets the title of the window using the currently running
        tab. This is called at 1Hz. '''
        Window.set_title('Ceed v{}, CPL lab'.format(
            ceed.__version__))

    def check_close(self):
        return True

def _cleanup():
    pass

run_app = partial(run_cpl_app, CeedApp, _cleanup)
'''The function that starts the GUI and the entry point for
the main script.
'''

if __name__ == '__main__':
    run_app()
