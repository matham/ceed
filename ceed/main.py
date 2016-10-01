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
from kivy.core.window import Window
from kivy.clock import Clock

import ceed
from ceed.player import CeedPlayer
import ceed.function.plugin
import ceed.shape
import ceed.function
import ceed.stage
import ceed.utils

__all__ = ('CeedApp', 'run_app')


class CeedApp(CPLComApp):
    '''The app which runs the GUI.
    '''

    player = None

    @classmethod
    def get_config_classes(cls):
        d = super(CeedApp, cls).get_config_classes()
        return d

    def __init__(self, **kwargs):
        self.player = CeedPlayer()
        super(CeedApp, self).__init__(**kwargs)
        settings = self.app_settings = populate_dump_config(
            self.ensure_config_file(self.json_config_path),
            self.get_config_classes())

        for k, v in settings['app'].items():
            setattr(self, k, v)

    def build(self):
        Builder.load_file(join(dirname(__file__), 'ceed_style.kv'))
        Builder.load_file(join(dirname(__file__), 'player_style.kv'))
        Builder.load_file(join(dirname(__file__), 'shape', 'shape_style.kv'))
        Builder.load_file(join(dirname(__file__), 'function', 'func_style.kv'))
        Builder.load_file(join(dirname(__file__), 'stage', 'stage_style.kv'))
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
        if CeedPlayer.is_player_active():
            self._close_message = 'Cannot close while active.'
            return False
        return True

def _cleanup():
    CeedPlayer.exit_players()

run_app = partial(run_cpl_app, CeedApp, _cleanup)
'''The function that starts the GUI and the entry point for
the main script.
'''

if __name__ == '__main__':
    run_app()
