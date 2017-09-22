'''Ceed App
=====================

The main module that runs the GUI.
'''

import ceed
if __name__ == '__main__':
    ceed.has_gui_control = True

from functools import partial
from os.path import join, dirname, isdir

from cplcom.app import CPLComApp, run_app as run_cpl_app
from cplcom.config import populate_dump_config
from cplcom.graphics import HighightButtonBehavior

from kivy.properties import ObjectProperty, NumericProperty, BooleanProperty
from kivy.resources import resource_add_path
from kivy.uix.behaviors.knspace import knspace
from kivy.garden.filebrowser import FileBrowser
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.factory import Factory

import ceed.graphics
Builder.load_file(join(dirname(__file__), 'graphics', 'graphics.kv'))
from ceed.player import CeedPlayer, CeedFFmpegPlayer, CeedPTGrayPlayer
import ceed.function.plugin
import ceed.shape
import ceed.stage
import ceed.function.func_widgets
import ceed.stage.stage_widgets
import ceed.shape.shape_widgets
import ceed.view.view_widgets

from ceed.function import FunctionFactory
from ceed.stage import StageFactory
from ceed.view.controller import ViewController
from ceed.storage.controller import CeedData, DataSerializer
from ceed.graphics import CeedDragNDrop

if ceed.has_gui_control:
    from kivy.core.window import Window

__all__ = ('CeedApp', 'run_app')


class CeedApp(CPLComApp):
    '''The app which runs the GUI.
    '''

    player = None

    agreed_discard = False

    drag_controller = None
    '''
    '''

    @classmethod
    def get_config_classes(cls):
        d = super(CeedApp, cls).get_config_classes()
        d['view'] = ViewController
        d['data'] = CeedData
        d['serializer'] = DataSerializer
        d['player'] = CeedPlayer
        d['point_gray_cam'] = CeedPTGrayPlayer
        d['video_file_playback'] = CeedFFmpegPlayer
        d['function'] = FunctionFactory
        if cls.get_running_app():
            p = d['player'] = cls.get_running_app().player
            d['point_gray_cam'] = p.pt_player
            d['video_file_playback'] = p.ff_player
        return d

    def __init__(self, **kwargs):
        self.player = CeedPlayer()
        super(CeedApp, self).__init__(**kwargs)
        self.load_app_settings_from_file()
        self.apply_app_settings()

    def build(self):
        base = dirname(__file__)
        # Builder.load_file(join(base, 'graphics', 'graphics.kv'))
        Builder.load_file(join(base, 'ceed_style.kv'))
        Builder.load_file(join(base, 'player_style.kv'))
        Builder.load_file(join(base, 'shape', 'shape_style.kv'))
        Builder.load_file(join(base, 'function', 'func_style.kv'))
        Builder.load_file(join(base, 'stage', 'stage_style.kv'))
        Builder.load_file(join(base, 'view', 'view_style.kv'))
        Builder.load_file(join(base, 'storage', 'storage_style.kv'))
        self.yesno_prompt = Factory.CeedYesNoPrompt()
        drag = self.drag_controller = CeedDragNDrop()
        drag.knsname = 'dragger'
        return super(CeedApp, self).build()

    def on_start(self):
        knspace.painter.show_widgets = True
        FunctionFactory.show_widgets = True
        StageFactory.show_widgets = True

        HighightButtonBehavior.init_class()

        CeedData.create_file('')

        StageFactory.fbind('on_changed', self.changed_callback)
        FunctionFactory.fbind('on_changed', self.changed_callback)
        knspace.painter.fbind('on_changed', self.changed_callback)
        ViewController.fbind('on_changed', self.changed_callback)

        self.set_tittle()
        CeedData.fbind('filename', self.set_tittle)
        CeedData.fbind('config_changed', self.set_tittle)
        CeedData.fbind('has_unsaved', self.set_tittle)

        try:
            ViewController.set_led_mode(ViewController.LED_mode_idle)
        except ImportError:
            pass

    def set_tittle(self, *largs):
        ''' Sets the title of the window using the currently running
        tab. This is called at 1Hz. '''
        star = ''
        if CeedData.has_unsaved or CeedData.config_changed:
            star = '*'
        if CeedData.filename:
            filename = ' - {}'.format(CeedData.filename)
        else:
            filename = ' - Unnamed File'
        Window.set_title('Ceed v{}, CPL lab{}{}'.format(
            ceed.__version__, star, filename))

    def changed_callback(self, *largs, **kwargs):
        CeedData.config_changed = True

    def check_close(self):
        if CeedPlayer.is_player_active():
            self._close_message = 'Cannot close while player is active.'
            return False
        if ViewController.stage_active or CeedData.data_thread:
            self._close_message = 'Cannot close during an experiment.'
            return False
        ViewController.stop_process()
        ViewController.finish_stop_process()
        if not CeedData.ui_close(app_close=True):
            self._close_message = ''
            return False
        return True

def _cleanup(app, *largs):
    CeedPlayer.exit_players()
    ViewController.stop_process()
    ViewController.finish_stop_process()
    CeedData.stop_experiment()
    app.dump_app_settings_to_file()

run_app = partial(run_cpl_app, CeedApp, _cleanup)
'''The function that starts the GUI and the entry point for
the main script.
'''

if __name__ == '__main__':
    run_app()
