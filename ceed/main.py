'''Ceed App
=====================

The main module that runs the GUI.
'''

import ceed
if __name__ == '__main__':
    ceed.has_gui_control = True

from functools import partial
from os.path import join, dirname

from cplcom.app import CPLComApp, run_app as run_cpl_app
from cplcom.graphics import HighightButtonBehavior

from kivy.uix.behaviors.knspace import knspace
from kivy.lang import Builder
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

from ceed.function import FunctionFactoryBase, register_all_functions
from ceed.stage import StageFactoryBase, _bind_remove
from ceed.view.controller import ControllerSideViewControllerBase
from ceed.storage.controller import CeedDataWriterBase, DataSerializerBase
from ceed.graphics import CeedDragNDrop
from ceed.remote.remote_view import RemoteViewerListenerBase

if ceed.has_gui_control:
    from kivy.core.window import Window

__all__ = ('CeedApp', 'run_app')


class CeedApp(CPLComApp):
    '''The app which runs the GUI.
    '''

    function_factory = None

    player = None

    view_controller = None

    ceed_data = None

    data_serializer = None

    remote_viewer = None

    stage_factory = None

    shape_factory = None

    agreed_discard = False

    drag_controller = None
    '''
    '''

    @classmethod
    def get_config_classes(cls):
        d = super(CeedApp, cls).get_config_classes()
        d['function'] = FunctionFactoryBase
        d['view'] = ControllerSideViewControllerBase
        d['data'] = CeedDataWriterBase
        d['serializer'] = DataSerializerBase
        d['player'] = CeedPlayer
        d['point_gray_cam'] = CeedPTGrayPlayer
        d['video_file_playback'] = CeedFFmpegPlayer
        d['remote_viewer'] = RemoteViewerListenerBase

        app = cls.get_running_app()
        if app is not None:
            d['function'] = app.function_factory
            d['view'] = app.view_controller
            d['data'] = app.ceed_data
            d['serializer'] = app.data_serializer
            p = d['player'] = app.player
            d['point_gray_cam'] = p.pt_player
            d['video_file_playback'] = p.ff_player
            d['remote_viewer'] = app.remote_viewer
        return d

    def __init__(self, **kwargs):
        self.function_factory = FunctionFactoryBase()
        register_all_functions(self.function_factory)
        self.stage_factory = StageFactoryBase(
            function_factory=self.function_factory)
        _bind_remove(self.stage_factory, self.shape_factory)
        self.player = CeedPlayer()
        self.view_controller = ControllerSideViewControllerBase()
        self.ceed_data = CeedDataWriterBase()
        self.data_serializer = DataSerializerBase()
        self.remote_viewer = RemoteViewerListenerBase()
        super(CeedApp, self).__init__(**kwargs)
        self.load_app_settings_from_file()
        self.apply_app_settings()

    def build(self):
        base = dirname(__file__)
        # Builder.load_file(join(base, 'graphics', 'graphics.kv'))
        Builder.load_file(join(base, 'ceed_style.kv'))
        Builder.load_file(join(base, 'player', 'player_style.kv'))
        Builder.load_file(join(base, 'shape', 'shape_style.kv'))
        Builder.load_file(join(base, 'function', 'func_style.kv'))
        Builder.load_file(join(base, 'stage', 'stage_style.kv'))
        Builder.load_file(join(base, 'view', 'view_style.kv'))
        Builder.load_file(join(base, 'storage', 'storage_style.kv'))
        self.yesno_prompt = Factory.CeedYesNoPrompt()
        drag = self.drag_controller = CeedDragNDrop()
        drag.knsname = 'dragger'

        root = Factory.get('MainView')()
        return super(CeedApp, self).build(root)

    def on_start(self):
        self.stage_factory.shape_factory = self.shape_factory = knspace.painter
        knspace.painter.add_shapes_to_canvas = \
            knspace.painter.show_widgets = True
        self.function_factory.show_widgets = True
        self.stage_factory.show_widgets = True

        HighightButtonBehavior.init_class()

        self.ceed_data.create_file('')

        self.stage_factory.fbind('on_changed', self.changed_callback)
        self.function_factory.fbind('on_changed', self.changed_callback)
        knspace.painter.fbind('on_changed', self.changed_callback)
        self.view_controller.fbind('on_changed', self.changed_callback)

        self.set_tittle()
        self.ceed_data.fbind('filename', self.set_tittle)
        self.ceed_data.fbind('config_changed', self.set_tittle)
        self.ceed_data.fbind('has_unsaved', self.set_tittle)

        try:
            self.view_controller.set_led_mode(self.view_controller.LED_mode_idle)
        except ImportError:
            pass

    def set_tittle(self, *largs):
        ''' Sets the title of the window using the currently running
        tab. This is called at 1Hz. '''
        star = ''
        if self.ceed_data.has_unsaved or self.ceed_data.config_changed:
            star = '*'
        if self.ceed_data.filename:
            filename = ' - {}'.format(self.ceed_data.filename)
        else:
            filename = ' - Unnamed File'
        Window.set_title('Ceed v{}, CPL lab{}{}'.format(
            ceed.__version__, star, filename))

    def changed_callback(self, *largs, **kwargs):
        self.ceed_data.config_changed = True

    def check_close(self):
        if CeedPlayer.is_player_active():
            self._close_message = 'Cannot close while player is active.'
            return False
        if self.view_controller.stage_active or self.ceed_data.data_thread:
            self._close_message = 'Cannot close during an experiment.'
            return False
        self.view_controller.stop_process()
        self.view_controller.finish_stop_process()
        if not self.ceed_data.ui_close(app_close=True):
            self._close_message = ''
            return False
        return True


def _cleanup(app, *largs):
    if app.remote_viewer:
        app.remote_viewer.stop_listener()
    CeedPlayer.exit_players()
    if app.view_controller is not None:
        app.view_controller.stop_process()
        app.view_controller.finish_stop_process()
    if app.ceed_data is not None:
        app.ceed_data.stop_experiment()
    app.dump_app_settings_to_file()

run_app = partial(run_cpl_app, CeedApp, _cleanup)
'''The function that starts the GUI and the entry point for
the main script.
'''

if __name__ == '__main__':
    run_app()
