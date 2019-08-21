'''Ceed App
=====================

The main module that runs the GUI.
'''
import ceed

from functools import partial
from os.path import join, dirname
import time

from cplcom.app import CPLComApp, run_app as run_cpl_app
from cplcom.graphics import HighightButtonBehavior, BufferImage

from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import ObjectProperty, BooleanProperty

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
import ceed.storage.storage_widgets

from ceed.function import FunctionFactoryBase, register_all_functions
from ceed.stage import StageFactoryBase, remove_shapes_upon_deletion
from ceed.stage.stage_widgets import StageList
from ceed.view.controller import ControllerSideViewControllerBase
from ceed.storage.controller import CeedDataWriterBase, DataSerializerBase
from ceed.graphics import CeedDragNDrop
from ceed.remote.remote_view import RemoteViewerListenerBase
from ceed.player import CeedRemotePlayer
from ceed.function.func_widgets import FuncNoiseDropDown, FuncList
from ceed.shape.shape_widgets import CeedPainter, ShapeList, ShapeGroupList
from ceed.view.view_widgets import MEAArrayAlign

from kivy.core.window import Window

__all__ = ('CeedApp', 'run_app')


class CeedApp(CPLComApp):
    '''The app which runs the GUI.
    '''

    kv_loaded = False
    """For tests, we don't want to load kv multiple times.
    """

    function_factory = None  # type: FunctionFactoryBase

    player = None  # type: CeedPlayer

    view_controller = None  # type: ControllerSideViewControllerBase

    ceed_data = None  # type: CeedDataWriterBase

    data_serializer = None  # type: DataSerializerBase

    remote_viewer = None  # type: RemoteViewerListenerBase

    stage_factory = None  # type: StageFactoryBase

    shape_factory = ObjectProperty(None, rebind=True)  # type: CeedPainter

    remote_player = None  # type: CeedRemotePlayer

    agreed_discard = False

    drag_controller = ObjectProperty(None, rebind=True)  # type: CeedDragNDrop

    noise_dropdown_widget = None
    '''
    '''

    stages_container = ObjectProperty(None, rebind=True)  # type: StageList

    funcs_container = ObjectProperty(None, rebind=True)  # type: FuncList

    shapes_container = ObjectProperty(None, rebind=True)  # type: ShapeList

    shape_groups_container = ObjectProperty(
        None, rebind=True)  # type: ShapeGroupList

    pinned_graph = None
    """PinnedGraph into which the stage graph may be pinned.
    """

    mea_align_widget = ObjectProperty(None, rebind=True)  # type: MEAArrayAlign

    central_display = ObjectProperty(None, rebind=True)  # type: BufferImage

    use_remote_view = BooleanProperty(False)

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
        return d

    def get_app_config_classes(self):
        d = super(CeedApp, self).get_app_config_classes()
        d['function'] = self.function_factory
        d['view'] = self.view_controller
        d['data'] = self.ceed_data
        d['serializer'] = self.data_serializer

        p = d['player'] = self.player
        d['point_gray_cam'] = p.pt_player
        d['video_file_playback'] = p.ff_player
        d['remote_viewer'] = self.remote_viewer

        return d

    def __init__(self, **kwargs):
        self.drag_controller = CeedDragNDrop()
        self.function_factory = FunctionFactoryBase()
        register_all_functions(self.function_factory)
        self.stage_factory = StageFactoryBase(
            function_factory=self.function_factory, shape_factory=None)
        self.player = CeedPlayer()
        self.view_controller = ControllerSideViewControllerBase()
        self.ceed_data = CeedDataWriterBase()
        self.data_serializer = DataSerializerBase()
        self.remote_viewer = RemoteViewerListenerBase()
        self.remote_player = CeedRemotePlayer()
        super(CeedApp, self).__init__(**kwargs)
        self.load_app_settings_from_file()
        self.apply_app_settings()

    def load_app_kv(self):
        if CeedApp.kv_loaded:
            return
        CeedApp.kv_loaded = True

        base = dirname(__file__)
        # Builder.load_file(join(base, 'graphics', 'graphics.kv'))
        Builder.load_file(join(base, 'ceed_style.kv'))
        Builder.load_file(join(base, 'player', 'player_style.kv'))
        Builder.load_file(join(base, 'shape', 'shape_style.kv'))
        Builder.load_file(join(base, 'function', 'func_style.kv'))
        Builder.load_file(join(base, 'stage', 'stage_style.kv'))
        Builder.load_file(join(base, 'view', 'view_style.kv'))
        Builder.load_file(join(base, 'storage', 'storage_style.kv'))

    def build(self):
        self.load_app_kv()
        self.yesno_prompt = Factory.CeedYesNoPrompt()
        self.noise_dropdown_widget = FuncNoiseDropDown()

        root = Factory.get('MainView')()
        return super(CeedApp, self).build(root)

    def _clear_all(self):
        self.funcs_container.clear_all()
        self.stages_container.clear_all()

    def on_start(self):
        self.stage_factory.shape_factory = self.shape_factory
        remove_shapes_upon_deletion(
            self.stage_factory, self.shape_factory,
            self.stages_container.remove_shape_from_stage)
        self.shape_factory.shape_widgets_list = self.shapes_container

        self.ceed_data.stage_display_callback = \
            self.stages_container.show_stage
        self.ceed_data.func_display_callback = \
            self.funcs_container.show_function
        self.ceed_data.clear_all_callback = self._clear_all

        HighightButtonBehavior.init_class()

        self.ceed_data.create_file('')

        self.stage_factory.fbind('on_changed', self.changed_callback)
        self.function_factory.fbind('on_changed', self.changed_callback)
        for func in self.function_factory.funcs_inst_default.values():
            func.fbind('on_changed', self.changed_callback)
        self.shape_factory.fbind('on_changed', self.changed_callback)
        self.view_controller.fbind('on_changed', self.changed_callback)

        self.set_tittle()
        self.ceed_data.fbind('filename', self.set_tittle)
        self.ceed_data.fbind('config_changed', self.set_tittle)
        self.ceed_data.fbind('has_unsaved', self.set_tittle)
        self.ceed_data.fbind('read_only_file', self.set_tittle)

        try:
            self.view_controller.set_led_mode(
                self.view_controller.LED_mode_idle)
        except ImportError:
            pass

    def set_tittle(self, *largs):
        ''' Sets the title of the window using the currently running
        tab. This is called at 1Hz. '''
        star = ''
        if self.ceed_data.has_unsaved or self.ceed_data.config_changed:
            star = '*'

        read_only = ''
        if self.ceed_data.read_only_file:
            read_only = ' - Read Only'

        if self.ceed_data.filename:
            filename = ' - {}'.format(self.ceed_data.filename)
        else:
            filename = ' - Unnamed File'

        Window.set_title('Ceed v{}, CPL lab{}{}{}'.format(
            ceed.__version__, star, filename, read_only))

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

    def handle_exception(self, *largs, **kwargs):
        val = super(CeedApp, self).handle_exception(*largs, **kwargs)
        self.view_controller.request_stage_end()
        return val

    def clean_up(self):
        super(CeedApp, self).clean_up()
        if self.ceed_data is not None:
            if self.ceed_data.backup_event is not None:
                self.ceed_data.backup_event.cancel()
                self.ceed_data.backup_event = None
            self.ceed_data.clear_all_callback = None
        if self.stage_factory is not None:
            self.stage_factory.funbind('on_changed', self.changed_callback)
        if self.function_factory is not None:
            self.function_factory.funbind('on_changed', self.changed_callback)
        if self.shape_factory is not None:
            self.shape_factory.funbind('on_changed', self.changed_callback)
        if self.view_controller is not None:
            self.view_controller.funbind('on_changed', self.changed_callback)

        for func in self.function_factory.funcs_inst_default.values():
            func.funbind('on_changed', self.changed_callback)
        if self.remote_viewer:
            self.remote_viewer.stop_listener()
        CeedPlayer.exit_players()
        if self.view_controller is not None:
            self.view_controller.stop_process()
            self.view_controller.finish_stop_process()
        if self.ceed_data is not None:
            self.ceed_data.stop_experiment()

            self.ceed_data.funbind('filename', self.set_tittle)
            self.ceed_data.funbind('config_changed', self.set_tittle)
            self.ceed_data.funbind('has_unsaved', self.set_tittle)
            self.ceed_data.funbind('read_only_file', self.set_tittle)

        self.dump_app_settings_to_file()
        HighightButtonBehavior.uninit_class()


run_app = partial(run_cpl_app, CeedApp)
'''The function that starts the GUI and the entry point for
the main script.
'''

if __name__ == '__main__':
    run_app()
