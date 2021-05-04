"""Ceed App
=====================

The main module that runs the Ceed GUI.
"""
import ceed

from functools import partial
from os.path import join, dirname, expanduser
import time

from base_kivy_app.app import BaseKivyApp, run_app as run_cpl_app
from base_kivy_app.graphics import HighightButtonBehavior, BufferImage
from more_kivy_app.app import report_exception_in_app
import cpl_media

from kivy.lang import Builder
from kivy.factory import Factory
from kivy.properties import ObjectProperty, StringProperty

import ceed.graphics
Builder.load_file(join(dirname(__file__), 'graphics', 'graphics.kv'))
import ceed.function.plugin
import ceed.shape
import ceed.stage
import ceed.function.func_widgets
import ceed.stage.stage_widgets
import ceed.shape.shape_widgets
import ceed.view.view_widgets
import ceed.storage.storage_widgets

from ceed.function import FunctionFactoryBase, register_all_functions, \
    register_external_functions
from ceed.stage import StageFactoryBase, remove_shapes_upon_deletion, \
    register_external_stages, register_all_stages
from ceed.stage.stage_widgets import StageList
from ceed.view.controller import ControllerSideViewControllerBase
from ceed.storage.controller import CeedDataWriterBase, DataSerializerBase
from ceed.graphics import CeedDragNDrop
from ceed.player import CeedPlayer
from ceed.function.func_widgets import FuncList
from ceed.shape.shape_widgets import CeedPainter, ShapeList, ShapeGroupList
from ceed.view.view_widgets import MEAArrayAlign

from kivy.core.window import Window

__all__ = ('CeedApp', 'run_app')


class CeedApp(BaseKivyApp):
    """The app which runs the main Ceed GUI.
    """

    _config_props_ = (
        'last_directory', 'external_function_plugin_package',
        'external_stage_plugin_package'
    )

    _config_children_ = {
        'function': 'function_factory', 'view': 'view_controller',
        'data': 'ceed_data', 'serializer': 'data_serializer',
        'player': 'player',
    }

    last_directory = StringProperty('~')
    """The last directory opened in the GUI.
    """

    external_function_plugin_package: str = ''
    """The name of an external function plugin package that contains additional
    functions to be displayed in the GUI to the user.

    See :mod:`~ceed.function.plugin` for details.
    """

    external_stage_plugin_package: str = ''
    """The name of an external stage plugin package that contains additional
    stages to be displayed in the GUI to the user.

    See :mod:`~ceed.stage.plugin` for details.
    """

    kv_loaded = False
    """For tests, we don't want to load kv multiple times so we only load kv if
    it wasn't loaded before.
    """

    yesno_prompt = ObjectProperty(None, allownone=True)
    '''Stores a instance of :class:`YesNoPrompt` that is automatically created
    by this app class. That class is described in ``base_kivy_app/graphics.kv``
    and shows a prompt with yes/no options and callback.
    '''

    function_factory: FunctionFactoryBase = None
    """The :class:`~ceed.function.FunctionFactoryBase` that contains all the
    functions shown in the GUI.
    """

    player: CeedPlayer = None
    """The :class:`ceed.player.CeedPlayer` used to play the video camera and
    record the images to disk.
    """

    view_controller: ControllerSideViewControllerBase = None
    """The :class:`~ceed.view.controller.ControllerSideViewControllerBase` used
    to run the experiment and display the stages.
    """

    ceed_data: CeedDataWriterBase = None
    """The :class:`~ceed.storage.controller.CeedDataWriterBase` used to load
    and save the data to disk.
    """

    data_serializer: DataSerializerBase = None
    """The :class:`~ceed.storage.controller.DataSerializerBase` used to
    generate the corner pixel values for Ceed-MCS temporal synchronization.
    """

    stage_factory: StageFactoryBase = None
    """The :class:`~ceed.stage.StageFactoryBase` that contains all the
    stages shown in the GUI.
    """

    shape_factory: CeedPainter = ObjectProperty(None, rebind=True)
    """The :class:`~ceed.shape.shape_widgets.CeedPainter` used to draw shapes
    and contains all the shapes shown in the GUI.
    """

    agreed_discard = False

    drag_controller: CeedDragNDrop = ObjectProperty(None, rebind=True)
    """The :class:`~ceed.graphics.CeedDragNDrop` used for dragging and
    dropping widgets on in the GUI.
    """

    stages_container: StageList = ObjectProperty(None, rebind=True)
    """The :class:`~ceed.stage.stage_widgets.StageList` widget that contains
    all the root stages' widgets in the GUI.
    """

    funcs_container: FuncList = ObjectProperty(None, rebind=True)
    """The :class:`~ceed.function.func_widgets.FuncList` widget that contains
    all the root functions' widgets in the GUI.
    """

    shapes_container: ShapeList = ObjectProperty(None, rebind=True)
    """The :class:`~ceed.shape.shape_widgets.ShapeList` widget that contains
    all the shapes' widgets in the GUI.
    """

    shape_groups_container = ObjectProperty(
        None, rebind=True)  # type: ShapeGroupList
    """The :class:`~ceed.shape.shape_widgets.ShapeGroupList` widget that
    contains all the shape groups' widgets in the GUI.
    """

    pinned_graph = None
    """PinnedGraph widget into which the experiment preview graph may be pinned.

    When pinned, it's displayed not as a popup, but as a flat widget.
    """

    mea_align_widget: MEAArrayAlign = ObjectProperty(None, rebind=True)
    """The :class:`~ceed.view.view_widgets.MEAArrayAlign` widget used to align
    the MEA grid to the camera.
    """

    central_display: BufferImage = ObjectProperty(None, rebind=True)
    """The :class:`~base_kivy_app.graphics.BufferImage` widget into which the
    camera widget is drawn.
    """

    _processing_error = False

    def __init__(self, open_player_thread=True, **kwargs):
        self.drag_controller = CeedDragNDrop()
        self.function_factory = FunctionFactoryBase()
        register_all_functions(self.function_factory)

        self.stage_factory = StageFactoryBase(
            function_factory=self.function_factory, shape_factory=None)
        register_all_stages(self.stage_factory)
        self.player = CeedPlayer(open_player_thread=open_player_thread)
        self.view_controller = ControllerSideViewControllerBase()
        self.ceed_data = CeedDataWriterBase()
        self.data_serializer = DataSerializerBase()
        super(CeedApp, self).__init__(**kwargs)
        self.load_app_settings_from_file()
        self.apply_app_settings()

    def load_app_kv(self):
        """Loads the app's kv files, if not yet loaded.
        """
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
        self.yesno_prompt = Factory.FlatYesNoPrompt()
        self.player.create_widgets()

        root = Factory.get('MainView')()
        return super(CeedApp, self).build(root)

    def _clear_all(self):
        self.funcs_container.clear_all()
        self.stages_container.clear_all()

    def on_start(self):
        if self.external_function_plugin_package:
            register_external_functions(
                self.function_factory,
                self.external_function_plugin_package)

        if self.external_stage_plugin_package:
            register_external_stages(
                self.stage_factory,
                self.external_stage_plugin_package)

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
        for stage in self.stage_factory.stages_inst_default.values():
            stage.fbind('on_changed', self.changed_callback)
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

        self.view_controller.set_led_mode(self.view_controller.LED_mode_idle)

    def set_tittle(self, *largs):
        """Periodically called by the Kivy Clock to update the title.
        """
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
        """Callback bound to anything that can change the Ceed data to indicate
        whether it needs to be re-saved.
        """
        self.ceed_data.config_changed = True

    def check_close(self):
        if self.view_controller.stage_active or self.ceed_data.data_thread:
            self._close_message = 'Cannot close during an experiment.'
            return False
        self.player.stop()
        self.view_controller.stop_process()
        self.view_controller.finish_stop_process()
        if not self.ceed_data.ui_close(app_close=True):
            self._close_message = ''
            return False
        return True

    def handle_exception(self, *largs, **kwargs):
        processing = self._processing_error
        self._processing_error = True
        val = super(CeedApp, self).handle_exception(*largs, **kwargs)
        if not processing:
            self.view_controller.request_stage_end()
            self._processing_error = False
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

        for stage in self.stage_factory.stages_inst_default.values():
            stage.funbind('on_changed', self.changed_callback)
        for func in self.function_factory.funcs_inst_default.values():
            func.funbind('on_changed', self.changed_callback)

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
        self.player.clean_up()
        HighightButtonBehavior.uninit_class()


def run_app():
    """The function that starts the main Ceed GUI and the entry point for
    the main script.
    """
    cpl_media.error_callback = report_exception_in_app
    return run_cpl_app(CeedApp)
