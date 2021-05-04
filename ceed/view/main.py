'''Ceed Viewer side App
=======================

Ceed starts a second process when it runs an actual experiment and all the
shapes and their intensities that are ultimately displayed on the projector are
displayed in this second process, full-screen.

This module is the Ceed app that the second process runs. It shares most of the
same configuration as the main :class:`~ceed.main.CeedApp`, but the GUI
is very simple because it only plays the shapes and forwards any keyboard
interactions to the main Ceed process.
'''
import os
os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'

from functools import partial

from base_kivy_app.app import run_app as run_cpl_app, app_error, BaseKivyApp

from kivy.lang import Builder
from kivy.graphics import Fbo, Scale, Translate, ClearColor, ClearBuffers
from kivy.uix.widget import Widget
from kivy.graphics.opengl import GL_DITHER, glDisable
from kivy.logger import Logger
from kivy.compat import string_types

import ceed
from ceed.function import FunctionFactoryBase, register_all_functions, \
    register_external_functions
from ceed.view.controller import ViewSideViewControllerBase
from ceed.view.view_widgets import ViewRootFocusBehavior
from ceed.storage.controller import DataSerializerBase, CeedDataWriterBase
from ceed.stage import StageFactoryBase, register_external_stages, \
    register_all_stages
from ceed.shape import CeedPaintCanvasBehavior

if ceed.is_view_inst or __name__ == '__main__':
    from kivy.core.window import Window

__all__ = ('CeedViewApp', 'run_app', 'ViewRootWidget')


kv = '''
<ViewRootWidget>:
    Widget:
        size: root.width, root.height
        canvas.before:
            Color:
                rgba: [0, 0, 0, 1]
            Rectangle:
                pos: self.pos
                size: self.size
        Widget:
            size: root.size
            id: display_canvas
'''


class ViewRootWidget(ViewRootFocusBehavior, Widget):
    """The widget that is the root kivy widget in the app.
    """
    pass


class CeedViewApp(BaseKivyApp):
    """The app which runs the GUI in the second Ceed process.
    """

    _config_props_ = (
        'external_function_plugin_package',
        'external_stage_plugin_package')

    _config_children_ = {
        'view': 'view_controller', 'data': 'ceed_data',
        'serializer': 'data_serializer', 'function': 'function_factory',
    }

    view_controller: ViewSideViewControllerBase = None
    """Same as :attr:`~ceed.main.CeedApp.view_controller`.
    """

    data_serializer: DataSerializerBase = None
    """Same as :attr:`~ceed.main.CeedApp.data_serializer`.
    """

    function_factory: FunctionFactoryBase = None
    """Same as :attr:`~ceed.main.CeedApp.function_factory`.
    """

    stage_factory: StageFactoryBase = None
    """Same as :attr:`~ceed.main.CeedApp.stage_factory`.
    """

    shape_factory: CeedPaintCanvasBehavior = None
    """Same as :attr:`~ceed.main.CeedApp.shape_factory`.
    """

    ceed_data: CeedDataWriterBase = None
    """Same as :attr:`~ceed.main.CeedApp.ceed_data`.
    """

    external_function_plugin_package: str = ''
    """Same as :attr:`~ceed.main.CeedApp.external_function_plugin_package`.
    """

    external_stage_plugin_package: str = ''
    """Same as :attr:`~ceed.main.CeedApp.external_stage_plugin_package`.
    """

    def __init__(self, **kwargs):
        self.view_controller = ViewSideViewControllerBase()
        self.ceed_data = CeedDataWriterBase()
        self.data_serializer = DataSerializerBase()

        self.function_factory = FunctionFactoryBase()
        register_all_functions(self.function_factory)

        self.shape_factory = CeedPaintCanvasBehavior()
        self.stage_factory = StageFactoryBase(
            function_factory=self.function_factory,
            shape_factory=self.shape_factory)
        register_all_stages(self.stage_factory)

        super(CeedViewApp, self).__init__(**kwargs)

    def init_load(self):
        pass

    def get_display_canvas(self):
        """Gets the canvas of the widget to which all the shapes graphics will
        be added.
        """
        return self.root.ids.display_canvas.canvas

    def build(self):
        Builder.load_string(kv)
        widget = ViewRootWidget()
        return widget

    def get_root_pixels(self):
        """Returns all the pixels values of the widget containing the shapes,
        as well as the size of that widget.

        This is how you can save an image of whatever is currently displayed on
        screen.
        """
        widget = self.root.ids.display_canvas

        canvas_parent_index = widget.parent.canvas.indexof(widget.canvas)
        if canvas_parent_index > -1:
            widget.parent.canvas.remove(widget.canvas)

        fbo = Fbo(size=widget.size, with_stencilbuffer=True)

        with fbo:
            ClearColor(0, 0, 0, 1)
            ClearBuffers()
            Scale(1, -1, 1)
            Translate(0, -widget.height, 0)

        fbo.add(widget.canvas)
        fbo.draw()
        pixels = fbo.pixels
        fbo.remove(widget.canvas)

        if canvas_parent_index > -1:
            widget.parent.canvas.insert(canvas_parent_index, widget.canvas)
        return pixels, widget.size

    def on_start(self):
        glDisable(GL_DITHER)
        Window.clearcolor = (0, 0, 0, 1)
        # Window.minimize()
        self.root.focus = True
        Window.show_cursor = False

        if self.external_function_plugin_package:
            register_external_functions(
                self.function_factory,
                self.external_function_plugin_package)

        if self.external_stage_plugin_package:
            register_external_stages(
                self.stage_factory,
                self.external_stage_plugin_package)

    def ask_cannot_close(self, *largs, **kwargs):
        return False

    def handle_exception(
            self, exception, exc_info=None, event=None, obj=None,
            level='error', *largs):
        if isinstance(exc_info, string_types):
            self.get_logger().error(exception)
            self.get_logger().error(exc_info)
        else:
            self.get_logger().error(exception, exc_info=exc_info)
        if obj is None:
            err = exception
        else:
            err = '{} from {}'.format(exception, obj)
        self.view_controller.handle_exception(err, exc_info)

    def get_logger(self):
        return Logger


run_app = partial(run_cpl_app, CeedViewApp)
'''The function that starts the GUI and runs :class:`CeedViewApp`.

This is the entry point for the main script.
'''

if __name__ == '__main__':
    run_app()
