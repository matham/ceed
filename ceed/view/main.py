'''Ceed View App
=====================

The module that runs the :mod:`ceed.view` GUI for displaying the pixels on the
projector. This is run in a seperate process than the main server side GUI.
'''
import os
os.environ['SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS'] = '0'

from functools import partial
from os.path import join, dirname, isdir

from cplcom.app import run_app as run_cpl_app, app_error, CPLComApp
from cplcom.config import populate_dump_config

from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.resources import resource_add_path
from kivy.uix.behaviors.knspace import knspace, KNSpaceBehavior
from kivy.uix.widget import Widget
from kivy.garden.filebrowser import FileBrowser
from kivy.lang import Builder
from kivy.graphics.opengl import glEnable, GL_DITHER, glDisable
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.compat import string_types

import ceed
from ceed.view.controller import ViewController
from ceed.view.view_widgets import ViewRootFocusBehavior
from ceed.storage.controller import DataSerializer
from ceed.function import FunctionFactory

if ceed.is_view_inst or __name__ == '__main__':
    from kivy.core.window import Window

__all__ = ('CeedViewApp', 'run_app')


kv = '''
<ViewRootWidget>:
    canvas:
        Color:
            rgba: 0, 0, 0, 1
        Rectangle:
            size: self.size
            pos: 0, 0
'''

class ViewRootWidget(ViewRootFocusBehavior, Widget):
    pass


class CeedViewApp(CPLComApp):
    '''The app which runs the GUI.
    '''

    @classmethod
    def get_config_classes(cls):
        d = super(CeedViewApp, cls).get_config_classes()
        d['view'] = ViewController
        d['serializer'] = DataSerializer
        d['function'] = FunctionFactory
        return d

    def init_load(self):
        pass

    def build(self):
        Builder.load_string(kv)
        return ViewRootWidget()

    def on_start(self):
        glDisable(GL_DITHER)
        Window.clearcolor = (0, 0, 0, 1)
        Window.minimize()
        self.root.focus = True
        Window.show_cursor = False

    def _ask_close(self, *largs, **kwargs):
        return False

    def handle_exception(self, exception, exc_info=None, event=None, obj=None,
                         error_indicator='', level='error', *largs):
        '''Should be called whenever an exception is caught in the app.

        :parameters:

            `exception`: string
                The caught exception (i.e. the ``e`` in
                ``except Exception as e``)
            `exc_info`: stack trace
                If not None, the return value of ``sys.exc_info()``. It is used
                to log the stack trace.
            `event`: :class:`moa.threads.ScheduledEvent` instance
                If not None and the exception originated from within a
                :class:`moa.threads.ScheduledEventLoop`, it's the
                :class:`moa.threads.ScheduledEvent` that caused the execution.
            `obj`: object
                If not None, the object that caused the exception.
        '''
        if isinstance(exc_info, string_types):
            self.get_logger().error(exception)
            self.get_logger().error(exc_info)
        else:
            self.get_logger().error(exception, exc_info=exc_info)
        if obj is None:
            err = exception
        else:
            err = '{} from {}'.format(exception, obj)
        ViewController.handle_exception(err, exc_info)

    def get_logger(self):
        return Logger


def _cleanup(app):
    pass

run_app = partial(run_cpl_app, CeedViewApp, _cleanup)
'''The function that starts the GUI and the entry point for
the main script.
'''

if __name__ == '__main__':
    run_app()