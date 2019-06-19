import os
os.environ['KIVY_EVENTLOOP'] = 'trio'
os.environ['KIVY_USE_DEFAULTCONFIG'] = '1'
import pytest
import trio
import gc
from collections import defaultdict

from kivy.config import Config
Config.set('graphics', 'width', '1200')
Config.set('graphics', 'height', '800')
for items in Config.items('input'):
    Config.remove_option('input', items[0])

from ceed.main import CeedApp, _cleanup

file_count = defaultdict(int)
kv_loaded = False


@pytest.fixture()
def temp_file(tmp_path):
    def temp_file_gen(fname):
        i = file_count[fname]
        file_count[fname] += 1

        root, ext = os.path.splitext(fname)
        return str(tmp_path / '{}_{}{}'.format(root, i, ext))

    return temp_file_gen


class CeedTestApp(CeedApp):

    app_has_started = False

    def __init__(self, ini_file, **kwargs):
        self._ini_config_filename = ini_file
        self._data_path = os.path.dirname(ini_file)
        super(CeedTestApp, self).__init__(**kwargs)

        def started_app(*largs):
            self.app_has_started = True
        self.fbind('on_start', started_app)

    def load_app_kv(self):
        global kv_loaded
        if kv_loaded:
            return

        super(CeedTestApp, self).load_app_kv()
        kv_loaded = True

    def check_close(self):
        super(CeedTestApp, self).check_close()
        return True

    def handle_exception(self, msg, exc_info=None, error_indicator='',
                         level='error', *largs):
        super(CeedApp, self).handle_exception(
            msg, exc_info, error_indicator, level, *largs)

        if isinstance(exc_info, str):
            self.get_logger().error(msg)
            self.get_logger().error(exc_info)
        elif exc_info is not None:
            tp, value, tb = exc_info
            try:
                if value is None:
                    value = tp()
                if value.__traceback__ is not tb:
                    raise value.with_traceback(tb)
                raise value
            finally:
                value = None
                tb = None
        elif level in ('error', 'exception'):
            raise Exception(msg)

    async def wait_clock_frames(self, n, sleep_time=0.1):
        from kivy.clock import Clock
        frames_start = Clock.frames
        while Clock.frames < frames_start + n:
            await trio.sleep(sleep_time)

    def get_widget_pos_pixel(self, widget, pos):
        from kivy.graphics import (
            Translate, Fbo, ClearColor, ClearBuffers, Scale)

        canvas_parent_index = -2
        if widget.parent is not None:
            canvas_parent_index = widget.parent.canvas.indexof(widget.canvas)
            if canvas_parent_index > -1:
                widget.parent.canvas.remove(widget.canvas)

        w, h = int(widget.width), int(widget.height)
        fbo = Fbo(size=(w, h), with_stencilbuffer=True)

        with fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()

        fbo.add(widget.canvas)
        fbo.draw()
        pixels = fbo.pixels
        fbo.remove(widget.canvas)

        if widget.parent is not None and canvas_parent_index > -1:
            widget.parent.canvas.insert(canvas_parent_index, widget.canvas)

        values = []
        for x, y in pos:
            x = int(x)
            y = int(y)
            i = y * w * 4 + x * 4
            values.append(tuple(pixels[i:i + 4]))

        return values


@pytest.fixture()
async def ceed_app(request, nursery, temp_file, tmp_path):
    from kivy.core.window import Window
    Window.create_window()
    Window.register()
    Window.initialized = True
    Window.canvas.clear()

    import ceed.view.controller
    ceed.view.controller.ignore_vpixx_import_error = True

    app = CeedTestApp(
        json_config_path=temp_file('config.yaml'),
        ini_file=temp_file('config.ini'))
    app.ceed_data.root_path = str(tmp_path)
    nursery.start_soon(app.async_run)

    def fin():
        from kivy.base import stopTouchApp
        stopTouchApp()
        for child in Window.children[:]:
            Window.remove_widget(child)
        _cleanup(app)
        gc.collect()

    request.addfinalizer(fin)

    while not app.app_has_started:
        await trio.sleep(.1)
    await app.wait_clock_frames(5)
    return app
