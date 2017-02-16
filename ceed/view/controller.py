from multiprocessing import Process, Queue
import os
import sys
import traceback
try:
    from Queue import Empty
except ImportError:
    from queue import Empty

from kivy.event import EventDispatcher
from kivy.properties import NumericProperty, StringProperty, BooleanProperty
from kivy.clock import Clock
from kivy.compat import clock
from kivy.graphics import Color, Point
from kivy.app import App
from kivy.uix.behaviors.knspace import knspace

from cplcom.app import app_error
from cplcom.utils import json_dumps, json_loads

import ceed
from ceed.stage import StageFactory, StageDoneException
from ceed.storage.controller import DataSerializer, CeedData

if ceed.has_gui_control or ceed.is_view_inst:
    from kivy.core.window import Window


class ViewControllerBase(EventDispatcher):

    __settings_attrs__ = (
        'screen_width', 'screen_height', 'frame_rate',
        'use_software_frame_rate', 'cam_scale', 'cam_offset_x', 'cam_offset_y',
        'cam_rotation', 'output_count', 'screen_offset_x', 'preview',
        'fullscreen')

    screen_width = NumericProperty(1920)

    screen_height = NumericProperty(1080)

    screen_offset_x = NumericProperty(0)

    frame_rate = NumericProperty(60.)

    use_software_frame_rate = BooleanProperty(True)

    cam_scale = NumericProperty(1.)

    cam_offset_x = NumericProperty(0)

    cam_offset_y = NumericProperty(0)

    cam_rotation = NumericProperty(0)

    output_count = BooleanProperty(True)

    preview = BooleanProperty(True)

    fullscreen = BooleanProperty(False)

    stage_active = BooleanProperty(False)

    cpu_fps = NumericProperty(0)

    gpu_fps = NumericProperty(0)

    _original_fps = Clock._max_fps

    canvas_name = 'view_controller'

    current_canvas = None

    shape_views = {}

    tick_event = None

    tick_func = None

    count = 0

    _cpu_stats = {'last_call_t': 0, 'count': 0, 'tstart': 0}

    _flip_stats = {'last_call_t': 0, 'dt': []}

    flip_fps = 0

    serializer = None

    serializer_color = None

    queue_view_read = None

    queue_view_write = None

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(ViewControllerBase, self).__init__(**kwargs)
        for name in ViewControllerBase.__settings_attrs__:
            self.fbind(name, self.dispatch, 'on_changed')

    def on_changed(self, *largs):
        pass

    def save_state(self):
        d = {}
        for name in ViewControllerBase.__settings_attrs__:
            d[name] = getattr(self, name)
        return d

    def set_state(self, data):
        for name, value in data.items():
            setattr(self, name, value)

    def request_process_data(self, data_type, data):
        pass

    def start_stage(self, stage_name, canvas):
        if self.tick_event:
            raise TypeError('Cannot start new stage while stage is active')

        Clock._max_fps = 0
        self.tick_event = Clock.schedule_interval(self.tick_callback, 0)
        Window.fbind('on_flip', self.flip_callback)

        self.current_canvas = canvas
        self.shape_views = StageFactory.get_shapes_gl_color_instructions(
            canvas, self.canvas_name)
        self.tick_func = StageFactory.tick_stage(stage_name)

        self._flip_stats['last_call_t'] = self._cpu_stats['last_call_t'] = \
            self._cpu_stats['tstart'] = clock()

        if self.output_count:
            kwargs = App.get_running_app().app_settings['serializer']
            self.serializer = DataSerializer(**kwargs).get_bits(-1)
            with canvas:
                self.serializer_color = Color(
                    0, 0, 0, 1, group=self.canvas_name)
                Point(points=[0, 0], pointsize=.5, group=self.canvas_name)

    def end_stage(self):
        if not self.tick_event:
            return

        self.tick_event.cancel()
        Window.funbind('on_flip', self.flip_callback)
        Clock._max_fps = self._original_fps
        StageFactory.remove_shapes_gl_color_instructions(
            self.current_canvas, self.canvas_name)

        self.tick_func = self.tick_event = self.current_canvas = None
        self.shape_views = {}
        self.count = 0
        self._cpu_stats['count'] = 0
        del self._flip_stats['dt'][:]

        self.serializer_color = self.serializer = None

    def tick_callback(self, *largs):
        t = clock()
        stats = self._cpu_stats
        tdiff = t - stats['last_call_t']
        rate = float(self.frame_rate)

        stats['count'] += 1
        if t - stats['tstart'] >= 1:
            fps = stats['count'] / (t - stats['tstart'])
            self.request_process_data('CPU', fps)
            stats['tstart'] = t
            stats['count'] = 0

        if self.use_software_frame_rate and tdiff < 1 / rate:
            return

        stats['last_call_t'] = t

        shape_views = self.shape_views
        tick = self.tick_func
        self.count += 1

        try:
            tick.next()
            shape_values = tick.send(self.count / self.frame_rate)
        except StageDoneException:
            self.end_stage()
            return
        except Exception:
            self.end_stage()
            raise

        bits = 0
        if self.serializer:
            self.serializer.next()
            bits = self.serializer.send(self.count)
            r, g, b = bits & 0xFF, (bits & 0xFF00) >> 8, \
                (bits & 0xFF0000) >> 16
            self.serializer_color.rgba = r / 255., g / 255., b / 255., 1.

        values = StageFactory.fill_shape_gl_color_values(
            shape_views, shape_values)
        self.request_process_data('frame', (self.count, bits, values))

    def flip_callback(self, *largs):
        Window.on_flip()

        t = clock()
        self.request_process_data('frame_flip', (self.count, t))

        stats = self._flip_stats
        tdiff = t - stats['last_call_t']
        rate = float(self.frame_rate)

        stats['dt'].append(tdiff)
        stats['last_call_t'] = t
        if len(stats['dt']) >= rate:
            fps = self.flip_fps = len(stats['dt']) / sum(stats['dt'])
            self.request_process_data('GPU', fps)
            del stats['dt'][:]
        return True


class ViewSideViewControllerBase(ViewControllerBase):

    def start_stage(self, stage_name, canvas):
        self.prepare_view_window()
        return super(ViewSideViewControllerBase, self).start_stage(
            stage_name, canvas)

    def end_stage(self):
        val = super(ViewSideViewControllerBase, self).end_stage()
        self.queue_view_write.put_nowait(('end_stage', None))
        return val

    def request_process_data(self, data_type, data):
        self.queue_view_write.put_nowait((data_type, json_dumps(data)))

    def view_process_enter(self, read, write, settings, app_settings):
        def assign_settings(*largs):
            App.get_running_app().app_settings = app_settings
        for k, v in settings.items():
            setattr(self, k, v)
        from ceed.view.main import run_app
        self.queue_view_read = read
        self.queue_view_write = write
        Clock.schedule_once(assign_settings, 0)
        Clock.schedule_interval(self.view_read, .25)
        Clock.schedule_once(self.prepare_view_window, 0)

        try:
            run_app()
        except Exception as e:
            App.get_running_app().handle_exception(e, exc_info=sys.exc_info())

        self.queue_view_write.put_nowait(('eof', None))

    def handle_exception(self, exception, exc_info=None):
        if exc_info is not None:
            exc_info = ''.join(traceback.format_exception(*exc_info))
        self.queue_view_write.put_nowait(
            ('exception', json_dumps(exception, exc_info)))

    @app_error
    def view_read(self, *largs):
        read = self.queue_view_read
        write = self.queue_view_write
        while True:
            try:
                msg, value = read.get(False)
                if msg == 'eof':
                    App.get_running_app().stop()
                    break
                elif msg == 'config':
                    if self.tick_event:
                        raise Exception('Cannot configure while running stage')
                    CeedData.clear_existing_config_data()
                    CeedData.apply_config_data_dict(json_loads(value))
                elif msg == 'start_stage':
                    self.start_stage(value, App.get_running_app().root.canvas)
                elif msg == 'end_stage':
                    self.end_stage()
                elif msg == 'fullscreen':
                    Window.fullscreen = self.fullscreen = value
                write.put_nowait(('response', msg))
            except Empty:
                break

    def prepare_view_window(self, *largs):
        if Window.fullscreen != self.fullscreen or not self.fullscreen:
            Window.maximize()
            Window.size = self.screen_width, self.screen_height
            Window.left = self.screen_offset_x
            Window.fullscreen = self.fullscreen


def view_process_enter(*largs):
    return ViewController.view_process_enter(*largs)


class ControllerSideViewControllerBase(ViewControllerBase):

    view_process = None

    def request_stage_start(self, stage_name):
        CeedData.prepare_experiment(stage_name)
        if self.preview:
            self.start_stage(stage_name, knspace.painter.canvas)
            self.stage_active = True
        elif self.view_process:
            self.queue_view_read.put_nowait(
                ('config', json_dumps(CeedData.gather_config_data_dict())))
            self.queue_view_read.put_nowait(('start_stage', stage_name))
            self.stage_active = True

    def request_stage_end(self):
        if self.preview:
            self.end_stage()
            CeedData.stop_experiment()
            self.stage_active = False
        elif self.view_process:
            self.queue_view_read.put_nowait(('end_stage', None))

    def request_fullscreen(self, state):
        self.fullscreen = state
        if self.view_process:
            self.queue_view_read.put_nowait(('fullscreen', state))

    def request_process_data(self, data_type, data):
        if data_type == 'GPU':
            self.gpu_fps = data
        elif data_type == 'CPU':
            self.cpu_fps = data
        elif data_type == 'frame':
            CeedData.add_frame(*data)
        elif data_type == 'frame_flip':
            if data[0]:  # counts of zero is too early
                CeedData.add_frame_flip(*data)

    def start_process(self):
        if self.view_process:
            return

        settings = {name: getattr(self, name)
                    for name in ViewControllerBase.__settings_attrs__}
        r = self.queue_view_read = Queue()
        w = self.queue_view_write = Queue()
        os.environ['CEED_IS_VIEW'] = '1'
        self.view_process = process = Process(
            target=view_process_enter,
            args=(r, w, settings, App.get_running_app().app_settings))
        process.start()
        del os.environ['CEED_IS_VIEW']
        Clock.schedule_interval(self.controller_read, .25)

    def stop_process(self):
        if self.view_process:
            self.queue_view_read.put_nowait(('eof', None))

    def finish_stop_process(self):
        '''Can only be called after recieving eof from view side.
        '''
        if not self.view_process:
            return

        self.view_process.join()
        self.view_process = self.queue_view_read = self.queue_view_write = None
        Clock.unschedule(self.controller_read)

    def controller_read(self, *largs):
        write = self.queue_view_read
        read = self.queue_view_write
        while True:
            try:
                msg, value = read.get(False)
                if msg == 'eof':
                    self.finish_stop_process()
                    CeedData.stop_experiment()
                    self.stage_active = False
                    break
                elif msg == 'exception':
                    e, exec_info = json_loads(value)
                    App.get_running_app().handle_exception(
                        e, exc_info=exec_info)
                elif msg in ('GPU', 'CPU', 'frame', 'frame_flip'):
                    self.request_process_data(msg, json_loads(value))
                elif msg == 'response' and value == 'end_stage':
                    self.stage_active = False
                elif msg == 'end_stage':
                    CeedData.stop_experiment()
                    self.stage_active = False
            except Empty:
                break


def process_enter(*largs, **kwargs):
    ViewController.view_process_enter(*largs, **kwargs)

if ceed.is_view_inst:
    ViewController = ViewSideViewControllerBase()
else:
    ViewController = ControllerSideViewControllerBase()
