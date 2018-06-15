'''View Controller
======================

Displays the preview or live pixel output of the experiment.
'''
import multiprocessing as mp
import os
from math import radians
import sys
from fractions import Fraction
import traceback
from collections import defaultdict
from functools import partial
from threading import Thread
# import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
try:
    from Queue import Empty
except ImportError:
    from queue import Empty
import uuid
from ffpyplayer.pic import Image, SWScale

from kivy.event import EventDispatcher
from kivy.properties import NumericProperty, StringProperty, BooleanProperty, \
    ObjectProperty, OptionProperty, AliasProperty
from kivy.clock import Clock
from kivy.compat import clock, PY2
from kivy.graphics import Color, Point, Fbo, Rectangle, Scale, PushMatrix, \
    PopMatrix, Translate
from kivy.graphics.texture import Texture
from kivy.app import App
from kivy.graphics.transformation import Matrix
from kivy.uix.behaviors.knspace import knspace

from cplcom.app import app_error
from cplcom.utils import yaml_dumps, yaml_loads

import ceed
from ceed.stage import StageDoneException

if ceed.has_gui_control or ceed.is_view_inst:
    from kivy.core.window import Window

try:
    from pypixxlib import _libdpx as libdpx
    from pypixxlib.propixx import PROPixx
    from pypixxlib.propixx import PROPixxCTRL
except ImportError:
    libdpx = PROPixx = PROPixxCTRL = None

__all__ = (
    'ViewControllerBase', 'ViewSideViewControllerBase',
    'view_process_enter', 'ControllerSideViewControllerBase')

_get_app = App.get_running_app


class ViewControllerBase(EventDispatcher):
    '''A base class for visualizing the output of a :mod:`ceed.stage` on the
    projector or to preview it in the main GUI.

    The usage of ceed is to run a GUI in which stages, shapes, and functions
    are designed. Subsequently, the stage is played on the projector or
    previewed in the main GUI and displays shapes varying with intensity as
    time progresses, as designed.

    When the stage is played as a preview in the main GUI, all the code is
    executed within the main process. In this case the controller is a
    :class:`ControllerSideViewControllerBase` instance.

    When the stage is played for real, it is played in a second process in
    a second window which can be displayed on the projector window. In
    this case, the controller in the second process is a
    :class:`ViewSideViewControllerBase` instance while in the main GUI it
    is a :class:`ControllerSideViewControllerBase` instance. Data is constantly
    sent between the two processes, specifically, the second process is
    initialized with the data to be displayed at the start. Once the playing
    starts, the client continuously sends data back to the main GUI for
    processing and storage.

    This class controls all aspects of how the data is presented, e.g. whether
    the window is full screen, the various modes, etc.

    :Events:

        `on_changed`:
            Triggered whenever a configuration option of the class is changed.
    '''

    __settings_attrs__ = (
        'screen_width', 'screen_height', 'frame_rate',
        'use_software_frame_rate', 'cam_rotation', 'cam_scale',
        'cam_center_x', 'cam_center_y', 'output_count', 'screen_offset_x',
        'preview', 'fullscreen', 'video_mode', 'LED_mode', 'LED_mode_idle')

    screen_width = NumericProperty(1920)
    '''The screen width on which the data is played. This is the full-screen
    size.
    '''

    screen_height = NumericProperty(1080)
    '''The screen height on which the data is played. This is the full-screen
    size.
    '''

    screen_offset_x = NumericProperty(0)
    '''When there are multiple monitors, the window on which the data is played
    is controlled by the position of the screen. E.g. to set it on the right
    screen of two screens, each 1920 pixel wide and with the main screen being
    on the left. Then the :attr:`screen_offset_x` should be set to ``1920``.
    '''

    frame_rate = NumericProperty(120.)
    '''The frame rate at which the data is played. This should match the
    currently selected monitor's refresh rate.
    '''

    use_software_frame_rate = BooleanProperty(False)
    '''Depending on the GPU, the software is unable to render faster than the
    GPU refresh rate. In that case, :attr:`frame_rate`, should match the value
    that the GPU is playing at and this should be False.

    If the GPU isn't forcing a frame rate. Then this should be True and
    :attr:`frame_rate` should be the desired frame rate.

    One can tell whether the GPU is forcing a frame rate by setting
    :attr:`frame_rate` to a large value and setting
    :attr:`use_software_frame_rate` to False and seeing what the resultant
    frame rate is. If it isn't capped at some value, e.g. 120Hz, it means that
    the GPU isn't forcing it.
    '''

    cam_scale = NumericProperty(1.)
    '''The scaling factor of the background image.
    '''

    cam_center_x = NumericProperty(0)
    '''The x center of the background image.
    '''

    cam_center_y = NumericProperty(0)
    '''The y center of the background image.
    '''

    cam_rotation = NumericProperty(0)
    '''The rotation angle of the background image.
    '''

    output_count = BooleanProperty(True)
    '''Whether the corner pixel is used to output frame information on the
    PROPixx controller IO pot. If True,
    :class:`ceed.storage.controller.DataSerializerBase` is used to set the 24 
    bits of the corner pixel.
    '''

    preview = BooleanProperty(True)
    '''When run, if True, the data is played in the main GUI. When False,
    the data id played on the second window.
    '''

    fullscreen = BooleanProperty(False)
    '''Whether the second window should run in fullscreen mode. In fullscreen
    mode the window has no borders.
    '''

    stage_active = BooleanProperty(False)
    '''True when a stage is playing. Read-only.
    '''

    cpu_fps = NumericProperty(0)
    '''The estimated CPU frames-per-second of the window playing the data.
    '''

    gpu_fps = NumericProperty(0)
    '''The estimated GPU frames-per-second of the window playing the data.
    '''

    propixx_lib = BooleanProperty(False)
    '''True when the propixx python library is available. Read-only.
    '''

    video_modes = ['RGB', 'RB3D', 'RGB240', 'RGB180', 'QUAD4X', 'QUAD12X',
                   'GREY3X']
    '''The video modes that the PROPixx projector can be set to.
    '''

    led_modes = {'RGB': 0, 'GB': 1, 'RB': 2, 'B': 3, 'RG': 4, 'G': 5, 'R': 6,
                 'none': 7}
    '''The color modes the PROPixx projector can be set to. It determines which
    of the RGB LEDs are turned OFF.
    '''

    video_mode = StringProperty('RGB')
    '''The current video mode from the :attr:`video_modes`.
    '''

    LED_mode = StringProperty('RGB')
    '''The LED mode the projector is set to during the experiment.
    Its value is from the :attr:`led_modes`.
    '''

    LED_mode_idle = StringProperty('RGB')
    '''The LED mode the projector is set to before/after the experiment.
    Its value is from the :attr:`led_modes`.
    '''

    def _get_do_quad_mode(self):
        return self.video_mode.startswith('QUAD')

    do_quad_mode = AliasProperty(
        _get_do_quad_mode, None, cache=True, bind=('video_mode', ))
    '''Whether the video mode is a quad mode. Read-only.
    '''

    _original_fps = Clock._max_fps if not os.environ.get(
        'KIVY_DOC_INCLUDE', None) else 0
    '''Original kivy clock fps, so we can set it back.
    '''

    canvas_name = 'view_controller'
    '''Name used to add graphics instructions to the kivy canvas for easy
    removal later by name.
    '''

    current_canvas = None
    '''The last canvas used on which the shapes graphics and color instructions
    was added.
    '''

    shape_views = []
    '''List of kivy graphics instructions added to the :attr:`current_canvas`.
    '''

    tick_event = None
    '''The kivy clock event that updates the colors on every frame.
    '''

    tick_func = None
    '''The iterator that updates the colors on every frame.
    '''

    count = 0
    '''The current frame count.
    '''

    def _get_effective_rate(self):
        if self.video_mode == 'QUAD4X':
            return self.frame_rate * 4
        elif self.video_mode == 'QUAD12X':
            return self.frame_rate * 12
        return self.frame_rate

    effective_frame_rate = AliasProperty(
        _get_effective_rate, None, cache=True,
        bind=('video_mode', 'frame_rate'))
    '''The actual frame rate at which the projector is updated. E.g. in
    ``'QUAD4X'`` :attr:`video_mode` it is updated at 4 * 120Hz = 480Hz.

    It is read only and automatically computed.
    '''

    _cpu_stats = {'last_call_t': 0, 'count': 0, 'tstart': 0}

    _flip_stats = {'last_call_t': 0, 'dt': []}

    flip_fps = 0
    '''The GPU fps.
    '''

    serializer = None
    '''The :meth:`ceed.storage.controller.DataSerializerBase.get_bits`
    generator instance that generates the corner pixel value.
    '''

    serializer_tex = None
    '''The kivy texture that displays the corner pixel value.
    '''

    queue_view_read = None
    '''The queue used by the view side to receive messages from the main GUI
    controller side.
    '''

    queue_view_write = None
    '''The queue used by the view side to write messages to the main GUI
    controller side.
    '''

    _scheduled_pos_restore = False

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(ViewControllerBase, self).__init__(**kwargs)
        for name in ViewControllerBase.__settings_attrs__:
            self.fbind(name, self.dispatch, 'on_changed')
        self.propixx_lib = libdpx is not None
        self.shape_views = []

    def _restore_cam_pos(self):
        if self._scheduled_pos_restore:
            return

        self._scheduled_pos_restore = True
        center = self.cam_center_x, self.cam_center_y
        scale = self.cam_scale
        rotation = self.cam_rotation

        def restore_state(*largs):
            self.cam_rotation = rotation
            self.cam_scale = scale
            self.cam_center_x, self.cam_center_y = center
            self._scheduled_pos_restore = False

        Clock.schedule_once(restore_state, -1)

    def apply_settings_attrs(self, config):
        config = dict(config)
        rotation = config.pop('cam_rotation', self.cam_rotation)
        scale = config.pop('cam_scale', self.cam_scale)
        center_x = config.pop('cam_center_x', self.cam_center_x)
        center_y = config.pop('cam_center_y', self.cam_center_y)
        for k, v in config.items():
            setattr(self, k, v)

        self.cam_rotation = rotation
        self.cam_scale = scale
        self.cam_center_x = center_x
        self.cam_center_y = center_y

    def on_changed(self, *largs):
        pass

    def request_process_data(self, data_type, data):
        '''Called by the client that displays the shapes when it needs to
        update the controller with some data.
        '''
        pass

    def add_graphics(self, canvas, black_back=False):
        '''Adds all the graphics required to visualize the shapes to the
        canvas.
        '''
        _get_app().stage_factory.remove_shapes_gl_color_instructions(
            canvas, self.canvas_name)
        self.shape_views = []
        w, h = self.screen_width, self.screen_height

        if black_back:
            with canvas:
                Color(0, 0, 0, 1, group=self.canvas_name)
                Rectangle(size=(w, h), group=self.canvas_name)

        if self.do_quad_mode:
            half_w = w // 2
            half_h = h // 2

            for (x, y) in ((0, 1), (1, 1), (0, 0), (1, 0)):
                with canvas:
                    PushMatrix(group=self.canvas_name)
                    Translate(x * half_w, y * half_h, group=self.canvas_name)
                    s = Scale(group=self.canvas_name)
                    s.x = s.y = 0.5
                    s.origin = 0, 0
                instructs = _get_app(
                    ).stage_factory.get_shapes_gl_color_instructions(
                        canvas, self.canvas_name)
                with canvas:
                    PopMatrix(group=self.canvas_name)
                self.shape_views.append(instructs)
        else:
            self.shape_views = [
                _get_app().stage_factory.get_shapes_gl_color_instructions(
                    canvas, self.canvas_name)]

        if self.output_count and not self.serializer_tex:
            with canvas:
                Color(1, 1, 1, 1, group=self.canvas_name)
                tex = self.serializer_tex = Texture.create(size=(1, 1))
                tex.mag_filter = 'nearest'
                tex.min_filter = 'nearest'
                Rectangle(texture=tex, pos=(0, h - 1), size=(1, 1),
                          group=self.canvas_name)

    def get_all_shape_values(self, stage_name, frame_rate):
        '''For every shape in the stage ``stage_name`` it samples the shape
        at the frame rate and returns a list of intensity values for the shape
        for each frame.
        '''
        '''frame_rate is not :attr:`frame_rate` bur rather the rate at which we
        sample the functions.
        '''
        tick = _get_app().stage_factory.tick_stage(stage_name)
        # the sampling rate at which we sample the functions
        frame_rate = int(frame_rate)

        obj_values = defaultdict(list)
        count = 0
        while True:
            count += 1

            try:
                next(tick)
                shape_values = tick.send(Fraction(count, frame_rate))
            except StageDoneException:
                break

            values = _get_app().stage_factory.fill_shape_gl_color_values(
                None, shape_values)
            for name, r, g, b, a in values:
                obj_values[name].append((r, g, b, a))
        return obj_values

    def start_stage(self, stage_name, canvas):
        '''Starts the stage. It adds the graphics instructions to the canvas
        and starts playing the shapes.
        '''
        if self.tick_event:
            raise TypeError('Cannot start new stage while stage is active')

        Clock._max_fps = 0
        self.tick_event = Clock.schedule_interval(self.tick_callback, 0)
        Window.fbind('on_flip', self.flip_callback)

        self.current_canvas = canvas
        self.tick_func = _get_app().stage_factory.tick_stage(stage_name)

        self._flip_stats['last_call_t'] = self._cpu_stats['last_call_t'] = \
            self._cpu_stats['tstart'] = clock()

        if self.output_count:
            self.serializer = App.get_running_app().data_serializer.get_bits(
                -1, uuid.uuid4().bytes)

        self.add_graphics(canvas)

    def end_stage(self):
        '''Ends the stage if one is playing.
        '''
        if not self.tick_event:
            return

        self.tick_event.cancel()
        Window.funbind('on_flip', self.flip_callback)
        Clock._max_fps = self._original_fps
        _get_app().stage_factory.remove_shapes_gl_color_instructions(
            self.current_canvas, self.canvas_name)

        self.tick_func = self.tick_event = self.current_canvas = None
        self.shape_views = []
        self.count = 0
        self._cpu_stats['count'] = 0
        del self._flip_stats['dt'][:]

        self.serializer_tex = None
        self.serializer = None

    def tick_callback(self, *largs):
        '''Called before every CPU frame to handle any processing work.

        When graphics need to be updated this method will update them
        '''
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

        tick = self.tick_func
        if self.video_mode == 'QUAD4X':
            projections = [None, ] * 4
            views = self.shape_views
        elif self.video_mode == 'QUAD12X':
            projections = (['r', ] * 4) + (['g', ] * 4) + (['b', ] * 4)
            views = [view for _ in range(4) for view in self.shape_views]
        else:
            projections = [None, ]
            views = self.shape_views
        effective_rate = int(self.effective_frame_rate)

        for shape_views, proj in zip(views, projections):
            self.count += 1

            try:
                next(tick)
                shape_values = tick.send(Fraction(self.count, effective_rate))
            except StageDoneException:
                self.end_stage()
                return
            except Exception:
                self.end_stage()
                raise

            if self.serializer:
                next(self.serializer)
                bits = self.serializer.send(self.count)
                r, g, b = bits & 0xFF, (bits & 0xFF00) >> 8, \
                    (bits & 0xFF0000) >> 16
                self.serializer_tex.blit_buffer(
                    bytes([r, g, b]), colorfmt='rgb', bufferfmt='ubyte')

            values = _get_app().stage_factory.fill_shape_gl_color_values(
                shape_views, shape_values, proj)
            self.request_process_data('frame', (self.count, bits, values))

    def flip_callback(self, *largs):
        '''Called before every GPU frame by the graphics system.
        '''
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
    '''The instance that is created on the viewer side.
    '''

    alpha_color = NumericProperty(1.)

    filter_background = True

    def start_stage(self, stage_name, canvas):
        self.prepare_view_window()
        return super(ViewSideViewControllerBase, self).start_stage(
            stage_name, canvas)

    def end_stage(self):
        d = {}
        for key in ('cam_scale', 'cam_center_x', 'cam_center_y', 'cam_rotation'):
            d[key] = getattr(self, key)
        d['pixels'], d['proj_size'] = App.get_running_app().get_root_pixels()
        d['proj_size'] = tuple(d['proj_size'])

        val = super(ViewSideViewControllerBase, self).end_stage()
        self.queue_view_write.put_nowait(('end_stage', d))
        return val

    def request_process_data(self, data_type, data):
        self.queue_view_write.put_nowait((
            data_type, yaml_dumps(data)))

    def send_keyboard_down(self, key, modifiers):
        '''Gets called by the window for every keyboard key press, which it
        passes on to the main GUI process.
        '''
        self.queue_view_write.put_nowait((
            'key_down', yaml_dumps((key, list(modifiers)))))

    def send_keyboard_up(self, key):
        '''Gets called by the window for every keyboard key release, which it
        passes on to the main GUI process.
        '''
        self.queue_view_write.put_nowait((
            'key_up', yaml_dumps((key, ))))

    def handle_exception(self, exception, exc_info=None):
        '''Called by the second process upon an error which is passed on to the
        main process.
        '''
        if exc_info is not None:
            exc_info = ''.join(traceback.format_exception(*exc_info))
        self.queue_view_write.put_nowait(
            ('exception', yaml_dumps((str(exception), exc_info))))

    @app_error
    def view_read(self, *largs):
        '''Communication between the two process occurs through queues, this
        is run periodically to serve the queue and read messages from the main
        GUI.
        '''
        read = self.queue_view_read
        write = self.queue_view_write
        while True:
            try:
                msg, value = read.get(False)
                if msg == 'eof':
                    App.get_running_app().stop()
                    break
                elif msg == 'config':
                    app = App.get_running_app()
                    if self.tick_event:
                        raise Exception('Cannot configure while running stage')
                    app.ceed_data.clear_existing_config_data()
                    app.ceed_data.apply_config_data_dict(yaml_loads(value))
                elif msg == 'start_stage':
                    self.start_stage(
                        value, App.get_running_app().get_display_canvas())
                elif msg == 'end_stage':
                    self.end_stage()
                elif msg == 'fullscreen':
                    Window.fullscreen = self.fullscreen = value
                elif msg == 'image':
                    plane_buffers, pix_fmt, size, linesize = value

                    # img = Image(
                    #     plane_buffers=plane_buffers, pix_fmt=pix_fmt,
                    #     size=size, linesize=linesize)
                    # sws = SWScale(*size, pix_fmt, ofmt='gray', ow=size[0], oh=size[1])
                    # img = sws.scale(img)
                    #
                    # if self.filter_background:
                    #     buffer = np.array(img.to_bytearray()[0], dtype=np.uint8).reshape((size[1], size[0]))
                    #     blur = cv2.GaussianBlur(buffer, (5, 5), 0)
                    #     binarized = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    #     data = cv2.Canny(binarized, 100, 200).reshape((size[1] * size[0], ))
                    # else:
                    #     data = np.array(img.to_bytearray()[0], dtype=np.uint8)
                    #
                    # rgba = np.zeros((size[1] * size[0], 4), dtype=np.uint8)
                    # rgba[:, 3] = rgba[:, 0] = data
                    # img = Image(
                    #     plane_buffers=[rgba.tobytes(), 0, 0, 0], pix_fmt='rgba',
                    #     size=size)
                    #
                    # App.get_running_app().get_background_widget().update_img(img)
                elif msg == 'get_pixels_register':
                    pixels, size = App.get_running_app().get_root_pixels()
                    write.put_nowait(('get_pixels_register', (pixels, tuple(size))))
                elif msg == 'cam_params':

                    tx, ty, scale, angle, cam_size = value
                    self.cam_rotation = angle
                    self.cam_scale = scale
                    self.cam_center_x = cam_size[0] // 2 + tx
                    self.cam_center_y = cam_size[1] // 2 + ty
                write.put_nowait(('response', msg))
            except Empty:
                break

    def prepare_view_window(self, *largs):
        '''Called before the app is run to prepare the app according to the
        configuration parameters.
        '''
        Window.size = self.screen_width, self.screen_height
        Window.left = self.screen_offset_x
        Window.fullscreen = self.fullscreen


def view_process_enter(read, write, settings, app_settings):
    '''Called by the second internal view process when it is created.
    This calls :meth:`ViewSideViewControllerBase.view_process_enter`.
    '''
    from cplcom.app import run_app
    from ceed.view.main import CeedViewApp, _cleanup

    app = None
    try:
        app = CeedViewApp()

        classes = app.get_config_classes()
        app.app_settings = {cls: app_settings[cls] for cls in classes}
        app.apply_app_settings()

        viewer = app.view_controller
        for k, v in settings.items():
            setattr(viewer, k, v)

        viewer.queue_view_read = read
        viewer.queue_view_write = write
        Clock.schedule_interval(viewer.view_read, .25)
        Clock.schedule_once(viewer.prepare_view_window, 0)

        run_app(app, _cleanup)
    except Exception as e:
        if app is not None:
            app.handle_exception(e, exc_info=sys.exc_info())

    write.put_nowait(('eof', None))


class ControllerSideViewControllerBase(ViewControllerBase):
    '''The instance that is created in the main GUI.
    '''

    view_process = ObjectProperty(None, allownone=True)
    '''Process of the internal window that runs the experiment through
    a :class:`ViewSideViewControllerBase`.
    '''

    _ctrl_down = False
    '''True when ctrl is pressed down in the viewer side.
    '''

    selected_stage_name = ''
    '''The name of the stage currently selected in the GUI. This will be the
    one started.
    '''

    last_cam_image = None

    cam_image = None

    proj_size = None

    proj_pixels = None

    proj_t_pixels = None

    registration_thread = None

    def add_graphics(self, canvas, black_back=True):
        return super(ControllerSideViewControllerBase, self).add_graphics(
            canvas, black_back=black_back)

    @app_error
    def request_stage_start(self, stage_name):
        '''Starts the stage either in the GUI when previewing or in the
        viewer.
        '''
        # needs t be set here so button is reset on fail
        self.stage_active = True
        self.last_cam_image = self.cam_image = self.proj_pixels = \
            self.proj_size = self.proj_t_pixels = None
        if not stage_name:
            self.stage_active = False
            raise ValueError('No stage specified')
        if not self.preview and not self.view_process:
            self.stage_active = False
            raise Exception("No window to run experiment")

        # if we need to save the florescent image, start video cam
        # if knspace.gui_save_cam_stage.state == 'down':
        #     video_btn = knspace.gui_play
        #     if video_btn.state == 'normal':
        #         video_btn.trigger_action(0)

        App.get_running_app().dump_app_settings_to_file()
        App.get_running_app().load_app_settings_from_file()
        App.get_running_app().ceed_data.prepare_experiment(stage_name)

        if self.propixx_lib:
            m = self.LED_mode
            self.set_led_mode(m)
            App.get_running_app().ceed_data.add_led_state(
                0, 'R' in m, 'G' in m, 'B' in m)
            self.set_pixel_mode(True)
        else:
            App.get_running_app().ceed_data.add_led_state(0, 1, 1, 1)

        if self.preview:
            self.start_stage(stage_name, knspace.painter.canvas)
        elif self.view_process and self.queue_view_read:
            app = App.get_running_app()
            self.queue_view_read.put_nowait(
                ('config', yaml_dumps(app.ceed_data.gather_config_data_dict())))
            self.queue_view_read.put_nowait(('start_stage', stage_name))

    @app_error
    def request_stage_end(self):
        '''Ends the stage either in the GUI when previewing or in the
        viewer.
        '''
        if self.preview:
            self.end_stage()
        elif self.view_process and self.queue_view_read:
            if knspace.gui_save_cam_stage.state == 'down':
                video_btn = knspace.gui_play
                if video_btn.state == 'down':
                    video_btn.trigger_action(0)
                if knspace.gui_remote_view.state == 'down':
                    knspace.gui_remote_view.trigger_action(0)

            self.queue_view_read.put_nowait(('end_stage', None))

    def stage_end_cleanup(self, state={}):
        App.get_running_app().ceed_data.stop_experiment()
        self.stage_active = False

        # if we need to save the florescent image, stop video cam
        if knspace.gui_save_cam_stage.state == 'down':
            video_btn = knspace.gui_play
            if video_btn.state == 'down':
                video_btn.trigger_action(0)
            if knspace.gui_remote_view.state == 'down':
                knspace.gui_remote_view.trigger_action(0)
            knspace.gui_save_cam_stage.state = 'normal'

            if state:
                self.cam_rotation = state['cam_rotation']
                self.cam_scale = state['cam_scale']
                self.cam_center_x = state['cam_center_x']
                self.cam_center_y = state['cam_center_y']
                if self.last_cam_image:
                    self.cam_image = self.last_cam_image
                    self.proj_size = state['proj_size']
                    self.proj_pixels = state['pixels']
                    self.proj_t_pixels = None

        if self.propixx_lib:
            self.set_pixel_mode(False)
            self.set_led_mode(self.LED_mode_idle)

    @app_error
    def end_stage(self):
        val = super(ControllerSideViewControllerBase, self).end_stage()
        self.stage_end_cleanup()
        return val

    def request_fullscreen(self, state):
        '''Sets the fullscreen state to full or not of the second internal
        view process.
        '''
        self.fullscreen = state
        if self.view_process and self.queue_view_read:
            self.queue_view_read.put_nowait(('fullscreen', state))

    def send_background_image(self, image):
        '''Sets the fullscreen state to full or not of the second internal
        view process.
        '''
        if self.view_process and self.queue_view_read:
            self.queue_view_read.put_nowait((
                'image', (image.to_bytearray(), image.get_pixel_format(),
                          image.get_size(), image.get_linesizes())))
            self.last_cam_image = image

    def request_process_data(self, data_type, data):
        if data_type == 'GPU':
            self.gpu_fps = data
        elif data_type == 'CPU':
            self.cpu_fps = data
        elif data_type == 'frame':
            App.get_running_app().ceed_data.add_frame(*data)
        elif data_type == 'frame_flip':
            if data[0]:  # counts of zero is too early
                App.get_running_app().ceed_data.add_frame_flip(*data)

    def start_process(self):
        '''Starts the process of the internal window that runs the experiment
        through a :class:`ViewSideViewControllerBase`.
        '''
        if self.view_process:
            return

        App.get_running_app().dump_app_settings_to_file()
        App.get_running_app().load_app_settings_from_file()
        settings = {name: getattr(self, name)
                    for name in ViewControllerBase.__settings_attrs__}

        ctx = mp.get_context('spawn') if not PY2 else mp
        r = self.queue_view_read = ctx.Queue()
        w = self.queue_view_write = ctx.Queue()
        os.environ['CEED_IS_VIEW'] = '1'
        self.view_process = process = ctx.Process(
            target=view_process_enter,
            args=(r, w, settings, App.get_running_app().app_settings))
        process.start()
        del os.environ['CEED_IS_VIEW']
        Clock.schedule_interval(self.controller_read, .25)

    def stop_process(self):
        '''Ends the :class:`view_process` process by sending a EOF to
        the second process.
        '''
        if self.view_process and self.queue_view_read:
            self.queue_view_read.put_nowait(('eof', None))
            self.queue_view_read = None

    def finish_stop_process(self):
        '''Called by by the read queue thread when we receive the message that
        the second process received an EOF and that it stopped.
        '''
        if not self.view_process:
            return

        self.view_process.join()
        self.view_process = self.queue_view_read = self.queue_view_write = None
        Clock.unschedule(self.controller_read)

    def handle_key_press(self, key, modifiers=[], down=True):
        '''Called by by the read queue thread when we receive a keypress
        event from the second process.
        '''
        if key in ('ctrl', 'lctrl', 'rctrl'):
            self._ctrl_down = down
        if not self._ctrl_down or down:
            return

        if key == 'z':
            if self.stage_active:
                self.request_stage_end()
            self.stop_process()
        elif key == 'c' and self.stage_active:
            self.request_stage_end()
        elif key == 's':
            if not self.stage_active:
                self.request_stage_start(self.selected_stage_name)
        elif key == 'f':
            self.request_fullscreen(not self.fullscreen)
        elif key == 'r':
            if self.last_cam_image:
                self.queue_view_read.put_nowait(('get_pixels_register', None))

    def controller_read(self, *largs):
        '''Called periodically to serve the queue that receives messages from
        the second process.
        '''
        read = self.queue_view_write
        while True:
            try:
                msg, value = read.get(False)
                if msg == 'eof':
                    self.finish_stop_process()
                    self.stage_end_cleanup()
                    break
                elif msg == 'exception':
                    e, exec_info = yaml_loads(value)
                    App.get_running_app().handle_exception(
                        e, exc_info=exec_info)
                elif msg in ('GPU', 'CPU', 'frame', 'frame_flip'):
                    self.request_process_data(
                        msg, yaml_loads(value))
                elif msg == 'end_stage' and msg != 'response':
                    self.stage_end_cleanup(value)
                elif msg == 'key_down':
                    key, modifiers = yaml_loads(value)
                    self.handle_key_press(key, modifiers)
                elif msg == 'key_up':
                    key, = yaml_loads(value)
                    self.handle_key_press(key, down=False)
                elif msg == 'get_pixels_register':
                    if self.last_cam_image:
                        self.proj_pixels, self.proj_size = value
                        self.cam_image = self.last_cam_image
                        self.proj_t_pixels = None
                        self.do_cam_registration()
            except Empty:
                break

    @app_error
    def set_pixel_mode(self, state):
        if PROPixxCTRL is None:
            raise ImportError('Cannot open PROPixx library')

        ctrl = PROPixxCTRL()
        if state:
            ctrl.dout.enablePixelMode()
        else:
            ctrl.dout.disablePixelMode()
        ctrl.updateRegisterCache()
        ctrl.close()

    @app_error
    def set_led_mode(self, mode):
        '''Sets the projector's LED mode. ``mode`` can be one of
        :attr:`ViewControllerBase.led_modes`.
        '''
        if libdpx is None:
            raise ImportError('Cannot open PROPixx library')

        libdpx.DPxOpen()
        libdpx.DPxSelectDevice('PROPixx')
        libdpx.DPxSetPPxLedMask(self.led_modes[mode])
        libdpx.DPxUpdateRegCache()
        libdpx.DPxClose()

    @app_error
    def set_video_mode(self, mode):
        '''Sets the projector's video mode. ``mode`` can be one of
        :attr:`ViewControllerBase.video_modes`.
        '''
        if PROPixx is None:
            raise ImportError('Cannot open PROPixx library')

        dev = PROPixx()
        dev.setDlpSequencerProgram(mode)
        dev.updateRegisterCache()
        dev.close()

    def do_cam_registration(self):
        if not self.proj_pixels or not self.cam_image or self.registration_thread:
            return

        self.registration_thread = t = Thread(
            target=self.register_cam_to_projector,
            args=(self.proj_pixels, self.cam_image, self.proj_size))
        t.start()

    def get_cam_registration_results(self, res, *largs):
        ty, tx, scale, angle, cam_size = res
        self.cam_scale = 1.
        self.cam_center_x = self.screen_width // 2 + tx
        self.cam_center_y = self.screen_height // 2 - ty
        self.cam_rotation = angle
        self.cam_scale = scale

        if self.queue_view_read:
            self.queue_view_read.put_nowait(('cam_params', res))
        self.registration_thread = None

    def register_cam_to_projector(self, proj_pixels, cam_image, proj_size):
        try:
            import time
            import imreg_dft as ird
            ts = time.clock()
            cam = cam_image
            cam_size = cam.get_size()
            sws = SWScale(*cam_size, cam.get_pixel_format(), ofmt='gray')
            cam = sws.scale(cam)

            cam = np.array(
                cam.to_bytearray()[0], dtype=np.uint8).reshape(
                (cam_size[1], cam_size[0]))
            blur = cv2.GaussianBlur(cam, (5, 5), 0)
            cam2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            cam3 = zoom(cam2, .5)
            cam4 = np.zeros(cam_size[::-1])
            cam4[512:-512, 612:-612] = cam3

            proj_size = proj_size
            proj = np.frombuffer(proj_pixels, dtype=np.uint8).reshape(
                (proj_size[1], proj_size[0], 4))[:, :, :3]
            proj = np.mean(proj, axis=-1, dtype=np.uint8)
            proj_pad = np.zeros(cam_size[::-1], dtype=np.uint8)
            proj_pad[968 // 2:-968 // 2, 528 // 2:-528 // 2] = proj
            blur = cv2.GaussianBlur(proj_pad, (5, 5), 0)
            proj_pad2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            res = ird.similarity(proj_pad2, cam4, numiter=3)
            scale = float(res['scale']) / 2
            angle = float(res['angle'])
            tx, ty = map(float, res['tvec'])

            Clock.schedule_once(partial(
                self.get_cam_registration_results,
                (tx, ty, scale, angle, cam_size)), 0)
            print('It took', time.clock() - ts)
        except:
            self.registration_thread = None
            raise
