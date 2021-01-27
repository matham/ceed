'''View Controller
======================

Displays the preview or live pixel output of the experiment.
'''
import multiprocessing as mp
import os
import sys
from fractions import Fraction
import traceback
from queue import Empty
import uuid
from tree_config import get_config_children_names

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

from more_kivy_app.app import app_error
from more_kivy_app.utils import yaml_dumps, yaml_loads

from ceed.stage import StageDoneException, last_experiment_stage_name, \
    StageFactoryBase

ignore_vpixx_import_error = False
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

    _config_props_ = (
        'screen_width', 'screen_height', 'frame_rate',
        'use_software_frame_rate', 'output_count', 'screen_offset_x',
        'fullscreen', 'video_mode', 'LED_mode', 'LED_mode_idle',
        'mirror_mea', 'mea_num_rows', 'mea_num_cols',
        'mea_pitch', 'mea_diameter', 'mea_transform', 'cam_transform',
        'flip_projector', 'flip_camera', 'pad_to_stage_handshake',
        'pre_compute_stages',
    )

    screen_width = NumericProperty(1920)
    '''The screen width on which the data is played. This is the full-screen
    size.
    '''

    flip_projector = BooleanProperty(True)

    flip_camera = BooleanProperty(False)

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

    cam_transform = ObjectProperty(Matrix().tolist())

    mea_transform = ObjectProperty(Matrix().tolist())

    mirror_mea = BooleanProperty(True)

    mea_num_rows = NumericProperty(12)

    mea_num_cols = NumericProperty(12)

    mea_pitch = NumericProperty(20)

    mea_diameter = NumericProperty(3)

    pad_to_stage_handshake = BooleanProperty(True)
    """Ceed sends some handshaking info to MCS for each experiment, to help
    us align the ceed and MCS data afterwards. If the root stage of the
    experiment is too short, it's possible the full handshake would not be
    sent, preventing alignment afterwards.

    If :attr:`pad_to_stage_handshake`, then the root stage will be padded
    so it goes for the minimum number of clock frames required to finish
    the handshake, if it's too short. The shapes will be black for those
    padded frames.
    """

    output_count = BooleanProperty(True)
    '''Whether the corner pixel is used to output frame information on the
    PROPixx controller IO pot. If True,
    :class:`ceed.storage.controller.DataSerializerBase` is used to set the 24
    bits of the corner pixel.
    '''

    fullscreen = BooleanProperty(True)
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

    pre_compute_stages: bool = BooleanProperty(False)
    """
    """

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

    tick_delay_event = None
    '''The delay event that triggers tick_event after an initial delay to
    ensure everything is ready before we start showing actual frames.
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
        for name in ViewControllerBase._config_props_:
            self.fbind(name, self.dispatch, 'on_changed')
        self.propixx_lib = libdpx is not None
        self.shape_views = []

    def _restore_cam_pos(self):
        if self._scheduled_pos_restore:
            return

        self._scheduled_pos_restore = True
        transform = self.cam_transform

        def restore_state(*largs):
            self.cam_transform = transform
            self._scheduled_pos_restore = False

        Clock.schedule_once(restore_state, -1)

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

        with canvas:
            PushMatrix()
            s = Scale()
            if self.flip_projector:
                s.x = -1
            s.origin = w / 2., h / 2.

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
                instructs = _get_app().\
                    stage_factory.get_shapes_gl_color_instructions(
                    canvas, self.canvas_name)
                with canvas:
                    PopMatrix(group=self.canvas_name)
                self.shape_views.append(instructs)
        else:
            self.shape_views = [
                _get_app().stage_factory.get_shapes_gl_color_instructions(
                    canvas, self.canvas_name)]

        with canvas:
            PopMatrix()

        if self.output_count and not self.serializer_tex:
            with canvas:
                Color(1, 1, 1, 1, group=self.canvas_name)
                tex = self.serializer_tex = Texture.create(size=(1, 1))
                tex.mag_filter = 'nearest'
                tex.min_filter = 'nearest'
                Rectangle(texture=tex, pos=(0, h - 1), size=(1, 1),
                          group=self.canvas_name)

    def start_stage(self, stage_name, canvas):
        '''Starts the stage. It adds the graphics instructions to the canvas
        and starts playing the shapes.
        '''
        from kivy.core.window import Window
        if self.tick_event:
            raise TypeError('Cannot start new stage while stage is active')

        Clock._max_fps = 0
        self.tick_event = Clock.create_trigger(
            self.tick_callback, 0, interval=True)
        self.tick_delay_event = Clock.schedule_once(self.tick_event, .25)
        Window.fbind('on_flip', self.flip_callback)

        stage_factory: StageFactoryBase = _get_app().stage_factory
        stage = stage_factory.stage_names[last_experiment_stage_name]
        stage.pad_stage_ticks = 0

        if self.output_count:
            msg = uuid.uuid4().bytes
            n = len(msg)

            data_serializer = App.get_running_app().data_serializer
            if self.pad_to_stage_handshake:
                stage.pad_stage_ticks = data_serializer.num_ticks_handshake(n)
            self.serializer = data_serializer.get_bits(-1, msg)

        self.current_canvas = canvas
        self.tick_func = stage_factory.tick_stage(
            Fraction(1, int(self.effective_frame_rate)),
            self.effective_frame_rate, stage_name=last_experiment_stage_name,
            pre_compute=self.pre_compute_stages)
        next(self.tick_func)

        self._flip_stats['last_call_t'] = self._cpu_stats['last_call_t'] = \
            self._cpu_stats['tstart'] = clock()

        self.add_graphics(canvas)

    def end_stage(self):
        '''Ends the stage if one is playing.
        '''
        from kivy.core.window import Window
        if not self.tick_event:
            return

        self.tick_event.cancel()
        if self.tick_delay_event is not None:
            self.tick_delay_event.cancel()
        Window.funbind('on_flip', self.flip_callback)
        Clock._max_fps = self._original_fps
        _get_app().stage_factory.remove_shapes_gl_color_instructions(
            self.current_canvas, self.canvas_name)

        self.tick_func = self.tick_event = self.current_canvas = None
        self.tick_delay_event = None
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
            # we cannot skip frames (i.e. we may only increment frame by one).
            # Because stages/func can be pre-computed and it assumes a constant
            # frame rate. If need to skip n frames, tick n times and drop result
            self.count += 1

            try:
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
            else:
                bits = 0

            values = _get_app().stage_factory.fill_shape_gl_color_values(
                shape_views, shape_values, proj)
            self.request_process_data('frame', (self.count, bits, values))

    def flip_callback(self, *largs):
        '''Called before every GPU frame by the graphics system.
        '''
        from kivy.core.window import Window
        Window.on_flip()

        t = clock()
        # count of zero is discarded
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
        from kivy.core.window import Window
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
                write.put_nowait(('response', msg))
            except Empty:
                break

    def prepare_view_window(self, *largs):
        '''Called before the app is run to prepare the app according to the
        configuration parameters.
        '''
        from kivy.core.window import Window
        Window.size = self.screen_width, self.screen_height
        Window.left = self.screen_offset_x
        Window.fullscreen = self.fullscreen


def view_process_enter(read, write, settings, app_settings):
    '''Called by the second internal view process when it is created.
    This calls :meth:`ViewSideViewControllerBase.view_process_enter`.
    '''
    from more_kivy_app.app import run_app
    from ceed.view.main import CeedViewApp

    app = None
    try:
        app = CeedViewApp()

        classes = get_config_children_names(app)
        app.app_settings = {cls: app_settings[cls] for cls in classes}
        app.apply_app_settings()

        viewer = app.view_controller
        for k, v in settings.items():
            setattr(viewer, k, v)

        viewer.queue_view_read = read
        viewer.queue_view_write = write
        Clock.schedule_interval(viewer.view_read, .25)
        Clock.schedule_once(viewer.prepare_view_window, 0)

        run_app(app)
    except Exception as e:
        if app is not None:
            app.handle_exception(e, exc_info=sys.exc_info())
        else:
            exc_info = ''.join(traceback.format_exception(*sys.exc_info()))
            write.put_nowait(('exception', yaml_dumps((str(e), exc_info))))
    finally:
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

    initial_cam_image = None

    last_cam_image = ObjectProperty(None, allownone=True)

    proj_size = None

    proj_pixels = None

    def add_graphics(self, canvas, black_back=True):
        return super(ControllerSideViewControllerBase, self).add_graphics(
            canvas, black_back=black_back)

    @app_error
    def request_stage_start(self, stage_name):
        '''Starts the stage either in the GUI when previewing or in the
        viewer.

        Look into immediately erroring out if already running. So that we
        don't overwrite the initial image if we're already running.
        '''
        # needs to be set here so button is reset on fail
        self.stage_active = True
        self.last_cam_image = self.proj_pixels = self.proj_size = None
        self.initial_cam_image = None
        if not stage_name:
            self.stage_active = False
            raise ValueError('No stage specified')

        app = App.get_running_app()
        app.stages_container.\
            copy_and_resample_experiment_stage(stage_name)
        app.dump_app_settings_to_file()
        app.load_app_settings_from_file()
        app.ceed_data.prepare_experiment(
            stage_name,
            app.stage_factory.stage_names[stage_name].get_stage_shape_names())

        if self.propixx_lib:
            m = self.LED_mode
            self.set_led_mode(m)
            app.ceed_data.add_led_state(
                0, 'R' in m, 'G' in m, 'B' in m)
            self.set_pixel_mode(True)
        else:
            app.ceed_data.add_led_state(0, 1, 1, 1)

        if self.view_process is None:
            self.start_stage(stage_name, app.shape_factory.canvas)
        elif self.queue_view_read is not None:
            self.initial_cam_image = app.player.last_image
            self.queue_view_read.put_nowait(
                ('config', yaml_dumps(app.ceed_data.gather_config_data_dict())))
            self.queue_view_read.put_nowait(('start_stage', stage_name))
        else:
            self.stage_active = False
            raise ValueError('Already running stage')

    @app_error
    def request_stage_end(self):
        '''Ends the stage either in the GUI when previewing or in the
        viewer.
        '''
        if self.view_process is None:
            self.end_stage()
        elif self.queue_view_read is not None:
            self.last_cam_image = App.get_running_app().player.last_image
            if self.last_cam_image is self.initial_cam_image:
                self.last_cam_image = None
            self.queue_view_read.put_nowait(('end_stage', None))

    def stage_end_cleanup(self, state=None):
        ceed_data = App.get_running_app().ceed_data
        if ceed_data is not None:
            ceed_data.stop_experiment()

        self.stage_active = False
        if state:
            if self.last_cam_image is None:
                self.last_cam_image = App.get_running_app().player.last_image

            if self.last_cam_image is not None:
                self.proj_size = state['proj_size']
                self.proj_pixels = state['pixels']

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
                    for name in ViewControllerBase._config_props_}

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
            except Empty:
                break

    @app_error
    def set_pixel_mode(self, state):
        if PROPixxCTRL is None:
            if ignore_vpixx_import_error:
                return
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
            if ignore_vpixx_import_error:
                return
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
            if ignore_vpixx_import_error:
                return
            raise ImportError('Cannot open PROPixx library')

        dev = PROPixx()
        dev.setDlpSequencerProgram(mode)
        dev.updateRegisterCache()
        dev.close()
