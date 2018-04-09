'''Ceed View App
=====================

The module that runs the :mod:`ceed.view` GUI for displaying the pixels on the
projector. This is run in a seperate process than the main server side GUI.
'''
import os
from ffpyplayer.pic import Image
from functools import partial
from itertools import accumulate
import os
import numpy as np
from os.path import join, dirname, isdir
from threading import Thread
import socket
import sys
import struct
from queue import Queue, Empty
import traceback
import select

from cplcom.app import run_app as run_cpl_app, app_error, CPLComApp
from cplcom.utils import yaml_dumps, yaml_loads

from kivy.app import App
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, \
    BooleanProperty, ListProperty
from kivy.uix.slider import Slider
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types

if __name__ == '__main__':
    from kivy.core.window import Window

__all__ = ('CeedRemoteViewApp',)


class UpSlider(Slider):
    __events__ = ('on_release',)

    def on_release(self, *largs):
        pass

    def on_touch_up(self, touch):
        if super(UpSlider, self).on_touch_up(touch):
            if touch.grab_current == self:
                self.dispatch('on_release', self)
            return True


kv = '''
BoxLayout:
    orientation: 'vertical'
    spacing: '15dp'
    padding: '5dp',
    Spinner:
        id: cam_selection
        values: ['ThorLabs camera', 'Point Gray camera']
        text: 'ThorLabs camera' 
        size_hint_y: None
        height: '40dp'
    ScreenManager:
        size_hint: None, None
        size: tsi_grid.parent.size if self.current == 'ThorLabs camera' else pt_grid.parent.size
        current: cam_selection.text
        Screen:
            name: 'ThorLabs camera'
            size_hint: None, None
            size: tsi_grid.size
            disabled: app.tsi_sdk is None
            GridLayout:
                id: tsi_grid
                size_hint: None, None
                size: self.minimum_size
                cols: 1
                BoxLayout:
                    size_hint: None, None
                    size: self.minimum_size
                    spacing: '5dp'
                    SizedButton:
                        text: 'Refresh'
                        on_release: tsi_cams_spinner.values = app.get_tsi_cams()
                    Spinner:
                        id: tsi_cams_spinner
                        text_autoupdate: True
                        size_hint_x: None
                        width: '60dp'
                    SizedToggleButton:
                        disabled: not tsi_cams_spinner.text
                        text: 'Open' if self.state == 'normal' else 'Close'
                        on_release: app.open_tsi_cam(tsi_cams_spinner.text, root.ids) if self.state == 'down' else app.close_tsi_cam()
                    SizedToggleButton:
                        disabled: app.tsi_cam is None
                        text: 'Play' if self.state == 'normal' else 'Stop'
                        on_release: app.tsi_cam.send_message('play') if self.state == 'down' else app.tsi_cam.send_message('stop')
                BoxLayout:
                    disabled: app.tsi_cam is None
                    size_hint: None, None
                    size: self.minimum_size
                    spacing: '5dp'
                    SizedLabel:
                        text: 'Speed:'
                    Spinner:
                        text_autoupdate: True
                        size_hint_x: None
                        width: '80dp'
                        values: app.tsi_cam.supported_freqs if app.tsi_cam else []
                        text: app.tsi_cam.freq if app.tsi_cam else '20 MHz'
                        on_text: if app.tsi_cam: app.tsi_cam.send_message('setting', ('freq', self.text))
                    SizedLabel:
                        text: 'Taps:'
                    Spinner:
                        text_autoupdate: True
                        size_hint_x: None
                        width: '60dp'
                        values: app.tsi_cam.supported_taps if app.tsi_cam else []
                        text: app.tsi_cam.taps if app.tsi_cam else '1'
                        on_text: if app.tsi_cam: app.tsi_cam.send_message('setting', ('taps', self.text))
                    SizedLabel:
                        text: 'Exposure [{}ms, {}ms]:'.format(*(app.tsi_cam.exposure_range if app.tsi_cam else (0, 100)))
                    TextInput:
                        id: tsi_value
                        size_hint: None, None
                        size: '80dp', self.minimum_height
                        text: str(app.tsi_cam.exposure_ms) if app.tsi_cam else '0'
                        multiline: False
                        on_focus:
                            if not self.focus and app.tsi_cam: app.tsi_cam.send_message('setting', ('exposure_ms', float(self.text)))
                        input_filter: 'float'
        Screen:
            name: 'Point Gray camera'
            size_hint: None, None
            size: pt_grid.size
            GridLayout:
                id: pt_grid
                cols: 1
                size_hint: None, None
                size: pt_grid.size
                GridLayout:
                    size_hint: None, None
                    size: self.minimum_size
                    height: '50dp'
                    spacing: ['5dp']
                    rows: 1
                    disabled: not app.connected
                    Widget
                    Spinner:
                        id: pt_settings_opt
                        size_hint_x: None
                        width: '100dp'
                        values: ['brightness', 'exposure', 'sharpness', 'hue', 'saturation', 'gamma', 'shutter', 'gain', 'iris', 'frame_rate', 'pan', 'tilt']
                        on_text: 
                            app.send_client_cam_request('track_cam_setting', self.text)
                            app.cam_setting = self.text
                        controllable: gui_pt_settings_opt_disable.state == 'normal'
                    SizedToggleButton:
                        id: gui_pt_settings_opt_auto
                        text: 'Auto'
                        on_release: app.send_client_cam_request('set_cam_setting', (pt_settings_opt.text, 'auto', self.state == 'down'))
                        disabled: not pt_settings_opt.controllable
                    SizedButton:
                        id: gui_pt_settings_opt_push
                        text: 'One push'
                        on_release: app.send_client_cam_request('set_cam_setting', (pt_settings_opt.text, 'one_push', True))
                        disabled: not pt_settings_opt.controllable
                    SizedToggleButton:
                        id: gui_pt_settings_opt_disable
                        text: 'Disable'
                        on_release: app.send_client_cam_request('set_cam_setting', (pt_settings_opt.text, 'controllable', self.state == 'normal'))
                    SizedLabel:
                        id: gui_pt_settings_opt_min
                        disabled: not pt_settings_opt.controllable
                    TextInput:
                        id: gui_pt_settings_opt_value
                        size_hint: None, None
                        size: '50dp', self.minimum_height
                        on_focus: if not self.focus: app.send_client_cam_request('set_cam_setting', (pt_settings_opt.text, 'value', float(self.text or 0)))
                        input_filter: 'float'
                        disabled: not pt_settings_opt.controllable
                        multiline: False
                    SizedLabel:
                        id: gui_pt_settings_opt_max
                        disabled: not pt_settings_opt.controllable
                    SizedButton:
                        id: gui_pt_settings_opt_reload
                        text: 'Reload'
                        on_release: app.send_client_cam_request('reload_cam_setting', pt_settings_opt.text)
                    Widget
                UpSlider:
                    id: gui_pt_settings_opt_slider
                    size_hint_y: None
                    height: '50dp'
                    on_release: app.send_client_cam_request('set_cam_setting', (pt_settings_opt.text, 'value', self.value))
                    disabled: not pt_settings_opt.controllable
    BufferImage:
        canvas.before:
            Color:
                rgba: [0, 0, 0, 1]
            Rectangle:
                pos: self.pos
                size: self.size
        id: img
        auto_bring_to_front: False
        available_size: self.size 

        do_scale: True
        do_translation: True, True
        do_rotation: True

<ScreenGrid@Screen+GridLayout>
<SizedLabel@Label>:
    size_hint: None, None
    size: self.texture_size
    padding: '5dp', '5dp'
<SizedButton@Button>:
    size_hint: None, None
    size: self.texture_size
    padding: '5dp', '5dp'
<SizedToggleButton@ToggleButton>:
    size_hint: None, None
    size: self.texture_size
    padding: '5dp', '5dp'
'''


class EndConnection(Exception):
    pass


class TSICamera(EventDispatcher):

    supported_freqs = ListProperty(['20 MHz', ])

    freq = StringProperty('20 MHz')

    supported_taps = ListProperty(['1', ])

    taps = StringProperty('1')

    supports_color = BooleanProperty(False)

    exposure_range = ListProperty([0, 100])

    exposure_ms = NumericProperty(5)

    tsi_sdk = None

    tsi_interface = None

    serial = ''

    ids = {}

    _kivy_trigger = None

    from_kivy_queue = None

    to_kivy_queue = None

    camera_thread = None

    _freqs_to_str_map = {}

    _str_to_freqs_map = {}

    _taps_to_str_map = {}

    _str_to_taps_map = {}

    def __init__(self, tsi_sdk, tsi_interface, serial, ids):
        self.tsi_sdk = tsi_sdk
        self.tsi_interface = tsi_interface
        self.serial = serial
        self.ids = ids
        self._freqs_to_str_map = {
            tsi_interface.DataRate.ReadoutSpeed20MHz: '20 MHz',
            tsi_interface.DataRate.ReadoutSpeed40MHz: '40 MHz',
            tsi_interface.DataRate.FPS50: '50 FPS',
            tsi_interface.DataRate.FPS30: '30 FPS',
        }
        self._str_to_freqs_map = {v: k for k, v in self._freqs_to_str_map.items()}
        self._taps_to_str_map = {
            tsi_interface.Taps.QuadTap: '4',
            tsi_interface.Taps.DualTap: '2',
            tsi_interface.Taps.SingleTap: '1',
        }
        self._str_to_taps_map = {v: k for k, v in self._taps_to_str_map.items()}

        self._kivy_trigger = Clock.create_trigger(self.process_in_kivy_thread)
        from_kivy_queue = self.from_kivy_queue = Queue()
        to_kivy_queue = self.to_kivy_queue = Queue()
        thread = self.camera_thread = Thread(
            target=self.camera_run, args=(from_kivy_queue, to_kivy_queue))
        thread.start()

    def send_message(self, msg, value=None):
        if self.from_kivy_queue:
            self.from_kivy_queue.put((msg, value))

    def read_settings(self, cam):
        d = {}
        rang = cam.get_ExposureTimeRange_us()
        d['exposure_range'] = rang.Minimum / 1000., rang.Maximum / 1000.
        d['exposure_ms'] = cam.get_ExposureTime_us() / 1000.
        if cam.GetIsDataRateSupported(self.tsi_interface.DataRate.ReadoutSpeed20MHz):
            if cam.GetIsDataRateSupported(self.tsi_interface.DataRate.ReadoutSpeed40MHz):
                d['supported_freqs'] = ['20 MHz', '40 MHz']
            else:
                d['supported_freqs'] = ['20 MHz', ]
        else:
            if cam.GetIsDataRateSupported(self.tsi_interface.DataRate.FPS50):
                d['supported_freqs'] = ['30 FPS', '50 FPS']
            else:
                d['supported_freqs'] = ['30 FPS', ]
        d['freq'] = self._freqs_to_str_map[cam.get_DataRate()]

        if cam.GetIsTapsSupported(self.tsi_interface.Taps.QuadTap):
            d['supported_taps'] = ['1', '2', '4']
        elif cam.GetIsTapsSupported(self.tsi_interface.Taps.DualTap):
            d['supported_taps'] = ['1', '2']
        elif cam.GetIsTapsSupported(self.tsi_interface.Taps.SingleTap):
            d['supported_taps'] = ['1', ]
        else:
            d['supported_taps'] = []

        if cam.GetIsTapsSupported(self.tsi_interface.Taps.SingleTap):
            d['taps'] = self._taps_to_str_map[cam.get_Taps()]
        else:
            d['taps'] = ''
        return d

    def write_setting(self, cam, setting, value):
        if setting == 'exposure_ms':
            value = int(max(min(value, self.exposure_range[1]),
                        self.exposure_range[0]) * 1000)
            cam.set_ExposureTime_us(value)
            print(value, cam.get_ExposureTime_us(), cam.ExposureTime_us)
            value = value / 1000.
        elif setting == 'freq':
            cam.set_DataRate(self._str_to_freqs_map[value])
        elif setting == 'taps' and value:
            cam.set_Taps(self._str_to_taps_map[value])
        return setting, value

    def read_frame(self, cam, asNumpyArray):
        if cam.get_NumberOfQueuedFrames() <= 0:
            return

        frame = cam.GetPendingFrameOrNull()
        if not frame:
            return

        count = frame.FrameNumber
        h = frame.ImageDataUShort1D.Height_pixels
        w = frame.ImageDataUShort1D.Width_pixels
        color = frame.ImageDataUShort1D.NumberOfChannels == 3
        data = asNumpyArray(frame.ImageDataUShort1D.ImageData_monoOrBGR)
        img = Image(
            plane_buffers=[data.tobytes()],
            pix_fmt='bgr48le' if color else 'gray16le', size=(w, h))
        return img, count

    def camera_run(self, from_kivy_queue, to_kivy_queue):
        from ceed.remote.net_data import asNumpyArray
        trigger = self._kivy_trigger
        send_image = App.get_running_app().send_client_cam_request
        cam = None
        playing = False
        msg = ''
        try:
            cam = self.tsi_sdk.OpenCamera(self.serial, False)
            settings = self.read_settings(cam)
            to_kivy_queue.put(('settings', settings))
            trigger()

            while msg != 'eof':
                if playing:
                    while True:
                        try:
                            msg, value = from_kivy_queue.get(block=False)
                            if msg == 'eof':
                                break
                            elif msg == 'stop':
                                cam.Disarm()
                                playing = False
                                break
                            elif msg == 'setting':
                                to_kivy_queue.put((msg, self.write_setting(cam, *value)))
                                trigger()
                        except Empty:
                            break

                    if not playing or msg == 'eof':
                        continue
                    data = self.read_frame(cam, asNumpyArray)
                    if data is None:
                        continue
                    img, count = data
                    to_kivy_queue.put(('image', img))
                    trigger()
                    send_image('remote_image', img, max_qsize=2)

                else:
                    msg, value = from_kivy_queue.get(block=True)
                    if msg == 'eof':
                        break
                    elif msg == 'play':
                        cam.Arm()
                        if cam.get_OperationMode() != self.tsi_interface.OperationMode.HardwareTriggered:
                            cam.IssueSoftwareTrigger()
                        playing = True
                    elif msg == 'setting':
                        to_kivy_queue.put((msg, self.write_setting(cam, *value)))
                        trigger()

            if cam.IsArmed:
                cam.Disarm()
            cam.Dispose()
        except Exception as e:
            exc_info = ''.join(traceback.format_exception(*sys.exc_info()))
            to_kivy_queue.put(
                ('exception', (str(e), exc_info)))
            trigger()

            try:
                if cam is not None:
                    if cam.IsArmed:
                        cam.Disarm()
                    cam.Dispose()
            except:
                pass

    @app_error
    def process_in_kivy_thread(self, *largs):
        while self.to_kivy_queue is not None:
            try:
                msg, value = self.to_kivy_queue.get(block=False)

                if msg == 'exception':
                    e, exec_info = value
                    App.get_running_app().handle_exception(
                        e, exc_info=exec_info)
                elif msg == 'image':
                    self.ids.img.update_img(value)
                elif msg == 'settings':
                    print('got', value)
                    for key, val in value.items():
                        setattr(self, key, val)
                elif msg == 'setting':
                    print('setarrt', value)
                    setattr(self, *value)
            except Empty:
                break


class CeedRemoteViewApp(CPLComApp):
    '''The app which runs the GUI.
    '''

    server = StringProperty('')

    port = NumericProperty(10000)

    timeout = NumericProperty(.1)

    connected = BooleanProperty(False)

    cam_setting = ''

    from_kivy_queue = None

    to_kivy_queue = None

    _kivy_trigger = None

    server_thread = None

    tsi_sdk = ObjectProperty(None)

    tsi_interface = ObjectProperty(None)

    tsi_cam = ObjectProperty(None, rebind=True, allownone=True)

    def init_load(self):
        pass

    def build(self):
        return Builder.load_string(kv)

    def on_start(self, *largs):
        self._kivy_trigger = Clock.create_trigger(self.process_in_kivy_thread)
        from_kivy_queue = self.from_kivy_queue = Queue()
        to_kivy_queue = self.to_kivy_queue = Queue()
        thread = self.server_thread = Thread(
            target=self.server_run, args=(from_kivy_queue, to_kivy_queue))
        thread.start()
        self.load_tsi(r'E:\MATLAB')

    @app_error
    def load_tsi(self, dll_path):
        os.environ['PATH'] += os.pathsep + dll_path
        import clr
        clr.AddReference(join(dll_path, 'Thorlabs.TSI.TLCamera.dll'))
        clr.AddReference(join(dll_path, 'Thorlabs.TSI.TLCameraInterfaces.dll'))
        from Thorlabs.TSI.TLCamera import TLCameraSDK
        import Thorlabs.TSI.TLCameraInterfaces as tsi_interface
        self.tsi_sdk = TLCameraSDK.OpenTLCameraSDK()
        self.tsi_interface = tsi_interface
        tsi_interface.CameraSensorType

    @app_error
    def get_tsi_cams(self):
        cams = self.tsi_sdk.DiscoverAvailableCameras()
        names = []
        for i in range(cams.get_Count()):
            names.append(cams.get_Item(i))
        return list(sorted(cams))

    def open_tsi_cam(self, serial, ids):
        self.tsi_cam = TSICamera(
            tsi_sdk=self.tsi_sdk, tsi_interface=self.tsi_interface,
            serial=serial, ids=ids)

    def close_tsi_cam(self):
        if self.tsi_cam:
            self.tsi_cam.send_message('eof')
            self.tsi_cam = None

    def send_msg_to_client(self, sock, msg, value):
        if msg == 'remote_image':
            bin_data = value.to_bytearray()
            data = yaml_dumps((
                'remote_image', (
                    list(map(len, bin_data)), value.get_pixel_format(),
                    value.get_size(), value.get_linesizes())))
            data = data.encode('utf8')
        else:
            data = yaml_dumps((msg, value))
            data = data.encode('utf8')
            bin_data = []

        sock.sendall(struct.pack('>II', len(data), sum(map(len, bin_data))))
        sock.sendall(data)
        for item in bin_data:
            sock.sendall(item)

    def decode_data(self, msg_buff, msg_len):
        n, n_bin = msg_len
        assert n + n_bin == len(msg_buff)

        data = msg_buff[:n].decode('utf8')
        msg, value = yaml_loads(data)

        if msg == 'image':
            bin_data = msg_buff[n:]
            planes_sizes, pix_fmt, size, linesize = value
            starts = list(accumulate([0] + list(planes_sizes[:-1])))
            ends = accumulate(planes_sizes)
            planes = [bin_data[s:e] for s, e in zip(starts, ends)]

            value = planes, pix_fmt, size, linesize
        else:
            assert not n_bin
        return msg, value

    def read_msg_from_client(
            self, sock, to_kivy_queue, trigger, msg_len, msg_buff):
        # still reading msg size
        if not msg_len:
            assert 8 - len(msg_buff)
            data = sock.recv(8 - len(msg_buff))
            if not data:
                raise EndConnection('Remote server was closed')

            msg_buff += data
            if len(msg_buff) == 8:
                msg_len = struct.unpack('>II', msg_buff)
                msg_buff = b''
        else:
            total = sum(msg_len)
            assert total - len(msg_buff)
            data = sock.recv(total - len(msg_buff))
            if not data:
                raise EndConnection('Remote server was closed')

            msg_buff += data
            if len(msg_buff) == total:
                to_kivy_queue.put(self.decode_data(msg_buff, msg_len))
                trigger()

                msg_len = ()
                msg_buff = b''
        return msg_len, msg_buff

    def server_run(self, from_kivy_queue, to_kivy_queue):
        trigger = self._kivy_trigger
        timeout = self.timeout

        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the server is listening
        server_address = (self.server, self.port)
        Logger.info('RemoteView_Server: starting up on {} port {}'
                    .format(*server_address))

        try:
            sock.bind(server_address)
            sock.listen(1)

            while True:
                r, _, _ = select.select([sock], [], [], timeout)
                if not r:
                    try:
                        while True:
                            msg, value = from_kivy_queue.get_nowait()
                            if msg == 'eof':
                                return
                    except Empty:
                        pass

                    continue

                connection, client_address = sock.accept()
                msg_len, msg_buff = (), b''
                Clock.schedule_once(lambda x: setattr(self, 'connected', True))
                from_kivy_queue.put(('get_cam_settings', ''))
                from_kivy_queue.put(('track_cam_setting', self.cam_setting))

                try:
                    while True:
                        r, _, _ = select.select([connection], [], [], timeout)
                        if r:
                            msg_len, msg_buff = self.read_msg_from_client(
                                connection, to_kivy_queue, trigger, msg_len, msg_buff)

                        try:
                            while True:
                                msg, value = from_kivy_queue.get_nowait()
                                if msg == 'eof':
                                    return
                                else:
                                    self.send_msg_to_client(connection, msg, value)
                        except Empty:
                            pass
                except EndConnection:
                    pass
                finally:
                    Logger.info('closing client connection')
                    Clock.schedule_once(lambda x: setattr(self, 'connected', False))
                    connection.close()
        except Exception as e:
            exc_info = ''.join(traceback.format_exception(*sys.exc_info()))
            to_kivy_queue.put(
                ('exception', (str(e), exc_info)))
            trigger()
        finally:
            Logger.info('closing socket')
            sock.close()

    @app_error
    def send_client_cam_request(self, msg, value, max_qsize=0):
        if self.from_kivy_queue is None:
            return

        if not max_qsize or self.from_kivy_queue.qsize() < max_qsize:
            self.from_kivy_queue.put((msg, value))

    def populate_settings(self, setting, opts):
        ids = self.root.ids
        if ids.pt_settings_opt.text != setting:
            return

        ids.gui_pt_settings_opt_auto.state = 'down' if opts['auto'] else 'normal'
        ids.gui_pt_settings_opt_min.text = '{:0.2f}'.format(opts['min'])
        ids.gui_pt_settings_opt_max.text = '{:0.2f}'.format(opts['max'])
        ids.gui_pt_settings_opt_value.text = '{:0.2f}'.format(opts['value'])
        ids.gui_pt_settings_opt_disable.state = 'normal' if opts['controllable'] else 'down'
        ids.gui_pt_settings_opt_slider.min = opts['min']
        ids.gui_pt_settings_opt_slider.max = opts['max']
        ids.gui_pt_settings_opt_slider.value = opts['value']

    @app_error
    def process_in_kivy_thread(self, *largs):
        while self.to_kivy_queue is not None:
            try:
                msg, value = self.to_kivy_queue.get(block=False)

                if msg == 'exception':
                    e, exec_info = value
                    App.get_running_app().handle_exception(
                        e, exc_info=exec_info)
                elif msg == 'image':
                    plane_buffers, pix_fmt, size, linesize = value
                    img = Image(
                        plane_buffers=plane_buffers, pix_fmt=pix_fmt,
                        size=size, linesize=linesize)
                    self.root.ids.img.update_img(img)
                elif msg == 'cam_settings':
                    self.root.ids['pt_settings_opt'].values = value
                elif msg == 'cam_setting':
                    self.populate_settings(*value)
                else:
                    print('got', msg, value)
            except Empty:
                break

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
            # handle_exception(err, exc_info)

    def _ask_close(self, *largs, **kwargs):
        return False

    def get_logger(self):
        return Logger


def _cleanup(app):
    if app.from_kivy_queue is not None:
        app.from_kivy_queue.put(('eof', None))
        if app.server_thread is not None:
            app.server_thread.join()


run_app = partial(run_cpl_app, CeedRemoteViewApp, _cleanup)
'''The function that starts the GUI and the entry point for
the main script.
'''

if __name__ == '__main__':
    run_app()
