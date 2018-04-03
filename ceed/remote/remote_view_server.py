'''Ceed View App
=====================

The module that runs the :mod:`ceed.view` GUI for displaying the pixels on the
projector. This is run in a seperate process than the main server side GUI.
'''
import os
from ffpyplayer.pic import Image
from functools import partial
from itertools import accumulate
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
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, BooleanProperty
from kivy.uix.slider import Slider
from kivy.lang import Builder
from kivy.logger import Logger
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
        size: self.current_screen.size if self.current_screen else (100, 100)
        current: cam_selection.text
        Screen:
            name: 'ThorLabs camera'
            size_hint: None, None
            size: pt_grid.size
        Screen:
            name: 'Point Gray camera'
            size_hint: None, None
            size: pt_grid.size
            GridLayout:
                id: pt_grid
                size_hint: None, None
                size: self.minimum_size
                cols: 1
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
                    ToggleButton:
                        id: gui_pt_settings_opt_auto
                        text: 'Auto'
                        padding: '5dp', '5dp'
                        size_hint: None, None
                        size: self.texture_size
                        on_release: app.send_client_cam_request('set_cam_setting', (pt_settings_opt.text, 'auto', self.state == 'down'))
                        disabled: not pt_settings_opt.controllable
                    Button:
                        id: gui_pt_settings_opt_push
                        text: 'One push'
                        size_hint: None, None
                        size: self.texture_size
                        padding: '5dp', '5dp'
                        on_release: app.send_client_cam_request('set_cam_setting', (pt_settings_opt.text, 'one_push', True))
                        disabled: not pt_settings_opt.controllable
                    ToggleButton:
                        id: gui_pt_settings_opt_disable
                        text: 'Disable'
                        padding: '5dp', '5dp'
                        size_hint: None, None
                        size: self.texture_size
                        on_release: app.send_client_cam_request('set_cam_setting', (pt_settings_opt.text, 'controllable', self.state == 'normal'))
                    Label:
                        id: gui_pt_settings_opt_min
                        padding: '5dp', '5dp'
                        size_hint: None, None
                        size: self.texture_size
                        disabled: not pt_settings_opt.controllable
                    TextInput:
                        id: gui_pt_settings_opt_value
                        size_hint: None, None
                        size: '50dp', self.minimum_height
                        on_focus: if not self.focus: app.send_client_cam_request('set_cam_setting', (pt_settings_opt.text, 'value', float(self.text or 0)))
                        input_filter: 'float'
                        disabled: not pt_settings_opt.controllable
                    Label:
                        id: gui_pt_settings_opt_max
                        padding: '5dp', '5dp'
                        size_hint: None, None
                        size: self.texture_size
                        disabled: not pt_settings_opt.controllable
                    Button:
                        id: gui_pt_settings_opt_reload
                        text: 'Reload'
                        padding: '5dp', '5dp'
                        size_hint: None, None
                        size: self.texture_size
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
'''


class EndConnection(Exception):
    pass


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

    def send_msg_to_client(self, sock, msg, value):
        data = yaml_dumps((msg, value))
        data = data.encode('utf8')

        data = struct.pack('>II', len(data), 0) + data
        sock.sendall(data)

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
    def send_client_cam_request(self, msg, value):
        if self.from_kivy_queue is None:
            return

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
