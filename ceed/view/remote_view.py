from threading import Thread
import socket
import sys
import struct
from kivy.event import EventDispatcher
from kivy.properties import StringProperty, NumericProperty, BooleanProperty
from kivy.clock import Clock
from kivy.app import App
from kivy.uix.behaviors.knspace import knspace
from queue import Queue, Empty
from cplcom.app import app_error
from cplcom.utils import yaml_dumps, yaml_loads
import traceback
import select
from kivy.logger import Logger


class RemoteViewerListenerBase(EventDispatcher):

    __settings_attrs__ = ('server', 'port', 'timeout')

    server = StringProperty('')

    port = NumericProperty(0)

    timeout = NumericProperty(.1)

    running = BooleanProperty(False)

    from_kivy_queue = None

    to_kivy_queue = None

    _kivy_trigger = None

    listener_thread = None

    def __init__(self):
        self._kivy_trigger = Clock.create_trigger(self.process_in_kivy_thread)

    def send_msg_to_server(self, sock, msg, value):
        import time
        ts = time.monotonic()
        if msg == 'image':
            bin_data = value.to_bytearray()
            data = yaml_dumps((
                'image', (list(map(len, bin_data)), value.get_pixel_format(),
                          value.get_size(), value.get_linesizes())))
            data = data.encode('utf8')
        else:
            data = yaml_dumps((msg, value))
            data = data.encode('utf8')
            bin_data = []
        ts2 = time.monotonic()

        sock.sendall(struct.pack('>II', len(data), sum(map(len, bin_data))))
        sock.sendall(data)
        for item in bin_data:
            sock.sendall(item)
        # print('part1: {}, part2: {}'.format(ts2 - ts, time.monotonic() - ts2))

    def decode_data(self, msg_buff, msg_len):
        n, bin_n = msg_len
        assert n + bin_n == len(msg_buff)
        data = msg_buff[:n].decode('utf8')
        msg, value = yaml_loads(data)

        assert not bin_n
        return msg, value

    def read_msg_from_server(
            self, sock, to_kivy_queue, trigger, msg_len, msg_buff):
        # still reading msg size
        if not msg_len:
            assert 8 - len(msg_buff)
            data = sock.recv(8 - len(msg_buff))
            if not data:
                raise Exception('Remote server was closed')

            msg_buff += data
            if len(msg_buff) == 8:
                msg_len = struct.unpack('>II', msg_buff)
                msg_buff = b''
        else:
            total = sum(msg_len)
            assert total - len(msg_buff)
            data = sock.recv(total - len(msg_buff))
            if not data:
                raise Exception('Remote server was closed')

            msg_buff += data
            if len(msg_buff) == total:
                to_kivy_queue.put(self.decode_data(msg_buff, msg_len))
                trigger()

                msg_len = ()
                msg_buff = b''
        return msg_len, msg_buff

    def listener_run(self, from_kivy_queue, to_kivy_queue):
        trigger = self._kivy_trigger
        timeout = self.timeout

        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Connect the socket to the port where the server is listening
        server_address = (self.server, self.port)
        Logger.info('RemoteView_Listener: connecting to {} port {}'
                    .format(*server_address))

        msg_len, msg_buff = (), b''

        try:
            sock.connect(server_address)
            done = False

            while not done:
                r, _, _ = select.select([sock], [], [], timeout)
                if r:
                    msg_len, msg_buff = self.read_msg_from_server(
                        sock, to_kivy_queue, trigger, msg_len, msg_buff)

                try:
                    while True:
                        msg, value = from_kivy_queue.get_nowait()
                        if msg == 'eof':
                            done = True
                            break
                        else:
                            self.send_msg_to_server(sock, msg, value)
                except Empty:
                    pass
        except Exception as e:
            exc_info = ''.join(traceback.format_exception(*sys.exc_info()))
            to_kivy_queue.put(
                ('exception', (str(e), exc_info)))
            trigger()
        finally:
            Logger.info('closing socket')
            sock.close()

    def send_image(self, image):
        if self.from_kivy_queue is None:
            return

        if self.from_kivy_queue.qsize() < 2:
            self.from_kivy_queue.put(('image', image))

    def send_cam_settings(self, key, value):
        if self.from_kivy_queue is None:
            return
        self.from_kivy_queue.put((key, value))

    @app_error
    def start_listener(self):
        if self.listener_thread is not None:
            return

        from_kivy_queue = self.from_kivy_queue = Queue()
        to_kivy_queue = self.to_kivy_queue = Queue()
        thread = self.listener_thread = Thread(
            target=self.listener_run, args=(from_kivy_queue, to_kivy_queue))
        thread.start()
        self.running = True

    @app_error
    def process_in_kivy_thread(self, *largs):
        while self.to_kivy_queue is not None:
            try:
                msg, value = self.to_kivy_queue.get(block=False)

                if msg == 'exception':
                    e, exec_info = value
                    App.get_running_app().handle_exception(
                        e, exc_info=exec_info)
                    self.stop_listener()
                elif msg == 'track_cam_setting':
                    player = knspace.player
                    if player is not None:
                        player.bind_pt_remote_setting(value)
                elif msg == 'get_cam_settings':
                    player = knspace.player
                    if player is not None:
                        self.send_cam_settings('cam_settings', player.get_valid_pt_settings())
                elif msg == 'set_cam_setting':
                    player = knspace.player
                    if player is not None:
                        player.change_pt_setting_opt(*value)
                elif msg == 'reload_cam_setting':
                    player = knspace.player
                    if player is not None:
                        player.reload_pt_setting_opt(value)
                else:
                    print('got', msg, value)
            except Empty:
                break

    @app_error
    def stop_listener(self):
        if self.listener_thread is None:
            return

        self.from_kivy_queue.put(('eof', None))
        self.listener_thread = self.to_kivy_queue = self.from_kivy_queue = None
        self.running = False

RemoteViewerListener = RemoteViewerListenerBase()
