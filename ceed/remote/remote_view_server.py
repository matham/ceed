'''Ceed View App
=====================

The module that runs the :mod:`ceed.view` GUI for displaying the pixels on the
projector. This is run in a seperate process than the main server side GUI.
'''
import os
from ffpyplayer.pic import Image
from ffpyplayer.tools import get_supported_pixfmts
from ffpyplayer.writer import MediaWriter
from functools import partial
from itertools import accumulate
import os
import numpy as np
from os.path import join, dirname, isdir
from threading import Thread
import socket
import sys
import struct
from time import clock
from queue import Queue, Empty
import traceback
import select

from base_kivy_app.app import run_app as run_cpl_app, app_error, BaseKivyApp
from base_kivy_app.utils import yaml_dumps, yaml_loads
from cplcom.player import Player as cplcom_player

from kivy.app import App
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, \
    BooleanProperty, ListProperty
from kivy.uix.slider import Slider
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.clock import Clock

try:
    from pypixxlib import _libdpx as libdpx
    from pypixxlib.propixx import PROPixx
    from pypixxlib.propixx import PROPixxCTRL
except ImportError:
    libdpx = PROPixx = PROPixxCTRL = None

if __name__ == '__main__':
    # needed to be imported before iron python
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


class EndConnection(Exception):
    pass


connection_errors = (EndConnection, ConnectionAbortedError, ConnectionResetError)


class TSICamera(EventDispatcher):

    supported_freqs = ListProperty(['20 MHz', ])

    freq = StringProperty('20 MHz')

    supported_taps = ListProperty(['1', ])

    taps = StringProperty('1')

    supports_color = BooleanProperty(False)

    exposure_range = ListProperty([0, 100])

    exposure_ms = NumericProperty(5)

    binning_x_range = ListProperty([0, 0])

    binning_x = NumericProperty(0)

    binning_y_range = ListProperty([0, 0])

    binning_y = NumericProperty(0)

    sensor_size = ListProperty([0, 0])

    roi_x = NumericProperty(0)

    roi_y = NumericProperty(0)

    roi_width = NumericProperty(0)

    roi_height = NumericProperty(0)

    gain_range = ListProperty([0, 100])

    gain = NumericProperty(0)

    black_level_range = ListProperty([0, 100])

    black_level = NumericProperty(0)

    frame_queue_size = NumericProperty(1)

    supported_triggers = ListProperty(['SW Trigger', 'HW Trigger'])

    trigger_type = StringProperty('SW Trigger')

    trigger_count = NumericProperty(1)

    num_queued_frames = NumericProperty(0)

    playing = BooleanProperty(False)

    color_gain = [1, 1, 1]

    tsi_sdk = None

    tsi_interface = None

    serial = ''

    app = None

    _kivy_trigger = None

    from_kivy_queue = None

    to_kivy_queue = None

    camera_thread = None

    _freqs_to_str_map = {}

    _str_to_freqs_map = {}

    _taps_to_str_map = {}

    _str_to_taps_map = {}

    _update_group = {}

    def __init__(self, tsi_sdk, tsi_interface, serial, app):
        self.tsi_sdk = tsi_sdk
        self.tsi_interface = tsi_interface
        self.serial = serial
        self.app = app
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

        roi_bin = cam.get_ROIAndBin()

        rang = cam.get_BinXRange()
        d['binning_x_range'] = rang.Minimum, rang.Maximum
        d['binning_x'] = roi_bin.BinX

        rang = cam.get_BinYRange()
        d['binning_y_range'] = rang.Minimum, rang.Maximum
        d['binning_y'] = roi_bin.BinY

        d['sensor_size'] = [
            cam.get_SensorWidth_pixels(), cam.get_SensorHeight_pixels()]

        d['roi_x'] = roi_bin.ROIOriginX_pixels
        d['roi_y'] = roi_bin.ROIOriginY_pixels
        d['roi_width'] = roi_bin.ROIWidth_pixels
        d['roi_height'] = roi_bin.ROIHeight_pixels

        d['frame_queue_size'] = cam.get_MaximumNumberOfFramesToQueue()
        d['trigger_count'] = cam.get_FramesPerTrigger_zeroForUnlimited()
        hw_mode = cam.get_OperationMode() == self.tsi_interface.OperationMode.HardwareTriggered
        d['trigger_type'] = self.supported_triggers[1 if hw_mode else 0]

        rang = cam.get_GainRange()
        d['gain_range'] = rang.Minimum, rang.Maximum
        d['gain'] = cam.get_Gain()

        rang = cam.get_BlackLevelRange()
        d['black_level_range'] = rang.Minimum, rang.Maximum
        d['black_level'] = cam.get_BlackLevel()

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

        d['color_gain'] = self.color_gain
        return d

    def write_setting(self, cam, setting, value):
        values = {}
        if setting == 'exposure_ms':
            value = int(max(min(value, self.exposure_range[1]),
                        self.exposure_range[0]) * 1000)
            cam.set_ExposureTime_us(value)
            value = value / 1000.
        elif setting == 'binning_x':
            value = max(min(value, self.binning_x_range[1]), self.binning_x_range[0])
            roi_bin = cam.get_ROIAndBin()
            roi_bin.BinX = value
            cam.set_ROIAndBin(roi_bin)
        elif setting == 'binning_y':
            value = max(min(value, self.binning_y_range[1]), self.binning_y_range[0])
            roi_bin = cam.get_ROIAndBin()
            roi_bin.BinY = value
            cam.set_ROIAndBin(roi_bin)
        elif setting == 'roi_x':
            x = value = max(0, min(value, self.sensor_size[0] - 1))
            roi_bin = cam.get_ROIAndBin()
            width = min(self.sensor_size[0] - x, roi_bin.ROIWidth_pixels)

            roi_bin.ROIOriginX_pixels = x
            roi_bin.ROIWidth_pixels = width
            cam.set_ROIAndBin(roi_bin)
            values['roi_width'] = width
        elif setting == 'roi_y':
            y = value = max(0, min(value, self.sensor_size[1] - 1))
            roi_bin = cam.get_ROIAndBin()
            height = min(self.sensor_size[1] - y, roi_bin.ROIHeight_pixels)

            roi_bin.ROIOriginY_pixels = y
            roi_bin.ROIHeight_pixels = height
            cam.set_ROIAndBin(roi_bin)
            values['roi_height'] = height
        elif setting == 'roi_width':
            roi_bin = cam.get_ROIAndBin()
            x = roi_bin.ROIOriginX_pixels
            value = max(1, min(value, self.sensor_size[0] - x))
            roi_bin.ROIWidth_pixels = value
            cam.set_ROIAndBin(roi_bin)
            values['roi_x'] = x
        elif setting == 'roi_height':
            roi_bin = cam.get_ROIAndBin()
            y = roi_bin.ROIOriginY_pixels
            value = max(1, min(value, self.sensor_size[1] - y))
            roi_bin.ROIHeight_pixels = value
            cam.set_ROIAndBin(roi_bin)
            values['roi_y'] = y
        elif setting == 'trigger_type':
            hw_mode = value == self.supported_triggers[1]
            cam.set_OperationMode(
                self.tsi_interface.OperationMode.HardwareTriggered if hw_mode else
                self.tsi_interface.OperationMode.SoftwareTriggered)
        elif setting == 'trigger_count':
            cam.set_FramesPerTrigger_zeroForUnlimited(max(0, value))
        elif setting == 'frame_queue_size':
            cam.set_MaximumNumberOfFramesToQueue(max(1, value))
        elif setting == 'gain':
            value = int(max(min(value, self.gain_range[1]), self.gain_range[0]))
            cam.set_Gain(value)
        elif setting == 'black_level':
            value = int(max(min(value, self.black_level_range[1]), self.black_level_range[0]))
            cam.set_BlackLevel(value)
        elif setting == 'freq':
            cam.set_DataRate(self._str_to_freqs_map[value])
        elif setting == 'taps' and value:
            cam.set_Taps(self._str_to_taps_map[value])
        elif setting == 'color_gain':
            r, g, b = value
            mat = [r, 0, 0, 0, g, 0, 0, 0, b]
            color_pipeline = self.tsi_interface.ColorPipeline()
            color_pipeline.set_ColorMode(self.tsi_interface.ColorMode.StandardRGB)
            color_pipeline.InsertColorTransformMatrix(0, mat)
            color_pipeline.InsertColorTransformMatrix(1, cam.GetCameraColorCorrectionMatrix())
            cam.set_ColorPipelineOrNull(color_pipeline)
        values[setting] = value
        return values

    def read_frame(self, cam, asNumpyArray):
        queued_count = cam.get_NumberOfQueuedFrames()
        if queued_count <= 0:
            return

        frame = cam.GetPendingFrameOrNull()
        t = clock()
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
        return img, count, queued_count, t

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
            to_kivy_queue.put(('open', None))
            trigger()
            self.write_setting(cam, 'c_gain', (10, 0, 0))

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
                                to_kivy_queue.put(('settings', {'playing': False}))
                                trigger()
                                break
                            elif msg == 'setting':
                                to_kivy_queue.put(('settings', self.write_setting(cam, *value)))
                                trigger()
                        except Empty:
                            break

                    if not playing or msg == 'eof':
                        continue
                    data = self.read_frame(cam, asNumpyArray)
                    if data is None:
                        continue

                    to_kivy_queue.put(('image', data))
                    trigger()
                    send_image('remote_image', data[0], max_qsize=2)

                else:
                    msg, value = from_kivy_queue.get(block=True)
                    if msg == 'eof':
                        break
                    elif msg == 'play':
                        cam.Arm()
                        if self.trigger_type == self.supported_triggers[0]:
                            cam.IssueSoftwareTrigger()
                        playing = True
                        to_kivy_queue.put(('settings', {'playing': True}))
                        trigger()
                    elif msg == 'setting':
                        to_kivy_queue.put(('settings', self.write_setting(cam, *value)))
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
                    img, frame_num, self.num_queued_frames, t = value
                    App.get_running_app().process_acquired_image(img, frame_num, t)
                elif msg == 'open':
                    App.get_running_app().num_images_sent = 0
                elif msg == 'settings':
                    for key, val in value.items():
                        setattr(self, key, val)
                else:
                    assert False
            except Empty:
                break


class CeedRemoteViewApp(BaseKivyApp):
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

    last_image = ObjectProperty(None, allownone=True)

    num_images_received = NumericProperty(0)

    num_images_sent = NumericProperty(0)

    image_compression_save = StringProperty('raw')

    image_save_path = StringProperty('')

    stream_image_save = BooleanProperty(False)

    image_suffix_save = StringProperty('_#.tiff')

    image_queue_len = NumericProperty(0)

    max_play_rate = 4

    _last_frame_play_time = 0

    estimated_frame_rate = NumericProperty(0)

    _fps_frame_time_buffer = None

    _fps_buffer_next_frame_index = 0

    _fps_buffer_num_used = 0

    image_save_thread = None

    image_save_queue = None

    thor_api_binaries = r'D:\daghda-to-tuatha\thor_api_binaries'

    vpixx_thread = None

    vpixx_thread_queue = None

    def init_load(self):
        pass

    def build(self):
        return Builder.load_file(os.path.join(os.path.dirname(__file__), 'ceed_server.kv'))

    def on_start(self, *largs):
        self._kivy_trigger = Clock.create_trigger(self.process_in_kivy_thread)
        from_kivy_queue = self.from_kivy_queue = Queue()
        to_kivy_queue = self.to_kivy_queue = Queue()
        image_save_queue = self.image_save_queue = Queue()
        vpixx_thread_queue = self.vpixx_thread_queue = Queue()

        server_thread = self.server_thread = Thread(
            target=self.server_run, args=(from_kivy_queue, to_kivy_queue, vpixx_thread_queue))

        image_save_thread = self.image_save_thread = Thread(
            target=self.image_save_run,
            args=(image_save_queue, to_kivy_queue))

        vpixx_thread = self.vpixx_thread = Thread(
            target=self.vpixx_run,
            args=(vpixx_thread_queue, to_kivy_queue, from_kivy_queue))

        server_thread.start()
        image_save_thread.start()
        vpixx_thread.start()

        if not os.path.exists(self.thor_api_binaries):
            raise Exception(
                'Could not find the path to the thor binaries at {}'
                    .format(self.thor_api_binaries))
        self.load_tsi(self.thor_api_binaries)

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
        tsi_interface.ColorPipeline

    def increment_images_sent(self, *largs):
        self.num_images_sent += 1

    def process_acquired_image(self, image, frame_number, frame_time):
        self.num_images_received = frame_number
        self.last_image = image

        i = self._fps_buffer_next_frame_index
        buffer = self._fps_frame_time_buffer
        n = len(buffer)
        if self._fps_buffer_num_used == n:
            fps = n / (frame_time - buffer[i])
            buffer[i] = frame_time
        else:
            fps = i / (frame_time - buffer[0]) if i else 0
            buffer[i] = frame_time
            self._fps_buffer_num_used += 1

        i += 1
        self._fps_buffer_next_frame_index = 0 if i == n else i
        self.estimated_frame_rate = fps

        t = clock()
        if t - self._last_frame_play_time > 1 / self.max_play_rate:
            self._last_frame_play_time = t
            self.root.ids.img.update_img(image)

        if self.stream_image_save:
            self.image_save_queue.put((
                'image',
                (image, self.image_save_path,
                 frame_number, self.image_compression_save, True, self.image_suffix_save)
            ))
            self.image_queue_len += 1

    @app_error
    def save_last_image(self):
        self.image_save_queue.put((
            'image',
            (self.last_image, self.image_save_path,
             self.num_images_received, self.image_compression_save, False, '_#.tiff')
        ))

    @app_error
    def stop_streaming_disk(self):
        self.image_save_queue.put(('stop_stream', None))

    def image_save_run(self, image_save_queue, to_kivy_queue):
        trigger = self._kivy_trigger
        exists = os.path.exists
        save_image = cplcom_player.save_image
        media_writer = None
        last_suffix = ''
        movie_base_frame = 0

        try:
            while True:
                msg, value = image_save_queue.get(block=True)
                if msg == 'eof':
                    break
                elif msg == 'stop_stream':
                    if media_writer is not None:
                        media_writer.close()
                        media_writer = None
                    last_suffix = ''
                elif msg == 'image':
                    image, root_filename, frame_num, compression, seq, suffix = value

                    # we're streaming a sequence or movie now, but the type of stream changed
                    if seq and suffix != last_suffix:
                        if suffix.endswith('tiff'):  # we stopped the movie and are instead streaming the tiffs
                            if media_writer is not None:
                                media_writer.close()
                                media_writer = None
                        else:  # we started streaming a movie
                            lib_opts = {'crf': '0'}
                            codec = 'libx264'
                            ifmt = image.get_pixel_format()
                            w, h = image.get_size()
                            out_fmt = get_supported_pixfmts(codec, ifmt)[0]
                            stream = {
                                'pix_fmt_in': ifmt, 'pix_fmt_out': out_fmt,
                                'width_in': w, 'height_in': h, 'width_out': w,
                                'height_out': h, 'codec': codec,
                                'frame_rate': (1, 1)}

                            ending = '_{}.mp4'.format(frame_num)
                            filename = root_filename + ending
                            counter = 0
                            while exists(filename):
                                counter += 1
                                filename = '{}-{}{}'.format(root_filename, counter, ending)

                            movie_base_frame = frame_num
                            media_writer = MediaWriter(filename, [stream], fmt='mp4', lib_opts=lib_opts)

                        last_suffix = suffix

                    # we're saving a tiff, either a seq or single
                    if suffix.endswith('tiff'):
                        ending = '_{}.tiff'.format(frame_num)
                        filename = root_filename + ending
                        counter = 0
                        while exists(filename):
                            counter += 1
                            filename = '{}-{}{}'.format(root_filename, counter, ending)

                        save_image(
                            filename, image, codec='tiff',
                            pix_fmt=image.get_pixel_format(),
                            lib_opts={'compression_algo': 'deflate'
                                      if compression == 'zip' else compression})
                    else:  # we're streaming a movie
                        media_writer.write_frame(img=image, pts=frame_num - movie_base_frame, stream=0)

                    # we're streaming seq of tiff or movie so we counted the frames
                    if seq:
                        to_kivy_queue.put(('decrement_image_save', None))
                        trigger()
                else:
                    assert False
        except Exception as e:
            exc_info = ''.join(traceback.format_exception(*sys.exc_info()))
            to_kivy_queue.put(
                ('exception', (str(e), exc_info)))
            trigger()
        finally:
            Logger.info('closing image saving thread')

    @app_error
    def get_tsi_cams(self):
        cams = self.tsi_sdk.DiscoverAvailableCameras()
        names = []
        for i in range(cams.get_Count()):
            names.append(cams.get_Item(i))
        return list(sorted(cams))

    def play_tsi_cam(self):
        self._fps_buffer_next_frame_index = 0
        self._fps_buffer_num_used = 0
        self._fps_frame_time_buffer = [None, ] * 30
        self.tsi_cam.send_message('play')

    def open_tsi_cam(self, serial):
        self.tsi_cam = TSICamera(
            tsi_sdk=self.tsi_sdk, tsi_interface=self.tsi_interface,
            serial=serial, app=self)

    def close_tsi_cam(self, join=False):
        tsi_cam = self.tsi_cam
        if tsi_cam:
            self.tsi_cam = None
            tsi_cam.send_message('eof')
            if join and tsi_cam.camera_thread is not None:
                tsi_cam.camera_thread.join()

    def vpixx_run(self, vpixx_thread_queue, to_kivy_queue, network_response_queue):
        trigger = self._kivy_trigger
        if PROPixx is None:
            try:
                raise Exception('Could not import the PROPixx library')
            except Exception as e:
                lib_exc_info = ''.join(traceback.format_exception(*sys.exc_info()))
                lib_e = str(e)
        else:
            lib_exc_info = lib_e = ''

        try:
            while True:
                msg, value = vpixx_thread_queue.get(block=True)
                if msg == 'eof':
                    break
                elif msg == 'vpixx.command':
                    if PROPixx is None:
                        network_response_queue.put(
                            ('server_exception', (lib_e, lib_exc_info)))
                        continue

                    opt, settings = value
                    if opt == 'led_mode':
                        name, mode = settings
                        libdpx.DPxOpen()
                        libdpx.DPxSelectDevice(name)
                        libdpx.DPxSetPPxLedMask(mode)
                        libdpx.DPxUpdateRegCache()
                        libdpx.DPxClose()
                    elif opt == 'video_mode':
                        mode, = settings
                        dev = PROPixx()
                        dev.setDlpSequencerProgram(mode)
                        dev.updateRegisterCache()
                        dev.close()
                    elif opt == 'pixel_mode':
                        state, = settings
                        ctrl = PROPixxCTRL()
                        if state:
                            ctrl.dout.enablePixelMode()
                        else:
                            ctrl.dout.disablePixelMode()
                        ctrl.updateRegisterCache()
                        ctrl.close()
                else:
                    assert False
        except Exception as e:
            exc_info = ''.join(traceback.format_exception(*sys.exc_info()))
            to_kivy_queue.put(
                ('exception', (str(e), exc_info)))
            trigger()
        finally:
            Logger.info('closing vpixx thread')

    def send_msg_to_client(self, sock, msg, value):
        if msg == 'remote_image':
            Clock.schedule_once(App.get_running_app().increment_images_sent)
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
            self, sock, to_kivy_queue, vpixx_thread_queue, trigger,
            msg_len, msg_buff):
        # still reading msg size
        if not msg_len:
            assert 8 - len(msg_buff)
            data = sock.recv(8 - len(msg_buff))
            if not data:
                raise EndConnection('Remote client was closed')

            msg_buff += data
            if len(msg_buff) == 8:
                msg_len = struct.unpack('>II', msg_buff)
                msg_buff = b''
        else:
            total = sum(msg_len)
            assert total - len(msg_buff)
            data = sock.recv(total - len(msg_buff))
            if not data:
                raise EndConnection('Remote client was closed')

            msg_buff += data
            if len(msg_buff) == total:
                msg, value = self.decode_data(msg_buff, msg_len)
                if msg.startswith('vpixx.'):
                    vpixx_thread_queue.put((msg, value))
                else:
                    to_kivy_queue.put((msg, value))
                    trigger()

                msg_len = ()
                msg_buff = b''
        return msg_len, msg_buff

    def server_run(self, from_kivy_queue, to_kivy_queue, vpixx_thread_queue):
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

                def connected(*largs):
                    App.get_running_app().num_images_sent = 0
                    self.connected = True
                Clock.schedule_once(connected)
                from_kivy_queue.put(('get_cam_settings', ''))
                from_kivy_queue.put(('track_cam_setting', self.cam_setting))

                try:
                    while True:
                        r, _, _ = select.select([connection], [], [], timeout)
                        if r:
                            msg_len, msg_buff = self.read_msg_from_client(
                                connection, to_kivy_queue, vpixx_thread_queue,
                                trigger, msg_len, msg_buff)

                        try:
                            while True:
                                msg, value = from_kivy_queue.get_nowait()
                                if msg == 'eof':
                                    return
                                else:
                                    self.send_msg_to_client(connection, msg, value)
                        except Empty:
                            pass
                except connection_errors:
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
                    self.last_image = img
                    self.root.ids.img.update_img(img)
                elif msg == 'cam_settings':
                    self.root.ids['pt_settings_opt'].values = value
                elif msg == 'cam_setting':
                    self.populate_settings(*value)
                elif msg == 'decrement_image_save':
                    self.image_queue_len -= 1
                else:
                    print('got', msg, value)
                    assert False
            except Empty:
                break

    def _ask_close(self, *largs, **kwargs):
        return False

    def get_logger(self):
        return Logger

    def clean_up(self):
        super(CeedRemoteViewApp, self).clean_up()
        if self.from_kivy_queue is not None:
            self.from_kivy_queue.put(('eof', None))
            if self.server_thread is not None:
                self.server_thread.join()

        self.close_tsi_cam(join=True)

        if self.image_save_queue is not None:
            self.image_save_queue.put(('eof', None))
            if self.image_save_thread is not None:
                self.image_save_thread.join()

        if self.vpixx_thread_queue is not None:
            self.vpixx_thread_queue.put(('eof', None))
            if self.vpixx_thread is not None:
                self.vpixx_thread.join()


run_app = partial(run_cpl_app, CeedRemoteViewApp)
'''The function that starts the GUI and the entry point for
the main script.
'''

if __name__ == '__main__':
    run_app()
