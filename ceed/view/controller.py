"""Experiment Controller
========================

Runs, controls, and displays Ceed experiments on screen, both during a "real"
experiment and when previewing it.

The purpose of Ceed is to run visual-temporal experiments. Once a
:mod:`~ceed.stage` containing one or more :mod:`~ceed.function`s
and one or more :mod:`~ceed.shape`s has been designed, you're ready to run
the experiment.

Following are some of the experiment configuration options:

.. _view-flip:

Camera-projector-array alignment
--------------------------------

There are three systems interacting with the tissue, and they all need to be
aligned to each other; the projector, the camera, and the MEA electrode grid.

Camera-projector
^^^^^^^^^^^^^^^^

The first step is to draw any unique shape in Ceed and project this pattern
on the MEA plane and then capture the projected pattern using the camera.
Then in the GUI scale and align the captured image to the original shape. This
will give us the camera to projector :attr:`ViewControllerBase.cam_transform`
matrix.

With the camera aligned to the projector output, you can draw shapes and target
specific regions of the slice, visually (from a broad field stimulation camera
capture) and it will be projected at the correct place on the tissue. If
there's mirroring, making affine alignment impossible you can either
:attr:`ViewControllerBase.flip_camera` or
:attr:`ViewControllerBase.flip_projector` horizontally. These settings are
also exposed in the GUI.

Camera-array
^^^^^^^^^^^^

With the camera aligned to the projector, we just need to align the MEA grid to
the camera. First take a camera picture of the grid - you should be able to see
the 2D grid of electrode termination points. Then, in the GUI display the
virtual grid defined by :attr:`ViewControllerBase.mea_num_rows`,
:attr:`ViewControllerBase.mea_num_cols`,
:attr:`ViewControllerBase.mea_pitch`,
:attr:`ViewControllerBase.mea_diameter`, and
:attr:`ViewControllerBase.mirror_mea` and manually align it to the image.
This will generate the :attr:`ViewControllerBase.mea_transform`.

With the array aligned to the image and the projector aligned to the image
we can now know exactly the electrodes on which the drawn shapes will cover
and we can relate the activity of those cells to the stimulation.

.. _view-video-mode:

Video mode
----------

The projector supports 120 (119.96 more accurately) frames per second at its
full resolution of 1920x1080 pixels, but it also offers higher speed modes.

It can split the GPU image into 4 quadrants, such that it renders 4 960x540
pixel images for each overall frame. So at the cost of a half of the x, y
resolution we can play at 480Hz.

Normally each image has red, green, and blue channels. By instead outputting
a grayscale image, we can use each of the three channels to further multiply
our speed by three to render 960x540 grayscale at 1,440 Hz.

The video mode is configurable with :attr:`ViewControllerBase.video_mode`
and from the GUI. Ceed will automatically correctly render the images for each
mode when it is selected.

LED mode
--------

The projector has three LEDs; red, green, and blue. In Ceed you can draw shapes
and select their color(s). Internally, the projector will uses its LEDs to
display the image with the requested colors, like a normal projector.

However, you can manually turn OFF each of these LEDs and then that color will
be displayed. :attr:`ViewControllerBase.LED_mode_idle`
and :attr:`ViewControllerBase.LED_mode` configure which LEDs are active outside
and during an experiment, respectively.

Typically you'd select ``'none'`` for :attr:`ViewControllerBase.LED_mode_idle`
so that the projector is OFF outside the experiment. This way you don't
stimulate the tissue outside the experiment. During the experiment you can
either rely on the color selected for each stage, turn OFF specific LEDs or use
the optical filters to filter out unwanted color channels.

Frame rate and dropped frames
-----------------------------

Frame time
^^^^^^^^^^

The projector and GPU display frames at a specific
:attr:`ViewControllerBase.frame_rate`. In Ceed (GUI) you must enter the exact
GPU frame rate, otherwise Ceed will project the stages at an incorrect rate.
The frame rate will be internally converted to a fraction
(:attr:`ViewControllerBase.frame_rate_numerator`,
:attr:`ViewControllerBase.frame_rate_denominator`) that will be used to clock
the functions (see below).

Normally, the GPU limits us to the frame rate so we don't have to estimate from
software when to display the next frame, because we immediately display the next
frame when the GPU returns control to the CPU. However, if it's not available,
:attr:`ViewControllerBase.use_software_frame_rate` can be used to force the
frame rate. Although it's very unreliable and should not be used during an
experiment.

Long frames
^^^^^^^^^^^

In an ideal system, every frame is displayed for exactly the duration of
the period of the frame_rate before displaying the next frame. In this system
the time of each frame is 0, 1 * period,, 2 * period, ..., n * period.
Since the period is an exact fraction, the current time can be expressed as an
exact fraction and when passed to a stage's function it can accurately determine
when each function is done.

In a real system, some frames may be displayed for more than one frame duration.
This could happen if the CPU is too slow then the current frame is e.g.
displayed for 2 or more frames before the next frame is shown. If this is not
accounted for, all subsequent frames are temporally displaced by the number of
long frames.

For example, say the frame rate and period is exactly 1 second. Normally, we'd
display frames at 0s, 1s, ... ns, and use that time when computing the functions
for each frame (i.e. multiplying the frame count by the period to get the time).
Naively, if we display frame 0 at 0s, 1 at 1s, 2 at 2s. But then frame 2
actually goes long and is displayed for 2 seconds. Because we're counting
frames, the next frame time will be computed as frame 3 at 3s. However, in real
time, because frame 2 was two frames long the actual frame 3 time is 4s when
frame 3 is displayed. So all frames are delayed.

Dropping frames
^^^^^^^^^^^^^^^

To fix this, frame 3 should be dropped and we should instead display frame 4
next. Or more generally, we need to drop frames until the frame number times the
period catches up with the real time.

Ceed has two approaches to detecting when to drop frames; a software approach
and a hardware approach. The software approach uses the time after rendering
frames and a median filter for :class:`FrameEstimation`. With default settings
it may take a few frames before we correct the delay.

We also have a hardware solution using a `Teensy device
<https://github.com/matham/ceed/tree/master/ceed/view/teensy_estimation>`_
for :class:`TeensyFrameEstimation`. This device watches for dropped frames
and notifies us over USB when it happens. This lets us respond more quickly.

The hardware device can be turned OFF with
:attr:`TeensyFrameEstimation.use_teensy`, which is also configurable in the GUI.
If disabled, we fall back to the software approach, unless it's completely
disabled with :attr:`ViewControllerBase.skip_estimated_missed_frames`.

Experiment flow
---------------

pre-computed.

experiment copy to special name

frame rate

Frame warmup
^^^^^^^^^^^^

for projector to update LED and for median estimation.

Experiment control
------------------

There are two sides that communicate over queue.

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

Viewer GUI interaction
^^^^^^^^^^^^^^^^^^^^^^

Keyboard commands are sent to the main process to handle.

Synchronization
---------------

Corner pixel
"""
import multiprocessing as mp
import numpy as np
from decimal import Decimal
import os
import sys
from heapq import heapify, heappop, heappush, heapreplace
from fractions import Fraction
import traceback
from queue import Empty
import uuid
from typing import Optional, Dict, List, Any
from threading import Thread
from tree_config import get_config_children_names
import usb.core
import usb.util
from usb.core import Device as USBDevice, Endpoint
import logging
from ffpyplayer.pic import Image

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

try:
    from pypixxlib import _libdpx as libdpx
    from pypixxlib.propixx import PROPixx
    from pypixxlib.propixx import PROPixxCTRL
except ImportError:
    libdpx = PROPixx = PROPixxCTRL = None

__all__ = (
    'FrameEstimation', 'TeensyFrameEstimation', 'ViewControllerBase',
    'ViewSideViewControllerBase', 'view_process_enter',
    'ControllerSideViewControllerBase', 'ignore_vpixx_import_error'
)

ignore_vpixx_import_error = False
"""Ceed requires the pypixxlib package to control the projector. Ceed can still
run in demo mode with it being installed (it requires hardware drivers to
install) and it will ignore any projector commands.

Set this to True to make it skip the projector commands. E.g. during testing
on the CI.
"""

_get_app = App.get_running_app


class FrameEstimation:
    """A window-based running-median estimator.

    Starting from the first frame, you pass it (:meth:`add_frame`) the time
    just after each frame is rendered. With that, it estimates the time
    the first frame was rendered by estimating the whole number of frames passed
    since :attr:`first_frame_time`, rounding, and then back-computing the first
    frame time from the current frame time, the count, and the GPU period.

    So, each frame gives us an estimate of when the first frame was rendered.
    Next, we keep a :attr:`history` of this estimate from the last 100 frames
    and its median is the best estimate for the actual first frame render time.

    Next, given the best estimate of the first frame render time and the period,
    we compute how many frames have passed and round to whole frames. We record
    this number for the last ``n`` (:attr:`skip_detection_smoothing_n_frames`)
    frames in the circular :attr:`render_times` buffer. Our assumption is that
    starting from the first of the ``n`` frames until the nth frame, we should
    have rendered ``n - 1`` frames.

    Averaging this over the ``n`` frames, so that we are less sensitive to
    individual frame jitter, we get the best estimate of how many frames we
    should have rendered by now, given the start time and the period.
    Additionally, globally, we keep count of the total number of frame actually
    submitted to the GPU and rendered. If our estimate for the number of frames
    we should have rendered is larger than the number of actual rendered,
    we know that some frame took to long to render and we need to drop one or
    more frames to compensate.

    :meth:`add_frame` returns now many frames need to be dropped to catch up.

    Before the first real frame, we do some frame warmup and initialize
    :attr:`render_times` with :meth:`reset`.
    """

    _config_props_ = ('skip_detection_smoothing_n_frames', )

    min_heap: List = []
    """The min heap used to track the median.
    """

    max_heap: List = []
    """The max heap used to track the median.
    """

    history: List = []
    """A circular buffer of 100 items that tracks the estimate of the time that
    the first frame was rendered, using the last 100 frames.

    :attr:`count` is the index in :attr:`history` of the oldest timestamp
    (i.e. the next one to be overwritten).
    """

    count: int = 0
    """Index in :attr:`history`.
    """

    frame_rate: float = 0
    """The GPU frame rate.
    """

    last_render_times_i: int = 0

    render_times: List[float] = []
    """A circular buffer of :attr:`skip_detection_smoothing_n_frames` items
    that tracks the estimate of how many frames should have been rendered,
    using the last :attr:`skip_detection_smoothing_n_frames` frames.

    :attr:`last_render_times_i` is the index in :attr:`render_times` of the
    oldest estimate (i.e. the next one to be overwritten).
    """

    skip_detection_smoothing_n_frames: int = 4
    """How many frames ot average to detect when a frame needs to be skipped.

    See class description.
    """

    smoothing_frame_growth: float = 0.
    """When averaging :attr:`render_times`, we subtract
    :attr:`smoothing_frame_growth`, which is the average over ``range(n)``,
    which is the expected number of frames to added over the last
    :attr:`skip_detection_smoothing_n_frames` frames.

    If the remainder is not zero, it is the number of frames to be dropped.
    """

    first_frame_time: float = 0.
    """The best current estimate of the time that the first experiment frame was
    rendered.
    """

    def reset(self, frame_rate: float, render_times: List[float]) -> None:
        """Resets the instance and initializes it to the render times from the
        warm up frames.
        """
        self.frame_rate = frame_rate
        n = self.skip_detection_smoothing_n_frames
        times = np.asarray(render_times)

        # estimate number of frames between each render and first (expected)
        # render
        n_frames = np.round((times[-1] - times[:-1]) * frame_rate) + 1
        # GPU should force us to multiples of period. Given period, each frame
        # estimates last render time, use median as baseline
        end_time = times[:-1] + n_frames / frame_rate

        self.first_frame_time = float(np.median(end_time))
        # reset for skip detection. Last item will be first real frame
        self.render_times = render_times[-n + 1:] + [-1, ]
        self.last_render_times_i = n - 1

        end_times = np.sort(end_time).tolist()
        max_heap = [-v for v in end_times[:len(end_times) // 2]]
        min_heap = end_times[len(end_times) // 2:]
        heapify(max_heap)
        heapify(min_heap)

        self.max_heap = max_heap
        self.min_heap = min_heap
        self.history = end_time.tolist()
        self.count = len(self.history)

        if n:
            self.smoothing_frame_growth = sum(range(n)) / n
        else:
            self.smoothing_frame_growth = 0

    def update_first_render_time(self, render_time: float) -> None:
        """Adds the frame render time to the running-median history and updates
        :attr:`first_frame_time` with the new best estimate.
        """
        history = self.history
        frame_rate = self.frame_rate
        max_heap = self.max_heap
        min_heap = self.min_heap

        n_frames = round((render_time - self.first_frame_time) * frame_rate)
        new_first_render = render_time - n_frames / frame_rate

        # build up heaps to total 100 items (so it's even)
        if len(history) < 100:
            history.append(new_first_render)
            self.count = (self.count + 1) % 100

            # they can only be one item different
            if len(max_heap) < len(min_heap):
                if new_first_render <= min_heap[0]:
                    heappush(max_heap, -new_first_render)
                else:
                    heappush(max_heap, -heapreplace(min_heap, new_first_render))
                med = (-max_heap[0] + min_heap[0]) / 2
            elif len(max_heap) == len(min_heap):
                if new_first_render <= min_heap[0]:
                    heappush(max_heap, -new_first_render)
                    med = -max_heap[0]
                else:
                    heappush(min_heap, new_first_render)
                    med = min_heap[0]
            else:
                if new_first_render >= -max_heap[0]:
                    heappush(min_heap, new_first_render)
                else:
                    heappush(
                        min_heap, -heapreplace(max_heap, -new_first_render))
                med = (-max_heap[0] + min_heap[0]) / 2
        else:
            # same # items on each heap
            med = (-max_heap[0] + min_heap[0]) / 2

            oldest_val = history[self.count]
            history[self.count] = new_first_render
            self.count = (self.count + 1) % 100

            if oldest_val < min_heap[0]:
                i = max_heap.index(-oldest_val)
                if new_first_render <= min_heap[0]:
                    # replace oldest value with new value
                    max_heap[i] = -new_first_render
                else:
                    # remove oldest from max, replace with min
                    max_heap[i] = -heapreplace(min_heap, new_first_render)
                heapify(max_heap)
            else:
                i = min_heap.index(oldest_val)
                if new_first_render >= -max_heap[0]:
                    # replace oldest value with new value
                    min_heap[i] = new_first_render
                else:
                    # remove oldest from min, replace with max
                    min_heap[i] = -heapreplace(max_heap, -new_first_render)
                heapify(min_heap)

            assert len(min_heap) == len(max_heap)

        self.first_frame_time = med

    def add_frame(
            self, render_time: float, count: int, n_sub_frames: int) -> int:
        """Estimates number of missed frames during experiment, given the render
        time of the last frame and the total frames sent to the GPU.

        ``n_sub_frames`` is the number of sub-frames included in ``count``, e.g.
        in quad mode.

        Can only be called after it is initialized with warmup frames in
        :meth:`reset`.
        """
        self.update_first_render_time(render_time)

        n = self.skip_detection_smoothing_n_frames
        render_times = self.render_times

        render_times[self.last_render_times_i] = render_time
        self.last_render_times_i = (self.last_render_times_i + 1) % n

        # frame number of the first frame in render_times
        frame_n = count // n_sub_frames - n
        start_time = self.first_frame_time
        period = 1 / self.frame_rate
        frame_i = [(t - start_time) / period for t in render_times]
        # number of frames above expected number of frames. Round down
        n_skipped_frames = int(round(
            sum(frame_i) / n - frame_n - self.smoothing_frame_growth))
        n_missed_frames = max(0, n_skipped_frames)

        return n_missed_frames


class TeensyFrameEstimation(EventDispatcher):
    """Alternatively to :class:`FrameEstimation`, we can estimate when the GPU
    rendered a frame for too long and frame needs to be dropped using the
    attached Teensy microcontroller.

    This microcontroller watches the clock bit in the 24-bit corner pixel that
    is described in :class:`~ceed.storage.controller.DataSerializerBase`. Then,
    if a frame change is not seen after 1 / 119.96 seconds after the last clock
    change, we know the frame is going long and we'll need to drop a frame.

    This information is communicated over the USB and this class, in the main
    process but in a second thread, continuously reads the USB. When it
    indicates that a frame needs to be skipped, it updates the
    :attr:`shared_value` that is seen by the second Ceed process that runs the
    experiment and that drops the required number of frames.

    The Teensy can and is only used during an actual experiment when Ceed is run
    from a second process, because otherwise the corner pixel is not visible,
    and the GPU doesn't match the frame rate anyway.
    """

    _config_props_ = ('usb_vendor_id', 'usb_product_id', 'use_teensy')

    usb_vendor_id: int = 0x16C0
    """The Teensy vendor ID. This is how we find the attached Teensy on the
    bus. If there's more than one, this needs to be modified.
    """

    usb_product_id: int = 0x0486
    """The Teensy product ID. This is how we find the attached Teensy on the
    bus. If there's more than one, this needs to be modified.
    """

    use_teensy = BooleanProperty(True)
    """Whether to use the Teensy.

    If it's not attached, set this to False. When False, it falls back on
    :class:`FrameEstimation`.
    """

    is_available = False
    """Indicates whether the Teensy is available and found.

    If :attr:`use_teensy`, but not :attr:`is_available`, then we don't do
    any frame adjustment.
    """

    _magic_header = b'\xAB\xBC\xCD\xDF'
    """USB packet header.
    """

    _start_exp_msg = _magic_header + b'\x01' + b'\x00' * 59
    """Header for packets sent to USB that indicate experiment is starting.
    """

    _end_exp_msg = _magic_header + b'\x02' + b'\x00' * 59
    """Header for packets sent to USB that indicate experiment is ending.
    """

    usb_device: Optional[USBDevice] = None
    """The USB device handle.
    """

    endpoint_out: Optional[Endpoint] = None
    """The output endpoint of the USB that we use to send messages to the
    Teensy.
    """

    endpoint_in: Optional[Endpoint] = None
    """The input endpoint of the USB that we use to read messages from the
    Teensy.
    """

    _stop_thread = False
    """Indicates to thread to end.
    """

    _thread: Optional[Thread] = None

    _reattach_device = False

    shared_value: mp.Value = None
    """A values shared between the main process (that updates this value based
    on the Teensy and indicates the number of frames skipped) and the second
    experiment process that uses this value to drop frames based on how many
    the Teensy thinks were skipped.
    """

    def _endpoint_filter(self, endpoint_type):
        def filt(endpoint):
            return usb.util.endpoint_direction(endpoint.bEndpointAddress) == \
                endpoint_type

        return filt

    def configure_device(self):
        """Configures the Teensy.

        This is called by the main Ceed process before the second process is
        started and opens the device in the main process.
        """
        self.is_available = False
        self._reattach_device = False
        if not self.use_teensy:
            return

        self.usb_device = dev = usb.core.find(
            idVendor=self.usb_vendor_id, idProduct=self.usb_product_id)
        if dev is None:
            raise ValueError(
                'Teensy USB device not found, falling back to time based '
                'missed frame detection')

        if dev.is_kernel_driver_active(0):
            self._reattach_device = True
            dev.detach_kernel_driver(0)

        # use default/first config
        configuration = dev.get_active_configuration()
        interface = configuration[(0, 0)]

        # match the first OUT endpoint
        self.endpoint_out = endpoint_out = usb.util.find_descriptor(
            interface,
            custom_match=self._endpoint_filter(usb.util.ENDPOINT_OUT))

        # match the first IN endpoint
        self.endpoint_in = usb.util.find_descriptor(
            interface, custom_match=self._endpoint_filter(usb.util.ENDPOINT_IN))

        endpoint_out.write(self._end_exp_msg)
        self.is_available = True

    def release_device(self):
        """Releases a previously configured Teensy.

        This is called by the main Ceed process after the second process is
        stopped and closes the device.
        """
        if self.usb_device is not None:
            usb.util.dispose_resources(self.usb_device)
            if self._reattach_device:
                self.usb_device.attach_kernel_driver(0)
            self.usb_device = None
            self.endpoint_in = self.endpoint_out = None

    def start_estimation(self, frame_rate):
        """For each experiment, it notifies the Teensy that a new experiment
        started so that it starts counting skipped frames once it sees the first
        clock toggle in the corner pixel.

        Called by the main Ceed process and it starts a new thread and
        continuously reads from the Teensy and correspondingly updates
        :attr:`shared_value`.
        """
        if frame_rate != 119.96:
            raise ValueError(
                f'Tried to start teensy with a screen frame rate of '
                f'{frame_rate}, but teensy assumes a frame rate of 119.96 Hz')

        if self._thread is not None:
            raise TypeError('Cannot start while already running')

        self._stop_thread = False
        self.shared_value.value = 0

        # reset teensy for sure and then start
        endpoint_out = self.endpoint_out
        endpoint_out.write(self._end_exp_msg)
        endpoint_out.write(self._start_exp_msg)

        endpoint_in = self.endpoint_in
        m1, m2, m3, m4 = self._magic_header

        # make sure to flush packets from before. Device queues
        # up to 1 frame. We should get frames immediately
        flag = 0
        for _ in range(5):
            arr = endpoint_in.read(64)
            h1, h2, h3, h4, flag = arr[:5]
            if h1 != m1 or h2 != m2 or h3 != m3 or h4 != m4:
                raise ValueError('USB packet magic number corrupted')
            # got packet from current (waiting) state
            if flag == 0x01:
                break

        if flag != 0x01:
            raise TypeError('Cannot set Teensy to experiment mode')

        self._thread = Thread(target=self._thread_run)
        self._thread.start()

    @app_error
    def _thread_run(self):
        endpoint_in = self.endpoint_in
        m1, m2, m3, m4 = self._magic_header

        try:
            while not self._stop_thread:
                arr = endpoint_in.read(64)
                h1, h2, h3, h4, flag, a, b, c, value = arr[:9]

                if h1 != m1 or h2 != m2 or h3 != m3 or h4 != m4:
                    raise ValueError('USB packet magic number corrupted')
                if flag != 0x02:
                    continue
                # teensy may know when experiment ended before we are asked to
                # stop so we can't raise error for 0x03 as it may have ended
                # already but we just didn't get the message yet from process

                value <<= 8
                value |= c
                value <<= 8
                value |= b
                value <<= 8
                value |= a

                self.shared_value.value = value
        finally:
            # go back to waiting
            self.endpoint_out.write(self._end_exp_msg)

    def stop_estimation(self):
        """After each experiment it notifies the Teensy that the experiment
        ended so it goes back to waiting for the next experiment notification.

        Called by the main Ceed process and it also stops the second thread
        started by :meth:`start_estimation`.
        """
        if self._thread is None:
            return

        self._stop_thread = True
        self._thread.join()
        self._thread = None


class ViewControllerBase(EventDispatcher):
    """Base class for running a Ceed experiment and visualizing the output of a
    :mod:`ceed.stage` on the projector (full-screen) or during preview.

    There are two sub-classes; :class:`ControllerSideViewControllerBase` for
    playing the experiment when it is previewed in the Ceed GUI and
    :class:`ViewSideViewControllerBase` for playing the experiment in the
    second Ceed process when it is played "for real".

    Additionally, :class:`ControllerSideViewControllerBase` is used to control
    the experiment from within the main Ceed process in each case.
    A base class for visualizing the output of a :mod:`ceed.stage` on the
    projector or to preview it in the main GUI.

    :Events:

        `on_changed`:
            Triggered whenever a configuration option of the class is changed.
    """

    _config_props_ = (
        'screen_width', 'screen_height', 'frame_rate',
        'use_software_frame_rate', 'output_count', 'screen_offset_x',
        'fullscreen', 'video_mode', 'LED_mode', 'LED_mode_idle',
        'mirror_mea', 'mea_num_rows', 'mea_num_cols',
        'mea_pitch', 'mea_diameter', 'mea_transform', 'cam_transform',
        'flip_projector', 'flip_camera', 'pad_to_stage_handshake',
        'pre_compute_stages', 'experiment_uuid', 'log_debug_timing',
        'skip_estimated_missed_frames', 'frame_rate_numerator',
        'frame_rate_denominator',
    )

    _config_children_ = {
        'frame_estimation': 'frame_estimation',
        'teensy_frame_estimation': 'teensy_frame_estimation',
    }

    screen_width = NumericProperty(1920)
    '''The screen width in pixels on which the data is played. This is the
    full-screen width.
    '''

    flip_projector = BooleanProperty(True)
    """Whether to flip the projector output horizontally, around the center.
    See also :ref:`view-flip`.
    """

    flip_camera = BooleanProperty(False)
    """Whether to flip the camera images horizontally, around the center.
    See also :ref:`view-flip`.
    """

    screen_height = NumericProperty(1080)
    '''The screen height in pixels on which the data is played. This is the
    full-screen height.
    '''

    screen_offset_x = NumericProperty(0)
    '''When there are multiple monitors, the monitor on which the experiment is
    shown in full-screen mode is controlled by the x-position of the displayed
    window.

    E.g. to show it on the right monitor of two monitors each 1920 pixel wide,
    and with the main monitor being on the left. Then the
    :attr:`screen_offset_x` should be set to ``1920``.
    '''

    def _get_frame_rate(self):
        return self._frame_rate_numerator / self._frame_rate_denominator

    def _set_frame_rate(self, value):
        self._frame_rate_numerator, self._frame_rate_denominator = Decimal(
            str(value)).as_integer_ratio()

    frame_rate = AliasProperty(
        _get_frame_rate, _set_frame_rate, cache=True,
        bind=('_frame_rate_numerator', '_frame_rate_denominator'))
    '''The frame-rate of the GPU that plays the experiment.

    This should be set to the exact refresh rate of the GPU, as can be found in
    e.g. the nvidia control panel. Otherwise, the experiment will be out of sync
    and played incorrectly.

    This is internally converted to a fraction, so the number must be such
    that it can be converted to a fraction. E.g. 119.96 or 59.94 can be
    represented correctly as fractions.
    '''

    def _get_frame_rate_numerator(self):
        return self._frame_rate_numerator

    def _set_frame_rate_numerator(self, value):
        self._frame_rate_numerator = value

    frame_rate_numerator: int = AliasProperty(
        _get_frame_rate_numerator, _set_frame_rate_numerator, cache=True,
        bind=('_frame_rate_numerator',))
    """The numerator of the :attr:`frame_rate` fraction.
    """

    def _get_frame_rate_denominator(self):
        return self._frame_rate_denominator

    def _set_frame_rate_denominator(self, value):
        self._frame_rate_denominator = value

    frame_rate_denominator: int = AliasProperty(
        _get_frame_rate_denominator, _set_frame_rate_denominator, cache=True,
        bind=('_frame_rate_denominator',))
    """The denominator of the :attr:`frame_rate` fraction.
    """

    _frame_rate_numerator: int = NumericProperty(2999)

    _frame_rate_denominator: int = NumericProperty(25)

    use_software_frame_rate = BooleanProperty(False)
    '''Depending on the CPU/GPU, the software may be unable to render faster
    than the GPU refresh rate. In that case, the GPU limits us to the GPU frame
    rate and :attr:`frame_rate` should be set to match the GPU refresh rate
    and this should be False.

    If the GPU isn't forcing a frame rate. Then this should be True and
    :attr:`frame_rate` should be the desired frame rate. That will restrict us
    the given frame rate. However, the actual frame rate will be wildly
    inaccurate in this mode, so it's only useful for testing.

    One can tell whether the GPU is forcing a frame rate by setting
    :attr:`frame_rate` to a large value and setting
    :attr:`use_software_frame_rate` to False and seeing what the resultant
    frame rate is. If it isn't capped at some value constant, e.g. 120Hz, it
    means that the GPU isn't forcing a rate.
    '''

    log_debug_timing = BooleanProperty(False)
    """Whether to log the times that frames are drawn and rendered to a debug
    section in the h5 file.

    If True, this will additionally be logged for each displayed frame in a
    special section in the file.
    """

    skip_estimated_missed_frames = BooleanProperty(True)
    """Whether to drop frames to compensate when we detect that a previous
    frame was displayed for longer than a single GPU frame duration. Then, we
    may want to drop an equivalent number of frames, rather than
    displaying all the subsequent frames at a delay.

    See :class:`FrameEstimation` and :class:`TeensyFrameEstimation` for how
    we detect these long frames. Use :attr:`TeensyFrameEstimation.use_teensy`
    to control which estimator is used.
    """

    cam_transform = ObjectProperty(Matrix().tolist())
    """A 4x4 matrix that controls the rotation, offset, and scaling of the
    camera images relative to the projector.

    In the Ceed GUI, a user can transform the camera image, in addition to
    :attr:`flip_camera` until it fully aligns with the projector output.
    See also :ref:`view-flip`.
    """

    mea_transform = ObjectProperty(Matrix().tolist())
    """A 4x4 matrix that controls the rotation, offset, and scaling of the
    mea array relative to the camera.

    This is a grid that corresponds to the electrodes in the electrode array.
    In the Ceed GUI, a user can transform this grid, in addition to
    :attr:`mirror_mea` until it fully aligns with a camera image of the grid
    from the actual array.

    See also :ref:`view-flip` and the other ``mea_`` properties of this class.
    """

    mirror_mea = BooleanProperty(True)
    """When True, the MEA grid is mirrored vertically around the center.
    See :attr:`mea_transform` also.
    """

    mea_num_rows = NumericProperty(12)
    """The number of electrode rows in the array. See :attr:`mea_transform`
    also.
    """

    mea_num_cols = NumericProperty(12)
    """The number of electrode columns in the array. See :attr:`mea_transform`
    also.
    """

    mea_pitch = NumericProperty(20)
    """The distance in pixels, center-to-center, between neighboring
    rows/columns. It is assumed that it is the same for columns and rows.

    See :attr:`mea_transform` also.
    """

    mea_diameter = NumericProperty(3)
    """The diameter in pixels of the displayed electrode circles in the grid.

    See :attr:`mea_transform` also.
    """

    pad_to_stage_handshake = BooleanProperty(True)
    """Ad described in :class:`~ceed.storage.controller.DataSerializerBase`,
    Ceed sends handshaking data to the MCS system at the start of each
    experiment. This helps us align the Ceed and MCS data afterwards. If the
    root stage of the experiment is too short, it's possible the full handshake
    would not be sent, preventing alignment afterwards.

    If :attr:`pad_to_stage_handshake`, then the root stage will be padded
    so it goes for the minimum number of clock frames required to finish
    the handshake, if it's too short. The shapes will be black for those
    padded frames.
    """

    output_count = BooleanProperty(True)
    '''Whether the corner pixel is used to output frame information on the
    PROPixx controller IO pot as described in
    :class:`~ceed.storage.controller.DataSerializerBase`.

    If True, :class:`ceed.storage.controller.DataSerializerBase` is used to set
    the 24 bits of the corner pixel. Otherwise, that pixel is treated like the
    other normal pixels.
    '''

    fullscreen = BooleanProperty(True)
    '''Whether the second Ceed window that runs the "real experiment" is run
    in fullscreen mode.

    In fullscreen mode the window has no borders and takes over the whole
    screen.
    '''

    stage_active = BooleanProperty(False)
    '''True when an experiment is being played. Read-only.
    '''

    cpu_fps = NumericProperty(0)
    '''The estimated CPU frames-per-second of the window playing the experiment.
    '''

    gpu_fps = NumericProperty(0)
    '''The estimated GPU frames-per-second of the window playing the experiment.
    '''

    propixx_lib = BooleanProperty(False)
    '''True when the propixx python library (pypixxlib) is available. Read-only.
    '''

    video_modes = ['RGB', 'QUAD4X', 'QUAD12X']
    '''The video modes that the PROPixx projector can be set to.

    See also :ref:`view-video-mode`.
    '''

    led_modes = {'RGB': 0, 'GB': 1, 'RB': 2, 'B': 3, 'RG': 4, 'G': 5, 'R': 6,
                 'none': 7}
    '''The color modes the PROPixx projector can be set to.

    It determines which of the RGB LEDs are turned OFF. E.g. ``"RG"`` means that
    the blue LED is OFF.
    '''

    video_mode = StringProperty('RGB')
    '''The current video mode from among the :attr:`video_modes`.

    See also :ref:`view-video-mode`.
    '''

    LED_mode = StringProperty('RGB')
    '''The LED mode the projector will be set to during the experiment.

    Its value is from the :attr:`led_modes`.
    '''

    LED_mode_idle = StringProperty('none')
    '''The LED mode the projector will be set to before/after the experiment.
    This is used to turn OFF the projector LEDs in between experiments so that
    light is not projected on the tissue while stages are designed.

    Its value is from the :attr:`led_modes`.
    '''

    def _get_do_quad_mode(self):
        return self.video_mode.startswith('QUAD')

    do_quad_mode = AliasProperty(
        _get_do_quad_mode, None, cache=True, bind=('video_mode', ))
    '''Whether the video mode is one of the quad modes. Read-only.
    '''

    pre_compute_stages: bool = BooleanProperty(False)
    """Whether the stage run by the experiment should be pre-computed. See
    :mod:`~ceed.stage` for details.
    """

    _original_fps = Clock._max_fps if not os.environ.get(
        'KIVY_DOC_INCLUDE', None) else 0
    '''Original kivy clock fps, so we can set it back after each experiment.
    '''

    canvas_name = 'view_controller'
    '''Name used for the Kivy canvas to which we add the experiment's graphics
    instructions.
    '''

    current_canvas = None
    '''The last canvas used on which the experiment's shapes, graphics, and
    color instructions was added.
    '''

    shape_views: List[Dict[str, Color]] = []
    '''List of kivy shapes graphics instructions added to the
    :attr:`current_canvas`.

    These are the shape's whose color and intensity is controlled by the
    experiment.
    '''

    tick_event = None
    '''The kivy clock event that updates the shapes' colors on every frame.
    '''

    tick_func = None
    '''The :meth:`~ceed.stage.StageFactoryBase.tick_stage` generator that
    updates the shapes on every frame.
    '''

    count = 0
    '''The current global frame count, reset for each experiment.

    This number divided by the :attr:`frame_rate` is the current global
    experiment time.
    '''

    experiment_uuid: bytes = b''
    """A unique uuid that is re-generated before each experiment and sent along
    over the corner pixel as the initial uniquely-identifying handshake-pattern.
    It allows us to locate this experiment in the MCS data post-hoc.

    See :class:`~ceed.storage.controller.DataSerializerBase`.
    """

    def _get_effective_rate(self):
        rate = Fraction(
            self._frame_rate_numerator, self._frame_rate_denominator)
        if self.video_mode == 'QUAD4X':
            return rate * 4
        elif self.video_mode == 'QUAD12X':
            return rate * 12
        return rate

    effective_frame_rate: Fraction = AliasProperty(
        _get_effective_rate, None, cache=True,
        bind=('video_mode', '_frame_rate_numerator', '_frame_rate_denominator'))
    '''The effective frame rate at which the experiment's shapes is updated.

    E.g. in ``'QUAD4X'`` :attr:`video_mode` shapes are updated at about
    4 * 120Hz = 480Hz.

    It is read only and automatically computed.
    '''

    _cpu_stats = {'last_call_t': 0., 'count': 0, 'tstart': 0.}

    _flip_stats = {'last_call_t': 0., 'count': 0, 'tstart': 0.}

    flip_fps = 0

    serializer = None
    '''The :meth:`ceed.storage.controller.DataSerializerBase.get_bits`
    generator instance that generates the corner pixel value.

    It is advanced for each frame and its value set to the 24-bits of the
    corner pixel.
    '''

    serializer_tex = None
    '''The kivy texture that displays the corner pixel value on screen.
    '''

    queue_view_read: mp.Queue = None
    '''The queue used by the second viewer process side to receive messages
    from the main GUI controller side.
    '''

    queue_view_write: mp.Queue = None
    '''The queue used by the second viewer process side to send messages
    to the main GUI controller side.
    '''

    _scheduled_pos_restore = False
    """Whether we're in the middle of restoring the camera transform.
    """

    _stage_ended_last_frame = False
    """Set when in quad mode, when the last frames only cover some of the 4 or
    12 sub-frames. Then, we still draw those partial frames and only finish the
    experiment on the next tick.
    """

    _frame_buffers = None
    """Buffer used to batch send frame data to the logging system.

    This is data logged by
    :meth:`~ceed.storage.controller.CeedDataWriterBase.add_frame`.
    """

    _frame_buffers_i = 0
    """The index in :attr:`_frame_buffers` where to save the next data.
    """

    _flip_frame_buffer = None
    """Buffer used to batch send frame render data to the logging system.

    This is data logged by
    :meth:`~ceed.storage.controller.CeedDataWriterBase.add_frame_flip`.
    """

    _flip_frame_buffer_i = 0
    """The index in :attr:`_flip_frame_buffer` where to save the next data.
    """

    _debug_frame_buffer = None
    """Buffer used to batch send frame debugging data to the logging system.

    This is data logged by
    :meth:`~ceed.storage.controller.CeedDataWriterBase.add_debug_data`.
    """

    _debug_frame_buffer_i = 0
    """The index in :attr:`_debug_frame_buffer` where to save the next data.
    """

    _debug_last_tick_times = 0, 0
    """Saves the timing info for the last frame.
    """

    _n_missed_frames: int = 0
    """Estimated number of frames missed upto and during the last render
    that we have not yet compensated for by dropping frames.
    """

    _total_missed_frames: int = 0
    """The total number of frames that had to be dropped.
    """

    _n_sub_frames = 1
    """The number of sub-frames within a frame.

    E.g. in quad12 mode this is 12.
    """

    stage_shape_names: List[str] = []
    """List of all the :mod:`~ceed.shape` names used during this experiment.
    """

    frame_estimation: FrameEstimation = None
    """The running-median based frame dropping estimator.

    See :class:`FrameEstimation`.
    """

    teensy_frame_estimation: TeensyFrameEstimation = None
    """The Teensy based frame dropping estimator.

    See :class:`TeensyFrameEstimation`.
    """

    _warmup_render_times: List[float] = []
    """List of the render times of the frames rendered during the experiment
    warmup phase.
    """

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(ViewControllerBase, self).__init__(**kwargs)
        for name in ViewControllerBase._config_props_:
            self.fbind(name, self.dispatch, 'on_changed')
        self.propixx_lib = libdpx is not None
        self.shape_views = []
        self.frame_estimation = FrameEstimation()
        self.teensy_frame_estimation = TeensyFrameEstimation()

    def _restore_cam_pos(self):
        """Resets transformation to the value from before a viewport resize.
        """
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
        """Called during the experiment, either by the second or main Ceed
        process (when previewing) to pass data to the main controller to be
        logged or displayed.

        It is the general interface by which the frame callbacks pass data
        back to the controller.
        """
        pass

    def _process_data(self, data_type, data):
        """The default handler for :meth:`request_process_data` when the
        data generation and logging happens in the same process during preview.

        It simply saves the data as needed.
        """
        if data_type == 'GPU':
            self.gpu_fps = data
        elif data_type == 'CPU':
            self.cpu_fps = data
        elif data_type == 'frame':
            App.get_running_app().ceed_data.add_frame(data)
        elif data_type == 'frame_flip':
            App.get_running_app().ceed_data.add_frame_flip(data)
        elif data_type == 'debug_data':
            App.get_running_app().ceed_data.add_debug_data(*data)
        else:
            assert False

    def add_graphics(self, canvas, black_back=False):
        """Called at the start of the experiment to add all the graphics
        required to visualize the shapes, to the :attr:`current_canvas`.
        """
        _get_app().stage_factory.remove_shapes_gl_color_instructions(
            canvas, self.canvas_name)
        self.shape_views = []
        w, h = self.screen_width, self.screen_height
        half_w = w // 2
        half_h = h // 2

        if black_back:
            with canvas:
                Color(0, 0, 0, 1, group=self.canvas_name)
                Rectangle(size=(w, h), group=self.canvas_name)

        if self.do_quad_mode:

            for (x, y) in ((0, 1), (1, 1), (0, 0), (1, 0)):
                with canvas:
                    PushMatrix(group=self.canvas_name)
                    Translate(x * half_w, y * half_h, group=self.canvas_name)
                    s = Scale(group=self.canvas_name)
                    s.x = s.y = 0.5
                    s.origin = 0, 0

                    if self.flip_projector:
                        s = Scale(group=self.canvas_name)
                        s.x = -1
                        s.origin = half_w, half_h

                instructs = _get_app().\
                    stage_factory.get_shapes_gl_color_instructions(
                    canvas, self.canvas_name)
                with canvas:
                    PopMatrix(group=self.canvas_name)
                self.shape_views.append(instructs)
        else:
            if self.flip_projector:
                with canvas:
                    PushMatrix(group=self.canvas_name)
                    s = Scale(group=self.canvas_name)
                    s.x = -1
                    s.origin = half_w, half_h

            self.shape_views = [
                _get_app().stage_factory.get_shapes_gl_color_instructions(
                    canvas, self.canvas_name)]

            if self.flip_projector:
                with canvas:
                    PopMatrix(group=self.canvas_name)

        if self.output_count and not self.serializer_tex:
            with canvas:
                Color(1, 1, 1, 1, group=self.canvas_name)
                tex = self.serializer_tex = Texture.create(size=(1, 1))
                tex.mag_filter = 'nearest'
                tex.min_filter = 'nearest'
                Rectangle(texture=tex, pos=(0, h - 1), size=(1, 1),
                          group=self.canvas_name)

    def start_stage(self, stage_name: str, canvas):
        """Starts the experiment using the special
        :attr:`~ceed.stage.last_experiment_stage_name` stage.

        It adds the graphics instructions to the canvas, saves it as
        :attr:`current_canvas`, and starts playing the experiment using the
        stage.

        ``stage_name`` is ignored because we use the special stage instead.
        """
        from kivy.core.window import Window
        if self.tick_event:
            raise TypeError('Cannot start new stage while stage is active')

        self.count = 0
        self._stage_ended_last_frame = False
        Clock._max_fps = 0
        self._warmup_render_times = []
        self._n_missed_frames = 0
        self._total_missed_frames = 0

        self._n_sub_frames = 1
        if self.video_mode == 'QUAD4X':
            self._n_sub_frames = 4
        elif self.video_mode == 'QUAD12X':
            self._n_sub_frames = 12

        self.tick_event = Clock.create_trigger(
            self.tick_callback, 0, interval=True)
        self.tick_event()
        Window.fbind('on_flip', self.flip_callback)

        stage_factory: StageFactoryBase = _get_app().stage_factory
        stage = stage_factory.stage_names[last_experiment_stage_name]
        self.stage_shape_names = sorted(stage.get_stage_shape_names())
        stage.pad_stage_ticks = 0

        if self.output_count:
            msg = self.experiment_uuid
            n = len(msg)

            data_serializer = App.get_running_app().data_serializer
            if self.pad_to_stage_handshake:
                n_sub = 1
                if self.video_mode == 'QUAD4X':
                    n_sub = 4
                elif self.video_mode == 'QUAD12X':
                    n_sub = 12
                stage.pad_stage_ticks = data_serializer.num_ticks_handshake(
                    n, n_sub)
            self.serializer = data_serializer.get_bits(msg)
            next(self.serializer)

        self.current_canvas = canvas
        self.tick_func = stage_factory.tick_stage(
            1 / self.effective_frame_rate,
            self.effective_frame_rate, stage_name=last_experiment_stage_name,
            pre_compute=self.pre_compute_stages)
        next(self.tick_func)

        self._flip_stats['last_call_t'] = self._cpu_stats['last_call_t'] = \
            self._cpu_stats['tstart'] = self._flip_stats['tstart'] = clock()
        self._flip_stats['count'] = self._cpu_stats['count'] = 0

        self.add_graphics(canvas)

        self._frame_buffers_i = self._flip_frame_buffer_i = 0

        counter_bits = np.empty(
            512, dtype=[('count', np.uint64), ('bits', np.uint32)])
        shape_rgba = np.empty(
            (512, 4),
            dtype=[(name, np.float16) for name in self.stage_shape_names])
        self._frame_buffers = counter_bits, shape_rgba

        self._flip_frame_buffer = np.empty(
            512, dtype=[('count', np.uint64), ('t', np.float64)])

        self._debug_frame_buffer_i = 0
        self._debug_frame_buffer = np.empty((512, 5), dtype=np.float64)
        self._debug_last_tick_times = 0, 0

    def end_stage(self):
        """Ends the current experiment, if one is running.
        """
        from kivy.core.window import Window
        if not self.tick_event:
            return

        self.tick_event.cancel()
        Window.funbind('on_flip', self.flip_callback)
        Clock._max_fps = self._original_fps
        _get_app().stage_factory.remove_shapes_gl_color_instructions(
            self.current_canvas, self.canvas_name)

        self.tick_func = self.tick_event = self.current_canvas = None
        self.shape_views = []

        self.serializer_tex = None
        self.serializer = None

        # send off any unsent data
        counter_bits, shape_rgba = self._frame_buffers
        i = self._frame_buffers_i
        if i:
            self.request_process_data(
                'frame', (counter_bits[:i], shape_rgba[:i, :]))
        self._frame_buffers = None

        i = self._flip_frame_buffer_i
        if i:
            self.request_process_data('frame_flip', self._flip_frame_buffer[:i])
        self._flip_frame_buffer = None

        if self.log_debug_timing:
            i = self._debug_frame_buffer_i
            if i:
                self.request_process_data(
                    'debug_data', ('timing', self._debug_frame_buffer[:i, :]))
            self._debug_frame_buffer = None

    def tick_callback(self, *largs):
        """Called for every CPU Clock frame to handle any processing work.

        If not :attr:`use_software_frame_rate` and if the GPU restricts
        the CPU to the GPU refresh rate, then this is called once before
        each frame is rendered so we can update the projector at the expected
        frame rate.

        Before the experiment starts for real we do 50 empty warmup frames.
        Warmup is required to ensure the projector LED had time to change to the
        experiment value :attr:`LED_mode` (compared to :attr:`LED_mode_idle`).
        In addition to allowing us to estimate the time of the first experiment
        frame render for :class:`FrameEstimation`.
        """
        # are we finishing up in quad mode after there were some partial frame
        # at the end of last iteration so we couldn't finish then?
        if self._stage_ended_last_frame:
            self._stage_ended_last_frame = False
            self.count += 1
            self.end_stage()
            return

        # are we still warming up? We always warm up, even if frames not used
        if not self.count:
            if len(self._warmup_render_times) < 50:
                # make sure we flip the frame to record render time
                self.current_canvas.ask_update()
                return

        # warmup period done, estimate params after first post-warmup frame
        if not self.count and self.skip_estimated_missed_frames \
                and not self.use_software_frame_rate:
            self.frame_estimation.reset(
                frame_rate=self.frame_rate,
                render_times=self._warmup_render_times)

        t = clock()
        stats = self._cpu_stats
        tdiff = t - stats['last_call_t']

        stats['count'] += 1
        if t - stats['tstart'] >= 1:
            fps = stats['count'] / (t - stats['tstart'])
            self.request_process_data('CPU', fps)
            stats['tstart'] = t
            stats['count'] = 0

        if self.use_software_frame_rate and tdiff < 1 / self.frame_rate:
            return

        stats['last_call_t'] = t

        tick = self.tick_func
        if self.video_mode == 'QUAD4X':
            projections = [None, ] * 4
            # it already has 4 views
            views = self.shape_views
        elif self.video_mode == 'QUAD12X':
            projections = (['r', ] * 4) + (['g', ] * 4) + (['b', ] * 4)
            views = self.shape_views * 3
        else:
            projections = [None, ]
            views = self.shape_views

        effective_rate = self.effective_frame_rate
        # in software mode this is always zero. For skipped frames serializer is
        # not ticked
        for _ in range(self._n_missed_frames):
            for proj in projections:
                # we cannot skip frames (i.e. we may only increment frame by
                # one). Because stages/func can be pre-computed and it assumes
                # a constant frame rate. If need to skip n frames, tick n times
                # but don't draw result
                self.count += 1

                try:
                    shape_values = tick.send(self.count / effective_rate)
                except StageDoneException:
                    self.end_stage()
                    return
                except Exception:
                    self.end_stage()
                    raise

                values = _get_app().stage_factory.fill_shape_gl_color_values(
                    None, shape_values, proj)

                stage_shape_names = self.stage_shape_names
                counter_bits, shape_rgba = self._frame_buffers
                i = self._frame_buffers_i
                counter_bits['count'][i] = self.count
                counter_bits['bits'][i] = 0
                for name, r, g, b, a in values:
                    if name in stage_shape_names:
                        shape_rgba[name][i, :] = r, g, b, a
                i += 1

                if i == 512:
                    self.request_process_data(
                        'frame', (counter_bits, shape_rgba))
                    self._frame_buffers_i = 0
                else:
                    self._frame_buffers_i = i

        first_blit = True
        bits = 0
        for k, (shape_views, proj) in enumerate(zip(views, projections)):
            self.count += 1

            try:
                shape_values = tick.send(self.count / effective_rate)
            except StageDoneException:
                # we're on the last and it's a partial frame (some sub-frames
                # were rendered), set remaining shapes of frame to black
                if k:
                    # we'll increment it again at next frame
                    self.count -= 1
                    self._stage_ended_last_frame = True
                    for colors, rem_proj in zip(views[k:], projections[k:]):
                        if rem_proj is None:
                            # in quad4 we just set rgba to zero
                            for color in colors.values():
                                color.rgba = 0, 0, 0, 0
                        else:
                            # in quad12 we only set the unused color channels
                            for color in colors.values():
                                setattr(color, rem_proj, 0)
                    break

                self.end_stage()
                return
            except Exception:
                self.end_stage()
                raise

            if self.serializer:
                if first_blit:
                    bits = self.serializer.send(self.count)
                    # if in e.g. quad mode, only blit on first section
                    r, g, b = bits & 0xFF, (bits & 0xFF00) >> 8, \
                        (bits & 0xFF0000) >> 16
                    self.serializer_tex.blit_buffer(
                        bytes([r, g, b]), colorfmt='rgb', bufferfmt='ubyte')
                    first_blit = False
            else:
                bits = 0

            values = _get_app().stage_factory.fill_shape_gl_color_values(
                shape_views, shape_values, proj)

            stage_shape_names = self.stage_shape_names
            counter_bits, shape_rgba = self._frame_buffers
            i = self._frame_buffers_i
            counter_bits['count'][i] = self.count
            counter_bits['bits'][i] = bits
            for name, r, g, b, a in values:
                if name in stage_shape_names:
                    shape_rgba[name][i, :] = r, g, b, a
            i += 1

            if i == 512:
                self.request_process_data(
                    'frame', (counter_bits, shape_rgba))
                self._frame_buffers_i = 0
            else:
                self._frame_buffers_i = i

        self.current_canvas.ask_update()
        if self.log_debug_timing:
            self._debug_last_tick_times = t, clock()

    def flip_callback(self, *largs):
        """Called for every GPU rendered frame by the graphics system.

        This method lets us estimate the rendering times and if we need to drop
        frames.
        """
        ts = clock()
        from kivy.core.window import Window
        Window.on_flip()

        t = clock()
        # count of zero is discarded as it's during warmup
        if not self.count:
            # but do record the render time
            self._warmup_render_times.append(t)
            return True

        if self.skip_estimated_missed_frames \
                and not self.use_software_frame_rate:
            # doesn't make sense in software mode

            teensy = self.teensy_frame_estimation
            if teensy.use_teensy:
                if teensy.shared_value is not None:
                    skipped = teensy.shared_value.value
                    self._n_missed_frames = max(
                        0, skipped - self._total_missed_frames)
                    self._total_missed_frames = skipped
            else:
                time_based_n = self.frame_estimation.add_frame(
                    t, self.count, self._n_sub_frames)
                self._n_missed_frames = time_based_n
                self._total_missed_frames += time_based_n

        buffer = self._flip_frame_buffer
        i = self._flip_frame_buffer_i
        buffer['count'][i] = self.count
        buffer['t'][i] = t
        i += 1

        if i == 512:
            self.request_process_data('frame_flip', buffer)
            self._flip_frame_buffer_i = 0
        else:
            self._flip_frame_buffer_i = i

        stats = self._flip_stats
        stats['count'] += 1
        if t - stats['tstart'] >= 1:
            fps = stats['count'] / (t - stats['tstart'])
            self.request_process_data('GPU', fps)
            stats['tstart'] = t
            stats['count'] = 0

        stats['last_call_t'] = t

        if self.log_debug_timing:
            if self.count:
                buffer = self._debug_frame_buffer
                i = self._debug_frame_buffer_i
                buffer[i, :] = self.count, *self._debug_last_tick_times, ts, t
                i += 1

                if i == 512:
                    self.request_process_data('debug_data', ('timing', buffer))
                    self._debug_frame_buffer_i = 0
                else:
                    self._debug_frame_buffer_i = i
        return True


class ViewSideViewControllerBase(ViewControllerBase):
    """This class is used for experiment control when Ceed is running a
    real experiment in the second Ceed process.

    If Ceed is running in the second process started with
    :func:`view_process_enter`, then this is a "real" experiment and this class
    is used. It has a inter-process queue from which it gets messages from the
    main Ceed process, such as to start or stop an experiment. It also sends
    back messages to the main process including data about the rendered frames
    and data to be logged.
    """

    def start_stage(self, stage_name, canvas):
        self.prepare_view_window()
        super(ViewSideViewControllerBase, self).start_stage(
            stage_name, canvas)

    def end_stage(self):
        d = {}
        d['pixels'], d['proj_size'] = App.get_running_app().get_root_pixels()
        d['proj_size'] = tuple(d['proj_size'])

        super(ViewSideViewControllerBase, self).end_stage()
        self.queue_view_write.put_nowait(('end_stage', d))

    def request_process_data(self, data_type, data):
        if data_type == 'frame':
            counter_bits, shape_rgba = data
            self.queue_view_write.put_nowait(
                (data_type, (counter_bits.tobytes(), shape_rgba.tobytes())))
        elif data_type == 'frame_flip':
            self.queue_view_write.put_nowait((data_type, data.tobytes()))
        elif data_type == 'debug_data':
            name, arr = data
            self.queue_view_write.put_nowait(
                (data_type, (name, arr.tobytes(), arr.dtype, arr.shape)))
        else:
            assert data_type in ('CPU', 'GPU')
            self.queue_view_write.put_nowait((data_type, str(data)))

    def send_keyboard_down(self, key, modifiers, t):
        """Gets called by the window for every keyboard key press, which it
        sends on to the main GUI process to handle.
        """
        self.queue_view_write.put_nowait((
            'key_down', yaml_dumps((key, t, list(modifiers)))))

    def send_keyboard_up(self, key, t):
        """Gets called by the window for every keyboard key release, which it
        sends on to the main GUI process to handle.
        """
        self.queue_view_write.put_nowait(('key_up', yaml_dumps((key, t))))

    def handle_exception(self, exception, exc_info=None):
        """Called upon an error which is passed on to the main process.
        """
        if exc_info is not None and not isinstance(exc_info, str):
            exc_info = ''.join(traceback.format_exception(*exc_info))
        self.queue_view_write.put_nowait(
            ('exception', yaml_dumps((str(exception), exc_info))))

    @app_error
    def view_read(self, *largs):
        """Communication between the two process occurs through queues. This
        method is run periodically by the Kivy Clock to serve the queue and
        read and handle messages from the main GUI.
        """
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
        """Called before :class:`~ceed.view.main.CeedViewApp` is run, to
        prepare the new window according to the configuration parameters.
        """
        from kivy.core.window import Window
        Window.size = self.screen_width, self.screen_height
        Window.left = self.screen_offset_x
        Window.fullscreen = self.fullscreen


def view_process_enter(
        read: mp.Queue, write: mp.Queue, settings: Dict[str, Any],
        app_settings: dict, shared_value: mp.Value):
    """Entry method for the second Ceed process that runs "real" experiments.

    It is called by this process when it is created. This in turns configures
    the app and then runs it until it's closed.

    The experiment is run in this process by
    :class:`ViewSideViewControllerBase`. It receives control messages and sends
    back data to the main process over the provided queues.
    :class:`ControllerSideViewControllerBase` handles these queues on the main
    process side.
    """
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
        viewer.teensy_frame_estimation.shared_value = shared_value

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
    """This class is used by the main Ceed process to control experiments
    run either as previews (in the main Ceed process) or as a real experiment
    (in a second process).

    If the experiment is run in the second process, then that second process
    runs :class:`ViewSideViewControllerBase` and this class is used by the
    main process to send control messages and receive experiment data from that
    process over queues.

    Otherwise, this class directly controls the experiment.
    """

    view_process: Optional[mp.Process] = ObjectProperty(None, allownone=True)
    '''The second process that runs "real" experiments in full-screen mode.
    See :func:`view_process_enter`.
    '''

    _ctrl_down = False
    '''True when ctrl is pressed down in the viewer side.
    '''

    selected_stage_name = ''
    '''The name of the stage currently selected in the GUI to be run.

    This will be the stage that is copied and run.
    '''

    initial_cam_image: Optional[Image] = None
    """The last camera image received before the experiment starts, if any.

    See also :attr:`last_cam_image`.

    It is only set for a "real" experiment, not during preview.
    """

    last_cam_image: Optional[Image] = ObjectProperty(None, allownone=True)
    """After the experiment ends, this contains the last camera image acquired
    before the experiment ended. If no image was taken during the experiment,
    this is the image from before the experiment if there's one.

    This allows us to keep the last image generated by the tissue in response
    to experiment stimulation. In the GUI, after the experiment ended, there's
    a button which when pressed will take this image (if not None) and set it
    as the camera image.

    It is only set for a "real" experiment, not during preview.

    See also :attr:`proj_pixels`.
    """

    proj_size = None
    """If :attr:`last_cam_image` is an image and not None, this contains the
    screen size from which the :attr:`proj_pixels` were generated.

    It's the second index value of the tuple returned by
    :meth:`~ceed.view.main.CeedViewApp.get_root_pixels`.

    It is only set for a "real" experiment, not during preview.
    """

    proj_pixels = None
    """If :attr:`last_cam_image` is an image and not None, this contains the
    pixel intensity values for all the pixels shown during the last frame before
    the experiment ended.

    Together with :attr:`last_cam_image`, this lets you compare the pixels
    displayed on the projector to the image from the tissue lighting up in
    response to those pixels.

    It's the first index value of the tuple returned by
    :meth:`~ceed.view.main.CeedViewApp.get_root_pixels`.

    It is only set for a "real" experiment, not during preview.
    """

    _last_ctrl_release = 0

    def add_graphics(self, canvas, black_back=True):
        return super().add_graphics(canvas, black_back=black_back)

    @app_error
    def request_stage_start(
            self, stage_name: str, experiment_uuid: Optional[bytes] = None
    ) -> None:
        """Starts the experiment using the stage, either running it in the GUI
        when previewing or in the second process.

        This internally calls the appropriate
        :meth:`ViewControllerBase.start_stage` method either for
        :class:`ViewSideViewControllerBase` or
        :class:`ControllerSideViewControllerBase` so this should be used to
        start the experiment.
        """
        # Look into immediately erroring out if already running. So that we
        # don't overwrite the initial image if we're already running.
        # needs to be set here so button is reset on fail
        self.stage_active = True
        self.last_cam_image = self.proj_pixels = self.proj_size = None
        self.initial_cam_image = None
        if not stage_name:
            self.stage_active = False
            raise ValueError('No stage specified')

        if experiment_uuid is None:
            self.experiment_uuid = uuid.uuid4().bytes
        else:
            self.experiment_uuid = experiment_uuid

        app = App.get_running_app()
        app.stages_container.\
            copy_and_resample_experiment_stage(stage_name)
        app.dump_app_settings_to_file()
        app.load_app_settings_from_file()
        self.stage_shape_names = sorted(
            app.stage_factory.stage_names[stage_name].get_stage_shape_names())
        app.ceed_data.prepare_experiment(stage_name, self.stage_shape_names)

        if self.propixx_lib:
            self.set_video_mode(self.video_mode)
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
            # we only do teensy estimation on the second process
            self.teensy_frame_estimation.shared_value.value = 0
            if self.teensy_frame_estimation.is_available:
                self.teensy_frame_estimation.start_estimation(self.frame_rate)

            self.initial_cam_image = app.player.last_image
            self.queue_view_read.put_nowait(
                ('config', yaml_dumps(app.ceed_data.gather_config_data_dict())))
            self.queue_view_read.put_nowait(('start_stage', stage_name))
        else:
            self.stage_active = False
            raise ValueError('Already running stage')

    @app_error
    def request_stage_end(self):
        """Ends the currently running experiment, whether it's running in the
        GUI when previewing or in the second process.

        This internally calls the appropriate
        :meth:`ViewControllerBase.end_stage` method either for
        :class:`ViewSideViewControllerBase` or
        :class:`ControllerSideViewControllerBase` so this should be used to
        stop the experiment.
        """
        if self.view_process is None:
            self.end_stage()
        elif self.queue_view_read is not None:
            self.last_cam_image = App.get_running_app().player.last_image
            if self.last_cam_image is self.initial_cam_image:
                self.last_cam_image = None
            self.queue_view_read.put_nowait(('end_stage', None))

    def stage_end_cleanup(self, state=None):
        """Automatically called by Ceed after a :meth:`request_stage_end`
        request and it cleans up any resources and finalizes the last
        experiment.
        """
        # we only do teensy estimation on the second process
        if self.teensy_frame_estimation.is_available:
            self.teensy_frame_estimation.stop_estimation()

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
            self.set_pixel_mode(False, ignore_exception=True)
            self.set_led_mode(self.LED_mode_idle, ignore_exception=True)

    @app_error
    def end_stage(self):
        val = super(ControllerSideViewControllerBase, self).end_stage()
        self.stage_end_cleanup()

    def request_fullscreen(self, state):
        """Sets the :attr:`fullscreen` state of the second Ceed process.
        """
        self.fullscreen = state
        if self.view_process and self.queue_view_read:
            self.queue_view_read.put_nowait(('fullscreen', state))

    def request_process_data(self, data_type, data):
        # When we're not going IPC, we need to copy the buffers
        if data_type == 'frame':
            counter_bits, shape_rgba = data
            data = counter_bits.copy(), shape_rgba.copy()
        elif data_type == 'frame_flip':
            data = data.copy()
        elif data_type == 'debug_data':
            name, arr = data
            data = name, arr.copy()
        else:
            assert data_type in ('CPU', 'GPU')

        self._process_data(data_type, data)

    @app_error
    def start_process(self):
        """Starts the second Ceed process that runs the "real" experiment
        using :class:`ViewSideViewControllerBase`.
        """
        if self.view_process:
            return

        self.teensy_frame_estimation.shared_value = None
        self.teensy_frame_estimation.configure_device()

        App.get_running_app().dump_app_settings_to_file()
        App.get_running_app().load_app_settings_from_file()
        settings = {name: getattr(self, name)
                    for name in ViewControllerBase._config_props_}

        ctx = mp.get_context('spawn') if not PY2 else mp
        shared_value = self.teensy_frame_estimation.shared_value = ctx.Value(
            'i', 0)
        r = self.queue_view_read = ctx.Queue()
        w = self.queue_view_write = ctx.Queue()
        os.environ['CEED_IS_VIEW'] = '1'
        os.environ['KCFG_GRAPHICS_VSYNC'] = '1'
        self.view_process = process = ctx.Process(
            target=view_process_enter,
            args=(r, w, settings, App.get_running_app().app_settings,
                  shared_value))
        process.start()
        del os.environ['CEED_IS_VIEW']
        Clock.schedule_interval(self.controller_read, .25)

    def stop_process(self):
        """Ends the :class:`view_process` process by sending a EOF to
        the second process.
        """
        if self.view_process and self.queue_view_read:
            self.queue_view_read.put_nowait(('eof', None))
            self.queue_view_read = None

    @app_error
    def finish_stop_process(self):
        """Automatically called by Ceed through the read queue when we receive
        the message that the second process received the :meth:`stop_process`
        EOF and that it stopped.
        """
        if not self.view_process:
            return

        self.view_process.join()
        self.view_process = self.queue_view_read = self.queue_view_write = None
        Clock.unschedule(self.controller_read)

        self.teensy_frame_estimation.shared_value = None
        self.teensy_frame_estimation.release_device()

    def handle_key_press(self, key, t, modifiers=[], down=True):
        """Called by by the read queue thread when we receive a keypress
        event from the second process.

        In response it e.g. starts/stops the experiment, closes the second
        process etc.
        """
        if key in ('ctrl', 'lctrl', 'rctrl'):
            self._ctrl_down = down
            if not down:
                self._last_ctrl_release = t
        if (not self._ctrl_down and t - self._last_ctrl_release > .1) or down:
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
        """Called periodically by the Kivy Clock to serve the queue that
        receives messages from the second Ceed process.
        """
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
                elif msg in ('GPU', 'CPU'):
                    self._process_data(msg, float(value))
                elif msg == 'frame':
                    counter_bits, shape_rgba = value

                    counter_bits = np.frombuffer(
                        counter_bits,
                        dtype=[('count', np.uint64), ('bits', np.uint32)])
                    shape_rgba = np.frombuffer(
                        shape_rgba,
                        dtype=[(name, np.float16)
                               for name in self.stage_shape_names])
                    shape_rgba = shape_rgba.reshape(-1, 4)

                    self._process_data(msg, (counter_bits, shape_rgba))
                elif msg == 'frame_flip':
                    decoded = np.frombuffer(
                        value, dtype=[('count', np.uint64), ('t', np.float64)])

                    self._process_data(msg, decoded)
                elif msg == 'debug_data':
                    name, data, dtype, shape = value
                    decoded = np.frombuffer(data, dtype=dtype)
                    decoded = decoded.reshape(shape)

                    self._process_data(msg, (name, decoded))
                elif msg == 'end_stage' and msg != 'response':
                    self.stage_end_cleanup(value)
                elif msg == 'key_down':
                    self.handle_key_press(*yaml_loads(value))
                elif msg == 'key_up':
                    self.handle_key_press(*yaml_loads(value), down=False)
            except Empty:
                break

    @app_error
    def set_pixel_mode(self, state, ignore_exception=False):
        """Sets the projector pixel mode to show the corner pixel on the
        controller IO.

        It is called with True before the experiment starts and with False
        when it ends.
        """
        if PROPixxCTRL is None:
            if ignore_vpixx_import_error:
                return
            raise ImportError('Cannot open PROPixx library')

        try:
            ctrl = PROPixxCTRL()
        except Exception as e:
            if not ignore_exception:
                raise
            else:
                logging.error(e)
                return

        if state:
            ctrl.dout.enablePixelMode()
        else:
            ctrl.dout.disablePixelMode()
        ctrl.updateRegisterCache()
        ctrl.close()

    @app_error
    def set_led_mode(self, mode, ignore_exception=False):
        """Sets the projector's LED mode to one of the
        :attr:`ViewControllerBase.led_modes`.
        """
        if libdpx is None:
            if ignore_vpixx_import_error:
                return
            raise ImportError('Cannot open PROPixx library')

        libdpx.DPxOpen()
        if not libdpx.DPxSelectDevice('PROPixx'):
            if ignore_exception:
                return
            raise TypeError('Cannot set projector LED mode. Is it ON?')
        libdpx.DPxSetPPxLedMask(self.led_modes[mode])
        libdpx.DPxUpdateRegCache()
        libdpx.DPxClose()

    @app_error
    def set_video_mode(self, mode, ignore_exception=False):
        """Sets the projector's video mode to one of the
        :attr:`ViewControllerBase.video_modes`.
        """
        if PROPixx is None:
            if ignore_vpixx_import_error:
                return
            raise ImportError('Cannot open PROPixx library')

        modes = {'RGB': 'RGB 120Hz', 'QUAD4X': 'RGB Quad 480Hz',
                 'QUAD12X': 'GREY Quad 1440Hz'}
        try:
            dev = PROPixx()
        except Exception as e:
            if not ignore_exception:
                raise
            else:
                logging.error(e)
                return

        dev.setDlpSequencerProgram(modes[mode])
        dev.updateRegisterCache()
        dev.close()
