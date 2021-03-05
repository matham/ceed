from itertools import cycle, chain
from time import perf_counter

from ceed.storage.controller import DataSerializerBase
from ceed.analysis import CeedDataReader
from ceed.tests.ceed_app import CeedTestApp
from ceed.tests.test_app.examples.stages import SerialAllStage, StageWrapper
from ceed.tests.test_app.examples.funcs import LinearFunctionF1
from .shapes import CircleShapeP1


async def create_basic_experiment(
        stage_app: CeedTestApp, add_function=True, add_shape=True
) -> StageWrapper:
    """Creates a stage and optionally adds a simple function and shape to the
    stage and shows the stage/function/shape in the GUI.

    Also sets ups controller to use a low frame rate with no software timing.
    """
    shapes = []
    if add_shape:
        shapes = [CircleShapeP1(
            app=None, painter=stage_app.app.shape_factory, show_in_gui=True)]

    functions = []
    if add_function:
        functions = [LinearFunctionF1(app=stage_app, show_in_gui=True)]

    root = SerialAllStage(
        app=stage_app, show_in_gui=True, functions=functions, shapes=shapes)

    stage_app.app.view_controller.frame_rate = 10
    stage_app.app.view_controller.use_software_frame_rate = False

    await stage_app.wait_clock_frames(2)

    return root


async def run_experiment(
        stage_app: CeedTestApp, name: str, tmp_path, num_clock_frames=None
) -> CeedDataReader:
    """Runs the named stage until done, or number of frames passed, if provided
    and returns a reader for the resulting data file.
    """
    ts = await stage_app.wait_clock_frames(0)

    stage_app.app.view_controller.request_stage_start(name)
    while stage_app.app.view_controller.stage_active:
        te = await stage_app.wait_clock_frames(5)

        if num_clock_frames and te - ts >= num_clock_frames:
            break

    if stage_app.app.view_controller.stage_active:
        stage_app.app.view_controller.request_stage_end()
    await stage_app.wait_clock_frames(2)

    filename = str(tmp_path / 'saved_data.h5')
    stage_app.app.ceed_data.save(filename=filename)

    f = CeedDataReader(filename)

    return f


def set_serializer_even_count_bits(
        serializer: DataSerializerBase, n_sub_frames):
    serializer_config_bytes = [
        0b00000001, 0b00010010, 0b00100011, 0b00110100, 0b01000101, 0b01010110,
        0b01100111
    ]

    serializer.clock_idx = 3
    serializer.count_indices = [0, 2, 4, 5]
    serializer.short_count_indices = [1, 6, 8]
    serializer.counter_bit_width = 32

    num_handshake_ticks = 3 * 2 * 8 * n_sub_frames

    counter = [
        # length of config is 2 ints, duplicated
        0b0000_0100, 0b0000_0100,
        0b0000_0000, 0b0000_0000,
        0b0000_0000, 0b0000_0000,
        0b0000_0000, 0b0000_0000,
        0b0000_0000, 0b0000_0000,
        0b0000_0000, 0b0000_0000,
        0b0000_0000, 0b0000_0000,
        0b0000_0000, 0b0000_0000,
        # first 4 config bytes
        0b0000_0001, 0b0000_0001,
        0b0000_0000, 0b0000_0000,
        0b0000_0100, 0b0000_0100,
        0b0000_0001, 0b0000_0001,
        0b0000_0101, 0b0000_0101,
        0b0000_0100, 0b0000_0100,
        0b0001_0000, 0b0001_0000,
        0b0000_0101, 0b0000_0101,
        # next 3 config bytes
        0b0001_0001, 0b0001_0001,
        0b0001_0000, 0b0001_0000,
        0b0001_0100, 0b0001_0100,
        0b0001_0001, 0b0001_0001,
        0b0001_0101, 0b0001_0101,
        0b0001_0100, 0b0001_0100,
        0b0000_0000, 0b0000_0000,
        0b0000_0000, 0b0000_0000,
    ]
    # counter is incremented once per sub-frame, short counter is the same
    # for all sub-frames
    if n_sub_frames == 1:
        counter += [
            # counter is now 49 frames (started at 1)
            0b0000_0001, 0b0000_0001,
            0b0000_0101, 0b0011_0000,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            # counter is now 65 frames
            0b0000_0001, 0b0000_0001,
            0b0001_0000, 0b0010_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
        ]
    elif n_sub_frames == 4:
        counter += [
            # counter is now 193 frames (started at 1)
            0b0000_0001, 0b0000_0001,
            0b0011_0000, 0b0000_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            # counter is now 257 frames
            0b0000_0001, 0b0000_0001,
            0b0000_0000, 0b0011_0101,
            0b0000_0001, 0b0011_0100,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
        ]
    elif n_sub_frames == 12:
        counter += [
            # counter is now 577 frames (started at 1)
            0b0000_0001, 0b0000_0001,
            0b0001_0000, 0b0010_0101,
            0b0000_0100, 0b0011_0001,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            # counter is now 767 frames
            0b0000_0001, 0b0000_0001,
            0b0000_0000, 0b0011_0101,
            0b0000_0101, 0b0011_0000,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
            0b0000_0000, 0b0011_0101,
        ]

    clock_values = cycle([0b1000, 0])
    short_values = cycle([
        0b0_0000_0000,
        0b0_0000_0010,
        0b0_0100_0000,
        0b0_0100_0010,
        0b1_0000_0000,
        0b1_0000_0010,
        0b1_0100_0000,
        0b1_0100_0010,
    ])

    return bytes(serializer_config_bytes), num_handshake_ticks, counter, \
        short_values, clock_values


async def wait_stage_experiment_started(stage_app: CeedTestApp, timeout=50):
    ts = perf_counter()
    while not stage_app.app.view_controller.count:
        if perf_counter() - ts >= timeout:
            raise TimeoutError

        await stage_app.wait_clock_frames(2)

    await stage_app.wait_clock_frames(2)


async def wait_experiment_done(stage_app: CeedTestApp, timeout=50):
    ts = perf_counter()
    while stage_app.app.view_controller.stage_active:
        if perf_counter() - ts >= timeout:
            raise TimeoutError

        await stage_app.wait_clock_frames(2)

    await stage_app.wait_clock_frames(2)


async def measure_fps(stage_app: CeedTestApp, max_fps=120.) -> int:
    from kivy.clock import Clock
    original_fps = Clock._max_fps
    Clock._max_fps = 0
    times = []

    def callback(*args):
        times.append(perf_counter())
        stage_app.app.root.canvas.ask_update()

    try:
        event = Clock.create_trigger(callback, timeout=0, interval=True)
        event()

        await stage_app.wait_clock_frames(40)
        event.cancel()
    finally:
        Clock._max_fps = original_fps

    assert len(times) > 30
    diff = sorted([t2 - t1 for t1, t2 in zip(times[:-1], times[1:])])
    fps = max(diff[len(diff) // 2], 1 / max_fps)
    return int(round(1 / fps))
