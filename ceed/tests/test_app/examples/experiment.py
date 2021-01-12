
from ceed.analysis import CeedDataReader
from ceed.tests.ceed_app import CeedTestApp
from ceed.tests.test_app.examples.stages import SerialAllStage, StageWrapper
from ceed.tests.test_app.examples.funcs import LinearFunctionF1
from .shapes import CircleShapeP1

__all__ = ('create_basic_experiment', 'run_experiment')


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
