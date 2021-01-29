import pytest
import pathlib

from .examples.experiment import create_basic_experiment, run_experiment
from ceed.tests.ceed_app import CeedTestApp

pytestmark = pytest.mark.ceed_app


async def test_function_plugin_source_in_data_file(
        stage_app: CeedTestApp, tmp_path):
    import ceed.function.plugin
    src_contents = pathlib.Path(ceed.function.plugin.__file__).read_bytes()

    stage = await create_basic_experiment(stage_app)

    f = await run_experiment(
        stage_app, stage.name, tmp_path, num_clock_frames=10)

    target_root = tmp_path / 'test_dump_target'
    target_root.mkdir()

    with f:
        f.dump_plugin_sources('function', target_root)
    plugin = target_root / 'ceed.function.plugin' / '__init__.py'
    dumped_contents = plugin.read_bytes()

    assert dumped_contents == src_contents


async def test_stage_plugin_source_in_data_file(
        stage_app: CeedTestApp, tmp_path):
    import ceed.stage.plugin
    src_contents = pathlib.Path(ceed.stage.plugin.__file__).read_bytes()

    stage = await create_basic_experiment(stage_app)

    f = await run_experiment(
        stage_app, stage.name, tmp_path, num_clock_frames=10)

    target_root = tmp_path / 'test_dump_target'
    target_root.mkdir()

    with f:
        f.dump_plugin_sources('stage', target_root)
    plugin = target_root / 'ceed.stage.plugin' / '__init__.py'
    dumped_contents = plugin.read_bytes()

    assert dumped_contents == src_contents
