import pytest
from math import isclose
import numpy as np
import pathlib

import ceed
from .examples.shapes import CircleShapeP1, CircleShapeP2
from .examples import assert_image_same, create_test_image
from .examples.experiment import create_basic_experiment, run_experiment
from ceed.tests.ceed_app import CeedTestApp
from ceed.tests.test_stages import get_stage_time_intensity
from ceed.analysis.merge_data import CeedMCSDataMerger
from ceed.tests.test_app.examples.stages import ParaAllStage

stored_images = []
stored_b_values = [(.0, .2), (.1, .3)]
stored_shape_names = CircleShapeP1.name, CircleShapeP2.name
# r, g is active, b is inactive
stored_colors = [((0, 1, ), (2,)), ] * 2
filename_source = [
    'internal', 'external', 're_merged', 'merged', 'unmerged']
filename_source_merged = ['re_merged', 'merged']
stored_stage_name = ParaAllStage.name

pytestmark = pytest.mark.ceed_app


def verify_experiment(values, n, first):
    shape1, shape2 = stored_shape_names
    if first:
        b1, b2 = stored_b_values[0]
    else:
        b1, b2 = stored_b_values[1]
    assert n == 240

    for i in range(240):
        for name, b, (active, inactive) in zip(
                (shape1, shape2), (b1, b2), stored_colors):
            d = values[name]
            assert d[i][3] == 1
            for j in inactive:
                assert d[i][j] == 0

            # 2.4 = .6 * 120 / 30
            val = .6 * (i % 30) / 30 + b
            for j in active:
                assert isclose(float(d[i][j]), val, abs_tol=.001)


def exp_source(filename):
    filename = pathlib.Path(filename).name
    return not ('internal' in filename or 'external' in filename), \
        'merged' in filename


@pytest.fixture(scope="module", autouse=True)
def init_written_data():
    global stored_images
    stored_images = [
        create_test_image(500, 500), create_test_image(250, 500),
        create_test_image(500, 250), create_test_image(250, 200)
    ]


async def run_data_experiment(stage_app: CeedTestApp):
    from ..test_stages import create_2_shape_stage
    from ceed.function.plugin import LinearFunc

    root, s1, s2, shape1, shape2 = create_2_shape_stage(
        stage_app.app.stage_factory, show_in_gui=True, app=stage_app)
    await stage_app.wait_clock_frames(2)

    # 30 frames
    f1 = LinearFunc(
        function_factory=stage_app.app.function_factory, duration=.25, loop=8,
        m=2.4
    )
    f2 = LinearFunc(
        function_factory=stage_app.app.function_factory, duration=.25, loop=8,
        m=2.4
    )
    s1.stage.add_func(f1)
    s2.stage.add_func(f2)
    await stage_app.wait_clock_frames(2)

    stage_app.app.view_controller.frame_rate = 120
    # count frames
    stage_app.app.view_controller.use_software_frame_rate = False
    stage_app.app.view_controller.pad_to_stage_handshake = True

    for image, (b1, b2) in zip(stored_images, stored_b_values):
        f1.b = b1
        f2.b = b2

        # set background image
        stage_app.app.central_display.update_img(image)
        stage_app.app.player.last_image = image
        await stage_app.wait_clock_frames(2)

        stage_app.app.view_controller.request_stage_start(root.name)
        while stage_app.app.view_controller.stage_active:
            await stage_app.wait_clock_frames(5)
        await stage_app.wait_clock_frames(2)

    for i, image in enumerate(stored_images[2:]):
        stage_app.app.ceed_data.add_image_to_file(image, f'image {i}')
        await stage_app.wait_clock_frames(2)


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


@pytest.fixture(scope='module')
def internal_experiment_filename(tmp_path_factory):
    """All tests depending on this, also depend on
    test_create_internal_experiment."""
    return str(
        tmp_path_factory.mktemp('experiment') / 'new_experiment_internal.h5')


@pytest.fixture(scope='module')
def external_experiment_filename(tmp_path_factory):
    """All tests depending on this, also depend on
    test_create_external_experiment."""
    return str(
        tmp_path_factory.mktemp('experiment') / 'new_experiment_external.h5')


@pytest.fixture(scope='module')
def existing_experiment_filename():
    root = pathlib.Path(ceed.__file__).parent.joinpath('examples', 'data')
    return str(root.joinpath('ceed_data.h5'))


@pytest.fixture(scope='module')
def merge_experiment_filename(tmp_path_factory, existing_experiment_filename):
    """All tests depending on this, also depend on
    test_create_merge_experiment."""
    return str(
        tmp_path_factory.mktemp('experiment') / 'new_experiment_merged.h5')


@pytest.fixture(scope='module')
def existing_template_filename():
    root = pathlib.Path(ceed.__file__).parent.joinpath('examples', 'data')
    return str(root.joinpath('ceed_template.yml'))


@pytest.fixture(scope='module')
def existing_merged_experiment_filename():
    root = pathlib.Path(ceed.__file__).parent.joinpath('examples', 'data')
    return str(root.joinpath('ceed_mcs_data_merged.h5'))


@pytest.fixture(scope='module', params=filename_source)
def experiment_filename(
        request, internal_experiment_filename, external_experiment_filename,
        merge_experiment_filename, existing_experiment_filename,
        existing_merged_experiment_filename):
    src = request.param
    if src == 'internal':
        return internal_experiment_filename
    if src == 'external':
        return external_experiment_filename
    if src == 'internal_merged':
        return merge_experiment_filename
    if src == 'merged':
        return existing_merged_experiment_filename
    return existing_experiment_filename


@pytest.fixture(scope='module', params=filename_source_merged)
def merged_filename(
        request, merge_experiment_filename,
        existing_merged_experiment_filename):
    src = request.param
    if src == 're_merged':
        return merge_experiment_filename
    return existing_merged_experiment_filename


async def test_create_internal_experiment(
        stage_app: CeedTestApp, internal_experiment_filename):
    filename = internal_experiment_filename

    await run_data_experiment(stage_app)

    stage_app.app.ceed_data.save(filename=filename)
    base = filename[:-3]
    stage_app.app.ceed_data.write_yaml_config(base + '.yml', stages_only=True)
    stage_app.app.ceed_data.write_yaml_config(
        base + 'app.yml', stages_only=False)


async def test_create_external_experiment(
        stage_app: CeedTestApp, external_experiment_filename):
    filename = external_experiment_filename
    stage_app.app.view_controller.start_process()
    await stage_app.wait_clock_frames(2)

    await run_data_experiment(stage_app)

    stage_app.app.view_controller.stop_process()
    await stage_app.wait_clock_frames(2)

    stage_app.app.ceed_data.save(filename=filename)


def test_create_merge_experiment(
        merge_experiment_filename, existing_experiment_filename):
    filename = merge_experiment_filename

    root = pathlib.Path(ceed.__file__).parent.joinpath('examples', 'data')
    mcs_filename = str(root.joinpath('mcs_data.h5'))

    merger = CeedMCSDataMerger()
    merger.read_mcs_digital_data(mcs_filename)

    first = True
    alignment = {}
    for experiment in ('0', '1'):
        merger.read_ceed_digital_data(existing_experiment_filename, experiment)
        merger.parse_ceed_digital_data()

        if first:
            merger.parse_mcs_digital_data()
            first = False

        alignment[experiment] = merger.get_alignment()

    merger.merge_data(
        filename, existing_experiment_filename, mcs_filename, alignment,
        notes='app notes')


def test_saved_metadata(experiment_filename):
    from ceed.analysis import CeedDataReader
    existing_exp, merged_exp = exp_source(experiment_filename)

    with CeedDataReader(experiment_filename) as f:

        def verify_app_props():
            assert f.filename == experiment_filename
            assert f.experiments_in_file == ['0', '1']
            assert f.num_images_in_file == 2
            if merged_exp:
                assert f.app_notes == 'app notes'
            else:
                assert not f.app_notes

            for name in {
                    'view_controller', 'data_serializer', 'function_factory',
                    'shape_factory', 'stage_factory'}:
                assert f.app_config[name] is not None
        verify_app_props()

        assert f.view_controller is None
        assert f.data_serializer is None
        assert f.function_factory is None
        assert f.stage_factory is None
        assert f.shape_factory is None

        assert f.loaded_experiment is None
        assert f.experiment_cam_image is None
        assert not f.experiment_stage_name
        assert not f.experiment_notes
        assert not f.external_function_plugin_package
        assert not f.external_stage_plugin_package

        assert not f.electrodes_data
        assert not f.electrodes_metadata
        assert f.electrode_dig_data is None
        assert f.electrode_intensity_alignment is None

        mcs_props = [
            'electrodes_data', 'electrodes_metadata', 'electrode_dig_data']
        if merged_exp:
            f.load_mcs_data()
        else:
            with pytest.raises(TypeError):
                f.load_mcs_data()
        mcs_values = [getattr(f, name) for name in mcs_props]

        if merged_exp:
            for name in mcs_props:
                assert getattr(f, name) is not None and len(getattr(f, name))
        else:
            assert not f.electrodes_data
            assert not f.electrodes_metadata
            assert f.electrode_dig_data is None
        assert f.electrode_intensity_alignment is None
        alignment = f.electrode_intensity_alignment

        exp_props = [
            'view_controller', 'data_serializer', 'function_factory',
            'stage_factory', 'shape_factory']
        exp_values = [getattr(f, name) for name in exp_props]

        for exp in (0, 1):
            f.load_experiment(exp)

            verify_app_props()
            for name, value in zip(mcs_props, mcs_values):
                assert getattr(f, name) is value
            if merged_exp:
                assert f.electrode_intensity_alignment is not None
                assert len(f.electrode_intensity_alignment)
                assert f.electrode_intensity_alignment is not alignment
                alignment = f.electrode_intensity_alignment
            else:
                assert f.electrode_intensity_alignment is None

            assert f.loaded_experiment == str(exp)
            assert f.experiment_cam_image is not None

            assert f.view_controller is not None
            assert f.data_serializer is not None
            assert f.function_factory is not None
            assert f.stage_factory is not None
            assert f.shape_factory is not None
            # it should change for each exp
            for name, value in zip(exp_props, exp_values):
                assert getattr(f, name) is not value
            exp_values = [getattr(f, name) for name in exp_props]

            assert f.experiment_stage_name
            assert not f.experiment_notes
            assert not f.external_function_plugin_package
            assert not f.external_stage_plugin_package


def test_saved_data(experiment_filename):
    from ceed.analysis import CeedDataReader
    existing_exp, merged_exp = exp_source(experiment_filename)
    shape1, shape2 = stored_shape_names

    with CeedDataReader(experiment_filename) as f:
        assert f.led_state is None

        for exp, image, (b1, b2) in zip((0, 1), stored_images, stored_b_values):
            f.load_experiment(exp)
            d1 = np.array(f.shapes_intensity[shape1])
            d2 = np.array(f.shapes_intensity[shape2])

            assert d1.shape == (240, 4)
            assert d2.shape == (240, 4)

            for i in range(240):
                for d, b, (active, inactive) in zip(
                        (d1, d2), (b1, b2), stored_colors):
                    assert d[i, 3] == 1
                    for j in inactive:
                        assert d[i, j] == 0

                    # 2.4 = .6 * 120 / 30
                    val = .6 * (i % 30) / 30 + b
                    for j in active:
                        matched = isclose(float(d[i, j]), val, abs_tol=.001)
                        # original ceed sometimes treated last sample as first
                        # because of float point issues
                        if existing_exp and not (i % 30):
                            assert matched or \
                                isclose(float(d[i, j]), b + .6, abs_tol=.001)
                        else:
                            assert matched

            assert_image_same(
                image, f.experiment_cam_image, exact=not existing_exp)

            assert f.led_state.tolist() == [(0, 1, 1, 1)]


def test_saved_image(experiment_filename):
    from ceed.analysis import CeedDataReader
    existing_exp, merged_exp = exp_source(experiment_filename)

    with CeedDataReader(experiment_filename) as f:
        for i in range(2):
            image, notes, _ = f.get_image_from_file(i)
            assert f'image {i}' == notes
            assert_image_same(
                image, stored_images[2 + i], exact=not existing_exp)

        for exp in (0, 1):
            f.load_experiment(exp)

        for i in range(2):
            image, notes, _ = f.get_image_from_file(i)
            assert f'image {i}' == notes
            assert_image_same(
                image, stored_images[2 + i], exact=not existing_exp)


def test_replay_experiment_data(experiment_filename):
    from ceed.analysis import CeedDataReader
    shape1, shape2 = stored_shape_names
    b1, b2 = stored_b_values[1]

    def verify_values():
        assert n == 240

        for i in range(240):
            for name, b, (active, inactive) in zip(
                    (shape1, shape2), (b1, b2), stored_colors):
                d = values[name]
                assert d[i][3] == 1
                for j in inactive:
                    assert d[i][j] == 0

                # 2.4 = .6 * 120 / 30
                val = .6 * (i % 30) / 30 + b
                for j in active:
                    assert isclose(float(d[i][j]), val, abs_tol=.001)

    with CeedDataReader(experiment_filename) as f:
        values, n = get_stage_time_intensity(
            f.app_config['stage_factory'], stored_stage_name, 120)
        verify_values()

        for exp, image, (b1, b2) in zip((0, 1), stored_images, stored_b_values):
            f.load_experiment(exp)

            values, n = get_stage_time_intensity(
                f.stage_factory, f.experiment_stage_name, 120)
            verify_values()


def test_mcs_data(merged_filename):
    from ceed.analysis import CeedDataReader
    shape1, shape2 = stored_shape_names

    with CeedDataReader(merged_filename) as f:
        f.load_mcs_data()

        assert f.electrodes_data
        assert f.electrodes_data.keys() == f.electrodes_metadata.keys()
        assert f.electrode_dig_data is not None
        n = len(f.electrodes_data[list(f.electrodes_data.keys())[0]])
        assert f.electrode_dig_data.shape == (n, )

        for exp in (0, 1):
            f.load_experiment(exp)
            assert f.electrode_intensity_alignment is not None
            n = f.shapes_intensity[shape1].shape[0]
            shape = f.electrode_intensity_alignment.shape
            assert shape in ((n, ), (n - 1, ))


def test_create_movie(tmp_path, existing_merged_experiment_filename):
    from ceed.analysis import CeedDataReader

    with CeedDataReader(existing_merged_experiment_filename) as f:
        f.load_mcs_data()
        f.load_experiment(0)

        paint_funcs = [
            f.paint_background_image(
                f.experiment_cam_image,
                transform_matrix=f.view_controller.cam_transform),
            f.show_mea_outline(f.view_controller.mea_transform),
            # this function shows the electrode voltage data
            f.paint_electrodes_data_callbacks(
                f.get_electrode_names(), draw_pos_hint=(1, 0),
                volt_axis=50)
        ]

        filename = tmp_path / 'movie.mp4'
        assert not filename.exists()
        f.generate_movie(
            str(filename),
            end=.1,
            lum=1,
            canvas_size_hint=(2, 1),
            # show the data at the normal speed
            speed=1.,
            paint_funcs=paint_funcs
        )

        assert filename.exists()


@pytest.mark.parametrize('stages_only', [True, False])
@pytest.mark.parametrize('suffix', ['app.yml', '.yml'])
async def test_import_yml_stages(
        stage_app: CeedTestApp, internal_experiment_filename, suffix,
        stages_only):
    filename = internal_experiment_filename[:-3] + suffix

    if not stages_only and suffix == '.yml':
        with pytest.raises(KeyError):
            stage_app.app.ceed_data.read_yaml_config(filename)
        return

    stage_app.app.ceed_data.read_yaml_config(filename, stages_only=stages_only)
    await stage_app.wait_clock_frames(2)

    values, n = get_stage_time_intensity(
        stage_app.app.stage_factory, stored_stage_name, 120)
    verify_experiment(values, n, False)


async def test_import_yml_existing(
        stage_app: CeedTestApp, existing_template_filename):
    stage_app.app.ceed_data.read_yaml_config(
        existing_template_filename, stages_only=True)
    await stage_app.wait_clock_frames(2)

    values, n = get_stage_time_intensity(
        stage_app.app.stage_factory, stored_stage_name, 120)
    verify_experiment(values, n, True)


@pytest.mark.parametrize('stages_only', [True, False])
@pytest.mark.parametrize('src', ['internal', 'existing'])
async def test_import_h5_stages(
        stage_app: CeedTestApp, internal_experiment_filename,
        existing_experiment_filename, src, stages_only):
    if src == 'internal':
        filename = internal_experiment_filename
    else:
        filename = existing_experiment_filename

    stage_app.app.ceed_data.import_file(filename, stages_only=stages_only)
    await stage_app.wait_clock_frames(2)

    values, n = get_stage_time_intensity(
        stage_app.app.stage_factory, stored_stage_name, 120)
    verify_experiment(values, n, False)
