import pytest
from math import isclose, ceil
import numpy as np
import pathlib
from pytest_dependency import depends

import ceed
from .examples.shapes import CircleShapeP1, CircleShapeP2
from .examples import assert_image_same, create_test_image
from .examples.experiment import create_basic_experiment, run_experiment, \
    set_serializer_even_count_bits, wait_experiment_done, measure_fps
from ceed.tests.ceed_app import CeedTestApp
from ceed.tests.test_stages import get_stage_time_intensity
from ceed.analysis.merge_data import CeedMCSDataMerger
from ceed.tests.test_app.examples.stages import ParaAllStage

stored_images = []
stored_b_values = [(.0, .2), (.1, .3)]
stored_shape_names = CircleShapeP1.name, CircleShapeP2.name
# r, g is active, b is inactive
stored_colors = [((0, 1, ), (2,)), ] * 2
stored_stage_name = ParaAllStage.name

data_root = pathlib.Path(ceed.__file__).parent.joinpath('examples', 'data')
existing_experiment_filename_v1_0_0_dev0 = str(
    data_root.joinpath('ceed_data_v1.0.0.dev0.h5'))
existing_template_filename_v1_0_0_dev0 = str(
    data_root.joinpath('ceed_template_v1.0.0.dev0.yml'))
existing_merged_experiment_filename_v1_0_0_dev0 = str(
    data_root.joinpath('ceed_mcs_data_merged_v1.0.0.dev0.h5'))
mcs_filename_v1_0_0_dev0 = str(data_root.joinpath('mcs_data_v1.0.0.dev0.h5'))
existing_experiment_filename_v1_0_0_dev1 = str(
    data_root.joinpath('ceed_data_v1.0.0.dev1.h5'))
existing_template_filename_v1_0_0_dev1 = str(
    data_root.joinpath('ceed_template_v1.0.0.dev1.yml'))
existing_merged_experiment_filename_v1_0_0_dev1 = str(
    data_root.joinpath('ceed_mcs_data_merged_v1.0.0.dev1.h5'))
mcs_filename_v1_0_0_dev1 = str(data_root.joinpath('mcs_data_v1.0.0.dev1.h5'))

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
        await wait_experiment_done(stage_app, timeout=10 * 60)

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
def re_merge_experiment_filename_v1_0_0_0_dev0(tmp_path_factory):
    """All tests depending on this, also depend on
    test_create_merge_experiment."""
    filename = str(
        tmp_path_factory.mktemp(
            'experiment') / 'new_experiment_merged_v1_0_0_dev0.h5')
    return filename, existing_experiment_filename_v1_0_0_dev0, \
        mcs_filename_v1_0_0_dev0


@pytest.fixture(scope='module')
def re_merge_experiment_filename_v1_0_0_0_dev1(tmp_path_factory):
    """All tests depending on this, also depend on
    test_create_merge_experiment."""
    filename = str(
        tmp_path_factory.mktemp(
            'experiment') / 'new_experiment_merged_v1_0_0_dev1.h5')
    return filename, existing_experiment_filename_v1_0_0_dev1, \
        mcs_filename_v1_0_0_dev1


@pytest.fixture(params=[
    'internal', 'external', 're_merged_v1_0_0_dev0',
    'existing_merged_v1_0_0_dev0', 'existing_unmerged_v1_0_0_dev0',
    're_merged_v1_0_0_dev1', 'existing_merged_v1_0_0_dev1',
    'existing_unmerged_v1_0_0_dev1'])
def experiment_ceed_filename(
        request, internal_experiment_filename, external_experiment_filename,
        re_merge_experiment_filename_v1_0_0_0_dev0,
        re_merge_experiment_filename_v1_0_0_0_dev1):
    src = request.param
    if src == 'internal':
        depends(request, ['internal_experiment'])
        return internal_experiment_filename
    if src == 'external':
        depends(request, ['external_experiment'])
        return external_experiment_filename
    if src == 're_merged_v1_0_0_dev0':
        depends(request, ['merge_experiment'])
        return re_merge_experiment_filename_v1_0_0_0_dev0[0]
    if src == 're_merged_v1_0_0_dev1':
        depends(request, ['merge_experiment'])
        return re_merge_experiment_filename_v1_0_0_0_dev1[0]
    if src == 'existing_merged_v1_0_0_dev0':
        if not pathlib.Path(
                existing_merged_experiment_filename_v1_0_0_dev0).exists():
            pytest.skip(
                f'"{existing_merged_experiment_filename_v1_0_0_dev0}" '
                f'does not exist')
        return existing_merged_experiment_filename_v1_0_0_dev0
    if src == 'existing_merged_v1_0_0_dev1':
        if not pathlib.Path(
                existing_merged_experiment_filename_v1_0_0_dev1).exists():
            pytest.skip(
                f'"{existing_merged_experiment_filename_v1_0_0_dev1}" '
                f'does not exist')
        return existing_merged_experiment_filename_v1_0_0_dev1

    if src == 'existing_unmerged_v1_0_0_dev0':
        if not pathlib.Path(existing_experiment_filename_v1_0_0_dev0).exists():
            pytest.skip(
                f'"{existing_experiment_filename_v1_0_0_dev0}" does not exist')
        return existing_experiment_filename_v1_0_0_dev0

    if not pathlib.Path(existing_experiment_filename_v1_0_0_dev1).exists():
        pytest.skip(
            f'"{existing_experiment_filename_v1_0_0_dev1}" does not exist')
    return existing_experiment_filename_v1_0_0_dev1


@pytest.fixture(params=[
    're_merged_v1_0_0_dev0', 'existing_merged_v1_0_0_dev0',
    're_merged_v1_0_0_dev1', 'existing_merged_v1_0_0_dev1'])
def merged_filename(
        request, re_merge_experiment_filename_v1_0_0_0_dev0,
        re_merge_experiment_filename_v1_0_0_0_dev1):
    src = request.param
    if src == 're_merged_v1_0_0_dev0':
        depends(request, ['merge_experiment'])
        return re_merge_experiment_filename_v1_0_0_0_dev0[0]
    if src == 're_merged_v1_0_0_dev1':
        depends(request, ['merge_experiment'])
        return re_merge_experiment_filename_v1_0_0_0_dev1[0]
    if src == 'existing_merged_v1_0_0_dev0':
        if not pathlib.Path(
                existing_merged_experiment_filename_v1_0_0_dev0).exists():
            pytest.skip(
                f'"{existing_merged_experiment_filename_v1_0_0_dev0}" '
                f'does not exist')
        return existing_merged_experiment_filename_v1_0_0_dev0

    if not pathlib.Path(
            existing_merged_experiment_filename_v1_0_0_dev1).exists():
        pytest.skip(
            f'"{existing_merged_experiment_filename_v1_0_0_dev1}" '
            f'does not exist')
    return existing_merged_experiment_filename_v1_0_0_dev1


@pytest.mark.dependency(name='internal_experiment')
async def test_create_internal_experiment(
        stage_app: CeedTestApp, internal_experiment_filename):
    filename = internal_experiment_filename

    await run_data_experiment(stage_app)

    stage_app.app.ceed_data.save(filename=filename)
    base = filename[:-3]
    stage_app.app.ceed_data.write_yaml_config(base + '.yml', stages_only=True)
    stage_app.app.ceed_data.write_yaml_config(
        base + 'app.yml', stages_only=False)


@pytest.mark.dependency(name='external_experiment')
async def test_create_external_experiment(
        stage_app: CeedTestApp, external_experiment_filename):
    filename = external_experiment_filename
    stage_app.app.view_controller.start_process()
    await stage_app.wait_clock_frames(2)

    await run_data_experiment(stage_app)

    stage_app.app.view_controller.stop_process()
    await stage_app.wait_clock_frames(2)

    stage_app.app.ceed_data.save(filename=filename)


@pytest.mark.dependency(name='merge_experiment')
@pytest.mark.parametrize(
    'triplet', ['re_merged_v1_0_0_dev0', 're_merged_v1_0_0_dev1'])
def test_create_merge_experiment(
        triplet, re_merge_experiment_filename_v1_0_0_0_dev0,
        re_merge_experiment_filename_v1_0_0_0_dev1):
    if triplet == 're_merged_v1_0_0_dev0':
        filename, ceed_filename, mcs_filename = \
            re_merge_experiment_filename_v1_0_0_0_dev0
    else:
        filename, ceed_filename, mcs_filename = \
            re_merge_experiment_filename_v1_0_0_0_dev1

    if not pathlib.Path(ceed_filename).exists():
        pytest.skip(f'"{ceed_filename}" does not exist')
    if not pathlib.Path(mcs_filename).exists():
        pytest.skip(f'"{mcs_filename}" does not exist')

    merger = CeedMCSDataMerger(
        ceed_filename=ceed_filename, mcs_filename=mcs_filename)

    experiments = merger.get_experiment_numbers()
    assert experiments == ['0', '1'] or experiments == ['0', '1', '2', '3']

    merger.read_mcs_data()
    merger.read_ceed_data()
    merger.parse_mcs_data()

    alignment = {}
    for experiment in experiments:
        merger.read_ceed_experiment_data(experiment)
        merger.parse_ceed_experiment_data()

        alignment[experiment] = merger.get_alignment()

    merger.merge_data(filename, alignment, notes='app notes')


@pytest.mark.dependency()
def test_saved_metadata(experiment_ceed_filename):
    from ceed.analysis import CeedDataReader
    existing_exp, merged_exp = exp_source(experiment_ceed_filename)

    with CeedDataReader(experiment_ceed_filename) as f:
        experiments = f.experiments_in_file

        def verify_app_props():
            assert f.filename == experiment_ceed_filename
            assert experiments == ['0', '1'] or \
                experiments == ['0', '1', '2', '3']
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

        for exp in experiments:
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


@pytest.mark.dependency()
def test_saved_data(experiment_ceed_filename):
    from ceed.analysis import CeedDataReader
    existing_exp, merged_exp = exp_source(experiment_ceed_filename)
    shape1, shape2 = stored_shape_names

    with CeedDataReader(experiment_ceed_filename) as f:
        assert f.led_state is None

        for exp, image, (b1, b2) in zip((0, 1), stored_images, stored_b_values):
            f.load_experiment(exp)
            d1 = f.shapes_intensity[shape1]
            d2 = f.shapes_intensity[shape2]

            assert d1.shape == (240, 4)
            assert d2.shape == (240, 4)
            assert len(
                np.asarray(f._block.data_arrays['frame_time_counter'])) == 240

            assert len(np.asarray(f._block.data_arrays['frame_bits'])) == 240
            counter = np.asarray(f._block.data_arrays['frame_counter'])
            assert len(counter) == 240
            assert np.all(counter == np.arange(1, 241))
            render = np.asarray(f._block.data_arrays['frame_time_counter'])
            assert len(render) == 240
            assert np.all(render == np.arange(1, 241))
            assert len(np.asarray(f._block.data_arrays['frame_time'])) == 240

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


@pytest.mark.dependency()
def test_saved_image(experiment_ceed_filename):
    from ceed.analysis import CeedDataReader
    existing_exp, merged_exp = exp_source(experiment_ceed_filename)

    with CeedDataReader(experiment_ceed_filename) as f:
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


@pytest.mark.dependency()
def test_replay_experiment_data(experiment_ceed_filename):
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

    with CeedDataReader(experiment_ceed_filename) as f:
        values, n = get_stage_time_intensity(
            f.app_config['stage_factory'], stored_stage_name, 120)
        verify_values()

        for exp, image, (b1, b2) in zip((0, 1), stored_images, stored_b_values):
            f.load_experiment(exp)

            values, n = get_stage_time_intensity(
                f.stage_factory, f.experiment_stage_name, 120)
            verify_values()


@pytest.mark.dependency()
def test_mcs_data(merged_filename):
    from ceed.analysis import CeedDataReader
    shape1, shape2 = stored_shape_names

    with CeedDataReader(merged_filename) as f:
        dev0 = 'dev0' in f.ceed_version
        f.load_mcs_data()

        assert f.electrodes_data
        assert f.electrodes_data.keys() == f.electrodes_metadata.keys()
        assert f.electrode_dig_data is not None
        n = len(f.electrodes_data[list(f.electrodes_data.keys())[0]])
        assert f.electrode_dig_data.shape == (n, )

        last_end_sample = 0
        for exp in f.experiments_in_file:
            n_sub_frames = 1
            if exp == '2':
                n_sub_frames = 4
            elif exp == '3':
                n_sub_frames = 12

            f.load_experiment(exp)
            assert f.electrode_intensity_alignment is not None
            n = f.shapes_intensity[shape1].shape[0]
            assert n == 240 * n_sub_frames
            assert f.electrode_intensity_alignment[0] > last_end_sample
            last_end_sample = f.electrode_intensity_alignment[-1]

            n_align = f.electrode_intensity_alignment.shape[0]
            samples_per_frames = f.electrode_intensity_alignment[1:] - \
                f.electrode_intensity_alignment[:-1]
            n_samples_min = np.min(samples_per_frames)
            n_samples_max = np.max(samples_per_frames)

            if dev0:
                assert n_align == 240 or n_align == 239
                # we used 1khz, and no quad mode to generate data
                bot = 1000 // 120
                # in case of missed frame
                top = ceil(2 * 1000 / 120)
            else:
                assert n_align == 240 * n_sub_frames
                # sampled at 5khz
                bot = 5000 // (120 * n_sub_frames)
                # in case of missed frame
                top = ceil(2 * 5000 / (120 * n_sub_frames))

            assert bot <= n_samples_min <= n_samples_max <= top


def test_create_movie(tmp_path):
    from ceed.analysis import CeedDataReader
    if not pathlib.Path(
            existing_merged_experiment_filename_v1_0_0_dev1).exists():
        pytest.skip(f'"{existing_merged_experiment_filename_v1_0_0_dev1}" '
                    f'does not exist')

    with CeedDataReader(existing_merged_experiment_filename_v1_0_0_dev1) as f:
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
@pytest.mark.dependency(depends=['internal_experiment'])
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


@pytest.mark.parametrize('existing_template_filename', [
    existing_template_filename_v1_0_0_dev0,
    existing_template_filename_v1_0_0_dev1])
async def test_import_yml_existing(
        stage_app: CeedTestApp, existing_template_filename):
    if not pathlib.Path(existing_template_filename).exists():
        pytest.skip(f'"{existing_template_filename}" does not exist')

    stage_app.app.ceed_data.read_yaml_config(
        existing_template_filename, stages_only=True)
    await stage_app.wait_clock_frames(2)

    values, n = get_stage_time_intensity(
        stage_app.app.stage_factory, stored_stage_name, 120)
    verify_experiment(values, n, True)


@pytest.mark.parametrize('stages_only', [True, False])
@pytest.mark.parametrize('src', [
    'internal', 'existing_v1_0_0_dev0', 'existing_v1_0_0_dev1'])
async def test_import_h5_stages(
        request, stage_app: CeedTestApp, internal_experiment_filename,
        src, stages_only):
    if src == 'internal':
        depends(request, ['internal_experiment'])
        filename = internal_experiment_filename
    elif src == 'existing_v1_0_0_dev0':
        if not pathlib.Path(existing_experiment_filename_v1_0_0_dev0).exists():
            pytest.skip(
                f'"{existing_experiment_filename_v1_0_0_dev0}" does not exist')
        filename = existing_experiment_filename_v1_0_0_dev0
    else:
        if not pathlib.Path(existing_experiment_filename_v1_0_0_dev1).exists():
            pytest.skip(
                f'"{existing_experiment_filename_v1_0_0_dev1}" does not exist')
        filename = existing_experiment_filename_v1_0_0_dev1

    stage_app.app.ceed_data.import_file(filename, stages_only=stages_only)
    await stage_app.wait_clock_frames(2)

    values, n = get_stage_time_intensity(
        stage_app.app.stage_factory, stored_stage_name, 120)
    verify_experiment(values, n, False)


@pytest.mark.ceed_single_pixel
@pytest.mark.parametrize('video_mode', ['RGB', 'QUAD4X', 'QUAD12X'])
@pytest.mark.parametrize(
    'flip,skip', [(True, False), (False, True), (False, False)])
async def test_serializer_corner_pixel(
        stage_app: CeedTestApp, flip, skip, video_mode):
    # for can't use stage_app because that zooms out leading to pixel being too
    # small to see, seemingly
    from kivy.clock import Clock
    from ceed.function.plugin import ConstFunc
    from ..test_stages import create_2_shape_stage

    n_sub_frames = 1
    if video_mode == 'QUAD4X':
        n_sub_frames = 4
    elif video_mode == 'QUAD12X':
        n_sub_frames = 12
    config, num_handshake_ticks, counter, short_values, clock_values = \
        set_serializer_even_count_bits(
            stage_app.app.data_serializer, n_sub_frames)
    stage_app.app.data_serializer.projector_to_aquisition_map = {
        i: i for i in range(16)}

    root, s1, s2, shape1, shape2 = create_2_shape_stage(
        stage_app.app.stage_factory, show_in_gui=True, app=stage_app)
    s1.stage.add_func(ConstFunc(
        function_factory=stage_app.app.function_factory, duration=20))
    await stage_app.wait_clock_frames(2)

    fps = await measure_fps(stage_app) + 10
    stage_app.app.view_controller.frame_rate = fps
    stage_app.app.view_controller.use_software_frame_rate = False
    stage_app.app.view_controller.skip_estimated_missed_frames = skip
    stage_app.app.view_controller.pad_to_stage_handshake = True
    stage_app.app.view_controller.flip_projector = flip
    stage_app.app.view_controller.output_count = True
    stage_app.app.view_controller.video_mode = video_mode
    assert stage_app.app.view_controller.do_quad_mode == (video_mode != 'RGB')
    assert stage_app.app.view_controller.effective_frame_rate == \
        fps * n_sub_frames

    frame = 0
    expected_values = list(zip(counter, short_values, clock_values))
    clock_or_short = 1 << stage_app.app.data_serializer.clock_idx
    for i in stage_app.app.data_serializer.short_count_indices:
        clock_or_short |= 1 << i

    def verify_serializer(*largs):
        nonlocal frame
        # wait to start
        if not stage_app.app.view_controller.count:
            return

        # stop when we exhausted predicted frames
        if frame >= len(counter):
            stage_app.app.view_controller.request_stage_end()
            return

        (r, g, b, a), = stage_app.get_widget_pos_pixel(
            stage_app.app.shape_factory, [(0, 1079)])
        value = r | g << 8 | b << 16

        count, short, clock = expected_values[frame]
        print(frame, f'{value:010b}, {count:010b}, {short:010b}, {clock:08b}')

        if skip:
            # only count may be different if frames are skipped. Short and clock
            # are the same even if frames are dropped because corner pixel
            # values are not skipped
            assert value & clock_or_short == short | clock
        else:
            assert value == count | short | clock
            assert not stage_app.app.view_controller._n_missed_frames
        frame += 1

    stage_app.app.view_controller.request_stage_start(
        root.name, experiment_uuid=config)
    event = Clock.create_trigger(verify_serializer, timeout=0, interval=True)
    event()

    await wait_experiment_done(stage_app)

    assert frame == len(counter)


@pytest.mark.parametrize('video_mode', ['RGB', 'QUAD4X', 'QUAD12X'])
@pytest.mark.parametrize('skip', [True, False])
async def test_serializer_saved_data(
        stage_app: CeedTestApp, tmp_path, video_mode, skip):
    from kivy.clock import Clock
    from ceed.function.plugin import ConstFunc
    from ..test_stages import create_2_shape_stage

    n_sub_frames = 1
    if video_mode == 'QUAD4X':
        n_sub_frames = 4
    elif video_mode == 'QUAD12X':
        n_sub_frames = 12

    config, num_handshake_ticks, counter, short_values, clock_values = \
        set_serializer_even_count_bits(
            stage_app.app.data_serializer, n_sub_frames)
    stage_app.app.data_serializer.projector_to_aquisition_map = {
        i: i for i in range(16)}
    expected_values = list(zip(counter, short_values, clock_values))

    root, s1, s2, shape1, shape2 = create_2_shape_stage(
        stage_app.app.stage_factory, show_in_gui=True, app=stage_app)
    s1.stage.add_func(ConstFunc(
        function_factory=stage_app.app.function_factory, duration=4))
    await stage_app.wait_clock_frames(2)

    fps = await measure_fps(stage_app) + 10
    stage_app.app.view_controller.frame_rate = fps
    stage_app.app.view_controller.skip_estimated_missed_frames = skip
    stage_app.app.view_controller.use_software_frame_rate = False
    stage_app.app.view_controller.pad_to_stage_handshake = True
    stage_app.app.view_controller.output_count = True
    stage_app.app.view_controller.video_mode = video_mode

    flip_counter = []
    skip_counter = []

    def verify_serializer(*largs):
        count_val = stage_app.app.view_controller.count
        if not count_val or not stage_app.app.view_controller.stage_active:
            return

        for i in range(n_sub_frames - 1, -1, -1):
            flip_counter.append(count_val - i)
        skip_counter.append(stage_app.app.view_controller._n_missed_frames)

    event = Clock.create_trigger(verify_serializer, timeout=0, interval=True)
    event()
    stage_app.app.view_controller.request_stage_start(
        root.name, experiment_uuid=config)

    await wait_experiment_done(stage_app)
    event.cancel()

    filename = str(tmp_path / 'serializer_data.h5')
    stage_app.app.ceed_data.save(filename=filename)

    merger = CeedMCSDataMerger(ceed_filename=filename, mcs_filename='')

    merger.read_ceed_data()
    merger.read_ceed_experiment_data('0')
    merger.parse_ceed_experiment_data()

    # logged data is one per frame (where e.g. in 12x each is still a frame)
    # when skipping, this data doesn't include skipped frames so we don't have
    # to filter them out here
    raw_data = merger.ceed_data_container.data
    clock_data = merger.ceed_data_container.clock_data
    short_count_data = merger.ceed_data_container.short_count_data
    short_count_max = 2 ** len(
        stage_app.app.data_serializer.short_count_indices)
    clock_or_short = 1 << stage_app.app.data_serializer.clock_idx
    for i in stage_app.app.data_serializer.short_count_indices:
        clock_or_short |= 1 << i

    # clock and short count are one-per group of n_sub_frames
    for i, (raw_s, short_s, clock_s) in enumerate(
            zip(raw_data, short_count_data, clock_data)):
        root_frame_i = i // n_sub_frames

        if root_frame_i < len(expected_values):
            count, short, clock = expected_values[root_frame_i]
            if skip:
                # count may be different if frames are skipped
                assert raw_s & clock_or_short == short | clock
            else:
                assert raw_s == count | short | clock

        if root_frame_i % 2:
            assert not clock_s
        else:
            assert clock_s == 1

        assert short_s == root_frame_i % short_count_max

    # counter is one per frame, including sub frames
    n_skipped = sum(skip_counter) * n_sub_frames
    frame_counter = merger.ceed_data['frame_counter']
    if skip:
        assert len(frame_counter) > len(merger.ceed_data_container.counter)
        # last frame could have been indicted to be skipped, but stage ended
        assert len(flip_counter) + n_skipped \
            >= len(merger.ceed_data_container.counter)
    else:
        assert len(frame_counter) == len(merger.ceed_data_container.counter)
        assert np.all(merger.ceed_data_container.counter == np.arange(
            1, 1 + len(raw_data)))
        assert not n_skipped
    assert np.all(
        merger.ceed_data_container.counter == np.asarray(flip_counter))

    # even when skipping frames, we should have sent enough frames not not cut
    # off handshake (ideally)
    n_bytes_per_int = stage_app.app.data_serializer.counter_bit_width // 8
    config += b'\0' * (n_bytes_per_int - len(config) % n_bytes_per_int)
    if skip:
        # can't assume message was sent full in case of dropped frames
        if merger.ceed_data_container.handshake_data:
            assert merger.ceed_data_container.expected_handshake_len \
                == len(config)
        assert 0 <= merger.ceed_data_container.expected_handshake_len <= 50
        assert config.startswith(merger.ceed_data_container.handshake_data)
    else:
        assert merger.ceed_data_container.expected_handshake_len == len(config)
        assert merger.ceed_data_container.handshake_data == config
