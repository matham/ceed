import os
import pytest
from collections import defaultdict
from typing import Type, List
from pytest_trio.enable_trio_mode import \
    pytest_collection_modifyitems as trio_pytest_collection_modifyitems, \
    pytest_fixture_setup as trio_pytest_fixture_setup

pytest.register_assert_rewrite('ceed.tests.test_app.examples')

from kivy.config import Config
Config.set('modules', 'touchring', '')

from ceed.function import FunctionFactoryBase
from ceed.shape import CeedPaintCanvasBehavior
from ceed.stage import StageFactoryBase, register_all_stages
from ceed.tests.ceed_app import CeedTestGUIApp, CeedTestApp
from .common import add_prop_watch

file_count = defaultdict(int)


def pytest_addoption(parser):
    parser.addoption(
        "--ceed-skip-app",
        action="store_true",
        default=False,
        help='Whether to skip tests that test the GUI app.',
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ceed_app: mark test requires gui app")


def pytest_collection_modifyitems(config, items):
    trio_pytest_collection_modifyitems(items)

    if not config.getoption("--ceed-skip-app"):
        # --ceed-skip-app not given in cli: don't skip gui app tests
        return

    skip_ceed_app = pytest.mark.skip(reason="provided --ceed-skip-app")
    for item in items:
        if "ceed_app" in item.keywords:
            item.add_marker(skip_ceed_app)


def pytest_fixture_setup(fixturedef, request):
    # unfortunately we can't parameterize fixtures from fixtures, so we have to
    # use a hammer
    if fixturedef.argname == 'trio_kivy_app':
        request.param = {
            'kwargs': {'width': 1600, 'height': 900}, 'cls': CeedTestApp}

    trio_pytest_fixture_setup(fixturedef, request)


@pytest.fixture()
def temp_file(tmp_path):
    def temp_file_gen(fname):
        i = file_count[fname]
        file_count[fname] += 1

        root, ext = os.path.splitext(fname)
        return str(tmp_path / '{}_{}{}'.format(root, i, ext))

    return temp_file_gen


@pytest.fixture()
def temp_file_sess(tmp_path_factory):
    def temp_file_gen(fname):
        i = file_count[fname]
        file_count[fname] += 1

        root, ext = os.path.splitext(fname)
        return str(tmp_path_factory / '{}_{}{}'.format(root, i, ext))

    return temp_file_gen


@pytest.fixture
async def ceed_app(
        request, trio_kivy_app, temp_file, tmp_path, tmp_path_factory
) -> CeedTestApp:
    kivy_app = trio_kivy_app
    params = request.param if hasattr(
        request, 'param') and request.param else {}

    # context around app
    app_context = params.get('app_context')

    def create_app():
        import ceed.view.controller
        from more_kivy_app.config import dump_config
        ceed.view.controller.ignore_vpixx_import_error = True

        if params.get('persist_config'):
            base = str(
                tmp_path_factory.getbasetemp() / params['persist_config'])
            yaml_config_path = base + 'config.yaml'
            ini_file = base + 'config.ini'
        else:
            yaml_config_path = temp_file('config.yaml')
            ini_file = temp_file('config.ini')

        # creates config file from dict
        yaml_config = params.get('yaml_config')
        if yaml_config:
            dump_config(yaml_config_path, yaml_config)

        app = CeedTestGUIApp(
            yaml_config_path=yaml_config_path,
            ini_file=ini_file, open_player_thread=False)
        app.ceed_data.root_path = str(tmp_path)
        return app

    if app_context:
        with app_context(tmp_path):
            try:
                await kivy_app(create_app)
                yield kivy_app
            finally:
                if kivy_app.app is not None:
                    kivy_app.app.clean_up()
    else:
        try:
            await kivy_app(create_app)
            yield kivy_app
        finally:
            if kivy_app.app is not None:
                kivy_app.app.clean_up()


@pytest.fixture
async def func_app(ceed_app: CeedTestApp):
    from kivy.metrics import dp
    await ceed_app.wait_clock_frames(2)

    assert ceed_app.app.function_factory is not None
    assert not ceed_app.app.function_factory.funcs_user
    assert len(ceed_app.app.function_factory.funcs_cls) == \
        len(ceed_app.app.function_factory.funcs_inst)
    assert ceed_app.app.function_factory.funcs_inst == \
        ceed_app.app.function_factory.funcs_inst_default

    assert not ceed_app.app.funcs_container.children

    # expand shape splitter so shape widgets are fully visible
    splitter = ceed_app.resolve_widget().down(
        test_name='func splitter')().children[-1]
    async for _ in ceed_app.do_touch_drag(widget=splitter, dx=-dp(100)):
        pass
    await ceed_app.wait_clock_frames(2)

    yield ceed_app


async def zoom_screen_out(ceed_app: CeedTestApp):
    # expand shape splitter so shape widgets are fully visible
    slider = ceed_app.resolve_widget().down(test_name='screen zoom slider')()
    slider.value = slider.min
    await ceed_app.wait_clock_frames(2)


@pytest.fixture
async def paint_app(ceed_app: CeedTestApp):
    from kivy.metrics import dp
    await ceed_app.wait_clock_frames(2)

    assert ceed_app.app.shape_factory is not None
    assert not ceed_app.app.shape_factory.shapes

    painter_widget = ceed_app.resolve_widget().down(
        test_name='painter')()
    assert tuple(painter_widget.size) == (
        ceed_app.app.view_controller.screen_width,
        ceed_app.app.view_controller.screen_height)

    # expand shape splitter so shape widgets are fully visible
    splitter = ceed_app.resolve_widget().down(
        test_name='shape splitter')().children[-1]
    async for _ in ceed_app.do_touch_drag(widget=splitter, dx=-dp(100)):
        pass
    await ceed_app.wait_clock_frames(2)

    # expand group splitter to have more space
    splitter = ceed_app.resolve_widget().down(
        test_name='expand group splitter')().children[0]
    async for _ in ceed_app.do_touch_drag(widget=splitter, dy=-dp(200)):
        pass
    await ceed_app.wait_clock_frames(2)

    await zoom_screen_out(ceed_app)

    yield ceed_app


@pytest.fixture
async def stage_app(paint_app: CeedTestApp):
    from kivy.metrics import dp
    await paint_app.wait_clock_frames(2)

    assert paint_app.app.stage_factory is not None

    assert not paint_app.app.stages_container.children

    # expand stage splitter so stage widgets are fully visible
    splitter = paint_app.resolve_widget().down(
        test_name='stage splitter')().children[-1]
    async for _ in paint_app.do_touch_drag(widget=splitter, dx=-dp(100)):
        pass
    await paint_app.wait_clock_frames(2)

    await zoom_screen_out(paint_app)

    yield paint_app


@pytest.fixture
def function_factory() -> FunctionFactoryBase:
    from ceed.function import FunctionFactoryBase, register_all_functions
    function_factory = FunctionFactoryBase()
    register_all_functions(function_factory)
    add_prop_watch(function_factory, 'on_changed', 'test_changes_count')

    yield function_factory


@pytest.fixture
def shape_factory() -> CeedPaintCanvasBehavior:
    shape_factory = CeedPaintCanvasBehavior()
    add_prop_watch(
        shape_factory, 'on_remove_shape', 'test_changes_remove_shape_count')
    add_prop_watch(
        shape_factory, 'on_remove_group', 'test_changes_remove_group_count')
    add_prop_watch(shape_factory, 'on_changed', 'test_changes_count')

    yield shape_factory


@pytest.fixture
def stage_factory(
        function_factory: FunctionFactoryBase,
        shape_factory: CeedPaintCanvasBehavior) -> StageFactoryBase:
    stage_factory = StageFactoryBase(
        function_factory=function_factory, shape_factory=shape_factory)
    register_all_stages(stage_factory)
    add_prop_watch(stage_factory, 'on_changed', 'test_changes_count')

    yield stage_factory
