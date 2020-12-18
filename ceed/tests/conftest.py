import os
import pytest
import trio
import time
import gc
import weakref
from collections import defaultdict
import logging
import warnings
from typing import Type, List

pytest.register_assert_rewrite('ceed.tests.test_app.examples')

from ceed.function import FunctionFactoryBase
from ceed.shape import CeedPaintCanvasBehavior
from ceed.stage import StageFactoryBase
from ceed.tests.ceed_app import CeedTestApp
from .common import add_prop_watch

warnings.filterwarnings(
    "ignore",
    message="numpy.ufunc size changed, may indicate binary incompatibility. "
            "Expected 192 from C header, got 216 from PyObject"
)

os.environ['KIVY_USE_DEFAULTCONFIG'] = '1'

file_count = defaultdict(int)


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


@pytest.fixture(scope='session')
def app_list():
    apps = []

    yield apps

    gc.collect()
    alive_apps = []
    for i, (app, request) in enumerate(apps[1:-1]):
        app = app()
        request = request()
        if request is None:
            request = '<dead request>'

        if app is not None:
            alive_apps.append((app, request))
            logging.error(
                'Memory leak: failed to release app for test ' + repr(request))

            import objgraph
            objgraph.show_backrefs(
                [app], filename=r'E:\backrefs{}.png'.format(i), max_depth=50,
                too_many=1)
            # objgraph.show_chain(
            #     objgraph.find_backref_chain(
            #         last_app(), objgraph.is_proper_module),
            #     filename=r'E:\chain.png')

    assert not len(alive_apps), 'Memory leak: failed to release all apps'


@pytest.fixture()
async def ceed_app(
        request, nursery, temp_file, tmp_path, tmp_path_factory, app_list):

    params = request.param if hasattr(
        request, 'param') and request.param else {}
    ts0 = time.perf_counter()
    from kivy.core.window import Window
    from kivy.context import Context
    from kivy.clock import ClockBase
    from kivy.animation import Animation
    from kivy.base import stopTouchApp
    from kivy.factory import FactoryBase, Factory
    from kivy.lang.builder import BuilderBase, Builder
    from kivy.logger import LoggerHistory

    context = Context(init=False)
    context['Clock'] = ClockBase(async_lib='trio')
    # context['Factory'] = FactoryBase.create_from(Factory)
    # have to make sure all ceed files are imported before this because
    # globally read kv files will not be loaded again in the new builder,
    # except if manually loaded, which we don't do
    # context['Builder'] = BuilderBase.create_from(Builder)
    context.push()

    Window.create_window()
    Window.register()
    Window.initialized = True
    Window.canvas.clear()

    from kivy.clock import Clock
    Clock._max_fps = 0

    import ceed.view.controller
    ceed.view.controller.ignore_vpixx_import_error = True

    if params.get('persist_config'):
        base = str(tmp_path_factory.getbasetemp() / params['persist_config'])
        app = CeedTestApp(
            yaml_config_path=base + 'config.yaml',
            ini_file=base + 'config.ini', open_player_thread=False)
    else:
        app = CeedTestApp(
            yaml_config_path=temp_file('config.yaml'),
            ini_file=temp_file('config.ini'), open_player_thread=False)
    app.ceed_data.root_path = str(tmp_path)

    try:
        app.set_async_lib('trio')
        nursery.start_soon(app.async_run)

        ts = time.perf_counter()
        while not app.app_has_started:
            await trio.sleep(.1)
            if time.perf_counter() - ts >= 120:
                raise TimeoutError()

        await app.wait_clock_frames(5)

        ts1 = time.perf_counter()
        yield weakref.proxy(app)
        ts2 = time.perf_counter()

        stopTouchApp()

        ts = time.perf_counter()
        while not app.app_has_stopped:
            await trio.sleep(.1)
            if time.perf_counter() - ts >= 40:
                raise TimeoutError()

    finally:
        stopTouchApp()
        for anim in list(Animation._instances):
            anim._unregister()
        app.clean_up()
        for child in Window.children[:]:
            Window.remove_widget(child)

        context.pop()
        del context
        LoggerHistory.clear_history()

    app_list.append((weakref.ref(app), weakref.ref(request)))

    ts3 = time.perf_counter()
    print(ts1 - ts0, ts2 - ts1, ts3 - ts2)


@pytest.fixture
async def func_app(ceed_app: CeedTestApp):
    from kivy.metrics import dp
    await ceed_app.wait_clock_frames(2)

    assert ceed_app.function_factory is not None
    assert not ceed_app.function_factory.funcs_user
    assert len(ceed_app.function_factory.funcs_cls) == \
        len(ceed_app.function_factory.funcs_inst)
    assert ceed_app.function_factory.funcs_inst == \
        ceed_app.function_factory.funcs_inst_default

    assert not ceed_app.funcs_container.children

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

    assert ceed_app.shape_factory is not None
    assert not ceed_app.shape_factory.shapes

    painter_widget = ceed_app.resolve_widget().down(
        test_name='painter')()
    assert tuple(painter_widget.size) == (
        ceed_app.view_controller.screen_width,
        ceed_app.view_controller.screen_height)

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

    assert paint_app.stage_factory is not None
    assert not paint_app.stage_factory.stages
    assert not paint_app.stage_factory.stage_names

    assert not paint_app.stages_container.children

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
    add_prop_watch(stage_factory, 'on_changed', 'test_changes_count')

    yield stage_factory
