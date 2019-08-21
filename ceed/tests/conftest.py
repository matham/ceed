import os
import pytest
import trio
import time
import gc
import weakref
from collections import defaultdict
import logging
import warnings

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


apps = []


@pytest.fixture()
async def ceed_app(request, nursery, temp_file, tmp_path, tmp_path_factory):
    gc.collect()
    if len(apps) >= 2:
        last_app, last_request = apps.pop()

        if last_app() is not None:
            logging.error(
                'Memory leak: failed to release app for test ' +
                repr(last_request))
            # import objgraph
            # objgraph.show_backrefs(
            #     [last_app()], filename=r'E:\backrefs.png', max_depth=50,
            #     too_many=1)
            # objgraph.show_chain(
            #     objgraph.find_backref_chain(
            #         last_app(), objgraph.is_proper_module),
            #     filename=r'E:\chain.png')

        # assert last_app() is None, \
        #     'Memory leak: failed to release app for test ' + repr(last_request)

    params = request.param if hasattr(
        request, 'param') and request.param else {}
    ts0 = time.perf_counter()
    from ceed.tests.ceed_app import CeedTestApp
    from kivy.core.window import Window
    from kivy.context import Context
    from kivy.clock import ClockBase
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

    import ceed.view.controller
    ceed.view.controller.ignore_vpixx_import_error = True

    if params.get('persist_config'):
        base = str(tmp_path_factory.getbasetemp() / params['persist_config'])
        app = CeedTestApp(
            json_config_path=base + 'config.yaml',
            ini_file=base + 'config.ini')
    else:
        app = CeedTestApp(
            json_config_path=temp_file('config.yaml'),
            ini_file=temp_file('config.ini'))
    app.ceed_data.root_path = str(tmp_path)

    async def run_app():
        await app.async_run(async_lib='trio')

    nursery.start_soon(run_app)

    ts = time.perf_counter()
    while not app.app_has_started:
        await trio.sleep(.1)
        if time.perf_counter() - ts >= 10:
            raise TimeoutError()

    await app.wait_clock_frames(5)

    ts1 = time.perf_counter()
    yield app
    ts2 = time.perf_counter()

    stopTouchApp()

    ts = time.perf_counter()
    while not app.app_has_stopped:
        await trio.sleep(.1)
        if time.perf_counter() - ts >= 10:
            raise TimeoutError()

    app.clean_up()
    for child in Window.children[:]:
        Window.remove_widget(child)

    context.pop()
    del context
    LoggerHistory.clear_history()
    apps.append((weakref.ref(app), request))
    del app

    ts3 = time.perf_counter()
    print(ts1 - ts0, ts2 - ts1, ts3 - ts2)
