import os
import pytest
import trio
import time
import gc
import weakref
from collections import defaultdict

os.environ['KIVY_EVENTLOOP'] = 'trio'
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


apps = []

@pytest.fixture()
async def ceed_app(request, nursery, temp_file, tmp_path):
    ts0 = time.perf_counter()
    from ceed.tests.ceed_app import CeedTestApp
    from kivy.core.window import Window
    from kivy.context import Context
    from kivy.clock import ClockBase
    from kivy.base import stopTouchApp

    context = Context(init=False)
    context['Clock'] = ClockBase(async_lib='trio')
    context.push()

    Window.create_window()
    Window.register()
    Window.initialized = True
    Window.canvas.clear()

    import ceed.view.controller
    ceed.view.controller.ignore_vpixx_import_error = True

    app = CeedTestApp(
        json_config_path=temp_file('config.yaml'),
        ini_file=temp_file('config.ini'))
    app.ceed_data.root_path = str(tmp_path)

    async def run_app():
        await app.async_run()

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

    for child in Window.children[:]:
        Window.remove_widget(child)
    context.pop()
    del context
    app = weakref.ref(app)
    apps.append(app)
    gc.collect()
    ts3 = time.perf_counter()

    print(ts1 - ts0, ts2 - ts1, ts3 - ts2, app())
    if len(apps) >= 3 and apps[-3]() is not None:
        import objgraph
        objgraph.show_backrefs([apps[-3]()], filename=r'E:\sample-graph.png', too_many=1, max_depth=50)
        # objgraph.show_chain(
        #     objgraph.find_backref_chain(apps[-3](), objgraph.is_proper_module), filename=r'E:\sample-graph.png')
        assert False
