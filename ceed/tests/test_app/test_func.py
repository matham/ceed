import pytest
import math
from typing import Type, List
from ceed.tests.ceed_app import CeedTestApp
from .funcs import func_classes


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


class TestFunctions(object):

    func_app = None  # type: CeedTestApp

    funcs = []

    def setup_app(self, func_app: CeedTestApp):
        self.func_app = func_app
        self.funcs = [func_cls(app=func_app) for func_cls in func_classes]

    async def test_funcs_add(self, func_app: CeedTestApp):
        self.setup_app(func_app)
        await func_app.wait_clock_frames(2)

        for func in self.funcs:
            ceed_func = func.func
            func.assert_init()

            assert func.name == ceed_func.name
            assert func.duration == ceed_func.duration
            assert func.loop == ceed_func.loop
            if hasattr(ceed_func, 't_offset'):
                assert func.t_offset == ceed_func.t_offset
            assert math.isclose(func.duration_min_total, ceed_func.duration_min_total)
            assert func.timebase[0] == ceed_func.timebase_numerator
            assert func.timebase[1] == ceed_func.timebase_denominator
            assert func.timebase[0] / func.timebase[1] == \
                float(ceed_func.timebase)

            func.assert_func_values()
