import pytest
import math
from typing import Type, List
from ceed.tests.ceed_app import CeedTestApp
from fractions import Fraction
from math import exp, cos, pi
import itertools

from ceed.function import FuncBase, FuncGroup, FunctionFactoryBase


def create_funcs(
        func_app: CeedTestApp = None,
        function_factory: FunctionFactoryBase = None, show_in_gui=True):
    funcs = []
    for func_cls in (ConstFunctionF1, LinearFunctionF1, ExponentialFunctionF1,
                     CosFunctionF1, GroupFunctionF1):
        func = func_cls(app=func_app, function_factory=function_factory,
                        show_in_gui=show_in_gui)
        funcs.append(func)
    return funcs


class Function:

    app: CeedTestApp = None

    function_factory: FunctionFactoryBase = None

    funcs_container = None

    func: FuncBase = None

    name = ''

    cls_name = ''

    duration_min_total = 0

    loop = 1

    timebase = (0, 1)

    t_offset = 0

    duration = 0

    t_start = 0

    func_times = []

    func_values = []

    fail_func_times = []

    editor_props = {}

    def __init__(
            self, app: CeedTestApp = None,
            function_factory: FunctionFactoryBase = None, show_in_gui=True):
        self.timebase_numerator = self.timebase[0]
        self.timebase_denominator = self.timebase[1]
        super().__init__()
        self.app = app
        if app is None:
            self.function_factory = function_factory
        else:
            self.function_factory = app.app.function_factory
            self.funcs_container = app.app.funcs_container

        if show_in_gui:
            self.show_in_gui()

    def create_func(self):
        raise NotImplementedError

    def show_in_gui(self, add_func=None, display_func=None):
        self.create_func()
        (add_func or self.function_factory.add_func)(self.func)
        (display_func or self.funcs_container.show_func_in_listing)(self.func)

    def assert_init(self):
        raise NotImplementedError

    def assert_func_values(self):
        from ceed.function import FuncDoneException
        func = self.func
        func.init_func_tree()
        func.init_func(self.t_start)

        times = list(itertools.chain(*self.func_times))
        assert len(times) == len(self.func_values)

        i = 0
        values = self.func_values
        for loop, (pre_times, times) in enumerate(zip(
                self.fail_func_times, self.func_times)):
            # first call the function with times from the current loop
            for t in times:
                val = values[i]
                assert math.isclose(func(t), val), \
                    't={}, func(t)={}, expected={}'.format(t, func(t), val)
                i += 1

            # now we're sure that the current loop has been incremented as we
            # caled it with times from the this loop we can test the times
            # from before the current loop
            assert func.loop_count == loop
            for t in pre_times:
                with pytest.raises((FuncDoneException, ValueError)):
                    func(t)
                    pytest.fail('t={}'.format(t))

        for t in self.fail_func_times[-1]:
            with pytest.raises((FuncDoneException, ValueError)):
                func(t)
                pytest.fail('t={}'.format(t))


class FunctionF1(object):

    loop = 2

    timebase = (0, 1)

    t_start = 3

    duration = 5.5

    duration_min_total = loop * duration

    # each item is times for each loop
    func_times = [
        (t_start, t_start + .1),
        (t_start + duration, t_start + duration + 1,
         t_start + loop * duration - .001)
    ]

    fail_func_times = [
        (-1, 0, 1, 2, 2.5), (3, 5, 8), (t_start + loop * duration + .0001, )
    ]


class FunctionF2(object):

    loop = 3

    timebase = (1, 25)

    duration = 100

    t_start = 2

    duration_min_total = loop * duration

    func_times = [
        (t_start, t_start + .2),
        (t_start + duration / timebase[1],
         t_start + duration / timebase[1] + .2),
        (t_start + 2 * duration / timebase[1],
         t_start + loop * duration / timebase[1] - .2)
    ]

    fail_func_times = [
        (-1, 0, 1, 1.5, 1.8, 1.9),
        (2, 3, 5),
        (6, 8, 9),
        (t_start + loop * duration / timebase[1] + .1, )
    ]


class FunctionF3(object):

    loop = 3

    timebase = (1, 25)

    duration = 1

    t_start = 5

    duration_min_total = loop * duration

    func_times = [
        (t_start, ), (t_start + Fraction(1, 25), ),
        (t_start + Fraction(2, 25), )]

    fail_func_times = [
        (-1, 0, 1, 2, 3, 4.5),
        (t_start, ),
        (t_start + Fraction(1, 25), ),
        (t_start + Fraction(3, 25), t_start + Fraction(4, 25), )
    ]


class FunctionF4(object):

    loop = 1

    timebase = (1, 23)

    duration = 1

    t_start = 5

    duration_min_total = loop * duration

    func_times = [(t_start, )]

    fail_func_times = [
        (-1, 0, 1, 2, 3, 4.5),
        (t_start + Fraction(1, 23), )
    ]


class FunctionF5(object):

    loop = 1

    timebase = (1, 21)

    duration = 3

    t_start = 5

    duration_min_total = loop * duration

    func_times = [
        (t_start, t_start + Fraction(1, 21), t_start + Fraction(2, 21))
    ]

    fail_func_times = [
        (-1, 0, 1, 2, 3, 4.5),
        (t_start + Fraction(3, 21), )
    ]


class ConstFunction(Function):

    a = 0

    cls_name = 'Const'

    editor_props = {
        'loop': int,
        'timebase_numerator': int,
        'timebase_denominator': int,
        'duration': float,
        't_offset': float,
        'a': float,
    }

    def create_func(self):
        from ceed.function.plugin import ConstFunc
        cls = self.function_factory.get('ConstFunc')
        num, denom = self.timebase
        self.func: ConstFunc = cls(
            function_factory=self.function_factory, name=self.name,
            loop=self.loop, timebase_numerator=num, timebase_denominator=denom,
            a=self.a, t_offset=self.t_offset, duration=self.duration)

    def assert_init(self):
        assert self.a == self.func.a


class ConstFunctionF1(FunctionF1, ConstFunction):

    a = .1

    name = 'slow and flat'

    func_values = [a] * 5


class ConstFunctionF2(FunctionF2, ConstFunction):

    name = 'slow and up'

    a = .22

    func_values = [a] * 6


class ConstFunctionF3(FunctionF3, ConstFunction):

    name = 'slow and sides'

    a = .45

    func_values = [a] * 3


class ConstFunctionF4(FunctionF4, ConstFunction):

    name = 'slow and low'

    a = 1.

    func_values = [1.]


class ConstFunctionF5(FunctionF5, ConstFunction):

    name = 'slow and slide'

    a = .67

    func_values = [a] * 3


class LinearFunction(Function):

    m = 0

    b = 0

    cls_name = 'Linear'

    editor_props = {
        'loop': int,
        'timebase_numerator': int,
        'timebase_denominator': int,
        'duration': float,
        't_offset': float,
        'm': float,
        'b': float,
    }

    def create_func(self):
        from ceed.function.plugin import LinearFunc
        cls = self.function_factory.get('LinearFunc')
        num, denom = self.timebase
        self.func: LinearFunc = cls(
            function_factory=self.function_factory, name=self.name,
            loop=self.loop, timebase_numerator=num, timebase_denominator=denom,
            t_offset=self.t_offset, duration=self.duration, m=self.m, b=self.b)

    def assert_init(self):
        assert self.m == self.func.m
        assert self.b == self.func.b


class LinearFunctionF1(FunctionF1, LinearFunction):

    m = 10

    b = 5

    name = 'linear and flat'

    func_values = [5, 1 + 5, 5, 10 + 5, (FunctionF1.duration - .001) * 10 + 5]


class LinearFunctionF2(FunctionF2, LinearFunction):

    m = 2

    b = 4

    name = 'linear and up'

    func_values = [
        b, .2 * m + b, b, .2 * m + b, b,
        (FunctionF2.duration / FunctionF2.timebase[1] - .2) * m + b]


class LinearFunctionF3(FunctionF3, LinearFunction):

    m = 2

    b = 4

    name = 'linear and sides'

    func_values = [b, b, b]


class LinearFunctionF4(FunctionF4, LinearFunction):

    m = 2

    b = 2.2

    name = 'linear and low'

    func_values = [b]


class LinearFunctionF5(FunctionF5, LinearFunction):

    m = 2

    b = 3.3

    name = 'linear and slide'

    func_values = [b, m * 1 / 21 + b, m * 2 / 21 + b]


class ExponentialFunction(Function):

    A = 0

    B = 0

    tau1 = 0

    tau2 = 0

    cls_name = 'Exp'

    editor_props = {
        'loop': int,
        'timebase_numerator': int,
        'timebase_denominator': int,
        'duration': float,
        't_offset': float,
        'A': float,
        'B': float,
        'tau1': float,
        'tau2': float,
    }

    def create_func(self):
        from ceed.function.plugin import ExponentialFunc
        cls = self.function_factory.get('ExponentialFunc')
        num, denom = self.timebase
        self.func: ExponentialFunc = cls(
            function_factory=self.function_factory, name=self.name,
            loop=self.loop, timebase_numerator=num, timebase_denominator=denom,
            t_offset=self.t_offset, duration=self.duration, A=self.A, B=self.B,
            tau1=self.tau1, tau2=self.tau2)

    def assert_init(self):
        assert self.A == self.func.A
        assert self.B == self.func.B
        assert self.tau1 == self.func.tau1
        assert self.tau2 == self.func.tau2


class ExponentialFunctionF1(FunctionF1, ExponentialFunction):

    A = .75

    B = .54

    tau1 = 23

    tau2 = 34

    name = 'exponential and flat'

    func_values = [
        A * exp(0 / tau1) + B * exp(0 / tau2),
        A * exp(-.1 / tau1) + B * exp(-.1 / tau2),
        A * exp(0 / tau1) + B * exp(0 / tau2),
        A * exp(-1 / tau1) + B * exp(-1 / tau2),
        A * exp(-(FunctionF1.duration - .001) / tau1) +
        B * exp(-(FunctionF1.duration - .001) / tau2)
    ]


class ExponentialFunctionF2(FunctionF2, ExponentialFunction):

    A = .58

    B = .06

    tau1 = 27

    tau2 = 28

    name = 'exponential and up'

    func_values = [
        A * exp(0 / tau1) + B * exp(0 / tau2),
        A * exp(-.2 / tau1) + B * exp(-.2 / tau2),
        A * exp(0 / tau1) + B * exp(0 / tau2),
        A * exp(-.2 / tau1) + B * exp(-.2 / tau2),
        A * exp(0 / tau1) + B * exp(0 / tau2),
        A * exp(-(FunctionF2.duration / FunctionF2.timebase[1] - .2) / tau1) +
        B * exp(-(FunctionF2.duration / FunctionF2.timebase[1] - .2) / tau2)
    ]


class ExponentialFunctionF3(FunctionF3, ExponentialFunction):

    A = .43

    B = .34

    tau1 = 46

    tau2 = 8

    name = 'exponential and sides'

    func_values = [
        A * exp(0 / tau1) + B * exp(0 / tau2),
        A * exp(0 / tau1) + B * exp(0 / tau2),
        A * exp(0 / tau1) + B * exp(0 / tau2)
    ]


class ExponentialFunctionF4(FunctionF4, ExponentialFunction):

    A = .78

    B = .98

    tau1 = 11

    tau2 = 33

    name = 'exponential and low'

    func_values = [A * exp(0 / tau1) + B * exp(0 / tau2)]


class ExponentialFunctionF5(FunctionF5, ExponentialFunction):

    A = 1

    B = .5

    tau1 = 4

    tau2 = 12

    name = 'exponential and slide'

    func_values = [
        A * exp(0 / tau1) + B * exp(0 / tau2),
        A * exp(-(1 / 21) / tau1) + B * exp(-(1 / 21) / tau2),
        A * exp(-(2 / 21) / tau1) + B * exp(-(2 / 21) / tau2),
    ]


class CosFunction(Function):

    f = 0

    A = 0

    th0 = 0

    b = 0

    cls_name = 'Cos'

    editor_props = {
        'loop': int,
        'timebase_numerator': int,
        'timebase_denominator': int,
        'duration': float,
        't_offset': float,
        'f': float,
        'A': float,
        'th0': float,
        'b': float,
    }

    def create_func(self):
        from ceed.function.plugin import CosFunc
        cls = self.function_factory.get('CosFunc')
        num, denom = self.timebase
        self.func: CosFunc = cls(
            function_factory=self.function_factory, name=self.name,
            loop=self.loop, timebase_numerator=num, timebase_denominator=denom,
            t_offset=self.t_offset, duration=self.duration, f=self.f, A=self.A,
            th0=self.th0, b=self.b)

    def assert_init(self):
        assert self.f == self.func.f
        assert self.A == self.func.A
        assert self.th0 == self.func.th0
        assert self.b == self.func.b


class CosFunctionF1(FunctionF1, CosFunction):

    f = .75

    A = .54

    th0 = 23

    b = 34

    name = 'cos and flat'

    func_values = [
        A * cos(2 * pi * f * 0 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * .1 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * 0 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * 1 + th0 * pi / 180.) + b,
        A * cos(
            2 * pi * f * (FunctionF1.duration - .001) + th0 * pi / 180.) + b,
    ]


class CosFunctionF2(FunctionF2, CosFunction):

    f = .58

    A = .06

    th0 = 27

    b = 28

    name = 'cos and up'

    func_values = [
        A * cos(2 * pi * f * 0 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * .2 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * 0 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * .2 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * 0 + th0 * pi / 180.) + b,
        A * cos(
            2 * pi * f * (FunctionF2.duration / FunctionF2.timebase[1] - .2) +
            th0 * pi / 180.) + b,
    ]


class CosFunctionF3(FunctionF3, CosFunction):

    f = .43

    A = .34

    th0 = 46

    b = 8

    name = 'cos and sides'

    func_values = [
        A * cos(2 * pi * f * 0 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * 0 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * 0 + th0 * pi / 180.) + b,
    ]


class CosFunctionF4(FunctionF4, CosFunction):

    f = .78

    A = .98

    th0 = 11

    b = 33

    name = 'cos and low'

    func_values = [A * cos(2 * pi * f * 0 + th0 * pi / 180.) + b, ]


class CosFunctionF5(FunctionF5, CosFunction):

    f = 1

    A = .5

    th0 = 4

    b = 12

    name = 'cos and slide'

    func_values = [
        A * cos(2 * pi * f * 0 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * 1 / 21 + th0 * pi / 180.) + b,
        A * cos(2 * pi * f * 2 / 21 + th0 * pi / 180.) + b,
    ]


class GroupFunction(Function):

    wrapper_classes = []

    wrapper_funcs: List[Function] = []

    cls_name = 'Group'

    editor_props = {
        'loop': int,
        'timebase_numerator': int,
        'timebase_denominator': int,
    }

    @property
    def duration(self):
        return sum((cls.duration_min_total for cls in self.wrapper_classes))

    @property
    def duration_min_total(self):
        return self.duration * self.loop

    @property
    def func_values(self):
        return list(itertools.chain(
            *(cls.func_values for cls in self.wrapper_classes))) * self.loop

    def __init__(self, *largs, **kwargs):
        self.set_func_times()
        super(GroupFunction, self).__init__(*largs, **kwargs)

    @classmethod
    def set_func_times(cls):
        def tb(obj):
            num, denum = obj.timebase
            if not num:
                return 1
            return Fraction(num, denum)

        times = cls.func_times = []
        t_start = cls.t_start
        for i in range(cls.loop):
            loop_times = []
            for func_cls in cls.wrapper_classes:
                cls_t_start = func_cls.t_start
                for time_group in func_cls.func_times:
                    for t in time_group:
                        loop_times.append(t - cls_t_start + t_start)
                t_start += func_cls.duration_min_total * tb(func_cls)
            times.append(loop_times)

        fail_times = cls.fail_func_times = []
        t_start = cls.t_start
        func_cls = cls.wrapper_classes[0]
        cls_t_start = func_cls.t_start

        for i in range(cls.loop):
            loop_fail_times = []
            for t in func_cls.fail_func_times[0]:
                loop_fail_times.append(t - cls_t_start + t_start)
            t_start += func_cls.duration_min_total * tb(func_cls)
            fail_times.append(loop_fail_times)

        func_cls = cls.wrapper_classes[-1]
        func_min_duration = func_cls.duration_min_total * tb(func_cls)
        duration = sum((
            c.duration_min_total for c in cls.wrapper_classes)) * tb(func_cls)
        total_duration = duration * cls.loop
        func_fail_times = [
            t - func_min_duration + total_duration
            for t in func_cls.fail_func_times[-1]]
        fail_times.append(func_fail_times)

    def create_func(self):
        funcs = self.wrapper_funcs = []
        cls = self.function_factory.get('FuncGroup')
        num, denom = self.timebase
        func_group: FuncGroup = cls(
            function_factory=self.function_factory, loop=self.loop,
            name=self.name, timebase_numerator=num, timebase_denominator=denom)
        self.func: FuncGroup = func_group

        for wrapper_cls in self.wrapper_classes:
            wrapper_func = wrapper_cls(
                app=self.app, show_in_gui=False,
                function_factory=self.function_factory)
            wrapper_func.create_func()
            funcs.append(wrapper_func)
            func_group.add_func(wrapper_func.func)

    def show_in_gui(self, add_func=None, display_func=None):
        super(GroupFunction, self).show_in_gui(
            add_func=add_func, display_func=display_func)

        for func in self.wrapper_funcs:
            self.funcs_container.show_child_func_in_func(self.func, func.func)

    def assert_init(self):
        assert len(self.wrapper_funcs) == len(self.func.funcs)
        for i, func in enumerate(self.wrapper_funcs):
            assert func.func in self.func.funcs
            assert self.func.funcs.index(func.func) == i
            func.assert_init()


class GroupFunctionF1(GroupFunction):

    loop = 2

    wrapper_classes = [ConstFunctionF1, ExponentialFunctionF1]

    timebase = FunctionF1.timebase

    t_start = 12

    name = 'group and flat'


class GroupFunctionF2(GroupFunction):

    loop = 3

    wrapper_classes = [ConstFunctionF2, ExponentialFunctionF2]

    timebase = FunctionF2.timebase

    t_start = 11

    name = 'group and up'


class GroupFunctionF3(GroupFunction):

    loop = 2

    wrapper_classes = [LinearFunctionF3, CosFunctionF3]

    timebase = FunctionF3.timebase

    t_start = 10

    name = 'group and sides'


class GroupFunctionF4(GroupFunction):

    loop = 3

    wrapper_classes = [LinearFunctionF4, CosFunctionF4]

    timebase = FunctionF4.timebase

    t_start = 4

    name = 'group and low'


class GroupFunctionF5(GroupFunction):

    loop = 2

    wrapper_classes = [LinearFunctionF5, CosFunctionF5]

    timebase = FunctionF5.timebase

    t_start = 6

    name = 'group and slide'


func_classes = [
    ConstFunctionF1, ConstFunctionF2, ConstFunctionF3, ConstFunctionF4,
    ConstFunctionF5,
    LinearFunctionF1, LinearFunctionF2, LinearFunctionF3, LinearFunctionF4,
    LinearFunctionF5,
    ExponentialFunctionF1, ExponentialFunctionF2, ExponentialFunctionF3,
    ExponentialFunctionF4, ExponentialFunctionF5,
    CosFunctionF1, CosFunctionF2, CosFunctionF3, CosFunctionF4, CosFunctionF5,
    GroupFunctionF1, GroupFunctionF2, GroupFunctionF3, GroupFunctionF4,
    GroupFunctionF5
]

func_classes_cos = [
    CosFunctionF1, CosFunctionF2, CosFunctionF3, CosFunctionF4, CosFunctionF5]

func_classes_linear = [
    LinearFunctionF1, LinearFunctionF2, LinearFunctionF3, LinearFunctionF4,
    LinearFunctionF5]

func_classes_group = [
    GroupFunctionF1, GroupFunctionF2, GroupFunctionF3, GroupFunctionF4,
    GroupFunctionF5]

func_classes_dedup = [
    ConstFunctionF5, LinearFunctionF5, ExponentialFunctionF5, CosFunctionF5,
    GroupFunctionF5,
]

fake_plugin_function = """
from kivy.properties import NumericProperty
from ceed.function import CeedFunc, FuncBase

class FakeFunc(CeedFunc):

    val = NumericProperty(0.)

    def __init__(self, name='Fake', description='y(t) = val', **kwargs):
        super().__init__(name=name, description=description, **kwargs)

    def __call__(self, t):
        super().__call__(t)
        return self.val

    def get_gui_props(self):
        d = super().get_gui_props()
        d['val'] = None
        return d

    def get_state(self, *largs, **kwargs):
        d = super().get_state(*largs, **kwargs)
        d['val'] = self.val
        return d

def get_ceed_functions(function_factory):
    return FakeFunc,
"""

fake_plugin_distribution = """
from kivy.properties import NumericProperty
from ceed.function.param_noise import NoiseType, NoiseBase

class FakeNoise(NoiseBase):

    val = NumericProperty(0)

    def sample(self) -> float:
        return 43

    def get_config(self) -> dict:
        config = super().get_config()
        config['val'] = self.val
        return config

    def get_prop_pretty_name(self):
        names = super().get_prop_pretty_name()
        names['val'] = 'Value'
        return names

def get_ceed_distributions(function_factory):
    return FakeNoise,
"""

fake_plugin = fake_plugin_function + fake_plugin_distribution

noise_test_parameters = [
    ('GaussianNoise',
     {'min_val': -1.5, 'max_val': 1.5, 'mean_val': 0.1, 'stdev': .15}),
    ('UniformNoise',
     {'min_val': -1.4, 'max_val': 1.3}),
    ('DiscreteNoise',
     {'start_value': 10, 'step': 2, 'num_values': 1000,
      'with_replacement': False}),
    ('DiscreteNoise',
     {'start_value': 10, 'step': 2, 'num_values': 8,
      'with_replacement': True}),
    ('DiscreteListNoise',
     {'csv_list': '5.6, 34,65.,34.54,45  ,34 , 43,.445',
      'with_replacement': True}),
    ('DiscreteListNoise',
     {'csv_list': '5.6, 34,65.,34.54,' + ','.join(map(str, range(100))),
      'with_replacement': False}),
]
