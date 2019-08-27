import pytest
import math
from typing import Type, List
from ceed.function import FuncBase, FuncGroup, FunctionFactoryBase, \
    register_all_functions, FuncDoneException


@pytest.fixture
def function_factory() -> FunctionFactoryBase:
    function_factory = FunctionFactoryBase()
    register_all_functions(function_factory)

    function_factory.test_changes_count = 0

    def count_changes(*largs):
        function_factory.test_changes_count += 1
    function_factory.fbind('on_changed', count_changes)

    yield function_factory


def test_register_funcs():
    from ceed.function.plugin import ConstFunc, LinearFunc
    function_factory = FunctionFactoryBase()
    count = 0

    def count_changes(*largs):
        nonlocal count
        count += 1
    function_factory.fbind('on_changed', count_changes)

    assert not function_factory.funcs_cls
    assert not function_factory.funcs_user
    assert not function_factory.funcs_inst
    assert not function_factory.funcs_inst_default
    assert not function_factory.get_classes()
    assert not function_factory.get_names()

    function_factory.register(ConstFunc)
    assert count
    assert function_factory.funcs_cls['ConstFunc'] is ConstFunc
    assert isinstance(function_factory.funcs_inst['Const'], ConstFunc)
    assert isinstance(function_factory.funcs_inst_default['Const'], ConstFunc)
    assert ConstFunc in function_factory.get_classes()
    assert 'ConstFunc' in function_factory.get_names()

    f = LinearFunc(function_factory=function_factory)
    count = 0
    function_factory.register(LinearFunc, instance=f)
    assert count
    assert function_factory.funcs_cls['LinearFunc'] is LinearFunc
    assert function_factory.funcs_inst['Linear'] is f
    assert function_factory.funcs_inst_default['Linear'] is f
    assert LinearFunc in function_factory.get_classes()
    assert 'LinearFunc' in function_factory.get_names()
    assert not function_factory.funcs_user


def test_auto_register(function_factory: FunctionFactoryBase):
    from ceed.function.plugin import ConstFunc, LinearFunc, CosFunc, \
        ExponentialFunc
    assert not function_factory.funcs_user
    assert function_factory.get('ConstFunc') is ConstFunc
    assert function_factory.get('LinearFunc') is LinearFunc
    assert function_factory.get('CosFunc') is CosFunc
    assert function_factory.get('ExponentialFunc') is ExponentialFunc

    assert isinstance(function_factory.funcs_inst['Const'], ConstFunc)
    assert isinstance(function_factory.funcs_inst['Linear'], LinearFunc)
    assert isinstance(function_factory.funcs_inst['Cos'], CosFunc)
    assert isinstance(
        function_factory.funcs_inst['Exp'], ExponentialFunc)


def test_register_user_func(function_factory: FunctionFactoryBase):
    assert not function_factory.funcs_user

    const_cls = function_factory.get('ConstFunc')
    f = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f')
    f2 = const_cls(
        function_factory=function_factory, duration=5, a=.9, name='f2')

    function_factory.test_changes_count = 0
    function_factory.add_func(f)
    assert function_factory.test_changes_count
    assert f in function_factory.funcs_user
    assert function_factory.funcs_inst['f'] is f

    function_factory.test_changes_count = 0
    function_factory.add_func(f2)
    assert function_factory.test_changes_count
    assert f2 in function_factory.funcs_user
    assert function_factory.funcs_inst['f2'] is f2


def test_factory_re_register(function_factory: FunctionFactoryBase):
    from ceed.function.plugin import ConstFunc, LinearFunc
    with pytest.raises(Exception):
        function_factory.register(ConstFunc)

    with pytest.raises(Exception):
        function_factory.register(LinearFunc)


def test_factory_func_unique_names(function_factory: FunctionFactoryBase):
    assert not function_factory.funcs_user
    function_factory.test_changes_count = 0

    const_cls = function_factory.get('ConstFunc')
    f = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f')
    function_factory.add_func(f)
    f2 = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f')
    function_factory.add_func(f2)

    def assert_not_f():
        assert function_factory.test_changes_count
        assert f in function_factory.funcs_user
        assert f2 in function_factory.funcs_user
        assert len(function_factory.funcs_user) == 2
        f2_name = f2.name
        assert f.name == 'f'
        assert f2_name != 'f'
        assert f2_name

        assert function_factory.funcs_inst['f'] is f
        assert function_factory.funcs_inst[f2_name] is f2
    assert_not_f()

    function_factory.test_changes_count = 0
    f2.name = 'f2'
    assert function_factory.test_changes_count
    assert f in function_factory.funcs_user
    assert f2 in function_factory.funcs_user
    assert len(function_factory.funcs_user) == 2
    assert f.name == 'f'
    assert f2.name == 'f2'

    assert function_factory.funcs_inst['f'] is f
    assert function_factory.funcs_inst['f2'] is f2

    function_factory.test_changes_count = 0
    f2.name = 'f'
    assert_not_f()


def test_factory_func_remove(function_factory: FunctionFactoryBase):
    assert not function_factory.funcs_user
    initial_funcs_n = len(function_factory.funcs_inst)

    const_cls = function_factory.get('ConstFunc')
    f = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f')
    function_factory.add_func(f)
    f2 = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f2')
    function_factory.add_func(f2)

    assert len(function_factory.funcs_inst) == initial_funcs_n + 2

    function_factory.test_changes_count = 0
    function_factory.remove_func(f2)

    assert function_factory.test_changes_count
    assert f in function_factory.funcs_user
    assert f2 not in function_factory.funcs_user
    assert len(function_factory.funcs_user) == 1
    assert f.name == 'f'
    assert f2.name == 'f2'

    assert function_factory.funcs_inst['f'] is f
    assert 'f2' not in function_factory.funcs_inst
    assert len(function_factory.funcs_inst) == initial_funcs_n + 1

    function_factory.test_changes_count = 0
    f2.name = 'f'

    assert not function_factory.test_changes_count
    assert f.name == 'f'
    assert f2.name == 'f'

    function_factory.test_changes_count = 0
    function_factory.remove_func(f)

    assert function_factory.test_changes_count
    assert f not in function_factory.funcs_user
    assert f2 not in function_factory.funcs_user
    assert not function_factory.funcs_user
    assert f.name == 'f'
    assert f2.name == 'f'

    assert 'f' not in function_factory.funcs_inst
    assert 'f2' not in function_factory.funcs_inst

    assert len(function_factory.funcs_inst) == initial_funcs_n


def test_clear_funcs(function_factory: FunctionFactoryBase):
    assert not function_factory.funcs_user
    initial_funcs_n = len(function_factory.funcs_inst)

    const_cls = function_factory.get('ConstFunc')
    f = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f')
    function_factory.add_func(f)
    f2 = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f2')
    function_factory.add_func(f2)

    assert len(function_factory.funcs_inst) == initial_funcs_n + 2

    function_factory.test_changes_count = 0
    function_factory.clear_added_funcs()
    assert len(function_factory.funcs_inst) == initial_funcs_n
    assert not function_factory.funcs_user


def test_recover_funcs(function_factory: FunctionFactoryBase):
    f1 = function_factory.get('ConstFunc')(
        function_factory=function_factory, a=.5, name='f1')
    f2 = function_factory.get('LinearFunc')(
        function_factory=function_factory, m=.5, b=.2, name='f2')
    f3 = function_factory.get('ExponentialFunc')(
        function_factory=function_factory, A=.5, B=.2, tau1=1.3, tau2=1.5,
        name='f3')
    f4 = function_factory.get('CosFunc')(
        function_factory=function_factory, f=.5, A=3.4, th0=7.5, b=12.2,
        name='f4')

    function_factory.add_func(f1)
    function_factory.add_func(f2)
    function_factory.add_func(f3)
    function_factory.add_func(f4)

    funcs = function_factory.save_functions()
    assert len(funcs) == 4

    recovered_funcs, name_mapping = function_factory.recover_funcs(funcs)
    assert len(recovered_funcs) == 4
    assert len(name_mapping) == 4

    for f_name in ('f1', 'f2', 'f3', 'f4'):
        assert f_name in name_mapping
        assert name_mapping[f_name] != f_name
        assert f_name in function_factory.funcs_inst
        assert name_mapping[f_name] in function_factory.funcs_inst

        for name in function_factory.funcs_inst[f_name].get_gui_props():
            original_f = function_factory.funcs_inst[f_name]
            new_f = function_factory.funcs_inst[name_mapping[f_name]]

            if name == 'name':
                assert original_f.name != new_f.name
                continue

            assert getattr(original_f, name) == getattr(new_f, name)


def test_make_function(function_factory: FunctionFactoryBase):
    f1 = function_factory.get('ConstFunc')(
        function_factory=function_factory, a=.5, name='f1')
    f2 = function_factory.get('LinearFunc')(
        function_factory=function_factory, m=.5, b=.2, name='f2')
    f3 = function_factory.get('ExponentialFunc')(
        function_factory=function_factory, A=.5, B=.2, tau1=1.3, tau2=1.5,
        name='f3')
    f4 = function_factory.get('CosFunc')(
        function_factory=function_factory, f=.5, A=3.4, th0=7.5, b=12.2,
        name='f4')

    funcs = f1, f2, f3, f4
    states = [f.get_state() for f in funcs]
    new_funcs = [function_factory.make_func(state) for state in states]
    assert len(new_funcs) == len(funcs)

    for new_func, f in zip(new_funcs, funcs):
        for name in f.get_gui_props():
            if name == 'name':
                assert f.name != new_func.name
                continue

            assert getattr(f, name) == getattr(new_func, name)

    # close should make them identical in all ways
    new_funcs = [
        function_factory.make_func(state, clone=True) for state in states]
    assert len(new_funcs) == len(funcs)

    for new_func, f in zip(new_funcs, funcs):
        for name in f.get_gui_props():
            assert getattr(f, name) == getattr(new_func, name)

    # provide instances
    new_funcs = [
        function_factory.make_func(
            state, instance=function_factory.get(state['cls'])(
                function_factory=function_factory), clone=True)
        for state in states]
    assert len(new_funcs) == len(funcs)

    for new_func, f in zip(new_funcs, funcs):
        for name in f.get_gui_props():
            assert getattr(f, name) == getattr(new_func, name)


def test_group_recursive(function_factory: FunctionFactoryBase):
    factory = function_factory
    const_cls = factory.get('ConstFunc')

    g1 = FuncGroup(function_factory=factory)

    g2 = FuncGroup(function_factory=factory)
    g1.add_func(g2)
    f = const_cls(function_factory=factory, a=.9, duration=2)
    g2.add_func(f)
    f1 = const_cls(function_factory=factory, a=.1, duration=2)
    g2.add_func(f1)

    f2 = const_cls(function_factory=factory, a=.25, duration=2)
    g1.add_func(f2)
    f3 = const_cls(function_factory=factory, a=.75, duration=2)
    g1.add_func(f3)

    g3 = FuncGroup(function_factory=factory)
    g1.add_func(g3)
    f4 = const_cls(function_factory=factory, a=.8, duration=2)
    g3.add_func(f4)
    f5 = const_cls(function_factory=factory, a=.2, duration=2)
    g3.add_func(f5)

    def assert_times():
        assert list(g2.get_funcs()) == [g2, f, f1]
        assert g2.duration == 4
        assert g2.duration_min == 4
        assert g2.duration_min_total == 4

        assert list(g3.get_funcs()) == [g3, f4, f5]
        assert g3.duration == 4
        assert g3.duration_min == 4
        assert g3.duration_min_total == 4

        assert list(g1.get_funcs()) == [g1, g2, f, f1, f2, f3, g3, f4, f5]
        assert g1.duration == 12
        assert g1.duration_min == 12
        assert g1.duration_min_total == 12
    assert_times()

    g1.init_func(2)
    with pytest.raises(ValueError):
        g1(0)
    assert g1(2) == .9
    assert g1(4) == .1
    assert g1(6) == .25
    assert g1(8) == .75
    assert g1(10) == .8
    assert g1(12) == .2
    with pytest.raises(FuncDoneException):
        g1(14)

    assert_times()


def test_timebase_func(function_factory: FunctionFactoryBase):
    const_cls = function_factory.get('ConstFunc')
    f = const_cls(function_factory=function_factory)
    assert f.get_timebase() == 1

    f = const_cls(function_factory=function_factory, timebase_numerator=1,
                  timebase_denominator=2)
    assert f.get_timebase() == 0.5


def test_timebase_group(function_factory: FunctionFactoryBase):
    g = FuncGroup(function_factory=function_factory)
    assert g.get_timebase() == 1

    const_cls = function_factory.get('ConstFunc')
    f = const_cls(function_factory=function_factory)
    assert f.get_timebase() == 1

    g.add_func(f)
    assert f.get_timebase() == 1

    g.timebase_numerator = 1
    g.timebase_denominator = 4
    assert g.get_timebase() == 1 / 4
    assert f.get_timebase() == 1 / 4

    f.timebase_numerator = 1
    f.timebase_denominator = 8
    assert g.get_timebase() == 1 / 4
    assert f.get_timebase() == 1 / 8

    f.timebase_numerator = 0
    assert g.get_timebase() == 1 / 4
    assert f.get_timebase() == 1 / 4

    f.timebase_numerator = 1
    assert g.get_timebase() == 1 / 4
    assert f.get_timebase() == 1 / 8

    g.timebase_numerator = 0
    assert g.get_timebase() == 1
    assert f.get_timebase() == 1 / 8


def test_recursive_timebase(function_factory: FunctionFactoryBase):
    factory = function_factory
    const_cls = factory.get('ConstFunc')

    g1 = FuncGroup(
        function_factory=factory, timebase_denominator=2, timebase_numerator=1)

    g2 = FuncGroup(
        function_factory=factory, timebase_denominator=4, timebase_numerator=1)
    g1.add_func(g2)
    f = const_cls(
        function_factory=factory, a=.9, duration=2, timebase_denominator=8,
        timebase_numerator=1)
    g2.add_func(f)
    f1 = const_cls(function_factory=factory, a=.1, duration=2)
    g2.add_func(f1)

    f2 = const_cls(function_factory=factory, a=.25, duration=2)
    g1.add_func(f2)
    f3 = const_cls(function_factory=factory, a=.75, duration=2)
    g1.add_func(f3)

    g3 = FuncGroup(function_factory=factory)
    g1.add_func(g3)
    f4 = const_cls(
        function_factory=factory, a=.8, duration=2, timebase_denominator=8,
        timebase_numerator=1)
    g3.add_func(f4)
    f5 = const_cls(function_factory=factory, a=.2, duration=2)
    g3.add_func(f5)

    def assert_times():
        assert list(g2.get_funcs()) == [g2, f, f1]
        assert g2.duration == (2 / 8 + 2 / 4) * 4
        assert g2.duration_min == (2 / 8 + 2 / 4) * 4
        assert g2.duration_min_total == (2 / 8 + 2 / 4) * 4

        assert list(g3.get_funcs()) == [g3, f4, f5]
        assert g3.duration == (2 / 8 + 2 / 2) * 2
        assert g3.duration_min == (2 / 8 + 2 / 2) * 2
        assert g3.duration_min_total == (2 / 8 + 2 / 2) * 2

        assert list(g1.get_funcs()) == [g1, g2, f, f1, f2, f3, g3, f4, f5]
        assert g1.duration == (4 / 8 + 2 / 4 + 6 / 2) * 2
        assert g1.duration_min == (4 / 8 + 2 / 4 + 6 / 2) * 2
        assert g1.duration_min_total == (4 / 8 + 2 / 4 + 6 / 2) * 2
    assert_times()

    g1.init_func(2)
    with pytest.raises(ValueError):
        g1(0)
    assert g1(2) == .9
    assert g1(2 + 2 / 8) == .1
    assert g1(2 + 2 / 8 + 2 / 4) == .25
    assert g1(2 + 2 / 8 + 2 / 4 + 2 / 2) == .75
    assert g1(2 + 2 / 8 + 2 / 4 + 4 / 2) == .8
    assert g1(2 + 4 / 8 + 2 / 4 + 4 / 2) == .2
    with pytest.raises(FuncDoneException):
        g1(2 + 4 / 8 + 2 / 4 + 6 / 2)

    assert_times()
