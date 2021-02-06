import pytest
import sys
import copy
import math
from fractions import Fraction
import pathlib
import os
from typing import Type, List, Tuple

from ceed.function import FuncBase, FuncGroup, FunctionFactoryBase, \
    register_all_functions, FuncDoneException, \
    register_external_functions
from .common import add_prop_watch
from .test_app.examples.funcs import fake_plugin_function, \
    fake_plugin_distribution, fake_plugin, noise_test_parameters, \
    func_classes, func_classes_dedup
from ceed.utils import collapse_list_to_counts


def register_callback_distribution(
        function_factory: FunctionFactoryBase, counter: list, class_count: int
) -> Type:
    UniformNoise = function_factory.param_noise_factory.get_cls('UniformNoise')

    def sample(self) -> float:
        counter[0] += 1
        return super(cls, self).sample()

    cls = type(
        f'CallbackNoise{class_count}', (UniformNoise, ), {'sample': sample})

    function_factory.param_noise_factory.register_class(cls)
    return cls


def get_function_values(func: FuncBase, frame_rate: float) -> List[float]:
    frame_rate = int(frame_rate)
    i = 1

    func.init_func_tree()
    func.init_func(Fraction(i, frame_rate))

    values = []
    while True:
        try:
            values.append(func(Fraction(i, frame_rate)))
        except FuncDoneException:
            break
        i += 1

    return values


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

    assert isinstance(function_factory.funcs_inst_default['Const'], ConstFunc)
    assert isinstance(function_factory.funcs_inst_default['Linear'], LinearFunc)
    assert isinstance(function_factory.funcs_inst_default['Cos'], CosFunc)
    assert isinstance(
        function_factory.funcs_inst_default['Exp'], ExponentialFunc)

    assert isinstance(function_factory.funcs_inst['Const'], ConstFunc)
    assert isinstance(function_factory.funcs_inst['Linear'], LinearFunc)
    assert isinstance(function_factory.funcs_inst['Cos'], CosFunc)
    assert isinstance(
        function_factory.funcs_inst['Exp'], ExponentialFunc)

    assert function_factory.get('SomeFunc') is None


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
    with pytest.raises(ValueError):
        function_factory.register(ConstFunc)

    with pytest.raises(ValueError):
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
    assert function_factory.remove_func(f2)

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
    assert function_factory.remove_func(f)

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
    assert function_factory.test_changes_count


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
                assert new_f.name.startswith(original_f.name)
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


def create_recursive_funcs(function_factory):
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

    return g1, g2, f, f1, f2, f3, g3, f4, f5


def test_recursive_timebase(function_factory: FunctionFactoryBase):
    g1, g2, f, f1, f2, f3, g3, f4, f5 = create_recursive_funcs(
        function_factory)

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


def test_recursive_timebase_trigger(function_factory: FunctionFactoryBase):
    g1, g2, f, f1, f2, f3, g3, f4, f5 = create_recursive_funcs(
        function_factory)

    for func in g1.get_funcs():
        add_prop_watch(func, 'timebase', 'test_timebase_changes_count')

    g1.timebase_denominator = 16
    assert g1.test_timebase_changes_count
    assert not g2.test_timebase_changes_count
    assert not f.test_timebase_changes_count
    assert not f1.test_timebase_changes_count
    assert f2.test_timebase_changes_count
    assert f3.test_timebase_changes_count
    assert g3.test_timebase_changes_count
    assert not f4.test_timebase_changes_count
    assert f5.test_timebase_changes_count
    assert g1.duration == (4 / 8 + 2 / 4 + 6 / 16) * 16

    for func in g1.get_funcs():
        setattr(func, 'test_timebase_changes_count', 0)

    f2.timebase_numerator = 1
    assert not g1.test_timebase_changes_count
    assert not g2.test_timebase_changes_count
    assert not f.test_timebase_changes_count
    assert not f1.test_timebase_changes_count
    assert f2.test_timebase_changes_count
    assert not f3.test_timebase_changes_count
    assert not g3.test_timebase_changes_count
    assert not f4.test_timebase_changes_count
    assert not f5.test_timebase_changes_count
    assert g1.duration == (4 / 8 + 2 / 4 + 4 / 16 + 2) * 16


def test_recursive_parent_func(function_factory: FunctionFactoryBase):
    g1, g2, f, f1, f2, f3, g3, f4, f5 = create_recursive_funcs(
        function_factory)
    assert f5.parent_func is g3
    assert f4.parent_func is g3
    assert g3.parent_func is g1
    assert f3.parent_func is g1
    assert f2.parent_func is g1
    assert f1.parent_func is g2
    assert f.parent_func is g2
    assert g2.parent_func is g1


def test_can_other_func_be_added(function_factory: FunctionFactoryBase):
    g1, g2, f, f1, f2, f3, g3, f4, f5 = create_recursive_funcs(
        function_factory)

    assert g1.can_other_func_be_added(g2)
    assert g1.can_other_func_be_added(g3)
    assert not g1.can_other_func_be_added(g1)

    assert not g2.can_other_func_be_added(g1)
    assert g2.can_other_func_be_added(g3)
    assert not g2.can_other_func_be_added(g2)

    assert not g3.can_other_func_be_added(g1)
    assert g3.can_other_func_be_added(g2)
    assert not g3.can_other_func_be_added(g3)


def test_func_ref(function_factory: FunctionFactoryBase):
    const_cls = function_factory.get('ConstFunc')
    f = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f')
    f2 = const_cls(
        function_factory=function_factory, duration=5, a=.9, name='f')

    function_factory.add_func(f)

    ref1 = function_factory.get_func_ref(name='f')
    ref2 = function_factory.get_func_ref(func=f2)

    assert ref1.func is f
    assert ref2.func is f2
    assert f.has_ref
    assert f2.has_ref
    assert f in function_factory._ref_funcs
    assert f2 in function_factory._ref_funcs

    function_factory.return_func_ref(ref1)
    assert ref2.func is f2
    assert not f.has_ref
    assert f2.has_ref
    assert f not in function_factory._ref_funcs
    assert f2 in function_factory._ref_funcs

    function_factory.return_func_ref(ref2)
    assert not f.has_ref
    assert not f2.has_ref
    assert f not in function_factory._ref_funcs
    assert f2 not in function_factory._ref_funcs


def test_return_not_added_func_ref(function_factory: FunctionFactoryBase):
    from ceed.function import CeedFuncRef
    const_cls = function_factory.get('ConstFunc')
    f = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f')
    ref = CeedFuncRef(function_factory=function_factory, func=f)

    with pytest.raises(ValueError):
        function_factory.return_func_ref(ref)


def test_remove_func_with_ref(function_factory: FunctionFactoryBase):
    const_cls = function_factory.get('ConstFunc')
    f = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f')
    f2 = const_cls(
        function_factory=function_factory, duration=5, a=.9, name='f2')
    f3 = const_cls(
        function_factory=function_factory, duration=5, a=.9, name='f3')

    function_factory.add_func(f)
    function_factory.add_func(f2)
    function_factory.add_func(f3)

    assert function_factory.funcs_inst['f'] is f
    assert f in function_factory.funcs_user
    assert function_factory.funcs_inst['f2'] is f2
    assert f2 in function_factory.funcs_user
    assert function_factory.funcs_inst['f3'] is f3
    assert f3 in function_factory.funcs_user

    ref = function_factory.get_func_ref(name='f')
    ref3 = function_factory.get_func_ref(name='f3')
    assert not function_factory.remove_func(f)

    assert ref.func is f
    assert f.has_ref
    assert function_factory.funcs_inst['f'] is f
    assert f in function_factory.funcs_user
    assert function_factory.funcs_inst['f2'] is f2
    assert f2 in function_factory.funcs_user
    assert function_factory.funcs_inst['f3'] is f3
    assert f3 in function_factory.funcs_user

    assert function_factory.remove_func(f2)

    assert function_factory.funcs_inst['f'] is f
    assert f in function_factory.funcs_user
    assert 'f2' not in function_factory.funcs_inst
    assert f2 not in function_factory.funcs_user
    assert function_factory.funcs_inst['f3'] is f3
    assert f3 in function_factory.funcs_user

    assert not function_factory.remove_func(f3)

    assert ref3.func is f3
    assert f3.has_ref
    assert function_factory.funcs_inst['f'] is f
    assert f in function_factory.funcs_user
    assert function_factory.funcs_inst['f3'] is f3
    assert f3 in function_factory.funcs_user

    assert function_factory.remove_func(f3, force=True)

    assert ref3.func is f3
    assert f3.has_ref
    assert function_factory.funcs_inst['f'] is f
    assert f in function_factory.funcs_user
    assert 'f3' not in function_factory.funcs_inst
    assert f3 not in function_factory.funcs_user

    assert not function_factory.remove_func(f)

    assert ref.func is f
    assert f.has_ref
    assert function_factory.funcs_inst['f'] is f
    assert f in function_factory.funcs_user

    function_factory.return_func_ref(ref)
    assert not f.has_ref

    assert function_factory.remove_func(f)

    assert 'f' not in function_factory.funcs_inst
    assert f not in function_factory.funcs_user

    function_factory.return_func_ref(ref3)
    assert not f3.has_ref


def test_clear_funcs_with_ref(function_factory: FunctionFactoryBase):
    const_cls = function_factory.get('ConstFunc')
    f = const_cls(
        function_factory=function_factory, duration=4, a=.7, name='f')
    f2 = const_cls(
        function_factory=function_factory, duration=5, a=.9, name='f2')

    function_factory.add_func(f)
    function_factory.add_func(f2)

    assert function_factory.funcs_inst['f'] is f
    assert f in function_factory.funcs_user
    assert function_factory.funcs_inst['f2'] is f2
    assert f2 in function_factory.funcs_user

    ref = function_factory.get_func_ref(name='f')
    function_factory.clear_added_funcs()

    # f should not have been removed, but f2 was removed
    assert ref.func is f
    assert f.has_ref
    assert function_factory.funcs_inst['f'] is f
    assert f in function_factory.funcs_user
    assert 'f2' not in function_factory.funcs_inst
    assert f2 not in function_factory.funcs_user

    function_factory.clear_added_funcs(force=True)

    assert ref.func is f
    assert f.has_ref
    assert 'f' not in function_factory.funcs_inst
    assert f not in function_factory.funcs_user

    function_factory.return_func_ref(ref)
    assert not f.has_ref


def test_recover_ref_funcs(function_factory: FunctionFactoryBase):
    from ceed.function import FuncGroup, CeedFuncRef
    f1 = function_factory.get('ConstFunc')(
        function_factory=function_factory, a=.5, name='f1', duration=1.2)
    f2 = function_factory.get('LinearFunc')(
        function_factory=function_factory, m=.5, b=.2, name='f2', duration=1.2)
    f3 = function_factory.get('ExponentialFunc')(
        function_factory=function_factory, A=.5, B=.2, tau1=1.3, tau2=1.5,
        name='f3', duration=1.2)
    f4 = function_factory.get('CosFunc')(
        function_factory=function_factory, f=.5, A=3.4, th0=7.5, b=12.2,
        name='f4', duration=1.2)
    g = FuncGroup(function_factory=function_factory, name='g')

    function_factory.add_func(f1)
    function_factory.add_func(f2)
    function_factory.add_func(f3)
    function_factory.add_func(f4)
    function_factory.add_func(g)

    g.add_func(function_factory.get_func_ref(name='f1'))
    g.add_func(function_factory.get_func_ref(name='f2'))
    g.add_func(function_factory.get_func_ref(name='f3'))
    g.add_func(function_factory.get_func_ref(name='f4'))

    funcs = function_factory.save_functions()
    assert len(funcs) == 5

    recovered_funcs, name_mapping = function_factory.recover_funcs(funcs)
    assert len(recovered_funcs) == 5
    assert len(name_mapping) == 5

    for f_name in ('f1', 'f2', 'f3', 'f4', 'g'):
        assert f_name in name_mapping
        assert name_mapping[f_name] != f_name
        assert f_name in function_factory.funcs_inst
        assert name_mapping[f_name] in function_factory.funcs_inst

        for name in function_factory.funcs_inst[f_name].get_gui_props():
            original_f = function_factory.funcs_inst[f_name]
            new_f = function_factory.funcs_inst[name_mapping[f_name]]

            if name == 'name':
                assert original_f.name != new_f.name
                assert new_f.name.startswith(original_f.name)
                continue

            assert getattr(original_f, name) == getattr(new_f, name)

    new_g: FuncGroup = function_factory.funcs_inst[name_mapping['g']]
    assert len(new_g.funcs) == 4

    func: CeedFuncRef
    for func, name in zip(new_g.funcs, ('f1', 'f2', 'f3', 'f4')):
        assert isinstance(func, CeedFuncRef)
        assert func.func is function_factory.funcs_inst[name_mapping[name]]


def test_get_funcs_ref(function_factory: FunctionFactoryBase):
    factory = function_factory
    const_cls = factory.get('ConstFunc')

    g1 = FuncGroup(function_factory=factory)

    g2 = FuncGroup(function_factory=factory)
    ref_g2 = function_factory.get_func_ref(func=g2)
    g1.add_func(ref_g2)

    f = const_cls(function_factory=factory, a=.9, duration=2)
    g2.add_func(f)

    f2 = const_cls(function_factory=factory, a=.25, duration=2)
    g1.add_func(f2)

    assert list(g1.get_funcs()) == [g1, g2, f, f2]
    assert list(g1.get_funcs(step_into_ref=False)) == [g1, ref_g2, f2]
    assert g1.duration == 4
    assert g1.duration_min == 4
    assert g1.duration_min_total == 4


def test_call_funcs_ref(function_factory: FunctionFactoryBase):
    factory = function_factory
    const_cls = factory.get('ConstFunc')

    g1 = FuncGroup(function_factory=factory)

    g2 = FuncGroup(function_factory=factory)
    ref_g2 = function_factory.get_func_ref(func=g2)
    g1.add_func(ref_g2)

    f = const_cls(function_factory=factory, a=.9, duration=2)
    g2.add_func(f)

    f2 = const_cls(function_factory=factory, a=.25, duration=2)
    g1.add_func(f2)

    with pytest.raises(TypeError):
        g1.init_func(0)

    with pytest.raises(TypeError):
        ref_g2.init_func(0)

    with pytest.raises(TypeError):
        ref_g2(0)

    with pytest.raises(TypeError):
        g1(0)


def test_can_other_func_be_added_ref(function_factory: FunctionFactoryBase):
    factory = function_factory
    const_cls = factory.get('ConstFunc')

    g1 = FuncGroup(function_factory=factory)

    g2 = FuncGroup(function_factory=factory)
    ref_g2 = function_factory.get_func_ref(func=g2)
    g1.add_func(ref_g2)

    f = const_cls(function_factory=factory, a=.9, duration=2)
    g2.add_func(f)

    f2 = const_cls(function_factory=factory, a=.25, duration=2)
    g1.add_func(f2)

    g3 = FuncGroup(function_factory=factory)
    g1.add_func(g3)
    f4 = const_cls(function_factory=factory, a=.8, duration=2)
    g3.add_func(f4)

    assert g1.can_other_func_be_added(g2)
    assert g1.can_other_func_be_added(ref_g2)
    assert g1.can_other_func_be_added(g3)
    assert not g1.can_other_func_be_added(g1)

    assert not g2.can_other_func_be_added(g1)
    assert g2.can_other_func_be_added(g3)
    assert not g2.can_other_func_be_added(g2)
    assert not g2.can_other_func_be_added(ref_g2)

    assert not g3.can_other_func_be_added(g1)
    assert g3.can_other_func_be_added(g2)
    assert g3.can_other_func_be_added(ref_g2)
    assert not g3.can_other_func_be_added(g3)


def test_expand_ref_funcs(function_factory: FunctionFactoryBase):
    from ceed.function import FuncGroup, CeedFuncRef
    factory = function_factory
    const_cls = factory.get('ConstFunc')

    g1 = FuncGroup(function_factory=factory)

    g2 = FuncGroup(function_factory=factory)
    ref_g2 = function_factory.get_func_ref(func=g2)
    g1.add_func(ref_g2)

    f = const_cls(function_factory=factory, a=.9, duration=2)
    g2.add_func(f)
    f1 = const_cls(function_factory=factory, a=.1, duration=2)
    g2.add_func(f1)

    f2 = const_cls(function_factory=factory, a=.25, duration=2)
    g1.add_func(f2)
    f3 = const_cls(function_factory=factory, a=.75, duration=2)
    ref_f3 = function_factory.get_func_ref(func=f3)
    g1.add_func(ref_f3)

    g3 = FuncGroup(function_factory=factory)
    g1.add_func(g3)
    f4 = const_cls(function_factory=factory, a=.8, duration=2)
    ref_f4 = function_factory.get_func_ref(func=f4)
    g3.add_func(ref_f4)
    f5 = const_cls(function_factory=factory, a=.2, duration=2)
    g3.add_func(f5)

    assert list(g1.get_funcs(step_into_ref=False)) == \
        [g1, ref_g2, f2, ref_f3, g3, ref_f4, f5]
    assert list(g1.get_funcs(step_into_ref=True)) == \
        [g1, g2, f, f1, f2, f3, g3, f4, f5]

    g1_copy = g1.copy_expand_ref()
    # the copy shouldn't have any refs
    assert len(list(g1_copy.get_funcs(step_into_ref=False))) == \
        len(list(g1.get_funcs(step_into_ref=True)))

    for original_f, new_f in zip(
            g1.get_funcs(step_into_ref=True),
            g1_copy.get_funcs(step_into_ref=False)):
        for name in original_f.get_gui_props():
            if name == 'name':
                continue

            assert getattr(original_f, name) == getattr(new_f, name)


def test_t_offset(function_factory: FunctionFactoryBase):
    f1 = function_factory.get('ConstFunc')(
        function_factory=function_factory, a=.5, name='f1', t_offset=3.5,
        duration=1.2)
    f2 = function_factory.get('LinearFunc')(
        function_factory=function_factory, m=.5, b=.2, name='f2', t_offset=3.5,
        duration=1.2)
    f3 = function_factory.get('ExponentialFunc')(
        function_factory=function_factory, A=.5, B=.2, tau1=1.3, tau2=1.5,
        name='f3', t_offset=3.5, duration=1.2)
    f4 = function_factory.get('CosFunc')(
        function_factory=function_factory, f=.5, A=3.4, th0=7.5, b=12.2,
        name='f4', t_offset=3.5, duration=1.2)

    for f in (f1, f2, f3, f4):
        f.init_func(2.3)

    t = 1 + 3.5
    assert math.isclose(f1(3.3), .5)
    assert math.isclose(f2(3.3), t * .5 + .2)
    assert math.isclose(
        f3(3.3), .5 * math.exp(-t / 1.3) + .2 * math.exp(-t / 1.5)
    )
    assert math.isclose(
        f4(3.3),
        3.4 * math.cos(2 * math.pi * .5 * t + 7.5 * math.pi / 180.) + 12.2
    )


def test_copy_funcs(function_factory: FunctionFactoryBase):
    from ceed.function import FuncGroup, CeedFuncRef
    f1 = function_factory.get('ConstFunc')(
        function_factory=function_factory, a=.5, name='f1', duration=1.2)
    f2 = function_factory.get('LinearFunc')(
        function_factory=function_factory, m=.5, b=.2, name='f2', duration=1.2)
    f3 = function_factory.get('ExponentialFunc')(
        function_factory=function_factory, A=.5, B=.2, tau1=1.3, tau2=1.5,
        name='f3', duration=1.2)
    f4 = function_factory.get('CosFunc')(
        function_factory=function_factory, f=.5, A=3.4, th0=7.5, b=12.2,
        name='f4', duration=1.2)
    g = FuncGroup(function_factory=function_factory, name='g')

    function_factory.add_func(f1)
    function_factory.add_func(f2)
    function_factory.add_func(f3)
    function_factory.add_func(f4)

    g.add_func(function_factory.get_func_ref(func=f1))
    g.add_func(function_factory.get_func_ref(func=f2))
    g.add_func(function_factory.get_func_ref(func=f3))
    g.add_func(function_factory.get_func_ref(func=f4))

    for func in (f1, f2, f3, f4):
        func_copy = copy.deepcopy(func)
        assert func is not func_copy
        assert isinstance(func_copy, func.__class__)

        for name in func.get_gui_props():
            if name == 'name':
                continue

            assert getattr(func, name) == getattr(func_copy, name)

    func_copy = copy.deepcopy(g)
    assert len(func_copy.funcs) == 4
    for new_f, original_f in zip(func_copy.funcs, g.funcs):
        assert new_f is not original_f
        assert isinstance(new_f, CeedFuncRef)
        assert isinstance(original_f, CeedFuncRef)
        assert new_f.func is original_f.func


def test_replace_ref_func_with_source_funcs(
        function_factory: FunctionFactoryBase):
    from ceed.function import FuncGroup, CeedFuncRef
    factory = function_factory
    const_cls = factory.get('ConstFunc')

    g1 = FuncGroup(function_factory=factory, name='g1')

    g2 = FuncGroup(function_factory=factory, name='g2')
    ref_g2 = function_factory.get_func_ref(func=g2)
    g1.add_func(ref_g2)

    f = const_cls(function_factory=factory, a=.9, duration=2)
    g2.add_func(f)

    f1 = const_cls(function_factory=factory, a=.1, duration=2, name='f1')
    function_factory.add_func(f1)
    ref_f1 = function_factory.get_func_ref(func=f1)
    g2.add_func(ref_f1)

    f2 = const_cls(function_factory=factory, a=.25, duration=2)
    g1.add_func(f2)

    f3 = const_cls(function_factory=factory, a=.75, duration=2, name='f3')
    function_factory.add_func(f3)
    ref_f3 = function_factory.get_func_ref(func=f3)
    g1.add_func(ref_f3)

    with pytest.raises(ValueError):
        g1.replace_ref_func_with_source(f2)

    with pytest.raises(ValueError):
        g1.replace_ref_func_with_source(ref_f1)

    f3_new, i = g1.replace_ref_func_with_source(ref_f3)

    assert i == 2
    assert ref_f3 not in g1.funcs
    assert f3 not in g1.funcs
    assert not isinstance(f3_new, CeedFuncRef)
    assert isinstance(f3_new, f3.__class__)
    assert g1.funcs[i] is f3_new

    for name in f3.get_gui_props():
        if name == 'name':
            continue
        assert getattr(f3, name) == getattr(f3_new, name)

    g2_new: FuncGroup
    g2_new, i = g1.replace_ref_func_with_source(ref_g2)

    assert i == 0
    assert ref_g2 not in g1.funcs
    assert g2 not in g1.funcs
    assert not isinstance(g2_new, CeedFuncRef)
    assert isinstance(g2_new, FuncGroup)
    assert g1.funcs[i] is g2_new

    assert len(g2_new.funcs) == 2
    assert g2_new.funcs[0] is not g2.funcs[0]
    assert g2_new.funcs[1] is not g2.funcs[1]
    assert isinstance(g2_new.funcs[0], f.__class__)
    assert isinstance(g2_new.funcs[1], ref_f1.__class__)
    assert isinstance(g2_new.funcs[1], CeedFuncRef)

    for name in f.get_gui_props():
        if name == 'name':
            continue
        assert getattr(f, name) == getattr(g2_new.funcs[0], name)
    assert g2_new.funcs[1].func is f1


def test_group_remove_func(function_factory: FunctionFactoryBase):
    factory = function_factory
    const_cls = factory.get('ConstFunc')

    g1 = FuncGroup(function_factory=factory)

    g2 = FuncGroup(function_factory=factory)
    ref_g2 = function_factory.get_func_ref(func=g2)
    g1.add_func(ref_g2)
    f = const_cls(function_factory=factory, a=.9, duration=2)
    g2.add_func(f)
    f1 = const_cls(function_factory=factory, a=.9, duration=2)
    g2.add_func(f1)

    f2 = const_cls(function_factory=factory, a=.25, duration=2)
    g1.add_func(f2)

    assert list(g1.get_funcs(step_into_ref=False)) == [g1, ref_g2, f2]
    assert g1.duration == 6
    assert g1.duration_min == 6
    assert g1.duration_min_total == 6

    g1.remove_func(f2)
    assert list(g1.get_funcs(step_into_ref=False)) == [g1, ref_g2]
    assert g1.duration == 4
    assert g1.duration_min == 4
    assert g1.duration_min_total == 4

    g2.remove_func(f)
    assert list(g1.get_funcs(step_into_ref=False)) == [g1, ref_g2]
    assert g1.duration == 2
    assert g1.duration_min == 2
    assert g1.duration_min_total == 2

    g1.remove_func(ref_g2)
    assert list(g1.get_funcs(step_into_ref=False)) == [g1, ]
    assert g1.duration == 0
    assert g1.duration_min == 0
    assert g1.duration_min_total == 0


def test_internal_plugin_source_in_factory(
        function_factory: FunctionFactoryBase):
    import ceed.function.plugin
    root = pathlib.Path(ceed.function.plugin.__file__)

    assert 'ConstFunc' in function_factory.funcs_cls

    plugin_contents = function_factory.plugin_sources['ceed.function.plugin']
    contents = root.read_bytes()
    assert plugin_contents == [(('__init__.py', ), contents)]


@pytest.mark.parametrize('contents,config', [
    (fake_plugin, 'both'), (fake_plugin_function, 'func'),
    (fake_plugin_distribution, 'dist')
])
def test_external_plugin_source_in_factory(
        function_factory: FunctionFactoryBase, tmp_path, contents, config):
    sys.path.append(str(tmp_path))
    mod = tmp_path / 'my_func_plugin' / '__init__.py'
    try:
        mod.parent.mkdir()
        mod.write_text(contents)
        register_external_functions(function_factory, 'my_func_plugin')

        noise_classes = function_factory.param_noise_factory.noise_classes

        assert 'ConstFunc' in function_factory.funcs_cls
        if config in ('both', 'func'):
            assert 'FakeFunc' in function_factory.funcs_cls
        else:
            assert 'FakeFunc' not in function_factory.funcs_cls

        assert 'UniformNoise' in noise_classes
        assert 'GaussianNoise' in noise_classes
        if config in ('both', 'dist'):
            assert 'FakeNoise' in noise_classes
        else:
            assert 'FakeNoise' not in noise_classes

        assert 'ceed.function.plugin' in function_factory.plugin_sources
        plugin_contents = function_factory.plugin_sources['my_func_plugin']
        assert plugin_contents == [
            (('__init__.py', ),
             contents.replace('\n', os.linesep).encode())
        ]
    finally:
        sys.path.remove(str(tmp_path))
        del sys.modules['my_func_plugin']


def test_external_plugin_single_file(
        function_factory: FunctionFactoryBase, tmp_path):
    sys.path.append(str(tmp_path))
    mod = tmp_path / 'my_bad_plugin.py'
    try:
        mod.write_text(fake_plugin)
        with pytest.raises(ModuleNotFoundError):
            register_external_functions(function_factory, 'my_bad_plugin')
    finally:
        sys.path.remove(str(tmp_path))


def test_noise_factory(function_factory: FunctionFactoryBase):
    UniformNoise = function_factory.param_noise_factory.get_cls('UniformNoise')
    assert function_factory.param_noise_factory.noise_classes['UniformNoise'] \
        is UniformNoise
    uniform = UniformNoise()
    assert hasattr(uniform, 'min_val')
    assert hasattr(uniform, 'max_val')
    assert UniformNoise.__name__ == 'UniformNoise'


@pytest.mark.parametrize('cls_name,props', noise_test_parameters)
def test_noise_sampling(function_factory: FunctionFactoryBase, cls_name, props):
    cls = function_factory.param_noise_factory.get_cls(cls_name)
    assert cls.__name__ == cls_name
    assert function_factory.param_noise_factory.noise_classes[cls_name] is cls

    obj = cls(**props)
    samples = [obj.sample() for _ in range(100)]

    if 'min_val' in props:
        min_val = props['min_val']
        max_val = props['max_val']
        assert min_val <= min(samples) <= max(samples) <= max_val

    # at least two are different (if random is not broken)
    assert len(set(samples)) > 1


@pytest.mark.parametrize('cls_name,props', noise_test_parameters)
def test_noise_seq_sampling(
        function_factory: FunctionFactoryBase, cls_name, props):
    cls = function_factory.param_noise_factory.get_cls(cls_name)
    assert cls.__name__ == cls_name
    assert function_factory.param_noise_factory.noise_classes[cls_name] is cls

    obj = cls(**props)
    samples = obj.sample_seq(100)

    if 'min_val' in props:
        min_val = props['min_val']
        max_val = props['max_val']
        assert min_val <= min(samples) <= max(samples) <= max_val

    # at least two are different (if random is not broken)
    assert len(set(samples)) > 1


@pytest.mark.parametrize('seq', [True, False])
@pytest.mark.parametrize('cls_name,props', noise_test_parameters)
def test_noise_create_from_config(
        function_factory: FunctionFactoryBase, cls_name, props, seq):
    cls = function_factory.param_noise_factory.get_cls(cls_name)

    obj = cls(sample_each_loop=seq, **props)
    obj2 = function_factory.param_noise_factory.make_instance(obj.get_config())

    for key, value in props.items():
        assert getattr(obj, key) == value
        assert getattr(obj2, key) == value
    assert obj.sample_each_loop is seq
    assert obj2.sample_each_loop is seq


@pytest.mark.parametrize('cls_name,props', noise_test_parameters)
def test_func_sampling(function_factory: FunctionFactoryBase, cls_name, props):
    cls = function_factory.param_noise_factory.get_cls(cls_name)

    f: FuncBase = function_factory.get('LinearFunc')(
        function_factory=function_factory)
    b = f.b
    f.noisy_parameters['m'] = cls(**props)

    b_vals = []
    m_vals = []
    for _ in range(100):
        f.resample_parameters()
        b_vals.append(f.b)
        m_vals.append(f.m)

    assert set(b_vals) == {b}
    # at least two are different (if random is not broken)
    assert len(set(m_vals)) > 1


@pytest.mark.parametrize('cls_name,props', noise_test_parameters)
def test_func_seq_sampling(
        function_factory: FunctionFactoryBase, cls_name, props):
    cls = function_factory.param_noise_factory.get_cls(cls_name)

    f: FuncBase = function_factory.get('LinearFunc')(
        function_factory=function_factory, loop=100)
    f.noisy_parameters['m'] = cls(sample_each_loop=True, **props)

    assert 'm' not in f.noisy_parameter_samples
    assert 'b' not in f.noisy_parameter_samples
    f.resample_parameters()
    assert 'm' in f.noisy_parameter_samples
    assert 'b' not in f.noisy_parameter_samples

    # at least two are different (if random is not broken)
    assert len(set(f.noisy_parameter_samples['m'])) > 1


@pytest.mark.parametrize('lock_param', [True, False])
@pytest.mark.parametrize('is_forked', [True, False])
def test_noise_lock(
        function_factory: FunctionFactoryBase, lock_param, is_forked):
    cls = function_factory.param_noise_factory.get_cls('UniformNoise')

    f: FuncBase = function_factory.get('LinearFunc')(
        function_factory=function_factory)
    f.noisy_parameters['m'] = cls()
    f.noisy_parameters['b'] = cls(lock_after_forked=lock_param)
    b = f.b

    b_vals = []
    m_vals = []
    for _ in range(100):
        f.resample_parameters(is_forked=is_forked)
        b_vals.append(f.b)
        m_vals.append(f.m)

    if lock_param:
        if is_forked:
            # should not have changed
            assert set(b_vals) == {b}
        else:
            assert len(set(b_vals)) > 1
    else:
        assert len(set(b_vals)) > 1

    # at least two are different (if random is not broken)
    assert len(set(m_vals)) > 1


@pytest.mark.parametrize('lock_param', [True, False])
def test_noise_seq_lock(function_factory: FunctionFactoryBase, lock_param):
    cls = function_factory.param_noise_factory.get_cls('UniformNoise')

    f: FuncBase = function_factory.get('LinearFunc')(
        function_factory=function_factory, loop=100)
    f.noisy_parameters['b'] = cls(
        lock_after_forked=lock_param, sample_each_loop=True)

    assert 'b' not in f.noisy_parameter_samples
    f.resample_parameters(is_forked=False)
    b_vals = f.noisy_parameter_samples['b']
    f.resample_parameters(is_forked=True)
    b_vals2 = f.noisy_parameter_samples['b']

    if lock_param:
        assert b_vals is b_vals2
        assert b_vals == b_vals2
        assert len(b_vals) == 100
    else:
        assert b_vals is not b_vals2
        assert b_vals != b_vals2
        assert len(b_vals) == len(b_vals2)
        assert len(b_vals) == 100


def test_noise_ref_lock(function_factory: FunctionFactoryBase):
    cls = function_factory.param_noise_factory.get_cls('UniformNoise')

    f: FuncBase = function_factory.get('LinearFunc')(
        function_factory=function_factory)
    function_factory.add_func(f)

    f.noisy_parameters['m'] = cls()
    f.noisy_parameters['b'] = cls(lock_after_forked=True)

    ref2 = function_factory.get_func_ref(func=f)
    f.resample_parameters()

    m_vals = set()
    f2: FuncBase = ref2.copy_expand_ref()
    for _ in range(100):
        f2.resample_parameters(is_forked=True)
        m_vals.add(f2.m)
        assert f2.b == f.b

    assert len(set(m_vals)) > 1


@pytest.mark.parametrize('cls_name,props', noise_test_parameters)
def test_copy_func_noise(
        function_factory: FunctionFactoryBase, cls_name, props):
    cls = function_factory.param_noise_factory.get_cls(cls_name)

    f: FuncBase = function_factory.get('LinearFunc')(
        function_factory=function_factory)
    f.noisy_parameters['m'] = cls(**props)
    f.noisy_parameters['b'] = cls(
        **props, lock_after_forked=True, sample_each_loop=True)

    f2 = copy.deepcopy(f)

    for key, value in props.items():
        assert getattr(f.noisy_parameters['m'], key) == value
        assert getattr(f.noisy_parameters['b'], key) == value
        assert getattr(f2.noisy_parameters['m'], key) == value
        assert getattr(f2.noisy_parameters['b'], key) == value

    assert not f.noisy_parameters['m'].lock_after_forked
    assert f.noisy_parameters['b'].lock_after_forked
    assert not f.noisy_parameters['m'].sample_each_loop
    assert f.noisy_parameters['b'].sample_each_loop
    assert not f2.noisy_parameters['m'].lock_after_forked
    assert f2.noisy_parameters['b'].lock_after_forked
    assert not f2.noisy_parameters['m'].sample_each_loop
    assert f2.noisy_parameters['b'].sample_each_loop


def test_copy_func_noise_seq(function_factory: FunctionFactoryBase):
    cls = function_factory.param_noise_factory.get_cls('UniformNoise')

    f: FuncBase = function_factory.get('LinearFunc')(
        function_factory=function_factory, loop=100)
    f.noisy_parameters['m'] = cls()
    f.noisy_parameters['b'] = cls(sample_each_loop=True)

    f.resample_parameters()
    assert 'm' not in f.noisy_parameter_samples
    original = f.noisy_parameter_samples['b']
    b_vals = list(original)

    f2 = copy.deepcopy(f)

    assert 'm' not in f.noisy_parameter_samples
    assert 'm' not in f2.noisy_parameter_samples
    assert f.noisy_parameter_samples['b'] is original
    assert f.noisy_parameter_samples['b'] == b_vals
    assert f2.noisy_parameter_samples['b'] is not original
    assert f2.noisy_parameter_samples['b'] == b_vals


@pytest.mark.parametrize('rate', [60., 120., 100.])
@pytest.mark.parametrize('duration', [
    (.5, .5, .5), (.51, .5, .49), (.1, .1, .1), (.11, .33, .59),
    (31 / 60, 5 / 12, 1441 / 720)])
def test_func_float_duration(
        function_factory: FunctionFactoryBase, rate, duration):
    ConstFunc = function_factory.get('ConstFunc')
    root = FuncGroup(function_factory=function_factory, loop=5)
    child_a = ConstFunc(
        function_factory=function_factory, loop=4, duration=duration[0], a=1)
    child_b = ConstFunc(
        function_factory=function_factory, loop=3, duration=duration[1], a=2)
    child_c = ConstFunc(
        function_factory=function_factory, loop=5, duration=duration[2], a=3)

    root.add_func(child_a)
    root.add_func(child_b)
    root.add_func(child_c)

    values = get_function_values(root, rate)

    expected = int(
        rate * 5 * (4 * duration[0] + 3 * duration[1] + 5 * duration[2]))
    assert expected - 1 <= len(values) <= expected + 1

    counts = collapse_list_to_counts(values)
    loops = [4, 3, 5]
    a = [1, 2, 3]
    for i in range(5 * 3):
        a_val, count = counts[i]
        assert a_val == a[i % 3]
        assert count - 1 <= round(loops[i % 3] * duration[i % 3] * rate) \
            <= count + 1


def test_func_float_duration_epsilon(function_factory: FunctionFactoryBase):
    ConstFunc = function_factory.get('ConstFunc')
    root = FuncGroup(function_factory=function_factory, loop=5)
    child_a = ConstFunc(function_factory=function_factory, duration=1., a=1)
    child_b = ConstFunc(function_factory=function_factory, duration=3 * .1, a=2)
    child_c = ConstFunc(function_factory=function_factory, duration=3 * .1, a=3)
    child_d = ConstFunc(function_factory=function_factory, duration=2., a=4)

    root.add_func(child_a)
    root.add_func(child_b)
    root.add_func(child_c)
    root.add_func(child_d)

    values = get_function_values(root, 60.0)
    expected = int(60. * 5 * 3.6)
    assert expected - 1 <= len(values) <= expected + 1

    counts = collapse_list_to_counts(values)
    duration = [1., 3 * .1, 3 * .1, 2.]
    a = [1, 2, 3, 4]
    for i in range(5 * 4):
        a_val, count = counts[i]
        assert a_val == a[i % 4]
        assert count - 1 <= round(duration[i % 4] * 60) <= count + 1


@pytest.mark.parametrize('name,props,min_val,max_val,legal_vals', [
    ('DiscreteNoise', {'start_value': .1, 'step': .1, 'num_values': 4},
     .09, .41, {.1 + i * .1 for i in range(4)}),
    ('DiscreteListNoise', {'csv_list': '.1, .2,.3  ,.4  ,,'},
     .09, .41, {.1, .2, .3, .4}),
])
@pytest.mark.parametrize('with_replacement', [True, False])
def test_discrete_without_replacement(
        function_factory: FunctionFactoryBase, with_replacement, name, props,
        min_val, max_val, legal_vals):
    cls = function_factory.param_noise_factory.get_cls(name)
    obj = cls(with_replacement=with_replacement, **props)

    for _ in range(100):
        assert min_val < obj.sample() < max_val

    vals = obj.sample_seq(3)
    assert not (set(vals) - legal_vals)
    assert legal_vals - set(vals)
    assert len(vals) == 3
    if not with_replacement:
        assert len(set(vals)) == 3

    if not with_replacement:
        obj.sample_seq(4)
        with pytest.raises(ValueError):
            obj.sample_seq(5)
        with pytest.raises(ValueError):
            obj.sample_seq(10)
        return

    vals = obj.sample_seq(100)
    assert not (set(vals) - legal_vals)
    assert len(vals) == 100
    assert len(set(vals)) > 1


@pytest.mark.parametrize('name,props', [
    ('DiscreteNoise', {'num_values': 0}),
    ('DiscreteListNoise', {'csv_list': ''}),
    ('DiscreteListNoise', {'csv_list': ','}),
])
@pytest.mark.parametrize('with_replacement', [True, False])
def test_discrete_empty(
        function_factory: FunctionFactoryBase, with_replacement, name, props):
    cls = function_factory.param_noise_factory.get_cls(name)
    obj = cls(with_replacement=with_replacement, **props)

    with pytest.raises(ValueError):
        obj.sample()
    with pytest.raises(ValueError):
        obj.sample_seq(1)


def test_function_tree(function_factory: FunctionFactoryBase):
    factory = function_factory
    const_cls = factory.get('ConstFunc')

    g1 = FuncGroup(function_factory=factory, loop=2)

    g2 = FuncGroup(function_factory=factory, loop=2)
    g1.add_func(g2)
    f = const_cls(function_factory=factory, a=.9, duration=2, loop=2)
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

    def assert_counts(
            g1_count, g1_tree_count, g2_count, g2_tree_count, f_count,
            f_tree_count):
        assert g1.loop_count == g1_count
        assert g1.loop_tree_count == g1_tree_count
        assert g2.loop_count == g2_count
        assert g2.loop_tree_count == g2_tree_count
        assert f.loop_count == f_count
        assert f.loop_tree_count == f_tree_count

    g1.init_func_tree()
    g1.init_func(0)
    assert_counts(0, 0, 0, 0, 0, 0)

    g1(1)
    assert_counts(0, 0, 0, 0, 0, 0)
    g1(3)
    assert_counts(0, 0, 0, 0, 1, 1)
    g1(5)
    assert_counts(0, 0, 0, 0, 2, 2)

    g1(7)
    assert_counts(0, 0, 1, 1, 0, 2)
    g1(9)
    assert_counts(0, 0, 1, 1, 1, 3)
    g1(11)
    assert_counts(0, 0, 1, 1, 2, 4)

    g1(13)
    assert_counts(0, 0, 2, 2, 2, 4)

    g1(21)
    assert_counts(1, 1, 0, 2, 0, 4)
    g1(23)
    assert_counts(1, 1, 0, 2, 1, 5)
    g1(25)
    assert_counts(1, 1, 0, 2, 2, 6)

    g1(27)
    assert_counts(1, 1, 1, 3, 0, 6)
    g1(29)
    assert_counts(1, 1, 1, 3, 1, 7)
    g1(31)
    assert_counts(1, 1, 1, 3, 2, 8)

    g1(33)
    assert_counts(1, 1, 2, 4, 2, 8)

    with pytest.raises(FuncDoneException):
        g1(41)

    assert_counts(2, 2, 2, 4, 2, 8)


def test_call_func_loop_done(function_factory: FunctionFactoryBase):
    from ceed.function.plugin import ConstFunc
    factory = function_factory

    f = ConstFunc(function_factory=factory, duration=2, loop=3)
    f.init_func_tree()
    f.init_func(0)
    assert f.loop_count == 0
    assert f.loop_tree_count == 0
    f(1)
    assert f.loop_count == 0
    assert f.loop_tree_count == 0

    f(3)
    assert f.loop_count == 1
    assert f.loop_tree_count == 1

    f(5)
    assert f.loop_count == 2
    assert f.loop_tree_count == 2

    with pytest.raises(FuncDoneException):
        f(7)
    assert f.loop_count == 3
    assert f.loop_tree_count == 3
    with pytest.raises(FuncDoneException):
        f(8)

    with pytest.raises(ValueError):
        f.tick_loop(0)

    with pytest.raises(ValueError):
        f.tick_loop(8)


def test_add_func_unique_built_in_name(function_factory: FunctionFactoryBase):
    assert not function_factory.funcs_user

    const_cls = function_factory.get('ConstFunc')
    f = const_cls(function_factory=function_factory)
    orig_name = f.name
    n = len(function_factory.funcs_inst)
    assert orig_name in function_factory.funcs_inst

    function_factory.add_func(f)
    assert f in function_factory.funcs_user
    assert len(function_factory.funcs_inst) == n + 1
    assert function_factory.funcs_inst[f.name] is f
    assert orig_name in function_factory.funcs_inst
    assert f.name != orig_name


def test_funcs_add(function_factory: FunctionFactoryBase):
    for func_cls in func_classes:
        func = func_cls(function_factory=function_factory, show_in_gui=False)
        func.create_func()

        ceed_func = func.func
        func.assert_init()

        assert func.name == ceed_func.name
        assert func.duration == ceed_func.duration
        assert func.loop == ceed_func.loop
        if hasattr(ceed_func, 't_offset'):
            assert func.t_offset == ceed_func.t_offset
        assert math.isclose(
            func.duration_min_total, ceed_func.duration_min_total)
        assert func.timebase[0] == ceed_func.timebase_numerator
        assert func.timebase[1] == ceed_func.timebase_denominator
        assert func.timebase[0] / func.timebase[1] == \
            float(ceed_func.timebase)

        func.assert_func_values()
