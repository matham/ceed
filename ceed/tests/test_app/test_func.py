import pytest
from contextlib import contextmanager
import os
import sys
import math
from typing import Type, List, Union

from ceed.tests.ceed_app import CeedTestApp
from ceed.tests.test_app import replace_text, touch_widget, \
    select_spinner_value as select_spinner_func, escape
from ceed.function import FuncBase, FuncGroup, FunctionFactoryBase, \
    CeedFuncRef, register_external_functions
from .examples.funcs import GroupFunction, \
    GroupFunctionF5, Function, LinearFunctionF1, LinearFunctionF2, \
    GroupFunctionF4, CosFunctionF4, ExponentialFunctionF3, ConstFunctionF1, \
    ExponentialFunctionF1, CosFunctionF1, GroupFunctionF1, \
    func_classes_group, func_classes_linear
from .examples.funcs import fake_plugin_function, \
    fake_plugin_distribution, fake_plugin, noise_test_parameters

pytestmark = pytest.mark.ceed_app


async def assert_set_params_in_gui(
        func_app: CeedTestApp, func: Function, settings=None):
    opened_settings = settings is None
    if opened_settings:
        settings = await open_func_settings(func_app, func.func)

    name_map = func.func.get_prop_pretty_name()
    for name, prop_type in func.editor_props.items():

        # find the label for this property we want to edit
        label = func_app.resolve_widget(settings).down(
            test_name='func prop label', text=name_map.get(name, name))()
        edit_children = label.parent.children
        # the widget right after label is the widget used to edit the prop
        editor = edit_children[edit_children.index(label) - 1]

        # change the value in the GUI
        assert prop_type in (float, int)
        val = getattr(func, name)
        await replace_text(func_app, editor, str(val))
        assert math.isclose(getattr(func.func, name), val)

    if opened_settings:
        await escape(func_app)
    return settings


async def assert_func_params_in_gui(
        func_app: CeedTestApp, func: Function, settings=None):
    opened_settings = settings is None
    if opened_settings:
        settings = await open_func_settings(func_app, func.func)

    name_map = func.func.get_prop_pretty_name()
    for name, prop_type in func.editor_props.items():

        # find the label for this property we want to edit
        label = func_app.resolve_widget(settings).down(
            test_name='func prop label', text=name_map.get(name, name))()
        edit_children = label.parent.children
        # the widget right after label is the widget used to edit the prop
        editor = edit_children[edit_children.index(label) - 1]

        # check the value in the GUI
        assert prop_type in (float, int)
        assert math.isclose(prop_type(editor.text), getattr(func, name))
        assert math.isclose(getattr(func, name), getattr(func.func, name))

    if opened_settings:
        await escape(func_app)
    return settings


def assert_funcs_same(
        func1: Union[FuncBase, FuncGroup], func2: Union[FuncBase, FuncGroup],
        compare_name=False):
    assert type(func1) == type(func2)

    keys = set(func1.get_state(recurse=False).keys()) | \
        set(func2.get_state(recurse=False).keys())
    assert 'name' in keys
    if not compare_name:
        keys.remove('name')
    keys.remove('cls')

    if isinstance(func1, FuncGroup):
        keys.remove('funcs')
        assert len(func1.funcs) == len(func2.funcs)

    for key in keys:
        assert getattr(func1, key) == getattr(func2, key)


async def replace_last_ref_with_original_func(
        func_app: CeedTestApp,
        funcs: List[Union[CeedFuncRef, FuncBase, FuncGroup]], name: str):
    from ceed.function import CeedFuncRef

    start_funcs = funcs[:]
    ref_func = funcs[-1]
    # it should be a ref to start with
    assert isinstance(ref_func, CeedFuncRef)
    # make sure the class name matches - we added the right class
    assert ref_func.func.name == name

    # the label of the new sub-func
    sub_func_widget = ref_func.display
    name_w = func_app.resolve_widget(sub_func_widget).down(
        test_name='func label')()
    assert name_w.text == name
    # replace the ref with a copy of the func
    ref_btn = func_app.resolve_widget(sub_func_widget).down(
        test_name='func settings open')()
    await touch_widget(func_app, ref_btn)

    # should now have replaced the ref with a copy of the original
    assert ref_func not in funcs
    assert len(funcs) == len(start_funcs)

    new_func = funcs[-1]
    assert ref_func is not new_func
    assert funcs[:-1] == start_funcs[:-1]
    # it should not be a ref anymore
    assert not isinstance(new_func, CeedFuncRef)

    assert_funcs_same(ref_func.func, new_func)

    return new_func


async def open_func_settings(func_app: CeedTestApp, func: FuncBase):
    settings_btn = func_app.resolve_widget(func.display).down(
        test_name='func settings open')()
    await touch_widget(func_app, settings_btn)

    return func_app.resolve_widget().down(test_name='func settings')()


async def test_funcs_add(func_app: CeedTestApp):
    for func_cls in func_classes_linear[:2] + func_classes_group[:2]:
        func = func_cls(app=func_app)
        await func_app.wait_clock_frames(2)

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


async def test_funcs_params(func_app: CeedTestApp):
    from kivy.uix.textinput import TextInput
    funcs = []
    for func_cls in func_classes_linear[:2] + func_classes_group[:2]:
        func = func_cls(app=func_app)

        funcs.append(func)
        # don't keep more than two function so the list is not too long
        if len(funcs) >= 3:
            oldest_func = funcs.pop(0)
            remove_btn = func_app.resolve_widget().down(
                text=oldest_func.name).family_up(test_name='del_btn_func')()
            await touch_widget(func_app, remove_btn)

        await func_app.wait_clock_frames(2)

        settings = await open_func_settings(func_app, func.func)
        name_map = func.func.get_prop_pretty_name()
        for name, prop_type in func.editor_props.items():

            # find the label for this property we want to edit
            label = func_app.resolve_widget(settings).down(
                test_name='func prop label', text=name_map.get(name, name))()
            edit_children = label.parent.children
            # the widget right after label is the widget used to edit the prop
            editor = edit_children[edit_children.index(label) - 1]
            assert isinstance(editor, TextInput)

            assert math.isclose(
                prop_type(editor.text), getattr(func.func, name))

            # change the value in the GUI
            assert prop_type in (float, int)
            new_val = prop_type(getattr(func.func, name) + 2.2)
            await replace_text(func_app, editor, str(new_val))
            assert math.isclose(getattr(func.func, name), new_val)

            # change the value in the function
            new_val = prop_type(getattr(func.func, name) + 4.2)
            setattr(func.func, name, new_val)
            assert editor.text == str(new_val)

        name = func_app.resolve_widget(settings).down(test_name='func name')()
        assert not name.disabled, "root functions can be renamed"
        assert name.text == func.name
        desc = func_app.resolve_widget(settings).down(
            test_name='func description')()
        assert desc.text == func.func.description

        # close the settings widget
        await escape(func_app)


async def assert_add_func_to_group(
        func_app: CeedTestApp, func: GroupFunction,
        sub_func_cls: Type[Function], add=None, spinner=None):
    # group function should have been selected previously
    if add is None:
        add = func_app.resolve_widget().down(test_name='func add')()
    if spinner is None:
        spinner = func_app.resolve_widget().down(test_name='func spinner')()

    sub_func = sub_func_cls(app=func_app, show_in_gui=False)
    # select the function class to add
    await select_spinner_func(func_app, sub_func.cls_name, spinner)

    original_funcs = func.func.funcs[:]
    original_n = len(original_funcs)
    # add the func to the class
    await touch_widget(func_app, add)
    assert len(func.func.funcs) == original_n + 1
    assert func.func.funcs[-1] not in original_funcs

    sub_func.func = await replace_last_ref_with_original_func(
        func_app, func.func.funcs, sub_func.cls_name)

    name = func_app.resolve_widget(
        sub_func.func.display).down(text=sub_func.cls_name)()
    assert name.text == sub_func.cls_name
    # locate and open its settings
    settings = await open_func_settings(func_app, sub_func.func)
    await assert_set_params_in_gui(func_app, sub_func, settings)

    name = func_app.resolve_widget(settings).down(test_name='func name')()
    assert name.disabled, "sub-functions cannot be renamed"
    assert name.text == sub_func.cls_name
    desc = func_app.resolve_widget(settings).down(
        test_name='func description')()
    assert desc.text == sub_func.func.description

    # close the settings widget
    await escape(func_app)


async def test_gui_add_funcs(func_app: CeedTestApp):
    funcs = []
    spinner = func_app.resolve_widget().down(test_name='func spinner')()
    add = func_app.resolve_widget().down(test_name='func add')()
    for func_cls in func_classes_linear[:2] + func_classes_group[:2]:
        func = func_cls(app=func_app, show_in_gui=False)
        funcs.append(func)

        # don't keep more than two function so the list is not too long
        if len(funcs) >= 3:
            oldest_func = funcs.pop(0)
            remove_btn = func_app.resolve_widget().down(
                text=oldest_func.name).family_up(test_name='del_btn_func')()
            await touch_widget(func_app, remove_btn)

        # select the function class in the spinner to add this function class
        assert func.cls_name in spinner.values
        await select_spinner_func(func_app, func.cls_name, spinner)

        # add the function
        await touch_widget(func_app, add)
        func.func = func_app.app.function_factory.funcs_user[-1]

        # show the settings for the function
        widget = func.func.display
        settings = await open_func_settings(func_app, func.func)

        name = func_app.resolve_widget(settings).down(test_name='func name')()
        assert not name.disabled, "root functions can be renamed"
        name_label = func_app.resolve_widget(widget).down(
            test_name='func label')()
        original_name = name.text
        assert original_name == name_label.text
        assert original_name in func_app.app.function_factory.funcs_inst
        assert func.name not in func_app.app.function_factory.funcs_inst
        await replace_text(func_app, name, func.name)
        assert name.text == func.name
        assert name_label.text == func.name
        assert original_name not in func_app.app.function_factory.funcs_inst
        assert func.name in func_app.app.function_factory.funcs_inst

        await assert_set_params_in_gui(func_app, func, settings)

        if not issubclass(func_cls, GroupFunction):
            desc = func_app.resolve_widget(settings).down(
                test_name='func description')()
            assert desc.text == func.func.description

        # close the settings widget
        await escape(func_app)

        # for group func, test all the sub functions
        if issubclass(func_cls, GroupFunction):
            for sub_func_cls in func_cls.wrapper_classes:
                # select the group so the function will be added to it
                if not func.func.display.selected:
                    await touch_widget(func_app, name_label)

                await assert_add_func_to_group(
                    func_app, func, sub_func_cls, add, spinner)
                await func_app.wait_clock_frames(2)

                # deselect the group so the next group will not be added to it
                if func.func.display.selected:
                    await touch_widget(func_app, name_label)
                await func_app.wait_clock_frames(2)


async def group_recursive_add(func_app: CeedTestApp, add_func):
    from ceed.function.func_widgets import FuncWidgetGroup
    from ceed.function import FuncGroup, CeedFuncRef
    spinner = func_app.resolve_widget().down(test_name='func spinner')()

    assert 'Group' in spinner.values
    assert 'Linear' in spinner.values
    await select_spinner_func(func_app, 'Group', spinner)

    # first add a new function group
    assert not func_app.app.function_factory.funcs_user
    await add_func(None)
    assert len(func_app.app.function_factory.funcs_user) == 1
    g1: FuncGroup = func_app.app.function_factory.funcs_user[-1]
    g1_widget = g1.display
    assert isinstance(g1, FuncGroup)
    assert isinstance(g1_widget, FuncWidgetGroup)

    # now select the widget
    g1.name = 'deep func'
    name_label = func_app.resolve_widget(g1_widget).down(
        test_name='func label')()
    assert name_label.text == 'deep func'
    await touch_widget(func_app, name_label)

    # add a group to the group and replace the ref
    await select_spinner_func(func_app, 'Group', spinner)
    await add_func(g1)
    g2: FuncGroup = await replace_last_ref_with_original_func(
        func_app, g1.funcs, 'Group')
    g2_widget = g2.display

    # add linear to the inner group
    name_label = func_app.resolve_widget(g2_widget).down(
        test_name='func label')()
    assert name_label.text == 'Group'
    await touch_widget(func_app, name_label)
    await select_spinner_func(func_app, 'Linear', spinner)
    await func_app.wait_clock_frames(1)
    await add_func(g2)
    f2: FuncBase = await replace_last_ref_with_original_func(
        func_app, g2.funcs, 'Linear')

    # add linear to the outer group
    name_label = func_app.resolve_widget(g1_widget).down(
        test_name='func label')()
    await touch_widget(func_app, name_label)
    await add_func(g1)
    f1: FuncBase = await replace_last_ref_with_original_func(
        func_app, g1.funcs, 'Linear')

    settings = []
    for wrapper_cls, f in (
            (LinearFunctionF1, f2), (LinearFunctionF2, f1),
            (GroupFunctionF5, g1), (GroupFunctionF4, g2)):
        wrapper = wrapper_cls(app=func_app, show_in_gui=False)
        wrapper.func = f

        setting = await assert_set_params_in_gui(func_app, wrapper)
        assert setting not in settings
        settings.append(setting)


async def test_group_recursive_add(func_app: CeedTestApp):
    # this test adds the functions by pressing the add button
    add = func_app.resolve_widget().down(test_name='func add')()

    async def add_func(func):
        await touch_widget(func_app, add)
    await group_recursive_add(func_app, add_func)


async def test_group_recursive_drag(func_app: CeedTestApp):
    # this test adds the functions by dragging them from the function list
    start = func_app.resolve_widget().down(test_name='func list drag')()

    # add group to the function factory
    async def add_g1():
        async for _ in func_app.do_touch_drag(
                widget=start, target_widget=func_app.app.funcs_container):
            pass

    # add functions to an existing function
    async def add_others(func):
        async for _ in func_app.do_touch_drag_follow(
                widget=start, target_widget=func.display.children_container,
                target_widget_loc=('center_x', 'y')):
            pass

    async def add_func(func):
        if func is None:
            await add_g1()
        else:
            await add_others(func)

    await group_recursive_add(func_app, add_func)


async def test_duplicate_func_globally(func_app: CeedTestApp):
    from ceed.function.plugin import CosFunc
    spinner = func_app.resolve_widget().down(test_name='func spinner')()
    add = func_app.resolve_widget().down(test_name='func add')()
    # select the function class to add
    await select_spinner_func(func_app, 'Cos', spinner)
    # add it
    await touch_widget(func_app, add)

    # update its params
    wrapper_func = CosFunctionF4(app=func_app, show_in_gui=False)
    wrapper_func.func = func_app.app.function_factory.funcs_user[-1]
    assert isinstance(wrapper_func.func, CosFunc)
    await assert_set_params_in_gui(func_app, wrapper_func)

    drag_btn = func_app.resolve_widget(wrapper_func.func.display).down(
        test_name='func drag btn')()
    async for _ in func_app.do_touch_drag(
            widget=drag_btn, target_widget=func_app.app.funcs_container):
        pass

    wrapper_func2 = CosFunctionF4(app=func_app, show_in_gui=False)
    wrapper_func2.func = func_app.app.function_factory.funcs_user[-1]
    assert wrapper_func.func is not wrapper_func2.func
    assert isinstance(wrapper_func2.func, CosFunc)
    await assert_func_params_in_gui(func_app, wrapper_func2)


async def global_ref_func_and_replace(func_app: CeedTestApp, add_func):
    from ceed.function.plugin import ExponentialFunc
    from ceed.function import CeedFuncRef
    spinner = func_app.resolve_widget().down(test_name='func spinner')()
    add = func_app.resolve_widget().down(test_name='func add')()

    # add exp function to factory
    await select_spinner_func(func_app, 'Exp', spinner)
    await touch_widget(func_app, add)

    exp_wrapper = ExponentialFunctionF3(app=func_app, show_in_gui=False)
    exp_wrapper.func = func_app.app.function_factory.funcs_user[-1]
    assert isinstance(exp_wrapper.func, ExponentialFunc)
    await assert_set_params_in_gui(func_app, exp_wrapper)
    assert exp_wrapper.func.name in spinner.values

    # add group function to factory
    await select_spinner_func(func_app, 'Group', spinner)
    await touch_widget(func_app, add)
    g: FuncGroup = func_app.app.function_factory.funcs_user[-1]
    assert isinstance(g, FuncGroup)
    assert not g.funcs
    assert g.name in spinner.values

    # add a ref to the exp function to the group
    await add_func(exp_wrapper.func, g)

    assert len(g.funcs) == 1
    exp_ref = g.funcs[-1]
    assert isinstance(exp_ref, CeedFuncRef)
    assert exp_ref.func is exp_wrapper.func
    label = func_app.resolve_widget(exp_ref.display).down(
        test_name='func label')()
    assert label.text == exp_wrapper.func.name

    # replace it with a copy of the original
    ref_btn = func_app.resolve_widget(exp_ref.display).down(
        test_name='func settings open')()
    await touch_widget(func_app, ref_btn)

    assert len(g.funcs) == 1
    exp_wrapper2 = ExponentialFunctionF3(app=func_app, show_in_gui=False)
    exp_wrapper2.func = g.funcs[-1]
    assert isinstance(exp_wrapper2.func, ExponentialFunc)
    assert exp_wrapper2.func is not exp_wrapper.func
    label = func_app.resolve_widget(exp_wrapper2.func.display).down(
        test_name='func label')()
    assert label.text == exp_wrapper2.cls_name
    await assert_func_params_in_gui(func_app, exp_wrapper2)


async def test_drag_global_ref_func_and_replace(func_app: CeedTestApp):
    # this version drags the global function into the group
    async def add_func(src_func, target_func):
        # drag a copy of exp into the group
        drag_btn = func_app.resolve_widget(src_func.display).down(
            test_name='func drag btn')()
        async for _ in func_app.do_touch_drag(
                widget=drag_btn,
                target_widget=target_func.display.children_container):
            pass

    await global_ref_func_and_replace(func_app, add_func)


async def test_add_global_ref_func_and_replace(func_app: CeedTestApp):
    # this version adds the global function into the group with add button
    spinner = func_app.resolve_widget().down(test_name='func spinner')()
    add = func_app.resolve_widget().down(test_name='func add')()

    async def add_func(src_func, target_func):
        # spinner needs to show the function to be added
        await select_spinner_func(func_app, src_func.name, spinner)
        # select the group widget
        g_label = func_app.resolve_widget(target_func.display).down(
            test_name='func label')()
        await touch_widget(func_app, g_label)
        # now add func to group with add button
        await touch_widget(func_app, add)

    await global_ref_func_and_replace(func_app, add_func)


@contextmanager
def add_to_path(tmp_path, *args):
    sys.path.append(str(tmp_path))
    mod = tmp_path / 'my_gui_func_plugin' / '__init__.py'
    try:
        mod.parent.mkdir()
        mod.write_text(fake_plugin)
        yield None
    finally:
        sys.path.remove(str(tmp_path))
        if 'my_gui_func_plugin' in sys.modules:
            del sys.modules['my_gui_func_plugin']


@pytest.mark.parametrize(
    "ceed_app",
    [{'yaml_config': {'external_function_plugin_package': 'my_gui_func_plugin'},
      'app_context': add_to_path}, ],
    indirect=True
)
async def test_external_plugin_named_package(func_app: CeedTestApp, tmp_path):
    function_factory = func_app.app.function_factory
    noise_classes = function_factory.param_noise_factory.noise_classes

    assert 'FakeFunc' in function_factory.funcs_cls
    assert 'FakeNoise' in noise_classes
