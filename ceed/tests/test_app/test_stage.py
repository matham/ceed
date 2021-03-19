import sys
import math
from contextlib import contextmanager
from math import isclose
import numpy as np
import pytest

import ceed
from .examples.stages import create_test_stages, make_stage, StageWrapper, \
    stage_classes, assert_stages_same
from typing import Type, List, Union
from ceed.tests.ceed_app import CeedTestApp
from ceed.tests.test_app import replace_text, touch_widget, escape
from ceed.stage import CeedStage, CeedStageRef
from ceed.function import CeedFuncRef, FuncBase, FuncGroup
from ceed.shape import CeedShape, CeedShapeGroup
from .examples.shapes import assert_add_three_groups, CircleShapeP1
from .examples.funcs import create_funcs, GroupFunction
from .examples.stages import fake_plugin_stage, SerialAllStage
from .examples.experiment import wait_stage_experiment_started, \
    wait_experiment_done, measure_fps
from .test_func import assert_func_params_in_gui, \
    replace_last_ref_with_original_func, assert_funcs_same

pytestmark = pytest.mark.ceed_app


async def assert_set_params_in_gui(
        stage_app: CeedTestApp, stage: StageWrapper, settings=None,
        check_name=False):
    opened_settings = settings is None
    if opened_settings:
        settings = await open_stage_settings(stage_app, stage.stage)

    if check_name:
        name = stage_app.resolve_widget(settings).down(
            test_name='stage name')()
        assert name.text != stage.name
        assert name.text == stage.stage.name
        await replace_text(stage_app, name, stage.name)
        assert name.text == stage.name
        assert name.text == stage.stage.name

    # verify colors
    for color in ('r', 'g', 'b'):
        widget = stage_app.resolve_widget(settings).down(
            test_name='stage color {}'.format(color))()
        prop = 'color_{}'.format(color)
        # the stage values should always match the GUI values
        assert getattr(stage.stage, prop) == (widget.state == 'down')
        # if the wrapper need to change the value, do it
        if getattr(stage, prop) != getattr(stage.stage, prop):
            await touch_widget(stage_app, widget)

        # make sure it was changed
        assert getattr(stage.stage, prop) == (widget.state == 'down')
        assert getattr(stage, prop) == getattr(stage.stage, prop)

    # parallel vs serial
    serial = stage_app.resolve_widget(settings).down(
        test_name='stage serial')()
    parallel = stage_app.resolve_widget(settings).down(
        test_name='stage parallel')()
    assert (stage.stage.order == 'serial') == (serial.state == 'down') and \
        (stage.stage.order == 'parallel') == (parallel.state == 'down')

    # set the GUI to the correct value
    if stage.order == 'parallel' and parallel.state != 'down':
        await touch_widget(stage_app, parallel)
    elif stage.order == 'serial' and serial.state != 'down':
        await touch_widget(stage_app, serial)
    assert (stage.stage.order == 'serial') == (serial.state == 'down') and \
        (stage.stage.order == 'parallel') == (parallel.state == 'down')
    assert (stage.order == 'serial') == (serial.state == 'down') and \
        (stage.order == 'parallel') == (parallel.state == 'down')

    # complete_on all vs any
    all_w = stage_app.resolve_widget(settings).down(
        test_name='stage finish all')()
    any_w = stage_app.resolve_widget(settings).down(
        test_name='stage finish any')()
    assert (stage.stage.complete_on == 'all') == (all_w.state == 'down') and \
        (stage.stage.complete_on == 'any') == (any_w.state == 'down')

    # set the GUI to the correct value
    if stage.complete_on == 'all' and all_w.state != 'down':
        await touch_widget(stage_app, all_w)
    elif stage.complete_on == 'any' and any_w.state != 'down':
        await touch_widget(stage_app, any_w)

    assert (stage.stage.complete_on == 'all') == (all_w.state == 'down') and \
        (stage.stage.complete_on == 'any') == (any_w.state == 'down')
    assert (stage.complete_on == 'all') == (all_w.state == 'down') and \
        (stage.complete_on == 'any') == (any_w.state == 'down')

    if opened_settings:
        await escape(stage_app)
    return settings


async def assert_stage_params_in_gui(
        stage_app: CeedTestApp, stage: StageWrapper, settings=None,
        check_name=False):
    opened_settings = settings is None
    if opened_settings:
        settings = await open_stage_settings(stage_app, stage.stage)

    if check_name:
        name = stage_app.resolve_widget(settings).down(
            test_name='stage name')()
        name_label = stage_app.resolve_widget(stage.stage.display).down(
            test_name='stage label')()
        assert name.text == stage.name
        assert name_label.text == stage.name
        assert name.text == stage.stage.name

    # verify colors
    for color in ('r', 'g', 'b'):
        widget = stage_app.resolve_widget(settings).down(
            test_name='stage color {}'.format(color))()
        prop = 'color_{}'.format(color)
        assert getattr(stage.stage, prop) == (widget.state == 'down')
        assert getattr(stage, prop) == getattr(stage.stage, prop)

    # parallel vs serial
    serial = stage_app.resolve_widget(settings).down(
        test_name='stage serial')()
    parallel = stage_app.resolve_widget(settings).down(
        test_name='stage parallel')()
    assert (stage.stage.order == 'serial') == (serial.state == 'down') and \
        (stage.stage.order == 'parallel') == (parallel.state == 'down')

    # complete_on all vs any
    all_w = stage_app.resolve_widget(settings).down(
        test_name='stage finish all')()
    any_w = stage_app.resolve_widget(settings).down(
        test_name='stage finish any')()
    assert (stage.stage.complete_on == 'all') == (all_w.state == 'down') and \
        (stage.stage.complete_on == 'any') == (any_w.state == 'down')

    if opened_settings:
        await escape(stage_app)
    return settings


async def replace_last_ref_with_original_stage(
        stage_app: CeedTestApp,
        stages: List[Union[CeedStageRef, CeedStage]], name: str):
    start_stages = stages[:]
    ref_stage = stages[-1]
    # it should be a ref to start with
    assert isinstance(ref_stage, CeedStageRef)
    # make sure the class name matches - we added the right class
    assert ref_stage.stage.name == name

    # the label of the new sub-stage
    sub_stage_widget = ref_stage.display
    name_w = stage_app.resolve_widget(sub_stage_widget).down(
        test_name='stage label')()
    assert name_w.text == name
    # replace the ref with a copy of the stage
    ref_btn = stage_app.resolve_widget(sub_stage_widget).down(
        test_name='stage settings open')()
    await touch_widget(stage_app, ref_btn)

    # should now have replaced the ref with a copy of the original
    assert ref_stage not in stages
    assert len(stages) == len(start_stages)

    new_stage = stages[-1]
    assert ref_stage is not new_stage
    assert stages[:-1] == start_stages[:-1]
    # it should not be a ref anymore
    assert not isinstance(new_stage, CeedStageRef)

    assert_stages_same(ref_stage.stage, new_stage)

    return new_stage


async def open_stage_settings(app: CeedTestApp, stage: CeedStage):
    settings_btn = app.resolve_widget(stage.display).down(
        test_name='stage settings open')()
    await touch_widget(app, settings_btn)

    return app.resolve_widget().down(test_name='stage settings')()


async def test_stage_find_shape_in_all_stages(stage_app: CeedTestApp):
    (s1, s2, s3), (group, shape1, shape2, shape3) = create_test_stages(
        stage_app=stage_app, show_in_gui=True)
    await stage_app.wait_clock_frames(2)

    for shape in (shape1, shape2, shape3):
        for stage in (s1, s2, s3):
            assert shape.shape in [s.shape for s in stage.stage.shapes]
        assert shape.shape in group.shapes

    stage_app.app.shape_factory.remove_shape(shape2.shape)
    await stage_app.wait_clock_frames(2)

    for shape in (shape1, shape3):
        for stage in (s1, s2, s3):
            assert shape.shape in [s.shape for s in stage.stage.shapes]
        assert shape.shape in group.shapes
    for shape in (shape2, ):
        for stage in (s1, s2, s3):
            assert shape.shape not in [s.shape for s in stage.stage.shapes]
        assert shape.shape not in group.shapes

    stage_app.app.shape_factory.remove_shape(shape1.shape)
    await stage_app.wait_clock_frames(2)

    for shape in (shape3, ):
        for stage in (s1, s2, s3):
            assert shape.shape in [s.shape for s in stage.stage.shapes]
        assert shape.shape in group.shapes
    for shape in (shape2, shape1):
        for stage in (s1, s2, s3):
            assert shape.shape not in [s.shape for s in stage.stage.shapes]
        assert shape.shape not in group.shapes

    stage_app.app.shape_factory.remove_shape(shape3.shape)
    await stage_app.wait_clock_frames(2)

    for shape in (shape2, shape1, shape3):
        for stage in (s1, s2, s3):
            assert shape.shape not in [s.shape for s in stage.stage.shapes]
        assert shape.shape not in group.shapes


async def test_add_empty_stage(stage_app: CeedTestApp):
    stage_factory = stage_app.app.stage_factory
    assert not stage_factory.stages
    n = len(stage_factory.stage_names)

    # add first empty stage
    add_stage = stage_app.resolve_widget().down(test_name='stage add')()
    await touch_widget(stage_app, add_stage)

    assert stage_factory.stages
    stage = stage_factory.stages[0]
    assert stage in list(stage_factory.stage_names.values())
    assert len(stage_factory.stage_names) == n + 1
    assert stage.display.show_more

    # select the stage and add stage to it
    name_label = stage_app.resolve_widget(stage.display).down(
        test_name='stage label')()
    assert not stage.display.selected

    await touch_widget(stage_app, name_label)
    assert stage.display.selected
    await touch_widget(stage_app, add_stage)
    assert stage_factory.stages == [stage]

    # deselect the stage and add stage globally
    assert stage.display.selected
    await touch_widget(stage_app, name_label)
    await touch_widget(stage_app, add_stage)

    assert len(stage_factory.stages) == 2
    assert stage_factory.stages[0] is stage


async def test_gui_add_stages(stage_app: CeedTestApp):
    stages = []
    add_stage = stage_app.resolve_widget().down(test_name='stage add')()
    for i, stage_cls in enumerate(stage_classes):
        stage = stage_cls(app=stage_app, show_in_gui=False)
        stages.append(stage)

        # don't keep more than two stages so the list is not too long
        if i >= 2:
            oldest_stage = stages.pop(0)
            assert oldest_stage.stage in stage_app.app.stage_factory.stages
            remove_btn = stage_app.resolve_widget(
                oldest_stage.stage.display).down(test_name='del btn stage')()
            await touch_widget(stage_app, remove_btn)
            assert oldest_stage.stage not in stage_app.app.stage_factory.stages

        # add the stage
        await touch_widget(stage_app, add_stage)
        assert len(stage_app.app.stage_factory.stages) == min(2, i + 1)
        stage.stage = stage_app.app.stage_factory.stages[-1]

        # show the settings for the stage
        widget = stage.stage.display
        settings = await open_stage_settings(stage_app, stage.stage)

        # check default name
        name = stage_app.resolve_widget(settings).down(
            test_name='stage name')()
        assert not name.disabled, "root stages can be renamed"
        name_label = stage_app.resolve_widget(widget).down(
            test_name='stage label')()
        original_name = name.text
        assert stage.name != original_name
        assert original_name == name_label.text
        assert original_name in stage_app.app.stage_factory.stage_names
        assert stage.name not in stage_app.app.stage_factory.stage_names

        # change the stage name
        await replace_text(stage_app, name, stage.name)
        assert name.text == stage.name
        assert name_label.text == stage.name
        assert original_name not in stage_app.app.stage_factory.stage_names
        assert stage.name in stage_app.app.stage_factory.stage_names

        await assert_set_params_in_gui(stage_app, stage, settings)

        # close the settings widget
        await escape(stage_app)


async def test_gui_add_sub_stages(stage_app: CeedTestApp):
    add_stage = stage_app.resolve_widget().down(test_name='stage add')()
    await touch_widget(stage_app, add_stage)

    base_stage: CeedStage = stage_app.app.stage_factory.stages[0]
    name_label = stage_app.resolve_widget(base_stage.display).down(
        test_name='stage label')()
    await touch_widget(stage_app, name_label)
    assert base_stage.display.selected
    assert not base_stage.stages

    stages = []
    for i, stage_cls in enumerate(stage_classes[:4]):
        stage = stage_cls(app=stage_app, show_in_gui=False)
        stages.append(stage)

        # don't keep more than two stages so the list is not too long
        if i >= 2:
            oldest_stage = stages.pop(0)
            assert oldest_stage.stage in base_stage.stages
            remove_btn = stage_app.resolve_widget(
                oldest_stage.stage.display).down(test_name='del btn stage')()
            await touch_widget(stage_app, remove_btn)
            assert oldest_stage.stage not in base_stage.stages

        if not base_stage.display.selected:
            await touch_widget(stage_app, name_label)
            assert base_stage.display.selected

        # add the stage
        await touch_widget(stage_app, add_stage)
        assert len(base_stage.stages) == min(2, i + 1)
        stage.stage = base_stage.stages[-1]

        # replace the ref stage
        settings_btn = stage_app.resolve_widget(stage.stage.display).down(
            test_name='stage settings open')()
        await touch_widget(stage_app, settings_btn)
        stage.stage = base_stage.stages[-1]

        await assert_set_params_in_gui(stage_app, stage, check_name=False)


async def test_gui_drag_shape_to_stage(stage_app: CeedTestApp):
    (group, group2, group3), (shape1, shape2, shape3) = \
        assert_add_three_groups(
            shape_factory=stage_app.app.shape_factory, app=stage_app,
            show_in_gui=True)
    await stage_app.wait_clock_frames(2)

    (s1, s2, s3), _ = create_test_stages(
        stage_app=stage_app, add_func=False, add_shapes=False)
    await stage_app.wait_clock_frames(2)

    # multiple stages
    for stage in (s2, s3):
        container = stage.stage.display.shape_widget
        shapes = stage.stage.shapes
        assert not shapes

        # drag each shape to the stage
        added_shapes = []
        for i, shape in enumerate((shape1, group2, shape3, shape2)):
            if isinstance(shape, CeedShapeGroup):
                src = stage_app.resolve_widget(shape.widget).down(
                    test_name='group drag button')()
            else:
                shape = shape.shape
                src = stage_app.resolve_widget(shape.widget).down(
                    test_name='shape drag')()

            offset = (0, 5) if container.height else (0, 0)
            async for _ in stage_app.do_touch_drag_follow(
                    widget=src, target_widget=container,
                    target_widget_loc=('center_x', 'y'),
                    target_widget_offset=offset, drag_n=15):
                pass

            # check that shape was added
            assert len(shapes) == min(3, i + 1)
            assert shape is shapes[-1].shape

            # make sure label matches
            name_label = stage_app.resolve_widget(shapes[-1].display).down(
                test_name='stage shape name')()
            assert name_label.text == shape.name

            added_shapes.append(shapes[-1])

            # don't keep more than two stages so the list is not too long
            if i >= 2:
                oldest_shape = added_shapes.pop(0)
                assert oldest_shape in shapes
                remove_btn = stage_app.resolve_widget(
                    oldest_shape.display).down(
                    test_name='stage shape del')()
                await touch_widget(stage_app, remove_btn)
                assert oldest_shape not in shapes
            await stage_app.wait_clock_frames(2)


async def test_gui_drag_func_to_stage(stage_app: CeedTestApp):
    global_funcs = create_funcs(func_app=stage_app, show_in_gui=True)
    group_func: GroupFunction = global_funcs[-1]
    ff1 = group_func.wrapper_funcs[0]
    ff2 = group_func.wrapper_funcs[1]
    global_funcs = [
        (ff1, True)] + [(f, False) for f in global_funcs] + [(ff2, True)]
    await stage_app.wait_clock_frames(2)

    (s1, s2, s3), _ = create_test_stages(
        stage_app=stage_app, add_func=False, add_shapes=False)
    await stage_app.wait_clock_frames(2)

    # multiple funcs
    for stage in (s2, s3):
        container = stage.stage.display.func_widget
        functions = stage.stage.functions
        assert not functions

        # drag each func to the stage
        added_funcs = []
        for i, (func, is_sub_func) in enumerate(global_funcs):
            src = stage_app.resolve_widget(func.func.display).down(
                test_name='func drag btn')()

            async for _ in stage_app.do_touch_drag_follow(
                    widget=src, target_widget=container,
                    target_widget_loc=('center_x', 'y'),
                    target_widget_offset=(0, 5)):
                pass

            # check that shape was added
            assert len(functions) == min(3, i + 1)
            assert functions[-1] is not func.func
            if is_sub_func:
                assert isinstance(functions[-1], (FuncBase, FuncGroup))
                assert_funcs_same(functions[-1], func.func)
            else:
                assert isinstance(functions[-1], CeedFuncRef)
                assert func.func is functions[-1].func
                await replace_last_ref_with_original_func(
                    stage_app, functions, func.func.name)

            added_funcs.append(functions[-1])

            # don't keep more than two funcs so the list is not too long
            if i >= 2:
                oldest_func = added_funcs.pop(0)
                assert oldest_func in functions
                remove_btn = stage_app.resolve_widget(
                    oldest_func.display).down(
                    test_name='del_btn_func')()
                await touch_widget(stage_app, remove_btn)
                assert oldest_func not in functions

            await stage_app.wait_clock_frames(2)


async def test_gui_drag_stage_to_stage(stage_app: CeedTestApp):
    (s1, s2, s21), _ = create_test_stages(
        stage_app=stage_app, show_in_gui=True, add_func=False,
        add_shapes=False)
    (s3, s4, s41), _ = create_test_stages(
        stage_app=stage_app, show_in_gui=True, add_func=False,
        add_shapes=False)
    await stage_app.wait_clock_frames(2)

    # collapse stages to not take up space
    for stage in (s1, s21, s3):
        stage.stage.display.show_more = False
    await stage_app.wait_clock_frames(2)

    # multiple funcs
    for stage in (s4, s41):
        container = stage.stage.display.stage_widget
        stages = stage.stage.stages
        n_start = 0 if stage is s41 else 1
        assert len(stages) == n_start

        # drag each func to the stage
        added_stages = []
        for i, src_stage in enumerate((s1, s2, s21, s3)):
            src = stage_app.resolve_widget(src_stage.stage.display).down(
                test_name='stage drag btn')()

            async for _ in stage_app.do_touch_drag_follow(
                    widget=src, target_widget=container,
                    target_widget_loc=('center_x', 'y'),
                    target_widget_offset=(0, 5)):
                pass

            # check that shape was added
            assert len(stages) == min(3, i + 1) + n_start

            assert stages[-1] is not src_stage.stage
            if src_stage is s21:
                assert isinstance(stages[-1], CeedStage)
                assert_stages_same(stages[-1], src_stage.stage)
            else:
                assert isinstance(stages[-1], CeedStageRef)
                assert src_stage.stage is stages[-1].stage
                await replace_last_ref_with_original_stage(
                    stage_app, stages, src_stage.stage.name)

            added_stages.append(stages[-1])

            # don't keep more than two stages so the list is not too long
            if i >= 2:
                oldest_stage = added_stages.pop(0)
                assert oldest_stage in stages
                remove_btn = stage_app.resolve_widget(
                    oldest_stage.display).down(
                    test_name='del btn stage')()
                await touch_widget(stage_app, remove_btn)
                assert oldest_stage not in stages

            await stage_app.wait_clock_frames(2)


def verify_color(
        stage_app, shape_color, shape2_color, frame, centers, flip, video_mode):
    (cx1, cy1), (cx2, cy2) = centers
    if flip:
        cx1 = 1920 - cx1
        cx2 = 1920 - cx2

    centers = [[(cx1, cy1), (cx2, cy2)]]
    if 'QUAD' in video_mode:
        cx1, cx2, cy1, cy2 = cx1 // 2, cx2 // 2, cy1 // 2, cy2 // 2
        corners = ((0, 540), (960, 540), (0, 0), (960, 0))
        centers = [
            [(cx + x, cy + y) for cx, cy in [(cx1, cy1), (cx2, cy2)]]
            for x, y in corners]

    if video_mode == 'QUAD12X':
        # first get all 4 centers values, one for each quadrant
        rgb_values = []
        for i in range(4):
            rgb = stage_app.get_widget_pos_pixel(
                stage_app.app.shape_factory, centers[i])
            rgb = [[c / 255 for c in p] for p in rgb]
            rgb_values.append(rgb)

        # r, g, b
        for plane in [0, 1, 2]:
            # 4 quads
            for color1, color2 in rgb_values:
                assert isclose(
                    color1[plane], shape_color[frame][3], abs_tol=2 / 255)
                assert isclose(
                    color2[plane], shape2_color[frame][3], abs_tol=2 / 255)
                frame += 1
    else:
        n_sub_frames = 1
        if video_mode == 'QUAD4X':
            n_sub_frames = 4

        for i in range(n_sub_frames):
            points = stage_app.get_widget_pos_pixel(
                stage_app.app.shape_factory, centers[i])
            points = [[c / 255 for c in p] for p in points]
            (r1, g1, b1, _), (r2, g2, b2, _) = points

            val = shape_color[frame]
            assert isclose(r1, val[3], abs_tol=2 / 255) if val[0] else r1 == 0
            assert isclose(g1, val[3], abs_tol=2 / 255) if val[1] else g1 == 0
            assert isclose(b1, val[3], abs_tol=2 / 255) if val[2] else b1 == 0
            val = shape2_color[frame]
            assert isclose(r2, val[3], abs_tol=2 / 255) if val[0] else r2 == 0
            assert isclose(g2, val[3], abs_tol=2 / 255) if val[1] else g2 == 0
            assert isclose(b2, val[3], abs_tol=2 / 255) if val[2] else b2 == 0
            frame += 1

    return frame


@pytest.mark.parametrize('video_mode', ['RGB', 'QUAD4X', 'QUAD12X'])
@pytest.mark.parametrize(
    'flip,skip', [(True, False), (False, True), (False, False)])
async def test_recursive_play_stage_intensity(
        stage_app: CeedTestApp, tmp_path, flip, skip, video_mode):
    """Checks that proper frame rendering happens in all these modes.
    In skip mode, some frames are skipped if GPU/CPU is too slow.
    """
    from ..test_stages import create_recursive_stages
    from .examples.shapes import CircleShapeP1, CircleShapeP2
    from kivy.clock import Clock
    from ceed.analysis import CeedDataReader

    root, g1, g2, s1, s2, s3, s4, s5, s6 = create_recursive_stages(
        stage_app.app.stage_factory, app=stage_app)

    from ceed.function.plugin import LinearFunc
    for i, stage in enumerate((s1, s2, s3, s4, s5, s6)):
        stage.stage.add_func(LinearFunc(
            function_factory=stage_app.app.function_factory, b=0, m=.5,
            duration=(i % 2 + 1) * 1))

    shape = CircleShapeP1(
        app=None, painter=stage_app.app.shape_factory, show_in_gui=True)

    shape2 = CircleShapeP2(
        app=None, painter=stage_app.app.shape_factory, show_in_gui=True)
    s1.stage.add_shape(shape.shape)
    s4.stage.add_shape(shape.shape)
    s5.stage.add_shape(shape.shape)
    s2.stage.add_shape(shape2.shape)
    s3.stage.add_shape(shape2.shape)
    s6.stage.add_shape(shape2.shape)

    root.show_in_gui()
    await stage_app.wait_clock_frames(2)

    frame = 0
    event = None
    # make GPU too slow to force skipping frames, when enabled
    fps = await measure_fps(stage_app) + 10
    rate = stage_app.app.view_controller.frame_rate = fps
    stage_app.app.view_controller.use_software_frame_rate = False
    stage_app.app.view_controller.flip_projector = flip
    stage_app.app.view_controller.skip_estimated_missed_frames = skip
    stage_app.app.view_controller.video_mode = video_mode
    stage_app.app.view_controller.pad_to_stage_handshake = False

    n_sub_frames = 1
    if video_mode == 'QUAD4X':
        n_sub_frames = 4
    elif video_mode == 'QUAD12X':
        n_sub_frames = 12

    centers = shape.center, shape2.center
    num_frames = rate * n_sub_frames * (2 + 1 + 2 + 1)
    shape_color = [(False, False, False, 0.), ] * num_frames
    shape2_color = [(False, False, False, 0.), ] * num_frames
    skipped_frame_indices = set()
    n_missed_frames = 0

    for s, start, e in [(s1, 0, 1), (s4, 3, 5), (s5, 5, 6)]:
        for i in range(start * rate * n_sub_frames, e * rate * n_sub_frames):
            val = (i - start * rate * n_sub_frames) / (rate * n_sub_frames) * .5
            shape_color[i] = s.color_r, s.color_g, s.color_b, val

    for s, start, e in [(s2, 0, 2), (s3, 2, 3), (s6, 5, 6)]:
        for i in range(start * rate * n_sub_frames, e * rate * n_sub_frames):
            val = (i - start * rate * n_sub_frames) / (rate * n_sub_frames) * .5
            shape2_color[i] = s.color_r, s.color_g, s.color_b, val

    def verify_intensity(*largs):
        nonlocal frame, n_missed_frames
        # total frames is a multiple of n_sub_frames
        if not stage_app.app.view_controller.stage_active:
            assert stage_app.app.view_controller.count - 1 == num_frames
            if skip:
                # last frame could be passed actual frames
                assert frame - n_missed_frames * n_sub_frames <= num_frames
            else:
                assert frame == num_frames
            event.cancel()
            return
        # not yet started
        if not stage_app.app.view_controller.count:
            return

        # some frame may have been skipped, but num_frames is max frames
        # This callback happens after frame callback and after the frame flip.
        # This also means we record even the last skipped frames (if skipped)
        assert frame < num_frames

        frame = verify_color(
            stage_app, shape_color, shape2_color, frame, centers, flip,
            video_mode)
        assert stage_app.app.view_controller.count == frame

        if skip:
            # some frames may have been dropped for next frame
            n_missed_frames = stage_app.app.view_controller._n_missed_frames
            for k in range(n_missed_frames * n_sub_frames):
                # frame is next frame index, next frame is skipped
                skipped_frame_indices.add(frame)
                frame += 1
        else:
            assert not stage_app.app.view_controller._n_missed_frames

    event = Clock.create_trigger(verify_intensity, timeout=0, interval=True)
    event()
    stage_app.app.view_controller.request_stage_start(root.name)

    await wait_experiment_done(stage_app, timeout=num_frames / rate * 50)

    filename = str(tmp_path / 'recursive_play_stage_intensity.h5')
    stage_app.app.ceed_data.save(filename=filename)

    f = CeedDataReader(filename)
    f.open_h5()
    assert f.experiments_in_file == ['0']
    assert not f.num_images_in_file
    f.load_experiment(0)

    shape_data = f.shapes_intensity[shape.name]
    shape_data_rendered = f.shapes_intensity_rendered[shape.name]
    shape2_data = f.shapes_intensity[shape2.name]
    shape2_data_rendered = f.shapes_intensity_rendered[shape2.name]
    recorded_rendered_frames = f.rendered_frames

    # even when skipping, skipped frames are still logged but they are removed
    # in xxx_rendered arrays
    if skip:
        # because frame rate is high, we'll definitely drop frames
        assert skipped_frame_indices
    else:
        assert not skipped_frame_indices

    assert shape_data.shape[0] == num_frames
    assert shape2_data.shape[0] == num_frames

    n_skipped = len(skipped_frame_indices)
    if skip:
        # last frame may be recorded as skipped, but if stage is done frame is
        # not real. n_missed_frames is the n_missed_frames from last frame
        assert num_frames - n_skipped <= shape_data_rendered.shape[0] \
            <= num_frames - n_skipped + n_sub_frames * n_missed_frames
        assert num_frames - n_skipped <= shape2_data_rendered.shape[0] \
            <= num_frames - n_skipped + n_sub_frames * n_missed_frames
    else:
        assert shape_data_rendered.shape[0] == num_frames
        assert shape2_data_rendered.shape[0] == num_frames

    # in QUAD12X mode, all 3 channels have same value in the data (because we
    # show gray). But the projector outputs different values for each channel,
    # for each sub-frame
    gray = video_mode == 'QUAD12X'
    i = 0
    k = 0
    for (r, g, b, val), (r1, g1, b1, _) in zip(shape_color, shape_data):
        assert isclose(val, r1, abs_tol=2 / 255) if r or gray else r1 == 0
        assert isclose(val, g1, abs_tol=2 / 255) if g or gray else g1 == 0
        assert isclose(val, b1, abs_tol=2 / 255) if b or gray else b1 == 0

        if skip:
            assert recorded_rendered_frames[k] \
                == (k not in skipped_frame_indices)
        else:
            assert recorded_rendered_frames[k]

        if k not in skipped_frame_indices:
            r1, g1, b1, _ = shape_data_rendered[i, :]
            assert isclose(val, r1, abs_tol=2 / 255) if r or gray else r1 == 0
            assert isclose(val, g1, abs_tol=2 / 255) if g or gray else g1 == 0
            assert isclose(val, b1, abs_tol=2 / 255) if b or gray else b1 == 0
            i += 1
        k += 1

    i = 0
    k = 0
    for (r, g, b, val), (r1, g1, b1, _) in zip(shape2_color, shape2_data):
        assert isclose(val, r1, abs_tol=2 / 255) if r or gray else r1 == 0
        assert isclose(val, g1, abs_tol=2 / 255) if g or gray else g1 == 0
        assert isclose(val, b1, abs_tol=2 / 255) if b or gray else b1 == 0

        if skip:
            assert recorded_rendered_frames[k] \
                == (k not in skipped_frame_indices)
        else:
            assert recorded_rendered_frames[k]

        if k not in skipped_frame_indices:
            r1, g1, b1, _ = shape2_data_rendered[i, :]
            assert isclose(val, r1, abs_tol=2 / 255) if r or gray else r1 == 0
            assert isclose(val, g1, abs_tol=2 / 255) if g or gray else g1 == 0
            assert isclose(val, b1, abs_tol=2 / 255) if b or gray else b1 == 0
            i += 1
        k += 1

    f.close_h5()


async def test_moat_stage_shapes(stage_app: CeedTestApp, tmp_path):
    from ..test_stages import create_recursive_stages
    from .examples.shapes import CircleShapeP1, CircleShapeP1Internal
    from ceed.function.plugin import ConstFunc
    from ceed.analysis import CeedDataReader

    root, g1, g2, s1, s2, s3, s4, s5, s6 = create_recursive_stages(
        stage_app.app.stage_factory, app=stage_app)
    # internal shape
    s1.stage.color_r = False
    s1.stage.color_g = False
    s1.stage.color_b = True
    # surrounding shape
    s2.stage.color_r = True
    s2.stage.color_g = False
    s2.stage.color_b = True

    shape = CircleShapeP1(
        app=None, painter=stage_app.app.shape_factory, show_in_gui=True)
    internal_shape = CircleShapeP1Internal(
        app=None, painter=stage_app.app.shape_factory, show_in_gui=True)

    s1.stage.add_func(ConstFunc(
        function_factory=stage_app.app.function_factory, a=1, duration=5))
    s1.stage.add_shape(internal_shape.shape)

    s2.stage.add_func(ConstFunc(
        function_factory=stage_app.app.function_factory, a=1, duration=5))
    s2.stage.add_shape(shape.shape)

    root.show_in_gui()
    await stage_app.wait_clock_frames(2)

    stage_app.app.view_controller.frame_rate = 10
    stage_app.app.view_controller.use_software_frame_rate = False
    stage_app.app.view_controller.flip_projector = False

    stage_app.app.view_controller.request_stage_start(root.name)
    await wait_stage_experiment_started(stage_app)
    assert stage_app.app.view_controller.stage_active

    points = stage_app.get_widget_pos_pixel(
        stage_app.app.shape_factory, [internal_shape.center, shape.center])
    (r1, g1, b1, _), (r2, g2, b2, _) = points
    assert r1 == 0
    assert g1 == 0
    assert b1 == 255

    assert r2 == 255
    assert g2 == 0
    assert b2 == 255

    stage_app.app.view_controller.request_stage_end()
    await stage_app.wait_clock_frames(2)
    assert not stage_app.app.view_controller.stage_active

    # now hide internal shape behind larger circle
    stage_app.app.shape_factory.move_shape_upwards(shape.shape)
    await stage_app.wait_clock_frames(2)

    stage_app.app.view_controller.request_stage_start(root.name)
    await wait_stage_experiment_started(stage_app)
    assert stage_app.app.view_controller.stage_active

    points = stage_app.get_widget_pos_pixel(
        stage_app.app.shape_factory, [internal_shape.center, shape.center])
    (r1, g1, b1, _), (r2, g2, b2, _) = points
    assert r1 == 255
    assert g1 == 0
    assert b1 == 255

    assert r2 == 255
    assert g2 == 0
    assert b2 == 255

    stage_app.app.view_controller.request_stage_end()
    await stage_app.wait_clock_frames(2)

    filename = str(tmp_path / 'moat_stage_shapes.h5')
    stage_app.app.ceed_data.save(filename=filename)

    f = CeedDataReader(filename)
    f.open_h5()
    assert f.experiments_in_file == ['0', '1']
    assert not f.num_images_in_file

    f.load_experiment(0)
    assert tuple(np.array(f.shapes_intensity[shape.name])[0, :3]) == (1, 0, 1)
    assert tuple(
        np.array(f.shapes_intensity[internal_shape.name])[0, :3]) == (0, 0, 1)

    f.load_experiment(1)
    assert tuple(np.array(f.shapes_intensity[shape.name])[0, :3]) == (1, 0, 1)
    assert tuple(
        np.array(f.shapes_intensity[internal_shape.name])[0, :3]) == (0, 0, 1)

    f.close_h5()


async def test_moat_single_stage_shapes(stage_app: CeedTestApp, tmp_path):
    from ..test_stages import create_recursive_stages
    from .examples.shapes import CircleShapeP1, CircleShapeP1Internal
    from ceed.function.plugin import ConstFunc
    from ceed.analysis import CeedDataReader

    root, g1, g2, s1, s2, s3, s4, s5, s6 = create_recursive_stages(
        stage_app.app.stage_factory, app=stage_app)
    s1.stage.color_r = False
    s1.stage.color_g = False
    s1.stage.color_b = True

    shape = CircleShapeP1(
        app=None, painter=stage_app.app.shape_factory, show_in_gui=True)
    internal_shape = CircleShapeP1Internal(
        app=None, painter=stage_app.app.shape_factory, show_in_gui=True)

    s1.stage.add_func(ConstFunc(
        function_factory=stage_app.app.function_factory, a=1, duration=5))
    stage_shape = s1.stage.add_shape(internal_shape.shape)
    s1.stage.add_shape(shape.shape)
    stage_shape.keep_dark = True

    root.show_in_gui()
    await stage_app.wait_clock_frames(2)

    stage_app.app.view_controller.frame_rate = 10
    stage_app.app.view_controller.use_software_frame_rate = False
    stage_app.app.view_controller.flip_projector = False

    stage_app.app.view_controller.request_stage_start(root.name)
    await wait_stage_experiment_started(stage_app)
    assert stage_app.app.view_controller.stage_active

    points = stage_app.get_widget_pos_pixel(
        stage_app.app.shape_factory, [internal_shape.center, shape.center])
    (r1, g1, b1, _), (r2, g2, b2, _) = points
    assert r1 == 0
    assert g1 == 0
    assert b1 == 0

    assert r2 == 0
    assert g2 == 0
    assert b2 == 255

    stage_app.app.view_controller.request_stage_end()
    await stage_app.wait_clock_frames(2)
    assert not stage_app.app.view_controller.stage_active

    filename = str(tmp_path / 'moat_single_stage_shapes.h5')
    stage_app.app.ceed_data.save(filename=filename)

    f = CeedDataReader(filename)
    f.open_h5()
    assert f.experiments_in_file == ['0', ]
    assert not f.num_images_in_file

    f.load_experiment(0)
    assert tuple(np.array(f.shapes_intensity[shape.name])[0]) == (0, 0, 1, 1)
    assert tuple(
        np.array(f.shapes_intensity[internal_shape.name])[0]) == (0, 0, 0, 1)
    f.close_h5()


@pytest.mark.parametrize(
    'quad,sub_frames', [('RGB', 1), ('QUAD4X', 4), ('QUAD12X', 12)])
@pytest.mark.parametrize('skip', [False, True])
async def test_pad_stage_ticks(
        stage_app: CeedTestApp, tmp_path, quad, sub_frames, skip):
    from ceed.analysis import CeedDataReader

    root = SerialAllStage(
        stage_factory=stage_app.app.stage_factory, show_in_gui=False,
        app=stage_app, create_add_to_parent=True)

    shape = CircleShapeP1(
        app=None, painter=stage_app.app.shape_factory, show_in_gui=True)
    root.stage.add_shape(shape.shape)
    root.show_in_gui()
    await stage_app.wait_clock_frames(2)

    # use a larger frame rate so we have to drop frames
    stage_app.app.view_controller.frame_rate = await measure_fps(stage_app) + 10
    stage_app.app.view_controller.skip_estimated_missed_frames = skip
    stage_app.app.view_controller.use_software_frame_rate = False
    stage_app.app.view_controller.video_mode = quad

    stage_app.app.view_controller.pad_to_stage_handshake = False
    stage_app.app.view_controller.request_stage_start(root.name)
    await wait_experiment_done(stage_app)

    stage_app.app.view_controller.pad_to_stage_handshake = True
    stage_app.app.view_controller.request_stage_start(root.name)
    await wait_experiment_done(stage_app, 300)

    filename = str(tmp_path / 'pad_stage_ticks.h5')
    stage_app.app.ceed_data.save(filename=filename)

    f = CeedDataReader(filename)
    f.open_h5()
    assert f.experiments_in_file == ['0', '1']
    assert not f.num_images_in_file

    f.load_experiment('0')
    assert f.shapes_intensity[shape.name].shape == (0, 4)

    f.load_experiment('1')
    # sub_frames scales up the handshake since IO is same for each sub-frame
    # Even skipped frames are logged so size matches
    assert f.shapes_intensity[shape.name].shape == (
        stage_app.app.data_serializer.num_ticks_handshake(16, sub_frames), 4)
    assert f.shapes_intensity[shape.name].shape == (
        stage_app.app.data_serializer.num_ticks_handshake(16, 1) * sub_frames,
        4)

    frame_time_counter = np.asarray(f._block.data_arrays['frame_time_counter'])
    frame_time = np.asarray(f._block.data_arrays['frame_time'])
    rendered_frames_bool = f.rendered_frames
    assert len(frame_time_counter) == len(frame_time)
    assert np.sum(rendered_frames_bool) == len(frame_time_counter) * sub_frames

    frame_counter = np.asarray(f._block.data_arrays['frame_counter'])
    n = f.shapes_intensity[shape.name].shape[0]
    # some frames will have been skipped because of higher frame rate than GPU
    if skip:
        assert sub_frames * len(frame_time_counter) < n
    else:
        assert sub_frames * len(frame_time_counter) == n

    # we didn't stop early so all frames are rendered
    rendered_indices = np.arange(0, n, sub_frames)
    if skip:
        assert len(frame_time_counter) < len(frame_counter) // sub_frames
        assert len(rendered_indices) > len(frame_time_counter)
    else:
        assert len(frame_time_counter) == len(frame_counter) // sub_frames
        assert len(rendered_indices) == len(frame_time_counter)

    assert np.all(np.arange(1, n + 1) == frame_counter)
    # count recorded is last sub-frame
    if skip:
        assert np.all(
            np.isin(frame_time_counter, rendered_indices + sub_frames))
        assert np.all(frame_time_counter[1:] - frame_time_counter[:-1] > 0)

        assert np.all(np.isin(
            frame_time_counter,
            frame_counter[rendered_indices + sub_frames - 1]))
    else:
        assert np.all(frame_time_counter == rendered_indices + sub_frames)
        assert np.all(
            frame_counter[rendered_indices + sub_frames - 1]
            == frame_time_counter)

    f.close_h5()


@contextmanager
def add_to_path(tmp_path, *args):
    sys.path.append(str(tmp_path))
    mod = tmp_path / 'my_gui_stage_plugin' / '__init__.py'
    try:
        mod.parent.mkdir()
        mod.write_text(fake_plugin_stage)
        yield None
    finally:
        sys.path.remove(str(tmp_path))
        if 'my_gui_stage_plugin' in sys.modules:
            del sys.modules['my_gui_stage_plugin']


@pytest.mark.parametrize(
    "ceed_app",
    [{'yaml_config': {'external_stage_plugin_package': 'my_gui_stage_plugin'},
      'app_context': add_to_path}, ],
    indirect=True
)
async def test_external_plugin_named_package(stage_app: CeedTestApp, tmp_path):
    stage_factory = stage_app.app.stage_factory

    assert 'FakeStage' in stage_factory.stages_cls


@pytest.mark.parametrize(
    'quad,sub_frames', [('RGB', 1), ('QUAD4X', 4), ('QUAD12X', 12)])
@pytest.mark.parametrize('main_frames', [1, 1.5, 2])
async def test_short_stage(
        stage_app: CeedTestApp, tmp_path, quad, sub_frames, main_frames):
    from ceed.analysis import CeedDataReader
    from ceed.function.plugin import LinearFunc
    from kivy.clock import Clock

    num_frames = int(math.ceil(main_frames * sub_frames))
    rate = main_frames

    root = SerialAllStage(
        stage_factory=stage_app.app.stage_factory, show_in_gui=False,
        app=stage_app, create_add_to_parent=True)
    shape = CircleShapeP1(
        app=None, painter=stage_app.app.shape_factory, show_in_gui=True)
    root.stage.add_shape(shape.shape)
    root.stage.add_func(LinearFunc(
        function_factory=stage_app.app.function_factory, b=0, m=1,
        duration=1))
    root.show_in_gui()
    await stage_app.wait_clock_frames(2)

    # use a larger frame rate so we have to drop frames
    stage_app.app.view_controller.frame_rate = rate
    stage_app.app.view_controller.use_software_frame_rate = False
    stage_app.app.view_controller.video_mode = quad
    stage_app.app.view_controller.pad_to_stage_handshake = False
    stage_app.app.view_controller.flip_projector = False

    frame = 0
    event = None
    cx, cy = shape.shape.centroid
    if sub_frames == 1:
        centers = [(cx, cy)]
    else:
        cx1, cy1 = cx // 2, cy // 2
        corners = ((0, 540), (960, 540), (0, 0), (960, 0))
        centers = [(cx1 + x, cy1 + y) for x, y in corners]
    intensity = []
    total_rounded_frames = math.ceil(main_frames) * sub_frames

    def verify_intensity(*largs):
        nonlocal frame
        if not stage_app.app.view_controller.stage_active:
            event.cancel()
            return
        # not yet started
        if not stage_app.app.view_controller.count:
            return

        assert frame < num_frames

        rgb = stage_app.get_widget_pos_pixel(
            stage_app.app.shape_factory, centers)
        rgb = [[c / 255 for c in p] for p in rgb]
        if sub_frames == 12:
            for plane in range(3):
                for point in rgb:
                    value = point[plane]
                    intensity.append((value, value, value, 1))
        else:
            intensity.extend(rgb)
        frame += sub_frames

        assert frame in (
            stage_app.app.view_controller.count, total_rounded_frames)
        assert not stage_app.app.view_controller._n_missed_frames

    event = Clock.create_trigger(verify_intensity, timeout=0, interval=True)
    event()
    stage_app.app.view_controller.request_stage_start(root.name)

    await wait_experiment_done(stage_app, timeout=50)

    assert stage_app.app.view_controller.count == num_frames + 1
    # only counts whole frames
    assert frame == total_rounded_frames
    # have data for blank frames at end
    assert len(intensity) == total_rounded_frames
    assert total_rounded_frames >= num_frames

    filename = str(tmp_path / 'short_stage.h5')
    stage_app.app.ceed_data.save(filename=filename)
    with CeedDataReader(filename) as f:
        f.load_experiment(0)

        shape_data = f.shapes_intensity[shape.name]
        shape_data_rendered = f.shapes_intensity_rendered[shape.name]
        recorded_rendered_frames = f.rendered_frames

        assert shape_data.shape[0] == num_frames
        assert shape_data_rendered.shape[0] == num_frames
        assert len(recorded_rendered_frames) == num_frames

        # for each sub-frame
        gray = quad == 'QUAD12X'
        r, g, b = root.color_r, root.color_g, root.color_b
        for i, ((v1, v2, v3, _), (r1, g1, b1, _)) in enumerate(
                zip(intensity[:num_frames], shape_data)):
            # we saw the intensity we expect
            val = i / (main_frames * sub_frames)
            assert isclose(val, v1, abs_tol=2 / 255) if r or gray else v1 == 0
            assert isclose(val, v2, abs_tol=2 / 255) if g or gray else v2 == 0
            assert isclose(val, v3, abs_tol=2 / 255) if b or gray else v3 == 0

            # what we saw is what is recorded
            assert isclose(v1, r1, abs_tol=2 / 255)
            assert isclose(v2, g1, abs_tol=2 / 255)
            assert isclose(v3, b1, abs_tol=2 / 255)

            assert recorded_rendered_frames[i]
            assert shape_data_rendered[i, 0] == r1
            assert shape_data_rendered[i, 1] == g1
            assert shape_data_rendered[i, 2] == b1

        # remaining frames are blank in quad mode
        for r, g, b, _ in intensity[num_frames:]:
            assert not r
            assert not g
            assert not b
