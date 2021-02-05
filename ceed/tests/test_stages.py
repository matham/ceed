import pytest
import os
from copy import deepcopy, copy
import math
from typing import Tuple, List, Dict
from collections import defaultdict
import sys
from itertools import product
from fractions import Fraction

from ceed.shape import CeedPaintCanvasBehavior
from ceed.function import FunctionFactoryBase, FuncGroup, FuncBase
from ceed.stage import StageFactoryBase, CeedStage, CeedStageRef, \
    StageDoneException, register_all_stages, register_external_stages
from ceed.tests.test_app.examples.stages import ParaAllStage, ParaAnyStage, \
    SerialAllStage, SerialAnyStage, assert_stages_same, make_stage, \
    fake_plugin_stage
from ceed.tests.test_app.test_shape import make_4_shapes
from ceed.tests.test_app.examples.shapes import Shape, EllipseShapeP1, \
    CircleShapeP1, PolygonShapeP1, FreeformPolygonShapeP1, EllipseShapeP2, \
    CircleShapeP2, PolygonShapeP2, FreeformPolygonShapeP2, \
    CircleShapeP1Internal
from .test_functions import register_callback_distribution
from ceed.tests.test_app.examples.funcs import LinearFunctionF1, \
    ConstFunctionF1
from ceed.utils import collapse_list_to_counts


def create_recursive_stages(
        stage_factory: StageFactoryBase, show_in_gui=False, app=None):
    root = SerialAllStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        create_add_to_parent=not show_in_gui)

    g1 = ParaAllStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        parent_wrapper_stage=root,
        create_add_to_parent=not show_in_gui)

    s1 = SerialAllStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        parent_wrapper_stage=g1,
        create_add_to_parent=not show_in_gui)
    s2 = SerialAllStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        parent_wrapper_stage=g1,
        create_add_to_parent=not show_in_gui)

    s3 = SerialAnyStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        parent_wrapper_stage=root,
        create_add_to_parent=not show_in_gui)
    s4 = SerialAllStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        parent_wrapper_stage=root,
        create_add_to_parent=not show_in_gui)

    g2 = ParaAnyStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        parent_wrapper_stage=root,
        create_add_to_parent=not show_in_gui)
    s5 = SerialAnyStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        parent_wrapper_stage=g2,
        create_add_to_parent=not show_in_gui)
    s6 = SerialAllStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        parent_wrapper_stage=g2,
        create_add_to_parent=not show_in_gui)

    return root, g1, g2, s1, s2, s3, s4, s5, s6


def create_2_shape_stage(
        stage_factory: StageFactoryBase, show_in_gui=False, app=None):
    shape1 = CircleShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=show_in_gui)
    shape2 = CircleShapeP2(
        app=None, painter=stage_factory.shape_factory, show_in_gui=show_in_gui)

    root = ParaAllStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        create_add_to_parent=not show_in_gui)

    s1 = SerialAllStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        parent_wrapper_stage=root,
        create_add_to_parent=not show_in_gui)
    s2 = SerialAllStage(
        stage_factory=stage_factory, show_in_gui=show_in_gui, app=app,
        parent_wrapper_stage=root,
        create_add_to_parent=not show_in_gui)

    s1.stage.add_shape(shape1.shape)
    s2.stage.add_shape(shape2.shape)

    return root, s1, s2, shape1, shape2


def create_4_stages(stage_factory: StageFactoryBase):
    cls = stage_factory.get('CeedStage')
    s1 = cls(
        stage_factory=stage_factory,
        function_factory=stage_factory.function_factory,
        shape_factory=stage_factory.shape_factory, name='s', order='serial',
        complete_on='any', disable_pre_compute=True, loop=4, color_r=True,
        color_g=False, color_b=True, color_a=.4)
    s2 = cls(
        stage_factory=stage_factory,
        function_factory=stage_factory.function_factory,
        shape_factory=stage_factory.shape_factory, name='s2', order='parallel',
        complete_on='all', disable_pre_compute=False, loop=5, color_r=False,
        color_g=True, color_b=False, color_a=.5)
    s3 = cls(
        stage_factory=stage_factory,
        function_factory=stage_factory.function_factory,
        shape_factory=stage_factory.shape_factory, name='s3', order='serial',
        complete_on='any', disable_pre_compute=True, loop=6, color_r=False,
        color_g=False, color_b=False, color_a=.6)
    s4 = cls(
        stage_factory=stage_factory,
        function_factory=stage_factory.function_factory,
        shape_factory=stage_factory.shape_factory, name='s4', order='parallel',
        complete_on='all', disable_pre_compute=False, loop=7, color_r=True,
        color_g=True, color_b=True, color_a=.9)

    return s1, s2, s3, s4


def get_stage_time_intensity(
        stage_factory: StageFactoryBase, stage_name: str, frame_rate,
        pre_compute: bool = False
) -> Tuple[Dict[str, List[Tuple[float, float, float, float]]], int]:
    """Samples the stage with the given frame rate and returns the intensity
    value for each shape for each timestamp.
    """
    obj_values = stage_factory.get_all_shape_values(
        frame_rate, stage_name=stage_name, pre_compute=pre_compute)

    n = 0
    if obj_values:
        n = len(obj_values[list(obj_values.keys())[0]])
    return obj_values, n


def test_register_stages(
        function_factory: FunctionFactoryBase,
        shape_factory: CeedPaintCanvasBehavior):
    class MyCeedStage(CeedStage):

        def __init__(self, name='MyStage', **kwargs):
            super().__init__(name=name, **kwargs)

    class MyCeedStage2(CeedStage):

        def __init__(self, name='MyStage2', **kwargs):
            super().__init__(name=name, **kwargs)

    stage_factory = StageFactoryBase(
        function_factory=function_factory, shape_factory=shape_factory)
    count = 0

    def count_changes(*largs):
        nonlocal count
        count += 1
    stage_factory.fbind('on_changed', count_changes)

    assert not stage_factory.stages_cls
    assert not stage_factory.stage_names
    assert not stage_factory.stages
    assert not stage_factory.stages_inst_default
    assert not stage_factory.get_classes()
    assert not stage_factory.get_names()

    stage_factory.register(MyCeedStage)
    assert count
    assert stage_factory.stages_cls['MyCeedStage'] is MyCeedStage
    assert isinstance(stage_factory.stage_names['MyStage'], MyCeedStage)
    assert isinstance(stage_factory.stages_inst_default['MyStage'], MyCeedStage)
    assert MyCeedStage in stage_factory.get_classes()
    assert 'MyCeedStage' in stage_factory.get_names()

    s = MyCeedStage2(
        stage_factory=stage_factory, function_factory=function_factory,
        shape_factory=shape_factory)
    count = 0
    stage_factory.register(MyCeedStage2, instance=s)
    assert count
    assert stage_factory.stages_cls['MyCeedStage2'] is MyCeedStage2
    assert stage_factory.stage_names['MyStage2'] is s
    assert stage_factory.stages_inst_default['MyStage2'] is s
    assert MyCeedStage2 in stage_factory.get_classes()
    assert 'MyCeedStage2' in stage_factory.get_names()
    assert not stage_factory.stages


def test_auto_register(stage_factory: StageFactoryBase):
    assert not stage_factory.stages
    assert stage_factory.get('CeedStage') is CeedStage
    assert isinstance(stage_factory.stage_names['Stage'], CeedStage)
    assert isinstance(stage_factory.stages_inst_default['Stage'], CeedStage)

    assert stage_factory.get('SomeStage') is None


def test_register_user_stage(stage_factory: StageFactoryBase):
    assert not stage_factory.stages

    s, s2, _, _ = create_4_stages(stage_factory)

    stage_factory.test_changes_count = 0
    stage_factory.add_stage(s)
    assert stage_factory.test_changes_count
    assert s in stage_factory.stages
    assert stage_factory.stage_names['s'] is s

    stage_factory.test_changes_count = 0
    stage_factory.add_stage(s2)
    assert stage_factory.test_changes_count
    assert s2 in stage_factory.stages
    assert stage_factory.stage_names['s2'] is s2


def test_factory_re_register(stage_factory: StageFactoryBase):
    with pytest.raises(ValueError):
        stage_factory.register(CeedStage)


def test_factory_stage_unique_names(stage_factory: StageFactoryBase):
    assert not len(stage_factory.stages)
    n = len(stage_factory.stage_names)
    stage = SerialAllStage(stage_factory=stage_factory, show_in_gui=False)
    stage.create_stage()
    stage = stage.stage

    # add first stage
    stage_factory.test_changes_count = 0
    stage_factory.add_stage(stage)
    assert len(stage_factory.stages) == 1
    assert len(stage_factory.stage_names) == 1 + n
    assert stage in stage_factory.stages
    assert stage.name in stage_factory.stage_names
    assert stage is stage_factory.stage_names[stage.name]
    assert stage_factory.test_changes_count

    stage2 = SerialAllStage(stage_factory=stage_factory, show_in_gui=False)
    stage2.create_stage()
    stage2 = stage2.stage
    stage2.name = stage.name
    assert stage2.name == stage.name

    # add second stage
    stage_factory.test_changes_count = 0
    stage_factory.add_stage(stage2)
    assert stage2.name != stage.name
    assert len(stage_factory.stages) == 2
    assert len(stage_factory.stage_names) == 2 + n
    assert stage2 in stage_factory.stages
    assert stage2.name in stage_factory.stage_names
    assert stage2 is stage_factory.stage_names[stage2.name]
    assert stage_factory.test_changes_count

    # try making stage2 the same name as stage 1
    stage_factory.test_changes_count = 0
    stage2.name = stage.name
    assert stage2.name != stage.name
    assert len(stage_factory.stages) == 2
    assert len(stage_factory.stage_names) == 2 + n
    assert stage2 in stage_factory.stages
    assert stage2.name in stage_factory.stage_names
    assert stage2 is stage_factory.stage_names[stage2.name]
    assert stage_factory.test_changes_count

    # give stage2an explicit name
    stage_factory.test_changes_count = 0
    stage.name = 'stagnation'
    assert stage2.name != stage.name
    assert stage.name == 'stagnation'
    assert len(stage_factory.stages) == 2
    assert len(stage_factory.stage_names) == 2 + n
    assert stage in stage_factory.stages
    assert stage is stage_factory.stage_names['stagnation']
    assert stage_factory.test_changes_count

    # try setting stage2 to the empty name
    stage_factory.test_changes_count = 0
    stage2.name = ''
    assert stage2.name != stage.name
    assert stage2.name
    assert len(stage_factory.stages) == 2
    assert len(stage_factory.stage_names) == 2 + n
    assert stage2 in stage_factory.stages
    assert stage2.name in stage_factory.stage_names
    assert stage2 is stage_factory.stage_names[stage2.name]
    assert stage_factory.test_changes_count


def test_shape_add_remove(stage_factory: StageFactoryBase):
    assert not len(stage_factory.stages)
    n = len(stage_factory.stage_names)

    stage = SerialAllStage(stage_factory=stage_factory, show_in_gui=False)
    stage.create_stage()
    stage = stage.stage

    # add stage
    stage_factory.test_changes_count = 0
    stage_factory.add_stage(stage)
    assert stage in stage_factory.stages
    assert stage.name in stage_factory.stage_names
    assert stage is stage_factory.stage_names[stage.name]
    assert stage_factory.test_changes_count

    # remove shape
    stage_factory.test_changes_count = 0
    assert stage_factory.remove_stage(stage)
    assert stage not in stage_factory.stages
    assert stage.name not in stage_factory.stage_names
    assert stage_factory.test_changes_count

    # remove same shape again
    with pytest.raises(ValueError):
        stage_factory.remove_stage(stage)


def test_factory_stage_remove(stage_factory: StageFactoryBase):
    assert not stage_factory.stages
    initial_stages_n = len(stage_factory.stage_names)

    s, s2, _, _ = create_4_stages(stage_factory)
    stage_factory.add_stage(s)
    stage_factory.add_stage(s2)

    assert len(stage_factory.stage_names) == initial_stages_n + 2

    stage_factory.test_changes_count = 0
    assert stage_factory.remove_stage(s2)

    assert stage_factory.test_changes_count
    assert s in stage_factory.stages
    assert s2 not in stage_factory.stages
    assert len(stage_factory.stages) == 1
    assert s.name == 's'
    assert s2.name == 's2'

    assert stage_factory.stage_names['s'] is s
    assert 's2' not in stage_factory.stage_names
    assert len(stage_factory.stage_names) == initial_stages_n + 1

    stage_factory.test_changes_count = 0
    s2.name = 's'

    assert not stage_factory.test_changes_count
    assert s.name == 's'
    assert s2.name == 's'

    stage_factory.test_changes_count = 0
    assert stage_factory.remove_stage(s)

    assert stage_factory.test_changes_count
    assert s not in stage_factory.stages
    assert s2 not in stage_factory.stages
    assert not stage_factory.stages
    assert s.name == 's'
    assert s2.name == 's'

    assert 's' not in stage_factory.stage_names
    assert 's2' not in stage_factory.stage_names

    assert len(stage_factory.stage_names) == initial_stages_n


def test_clear_stages(stage_factory: StageFactoryBase):
    assert not len(stage_factory.stages)
    n = len(stage_factory.stage_names)

    stage = SerialAllStage(stage_factory=stage_factory, show_in_gui=False)
    stage.create_stage()
    stage = stage.stage
    stage2 = SerialAllStage(stage_factory=stage_factory, show_in_gui=False)
    stage2.create_stage()
    stage2 = stage2.stage
    stage_factory.add_stage(stage)
    stage_factory.add_stage(stage2)

    assert stage_factory.stages == [stage, stage2]

    stage_factory.test_changes_count = 0
    stage_factory.clear_stages()
    assert not stage_factory.stages
    assert len(stage_factory.stage_names) == n
    assert stage_factory.test_changes_count


def test_recover_stages(stage_factory: StageFactoryBase):
    s1, s2, s3, s4 = create_4_stages(stage_factory)

    stage_factory.add_stage(s1)
    stage_factory.add_stage(s2)
    stage_factory.add_stage(s3)
    stage_factory.add_stage(s4)

    stages = stage_factory.save_stages()
    assert len(stages) == 4

    recovered_stages, name_mapping = stage_factory.recover_stages(
        stages, {}, {})
    assert len(recovered_stages) == 4
    assert len(name_mapping) == 4

    # name cannot be the same because both old and new are in the factory
    for s_name in ('s', 's2', 's3', 's4'):
        assert s_name in name_mapping
        assert name_mapping[s_name] != s_name
        assert s_name in stage_factory.stage_names
        assert name_mapping[s_name] in stage_factory.stage_names

        original_s = stage_factory.stage_names[s_name]
        new_s = stage_factory.stage_names[name_mapping[s_name]]
        assert_stages_same(original_s, new_s, compare_name=False)
        assert original_s.name != new_s.name
        assert new_s.name.startswith(original_s.name)


def test_make_function(stage_factory: StageFactoryBase):
    stages = s1, s2, s3, s4 = create_4_stages(stage_factory)

    states = [s.get_state() for s in stages]
    new_stages = [stage_factory.make_stage(state) for state in states]
    assert len(new_stages) == len(stages)

    for new_stage, s in zip(new_stages, stages):
        assert_stages_same(new_stage, s, compare_name=False)
        assert s.name != new_stage.name

    # close should make them identical in all ways
    new_stages = [
        stage_factory.make_stage(state, clone=True) for state in states]
    assert len(new_stages) == len(stages)

    for new_stage, s in zip(new_stages, stages):
        assert_stages_same(new_stage, s, compare_name=True)

    # provide instances
    new_stages = [
        stage_factory.make_stage(
            state, instance=CeedStage(
                stage_factory=stage_factory,
                function_factory=stage_factory.function_factory,
                shape_factory=stage_factory.shape_factory), clone=True)
        for state in states]
    assert len(new_stages) == len(stages)

    for new_stage, s in zip(new_stages, stages):
        assert_stages_same(new_stage, s, compare_name=True)


def test_can_other_stage_be_added(stage_factory: StageFactoryBase):
    root, g1, g2, s1, s2, s3, s4, s5, s6 = create_recursive_stages(
        stage_factory)

    assert root.stage.can_other_stage_be_added(g1.stage)
    assert root.stage.can_other_stage_be_added(g2.stage)
    assert not root.stage.can_other_stage_be_added(root.stage)

    assert not g1.stage.can_other_stage_be_added(root.stage)
    assert g1.stage.can_other_stage_be_added(g2.stage)
    assert not g1.stage.can_other_stage_be_added(g1.stage)

    assert not g2.stage.can_other_stage_be_added(root.stage)
    assert g2.stage.can_other_stage_be_added(g1.stage)
    assert not g2.stage.can_other_stage_be_added(g2.stage)


def test_stage_ref(stage_factory: StageFactoryBase):
    s = make_stage(stage_factory)
    s.name = 'me stage'
    s2 = make_stage(stage_factory)

    stage_factory.add_stage(s)

    ref1 = stage_factory.get_stage_ref(name='me stage')
    ref2 = stage_factory.get_stage_ref(stage=s2)

    assert ref1.stage is s
    assert ref2.stage is s2
    assert s.has_ref
    assert s2.has_ref
    assert s in stage_factory._stage_ref
    assert s2 in stage_factory._stage_ref

    stage_factory.return_stage_ref(ref1)
    assert ref2.stage is s2
    assert not s.has_ref
    assert s2.has_ref
    assert s not in stage_factory._stage_ref
    assert s2 in stage_factory._stage_ref

    stage_factory.return_stage_ref(ref2)
    assert not s.has_ref
    assert not s2.has_ref
    assert s not in stage_factory._stage_ref
    assert s2 not in stage_factory._stage_ref


def test_return_not_added_stage_ref(stage_factory: StageFactoryBase):
    from ceed.stage import CeedStageRef
    s = make_stage(stage_factory)
    ref = CeedStageRef(
        stage_factory=stage_factory,
        function_factory=stage_factory.function_factory,
        shape_factory=stage_factory.shape_factory, stage=s)

    with pytest.raises(ValueError):
        stage_factory.return_stage_ref(ref)


def test_remove_stage_with_ref(stage_factory: StageFactoryBase):
    s = make_stage(stage_factory)
    s2 = make_stage(stage_factory)
    s3 = make_stage(stage_factory)
    s.name = 'stage'
    s2.name = 'stage2'
    s3.name = 'stage3'

    stage_factory.add_stage(s)
    stage_factory.add_stage(s2)
    stage_factory.add_stage(s3)

    assert stage_factory.stage_names['stage'] is s
    assert s in stage_factory.stages
    assert stage_factory.stage_names['stage2'] is s2
    assert s2 in stage_factory.stages
    assert stage_factory.stage_names['stage3'] is s3
    assert s3 in stage_factory.stages

    ref = stage_factory.get_stage_ref(name='stage')
    ref3 = stage_factory.get_stage_ref(name='stage3')
    assert not stage_factory.remove_stage(s)

    assert ref.stage is s
    assert s.has_ref
    assert stage_factory.stage_names['stage'] is s
    assert s in stage_factory.stages
    assert stage_factory.stage_names['stage2'] is s2
    assert s2 in stage_factory.stages
    assert stage_factory.stage_names['stage3'] is s3
    assert s3 in stage_factory.stages

    assert stage_factory.remove_stage(s2)

    assert stage_factory.stage_names['stage'] is s
    assert s in stage_factory.stages
    assert 'stage2' not in stage_factory.stage_names
    assert s2 not in stage_factory.stages
    assert stage_factory.stage_names['stage3'] is s3
    assert s3 in stage_factory.stages

    assert not stage_factory.remove_stage(s3)

    assert ref3.stage is s3
    assert s3.has_ref
    assert stage_factory.stage_names['stage'] is s
    assert s in stage_factory.stages
    assert stage_factory.stage_names['stage3'] is s3
    assert s3 in stage_factory.stages

    assert stage_factory.remove_stage(s3, force=True)

    assert ref3.stage is s3
    assert s3.has_ref
    assert stage_factory.stage_names['stage'] is s
    assert s in stage_factory.stages
    assert 'stage3' not in stage_factory.stage_names
    assert s3 not in stage_factory.stages

    assert not stage_factory.remove_stage(s)

    assert ref.stage is s
    assert s.has_ref
    assert stage_factory.stage_names['stage'] is s
    assert s in stage_factory.stages

    stage_factory.return_stage_ref(ref)
    assert not s.has_ref

    assert stage_factory.remove_stage(s)

    assert 'stage' not in stage_factory.stage_names
    assert s not in stage_factory.stages

    stage_factory.return_stage_ref(ref3)
    assert not s3.has_ref


def test_clear_stages_with_ref(stage_factory: StageFactoryBase):
    s = make_stage(stage_factory)
    s2 = make_stage(stage_factory)
    s.name = 'stage'
    s2.name = 'stage2'

    stage_factory.add_stage(s)
    stage_factory.add_stage(s2)

    assert stage_factory.stage_names['stage'] is s
    assert s in stage_factory.stages
    assert stage_factory.stage_names['stage2'] is s2
    assert s2 in stage_factory.stages

    ref = stage_factory.get_stage_ref(name='stage')
    stage_factory.clear_stages()

    # s should not have been removed, but s2 was removed
    assert ref.stage is s
    assert s.has_ref
    assert stage_factory.stage_names['stage'] is s
    assert s in stage_factory.stages
    assert 'stage2' not in stage_factory.stage_names
    assert s2 not in stage_factory.stages

    stage_factory.clear_stages(force=True)

    assert ref.stage is s
    assert s.has_ref
    assert 'stage' not in stage_factory.stage_names
    assert s not in stage_factory.stages

    stage_factory.return_stage_ref(ref)
    assert not s.has_ref


def test_recover_ref_stages(stage_factory: StageFactoryBase):
    s1 = make_stage(
        stage_factory, name='stage1', order='parallel', complete_on='any',
        color_r=True)
    s2 = make_stage(
        stage_factory, name='stage2', order='serial', complete_on='all',
        color_g=True)
    s3 = make_stage(
        stage_factory, name='stage3', order='parallel', complete_on='all',
        color_b=True)
    s4 = make_stage(
        stage_factory, name='stage4', order='serial', complete_on='any',
        color_g=True)
    g = make_stage(
        stage_factory, name='grouped', order='parallel', complete_on='any',
        color_b=True)

    stage_factory.add_stage(s1)
    stage_factory.add_stage(s2)
    stage_factory.add_stage(s3)
    stage_factory.add_stage(s4)
    stage_factory.add_stage(g)

    g.add_stage(stage_factory.get_stage_ref(name='stage1'))
    g.add_stage(stage_factory.get_stage_ref(name='stage2'))
    g.add_stage(stage_factory.get_stage_ref(name='stage3'))
    g.add_stage(stage_factory.get_stage_ref(name='stage4'))

    stages = stage_factory.save_stages()
    assert len(stages) == 5

    func_name_map = {}
    old_name_to_shape_map = {}
    recovered_stages, name_mapping = stage_factory.recover_stages(
        stages, func_name_map, old_name_to_shape_map)
    assert len(recovered_stages) == 5
    assert len(name_mapping) == 5

    for s_name in ('stage1', 'stage2', 'stage3', 'stage4', 'grouped'):
        assert s_name in name_mapping
        assert name_mapping[s_name] != s_name
        assert s_name in stage_factory.stage_names
        assert name_mapping[s_name] in stage_factory.stage_names

        original_s = stage_factory.stage_names[s_name]
        new_s = stage_factory.stage_names[name_mapping[s_name]]
        assert_stages_same(original_s, new_s, compare_name=False)
        assert original_s.name != new_s.name
        assert new_s.name.startswith(original_s.name)

    new_g: CeedStage = stage_factory.stage_names[name_mapping['grouped']]
    assert len(new_g.stages) == 4

    stage: CeedStageRef
    for stage, name in zip(
            new_g.stages, ('stage1', 'stage2', 'stage3', 'stage4')):
        assert isinstance(stage, CeedStageRef)
        assert stage.stage is stage_factory.stage_names[name_mapping[name]]


def test_get_stage_ref(stage_factory: StageFactoryBase):
    g1 = make_stage(stage_factory)
    g2 = make_stage(stage_factory)
    ref_g2 = stage_factory.get_stage_ref(stage=g2)
    g1.add_stage(ref_g2)

    s1 = make_stage(stage_factory, order='parallel')
    g2.add_stage(s1)
    s2 = make_stage(stage_factory, complete_on='any')
    g1.add_stage(s2)

    assert list(g1.get_stages()) == [g1, g2, s1, s2]
    assert list(g1.get_stages(step_into_ref=False)) == [g1, ref_g2, s2]


def test_can_other_stage_be_added_ref(stage_factory: StageFactoryBase):
    g1 = make_stage(stage_factory)
    g2 = make_stage(stage_factory)
    ref_g2 = stage_factory.get_stage_ref(stage=g2)
    g1.add_stage(ref_g2)

    s1 = make_stage(stage_factory, order='parallel')
    g2.add_stage(s1)
    s2 = make_stage(stage_factory, complete_on='any')
    g1.add_stage(s2)

    g3 = make_stage(stage_factory)
    g1.add_stage(g3)
    s4 = make_stage(stage_factory, order='parallel', complete_on='all')
    g3.add_stage(s4)

    assert g1.can_other_stage_be_added(g2)
    assert g1.can_other_stage_be_added(ref_g2)
    assert g1.can_other_stage_be_added(g3)
    assert not g1.can_other_stage_be_added(g1)

    assert not g2.can_other_stage_be_added(g1)
    assert g2.can_other_stage_be_added(g3)
    assert not g2.can_other_stage_be_added(g2)
    assert not g2.can_other_stage_be_added(ref_g2)

    assert not g3.can_other_stage_be_added(g1)
    assert g3.can_other_stage_be_added(g2)
    assert g3.can_other_stage_be_added(ref_g2)
    assert not g3.can_other_stage_be_added(g3)


def test_expand_ref_stages(stage_factory: StageFactoryBase):
    g1 = make_stage(stage_factory, order='parallel')

    g2 = make_stage(stage_factory, order='parallel')
    ref_g2 = stage_factory.get_stage_ref(stage=g2)
    g1.add_stage(ref_g2)

    s = make_stage(
        stage_factory, order='parallel', color_r=True, complete_on='any')
    g2.add_stage(s)
    s1 = make_stage(
        stage_factory, order='parallel', color_b=True, complete_on='any')
    g2.add_stage(s1)

    s2 = make_stage(
        stage_factory, order='parallel', color_r=False, complete_on='all')
    g1.add_stage(s2)
    s3 = make_stage(
        stage_factory, order='serial', color_r=True, color_g=True,
        complete_on='any')
    ref_f3 = stage_factory.get_stage_ref(stage=s3)
    g1.add_stage(ref_f3)

    g3 = make_stage(stage_factory, order='parallel')
    g1.add_stage(g3)
    s4 = make_stage(
        stage_factory, order='parallel', color_b=True, color_r=True,
        complete_on='any')
    ref_f4 = stage_factory.get_stage_ref(stage=s4)
    g3.add_stage(ref_f4)
    s5 = make_stage(
        stage_factory, order='parallel', color_b=True, color_g=True,
        complete_on='all')
    g3.add_stage(s5)

    assert list(g1.get_stages(step_into_ref=False)) == \
        [g1, ref_g2, s2, ref_f3, g3, ref_f4, s5]
    assert list(g1.get_stages(step_into_ref=True)) == \
        [g1, g2, s, s1, s2, s3, g3, s4, s5]

    g1_copy = g1.copy_expand_ref()
    # the copy shouldn't have any refs
    assert len(list(g1_copy.get_stages(step_into_ref=False))) == \
        len(list(g1.get_stages(step_into_ref=True)))

    for original_f, new_f in zip(
            g1.get_stages(step_into_ref=True),
            g1_copy.get_stages(step_into_ref=False)):
        assert_stages_same(original_f, new_f, compare_name=False)


def test_copy_stages(stage_factory: StageFactoryBase):
    s1 = make_stage(
        stage_factory, name='stage1', order='parallel', complete_on='any',
        color_r=True)
    s2 = make_stage(
        stage_factory, name='stage2', order='serial', complete_on='all',
        color_g=True)
    s3 = make_stage(
        stage_factory, name='stage3', order='parallel', complete_on='all',
        color_b=True)
    s4 = make_stage(
        stage_factory, name='stage4', order='serial', complete_on='any',
        color_g=True)
    g = make_stage(
        stage_factory, name='grouped', order='parallel', complete_on='any',
        color_b=True)

    stage_factory.add_stage(s1)
    stage_factory.add_stage(s2)
    stage_factory.add_stage(s3)
    stage_factory.add_stage(s4)

    g.add_stage(stage_factory.get_stage_ref(name='stage1'))
    g.add_stage(stage_factory.get_stage_ref(name='stage2'))
    g.add_stage(stage_factory.get_stage_ref(name='stage3'))
    g.add_stage(stage_factory.get_stage_ref(name='stage4'))

    for stage in (s1, s2, s3, s4):
        stage_copy = deepcopy(stage)
        assert stage is not stage_copy
        assert isinstance(stage_copy, stage.__class__)

        assert_stages_same(stage, stage_copy, compare_name=False)

    stage_copy = deepcopy(g)
    assert len(stage_copy.stages) == 4
    for new_s, original_s in zip(stage_copy.stages, g.stages):
        assert new_s is not original_s
        assert isinstance(new_s, CeedStageRef)
        assert isinstance(original_s, CeedStageRef)
        assert new_s.stage is original_s.stage


def test_replace_ref_stage_with_source_stages(
        stage_factory: StageFactoryBase):
    g1 = make_stage(stage_factory, order='parallel', name='g1')

    g2 = make_stage(stage_factory, order='parallel', name='g2')
    ref_g2 = stage_factory.get_stage_ref(stage=g2)
    g1.add_stage(ref_g2)

    s = make_stage(
        stage_factory, order='parallel', color_r=True, complete_on='any')
    g2.add_stage(s)
    s1 = make_stage(
        stage_factory, order='parallel', color_b=True, complete_on='any',
        name='s1')
    stage_factory.add_stage(s1)
    ref_s1 = stage_factory.get_stage_ref(stage=s1)
    g2.add_stage(ref_s1)

    s2 = make_stage(
        stage_factory, order='parallel', color_r=False, complete_on='all')
    g1.add_stage(s2)
    s3 = make_stage(
        stage_factory, order='serial', color_r=True, color_g=True,
        complete_on='any', name='s3')
    stage_factory.add_stage(s3)
    ref_s3 = stage_factory.get_stage_ref(stage=s3)
    g1.add_stage(ref_s3)

    with pytest.raises(ValueError):
        g1.replace_ref_stage_with_source(s2)

    with pytest.raises(ValueError):
        g1.replace_ref_stage_with_source(ref_s1)

    s3_new, i = g1.replace_ref_stage_with_source(ref_s3)

    assert i == 2
    assert ref_s3 not in g1.stages
    assert s3 not in g1.stages
    assert not isinstance(s3_new, CeedStageRef)
    assert isinstance(s3_new, s3.__class__)
    assert g1.stages[i] is s3_new

    assert_stages_same(s3, s3_new, compare_name=False)

    g2_new: CeedStage
    g2_new, i = g1.replace_ref_stage_with_source(ref_g2)

    assert i == 0
    assert ref_g2 not in g1.stages
    assert g2 not in g1.stages
    assert not isinstance(g2_new, CeedStageRef)
    assert isinstance(g2_new, CeedStage)
    assert g1.stages[i] is g2_new

    assert len(g2_new.stages) == 2
    assert g2_new.stages[0] is not g2.stages[0]
    assert g2_new.stages[1] is not g2.stages[1]
    assert isinstance(g2_new.stages[0], s.__class__)
    assert isinstance(g2_new.stages[1], ref_s1.__class__)
    assert isinstance(g2_new.stages[1], CeedStageRef)

    assert_stages_same(s, g2_new.stages[0], compare_name=False)
    assert g2_new.stages[1].stage is s1


def test_group_remove_stage(stage_factory: StageFactoryBase):
    g1 = make_stage(stage_factory)
    g2 = make_stage(stage_factory)
    ref_g2 = stage_factory.get_stage_ref(stage=g2)
    g1.add_stage(ref_g2)

    s = make_stage(stage_factory, order='parallel')
    g2.add_stage(s)
    s1 = make_stage(stage_factory, order='parallel')
    g2.add_stage(s1)
    s2 = make_stage(stage_factory, complete_on='any')
    g1.add_stage(s2)

    assert list(g1.get_stages(step_into_ref=False)) == [g1, ref_g2, s2]

    g1.remove_stage(s2)
    assert list(g1.get_stages(step_into_ref=False)) == [g1, ref_g2]

    g2.remove_stage(s)
    assert list(g1.get_stages(step_into_ref=False)) == [g1, ref_g2]

    g1.remove_stage(ref_g2)
    assert list(g1.get_stages(step_into_ref=False)) == [g1, ]


@pytest.mark.parametrize('pre_compute', [True, False])
def test_simple_stage_intensity(stage_factory: StageFactoryBase, pre_compute):
    from ceed.function.plugin import LinearFunc
    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)

    shape2 = EllipseShapeP2(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)

    f: LinearFunc = LinearFunc(
        function_factory=stage_factory.function_factory, b=0, m=.1, duration=5)

    stage = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=True)
    stage_factory.add_stage(stage)
    stage.add_func(f)
    stage.add_shape(shape.shape)
    stage.add_shape(shape2.shape)

    values, n = get_stage_time_intensity(
        stage_factory, stage.name, 10, pre_compute=pre_compute)
    assert n == 5 * 10
    assert len(values) == 2
    colors = values[shape.name]
    colors2 = values[shape2.name]
    assert len(colors) == 10 * 5

    for i, (r, g, b, a) in enumerate(colors):
        assert math.isclose(r, i / 10 * .1)
        assert math.isclose(b, i / 10 * .1)
        assert g == 0

    for i, (r, g, b, a) in enumerate(colors2):
        assert math.isclose(r, i / 10 * .1)
        assert math.isclose(b, i / 10 * .1)
        assert g == 0


@pytest.mark.parametrize('pre_compute', [True, False])
def test_recursive_stage_intensity(
        stage_factory: StageFactoryBase, pre_compute):
    root, g1, g2, s1, s2, s3, s4, s5, s6 = create_recursive_stages(
        stage_factory)

    from ceed.function.plugin import LinearFunc
    for i, stage in enumerate((s1, s2)):
        stage.stage.add_func(LinearFunc(
            function_factory=stage_factory.function_factory, b=0, m=.1,
            duration=(i + 1) * 5))

    shape = CircleShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)

    shape2 = CircleShapeP1Internal(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    s1.stage.add_shape(shape.shape)
    s2.stage.add_shape(shape2.shape)

    values, n = get_stage_time_intensity(
        stage_factory, root.name, 10, pre_compute=pre_compute)
    assert n == 10 * 10
    assert len(values) == 2
    colors = values[shape.name]
    colors2 = values[shape2.name]
    assert len(colors) == n

    for i, (r, g, b, a) in enumerate(colors):
        if i < 5 * 10:
            assert math.isclose(r, i / 10 * .1) if s2.color_r else r == 0
            assert math.isclose(g, i / 10 * .1) if s2.color_g else g == 0
            assert math.isclose(b, i / 10 * .1) if s2.color_b else b == 0
        else:
            assert r == 0
            assert g == 0
            assert b == 0

    for i, (r, g, b, a) in enumerate(colors2):
        assert math.isclose(r, i / 10 * .1) if s2.color_r else r == 0
        assert math.isclose(g, i / 10 * .1) if s2.color_g else g == 0
        assert math.isclose(b, i / 10 * .1) if s2.color_b else b == 0


def assert_recursive_stages_intensity(
        s1, s2, s3, s4, s5, s6, values, n, shape, shape2, rate=10):
    assert n == rate * (2 * 10 + 5 + 2 * 10 + 3 * 5)
    assert len(values) == 2
    colors = values[shape.name]
    colors2 = values[shape2.name]
    assert len(colors) == n

    # first shape is in s1, s4, s5
    for s, start, e in [
            (s1, 0, 5), (s1, 10, 15), (s4, 25, 35), (s4, 35, 45), (s5, 45, 50),
            (s5, 50, 55), (s5, 55, 60)]:
        for i in range(start * 10, e * 10):
            r, g, b, a = colors[i]

            i -= start * rate  # convert zero offset
            assert math.isclose(r, i / 10 * .1) if s.color_r else r == 0
            assert math.isclose(g, i / 10 * .1) if s.color_g else g == 0
            assert math.isclose(b, i / 10 * .1) if s.color_b else b == 0

    # shape is not active during s2, s3
    for start, e in [(5, 10), (15, 25)]:
        for i in range(start * 10, e * 10):
            r, g, b, a = colors[i]
            assert r == 0
            assert g == 0
            assert b == 0

    # second shape is in s2, s3, s6
    for s, start, e in [
            (s2, 0, 10), (s2, 10, 20), (s3, 20, 25), (s6, 45, 50),
            (s6, 50, 55), (s6, 55, 60)]:
        for i in range(start * 10, e * 10):
            r, g, b, a = colors2[i]

            i -= start * rate  # convert zero offset
            assert math.isclose(r, i / 10 * .1) if s.color_r else r == 0
            assert math.isclose(g, i / 10 * .1) if s.color_g else g == 0
            assert math.isclose(b, i / 10 * .1) if s.color_b else b == 0

    # shape2 is not active during s4, s5
    for start, e in [(25, 45), ]:
        for i in range(start * 10, e * 10):
            r, g, b, a = colors2[i]
            assert r == 0
            assert g == 0
            assert b == 0


@pytest.mark.parametrize('pre_compute', [True, False])
def test_recursive_full_stage_with_loop_intensity(
        stage_factory: StageFactoryBase, pre_compute):
    root, g1, g2, s1, s2, s3, s4, s5, s6 = create_recursive_stages(
        stage_factory)

    from ceed.function.plugin import LinearFunc
    for i, stage in enumerate((s1, s2, s3, s4, s5, s6)):
        stage.stage.add_func(LinearFunc(
            function_factory=stage_factory.function_factory, b=0, m=.1,
            duration=(i % 2 + 1) * 5))

    g1.stage.loop = 2
    s4.stage.loop = 2
    g2.stage.loop = 3

    shape = CircleShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)

    shape2 = CircleShapeP1Internal(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    s1.stage.add_shape(shape.shape)
    s4.stage.add_shape(shape.shape)
    s5.stage.add_shape(shape.shape)
    s2.stage.add_shape(shape2.shape)
    s3.stage.add_shape(shape2.shape)
    s6.stage.add_shape(shape2.shape)

    values, n = get_stage_time_intensity(
        stage_factory, root.name, 10, pre_compute=pre_compute)
    assert_recursive_stages_intensity(
        s1, s2, s3, s4, s5, s6, values, n, shape, shape2)


@pytest.mark.parametrize('pre_compute', [True, False])
def test_single_frame_stage_intensity(
        stage_factory: StageFactoryBase, pre_compute):
    from ceed.function.plugin import ConstFunc

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)

    f: FuncGroup = FuncGroup(
        function_factory=stage_factory.function_factory, timebase_numerator=1,
        timebase_denominator=120, loop=3)
    # give it a float duration to see if it can handle a integer float
    f1 = ConstFunc(
        function_factory=stage_factory.function_factory, a=0, duration=1.)
    f2 = ConstFunc(
        function_factory=stage_factory.function_factory, a=1, duration=1)
    f.add_func(f1)
    f.add_func(f2)

    stage = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=True)
    stage_factory.add_stage(stage)
    stage.add_func(f)
    stage.add_shape(shape.shape)

    values, n = get_stage_time_intensity(
        stage_factory, stage.name, 120, pre_compute=pre_compute)
    assert n == 3 * 2
    assert len(values) == 1
    colors = values[shape.name]
    assert len(colors) == 3 * 2

    for i, (r, g, b, a) in enumerate(colors):
        assert math.isclose(r, i % 2)
        assert math.isclose(b, i % 2)
        assert g == 0


def test_sample_ref_function_stage(stage_factory: StageFactoryBase):
    function_factory = stage_factory.function_factory
    f: FuncBase = function_factory.get('LinearFunc')(
        function_factory=function_factory)
    function_factory.add_func(f)

    counter_m = [0]
    counter_b = [0]
    cls = register_callback_distribution(function_factory, counter_m, 0)
    f.noisy_parameters['m'] = cls()
    cls = register_callback_distribution(function_factory, counter_b, 1)
    f.noisy_parameters['b'] = cls(lock_after_forked=True)

    ref = function_factory.get_func_ref(func=f)

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)

    root = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=True)
    stage = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=True)
    stage.add_func(ref)
    stage.add_shape(shape.shape)
    root.add_stage(stage)

    copied = root.copy_and_resample()
    assert counter_m[0] == 2
    assert counter_b[0] == 1

    assert f.b == copied.stages[0].functions[0].b


@pytest.mark.parametrize('pre_compute', [True, False])
@pytest.mark.parametrize('rate', [60., 120., 100.])
@pytest.mark.parametrize('duration', [
    (.5, .5, .5), (.51, .5, .49), (.1, .1, .1), (.11, .33, .59),
    (31 / 60, 5 / 12, 1441 / 720)])
def test_stage_func_float_duration(
        stage_factory: StageFactoryBase, rate, duration, pre_compute):
    function_factory = stage_factory.function_factory

    root = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=False)
    stage_factory.add_stage(root)

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    root.add_shape(shape.shape)

    ConstFunc = function_factory.get('ConstFunc')
    child_a = ConstFunc(
        function_factory=function_factory, loop=4, duration=duration[0], a=.1)
    child_b = ConstFunc(
        function_factory=function_factory, loop=3, duration=duration[1], a=.2)
    child_c = ConstFunc(
        function_factory=function_factory, loop=5, duration=duration[2], a=.3)
    root.add_func(child_a)
    root.add_func(child_b)
    root.add_func(child_c)

    values, n = get_stage_time_intensity(
        stage_factory, root.name, rate, pre_compute=pre_compute)
    expected = int(
        rate * (4 * duration[0] + 3 * duration[1] + 5 * duration[2]))
    assert expected - 1 <= n <= expected + 1

    counts = collapse_list_to_counts(values[shape.name])
    loops = [4, 3, 5]
    a = [.1, .2, .3]
    for i in range(3):
        a_val, count = counts[i]
        assert a_val == (a[i], 0, 0, 1)
        assert count - 1 <= round(loops[i] * duration[i] * rate) <= count + 1


@pytest.mark.parametrize('pre_compute', [True, False])
def test_stage_func_float_duration_epsilon(
        stage_factory: StageFactoryBase, pre_compute):
    # before isclose was used, this would error with floating point equality
    # issues
    function_factory = stage_factory.function_factory

    root = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=False)
    stage_factory.add_stage(root)

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    root.add_shape(shape.shape)

    ConstFunc = function_factory.get('ConstFunc')
    child_a = ConstFunc(
        function_factory=function_factory, duration=1., a=.1)
    child_b = ConstFunc(
        function_factory=function_factory, duration=3 * .1, a=.2)
    child_c = ConstFunc(
        function_factory=function_factory, duration=3 * .1, a=.3)
    child_d = ConstFunc(
        function_factory=function_factory, duration=2., a=.4)
    root.add_func(child_a)
    root.add_func(child_b)
    root.add_func(child_c)
    root.add_func(child_d)

    values, n = get_stage_time_intensity(
        stage_factory, root.name, 60., pre_compute=pre_compute)
    expected = int(60. * 3.6)
    assert expected - 1 <= n <= expected + 1

    counts = collapse_list_to_counts(values[shape.name])
    duration = [1., 3 * .1, 3 * .1, 2.]
    a = [.1, .2, .3, .4]
    for i in range(4):
        a_val, count = counts[i]
        assert a_val == (a[i], 0, 0, 1)
        assert count - 1 <= round(duration[i] * 60) <= count + 1


@pytest.mark.parametrize('pre_compute', [True, False])
def test_stage_func_clip_range(stage_factory: StageFactoryBase, pre_compute):
    function_factory = stage_factory.function_factory

    root = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=False)
    stage_factory.add_stage(root)

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    root.add_shape(shape.shape)

    ConstFunc = function_factory.get('ConstFunc')
    child_a = ConstFunc(
        function_factory=function_factory, duration=1., a=-.1)
    child_b = ConstFunc(
        function_factory=function_factory, duration=3 * .1, a=.2)
    child_c = ConstFunc(
        function_factory=function_factory, duration=3 * .1, a=.3)
    child_d = ConstFunc(
        function_factory=function_factory, duration=2., a=1.4)
    root.add_func(child_a)
    root.add_func(child_b)
    root.add_func(child_c)
    root.add_func(child_d)

    values, n = get_stage_time_intensity(
        stage_factory, root.name, 60., pre_compute=pre_compute)
    intensities = {v[0] for v in values[shape.name]}
    assert intensities == {0, .2, .3, 1}


@pytest.mark.parametrize('pre_compute', [True, False])
def test_stage_func_tree_init(stage_factory: StageFactoryBase, pre_compute):
    from ceed.function.plugin import ConstFunc
    tree_counter = defaultdict(int)
    init_counter = defaultdict(int)
    init_loop_counter = defaultdict(int)

    class CounterMixIn:
        def init_func_tree(self, *args, **kwargs):
            tree_counter[self.name] += 1
            super().init_func_tree(*args, **kwargs)

        def init_func(self, *args, **kwargs):
            init_counter[self.name] += 1
            super().init_func(*args, **kwargs)

        def init_loop_iteration(self, *args, **kwargs):
            init_loop_counter[self.name] += 1
            super().init_loop_iteration(*args, **kwargs)

    class GroupHook(CounterMixIn, FuncGroup):
        pass

    class ConstHook(CounterMixIn, ConstFunc):
        pass

    class HookStage(CeedStage):
        def init_stage_tree(self, *args, **kwargs):
            tree_counter[self.name] += 1
            super().init_stage_tree(*args, **kwargs)

    root = HookStage(
        stage_factory=stage_factory,
        function_factory=stage_factory.function_factory,
        shape_factory=stage_factory.shape_factory, name='root_stage')
    stage_factory.add_stage(root)
    s2 = HookStage(
        stage_factory=stage_factory,
        function_factory=stage_factory.function_factory,
        shape_factory=stage_factory.shape_factory, name='s2')
    root.add_stage(s2)

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    shape2 = EllipseShapeP2(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    root.add_shape(shape.shape)
    s2.add_shape(shape2.shape)

    root.add_func(ConstHook(
        function_factory=stage_factory.function_factory, name='f_root', loop=3,
        duration=1)
    )
    s2.add_func(ConstHook(
        function_factory=stage_factory.function_factory, name='f2_child',
        loop=2, duration=1)
    )

    g = GroupHook(
        function_factory=stage_factory.function_factory, name='root', loop=3)
    g2 = GroupHook(
        function_factory=stage_factory.function_factory, name='g_child', loop=4)
    f = ConstHook(
        function_factory=stage_factory.function_factory, name='gf_child',
        loop=5, duration=1)
    g2.add_func(f)
    g.add_func(g2)
    s2.add_func(g)

    get_stage_time_intensity(
        stage_factory, root.name, 10, pre_compute=pre_compute)

    for name in ('f_root', 'f2_child', 'root', 'g_child', 'gf_child',
                 'root_stage', 's2'):
        assert tree_counter[name] == 1

    for name in ('f_root', 'f2_child', 'root'):
        assert init_counter[name] == 1
    assert init_counter['g_child'] == 3
    assert init_counter['gf_child'] == 12

    assert init_loop_counter['f_root'] == 2
    assert init_loop_counter['f2_child'] == 1
    assert init_loop_counter['root'] == 2
    assert init_loop_counter['g_child'] == 9
    assert init_loop_counter['gf_child'] == 48


def test_stage_func_resample(stage_factory: StageFactoryBase):
    function_factory = stage_factory.function_factory

    stage = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=False)
    stage_factory.add_stage(stage)

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    stage.add_shape(shape.shape)

    LinearFunc = function_factory.get('LinearFunc')
    f = LinearFunc(
        function_factory=function_factory, duration=1., loop=4)
    stage.add_func(f)

    cls = function_factory.param_noise_factory.get_cls('UniformNoise')
    f.noisy_parameters['m'] = cls(sample_each_loop=False)
    f.noisy_parameters['b'] = cls(sample_each_loop=True)

    stage_copy = stage.copy_and_resample()
    f_copy = stage_copy.functions[0]
    assert 'm' not in f_copy.noisy_parameter_samples
    assert len(f_copy.noisy_parameter_samples['b']) == 4


@pytest.mark.parametrize(
    'disable_pre_compute', product([True, False], repeat=9))
def test_recursive_stage_pre_compute(
        stage_factory: StageFactoryBase, disable_pre_compute):
    stages = root, g1, g2, s1, s2, s3, s4, s5, s6 = create_recursive_stages(
        stage_factory)

    from ceed.function.plugin import LinearFunc
    for i, stage in enumerate((s1, s2, s3, s4, s5, s6)):
        stage.stage.add_func(LinearFunc(
            function_factory=stage_factory.function_factory, b=0, m=.1,
            duration=(i % 2 + 1) * 5))

    g1.stage.loop = 2
    s4.stage.loop = 2
    g2.stage.loop = 3

    shape = CircleShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)

    shape2 = CircleShapeP1Internal(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    s1.stage.add_shape(shape.shape)
    s4.stage.add_shape(shape.shape)
    s5.stage.add_shape(shape.shape)
    s2.stage.add_shape(shape2.shape)
    s3.stage.add_shape(shape2.shape)
    s6.stage.add_shape(shape2.shape)

    for disable, s in zip(disable_pre_compute, stages):
        s.stage.disable_pre_compute = disable

    values, n = get_stage_time_intensity(
        stage_factory, root.name, 10, pre_compute=True)
    assert_recursive_stages_intensity(
        s1, s2, s3, s4, s5, s6, values, n, shape, shape2)

    root, g1, g2, s1, s2, s3, s4, s5, s6 = [s.stage for s in stages]
    for s in (s1, s2, s3, s4, s5, s6):
        assert s.can_pre_compute != s.disable_pre_compute
    assert g1.can_pre_compute == (
        s1.can_pre_compute and s2.can_pre_compute and
        not g1.disable_pre_compute)
    assert g2.can_pre_compute == (
        s5.can_pre_compute and s6.can_pre_compute and
        not g2.disable_pre_compute)
    assert root.can_pre_compute == (
        g1.can_pre_compute and g2.can_pre_compute and s3.can_pre_compute and
        s4.can_pre_compute and not root.disable_pre_compute)

    for s, parent in [
            (s1, g1), (s2, g1), (s5, g2), (s6, g2), (s3, root), (s4, root),
            (g1, root), (g2, root)]:
        if s.can_pre_compute and not parent.can_pre_compute:
            assert s.runtime_stage is not None
        else:
            assert s.runtime_stage is None

    if root.can_pre_compute:
        assert root.runtime_stage is not None
    else:
        assert root.runtime_stage is None

    for s in (s1, s2, s3, s4, s5, s6):
        if not s.can_pre_compute and not s.disable_pre_compute:
            for f, items, t_end in s.runtime_functions:
                assert f is None
                assert items
                assert t_end in {5, 10}
        else:
            for f, items, t_end in s.runtime_functions:
                assert f is not None
                assert items is None
                assert t_end is None

    for s in (root, g1, g2):
        assert not s.runtime_functions

    # the global clock starts at 1 / 10 (that's how it's computed)
    for s, t_end in [
            (s1, 15.1), (s2, 20.1), (s3, 25.1), (s4, 45.1), (s5, 60.1),
            (s6, 60.1), (root, 60.1), (g1, 20.1), (g2, 60.1)]:
        # if parent can pre-compute, this stage is not set
        if not s.parent_stage or not s.parent_stage.can_pre_compute:
            assert s.t_end == t_end

    # s6 func is never finished as it's parallel any
    for s, t_end in [
            (s1, 15.1), (s2, 20.1), (s3, 25.1), (s4, 45.1), (s5, 60.1)]:
        if s.disable_pre_compute:
            assert s.functions[0].t_end == t_end


def test_pre_compute_infinite_func(stage_factory: StageFactoryBase):
    function_factory = stage_factory.function_factory

    root = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=False)
    stage_factory.add_stage(root)

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    root.add_shape(shape.shape)

    ConstFunc = function_factory.get('ConstFunc')
    f = FuncGroup(function_factory=stage_factory.function_factory)
    f1 = ConstFunc(function_factory=stage_factory.function_factory, duration=1)
    f2 = ConstFunc(
        function_factory=stage_factory.function_factory, duration=-1)
    f.add_func(f1)
    f.add_func(f2)

    root.add_func(f)

    root.init_stage_tree()
    root.apply_pre_compute(True, 60, 0, {shape.name})

    assert not root.can_pre_compute
    assert not root.disable_pre_compute


@pytest.mark.parametrize('pad', [True, False])
@pytest.mark.parametrize('pre_compute', [True, False])
@pytest.mark.parametrize('blank', [True, False])
def test_stage_padding(
        stage_factory: StageFactoryBase, pre_compute, pad, blank):
    function_factory = stage_factory.function_factory

    root = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=False)
    stage_factory.add_stage(root)

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    root.add_shape(shape.shape)

    ConstFunc = function_factory.get('ConstFunc')
    f = ConstFunc(function_factory=stage_factory.function_factory, duration=1)
    if not blank:
        root.add_func(f)

    if pad:
        root.pad_stage_ticks = 22

    _, n = get_stage_time_intensity(
        stage_factory, root.name, 10, pre_compute=pre_compute)

    if pad:
        assert n == 22
    elif blank:
        assert not n
    else:
        assert n == 10


@pytest.mark.parametrize('pre_compute', [True, False])
def test_t_end_empty_stage(stage_factory: StageFactoryBase, pre_compute):
    root = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=False)
    stage_factory.add_stage(root)

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    root.add_shape(shape.shape)

    get_stage_time_intensity(
        stage_factory, root.name, 10, pre_compute=pre_compute)

    # global time starts at 1 / 10
    assert root.t_end == Fraction(1, 10)


def test_t_end_empty_stage_func(stage_factory: StageFactoryBase):
    function_factory = stage_factory.function_factory

    root = make_stage(
        stage_factory, color_r=True, color_g=False, color_b=False)
    stage_factory.add_stage(root)

    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)
    root.add_shape(shape.shape)

    ConstFunc = function_factory.get('ConstFunc')
    f1 = ConstFunc(
        function_factory=stage_factory.function_factory, duration=.05)
    f2 = ConstFunc(
        function_factory=stage_factory.function_factory, duration=0)
    root.add_func(f1)
    root.add_func(f2)

    get_stage_time_intensity(stage_factory, root.name, 10, pre_compute=False)

    # global time starts at 1 / 10
    assert f1.t_end == Fraction(1, 10) + .05
    assert f2.t_end == Fraction(1, 10) + .05
    assert root.t_end == Fraction(1, 10) + .05


def test_external_plugin_source_in_factory(
        stage_factory: StageFactoryBase, tmp_path):
    sys.path.append(str(tmp_path))
    mod = tmp_path / 'my_stage_plugin' / '__init__.py'
    try:
        mod.parent.mkdir()
        mod.write_text(fake_plugin_stage)
        register_external_stages(stage_factory, 'my_stage_plugin')

        assert 'CeedStage' in stage_factory.stages_cls
        assert 'FakeStage' in stage_factory.stages_cls

        assert 'ceed.stage.plugin' in stage_factory.plugin_sources
        plugin_contents = stage_factory.plugin_sources['my_stage_plugin']
        assert plugin_contents == [
            (('__init__.py', ),
             fake_plugin_stage.replace('\n', os.linesep).encode())
        ]
    finally:
        sys.path.remove(str(tmp_path))
        del sys.modules['my_stage_plugin']


def test_external_plugin_single_file(
        stage_factory: StageFactoryBase, tmp_path):
    sys.path.append(str(tmp_path))
    mod = tmp_path / 'my_stage_plugin.py'
    try:
        mod.write_text(fake_plugin_stage)
        with pytest.raises(ModuleNotFoundError):
            register_external_stages(stage_factory, 'my_stage_plugin')
    finally:
        sys.path.remove(str(tmp_path))


@pytest.mark.parametrize('pre_compute', [True, False])
def test_custom_stage_evaluate(stage_factory: StageFactoryBase, pre_compute):
    shape = EllipseShapeP1(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)

    shape2 = EllipseShapeP2(
        app=None, painter=stage_factory.shape_factory, show_in_gui=False,
        create_add_shape=True)

    class MyStage(CeedStage):

        def evaluate_stage(self, shapes, last_end_t):
            t = yield
            for i in range(10):
                shapes[shape.name].append((.1, .2, (i % 2) * .3, None))
                shapes[shape2.name].append((.1, .2, (i % 2) * .5, None))
                t = yield

            self.t_end = t
            raise StageDoneException

    stage = MyStage(
        stage_factory=stage_factory,
        function_factory=stage_factory.function_factory,
        shape_factory=stage_factory.shape_factory)
    stage_factory.add_stage(stage)
    stage.add_shape(shape.shape)
    stage.add_shape(shape2.shape)

    values, n = get_stage_time_intensity(
        stage_factory, stage.name, 10, pre_compute=pre_compute)
    assert n == 10

    for i, (r, g, b, a) in enumerate(values[shape.name]):
        assert r == .1
        assert g == .2
        assert b == (i % 2) * .3
        assert a == 1

    for i, (r, g, b, a) in enumerate(values[shape2.name]):
        assert r == .1
        assert g == .2
        assert b == (i % 2) * .5
        assert a == 1

    assert stage.t_end == Fraction(11, 10)


def test_add_stage_unique_built_in_name(stage_factory: StageFactoryBase):
    assert not stage_factory.stages

    cls = stage_factory.get('CeedStage')
    s = cls(
        stage_factory=stage_factory,
        function_factory=stage_factory.function_factory,
        shape_factory=stage_factory.shape_factory)
    orig_name = s.name
    n = len(stage_factory.stage_names)
    assert orig_name in stage_factory.stage_names

    stage_factory.add_stage(s)
    assert s in stage_factory.stages
    assert len(stage_factory.stage_names) == n + 1
    assert stage_factory.stage_names[s.name] is s
    assert orig_name in stage_factory.stage_names
    assert s.name != orig_name
