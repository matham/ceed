import pytest
from copy import deepcopy, copy
import math
from typing import Tuple, List, Dict
from collections import defaultdict
from fractions import Fraction

from ceed.function import FunctionFactoryBase, FuncGroup, FuncBase
from ceed.stage import StageFactoryBase, CeedStage, CeedStageRef, \
    StageDoneException
from ceed.tests.test_app.examples.stages import ParaAllStage, ParaAnyStage, \
    SerialAllStage, SerialAnyStage, assert_stages_same, make_stage
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


def get_stage_time_intensity(
        stage_factory: StageFactoryBase, stage_name: str, frame_rate
) -> Tuple[Dict[str, List[Tuple[float, float, float, float]]], int]:
    """Samples the stage with the given frame rate and returns the intensity
    value for each shape for each timestamp.
    """
    tick = stage_factory.tick_stage(stage_name)
    # the sampling rate at which we sample the functions
    frame_rate = int(frame_rate)

    obj_values = defaultdict(list)
    count = 0
    while True:
        count += 1

        try:
            next(tick)
            shape_values = tick.send(Fraction(count, frame_rate))
        except StageDoneException:
            break

        values = stage_factory.fill_shape_gl_color_values(
            None, shape_values)
        for name, r, g, b, a in values:
            obj_values[name].append((r, g, b, a))

    return obj_values, count - 1


def test_factory_stage_unique_names(stage_factory: StageFactoryBase):
    assert not stage_factory.stages
    assert not stage_factory.stage_names
    stage = SerialAllStage(stage_factory=stage_factory, show_in_gui=False)
    stage.create_stage()
    stage = stage.stage

    # add first stage
    stage_factory.test_changes_count = 0
    stage_factory.add_stage(stage)
    assert len(stage_factory.stages) == 1
    assert len(stage_factory.stage_names) == 1
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
    assert len(stage_factory.stage_names) == 2
    assert stage2 in stage_factory.stages
    assert stage2.name in stage_factory.stage_names
    assert stage2 is stage_factory.stage_names[stage2.name]
    assert stage_factory.test_changes_count

    # try making stage2 the same name as stage 1
    stage_factory.test_changes_count = 0
    stage2.name = stage.name
    assert stage2.name != stage.name
    assert len(stage_factory.stages) == 2
    assert len(stage_factory.stage_names) == 2
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
    assert len(stage_factory.stage_names) == 2
    assert stage in stage_factory.stages
    assert stage is stage_factory.stage_names['stagnation']
    assert stage_factory.test_changes_count

    # try setting stage2 to the empty name
    stage_factory.test_changes_count = 0
    stage2.name = ''
    assert stage2.name != stage.name
    assert stage2.name
    assert len(stage_factory.stages) == 2
    assert len(stage_factory.stage_names) == 2
    assert stage2 in stage_factory.stages
    assert stage2.name in stage_factory.stage_names
    assert stage2 is stage_factory.stage_names[stage2.name]
    assert stage_factory.test_changes_count


def test_shape_add_remove(stage_factory: StageFactoryBase):
    assert not stage_factory.stages
    assert not stage_factory.stage_names

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


def test_clear_stages(stage_factory: StageFactoryBase):
    assert not stage_factory.stages
    assert not stage_factory.stage_names

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
    assert not stage_factory.stage_names


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


def test_simple_stage_intensity(stage_factory: StageFactoryBase):
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

    values, n = get_stage_time_intensity(stage_factory, stage.name, 10)
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


def test_recursive_stage_intensity(stage_factory: StageFactoryBase):
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

    values, n = get_stage_time_intensity(stage_factory, root.name, 10)
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


def test_recursive_full_stage_intensity(stage_factory: StageFactoryBase):
    root, g1, g2, s1, s2, s3, s4, s5, s6 = create_recursive_stages(
        stage_factory)

    from ceed.function.plugin import LinearFunc
    for i, stage in enumerate((s1, s2, s3, s4, s5, s6)):
        stage.stage.add_func(LinearFunc(
            function_factory=stage_factory.function_factory, b=0, m=.1,
            duration=(i % 2 + 1) * 5))

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

    values, n = get_stage_time_intensity(stage_factory, root.name, 10)
    assert n == 10 * (10 + 5 + 10 + 5)
    assert len(values) == 2
    colors = values[shape.name]
    colors2 = values[shape2.name]
    assert len(colors) == n

    for s, start, e in [(s1, 0, 5), (s4, 15, 25), (s5, 25, 30)]:
        for i in range(start * 10, e * 10):
            r, g, b, a = colors[i]

            i -= start * 10
            assert math.isclose(r, i / 10 * .1) if s.color_r else r == 0
            assert math.isclose(g, i / 10 * .1) if s.color_g else g == 0
            assert math.isclose(b, i / 10 * .1) if s.color_b else b == 0

    for start, e in [(5, 15), ]:
        for i in range(start * 10, e * 10):
            r, g, b, a = colors[i]
            assert r == 0
            assert g == 0
            assert b == 0

    for s, start, e in [(s2, 0, 10), (s3, 10, 15), (s6, 25, 30)]:
        for i in range(start * 10, e * 10):
            r, g, b, a = colors2[i]

            i -= start * 10
            assert math.isclose(r, i / 10 * .1) if s.color_r else r == 0
            assert math.isclose(g, i / 10 * .1) if s.color_g else g == 0
            assert math.isclose(b, i / 10 * .1) if s.color_b else b == 0

    for start, e in [(15, 25), ]:
        for i in range(start * 10, e * 10):
            r, g, b, a = colors2[i]
            assert r == 0
            assert g == 0
            assert b == 0


def test_single_frame_stage_intensity(stage_factory: StageFactoryBase):
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

    values, n = get_stage_time_intensity(stage_factory, stage.name, 120)
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


@pytest.mark.parametrize('rate', [60., 120., 100.])
@pytest.mark.parametrize('duration', [
    (.5, .5, .5), (.51, .5, .49), (.1, .1, .1), (.11, .33, .59),
    (31 / 60, 5 / 12, 1441 / 720)])
def test_stage_func_float_duration(
        stage_factory: StageFactoryBase, rate, duration):
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

    values, n = get_stage_time_intensity(stage_factory, root.name, rate)
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


def test_stage_func_float_duration_epsilon(stage_factory: StageFactoryBase):
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

    values, n = get_stage_time_intensity(stage_factory, root.name, 60.)
    expected = int(60. * 3.6)
    assert expected - 1 <= n <= expected + 1

    counts = collapse_list_to_counts(values[shape.name])
    duration = [1., 3 * .1, 3 * .1, 2.]
    a = [.1, .2, .3, .4]
    for i in range(4):
        a_val, count = counts[i]
        assert a_val == (a[i], 0, 0, 1)
        assert count - 1 <= round(duration[i] * 60) <= count + 1


def test_stage_func_clip_range(stage_factory: StageFactoryBase):
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

    values, n = get_stage_time_intensity(stage_factory, root.name, 60.)
    intensities = {v[0] for v in values[shape.name]}
    assert intensities == {0, .2, .3, 1}
