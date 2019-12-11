import pytest
from copy import deepcopy
import math

from ceed.shape import CeedPaintCanvasBehavior, CeedShape, CeedShapeGroup
from .test_app.examples.shapes import Shape, EllipseShapeP1, shape_classes, \
    CircleShapeP1, assert_add_three_shapes, assert_add_three_groups
from ceed.tests.common import add_prop_watch


def assert_shapes_same(
        shape1: CeedShape, shape2: CeedShape, compare_name=False):
    assert type(shape1) == type(shape2)

    keys = set(shape1.get_state().keys()) | set(shape2.get_state().keys())
    assert 'name' in keys
    if not compare_name:
        keys.remove('name')
    keys.remove('cls')

    for key in keys:
        assert getattr(shape1, key) == getattr(shape2, key)


@pytest.mark.parametrize("shape_cls", shape_classes)
def test_shape_instantiate(shape_factory: CeedPaintCanvasBehavior, shape_cls):
    shape = shape_cls(app=None, painter=shape_factory, show_in_gui=False)
    shape.make_shape()
    shape.assert_shape_prop_same(compare_name=True)


@pytest.mark.parametrize("shape_cls", shape_classes)
def test_shape_copy(shape_factory: CeedPaintCanvasBehavior, shape_cls):
    shape = shape_cls(app=None, painter=shape_factory, show_in_gui=False)
    shape.make_shape()
    shape.assert_shape_prop_same(compare_name=True)

    new_shape = deepcopy(shape.shape)
    assert_shapes_same(shape.shape, new_shape, compare_name=True)


@pytest.mark.parametrize("shape_cls", shape_classes)
def test_shape_add_remove(shape_factory: CeedPaintCanvasBehavior, shape_cls):
    assert not shape_factory.shapes
    assert not shape_factory.shape_names
    shape = shape_cls(app=None, painter=shape_factory, show_in_gui=False)
    shape.make_shape()

    # add shape
    shape_factory.test_changes_count = 0
    assert shape_factory.add_shape(shape.shape)
    assert shape.shape in shape_factory.shapes
    assert shape.name in shape_factory.shape_names
    assert shape.shape is shape_factory.shape_names[shape.name]
    assert shape_factory.test_changes_count

    # remove shape
    shape_factory.test_changes_remove_shape_count = 0
    shape_factory.test_changes_count = 0
    assert shape_factory.remove_shape(shape.shape)
    assert shape.shape not in shape_factory.shapes
    assert shape.name not in shape_factory.shape_names
    assert shape_factory.test_changes_remove_shape_count
    assert shape_factory.test_changes_count

    # remove same shape again
    shape_factory.test_changes_remove_shape_count = 0
    shape_factory.test_changes_count = 0
    assert not shape_factory.remove_shape(shape.shape)
    assert not shape_factory.test_changes_remove_shape_count
    assert not shape_factory.test_changes_count


def test_shape_name(shape_factory: CeedPaintCanvasBehavior):
    assert not shape_factory.shapes
    assert not shape_factory.shape_names
    shape = EllipseShapeP1(app=None, painter=shape_factory, show_in_gui=False)
    shape.make_shape()

    # add first shape
    shape_factory.test_changes_count = 0
    assert shape_factory.add_shape(shape.shape)
    assert len(shape_factory.shapes) == 1
    assert len(shape_factory.shape_names) == 1
    assert shape.shape in shape_factory.shapes
    assert shape.name in shape_factory.shape_names
    assert shape.shape is shape_factory.shape_names[shape.name]
    assert shape_factory.test_changes_count

    shape2 = EllipseShapeP1(
        app=None, painter=shape_factory, show_in_gui=False)
    shape2.make_shape()

    # add second shape
    shape_factory.test_changes_count = 0
    assert shape_factory.add_shape(shape2.shape)
    assert len(shape_factory.shapes) == 2
    assert len(shape_factory.shape_names) == 2
    assert shape2.shape in shape_factory.shapes
    assert shape_factory.test_changes_count

    assert shape2.name != shape2.shape.name
    assert shape.shape.name != shape2.shape.name
    assert shape2.shape.name in shape_factory.shape_names
    assert shape2.shape is shape_factory.shape_names[shape2.shape.name]

    # try making shape2 the same name as shape 1
    shape_factory.test_changes_count = 0
    shape2.shape.name = shape.shape.name
    assert len(shape_factory.shapes) == 2
    assert len(shape_factory.shape_names) == 2
    assert shape2.shape in shape_factory.shapes
    assert shape_factory.test_changes_count

    assert shape.shape.name != shape2.shape.name
    assert shape2.shape.name in shape_factory.shape_names
    assert shape2.shape is shape_factory.shape_names[shape2.shape.name]


def test_reorder_shape(shape_factory: CeedPaintCanvasBehavior):
    shape, shape2, shape3 = assert_add_three_shapes(shape_factory)

    shape_factory.test_changes_count = 0
    shape_factory.reorder_shape(shape3.shape, before_shape=shape2.shape)
    assert shape_factory.test_changes_count

    assert shape_factory.shapes == [shape.shape, shape3.shape, shape2.shape]
    assert len(shape_factory.shapes) == 3
    assert len(shape_factory.shape_names) == 3
    for s in (shape, shape2, shape3):
        assert shape_factory.shape_names[s.name] is s.shape


def test_move_shape_lower(shape_factory: CeedPaintCanvasBehavior):
    shape, shape2, shape3 = assert_add_three_shapes(shape_factory)

    shape_factory.test_changes_count = 0
    shape_factory.move_shape_lower(shape3.shape)
    assert shape_factory.test_changes_count

    assert shape_factory.shapes == [shape.shape, shape3.shape, shape2.shape]
    assert len(shape_factory.shapes) == 3
    assert len(shape_factory.shape_names) == 3
    for s in (shape, shape2, shape3):
        assert shape_factory.shape_names[s.name] is s.shape


def test_move_shape_upwards(shape_factory: CeedPaintCanvasBehavior):
    shape, shape2, shape3 = assert_add_three_shapes(shape_factory)

    shape_factory.test_changes_count = 0
    shape_factory.move_shape_upwards(shape.shape)
    assert shape_factory.test_changes_count

    assert shape_factory.shapes == [shape2.shape, shape.shape, shape3.shape]
    assert len(shape_factory.shapes) == 3
    assert len(shape_factory.shape_names) == 3
    for s in (shape, shape2, shape3):
        assert shape_factory.shape_names[s.name] is s.shape


def test_group_name(shape_factory: CeedPaintCanvasBehavior):
    assert not shape_factory.groups
    assert not shape_factory.shape_group_names

    # first group
    shape_factory.test_changes_count = 0
    group = shape_factory.add_group()

    assert isinstance(group, CeedShapeGroup)
    assert len(shape_factory.groups) == 1
    assert len(shape_factory.shape_group_names) == 1
    assert group in shape_factory.groups
    assert group.name in shape_factory.shape_group_names
    assert group is shape_factory.shape_group_names[group.name]
    assert shape_factory.test_changes_count

    # add second group
    shape_factory.test_changes_count = 0
    group2 = shape_factory.add_group()

    assert len(shape_factory.groups) == 2
    assert len(shape_factory.shape_group_names) == 2
    assert group2 in shape_factory.groups
    assert shape_factory.test_changes_count

    assert group.name != group2.name
    assert group2.name in shape_factory.shape_group_names
    assert group2 is shape_factory.shape_group_names[group2.name]

    # try making a duplicate name
    shape_factory.test_changes_count = 0
    group2.name = group.name
    assert len(shape_factory.groups) == 2
    assert len(shape_factory.shape_group_names) == 2
    assert group2 in shape_factory.groups
    assert shape_factory.test_changes_count

    assert group.name != group2.name
    assert group2.name in shape_factory.shape_group_names
    assert group2 is shape_factory.shape_group_names[group2.name]

    # try setting name
    shape_factory.test_changes_count = 0
    group2.name = 'something random'
    assert len(shape_factory.groups) == 2
    assert len(shape_factory.shape_group_names) == 2
    assert group2 in shape_factory.groups
    assert shape_factory.test_changes_count

    assert group2.name == 'something random'
    assert group2.name in shape_factory.shape_group_names
    assert group2 is shape_factory.shape_group_names['something random']


def test_add_remove_group(shape_factory: CeedPaintCanvasBehavior):
    assert not shape_factory.groups
    assert not shape_factory.shape_group_names

    # first group
    shape_factory.test_changes_count = 0
    group = shape_factory.add_group()

    assert isinstance(group, CeedShapeGroup)
    assert len(shape_factory.groups) == 1
    assert len(shape_factory.shape_group_names) == 1
    assert group in shape_factory.groups
    assert group.name in shape_factory.shape_group_names
    assert group is shape_factory.shape_group_names[group.name]
    assert shape_factory.test_changes_count

    # add second group
    shape_factory.test_changes_count = 0
    group2 = shape_factory.add_group()

    assert shape_factory.groups == [group, group2]
    assert shape_factory.shape_group_names == {
        group.name: group, group2.name: group2}
    assert shape_factory.test_changes_count

    # remove first group
    shape_factory.test_changes_count = 0
    shape_factory.test_changes_remove_group_count = 0
    assert shape_factory.remove_group(group)

    assert shape_factory.test_changes_count
    assert shape_factory.test_changes_remove_group_count
    assert shape_factory.groups == [group2]
    assert shape_factory.shape_group_names == {group2.name: group2}

    # add back first group
    shape_factory.test_changes_count = 0
    assert shape_factory.add_group(group) is group

    assert shape_factory.groups == [group2, group]
    assert shape_factory.shape_group_names == {
        group.name: group, group2.name: group2}
    assert shape_factory.test_changes_count

    # remove all groups
    shape_factory.test_changes_count = 0
    shape_factory.test_changes_remove_group_count = 0
    shape_factory.remove_all_groups()

    assert shape_factory.test_changes_count
    assert shape_factory.test_changes_remove_group_count
    assert not shape_factory.groups
    assert not shape_factory.shape_group_names


def test_add_to_group(shape_factory: CeedPaintCanvasBehavior):
    shape, shape2, shape3 = assert_add_three_shapes(shape_factory)

    group = shape_factory.add_group()
    add_prop_watch(group, 'on_changed', 'test_changes_count')
    group.test_changes_count = 0
    assert group.add_shape(shape.shape)
    assert group.test_changes_count
    assert group.shapes == [shape.shape]

    # can't add twice
    assert not group.add_shape(shape.shape)
    assert group.shapes == [shape.shape]

    # add second shape
    group.test_changes_count = 0
    assert group.add_shape(shape2.shape)
    assert group.test_changes_count
    assert group.shapes == [shape.shape, shape2.shape]

    # can't add twice
    assert not group.add_shape(shape.shape)
    assert not group.add_shape(shape2.shape)
    assert group.shapes == [shape.shape, shape2.shape]


def test_remove_from_group(shape_factory: CeedPaintCanvasBehavior):
    shape, shape2, shape3 = assert_add_three_shapes(shape_factory)

    group = shape_factory.add_group()
    add_prop_watch(group, 'on_changed', 'test_changes_count')
    assert group.add_shape(shape.shape)
    assert group.add_shape(shape2.shape)
    assert group.shapes == [shape.shape, shape2.shape]

    # remove first shape
    group.test_changes_count = 0
    group.remove_shape(shape.shape)
    assert group.test_changes_count
    assert group.shapes == [shape2.shape]

    # removing again does nothing
    group.remove_shape(shape.shape)
    assert group.shapes == [shape2.shape]

    # remove last
    group.test_changes_count = 0
    group.remove_shape(shape2.shape)
    assert group.test_changes_count
    assert not group.shapes

    # removing again does nothing
    group.remove_shape(shape2.shape)
    assert not group.shapes


def test_remove_all_from_group(shape_factory: CeedPaintCanvasBehavior):
    shape, shape2, shape3 = assert_add_three_shapes(shape_factory)

    group = shape_factory.add_group()
    add_prop_watch(group, 'on_changed', 'test_changes_count')
    assert group.add_shape(shape.shape)
    assert group.add_shape(shape2.shape)
    assert group.shapes == [shape.shape, shape2.shape]

    # remove first shape
    group.test_changes_count = 0
    group.remove_all()
    assert group.test_changes_count
    assert not group.shapes


def test_add_selection_to_group(shape_factory: CeedPaintCanvasBehavior):
    shape, shape2, shape3 = assert_add_three_shapes(shape_factory)
    # select first and second shape
    shape_factory.select_shape(shape.shape)
    shape_factory.select_shape(shape2.shape)

    group = shape_factory.add_group()
    add_prop_watch(group, 'on_changed', 'test_changes_count')

    # add first shape manually
    assert group.add_shape(shape.shape)
    assert group.shapes == [shape.shape]

    # add all selected shapes
    group.test_changes_count = 0
    shape_factory.add_selected_shapes_to_group(group)
    assert group.test_changes_count
    assert group.shapes == [shape.shape, shape2.shape]


def test_remove_from_all_groups(shape_factory: CeedPaintCanvasBehavior):
    (group, group2, group3), (shape, shape2, shape3) = \
        assert_add_three_groups(shape_factory)
    add_prop_watch(group, 'on_changed', 'test_changes_count')
    add_prop_watch(group2, 'on_changed', 'test_changes_count')
    add_prop_watch(group3, 'on_changed', 'test_changes_count')

    # remove first shape
    group.test_changes_count = 0
    group3.test_changes_count = 0
    shape_factory.remove_shape_from_groups(shape.shape)

    assert group.test_changes_count
    assert group3.test_changes_count
    assert group.shapes == [shape2.shape]
    assert group2.shapes == [shape2.shape, shape3.shape]
    assert group3.shapes == [shape2.shape, shape3.shape]

    # remove third shape
    group2.test_changes_count = 0
    group3.test_changes_count = 0
    shape_factory.remove_shape_from_groups(shape3.shape)

    assert group2.test_changes_count
    assert group3.test_changes_count
    assert group.shapes == [shape2.shape]
    assert group2.shapes == [shape2.shape]
    assert group3.shapes == [shape2.shape]

    # remove second shape
    group.test_changes_count = 0
    group2.test_changes_count = 0
    group3.test_changes_count = 0
    shape_factory.remove_shape_from_groups(shape2.shape)

    assert group.test_changes_count
    assert group2.test_changes_count
    assert group3.test_changes_count
    assert not group.shapes
    assert not group2.shapes
    assert not group3.shapes


def test_set_factory_state(shape_factory: CeedPaintCanvasBehavior):
    (group, group2, group3), (shape, shape2, shape3) = \
        assert_add_three_groups(shape_factory)

    name_map = {}
    shape_factory.test_changes_count = 0
    shape_factory.set_state(shape_factory.get_state(), name_map)
    assert shape_factory.test_changes_count

    assert shape_factory.shapes[:3] == [
        shape.shape, shape2.shape, shape3.shape]
    assert len(shape_factory.shapes) == 6
    assert len(set(shape_factory.shapes)) == 6
    assert len(shape_factory.shape_names) == 6

    assert shape_factory.groups[:3] == [group, group2, group3]
    assert len(shape_factory.groups) == 6
    assert len(set(shape_factory.groups)) == 6
    assert len(shape_factory.shape_group_names) == 6

    shape4, shape5, shape6 = shape_factory.shapes[3:]
    group4, group5, group6 = shape_factory.groups[3:]

    assert group4.shapes == [shape4, shape5]
    assert group5.shapes == [shape5, shape6]
    assert group6.shapes == [shape4, shape5, shape6]

    assert_shapes_same(shape.shape, shape4)
    assert_shapes_same(shape2.shape, shape5)
    assert_shapes_same(shape3.shape, shape6)


def test_bounding_box(shape_factory: CeedPaintCanvasBehavior):
    shape = CircleShapeP1(app=None, painter=shape_factory, show_in_gui=False)
    shape.make_shape()

    assert shape_factory.add_shape(shape.shape)
    assert shape.shape.bounding_box == shape.bounding_box


def test_center(shape_factory: CeedPaintCanvasBehavior):
    shape = CircleShapeP1(app=None, painter=shape_factory, show_in_gui=False)
    shape.make_shape()

    assert shape_factory.add_shape(shape.shape)
    assert shape.shape.centroid == tuple(shape.center)


def test_area(shape_factory: CeedPaintCanvasBehavior):
    shape = CircleShapeP1(app=None, painter=shape_factory, show_in_gui=False)
    shape.make_shape()

    assert shape_factory.add_shape(shape.shape)
    assert math.isclose(shape.shape.area, shape.area)

    shape.shape.set_area(shape.shape.area / 4)
    assert math.isclose(shape.shape.radius, shape.radius / 2)
