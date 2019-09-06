import pytest
from copy import deepcopy

from ceed.shape import CeedPaintCanvasBehavior, CeedShape, CeedShapeGroup
from .test_app.examples.shapes import Shape, EllipseShapeP1, shape_classes, \
    CircleShapeP1, PolygonShapeP1, FreeformPolygonShapeP1


def assert_shapes_same(
        shape1: CeedShape, shape2: CeedShape, compare_name=False):
    assert isinstance(shape1, shape2.__class__)

    keys = set(shape1.get_state().keys()) | set(shape2.get_state().keys())
    assert 'name' in keys
    if not compare_name:
        keys.remove('name')
    keys.remove('cls')

    for key in keys:
        assert getattr(shape1, key) == getattr(shape2, key)


@pytest.mark.parametrize("shape_cls", shape_classes)
def test_shape_instantiate(shape_factory: CeedPaintCanvasBehavior, shape_cls):
    shape = shape_cls(app=None, painter=shape_factory, manually_add=False)
    shape.make_shape()
    shape.assert_shape_prop_same(compare_name=True)


@pytest.mark.parametrize("shape_cls", shape_classes)
def test_shape_copy(shape_factory: CeedPaintCanvasBehavior, shape_cls):
    shape = shape_cls(app=None, painter=shape_factory, manually_add=False)
    shape.make_shape()
    shape.assert_shape_prop_same(compare_name=True)

    new_shape = deepcopy(shape.shape)
    assert_shapes_same(shape.shape, new_shape, compare_name=True)


@pytest.mark.parametrize("shape_cls", shape_classes)
def test_shape_add_remove(shape_factory: CeedPaintCanvasBehavior, shape_cls):
    assert not shape_factory.shapes
    assert not shape_factory.shape_names
    shape = shape_cls(app=None, painter=shape_factory, manually_add=False)
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
    shape = EllipseShapeP1(app=None, painter=shape_factory, manually_add=False)
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
        app=None, painter=shape_factory, manually_add=False)
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


def assert_add_three_shapes(shape_factory: CeedPaintCanvasBehavior):
    assert not shape_factory.shapes
    assert not shape_factory.shape_names
    shape = EllipseShapeP1(app=None, painter=shape_factory, manually_add=False)
    shape.make_shape()

    shape2 = PolygonShapeP1(
        app=None, painter=shape_factory, manually_add=False)
    shape2.make_shape()

    shape3 = CircleShapeP1(app=None, painter=shape_factory, manually_add=False)
    shape3.make_shape()

    assert shape_factory.add_shape(shape.shape)
    assert shape_factory.add_shape(shape2.shape)
    assert shape_factory.add_shape(shape3.shape)

    assert shape_factory.shapes == [shape.shape, shape2.shape, shape3.shape]
    assert len(shape_factory.shapes) == 3
    assert len(shape_factory.shape_names) == 3
    for s in (shape, shape2, shape3):
        assert shape_factory.shape_names[s.name] is s.shape
    return shape, shape2, shape3


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
