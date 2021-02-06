import pytest
from typing import Type, List

from ceed.tests.ceed_app import CeedTestApp
from ceed.tests.common import exhaust
from ceed.tests.test_app import replace_text, touch_widget
from ceed.tests.test_app.examples.shapes import paired_tests, PolygonShapeP1, \
    CircleShapeP1, EllipseShapeP1, FreeformPolygonShapeP1, Shape, \
    EnclosingPolygon, paired_tests_one

pytestmark = pytest.mark.ceed_app


@pytest.mark.parametrize(
    "shape_cls",
    [PolygonShapeP1, CircleShapeP1, EllipseShapeP1, FreeformPolygonShapeP1])
async def test_simple_shape(ceed_app: CeedTestApp, shape_cls):
    painter = ceed_app.app.shape_factory
    assert ceed_app.app.shape_factory == painter
    assert not ceed_app.app.shape_factory.shapes

    shape = shape_cls(ceed_app, painter)
    await ceed_app.wait_clock_frames(2)

    assert shape.shape in ceed_app.app.shape_factory.shapes
    assert shape.shape.name == shape.name
    shape.check_shape_visible(True)

    shape.remove()
    await ceed_app.wait_clock_frames(2)

    assert shape.shape not in ceed_app.app.shape_factory.shapes
    assert not ceed_app.app.shape_factory.shapes
    shape.check_shape_visible(False)

    painter.add_shape(shape.shape)
    assert shape.shape in ceed_app.app.shape_factory.shapes
    assert shape.shape.name == shape.name

    assert not shape.shape.locked
    painter.lock_shape(shape.shape)
    assert shape.shape.locked
    painter.unlock_shape(shape.shape)
    assert not shape.shape.locked


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_multiple_shapes_add(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_multiple_shapes_remove(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)
    assert painter.shapes == [shape1.shape, shape2.shape]

    # delete the first shape
    shape1_widget = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape remove')()
    async for _ in paint_app.do_touch_down_up(widget=shape1_widget):
        pass
    assert painter.shapes == [shape2.shape]
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(False)
    shape2.check_shape_visible(True)

    # add the shape back
    shape1.show_in_gui()
    assert painter.shapes == [shape2.shape, shape1.shape]
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_multiple_shapes_move_depth(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    # tests move_shape_lower, move_shape_upwards, and reorder_shape
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)
    assert painter.shapes == [shape1.shape, shape2.shape]

    shape1_down = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape down')()
    shape1_up = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape up')()

    async for _ in paint_app.do_touch_down_up(widget=shape1_down):
        pass
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape2.shape, shape1.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    async for _ in paint_app.do_touch_down_up(widget=shape1_up):
        pass
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_multiple_shapes_duplicate(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)
    assert painter.shapes == [shape1.shape, shape2.shape]

    shape1_widget = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape drag')()
    shape2_widget = paint_app.resolve_widget().down(
        text=shape2.name).family_up(test_name='shape drag')()

    # duplicate first shape
    async for _ in paint_app.do_touch_drag(
            widget=shape1_widget, target_widget=shape2_widget):
        pass
    await paint_app.wait_clock_frames(2)

    assert painter.shapes[:2] == [shape1.shape, shape2.shape]
    assert len(painter.shapes) == 3
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    assert len({shape.name for shape in painter.shapes}) == 3, \
        'Names must be unique'

    # duplicate second shape
    async for _ in paint_app.do_touch_drag(
            widget=shape2_widget, target_widget=shape1_widget):
        pass
    await paint_app.wait_clock_frames(2)

    assert painter.shapes[:2] == [shape1.shape, shape2.shape]
    assert len(painter.shapes) == 4
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    assert len({shape.name for shape in painter.shapes}) == 4, \
        'Names must be unique'


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_shape_lock(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)
    assert painter.shapes == [shape1.shape, shape2.shape]

    shape1_lock = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape lock')()
    assert not shape1.shape.locked

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    v1_unlocked, = paint_app.get_widget_pos_pixel(
        painter, [shape1.test_points[2:4]])
    intensity_unlocked = max(v1_unlocked[:3])

    # lock
    async for _ in paint_app.do_touch_down_up(widget=shape1_lock):
        pass
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    assert shape1.shape.locked
    shape2.check_shape_visible(True)

    v1_locked, = paint_app.get_widget_pos_pixel(
        painter, [shape1.test_points[2:4]])
    intensity_locked = sum(v1_locked[:3]) / 3.
    assert intensity_locked < intensity_unlocked / 5. * 3.

    # unlock
    async for _ in paint_app.do_touch_down_up(widget=shape1_lock):
        pass
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    assert not shape1.shape.locked
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    v1_unlocked, = paint_app.get_widget_pos_pixel(
        painter, [shape1.test_points[2:4]])
    intensity_unlocked = max(v1_unlocked[:3])
    assert intensity_locked < intensity_unlocked / 5. * 3


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_shape_hide(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)
    assert painter.shapes == [shape1.shape, shape2.shape]

    shape1_hide = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape hide')()

    assert not shape1.shape.locked
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    # hide
    async for _ in paint_app.do_touch_down_up(widget=shape1_hide):
        pass
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    assert shape1.shape.locked
    shape1.check_shape_visible(False)
    shape2.check_shape_visible(True)

    # unhide
    async for _ in paint_app.do_touch_down_up(widget=shape1_hide):
        pass
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    assert shape1.shape.locked
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_shape_name(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    # tests _change_shape_name
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    assert shape1.shape.name == shape1.name
    assert shape2.shape.name == shape2.name
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    # expand options
    shape1_expand = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape expand')()
    async for _ in paint_app.do_touch_down_up(widget=shape1_expand):
        pass
    await paint_app.wait_clock_frames(2)

    # give it a name that already exists
    shape1_name = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape textname')()

    assert shape1_name.text == shape1.name

    await replace_text(paint_app, shape1_name, shape2.name)
    await paint_app.wait_clock_frames(2)

    assert shape1.shape.name != shape2.name
    assert shape2.shape.name == shape2.name
    assert shape1.shape.name
    assert shape1_name.text == shape1.shape.name

    # give it a name that doesn't exists
    await replace_text(paint_app, shape1_name, 'random shape')
    await paint_app.wait_clock_frames(2)

    assert shape1.shape.name != shape2.name
    assert shape2.shape.name == shape2.name
    assert shape1.shape.name == 'random shape'
    assert shape1_name.text == shape1.shape.name


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_pos(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    # expand options
    shape1_expand = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape expand')()
    async for _ in paint_app.do_touch_down_up(widget=shape1_expand):
        pass
    await paint_app.wait_clock_frames(2)

    # change the position
    for axis in ('x', 'y'):
        shape1_pos = paint_app.resolve_widget().down(
            text=shape1.name).family_up(
            test_name='shape pos{}'.format(axis))()
        pos = int(shape1_pos.text)

        await replace_text(paint_app, shape1_pos, str(pos + 25))
        await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True, offset=25)
    shape2.check_shape_visible(True)


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_area(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert shape1.area * 0.99 < shape1.shape.area < shape1.area * 1.01
    assert shape2.area * 0.99 < shape2.shape.area < shape2.area * 1.01

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    # expand options
    shape1_expand = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape expand')()
    async for _ in paint_app.do_touch_down_up(widget=shape1_expand):
        pass
    await paint_app.wait_clock_frames(2)

    shape1_area = paint_app.resolve_widget().down(
        text=shape1.name).family_up(test_name='shape area')()
    area = float(shape1_area.text)

    await replace_text(paint_app, shape1_area, '{:0.2f}'.format(area / 2.))
    await paint_app.wait_clock_frames(2)

    # shape1.check_shape_visible(True)
    # shape2.check_shape_visible(True)

    assert shape1.area * 0.99 / 2. < \
        shape1.shape.area < shape1.area * 1.01 / 2.
    assert shape2.area * 0.99 < shape2.shape.area < shape2.area * 1.01
    shape1.check_resize_by_area(0.5)


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_multiple_shapes_draw(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (
        cls(paint_app, painter, show_in_gui=False) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert not painter.shapes

    # set the correct draw mode
    draw_mode = paint_app.resolve_widget().down(
        test_name=shape1.activate_btn_name)()
    if draw_mode.state != 'down':
        async for _ in paint_app.do_touch_down_up(widget=draw_mode):
            pass
    await paint_app.wait_clock_frames(2)

    await shape1.draw()
    await paint_app.wait_clock_frames(2)
    assert painter.shapes == [shape1.shape]
    assert shape1.area * 0.99 < shape1.shape.area < shape1.area * 1.01
    shape1.check_shape_visible(True)
    assert len({shape.name for shape in painter.shapes}) == 1, \
        'Names must be unique'

    await shape2.draw()
    await paint_app.wait_clock_frames(2)
    assert painter.shapes == [shape1.shape, shape2.shape]
    assert shape2.area * 0.99 < shape2.shape.area < shape2.area * 1.01
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    assert len({shape.name for shape in painter.shapes}) == 2, \
        'Names must be unique'


async def test_shape_add_enclosing_polygon(paint_app: CeedTestApp):
    painter = paint_app.app.shape_factory
    assert not painter.shapes

    # set the correct draw mode
    add = paint_app.resolve_widget().down(
        test_name='add enclosing polygon')()
    async for _ in paint_app.do_touch_down_up(widget=add):
        pass
    await paint_app.wait_clock_frames(2)

    assert len(painter.shapes) == 1
    w = paint_app.app.view_controller.screen_width
    h = paint_app.app.view_controller.screen_height
    points = [0, 0, w, 0, w, h, 0, h]
    shape = EnclosingPolygon(
        app=paint_app, painter=painter, shape=painter.shapes[0],
        points=points, outside_point=[w / 2, h / 2], show_in_gui=False)

    assert shape.shape.name == 'enclosed'
    shape.check_shape_visible(True)


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_shape_duplicate_selection(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    # make sure we're in single select mode
    multiselect = paint_app.resolve_widget().down(
        test_name='shape multiselect')()
    assert multiselect.state == 'normal'

    # select first shape
    shape1_widget = paint_app.resolve_widget().down(text=shape1.name)()
    async for _ in paint_app.do_touch_down_up(widget=shape1_widget):
        pass
    await paint_app.wait_clock_frames(2)

    # duplicate it
    duplicate = paint_app.resolve_widget().down(test_name='shape duplicate')()
    async for _ in paint_app.do_touch_down_up(widget=duplicate):
        pass

    assert painter.shapes[:2] == [shape1.shape, shape2.shape]
    assert len(painter.shapes) == 3
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    assert len({shape.name for shape in painter.shapes}) == 3, \
        'Names must be unique'

    # select second shape
    shape2_widget = paint_app.resolve_widget().down(text=shape2.name)()
    async for _ in paint_app.do_touch_down_up(widget=shape2_widget):
        pass
    await paint_app.wait_clock_frames(2)

    # duplicate it
    async for _ in paint_app.do_touch_down_up(widget=duplicate):
        pass
    await paint_app.wait_clock_frames(2)

    assert painter.shapes[:2] == [shape1.shape, shape2.shape]
    assert len(painter.shapes) == 4
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    assert len({shape.name for shape in painter.shapes}) == 4, \
        'Names must be unique'


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_shape_delete_selection(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)

    # make sure we're in single select mode
    multiselect = paint_app.resolve_widget().down(
        test_name='shape multiselect')()
    assert multiselect.state == 'normal'

    # select first shape
    shape1_widget = paint_app.resolve_widget().down(text=shape1.name)()
    async for _ in paint_app.do_touch_down_up(widget=shape1_widget):
        pass
    await paint_app.wait_clock_frames(2)

    # delete it
    delete = paint_app.resolve_widget().down(test_name='shape delete')()
    async for _ in paint_app.do_touch_down_up(widget=delete):
        pass

    assert painter.shapes == [shape2.shape, ]
    shape1.check_shape_visible(False)
    shape2.check_shape_visible(True)

    # select second shape
    shape2_widget = paint_app.resolve_widget().down(text=shape2.name)()
    async for _ in paint_app.do_touch_down_up(widget=shape2_widget):
        pass
    await paint_app.wait_clock_frames(2)

    # delete it
    async for _ in paint_app.do_touch_down_up(widget=delete):
        pass
    await paint_app.wait_clock_frames(2)

    assert not painter.shapes
    shape1.check_shape_visible(False)
    shape2.check_shape_visible(False)


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_shape_single_select_widget(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # make sure we're in single select mode
    multiselect = paint_app.resolve_widget().down(
        test_name='shape multiselect')()
    assert multiselect.state == 'normal'

    shape1_widget = paint_app.resolve_widget().down(text=shape1.name)()
    shape2_widget = paint_app.resolve_widget().down(text=shape2.name)()

    # select first shape
    async for _ in paint_app.do_touch_down_up(widget=shape1_widget):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_not_selected()

    # select second shape
    async for _ in paint_app.do_touch_down_up(widget=shape2_widget):
        pass
    await paint_app.wait_clock_frames(2)
    painter.export_to_png(r'E:\img3.png')

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_not_selected()
    shape2.assert_shape_visible_selected()
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert shape2.shape.selected
    assert shape2.shape.widget.selected

    # select first shape again
    async for _ in paint_app.do_touch_down_up(widget=shape1_widget):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_not_selected()
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # deselect first shape
    async for _ in paint_app.do_touch_down_up(widget=shape1_widget):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_not_selected()
    shape2.assert_shape_visible_not_selected()
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_shape_single_select_shape(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # make sure we're in single select mode
    multiselect = paint_app.resolve_widget().down(
        test_name='shape multiselect')()
    assert multiselect.state == 'normal'

    # select first shape
    async for _ in paint_app.do_touch_down_up(
            widget=painter, pos=shape1.drag_point, duration=0):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_not_selected()
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # select second shape
    async for _ in paint_app.do_touch_down_up(
            widget=painter, pos=shape2.drag_point):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_not_selected()
    shape2.assert_shape_visible_selected()
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert shape2.shape.selected
    assert shape2.shape.widget.selected

    # select first shape again
    async for _ in paint_app.do_touch_down_up(
            widget=painter, pos=shape1.drag_point):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_not_selected()
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # deselect first shape, but it should stay selected
    async for _ in paint_app.do_touch_down_up(
            widget=painter, pos=shape1.drag_point):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_not_selected()
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_shape_multiselect_widget(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # make sure we're in multiselect mode
    multiselect = paint_app.resolve_widget().down(
        test_name='shape multiselect')()
    assert multiselect.state == 'normal'

    async for _ in paint_app.do_touch_down_up(widget=multiselect):
        pass
    await paint_app.wait_clock_frames(2)
    assert multiselect.state == 'down'

    shape1_widget = paint_app.resolve_widget().down(text=shape1.name)()
    shape2_widget = paint_app.resolve_widget().down(text=shape2.name)()

    # select first shape
    async for _ in paint_app.do_touch_down_up(widget=shape1_widget):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_not_selected()
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # select second shape
    async for _ in paint_app.do_touch_down_up(widget=shape2_widget):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_selected()
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert shape2.shape.selected
    assert shape2.shape.widget.selected

    # deselect first shape
    async for _ in paint_app.do_touch_down_up(widget=shape1_widget):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_not_selected()
    shape2.assert_shape_visible_selected()
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert shape2.shape.selected
    assert shape2.shape.widget.selected

    # deselect second shape
    async for _ in paint_app.do_touch_down_up(widget=shape2_widget):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_not_selected()
    shape2.assert_shape_visible_not_selected()
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # select first shape again
    async for _ in paint_app.do_touch_down_up(widget=shape1_widget):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_not_selected()
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected


@pytest.mark.parametrize("shape_classes", paired_tests_one)
async def test_shape_multiselect_shape(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # make sure we're in multiselect mode
    multiselect = paint_app.resolve_widget().down(
        test_name='shape multiselect')()
    assert multiselect.state == 'normal'

    async for _ in paint_app.do_touch_down_up(widget=multiselect):
        pass
    await paint_app.wait_clock_frames(2)
    assert multiselect.state == 'down'

    # select first shape
    async for _ in paint_app.do_touch_down_up(
            widget=painter, pos=shape1.drag_point):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_not_selected()
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # select second shape
    async for _ in paint_app.do_touch_down_up(
            widget=painter, pos=shape2.drag_point):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_selected()
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert shape2.shape.selected
    assert shape2.shape.widget.selected

    # deselect first shape
    async for _ in paint_app.do_touch_down_up(
            widget=painter, pos=shape1.drag_point):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_not_selected()
    shape2.assert_shape_visible_selected()
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert shape2.shape.selected
    assert shape2.shape.widget.selected

    # deselect second shape
    async for _ in paint_app.do_touch_down_up(
            widget=painter, pos=shape2.drag_point):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_not_selected()
    shape2.assert_shape_visible_not_selected()
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected

    # select first shape
    async for _ in paint_app.do_touch_down_up(
            widget=painter, pos=shape1.drag_point):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    shape1.assert_shape_visible_selected()
    shape2.assert_shape_visible_not_selected()
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected


async def make_4_shapes(paint_app: CeedTestApp) -> List[Shape]:
    painter = paint_app.app.shape_factory
    assert len(painter.shapes) == len(painter.shape_names)

    shapes = []
    i = len(painter.shapes)
    for cls in (PolygonShapeP1, EllipseShapeP1, CircleShapeP1,
                FreeformPolygonShapeP1):
        shapes.append(cls(paint_app, painter))

        i += 1
        assert len(painter.shapes) == i
        assert len(painter.shape_names) == i
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape.shape for shape in shapes]
    return shapes


async def test_group_add_remove(paint_app: CeedTestApp):
    painter = paint_app.app.shape_factory
    from ceed.shape import CeedShapeGroup
    # add the 4 shapes
    shapes: List[Shape] = await make_4_shapes(paint_app)
    s1, s2, s3, s4 = shapes
    assert not paint_app.app.shape_factory.groups

    # add the group
    add = paint_app.resolve_widget().down(test_name='add group')()
    await touch_widget(paint_app, add)
    assert len(paint_app.app.shape_factory.groups) == 1
    group = paint_app.app.shape_factory.groups[0]
    assert isinstance(group, CeedShapeGroup)
    container = paint_app.resolve_widget(group.widget).down(
        test_name='shape group obj container')()

    # drag the shapes into the group
    group_shape = [s2, s1, s4, s3]
    s: Shape
    for s in group_shape:
        drag_widget = paint_app.resolve_widget(s.shape.widget).down(
            test_name='shape drag')()
        async for _ in paint_app.do_touch_drag_follow(
                widget=drag_widget, target_widget=container,
                target_widget_loc=('center_x', 'y')):
            pass

    assert group.shapes == [s.shape for s in group_shape]
    assert group.paint_widget == painter

    # check that the labels in the group match
    group_shapes = group.widget.shape_widgets[::-1]
    widgets = list(zip(group_shape, group_shapes))
    assert len(widgets) == 4
    for s, widget in widgets:
        label = paint_app.resolve_widget(widget).down(
            test_name='group shape item label')()
        assert label.text == s.shape.name

    # delete the first item
    remove = paint_app.resolve_widget(group_shapes[0]).down(
        test_name='delete group item')()
    await touch_widget(paint_app, remove)
    assert group.widget.shape_widgets[::-1] == group_shapes[1:]
    assert group.shapes == [s.shape for s in group_shape[1:]]

    # delete the middle item
    remove = paint_app.resolve_widget(group_shapes[2]).down(
        test_name='delete group item')()
    await touch_widget(paint_app, remove)
    assert group.widget.shape_widgets[::-1] == \
        [group_shapes[1], group_shapes[3]]
    assert group.shapes == [s1.shape, s3.shape]


async def test_group_drag_in(paint_app: CeedTestApp):
    painter = paint_app.app.shape_factory
    shapes: List[Shape] = await make_4_shapes(paint_app)
    s1, *_ = shapes
    assert not painter.groups

    group_container = paint_app.resolve_widget().down(
        test_name='shape group container')()

    # add groups by dragging a shape to the group area
    for i in range(3):
        # initial groups and group's widget container count
        groups = painter.groups[:]
        container = paint_app.app.shape_groups_container.children[:]

        # drag it in
        drag_widget = paint_app.resolve_widget(s1.shape.widget).down(
            test_name='shape drag')()
        await exhaust(paint_app.do_touch_drag_follow(
            widget=drag_widget, target_widget=group_container,
            target_widget_loc=('center_x', 'y')))

        assert len(painter.groups) == i + 1
        assert len(paint_app.app.shape_groups_container.children) == i + 1
        assert painter.groups[:-1] == groups
        assert paint_app.app.shape_groups_container.children[1:] == container

        group = painter.groups[-1]
        assert len(group.shapes) == 1
        assert group.shapes[0] is s1.shape

        # close the group
        close = paint_app.resolve_widget(group.widget).down(
            test_name='shape group expand')()
        await touch_widget(paint_app, close)
        assert close.state == 'normal'


async def test_group_drag_duplicate(paint_app: CeedTestApp):
    painter = paint_app.app.shape_factory
    shapes: List[Shape] = await make_4_shapes(paint_app)
    s1, s2, *_ = shapes
    assert not painter.groups

    # add new group
    add = paint_app.resolve_widget().down(test_name='add group')()
    await touch_widget(paint_app, add)
    group = painter.groups[-1]

    # drag shape into it
    group_container = paint_app.resolve_widget(group.widget).down(
        test_name='shape group obj container')()
    drag_widget = paint_app.resolve_widget(s1.shape.widget).down(
        test_name='shape drag')()
    await exhaust(paint_app.do_touch_drag_follow(
        widget=drag_widget, target_widget=group_container,
        target_widget_loc=('center_x', 'y')))

    assert group.shapes == [s1.shape, ]
    assert len(group.widget.shape_widgets) == 1

    # try adding it again, it shouldn't be added
    await exhaust(paint_app.do_touch_drag_follow(
        widget=drag_widget, target_widget=group_container,
        target_widget_loc=('center_x', 'y')))
    assert group.shapes == [s1.shape, ]
    assert len(group.widget.shape_widgets) == 1

    # now add a different shape, which should work
    drag_widget = paint_app.resolve_widget(s2.shape.widget).down(
        test_name='shape drag')()
    await exhaust(paint_app.do_touch_drag_follow(
        widget=drag_widget, target_widget=group_container,
        target_widget_loc=('center_x', 'y')))
    assert group.shapes == [s1.shape, s2.shape]
    assert len(group.widget.shape_widgets) == 2


async def test_group_drag_multiple(paint_app: CeedTestApp):
    painter = paint_app.app.shape_factory
    shapes: List[Shape] = await make_4_shapes(paint_app)
    s1, s2, s3, s4 = shapes

    # make sure we're in multi-select mode
    multiselect = paint_app.resolve_widget().down(
        test_name='shape multiselect')()
    multiselect.state = 'down'

    # add a group
    add = paint_app.resolve_widget().down(test_name='add group')()
    await touch_widget(paint_app, add)
    group = painter.groups[-1]
    group_container = paint_app.resolve_widget(group.widget).down(
        test_name='shape group obj container')()

    # select the shapes
    for s in (s1, s3, s4):
        select = paint_app.resolve_widget(s.shape.widget).down(
            test_name='shape name')()
        await touch_widget(paint_app, select)

    # drag button
    drag_widget = paint_app.resolve_widget(s1.shape.widget).down(
        test_name='shape drag')()
    await exhaust(paint_app.do_touch_drag_follow(
        widget=drag_widget, target_widget=group_container,
        target_widget_loc=('center_x', 'y')))

    assert group.shapes == [s1.shape, s3.shape, s4.shape]
    assert len(group.widget.shape_widgets) == 3


async def test_group_shape_name(paint_app: CeedTestApp):
    painter = paint_app.app.shape_factory
    shapes: List[Shape] = await make_4_shapes(paint_app)
    s1, s2, s3, s4 = shapes

    # drag shape into group area
    group_container = paint_app.resolve_widget().down(
        test_name='shape group container')()
    drag_widget = paint_app.resolve_widget(s1.shape.widget).down(
        test_name='shape drag')()
    await exhaust(paint_app.do_touch_drag_follow(
        widget=drag_widget, target_widget=group_container,
        target_widget_loc=('center_x', 'y')))

    shape_widget = painter.groups[0].widget.shape_widgets[0]
    label = paint_app.resolve_widget(shape_widget).down(
        test_name='group shape item label')()
    assert label.text == s1.shape.name
    s1.shape.name = 'new name'
    assert label.text == s1.shape.name


async def test_group_name(paint_app: CeedTestApp):
    painter = paint_app.app.shape_factory

    # add a group
    add = paint_app.resolve_widget().down(test_name='add group')()
    await touch_widget(paint_app, add)
    group = painter.groups[-1]

    label = paint_app.resolve_widget(group.widget).down(
        test_name='group name')()
    textbox = paint_app.resolve_widget(group.widget).down(
        test_name='group name input')()
    assert label.text == group.name
    assert textbox.text == group.name
    assert painter.groups == [group]
    assert painter.shape_group_names[group.name] is group
    assert len(painter.shape_group_names) == 1

    group.name = 'new name'
    assert label.text == 'new name'
    assert textbox.text == 'new name'
    assert group.name == 'new name'
    assert painter.groups == [group]
    assert painter.shape_group_names['new name'] is group
    assert len(painter.shape_group_names) == 1

    await replace_text(paint_app, textbox, 'other name')
    assert label.text == 'other name'
    assert textbox.text == 'other name'
    assert group.name == 'other name'
    assert painter.groups == [group]
    assert painter.shape_group_names['other name'] is group
    assert len(painter.shape_group_names) == 1

    group.name = 'final name'
    assert label.text == 'final name'
    assert textbox.text == 'final name'
    assert group.name == 'final name'
    assert painter.groups == [group]
    assert painter.shape_group_names['final name'] is group
    assert len(painter.shape_group_names) == 1


async def test_group_duplicate(paint_app: CeedTestApp):
    from ceed.shape import CeedShapeGroup
    painter = paint_app.app.shape_factory
    shapes: List[Shape] = await make_4_shapes(paint_app)
    s1, s2, s3, s4 = shapes

    # drag shape into group area
    group_container = paint_app.resolve_widget().down(
        test_name='shape group container')()
    drag_widget = paint_app.resolve_widget(s1.shape.widget).down(
        test_name='shape drag')()
    await exhaust(paint_app.do_touch_drag_follow(
        widget=drag_widget, target_widget=group_container,
        target_widget_loc=('center_x', 'y')))

    assert len(painter.groups) == 1
    group1 = painter.groups[0]
    assert isinstance(group1, CeedShapeGroup)
    assert group1.shapes == [s1.shape]

    # copy the group by dragging it
    drag_widget = paint_app.resolve_widget(group1.widget).down(
        test_name='group drag button')()
    await exhaust(paint_app.do_touch_drag_follow(
        widget=drag_widget, target_widget=group_container,
        target_widget_loc=('center_x', 'y')))

    assert len(painter.groups) == 2
    group2 = painter.groups[1]
    assert isinstance(group2, CeedShapeGroup)
    assert len(group2.shapes) == 1
    assert group2.shapes == [s1.shape]
    assert group1 is not group2
    assert group1 in painter.groups
    assert group2 in painter.groups
    assert group1.name != group2.name


async def test_group_delete(paint_app: CeedTestApp):
    painter = paint_app.app.shape_factory
    shapes: List[Shape] = await make_4_shapes(paint_app)
    s1, s2, s3, s4 = shapes

    # drag shape into group area
    group_container = paint_app.resolve_widget().down(
        test_name='shape group container')()
    drag_widget = paint_app.resolve_widget(s1.shape.widget).down(
        test_name='shape drag')()
    await exhaust(paint_app.do_touch_drag_follow(
        widget=drag_widget, target_widget=group_container,
        target_widget_loc=('center_x', 'y')))

    assert len(painter.groups) == 1
    assert len(painter.shape_group_names) == 1
    group = painter.groups[0]
    assert painter.shape_group_names[group.name] is group
    assert painter.shapes == [shape.shape for shape in shapes]
    assert group.shapes == [s1.shape]

    # delete the group
    delete = paint_app.resolve_widget(group.widget).down(
        test_name='shape group delete')()
    await touch_widget(paint_app, delete)

    assert not painter.groups
    assert not painter.shape_group_names
    assert painter.shapes == [shape.shape for shape in shapes]


async def test_group_shape_rename(paint_app: CeedTestApp):
    painter = paint_app.app.shape_factory
    shapes: List[Shape] = await make_4_shapes(paint_app)
    s1, s2, s3, s4 = shapes
