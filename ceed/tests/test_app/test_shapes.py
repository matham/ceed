import pytest
import math
from typing import Type, List
from ceed.tests.ceed_app import CeedTestApp


class Shape(object):

    painter = None

    app = None

    shape = None

    drag_point = None

    test_points = []

    outside_point = None

    name = ''

    activate_btn_name = ''

    area = 0

    center = None

    inside_point = None

    def __init__(self, app, painter, manually_add=True):
        super(Shape, self).__init__()
        self.painter = painter
        self.app = app
        if manually_add:
            self.manually_add()

    def manually_add(self):
        raise NotImplementedError

    def remove(self):
        self.painter.remove_shape(self.shape)

    def check_shape_visible(self, assert_visible, offset=0, pad=3):
        test_points = []
        orig_points = list(zip(self.test_points[::2], self.test_points[1::2]))
        for x, y in orig_points:
            points = []
            for x_ in range(max(0, x + offset - pad), x + offset + pad + 1):
                for y_ in range(
                        max(0, y + offset - pad), y + offset + pad + 1):
                    points.append((x_, y_))
            test_points.append(points)
        lengths = list(map(len, test_points))

        flat_points = []
        for points in test_points:
            flat_points.extend(points)
        flat_points.append([p + offset for p in self.outside_point])

        points = self.app.get_widget_pos_pixel(self.painter, flat_points)

        region_max = []
        s = 0
        for size in lengths:
            region_max.append((
                max((p[0] for p in points[s: s + size] if p)),
                max((p[1] for p in points[s: s + size] if p)),
                max((p[2] for p in points[s: s + size] if p)),
                max((p[3] for p in points[s: s + size] if p)),
            ))
            s += size

        if assert_visible:
            for color, point in zip(region_max, orig_points):
                assert max(color[:3]) > 50, str(point)
        else:
            for color, point in zip(region_max, orig_points):
                assert max(color[:3]) == 0, str(point)
        assert max(points[-1][:3]) == 0

    async def draw(self):
        raise NotImplementedError

    def check_resize_by_area(self, ratio):
        raise NotImplementedError

    def get_mean_visible_pixel_intensity(self, pad=3):
        test_points = []
        orig_points = list(zip(self.test_points[::2], self.test_points[1::2]))
        for x, y in orig_points:
            for x_ in range(max(0, x - pad), x + pad + 1):
                for y_ in range(max(0, y - pad), y + pad + 1):
                    test_points.append((x_, y_))

        points = self.app.get_widget_pos_pixel(self.painter, test_points)
        pixels_vals = [max(p[:3]) for p in points]
        return sum(pixels_vals) / len(pixels_vals)


class CircleShape(Shape):

    radius = 100

    area = math.pi * 100 ** 2

    activate_btn_name = 'draw circle'

    def manually_add(self):
        self.shape = self.painter.create_add_shape(
            'circle', center=self.center, radius=self.radius, name=self.name)

    async def draw(self):
        # place center
        async for _ in self.app.do_touch_down_up(
                pos=self.center, widget=self.painter):
            pass
        await self.app.wait_clock_frames(2)
        self.shape = self.painter.shapes[-1]

        # increase size
        x, y = self.center
        async for _ in self.app.do_touch_drag(
                pos=(x + self.shape.radius, y), widget=self.painter,
                long_press=self.painter.long_touch_delay + 0.2,
                dx=self.radius - self.shape.radius):
            pass

        # deselect
        async for _ in self.app.do_touch_down_up(
                pos=self.outside_point, widget=self.painter):
            pass

    def check_resize_by_area(self, ratio):
        new_radius = math.sqrt(ratio) * self.radius
        assert new_radius * 0.99 < self.shape.radius < new_radius * 1.01


class EllipseShape(Shape):

    radius_x = 100

    radius_y = 50

    area = math.pi * radius_x * radius_y

    activate_btn_name = 'draw ellipse'

    def manually_add(self):
        self.shape = self.painter.create_add_shape(
            'ellipse', center=self.center, radius_x=self.radius_x,
            radius_y=self.radius_y, name=self.name)

    async def draw(self):
        # place center
        async for _ in self.app.do_touch_down_up(
                pos=self.center, widget=self.painter):
            pass
        await self.app.wait_clock_frames(2)
        self.shape = self.painter.shapes[-1]

        # increase size
        x, y = self.center
        async for _ in self.app.do_touch_drag(
                pos=(x + self.shape.radius_x, y), widget=self.painter,
                long_press=self.painter.long_touch_delay + 0.2,
                dx=self.radius_x - self.shape.radius_x):
            pass
        async for _ in self.app.do_touch_drag(
                pos=(x, y + self.shape.radius_y), widget=self.painter,
                long_press=self.painter.long_touch_delay + 0.2,
                dy=self.radius_y - self.shape.radius_y):
            pass

        # deselect
        async for _ in self.app.do_touch_down_up(
                pos=self.outside_point, widget=self.painter):
            pass

    def check_resize_by_area(self, ratio):
        new_radius_x = math.sqrt(ratio) * self.radius_x
        assert new_radius_x * 0.99 < self.shape.radius_x < new_radius_x * 1.01
        new_radius_y = math.sqrt(ratio) * self.radius_y
        assert new_radius_y * 0.99 < self.shape.radius_y < new_radius_y * 1.01


class PolygonShape(Shape):

    points = []

    activate_btn_name = 'draw polygon'

    def manually_add(self):
        self.shape = self.painter.create_add_shape(
            'polygon', points=self.points, selection_point=self.points[:2],
            name=self.name)

    async def draw(self):
        # place until last point
        for x, y in zip(self.points[:-2:2], self.points[1:-2:2]):
            async for _ in self.app.do_touch_down_up(
                    pos=(x, y), widget=self.painter):
                pass
            await self.app.wait_clock_frames(2)

        # put final point
        async for _ in self.app.do_touch_down_up(
                pos=self.points[-2:], widget=self.painter, duration=0.01):
            pass
        await self.app.wait_clock_frames(2)
        async for _ in self.app.do_touch_down_up(
                pos=self.points[-2:], widget=self.painter, duration=0.01):
            pass
        self.shape = self.painter.shapes[-1]

    def check_resize_by_area(self, ratio):
        factor = math.sqrt(ratio)
        points = list(zip(self.shape.points[::2], self.shape.points[1::2]))
        opoints = list(zip(self.points[::2], self.points[1::2]))
        cx, cy = self.center

        for (x, y), (ox, oy) in zip(points, opoints):
            if ox < cx:
                new_x = ox + (1 - factor) * (cx - ox)
            else:
                new_x = ox - (1 - factor) * (ox - cx)
            assert new_x * 0.99 < x < new_x * 1.01

            if oy < cy:
                new_y = oy + (1 - factor) * (cy - oy)
            else:
                new_y = oy - (1 - factor) * (oy - cy)
            assert new_y * 0.99 < y < new_y * 1.01


class FreeofrmPolygonShape(PolygonShape):

    activate_btn_name = 'draw freeform'

    def manually_add(self):
        self.shape = self.painter.create_add_shape(
            'freeform', points=self.points, selection_point=self.points[:2],
            name=self.name)

    async def draw(self):
        # place until last point
        points = list(zip(self.points[::2], self.points[1::2]))
        async for _ in self.app.do_touch_drag_path(
                path=points, axis_widget=self.painter):
            pass
        self.shape = self.painter.shapes[-1]


class CircleShapeP1(CircleShape):

    center = [450, 300]

    test_points = [550, 300, 450, 200, 350, 300]

    outside_point = [450, 450]

    inside_point = center

    name = 'happy circle'

    drag_point = test_points[:2]


class CircleShapeP2(CircleShape):

    center = [450, 600]

    test_points = [550, 600, 450, 500, 350, 600]

    outside_point = [450, 750]

    inside_point = center

    name = 'sad circle'

    drag_point = test_points[:2]


class EllipseShapeP1(EllipseShape):

    center = [700, 300]

    test_points = [800, 300, 700, 250, 600, 300]

    outside_point = [700, 450]

    inside_point = center

    name = 'happy circle'

    drag_point = test_points[:2]


class EllipseShapeP2(EllipseShape):

    center = [700, 600]

    test_points = [800, 600, 700, 550, 600, 600]

    outside_point = [700, 750]

    inside_point = center

    name = 'sad circle'

    drag_point = test_points[:2]


class PolygonShapeP1(PolygonShape):

    points = [0, 0, 300, 0, 300, 300, 0, 300]

    center = [150, 150]

    test_points = points

    outside_point = [200, 350]

    inside_point = [200, 200]

    name = 'happy shape'

    area = 300 * 300

    drag_point = test_points[:2]


class PolygonShapeP2(PolygonShape):

    points = [0, 400, 300, 400, 300, 700, 0, 700]

    center = [150, 550]

    test_points = points

    outside_point = [200, 800]

    inside_point = [200, 500]

    name = 'sad shape'

    area = 300 * 300

    drag_point = test_points[:2]


class FreeformPolygonShapeP1(FreeofrmPolygonShape):

    points = [900, 0, 1200, 0, 1200, 300, 900, 300]

    center = [1050, 150]

    test_points = points

    outside_point = [1100, 350]

    inside_point = [1100, 200]

    name = 'eating shape'

    area = 300 * 300

    drag_point = test_points[:2]


class FreeformPolygonShapeP2(FreeofrmPolygonShape):

    points = [900, 400, 1200, 400, 1200, 700, 900, 700]

    center = [1050, 550]

    test_points = points

    outside_point = [1100, 800]

    inside_point = [1100, 500]

    name = 'sleeping shape'

    area = 300 * 300

    drag_point = test_points[:2]


class EnclosingPolygon(PolygonShape):

    def __init__(self, *largs, shape=None, points=(), outside_point=(),
                 **kwargs):
        super(EnclosingPolygon, self).__init__(*largs, **kwargs)
        self.shape = shape
        self.points = self.test_points = points
        self.outside_point = outside_point


paired_tests = [
    (PolygonShapeP1, PolygonShapeP2),
    (CircleShapeP1, CircleShapeP2),
    (EllipseShapeP1, EllipseShapeP2),
    (FreeformPolygonShapeP1, FreeformPolygonShapeP2),
]

@pytest.fixture
async def paint_app(ceed_app: CeedTestApp):
    from kivy.metrics import dp
    await ceed_app.wait_clock_frames(2)

    assert ceed_app.shape_factory is not None
    assert not ceed_app.shape_factory.shapes

    painter_widget = ceed_app.resolve_widget().down(
        test_name='painter')()
    assert tuple(painter_widget.size) == (
        ceed_app.view_controller.screen_width,
        ceed_app.view_controller.screen_height)

    # expand shape splitter so shape widgets are fully visible
    splitter = ceed_app.resolve_widget().down(
        test_name='shape splitter')().children[-1]
    async for _ in ceed_app.do_touch_drag(widget=splitter, dx=-dp(100)):
        pass
    await ceed_app.wait_clock_frames(2)

    # expand shape splitter so shape widgets are fully visible
    slider = ceed_app.resolve_widget().down(test_name='screen zoom silder')()
    slider.value = slider.min
    await ceed_app.wait_clock_frames(2)

    yield ceed_app


async def replace_text(app, text_widget, new_text):
    # activate it
    async for _ in app.do_touch_down_up(widget=text_widget):
        pass

    # select all
    ctrl_it = app.do_keyboard_key(key='lctrl', modifiers=['ctrl'])
    await ctrl_it.__anext__()  # down
    async for _ in app.do_keyboard_key(key='a', modifiers=['ctrl']):
        pass
    await ctrl_it.__anext__()  # up

    # replace text
    for key in ['delete'] + list(new_text) + ['enter']:
        async for _ in app.do_keyboard_key(key=key):
            pass


@pytest.mark.parametrize(
    "shape_cls",
    [PolygonShapeP1, CircleShapeP1, EllipseShapeP1, FreeformPolygonShapeP1])
async def test_simple_shape(ceed_app: CeedTestApp, shape_cls):
    painter = ceed_app.shape_factory
    assert ceed_app.shape_factory == painter
    assert not ceed_app.shape_factory.shapes

    shape = shape_cls(ceed_app, painter)
    await ceed_app.wait_clock_frames(2)

    assert shape.shape in ceed_app.shape_factory.shapes
    assert shape.shape.name == shape.name
    shape.check_shape_visible(True)

    shape.remove()
    await ceed_app.wait_clock_frames(2)

    assert shape.shape not in ceed_app.shape_factory.shapes
    assert not ceed_app.shape_factory.shapes
    shape.check_shape_visible(False)

    painter.add_shape(shape.shape)
    assert shape.shape in ceed_app.shape_factory.shapes
    assert shape.shape.name == shape.name

    assert not shape.shape.locked
    painter.lock_shape(shape.shape)
    assert shape.shape.locked
    painter.unlock_shape(shape.shape)
    assert not shape.shape.locked


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_multiple_shapes_add(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_multiple_shapes_remove(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
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
    shape1.manually_add()
    assert painter.shapes == [shape2.shape, shape1.shape]
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_multiple_shapes_move_depth(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
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


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_multiple_shapes_duplicate(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
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


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_lock(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
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


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_hide(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
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


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_name(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
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
    painter = paint_app.shape_factory
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
    painter = paint_app.shape_factory
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
    painter = paint_app.shape_factory
    shape1, shape2 = (
        cls(paint_app, painter, manually_add=False) for cls in shape_classes)
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
    painter = paint_app.shape_factory
    assert not painter.shapes

    # set the correct draw mode
    add = paint_app.resolve_widget().down(
        test_name='add enclosing polygon')()
    async for _ in paint_app.do_touch_down_up(widget=add):
        pass
    await paint_app.wait_clock_frames(2)

    assert len(painter.shapes) == 1
    w = paint_app.view_controller.screen_width
    h = paint_app.view_controller.screen_height
    points = [0, 0, w, 0, w, h, 0, h]
    shape = EnclosingPolygon(
        app=paint_app, painter=painter, shape=painter.shapes[0],
        points=points, outside_point=[w / 2, h / 2], manually_add=False)

    assert shape.shape.name == 'enclosed'
    shape.check_shape_visible(True)


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_duplicate_selection(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
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


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_delete_selection(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
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


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_single_select_widget(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    noselect1 = shape1.get_mean_visible_pixel_intensity()
    noselect2 = shape2.get_mean_visible_pixel_intensity()
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
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() \
       < noselect2 * 1.01

    # select second shape
    async for _ in paint_app.do_touch_down_up(widget=shape2_widget):
        pass
    await paint_app.wait_clock_frames(2)
    painter.export_to_png(r'E:\img3.png')

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    assert noselect1 * .99 < shape1.get_mean_visible_pixel_intensity() < \
           noselect1 * 1.01
    assert noselect2 * 1.35 < shape2.get_mean_visible_pixel_intensity()
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
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
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
    assert noselect1 * .99 < shape1.get_mean_visible_pixel_intensity() < \
           noselect1 * 1.01
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
    assert not shape1.shape.selected
    assert not shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_single_select_shape(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    noselect1 = shape1.get_mean_visible_pixel_intensity()
    noselect2 = shape2.get_mean_visible_pixel_intensity()
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
            widget=painter, pos=shape1.drag_point):
        pass
    await paint_app.wait_clock_frames(2)

    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
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
    assert noselect1 * .99 < shape1.get_mean_visible_pixel_intensity() < \
           noselect1 * 1.01
    assert noselect2 * 1.35 < shape2.get_mean_visible_pixel_intensity()
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
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
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
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_multiselect_widget(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    noselect1 = shape1.get_mean_visible_pixel_intensity()
    noselect2 = shape2.get_mean_visible_pixel_intensity()
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
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
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
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * 1.35 < shape2.get_mean_visible_pixel_intensity()
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
    assert noselect1 * .99 < shape1.get_mean_visible_pixel_intensity() < \
           noselect1 * 1.01
    assert noselect2 * 1.35 < shape2.get_mean_visible_pixel_intensity()
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
    assert noselect1 * .99 < shape1.get_mean_visible_pixel_intensity() < \
           noselect1 * 1.01
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
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
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected


@pytest.mark.parametrize("shape_classes", paired_tests)
async def test_shape_multiselect_shape(
        paint_app: CeedTestApp, shape_classes: List[Type[Shape]]):
    painter = paint_app.shape_factory
    shape1, shape2 = (cls(paint_app, painter) for cls in shape_classes)
    await paint_app.wait_clock_frames(2)

    assert painter.shapes == [shape1.shape, shape2.shape]
    shape1.check_shape_visible(True)
    shape2.check_shape_visible(True)
    noselect1 = shape1.get_mean_visible_pixel_intensity()
    noselect2 = shape2.get_mean_visible_pixel_intensity()
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
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
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
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * 1.35 < shape2.get_mean_visible_pixel_intensity()
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
    assert noselect1 * .99 < shape1.get_mean_visible_pixel_intensity() < \
           noselect1 * 1.01
    assert noselect2 * 1.35 < shape2.get_mean_visible_pixel_intensity()
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
    assert noselect1 * .99 < shape1.get_mean_visible_pixel_intensity() < \
           noselect1 * 1.01
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
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
    assert noselect1 * 1.35 < shape1.get_mean_visible_pixel_intensity()
    assert noselect2 * .99 < shape2.get_mean_visible_pixel_intensity() < \
           noselect2 * 1.01
    assert shape1.shape.selected
    assert shape1.shape.widget.selected
    assert not shape2.shape.selected
    assert not shape2.shape.widget.selected
