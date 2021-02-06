import math
from typing import Type, List, Union
from ceed.shape import CeedPaintCanvasBehavior, CeedShape
from ceed.tests.ceed_app import CeedTestApp
from kivy_garden.painter import PaintShape


def assert_add_three_shapes(
        shape_factory: CeedPaintCanvasBehavior = None,
        app: CeedTestApp = None, show_in_gui=False):
    assert not shape_factory.shapes
    assert not shape_factory.shape_names

    shape = EllipseShapeP1(
        app=app, painter=shape_factory, show_in_gui=show_in_gui)
    shape2 = PolygonShapeP1(
        app=app, painter=shape_factory, show_in_gui=show_in_gui)
    shape3 = CircleShapeP1(
        app=app, painter=shape_factory, show_in_gui=show_in_gui)

    if not show_in_gui:
        shape.make_shape()
        shape2.make_shape()
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


def assert_add_three_groups(
        shape_factory: CeedPaintCanvasBehavior = None,
        app: CeedTestApp = None, show_in_gui=False):
    shape, shape2, shape3 = assert_add_three_shapes(
        shape_factory, app, show_in_gui)
    assert not shape_factory.groups
    assert not shape_factory.shape_group_names

    group = shape_factory.add_group()
    group2 = shape_factory.add_group()
    group3 = shape_factory.add_group()

    group.add_shape(shape.shape)
    group.add_shape(shape2.shape)
    group2.add_shape(shape2.shape)
    group2.add_shape(shape3.shape)
    group3.add_shape(shape.shape)
    group3.add_shape(shape2.shape)
    group3.add_shape(shape3.shape)

    assert shape_factory.groups == [group, group2, group3]
    assert len(shape_factory.groups) == 3
    assert len(shape_factory.shape_group_names) == 3
    for g in (group, group2, group3):
        assert shape_factory.shape_group_names[g.name] is g

    assert group.shapes == [shape.shape, shape2.shape]
    assert group2.shapes == [shape2.shape, shape3.shape]
    assert group3.shapes == [shape.shape, shape2.shape, shape3.shape]

    return (group, group2, group3), (shape, shape2, shape3)


class Shape:

    painter: CeedPaintCanvasBehavior = None

    app: CeedTestApp = None

    shape: Union[CeedShape, PaintShape] = None

    drag_point = None

    test_points = []

    outside_point = None

    name = ''

    activate_btn_name = ''

    area = 0

    center = None

    bounding_box = []

    inside_point = None

    def __init__(self, app, painter, show_in_gui=True, create_add_shape=False):
        super(Shape, self).__init__()
        self.painter = painter
        self.app = app

        if show_in_gui:
            self.show_in_gui()
        elif create_add_shape:
            self.create_add_shape()

    def make_shape(self):
        raise NotImplementedError

    def create_add_shape(self):
        if self.shape is None:
            self.make_shape()
        self.painter.add_shape(self.shape)

    def show_in_gui(self):
        self.create_add_shape()
        self.shape.add_shape_to_canvas(self.painter)

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
        r = [p[0] / 255 for p in points]
        g = [p[1] / 255 for p in points]
        b = [p[2] / 255 for p in points]
        n = len(r)
        return sum(r) / n, sum(g) / n, sum(b) / n

    def assert_shape_visible_selected(self):
        r, g, b = self.get_mean_visible_pixel_intensity()

    def assert_shape_visible_not_selected(self):
        r, g, b = self.get_mean_visible_pixel_intensity()

    def assert_shape_prop_same(self, compare_name=False):
        raise NotImplementedError


class CircleShape(Shape):

    radius = 100

    area = math.pi * 100 ** 2

    activate_btn_name = 'draw circle'

    def make_shape(self):
        self.shape = self.painter.create_shape(
            'circle', center=self.center, radius=self.radius, name=self.name)
        return self.shape

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

    def assert_shape_prop_same(self, compare_name=False):
        from ceed.shape import CeedPaintCircle
        from kivy_garden.painter import PaintCircle
        assert isinstance(self.shape, CeedPaintCircle)
        assert isinstance(self.shape, PaintCircle)
        if compare_name:
            assert self.shape.name == self.name

        assert self.shape.center == self.center
        assert self.shape.radius == self.radius


class EllipseShape(Shape):

    radius_x = 100

    radius_y = 50

    area = math.pi * radius_x * radius_y

    activate_btn_name = 'draw ellipse'

    def make_shape(self):
        self.shape = self.painter.create_shape(
            'ellipse', center=self.center, radius_x=self.radius_x,
            radius_y=self.radius_y, name=self.name)
        return self.shape

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

    def assert_shape_prop_same(self, compare_name=False):
        from ceed.shape import CeedPaintEllipse
        from kivy_garden.painter import PaintEllipse
        assert isinstance(self.shape, CeedPaintEllipse)
        assert isinstance(self.shape, PaintEllipse)
        if compare_name:
            assert self.shape.name == self.name

        assert self.shape.center == self.center
        assert self.shape.radius_x == self.radius_x
        assert self.shape.radius_y == self.radius_y


class PolygonShape(Shape):

    points = []

    activate_btn_name = 'draw polygon'

    def make_shape(self):
        self.shape = self.painter.create_shape(
            'polygon', points=self.points, selection_point=self.points[:2],
            name=self.name)
        return self.shape

    async def draw(self):
        # place until last point
        for x, y in zip(self.points[:-2:2], self.points[1:-2:2]):
            async for _ in self.app.do_touch_down_up(
                    pos=(x, y), widget=self.painter):
                pass
            await self.app.wait_clock_frames(2)

        # put final point
        async for _ in self.app.do_touch_down_up(
                pos=self.points[-2:], widget=self.painter, duration=0):
            pass
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
            assert new_x * 0.95 < x < new_x * 1.05

            if oy < cy:
                new_y = oy + (1 - factor) * (cy - oy)
            else:
                new_y = oy - (1 - factor) * (oy - cy)
            assert new_y * 0.95 < y < new_y * 1.05

    def assert_shape_prop_same(self, compare_name=False):
        from ceed.shape import CeedPaintPolygon
        from kivy_garden.painter import PaintPolygon
        assert isinstance(self.shape, CeedPaintPolygon)
        assert isinstance(self.shape, PaintPolygon)
        if compare_name:
            assert self.shape.name == self.name

        assert self.shape.points == self.points
        assert self.shape.selection_point == self.points[:2]


class FreeofrmPolygonShape(PolygonShape):

    activate_btn_name = 'draw freeform'

    def make_shape(self):
        self.shape = self.painter.create_shape(
            'freeform', points=self.points, selection_point=self.points[:2],
            name=self.name)
        return self.shape

    async def draw(self):
        # place until last point
        points = list(zip(self.points[::2], self.points[1::2]))
        async for _ in self.app.do_touch_drag_path(
                path=points, axis_widget=self.painter, duration=0):
            pass
        self.shape = self.painter.shapes[-1]

    def assert_shape_prop_same(self, compare_name=False):
        from ceed.shape import CeedPaintFreeformPolygon
        from kivy_garden.painter import PaintFreeformPolygon
        assert isinstance(self.shape, CeedPaintFreeformPolygon)
        assert isinstance(self.shape, PaintFreeformPolygon)
        if compare_name:
            assert self.shape.name == self.name

        assert self.shape.points == self.points
        assert self.shape.selection_point == self.points[:2]


class CircleShapeP1(CircleShape):

    center = [450, 300]

    test_points = [550, 300, 450, 200, 350, 300]

    outside_point = [450, 450]

    inside_point = center

    name = 'happy circle'

    drag_point = test_points[:2]

    bounding_box = (350, 200, 551, 401)

    area = math.pi * CircleShape.radius ** 2


class CircleShapeP1Internal(CircleShape):

    center = [450 + CircleShape.radius / 2, 300]

    outside_point = CircleShapeP1.outside_point

    inside_point = center

    name = 'happy circle internal'

    radius = CircleShape.radius / 3


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

    name = 'happy ellipse'

    drag_point = test_points[:2]


class EllipseShapeP2(EllipseShape):

    center = [700, 600]

    test_points = [800, 600, 700, 550, 600, 600]

    outside_point = [700, 750]

    inside_point = center

    name = 'sad ellipse'

    drag_point = test_points[:2]


class PolygonShapeP1(PolygonShape):

    points = [10, 10, 300, 10, 300, 300, 10, 300]

    center = [150, 150]

    test_points = points

    outside_point = [200, 350]

    inside_point = [200, 200]

    name = 'happy shape'

    area = 290 * 290

    drag_point = test_points[:2]


class PolygonShapeP2(PolygonShape):

    points = [10, 400, 300, 400, 300, 700, 10, 700]

    center = [150, 550]

    test_points = points

    outside_point = [200, 800]

    inside_point = [200, 500]

    name = 'sad shape'

    area = 290 * 300

    drag_point = test_points[:2]


class FreeformPolygonShapeP1(FreeofrmPolygonShape):

    points = [900, 10, 1200, 10, 1200, 300, 900, 300]

    center = [1050, 150]

    test_points = points

    outside_point = [1100, 350]

    inside_point = [1100, 200]

    name = 'eating shape'

    area = 290 * 300

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
        self.shape = shape
        self.points = self.test_points = points
        self.outside_point = outside_point
        super(EnclosingPolygon, self).__init__(*largs, **kwargs)


paired_tests = [
    (PolygonShapeP1, PolygonShapeP2),
    (CircleShapeP1, CircleShapeP2),
    (EllipseShapeP1, EllipseShapeP2),
    (FreeformPolygonShapeP1, FreeformPolygonShapeP2),
]

paired_tests_one = [
    (CircleShapeP1, CircleShapeP2),
]

shape_classes = [
    PolygonShapeP1, PolygonShapeP2,
    CircleShapeP1, CircleShapeP2,
    EllipseShapeP1, EllipseShapeP2,
    FreeformPolygonShapeP1, FreeformPolygonShapeP2,
]
