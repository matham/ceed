import math
from typing import Type, List


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
