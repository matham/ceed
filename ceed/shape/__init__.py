
from itertools import product

from kivy.event import EventDispatcher
from kivy.garden.collider import CollideEllipse
from kivy.uix.behaviors.knspace import KNSpaceBehavior
from cplcom.painter import PaintCanvas, PaintCircle, PaintEllipse, \
    PaintPolygon, PaintFreeform, PaintBezier


class CeedPainter(KNSpaceBehavior, PaintCanvas):
    pass


class Shape(EventDispatcher):

    _collider = None

    canvas_size = 640, 480

    def __init__(self, canvas_size=(640, 480), **kwargs):
        super(Shape, self).__init__(**kwargs)
        self.canvas_size = canvas_size

    def get_config_widget(self):
        pass

    def get_canvas_elements(self):
        pass

    def edit_shape(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._collider = None

    @property
    def shape(self):
        return self.__class__.__name__

    def get_inside_points(self):
        points = []
        append = points.append
        collide = self.collider.collide_point
        for x, y in product(
                range(self.canvas_size[0]), range(self.canvas_size[1])):
            if collide(x, y):
                append((x, y))
        return points

    @property
    def collider(self):
        pass


class Ellipse(Shape):

    x = y = 0

    rx = ry = 0

    def __init__(self, x=0, y=0, rx=0, ry=0, **kwargs):
        super(Ellipse, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.rx = rx
        self.ry = ry

    @property
    def collider(self):
        if self._collider is None:
            self._collider = CollideEllipse(self.x, self.y, self.rx, self.ry)
        return self._collider
