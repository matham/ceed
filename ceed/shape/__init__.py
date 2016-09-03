from collections import OrderedDict

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.behaviors.togglebutton import ToggleButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty
from kivy.core.window import Window
from kivy.factory import Factory

from cplcom.painter import PaintCanvas, PaintCircle, PaintEllipse, \
    PaintPolygon, PaintBezier

from ceed.utils import WidgetList


class CeedPainter(KNSpaceBehavior, PaintCanvas):

    shape_map = OrderedDict()

    show_label = BooleanProperty(False)

    pos_label = None

    def __init__(self, **kwargs):
        super(CeedPainter, self).__init__(**kwargs)
        self.shape_map = OrderedDict()
        self.pos_label = Factory.XYSizedLabel()

    def add_shape(self, shape, **kwargs):
        if not super(CeedPainter, self).add_shape(shape, **kwargs):
            return False

        shape_map = self.shape_map
        widget = shape_map[shape] = WidgetShape(
            index='{}'.format(len(shape_map)))
        widget.connect_shape(shape, self)
        return True

    def remove_shape(self, shape):
        if not super(CeedPainter, self).remove_shape(shape):
            return False

        shape_map = self.shape_map
        widget = shape_map.pop(shape)
        widget.disconnect_shape()

        for i, widget in enumerate(shape_map.values()):
            widget.index = '{}'.format(i)
        return True

    def select_shape(self, shape):
        if not super(CeedPainter, self).select_shape(shape):
            return False
        knspace.shapes.select_node(self.shape_map[shape])
        return True

    def deselect_shape(self, shape):
        if not super(CeedPainter, self).deselect_shape(shape):
            return False
        knspace.shapes.deselect_node(self.shape_map[shape])
        return True

    def on_show_label(self, *largs):
        state = self.show_label
        for w in self.shape_map.values():
            w.show_label = state

        label = self.pos_label
        if state:
            Window.add_widget(label)
            Window.fbind('mouse_pos', self._update_mouse_pos)
            self._update_mouse_pos(None, Window.mouse_pos)
        else:
            Window.remove_widget(label)
            Window.funbind('mouse_pos', self._update_mouse_pos)

    def _update_mouse_pos(self, instance, pos):
        x, y = map(int, self.to_widget(*pos))
        if self.collide_point(x, y):
            self.pos_label.pos = pos
            self.pos_label.text = '{}, {}'.format(x, y)
        else:
            self.pos_label.text = ''


class ShapeList(WidgetList, BoxLayout):

    def select_node(self, node):
        if super(ShapeList, self).select_node(node):
            knspace.painter.select_shape(node.shape)
            node.selected = True
            return True
        return False

    def deselect_node(self, node):
        if super(ShapeList, self).deselect_node(node):
            knspace.painter.deselect_shape(node.shape)
            node.selected = False
            return True
        return False


class WidgetShape(BoxLayout):

    painter = None

    shape = None

    label = None

    show_label = BooleanProperty(False)

    index = StringProperty('0')

    _name = StringProperty('')

    name = StringProperty('')

    show_more = BooleanProperty(False)

    centroid_x = NumericProperty(0)

    centroid_y = NumericProperty(0)

    more = ObjectProperty(None)

    selected = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(WidgetShape, self).__init__(**kwargs)
        self.fbind('show_label', self._show_label)
        self.fbind('index', self._label_text)
        self.fbind('_name', self._label_text)
        self.fbind('show_more', self._show_more)
        self.remove_widget(self.more)

    def connect_shape(self, shape, painter):
        self.painter = painter
        self.shape = shape

        label = self.label = Label()
        knspace.shapes.add_widget(self)
        shape.fbind('on_update', self._shape_update)
        label.fbind('size', self._shape_update)

        self._label_text()
        self._shape_update()
        if self.show_label:
            self._show_label()

    def disconnect_shape(self):
        self.shape.funbind('on_update', self._shape_update)
        knspace.shapes.remove_widget(self)

        if self.show_label:
            label = self.label
            label.funbind('size', self._shape_update)
            self.painter.remove_widget(label)

    def _show_label(self, *largs):
        if self.show_label:
            self.painter.add_widget(self.label)
            self._shape_update()
            self._label_text()
        else:
            self.painter.remove_widget(self.label)

    def _label_text(self, *largs):
        name = self.name = self._name if self._name else self.index
        if self.show_label:
            self.label.text = name

    def _show_more(self, *largs):
        if self.show_more:
            self.add_widget(self.more)
            self._shape_update()
        else:
            self.remove_widget(self.more)

    def _shape_update(self, *largs):
        if self.show_label or self.show_more:
            self.centroid_x, self.centroid_y = self.label.center = \
                tuple(map(round, self.shape.centroid))

    def _update_centroid(self, x=None, y=None):
        x1, y1 = map(round, self.shape.centroid)
        dx = 0 if x is None else x - x1
        dy = 0 if y is None else y - y1
        if dx or dy:
            self.shape.translate(dpos=(dx, dy))


class ShapeDisplayRoot(BoxLayout):

    def on_touch_up(self, touch):
        if super(ShapeDisplayRoot, self).on_touch_up(touch):
            return True
        if self.collide_point(*touch.pos):
            knspace.shapes.select_with_touch(self.parent, touch)
            return True
        return False
