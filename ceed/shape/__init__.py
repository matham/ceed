from collections import OrderedDict

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.behaviors.togglebutton import ToggleButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty
from kivy.core.window import Window
from kivy.factory import Factory

from cplcom.painter import PaintCanvas, PaintCircle, PaintEllipse, \
    PaintPolygon, PaintBezier

from ceed.utils import WidgetList, ShowMoreSelection, BoxSelector, \
    ShowMoreBehavior, fix_name


class CeedPainter(KNSpaceBehavior, PaintCanvas):

    shape_map = DictProperty({})

    show_label = BooleanProperty(False)

    pos_label = None

    def __init__(self, **kwargs):
        super(CeedPainter, self).__init__(**kwargs)
        self.pos_label = Factory.XYSizedLabel()

    def add_shape(self, shape, **kwargs):
        if not super(CeedPainter, self).add_shape(shape, **kwargs):
            return False

        shape.name = fix_name(
            shape.name, knspace.shapes.names, knspace.shape_groups.names)
        self.shape_map[shape] = WidgetShape(shape=shape, painter=self)
        knspace.shapes._update_names()
        return True

    def remove_shape(self, shape):
        if not super(CeedPainter, self).remove_shape(shape):
            return False

        self.shape_map.pop(shape).remove()
        knspace.shapes._update_names()
        knspace.shape_groups.remove_shape_from_groups(shape)
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

    def get_state(self):
        return self.save_shapes(), knspace.shape_groups.get_state()

    def set_state(self, state):
        shapes, groups = state
        self.restore_shapes(shapes)
        shape_map = {s.name: s for s in self.shapes}
        knspace.shape_groups.set_state(groups, shape_map)


class ShapeGroupList(ShowMoreSelection, WidgetList, BoxLayout):

    names = ListProperty([])

    recurse_select = True

    def add_group(self):
        group = WidgetShapeGroup()
        self.add_widget(group)
        self._update_names()
        return group

    def remove_group(self, group):
        self.deselect_node(group)
        self.remove_widget(group)
        self._update_names()

    def add_selected_shapes(self):
        if not self.selected_nodes:
            group = self.add_group()
            for shape in self.knspace.painter.selected_shapes:
                group.add_shape(shape)
            return

        node = self.selected_nodes[-1]
        if isinstance(node, ShapeGroupItem):
            node = node.group

        for shape in self.knspace.painter.selected_shapes:
            node.add_shape(shape)

    def remove_shape_from_groups(self, shape):
        for group in self.children:
            if shape in group.shapes:
                group.remove_shape(shape=shape)

    def deselect_shape_everywhere(self, shape):
        for group in self.children:
            for widget in group.shape_widgets:
                if widget.selected and widget.shape is shape:
                    self.deselect_node(widget)

    def count_shape_selection(self, shape):
        count = 0
        for group in self.children:
            for widget in group.shape_widgets:
                if widget.selected and widget.shape is shape:
                    count += 1
        return count

    def _change_group_name(self, group, name):
        if group.name == name:
            return name
        name = fix_name(name, knspace.shapes.names, self.names)
        group.name = name
        self._update_names()
        return name

    def _update_names(self, *largs):
        self.names = [group.name for group in reversed(self.children)]

    def select_node(self, shape_widget):
        if not super(ShapeGroupList, self).select_node(shape_widget):
            return False

        if isinstance(shape_widget, WidgetShapeGroup):
            for s in shape_widget.shape_widgets:
                self.select_node(s)
        else:
            knspace.painter.select_shape(shape_widget.shape)
        return True

    def deselect_node(self, shape_widget, keep_all=False):
        if not super(ShapeGroupList, self).deselect_node(shape_widget):
            return False

        if isinstance(shape_widget, WidgetShapeGroup):
            if not keep_all:
                for s in shape_widget.shape_widgets:
                    self.deselect_node(s)
        else:
            if not self.count_shape_selection(shape_widget.shape):
                knspace.painter.deselect_shape(shape_widget.shape)
            if shape_widget.group.selected:
                self.deselect_node(shape_widget.group, keep_all=True)
        return True

    def get_selectable_nodes(self):
        nodes = []
        for child in reversed(self.children):
            nodes.append(child)
            if child.show_more:
                for c in reversed(child.shape_widgets):
                    nodes.append(c)
        return nodes

    def get_state(self):
        return [group.get_state() for group in reversed(self.children)]

    def set_state(self, state, shape_name_map):
        for group_state in state:
            group = self.add_group()
            group.set_state(group_state, shape_name_map)


class WidgetShapeGroup(ShowMoreBehavior, BoxLayout):

    _name_count = 0

    selected = BooleanProperty(False)

    name = StringProperty('')

    shapes = []

    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'G{}'.format(WidgetShapeGroup._name_count)
            WidgetShapeGroup._name_count += 1
        super(WidgetShapeGroup, self).__init__(**kwargs)
        self.shapes = []
        self.name = fix_name(
            self.name, knspace.shapes.names, knspace.shape_groups.names)

    @property
    def shape_widgets(self):
        return self.more.children[:-1]

    def add_shape(self, shape):
        if shape in self.shapes:
            return

        self.more.add_widget(ShapeGroupItem(shape=shape, group=self))
        self.shapes.append(shape)

    def remove_shape(self, shape=None, widget=None):
        if widget is not None:
            knspace.shape_groups.deselect_node(widget)
            self.shapes.remove(widget.shape)
            self.more.remove_widget(widget)
        elif shape is not None:
            for widget in self.shape_widgets:
                if widget.shape == shape:
                    knspace.shape_groups.deselect_node(widget)
                    self.more.remove_widget(widget)
                    self.shapes.remove(shape)
                    return

    def get_state(self):
        return {'name': self.name, 'shapes': [s.name for s in self.shapes]}

    def set_state(self, state, shape_name_map):
        self.name = state['name']
        for name in state['shapes']:
            self.add_shape(shape_name_map[name])


class ShapeGroupItem(BoxSelector):

    selected = BooleanProperty(False)

    shape = ObjectProperty(None, rebind=True)

    group = ObjectProperty(None)

    @property
    def name(self):
        return self.shape.name


class ShapeList(ShowMoreSelection, WidgetList, BoxLayout):

    names = ListProperty([])

    def select_node(self, node):
        if super(ShapeList, self).select_node(node):
            knspace.painter.select_shape(node.shape)
            return True
        return False

    def deselect_node(self, node):
        if super(ShapeList, self).deselect_node(node):
            if knspace.painter.deselect_shape(node.shape):
                knspace.shape_groups.deselect_shape_everywhere(node.shape)
            return True
        return False

    def _change_shape_name(self, shape, name):
        if shape.name == name:
            return name
        name = fix_name(name, knspace.shape_groups.names, self.names)
        shape.name = name
        self._update_names()
        return name

    def _update_names(self, *largs):
        self.names = [shape.name for shape in knspace.painter.shapes]


class WidgetShape(ShowMoreBehavior, BoxLayout):

    painter = ObjectProperty(None, rebind=True)

    shape = ObjectProperty(None, rebind=True)

    label = None

    show_label = BooleanProperty(False)

    centroid_x = NumericProperty(0)

    centroid_y = NumericProperty(0)

    selected = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(WidgetShape, self).__init__(**kwargs)
        self.fbind('show_label', self._show_label)
        self.shape.fbind('name', self._label_text)

        knspace.shapes.add_widget(self)

        label = self.label = Label()
        self.shape.fbind('on_update', self._shape_update)
        label.fbind('size', self._shape_update)

        self._label_text()
        self._shape_update()
        if self.show_label:
            self._show_label()

    @property
    def name(self):
        return self.shape.name

    def remove(self):
        self.shape.funbind('on_update', self._shape_update)
        knspace.shapes.remove_widget(self)

        label = self.label
        label.funbind('size', self._shape_update)
        if self.show_label:
            self.painter.remove_widget(label)

    def _show_label(self, *largs):
        if self.show_label:
            self.painter.add_widget(self.label)
            self._shape_update()
            self._label_text()
        else:
            self.painter.remove_widget(self.label)

    def _label_text(self, *largs):
        knspace.shapes._update_names()
        if self.show_label:
            self.label.text = self.shape.name

    def _show_more(self, *largs):
        super(WidgetShape, self)._show_more(*largs)
        if self.show_more:
            self._shape_update()

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
