'''Shapes
======================

Defines the shapes which are intensity-varying durign experiments.

'''

from collections import OrderedDict

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.behaviors.togglebutton import ToggleButtonBehavior
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty
from kivy.event import EventDispatcher
from kivy.factory import Factory

from cplcom.painter import PaintCanvasBehavior, PaintCircle, PaintEllipse, \
    PaintPolygon, PaintBezier, PaintShape

import ceed
from ceed.utils import fix_name


class CeedPaintCanvasBehavior(KNSpaceBehavior, PaintCanvasBehavior):

    show_widgets = False

    groups = ListProperty([])

    selected_groups = ListProperty([])

    shape_names = DictProperty([])

    shape_group_names = DictProperty([])

    __events__ = ('on_remove_shape', 'on_remove_group', 'on_changed')

    def add_shape(self, shape, **kwargs):
        if not super(CeedPaintCanvasBehavior, self).add_shape(shape, **kwargs):
            return False

        shape.name = fix_name(
            shape.name, self.shape_names, self.shape_group_names)
        self.shape_names[shape.name] = shape
        if self.show_widgets:
            shape.display.show_widget()
        self.dispatch('on_changed')
        return True

    def remove_shape(self, shape):
        if not super(CeedPaintCanvasBehavior, self).remove_shape(shape):
            return False

        if shape._display:
            shape.display.hide_widget()
        self.remove_shape_from_groups(shape)
        self.dispatch('on_remove_shape', shape)
        del self.shape_names[shape.name]
        self.dispatch('on_changed')
        return True

    def add_group(self, group=None):
        if group is None:
            group = CeedShapeGroup(paint_widget=self)
        self.groups.append(group)

        if self.show_widgets:
            group.display.show_widget()
        self.shape_group_names[group.name] = group
        self.dispatch('on_changed')
        return group

    def remove_group(self, group):
        group.remove_all()
        group.deselect(keep_shapes=True)
        if group._display:
            group.display.hide_widget()

        self.dispatch('on_remove_group', group)
        del self.shape_group_names[group.name]
        self.groups.remove(group)
        self.dispatch('on_changed')

    def remove_all_groups(self):
        for group in self.groups[:]:
            self.remove_group(group)

    def add_selected_shapes(self, group=None):
        if group is None:
            group = self.add_group()

        for shape in self.selected_shapes:
            group.add_shape(shape)
        return group

    def remove_shape_from_groups(self, shape):
        for group in self.groups:
            if shape in group.shapes:
                group.remove_shape(shape)

    def deselect_shape_everywhere(self, shape):
        for group in self.groups:
            if shape in group.shapes:
                group.deselect_shape(shape)

    def count_shape_selection(self, shape):
        count = 0
        for group in self.groups:
            if shape in group.selected_shapes:
                count += 1
        return count

    def save_state(self):
        d = {}
        d['shapes__name_count'] = PaintShape._name_count
        d['groups__name_count'] = CeedShapeGroup._name_count
        d['shapes'] = self.save_shapes()
        d['groups'] = [{'name': g.name, 'shapes': [s.name for s in g.shapes]}
                       for g in self.groups]
        return d

    def restore_shape(self, state, old_name_map):
        shape = super(CeedPaintCanvasBehavior, self).restore_shape(state)
        if 'name' in state:
            old_name_map[state['name']] = shape.name
        return shape

    def set_state(self, state):
        PaintShape._name_count = max(
            PaintShape._name_count, state['shapes__name_count'])
        CeedShapeGroup._name_count = max(
            CeedShapeGroup._name_count, state['groups__name_count'])

        old_name_map = {}
        for s in state['shapes']:
            self.restore_shape(s, old_name_map)
        shape_names = self.shape_names

        for group_state in state['groups']:
            group = CeedShapeGroup(paint_widget=self, name=group_state['name'])
            self.add_group(group)
            for name in group_state['shapes']:
                if name in old_name_map and old_name_map[name] in shape_names:
                    group.add_shape(shape_names[old_name_map[name]])
        self.dispatch('on_changed')

    def on_remove_shape(self, shape):
        pass

    def on_remove_group(self, group):
        pass

    def on_changed(self, *largs):
        pass

    def _change_shape_name(self, shape, name):
        if shape.name == name:
            return name

        if isinstance(shape, CeedShape):
            cont = self.shape_names
        else:
            cont = self.shape_group_names

        del cont[shape.name]
        name = fix_name(name, self.shape_names, self.shape_group_names)
        shape.name = name
        cont[name] = shape
        return name


class CeedShape(object):

    _display = None

    def __init__(self, **kwargs):
        super(CeedShape, self).__init__(**kwargs)
        self.add_to_canvas = self.paint_widget.show_widgets
        self.fbind('on_update', get_painter().dispatch, 'on_changed')
        self.fbind('name', get_painter().dispatch, 'on_changed')

    @property
    def display(self):
        if self._display:
            return self._display

        w = self._display = Factory.WidgetShape(
            shape=self, painter=self.paint_widget)
        return w

    def select(self):
        if super(CeedShape, self).select():
            if self._display:
                self.display.select_widget()
            return True
        return False

    def deselect(self):
        if super(CeedShape, self).deselect():
            self.paint_widget.deselect_shape_everywhere(self)
            if self._display:
                self.display.deselect_widget()
            return True
        return False

    def get_state(self, state=None):
        d = super(CeedShape, self).get_state(state)
        d['cls'] = self.__class__.__name__[9:].lower()
        return d


class CeedShapeGroup(EventDispatcher):

    paint_widget = ObjectProperty(None)

    _name_count = 0

    name = StringProperty('')

    selected = BooleanProperty(False)

    shapes = []

    selected_shapes = []

    _display = None

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'G{}'.format(CeedShapeGroup._name_count)
            CeedShapeGroup._name_count += 1
        super(CeedShapeGroup, self).__init__(**kwargs)
        self.shapes = []
        self.selected_shapes = []
        self.name = fix_name(
            self.name, self.paint_widget.shape_names,
            self.paint_widget.shape_group_names)
        self.fbind('name', self.dispatch, 'on_changed')
        self.fbind('on_changed', get_painter().dispatch, 'on_changed')

    def on_changed(self, *largs):
        pass

    @property
    def display(self):
        if self._display:
            return self._display

        w = self._display = Factory.WidgetShapeGroup(group=self)
        return w

    def add_shape(self, shape):
        if shape in self.shapes:
            return

        if self._display:
            self.display.add_shape(shape)
        self.shapes.append(shape)
        self.dispatch('on_changed')

    def remove_shape(self, shape):
        self.deselect_shape(shape, deselect_base=False)
        if shape not in self.shapes:
            return

        if self._display:
            self.display.remove_shape(shape)
        self.shapes.remove(shape)
        self.dispatch('on_changed')

    def remove_all(self):
        for shape in self.shapes:
            self.remove_shape(shape)

    def select_shape(self, shape):
        if shape not in self.selected_shapes:
            self.selected_shapes.append(shape)
            if self._display:
                self.display.select_shape_widget(shape)
            self.paint_widget.select_shape(shape)
            return True
        return False

    def deselect_shape(self, shape, deselect_base=True):
        if shape in self.selected_shapes:
            self.deselect(keep_shapes=True)
            self.selected_shapes.remove(shape)
            if self._display:
                self.display.deselect_shape_widget(shape)

            if not self.paint_widget.count_shape_selection(shape):
                self.paint_widget.deselect_shape(shape)
            return True
        return False

    def select(self, clear_others=False):
        if self.selected:
            return False
        self.selected = True

        if self._display:
            self.display.select_widget()
        for shape in self.shapes:
            self.select_shape(shape)
        return True

    def deselect(self, keep_shapes=False):
        if not self.selected:
            return False
        self.selected = False

        if self._display:
            self.display.deselect_widget()
        if not keep_shapes:
            for shape in self.shapes:
                self.deselect_shape(shape)
        return True


class CeedPaintCircle(CeedShape, PaintCircle):
    pass


class CeedPaintEllipse(CeedShape, PaintEllipse):
    pass


class CeedPaintPolygon(CeedShape, PaintPolygon):
    pass


class CeedPaintPolygon(CeedShape, PaintPolygon):
    pass


class CeedPaintBezier(CeedShape, PaintBezier):
    pass

CeedPaintCanvasBehavior.shape_cls_map = {
    'circle': CeedPaintCircle, 'ellipse': CeedPaintEllipse,
    'polygon': CeedPaintPolygon, 'freeform': CeedPaintPolygon,
    'bezier': CeedPaintBezier
}

ceed_painter = False
if not ceed.has_gui_control:
    ceed_painter = CeedPaintCanvasBehavior()
    ceed_painter.knsname = 'painter'


def get_painter():
    if ceed.has_gui_control:
        return knspace.painter
    return ceed_painter
