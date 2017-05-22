'''Shape Widgets
=======================

Defines the GUI components used with :mod:`ceed.shape`.
'''
from collections import OrderedDict

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.uix.behaviors.togglebutton import ToggleButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.factory import Factory
from kivy.clock import Clock

from cplcom.painter import PaintCanvasBehavior, PaintCircle, PaintEllipse, \
    PaintPolygon, PaintBezier

from ceed.utils import fix_name
from ceed.graphics import WidgetList, ShowMoreSelection, BoxSelector, \
    ShowMoreBehavior, ColorBackgroundBehavior
from ceed.shape import CeedPaintCanvasBehavior

__all__ = (
    'CeedPainter', 'ShapeGroupList', 'WidgetShapeGroup', 'ShapeGroupItem',
    'ShapeList', 'WidgetShape')


class CeedPainter(CeedPaintCanvasBehavior, Widget):
    '''The shapes controller used when the GUI is present. It is the
    paint widget itself in this case. See :func:`ceed.shape.get_painter`.
    '''

    show_label = BooleanProperty(False)
    '''If True, a label showing the current mouse position is displayed.
    '''

    pos_label = None
    '''The label instance that shows the mouse position.
    '''

    def __init__(self, **kwargs):
        super(CeedPainter, self).__init__(**kwargs)
        self.pos_label = Factory.XYSizedLabel()

    def on_show_label(self, *largs):
        '''Shows/hides the :attr:`pos_label` label.
        '''
        state = self.show_label
        for shape in self.shapes:
            shape.display.show_label = state

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


class ShapeGroupList(ShowMoreSelection, WidgetList, BoxLayout):
    '''Widget that shows the list of all the groups.
    '''

    def add_selected_shapes(self):
        '''Selects in :class:`CeedPainter` the shapes that are the children
        of the group currently selected in the widget list.
        '''
        group = None
        if self.selected_nodes:
            node = self.selected_nodes[-1]
            if isinstance(node, ShapeGroupItem):
                node = node.group
            group = node.group

        self.knspace.painter.add_selected_shapes(group)

    def select_node(self, shape_widget):
        if not super(ShapeGroupList, self).select_node(shape_widget):
            return False

        if isinstance(shape_widget, WidgetShapeGroup):
            shape_widget.group.select()
            self._anchor = self
            self._last_selected_node = self
        else:
            shape_widget.group.group.select_shape(shape_widget.shape)
        return True

    def deselect_node(self, shape_widget):
        if not super(ShapeGroupList, self).deselect_node(shape_widget):
            return False

        if isinstance(shape_widget, WidgetShapeGroup):
            shape_widget.group.deselect()
        else:
            shape_widget.group.group.deselect_shape(shape_widget.shape)
        return True

    def get_selectable_nodes(self):
        nodes = []
        for child in reversed(self.children):
            nodes.append(child)
            if child.show_more:
                for c in reversed(child.shape_widgets):
                    nodes.append(c)
        return nodes


class WidgetShapeGroup(ShowMoreBehavior, BoxLayout):
    '''The widget displayed for a :class:`ceed.shape.CeedShapeGroup` instance.
    '''

    selected = BooleanProperty(False)
    '''If the group is :attr:`ceed.shape.CeedShapeGroup.selected`.
    '''

    group = ObjectProperty(None, rebind=True)
    '''The :class:`ceed.shape.CeedShapeGroup` this widget represents.
    '''

    @property
    def name(self):
        '''The :attr:`ceed.shape.CeedShapeGroup.name` of the group.
        '''
        return self.group.name

    def show_widget(self):
        '''Displays this widget group.
        '''
        knspace.shape_groups.add_widget(self)

    def hide_widget(self):
        '''Hides this widget group.
        '''
        knspace.shape_groups.remove_widget(self)

    def select_widget(self):
        '''Selects the group.
        '''
        knspace.shape_groups.select_node(self)

    def deselect_widget(self):
        '''Deselects the group.
        '''
        knspace.shape_groups.deselect_node(self)

    @property
    def shape_widgets(self):
        '''Returns the :class:`ShapeGroupItem` instances representing the
        shapes in this group.
        '''
        return self.more.children[:-1]

    def add_shape(self, shape):
        '''Adds and displays a :class:`ShapeGroupItem` widget representing the
        :class:`ceed.shape.CeedShape` to this group's widget.
        '''
        self.more.add_widget(ShapeGroupItem(shape=shape, group=self))

    def remove_shape(self, shape):
        '''Hides the :class:`ShapeGroupItem` associated with the
        :class:`ceed.shape.CeedShape` from this groups widget.
        '''
        for widget in self.shape_widgets:
            if widget.shape is shape:
                self.more.remove_widget(widget)
                return

    def select_shape_widget(self, shape):
        '''Selects the :class:`ceed.shape.CeedShape` from within this group.
        '''
        for widget in self.shape_widgets:
            if widget.shape is shape:
                knspace.shape_groups.select_node(widget)
                return

    def deselect_shape_widget(self, shape):
        '''Deselects the :class:`ceed.shape.CeedShape` from within this group.
        '''
        for widget in self.shape_widgets:
            if widget.shape is shape:
                knspace.shape_groups.deselect_node(widget)
                return


class ShapeGroupItem(BoxSelector):
    '''The shape's widget displayed in the :class:`WidgetShapeGroup` widget
    tree for a shape in that group.
    '''

    selected = BooleanProperty(False)
    '''Whether this shape is selected in the group.
    '''

    shape = ObjectProperty(None, rebind=True)
    '''The :class:`ceed.shape.CeedShape` to which this widget belongs.
    '''

    group = ObjectProperty(None)
    '''The :class:`ceed.shape.CeedShapeGroup` to which this shape belongs.
    '''

    @property
    def name(self):
        '''The :attr:`cplcom.painter.PaintShape.name` of the shape.
        '''
        return self.shape.name


class ShapeList(ShowMoreSelection, WidgetList, BoxLayout):
    '''Widget that shows the list of all the shapes.
    '''

    def select_node(self, node):
        if super(ShapeList, self).select_node(node):
            knspace.painter.select_shape(node.shape)
            return True
        return False

    def deselect_node(self, node):
        if super(ShapeList, self).deselect_node(node):
            knspace.painter.deselect_shape(node.shape)
            return True
        return False


class WidgetShape(ShowMoreBehavior, BoxLayout):
    '''The widget displayed for and associated with a
    :class:`ceed.shape.CeedShape` instance.
    '''

    painter = ObjectProperty(None, rebind=True)
    '''The :class:`CeedPainter` this shape belongs to.
    '''

    shape = ObjectProperty(None, rebind=True)
    '''The :class:`ceed.shape.CeedShape` instance associated with the widget.
    '''

    label = None
    '''The label widget that displays the name of the shape in the center
    of the shape in the drawing area.
    '''

    show_label = BooleanProperty(False)
    '''Whether :attr:`label` is currently displayed.
    '''

    centroid_x = NumericProperty(0)
    '''The x center of the shape (e.g. the x-center of the polygon).
    '''

    centroid_y = NumericProperty(0)
    '''The y center of the shape (e.g. the y-center of the polygon).
    '''

    area = NumericProperty(0)
    '''The area in the shape (e.g. the area of the polygon).
    '''

    selected = BooleanProperty(False)
    '''Whether the shape is :attr:`cplcom.painter.PaintShape.selected`.
    '''

    _shape_update_trigger = None

    def __init__(self, **kwargs):
        super(WidgetShape, self).__init__(**kwargs)
        self.show_label = self.painter.show_label
        self.fbind('show_label', self._show_label)
        self.shape.fbind('name', self._label_text)

        label = self.label = Label()
        trigger = Clock.create_trigger(self._shape_update, 0)
        f = self._shape_update_trigger = lambda *largs: trigger() and False
        self.shape.fbind('on_update', f)
        label.fbind('size', f)
        f()

        self._label_text()
        if self.show_label:
            self._show_label()

    @property
    def name(self):
        '''The :attr:`cplcom.painter.PaintShape.name` of the shape.
        '''
        return self.shape.name

    def show_widget(self):
        '''Displays this widget in the list of shape widgets.
        '''
        knspace.shapes.add_widget(self)

    def hide_widget(self):
        '''Hides this widget from the list of shape widgets.
        '''
        self.shape.funbind('on_update', self._shape_update_trigger)
        knspace.shapes.remove_widget(self)

        label = self.label
        label.funbind('size', self._shape_update_trigger)
        if self.show_label:
            self.painter.remove_widget(label)

    def select_widget(self):
        '''Selects the shape of this widget.
        '''
        knspace.shapes.select_node(self)

    def deselect_widget(self):
        '''Deselects the shape of this widget.
        '''
        knspace.shapes.deselect_node(self)

    def _show_label(self, *largs):
        '''Displays/hides the label in the shapes center containing the name of
        shape.
        '''
        if self.show_label:
            self.painter.add_widget(self.label)
            self._shape_update_trigger()
            self._label_text()
        else:
            self.painter.remove_widget(self.label)

    def _label_text(self, *largs):
        '''Updates the :attr:`label` with the current name of the shape.
        '''
        if self.show_label:
            self.label.text = self.shape.name

    def _show_more(self, *largs):
        '''Shows the settings
        '''
        super(WidgetShape, self)._show_more(*largs)
        if self.show_more:
            self._shape_update_trigger()

    def _shape_update(self, *largs):
        '''Update the centroids and area when the shape is changed.
        '''
        self.centroid_x, self.centroid_y = tuple(
            map(round, self.shape.centroid))
        self.area = round(self.shape.area)
        if self.show_label:
            self.label.center = self.shape.centroid

    def _update_centroid(self, x=None, y=None):
        '''Sets the centroid from the GUI.
        '''
        x1, y1 = map(round, self.shape.centroid)
        dx = 0 if x is None else x - x1
        dy = 0 if y is None else y - y1
        if dx or dy:
            self.shape.translate(dpos=(dx, dy))

    def _update_area(self, area):
        '''Sets the area from the GUI.
        '''
        self.shape.set_area(area)
