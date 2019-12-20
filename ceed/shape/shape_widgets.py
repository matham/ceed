"""Shape Widgets
=======================

Defines the GUI components used with :mod:`ceed.shape`.
"""
import math
from typing import Type, List, Tuple, Dict, Optional, Union

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.app import App

from kivy_garden.drag_n_drop import DraggableLayoutBehavior

from ceed.graphics import WidgetList, ShowMoreSelection, ShowMoreBehavior
from ceed.shape import CeedPaintCanvasBehavior, CeedShapeGroup, CeedShape

__all__ = (
    'CeedPainter', 'ShapeGroupList', 'WidgetShapeGroup', 'ShapeGroupItem',
    'ShapeList', 'WidgetShape')


class CeedPainter(CeedPaintCanvasBehavior, Widget):
    """The shapes controller used in the GUI. It is the
    canvas widget upon which the shapes are drawn.
    """

    show_label = BooleanProperty(False)
    '''If True, a label showing the current mouse position is displayed.
    '''

    pos_label = None
    '''The label instance that shows the mouse position.
    '''

    shape_widgets_list: 'ShapeList' = None
    """The :class:`ShapeList` that contains the shape widgets.
    """

    def __init__(self, **kwargs):
        super(CeedPainter, self).__init__(**kwargs)
        self.pos_label = Factory.XYSizedLabel()

    @property
    def selected_groups(self) -> List[CeedShapeGroup]:
        """Returns the list of :class:`CeedShapeGroup` that are currently
        selected in the GUI.
        """
        app = App.get_running_app()
        return [widget.group
                for widget in app.shape_groups_container.selected_nodes]

    def create_shape_with_touch(self, touch):
        shape = super(CeedPainter, self).create_shape_with_touch(touch)
        if shape is not None:
            shape.add_shape_to_canvas(self)
        return shape

    def reorder_shape(self, shape, before_shape=None):
        super(CeedPainter, self).reorder_shape(shape, before_shape=before_shape)

        self.shape_widgets_list.remove_widget(shape.widget)
        if before_shape is None:
            self.shape_widgets_list.add_widget(shape.widget)
        else:
            i = self.shape_widgets_list.children.index(before_shape.widget)
            self.shape_widgets_list.add_widget(shape.widget, index=i + 1)

    def add_shape(self, shape):
        if super(CeedPainter, self).add_shape(shape):
            shape.add_shape_to_canvas(self)
            widget = shape.widget = WidgetShape(painter=self, shape=shape)
            widget.show_widget()
            shape.fbind('on_update', App.get_running_app().changed_callback)
            return True
        return False

    def remove_shape(self, shape):
        if super(CeedPainter, self).remove_shape(shape):
            shape.remove_shape_from_canvas()
            shape.widget.hide_widget()
            shape.widget = None
            shape.funbind('on_update', App.get_running_app().changed_callback)
            return True
        return False

    def add_group(self, group=None):
        group = super(CeedPainter, self).add_group(group)
        widget = group.widget = WidgetShapeGroup(group=group)
        widget.show_widget()
        group.fbind('on_changed', App.get_running_app().changed_callback)
        return group

    def remove_group(self, group):
        if super(CeedPainter, self).remove_group(group):
            group.widget.hide_widget()
            group.widget = None
            group.funbind('on_changed', App.get_running_app().changed_callback)
            return True
        return False

    def on_show_label(self, *largs):
        """Shows/hides the :attr:`pos_label` label depending on the value
        of :attr:`show_label`.
        """
        state = self.show_label
        for shape in self.shapes:
            shape.widget.show_label = state

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

    def add_enclosing_polygon(self):
        """Adds a polygon shape named ``'enclosed'`` that encloses the whole
        drawing area.
        """
        w, h = self.size
        self.create_add_shape(
            'polygon', points=[0, 0, w, 0, w, h, 0, h], name='enclosed')

    def select_shape(self, shape):
        if super(CeedPainter, self).select_shape(shape):
            if shape.widget is not None:
                App.get_running_app().shapes_container.select_node(
                    shape.widget)
            return True
        return False

    def deselect_shape(self, shape):
        if super(CeedPainter, self).deselect_shape(shape):
            if shape.widget is not None:
                App.get_running_app().shapes_container.deselect_node(
                    shape.widget)
            return True
        return False


class ShapeGroupDraggableLayoutBehavior(DraggableLayoutBehavior):
    """The container widget of a group that displays the shapes of the group.
    """

    group_widget: 'WidgetShapeGroup' = ObjectProperty(None)

    def handle_drag_release(self, index, drag_widget):
        group = self.group_widget.group

        group.add_shape(drag_widget.obj_dragged.shape)
        if drag_widget.obj_dragged.selected:
            App.get_running_app().shape_factory.add_selected_shapes_to_group(
                group)


class ShapeGroupList(
        DraggableLayoutBehavior, ShowMoreSelection, WidgetList, BoxLayout):
    """Container widget that shows all the groups.
    """

    def add_selected_shapes(self):
        """Adds all the shapes currently selected in the painter to the
        currently selected group. If no group is selected, a new one is
        created.
        """
        group = None
        if self.selected_nodes:
            group = self.selected_nodes[-1].group

        App.get_running_app().shape_factory.add_selected_shapes_to_group(group)

    def handle_drag_release(self, index, drag_widget):
        app = App.get_running_app()
        if drag_widget.drag_cls == 'shape':
            group = app.shape_factory.add_group()
            group.add_shape(drag_widget.obj_dragged.shape)
            if drag_widget.obj_dragged.selected:
                app.shape_factory.add_selected_shapes_to_group(group)
            group.widget.expand_widget.state = 'down'
        else:
            group = app.shape_factory.add_group()
            for shape in drag_widget.obj_dragged.group.shapes:
                group.add_shape(shape)
            group.widget.expand_widget.state = 'down'


class WidgetShapeGroup(ShowMoreBehavior, BoxLayout):
    """The widget that is displayed for a :class:`ceed.shape.CeedShapeGroup`
    instance.
    """

    selected = BooleanProperty(False)
    '''If the group is :attr:`~ceed.shape.CeedShapeGroup.selected`.
    '''

    group: CeedShapeGroup = ObjectProperty(None, rebind=True)
    '''The :class:`~ceed.shape.CeedShapeGroup` this widget represents.
    '''

    expand_widget = None
    """The ExpandWidget that when hit wil show the list of shapes of the group.
    """

    @property
    def name(self):
        """The :attr:`ceed.shape.CeedShapeGroup.name` of the group.
        """
        return self.group.name

    def show_widget(self):
        """Displays this widget group in the GUI.
        """
        App.get_running_app().shape_groups_container.add_widget(self)

    def hide_widget(self):
        """Hides this widget group from the GUI.
        """
        App.get_running_app().shape_groups_container.deselect_node(self)
        App.get_running_app().shape_groups_container.remove_widget(self)

    @property
    def shape_widgets(self) -> List['ShapeGroupItem']:
        """Returns the :class:`ShapeGroupItem` instances representing the
        shapes in this group.
        """
        return self.more.children[:-1]

    def add_shape(self, shape):
        """Adds and displays a :class:`ShapeGroupItem` widget representing the
        :class:`ceed.shape.CeedShape`, to this group's widget.
        """
        self.more.add_widget(ShapeGroupItem(shape=shape, group=self))

    def remove_shape(self, shape):
        """Hides the :class:`ShapeGroupItem` associated with the
        :class:`ceed.shape.CeedShape` from this group's widget.
        """
        for widget in self.shape_widgets:
            if widget.shape is shape:
                self.more.remove_widget(widget)
                return


class ShapeGroupItem(BoxLayout):
    """The shape's widget displayed in the :class:`WidgetShapeGroup` widget
    tree for a shape from that group.
    """

    shape: CeedShape = ObjectProperty(None, rebind=True)
    '''The :class:`~ceed.shape.CeedShape` with which this widget is associated.
    '''

    group: CeedShapeGroup = ObjectProperty(None)
    '''The :class:`~ceed.shape.CeedShapeGroup` to which this shape belongs.
    '''

    @property
    def name(self):
        """The :attr:`~ceed.shape.CeedShape.name` of the shape.
        """
        return self.shape.name


class ShapeList(DraggableLayoutBehavior, ShowMoreSelection, WidgetList,
                BoxLayout):
    """Container widget that shows all the shapes.
    """

    def select_node(self, node):
        if super(ShapeList, self).select_node(node):
            App.get_running_app().shape_factory.select_shape(node.shape)
            return True
        return False

    def deselect_node(self, node):
        if super(ShapeList, self).deselect_node(node):
            App.get_running_app().shape_factory.deselect_shape(node.shape)
            return True
        return False

    def handle_drag_release(self, index, drag_widget):
        if drag_widget.obj_dragged.selected:
            App.get_running_app().shape_factory.duplicate_selected_shapes()
        else:
            App.get_running_app().shape_factory.duplicate_shape(
                drag_widget.obj_dragged.shape)


class WidgetShape(ShowMoreBehavior, BoxLayout):
    """The widget displayed for and associated with a
    :class:`~ceed.shape.CeedShape` instance.
    """

    painter: CeedPainter = ObjectProperty(None, rebind=True)
    '''The :class:`CeedPainter` this shape belongs to.
    '''

    shape: CeedShape = ObjectProperty(None, rebind=True)
    '''The :class:`~ceed.shape.CeedShape` instance associated with the widget.
    '''

    label = None
    '''The label widget that displays the name of the shape in the center
    of the shape, in the drawing area when enabled.
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
    '''Whether the shape is :attr:`kivy_garden.painter.PaintShape.selected`.
    '''

    _shape_update_trigger = None

    def __init__(self, **kwargs):
        super(WidgetShape, self).__init__(**kwargs)
        self.show_label = self.painter.show_label

        self.label = Label()
        trigger = Clock.create_trigger(self._shape_update, 0)
        self._shape_update_trigger = lambda *largs: trigger() and False

    @property
    def name(self):
        """The :attr:`kivy_garden.painter.PaintShape.name` of the shape.
        """
        return self.shape.name

    def show_widget(self, index=None):
        """Displays this widget in the list of shape widgets at the given
        index. The index is in the same order as the shapes, i.e. zero is shape
        zero etc.
        """
        if index is None:
            App.get_running_app().shapes_container.add_widget(self)
        else:
            App.get_running_app().shapes_container.add_widget(
                self, index=len(
                    App.get_running_app().shapes_container.children) - index)

        self.fbind('show_label', self._show_label)
        self.shape.fbind('name', self._label_text)

        f = self._shape_update_trigger
        self.shape.fbind('on_update', f)
        self.label.fbind('size', f)
        f()

        self._label_text()
        self._show_label()

    def hide_widget(self):
        """Hides this widget from the list of shape widgets.
        """
        self.shape.funbind('on_update', self._shape_update_trigger)
        App.get_running_app().shapes_container.remove_widget(self)

        label = self.label
        label.funbind('size', self._shape_update_trigger)

        self.funbind('show_label', self._show_label)
        self.shape.funbind('name', self._label_text)
        self._show_label(force_hide=True)

    def _show_label(self, *largs, force_hide=False):
        """Displays/hides the label in the shapes center containing the name of
        shape.
        """
        if self.show_label and not force_hide:
            if self.label.parent is not None:  # already showing
                return

            self.painter.add_widget(self.label)
            self._shape_update_trigger()
            self._label_text()
        elif self.label.parent is not None:
            self.painter.remove_widget(self.label)

    def _label_text(self, *largs):
        """Updates the :attr:`label` with the current name of the shape.
        """
        if self.show_label:
            self.label.text = self.shape.name

    def _show_more(self, *largs):
        super(WidgetShape, self)._show_more(*largs)
        if self.show_more:
            self._shape_update_trigger()

    def _shape_update(self, *largs):
        """Update the centroids and area when the shape is changed.
        """
        if not self.shape.finished:
            return
        self.centroid_x, self.centroid_y = tuple(
            map(round, self.shape.centroid))
        self.area = round(self.shape.area)
        if self.show_label:
            self.label.center = self.shape.centroid

    def _update_centroid(self, x=None, y=None):
        """Sets the centroid from the GUI.
        """
        x1, y1 = map(round, self.shape.centroid)
        dx = 0 if x is None else x - x1
        dy = 0 if y is None else y - y1
        if dx or dy:
            self.shape.translate(dpos=(dx, dy))

    def _update_area(self, area):
        """Sets the area from the GUI.
        """
        if not math.isclose(area, self.area):
            self.shape.set_area(area)


# make it available from kv
Factory.register('ShapeGroupDraggableLayoutBehavior',
                 cls=ShapeGroupDraggableLayoutBehavior)
