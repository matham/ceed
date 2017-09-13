'''Shapes
============

Defines the shapes which are used with a :mod:`ceed.function` to create regions
with time-varying intensity during the experiemtal :mod:`ceed.stage`.

Shapes are created automatically when the user draws regions in the GUI. The
controller keeping track of these shapes is a :class:`CeedPaintCanvasBehavior`
instance and is returned by :func:`get_painter`.
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

__all__ = (
    'get_painter', 'CeedPaintCanvasBehavior', 'CeedShape', 'CeedShapeGroup',
    'CeedPaintCircle', 'CeedPaintEllipse', 'CeedPaintPolygon',
    'CeedPaintBezier')


class CeedPaintCanvasBehavior(KNSpaceBehavior, PaintCanvasBehavior):
    '''Controller base class for drawing and managing the shapes.

    A shape is drawn by the user in the GUI and added by
    :class:`cplcom.painter.PaintCanvasBehavior` to its
    :attr:`cplcom.painter.PaintCanvasBehavior.shapes` list. This class
    manages all that without any of the associated GUI components that
    is shown to the user. The GUI components is added by the
    :class:`ceed.shape.shape_widgets.CeedPainter`.

    So when run from the GUI :class:`ceed.shape.shape_widgets.CeedPainter` is
    the class used, while this class used when e.g. running from the
    interpreter. :func:`get_painter` will return the correct instance for
    each case.

    In addition to the :attr:`cplcom.painter.PaintCanvasBehavior.shapes`, the
    class adds :attr:`groups` for grouping shapes; a :class:`CeedShapeGroup`
    simply groups a collection of :class:`CeedShape`.

    :Events:

        `on_remove_shape`:
            Triggered when a :class:`CeedShape` is removed. The first parameter
            is the shape removed.
        `on_remove_group`:
            Triggered when a :class:`CeedShapeGroup` is removed. The first
            parameter is the group removed.
        `on_changed`:
            Triggered whenever a :class:`CeedShape` or :class:`CeedShapeGroup`
            is added or removed, or if a configuration option of the objects
            is changed.
    '''

    show_widgets = False
    '''Whether the class is used through the GUI, in which case the shape's
    widget are displayed, otherwise, no widgets are displayed.
    '''

    groups = ListProperty([])
    '''List of :class:`CeedShapeGroup` instances.
    '''

    selected_groups = ListProperty([])
    '''Similar to :attr:`cplcom.painter.PaintCanvasBehavior.selected_shapes`,
    but for the groups. It's the list of :class:`CeedShapeGroup` currently
    selected in the GUI.
    '''

    shape_names = DictProperty([])
    '''The name -> :class:`CeedShape` dict. The key is the shape's name
    and the corresponding value is the :class:`CeedShape` instance.
    '''

    shape_group_names = DictProperty([])
    '''The name -> :class:`CeedShapeGroup` dict. The key is the group's name
    and the corresponding value is the :class:`CeedShapeGroup` instance.
    '''

    __events__ = ('on_remove_shape', 'on_remove_group', 'on_changed')

    def add_shape(self, shape, **kwargs):
        '''Overrides :meth:`cplcom.painter.PaintCanvasBehavior.add_shape`.

        It ensures the the name of the :class:`CeedShape` added is unique and
        also displays the widget associated with the :class:`CeedShape` if
        :attr:`show_widgets`.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to add.

        :returns:

            True if the shape was added, False otherwise.
        '''
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
        '''Overrides :meth:`cplcom.painter.PaintCanvasBehavior.remove_shape`.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to remove.

        :returns:

            True if the shape was removed, False otherwise.
        '''
        if not super(CeedPaintCanvasBehavior, self).remove_shape(shape):
            return False

        if shape._display:
            shape.display.hide_widget()
        self.remove_shape_from_groups(shape)
        self.dispatch('on_remove_shape', shape)
        del self.shape_names[shape.name]
        self.dispatch('on_changed')
        return True

    def reorder_shape(self, shape, before_shape=None):
        if shape._display:
            shape.display.hide_widget()
        super(CeedPaintCanvasBehavior, self).reorder_shape(
            shape, before_shape=before_shape)
        if self.show_widgets:
            shape.display.show_widget(self.shapes.index(shape))
        self.dispatch('on_changed')

    def move_shape_lower(self, shape):
        '''Moves it below the shape below it
        '''
        i = self.shapes.index(shape)
        if not i:
            return

        before_shape = self.shapes[i - 1]
        self.reorder_shape(shape, before_shape)

    def move_shape_upwards(self, shape):
        i = self.shapes.index(shape)
        if i == len(self.shapes) - 1:
            return

        if i == len(self.shapes) - 2:
            before_shape = None
        else:
            before_shape = self.shapes[i + 2]
        self.reorder_shape(shape, before_shape)

    def add_group(self, group=None):
        '''Similar to :meth:`add_shape` but for a :class:`CeedShapeGroup`.

        :Params:

            `group`: :class:`CeedShapeGroup`
                The group to add. If None, the default, a new
                :class:`CeedShapeGroup` is created.

        :returns:

            The :class:`CeedShapeGroup` added.
        '''
        if group is None:
            group = CeedShapeGroup(paint_widget=self)
        self.groups.append(group)

        if self.show_widgets:
            group.display.show_widget()
        self.shape_group_names[group.name] = group
        self.dispatch('on_changed')
        return group

    def remove_group(self, group):
        '''Similar to :meth:`remove_shape` but for a :class:`CeedShapeGroup`.

        :Params:

            `group`: :class:`CeedShapeGroup`
                The group to remove.

        :returns:

            True if the group was removed, False otherwise.
        '''
        group.remove_all()
        group.deselect(keep_shapes=True)
        if group._display:
            group.display.hide_widget()

        self.dispatch('on_remove_group', group)
        del self.shape_group_names[group.name]
        self.groups.remove(group)
        self.dispatch('on_changed')
        return True

    def remove_all_groups(self):
        '''Removes all the :class:`CeedShapeGroup` instances.
        '''
        for group in self.groups[:]:
            self.remove_group(group)

    def add_selected_shapes(self, group=None):
        '''Adds all the
        :attr:`cplcom.painter.PaintCanvasBehavior.selected_shapes` to the
        ``group``.

        :Params:

            `group`: :class:`CeedShapeGroup`
                The group to which to add the shapes. If None, the default, a
                new :class:`CeedShapeGroup` is created.

        :returns:

            The :class:`CeedShapeGroup` passed in or created.
        '''
        if group is None:
            group = self.add_group()

        for shape in self.selected_shapes:
            group.add_shape(shape)
        return group

    def remove_shape_from_groups(self, shape):
        '''Removes the :class:`CeedShape` from all the groups.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to remove.
        '''
        for group in self.groups:
            if shape in group.shapes:
                group.remove_shape(shape)

    def deselect_shape_everywhere(self, shape):
        '''Deselects the :class:`CeedShape` from all the groups. If the
        shape was selected, it is selected in all the groups. This clears that.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to deselect.
        '''
        for group in self.groups:
            if shape in group.shapes:
                group.deselect_shape(shape)

    def count_shape_selection(self, shape):
        '''The number of :class:`CeedShapeGroup` that the :class:`CeedShape`
        is selected in. If the shape was selected, it is selected in all the
        groups that contains it. This counts the number of groups that
        contain it and in which it's selected.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to deselects.

        :returns:

            The number of groups in which it's selected.
        '''
        count = 0
        for group in self.groups:
            if shape in group.selected_shapes:
                count += 1
        return count

    def save_state(self):
        '''Returns a dictionary containing all the configuration data for all
        the shapes and groups. It is used with :meth:`set_state` to later
        restore the state.
        '''
        d = {}
        d['shapes__name_count'] = PaintShape._name_count
        d['groups__name_count'] = CeedShapeGroup._name_count
        d['shapes'] = self.save_shapes()
        d['groups'] = [{'name': g.name, 'shapes': [s.name for s in g.shapes]}
                       for g in self.groups]
        return d

    def restore_shape(self, state, old_name_map):
        '''Overrides :meth:`cplcom.painter.PaintCanvasBehavior.restore_shape`.

        It takes an additional parameter, ``old_name_map``. When a shape is
        created from the ``state``, the shape's new name could be changed so
        that it remains unique. ``old_name_map`` is a dict that is filled in so
        the key is the old name (if present in ``state``) and the associated
        value is the actual final shape name.
        '''
        shape = super(CeedPaintCanvasBehavior, self).restore_shape(state)
        if 'name' in state:
            old_name_map[state['name']] = shape.name
        return shape

    def set_state(self, state, old_to_new_name_map):
        '''Takes the dict returned by :meth:`save_state` and adds the shapes
        and groups to the controller.
        '''
        PaintShape._name_count = max(
            PaintShape._name_count, state['shapes__name_count'])
        CeedShapeGroup._name_count = max(
            CeedShapeGroup._name_count, state['groups__name_count'])

        for s in state['shapes']:
            self.restore_shape(s, old_to_new_name_map)
        shape_names = self.shape_names

        for group_state in state['groups']:
            group = CeedShapeGroup(paint_widget=self, name=group_state['name'])
            if 'name' in group_state:
                old_to_new_name_map[group_state['name']] = group.name
            self.add_group(group)
            for name in group_state['shapes']:
                if (name in old_to_new_name_map and
                        old_to_new_name_map[name] in shape_names):
                    group.add_shape(shape_names[old_to_new_name_map[name]])

        self.dispatch('on_changed')

    def on_remove_shape(self, shape):
        pass

    def on_remove_group(self, group):
        pass

    def on_changed(self, *largs):
        pass

    def _change_shape_name(self, shape, name):
        '''Makes sure that the shape or group name is unique.
        '''
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
    '''A co-base class used with :class:`cplcom.painter.PaintShape` derived
    classes to add ceed specific functionality to the
    :class:`cplcom.painter.PaintShape` classes.
    '''

    _display = None

    def __init__(self, **kwargs):
        super(CeedShape, self).__init__(**kwargs)
        self.add_to_canvas = self.paint_widget.show_widgets
        self.fbind('on_update', get_painter().dispatch, 'on_changed')
        self.fbind('name', get_painter().dispatch, 'on_changed')

    @property
    def display(self):
        '''The GUI widget associated with the shape and displayed to the
        user.
        '''
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
    '''Holds a collection of :class:`CeedShape` instances.

    It is helpful to group them when the same :class:`ceed.function` is to be
    applied to multiple shapes.

    :Events:

        `on_changed`:
            Triggered whenever a child :class:`CeedShape`
            is added or removed, or if a configuration option of the objects
            is changed.
    '''

    paint_widget = ObjectProperty(None)
    '''See :attr:`cplcom.painter.PaintShape.paint_widget`.
    '''

    _name_count = 0
    '''A counter to ensure name uniqueness.
    '''

    name = StringProperty('')
    '''The name of the group. Similar to
    See :attr:`cplcom.painter.PaintShape.name`.
    '''

    selected = BooleanProperty(False)
    '''Whether the group is selected. Similar to
    See :attr:`cplcom.painter.PaintShape.selected`.
    '''

    shapes = []
    '''A list that contains the :class:`CeedShape` instances that is part of
    this group.
    '''

    selected_shapes = []
    '''A list that contains the :class:`CeedShape` instances that is part of
    this group and are also selected.
    '''

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
        '''The GUI widget associated with the group and displayed to the
        user.
        '''
        if self._display:
            return self._display

        w = self._display = Factory.WidgetShapeGroup(group=self)
        return w

    def add_shape(self, shape):
        '''Adds the shape to the group if it is not already in the group.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to add to :attr:`shapes`.
        '''
        if shape in self.shapes:
            return

        if self._display:
            self.display.add_shape(shape)
        self.shapes.append(shape)
        self.dispatch('on_changed')

    def remove_shape(self, shape):
        '''Removes the shape from the group (and its :attr:`CeedShape.display`)
        if it is present.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to remove from :attr:`shapes`.
        '''
        self.deselect_shape(shape)
        if shape not in self.shapes:
            return

        if self._display:
            self.display.remove_shape(shape)
        self.shapes.remove(shape)
        self.dispatch('on_changed')

    def remove_all(self):
        '''Removes all the shapes from the group.
        '''
        for shape in self.shapes:
            self.remove_shape(shape)

    def select_shape(self, shape):
        '''Selects the shape within the group. Selected shapes are added to
        :attr:`selected_shapes`.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to select.

        :returns:

            True if the shape was selected, False otherwise.
        '''
        if shape not in self.selected_shapes:
            self.selected_shapes.append(shape)
            if self._display:
                self.display.select_shape_widget(shape)
            self.paint_widget.select_shape(shape)
            return True
        return False

    def deselect_shape(self, shape):
        '''deselects the shape if it's selected within the group.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to deselect.

        :returns:

            True if the shape was deselected, False otherwise.
        '''
        if shape in self.selected_shapes:
            self.deselect(keep_shapes=True)
            self.selected_shapes.remove(shape)
            if self._display:
                self.display.deselect_shape_widget(shape)

            if not self.paint_widget.count_shape_selection(shape):
                self.paint_widget.deselect_shape(shape)
            return True
        return False

    def select(self):
        '''Selects the group and all its children shapes.

        :returns:

            True if the group was selected, False otherwise.
        '''
        if self.selected:
            return False
        self.selected = True

        if self._display:
            self.display.select_widget()
        for shape in self.shapes:
            self.select_shape(shape)
        return True

    def deselect(self, keep_shapes=False):
        '''Deselects the group and all its shapes if ``keep_shapes`` is True,
        otherwise, only the group is deselected.

        :returns:

            True if the group was deselected, False otherwise.
        '''
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
    '''A circle shape.
    '''
    pass


class CeedPaintEllipse(CeedShape, PaintEllipse):
    '''An ellipse shape.
    '''
    pass


class CeedPaintPolygon(CeedShape, PaintPolygon):
    '''A polygonal shape.
    '''
    pass


class CeedPaintBezier(CeedShape, PaintBezier):
    '''A bezier shape.
    '''
    pass

# make sure the classes above is used rather than the defaults.
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
    '''Returns the controller that stores the shapes etc.

    When running from the GUI, the
    :class:`ceed.shape.shape_widgets.CeedPainter` is used as the controller
    and is returned. When running outside the GUI, e.g. imported on the command
    line widgets are not used and :class:`CeedPaintCanvasBehavior` is used
    as the controller class.
    '''
    if ceed.has_gui_control:
        return knspace.painter
    return ceed_painter
