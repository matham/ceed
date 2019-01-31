'''Shapes
============

Defines the shapes which are used with a :mod:`ceed.function` to create regions
with time-varying intensity during the experiemtal :mod:`ceed.stage`.

Shapes are created automatically when the user draws regions in the GUI. The
controller keeping track of these shapes is a :class:`CeedPaintCanvasBehavior`
instance.
'''
import math

from kivy.uix.behaviors.knspace import KNSpaceBehavior, knspace
from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty
from kivy.event import EventDispatcher
from kivy.factory import Factory
from kivy.garden.collider import Collide2DPoly, CollideEllipse

from cplcom.painter import PaintCanvasBehavior, PaintCircle, PaintEllipse, \
    PaintPolygon, PaintFreeformPolygon

from ceed.utils import fix_name

__all__ = (
    'CeedPaintCanvasBehavior', 'CeedShape', 'CeedShapeGroup',
    'CeedPaintCircle', 'CeedPaintEllipse', 'CeedPaintPolygon')


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
    interpreter.

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

    groups = ListProperty([])
    '''List of :class:`CeedShapeGroup` instances.
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

    def add_shape(self, shape):
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
        if not super(CeedPaintCanvasBehavior, self).add_shape(shape):
            return False

        name = fix_name(
            shape.name, self.shape_names, self.shape_group_names)
        self.shape_names[name] = shape
        shape.name = name
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

        self.remove_shape_from_groups(shape)
        del self.shape_names[shape.name]
        self.dispatch('on_remove_shape', shape)
        self.dispatch('on_changed')
        return True

    def reorder_shape(self, shape, before_shape=None):
        self.dispatch('on_changed')
        return super(CeedPaintCanvasBehavior, self).reorder_shape(
            shape, before_shape=before_shape)

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

        name = fix_name(
            group.name, self.shape_names, self.shape_group_names)
        self.shape_group_names[name] = group
        group.name = name
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
        del self.shape_group_names[group.name]
        self.groups.remove(group)
        self.dispatch('on_remove_group', group)
        self.dispatch('on_changed')
        return True

    def remove_all_groups(self):
        '''Removes all the :class:`CeedShapeGroup` instances.
        '''
        for group in self.groups[:]:
            self.remove_group(group)

    def add_shape_to_group(self, group, shape):
        group.add_shape(shape)
        self.dispatch('on_changed')

    def remove_shape_from_group(self, group, shape):
        group.remove_shape(shape)
        self.dispatch('on_changed')

    def add_selected_shapes_to_group(self, group=None):
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
            self.add_shape_to_group(group, shape)
        return group

    def remove_shape_from_groups(self, shape):
        '''Removes the :class:`CeedShape` from all the groups.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to remove.
        '''
        for group in self.groups:
            if shape in group.shapes:
                self.remove_shape_from_group(group, shape)

    def get_state(self):
        '''Returns a dictionary containing all the configuration data for all
        the shapes and groups. It is used with :meth:`set_state` to later
        restore the state.
        '''
        d = {
            'shapes': [s.get_state() for s in self.shapes],
            'groups': [{'name': g.name, 'shapes': [s.name for s in g.shapes]}
                       for g in self.groups],
        }
        return d

    def create_shape_from_state(self, state, old_name_map):
        '''Overrides :meth:`cplcom.painter.PaintCanvasBehavior.create_shape_from_state`.

        It takes an additional parameter, ``old_name_map``. When a shape is
        created from the ``state``, the shape's new name could be changed so
        that it remains unique. ``old_name_map`` is a dict that is filled in so
        the key is the old name (if present in ``state``) and the associated
        value is the actual final shape name.
        '''
        old_name = state.get('name', '')
        shape = super(CeedPaintCanvasBehavior, self).create_shape_from_state(
            state)
        if old_name:
            old_name_map[old_name] = shape
        return shape

    def set_state(self, state, old_name_map):
        '''Takes the dict returned by :meth:`save_state` and adds the shapes
        and groups to the controller.
        '''
        for s in state['shapes']:
            self.create_shape_from_state(s, old_name_map)

        for group_state in state['groups']:
            group = CeedShapeGroup(paint_widget=self, name=group_state['name'])
            if group_state['name']:
                old_name_map[group_state['name']] = group
            self.add_group(group)

            for name in group_state['shapes']:
                shape = old_name_map.get(name, None)
                if shape is None:
                    raise Exception('Cannot find shape {}'.format(name))
                self.add_shape_to_group(group, shape)

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
            container = self.shape_names
        else:
            container = self.shape_group_names

        del container[shape.name]
        name = fix_name(name, self.shape_names, self.shape_group_names)
        container[name] = shape
        shape.name = name
        return name


class CeedShape(object):
    '''A co-base class used with :class:`cplcom.painter.PaintShape` derived
    classes to add ceed specific functionality to the
    :class:`cplcom.painter.PaintShape` classes.
    '''

    name = StringProperty('Shape')

    _collider = None

    _area = None

    _bounding_box = None

    _centroid = None

    paint_widget_size = 0, 0

    widget = None

    def __init__(self, paint_widget_size=(0, 0), **kwargs):
        super(CeedShape, self).__init__(**kwargs)
        self.paint_widget_size = paint_widget_size
        self.fbind('name', self.dispatch, 'on_update')

        def on_update(*largs):
            self._bounding_box = None
            self._centroid = None
            self._area = None
            self._collider = None
        self.fbind('on_update', on_update)

    def get_state(self, state=None):
        d = super(CeedShape, self).get_state(state)
        d['name'] = self.name
        return d

    def set_state(self, state):
        state = dict(state)
        state.pop('cls', None)
        self.name = state.pop('name')
        return super(CeedShape, self).set_state(state)

    def _get_collider(self, size):
        pass

    @property
    def bounding_box(self):
        if self._bounding_box is not None:
            return self._bounding_box

        collider = self.collider
        if collider is None:
            return 0, 0, 0, 0

        x1, y1, x2, y2 = collider.bounding_box()
        box = self._bounding_box = x1, y1, x2 + 1, y2 + 1
        return box

    @property
    def centroid(self):
        if self._centroid is not None:
            return self._centroid

        collider = self.collider
        if collider is None:
            return 0, 0

        self._centroid = xc, yc = self.collider.get_centroid()
        return xc, yc

    @property
    def area(self):
        if self._area is not None:
            return self._area

        collider = self.collider
        if collider is None:
            return 0

        self._area = area = float(self.collider.get_area())
        return area

    def set_area(self, area):
        if not area:
            return

        scale = 1 / math.sqrt(self.area / float(area))
        self.rescale(scale)

    @property
    def collider(self):
        if not self.is_valid or not self.finished:
            return None
        if self._collider is not None:
            return self._collider

        self._collider = collider = self._get_collider(self.paint_widget_size)
        return collider


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

    name = StringProperty('Group')
    '''The name of the group. Similar to
    See :attr:`cplcom.painter.PaintShape.name`.
    '''

    shapes = []
    '''A list that contains the :class:`CeedShape` instances that is part of
    this group.
    '''

    widget = None

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(CeedShapeGroup, self).__init__(**kwargs)
        self.shapes = []
        self.fbind('name', self.dispatch, 'on_changed')

    def on_changed(self, *largs):
        pass

    def add_shape(self, shape):
        '''Adds the shape to the group if it is not already in the group.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to add to :attr:`shapes`.
        '''
        if shape in self.shapes:
            return

        self.shapes.append(shape)
        self.dispatch('on_changed')

    def remove_shape(self, shape):
        '''Removes the shape from the group (and its :attr:`CeedShape.display`)
        if it is present.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to remove from :attr:`shapes`.
        '''
        if shape not in self.shapes:
            return

        self.shapes.remove(shape)
        self.dispatch('on_changed')

    def remove_all(self):
        '''Removes all the shapes from the group.
        '''
        for shape in self.shapes:
            self.remove_shape(shape)


class CeedPaintCircle(CeedShape, PaintCircle):
    '''A circle shape.
    '''

    def _get_collider(self, size):
        x, y = self.center
        r = self.radius
        return CollideEllipse(x=x, y=y, rx=r, ry=r)


class CeedPaintEllipse(CeedShape, PaintEllipse):
    '''An ellipse shape.
    '''

    def _get_collider(self, size):
        x, y = self.center
        rx, ry = self.radius_x, self.radius_y
        return CollideEllipse(x=x, y=y, rx=rx, ry=ry, angle=self.angle)


class CeedPaintPolygon(CeedShape, PaintPolygon):
    '''A polygonal shape.
    '''

    def _get_collider(self, size):
        return Collide2DPoly(points=self.points, cache=False)


class CeedPaintFreeformPolygon(CeedShape, PaintFreeformPolygon):
    '''A polygonal shape.
    '''

    def _get_collider(self, size):
        return Collide2DPoly(points=self.points, cache=False)


# make sure the classes above is used rather than the defaults.
CeedPaintCanvasBehavior.shape_cls_map = {
    'circle': CeedPaintCircle, 'ellipse': CeedPaintEllipse,
    'polygon': CeedPaintPolygon, 'freeform': CeedPaintFreeformPolygon,
    'none': None
}
