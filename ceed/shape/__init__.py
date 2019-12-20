"""Shapes
============

Defines the shapes which are used with a :mod:`ceed.function` to create regions
with time-varying intensity during an experiment :mod:`ceed.stage`.

:class:`~kivy_garden.painter.PaintCanvasBehavior` provides the widget
functionality such that when a user draws on the screen,
:class:`~kivy_garden.painter.PaintShape` instances are created and added to the
widget.

:class:`CeedPaintCanvasBehavior` provides a more specialized painter canvas.
Specifically, in addition to being able to draw shapes, it adds the ability
to organize shapes into :attr:`CeedPaintCanvasBehavior.groups`.
"""
import math
from typing import Type, List, Tuple, Dict, Optional, Union

from kivy.properties import BooleanProperty, NumericProperty, StringProperty, \
    ObjectProperty, DictProperty, ListProperty
from kivy.event import EventDispatcher
from kivy.factory import Factory
from kivy_garden.collider import Collide2DPoly, CollideEllipse

from kivy_garden.painter import PaintCanvasBehavior, PaintCircle,\
    PaintEllipse, PaintPolygon, PaintFreeformPolygon

from ceed.utils import fix_name

__all__ = (
    'CeedPaintCanvasBehavior', 'CeedShape', 'CeedShapeGroup',
    'CeedPaintCircle', 'CeedPaintEllipse', 'CeedPaintPolygon',
    'CeedPaintFreeformPolygon')


class CeedPaintCanvasBehavior(PaintCanvasBehavior):
    """Controller base class for drawing and managing the shapes.

    A shape is drawn by the user in the GUI and added by
    :class:`kivy_garden.painter.PaintCanvasBehavior` to its
    :attr:`kivy_garden.painter.PaintCanvasBehavior.shapes` list. This class
    manages all that without any of the associated GUI components that
    is shown to the user (e.g. the ability name shapes etc.). The GUI
    components is added by the :class:`ceed.shape.shape_widgets.CeedPainter`.

    So when run from the GUI, an instance of
    :class:`ceed.shape.shape_widgets.CeedPainter` is the class used to manage
    the shapes. When running during analysis, and instance of this class is
    used instead as no visualizations is required..

    In addition to the :attr:`kivy_garden.painter.PaintCanvasBehavior.shapes`,
    the class adds :attr:`groups` for grouping shapes; a
    :class:`CeedShapeGroup` simply groups a collection of :class:`CeedShape`
    by their name.

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
    """

    groups: List['CeedShapeGroup'] = ListProperty([])
    '''List of :class:`CeedShapeGroup` instances.
    '''

    shape_names: Dict[str, 'CeedShape'] = DictProperty({})
    '''The name -> :class:`CeedShape` dict. The key is the shape's name
    and the corresponding value is the :class:`CeedShape` instance.
    '''

    shape_group_names: Dict[str, 'CeedShapeGroup'] = DictProperty({})
    '''The name -> :class:`CeedShapeGroup` dict. The key is the group's name
    and the corresponding value is the :class:`CeedShapeGroup` instance.
    '''

    __events__ = ('on_remove_shape', 'on_remove_group', 'on_changed')

    def add_shape(self, shape: 'CeedShape'):
        if not super(CeedPaintCanvasBehavior, self).add_shape(shape):
            return False

        # make sure the name is unique
        name = fix_name(
            shape.name, self.shape_names, self.shape_group_names)
        self.shape_names[name] = shape
        shape.name = name
        shape.fbind('name', self._change_shape_name)
        self.dispatch('on_changed')
        return True

    def remove_shape(self, shape: 'CeedShape'):
        if not super(CeedPaintCanvasBehavior, self).remove_shape(shape):
            return False

        shape.funbind('name', self._change_shape_name)
        self.remove_shape_from_groups(shape)
        del self.shape_names[shape.name]
        self.dispatch('on_remove_shape', shape)
        self.dispatch('on_changed')
        return True

    def reorder_shape(self,
                      shape: 'CeedShape', before_shape: 'CeedShape' = None):
        self.dispatch('on_changed')
        return super(CeedPaintCanvasBehavior, self).reorder_shape(
            shape, before_shape=before_shape)

    def move_shape_lower(self, shape: 'CeedShape'):
        """Moves the shape one shape down. I.e. if there are two shapes and
        this shape is at index 1, it gets moved to index 0.

        This changes the depth ordering of the shapes.

        :param shape: The :class:`CeedShape` instance to move. It must exist in
            :attr:`shapes`.
        """
        i = self.shapes.index(shape)
        if not i:
            return

        before_shape = self.shapes[i - 1]
        self.reorder_shape(shape, before_shape)

    def move_shape_upwards(self, shape: 'CeedShape'):
        """Moves the shape one shape up. I.e. if there are two shapes and
        this shape is at index 0, it gets moved to index 1.

        This changes the depth ordering of the shapes.

        :param shape: The :class:`CeedShape` instance to move. It must exist in
            :attr:`shapes`.
        """
        i = self.shapes.index(shape)
        if i == len(self.shapes) - 1:
            return

        if i == len(self.shapes) - 2:
            before_shape = None
        else:
            before_shape = self.shapes[i + 2]
        self.reorder_shape(shape, before_shape)

    def add_group(self, group: Optional['CeedShapeGroup'] = None):
        """Similar to :meth:`add_shape` but for a :class:`CeedShapeGroup`.

        :Params:

            `group`: :class:`CeedShapeGroup`
                The group to add. If None, the default, a new
                :class:`CeedShapeGroup` is created and added.

        :returns:

            The :class:`CeedShapeGroup` added.
        """
        if group is None:
            group = CeedShapeGroup(paint_widget=self)
        self.groups.append(group)

        name = fix_name(
            group.name, self.shape_names, self.shape_group_names)
        self.shape_group_names[name] = group
        group.name = name
        group.fbind('name', self._change_shape_name)
        self.dispatch('on_changed')
        return group

    def remove_group(self, group: 'CeedShapeGroup'):
        """Similar to :meth:`remove_shape` but for a :class:`CeedShapeGroup`.

        :Params:

            `group`: :class:`CeedShapeGroup`
                The group to remove.

        :returns:

            True if the group was removed, False otherwise.
        """
        group.funbind('name', self._change_shape_name)
        del self.shape_group_names[group.name]
        self.groups.remove(group)
        self.dispatch('on_remove_group', group)
        self.dispatch('on_changed')
        return True

    def remove_all_groups(self):
        """Removes all the :class:`CeedShapeGroup` instances in :attr:`groups`.
        """
        for group in self.groups[:]:
            self.remove_group(group)

    def add_selected_shapes_to_group(self, group: 'CeedShapeGroup' = None):
        """Adds all the
        :attr:`kivy_garden.painter.PaintCanvasBehavior.selected_shapes` to the
        ``group``.

        :Params:

            `group`: :class:`CeedShapeGroup`
                The group to which to add the shapes. If None, the default, a
                new :class:`CeedShapeGroup` is created.

        :returns:

            The :class:`CeedShapeGroup` passed in or created.
        """
        if group is None:
            group = self.add_group()

        for shape in self.selected_shapes:
            if shape not in group.shapes:
                group.add_shape(shape)
        return group

    def remove_shape_from_groups(self, shape: 'CeedShape'):
        """Removes the :class:`CeedShape` from all the groups.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to remove.
        """
        for group in self.groups:
            if shape in group.shapes:
                group.remove_shape(shape)

    def get_state(self):
        """Returns a dictionary containing all the configuration data for all
        the shapes and groups. It is used with :meth:`set_state` to later
        restore the state.
        """
        d = {
            'shapes': [s.get_state() for s in self.shapes],
            'groups': [{'name': g.name, 'shapes': [s.name for s in g.shapes]}
                       for g in self.groups],
        }
        return d

    def create_shape_from_state(
            self, state: dict, old_name_map: Dict[str, 'CeedShape']):
        """Overrides
        :meth:`kivy_garden.painter.PaintCanvasBehavior.create_shape_from_state`
        and changes its signature.

        It takes an additional parameter, ``old_name_map``. When a shape is
        created from the given ``state``, the shape's new name could have been
        changed automatically so that it remained unique. ``old_name_map`` is a
        dict that is filled in so the key is the old name (if present in
        ``state``) and the associated value is the actual final shape name.
        """
        old_name = state.get('name', '')
        shape = super(CeedPaintCanvasBehavior, self).create_shape_from_state(
            state)
        if old_name:
            old_name_map[old_name] = shape
        return shape

    def set_state(
            self, state: dict,
            old_name_map: Dict[str, Union['CeedShape', 'CeedShapeGroup']]):
        """Takes the dict returned by :meth:`get_state` and adds the shapes
        and groups created form them.

        ``old_name_map`` is the same as in :meth:`create_shape_from_state`.
        """
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
                group.add_shape(shape)

        self.dispatch('on_changed')

    def on_remove_shape(self, shape):
        pass

    def on_remove_group(self, group):
        pass

    def on_changed(self, *largs):
        pass

    def _change_shape_name(self, shape: 'CeedShape', new_name):
        """Makes sure that the shape or group name is unique.
        """
        if isinstance(shape, CeedShape):
            container = self.shape_names
        else:
            container = self.shape_group_names

        # get the new name
        for name, s in container.items():
            if s is shape:
                if shape.name == name:
                    return name

                del container[name]
                # only one change at a time happens because of binding
                break
        else:
            raise ValueError(
                '{} has not been added to the factory'.format(shape))

        new_name = fix_name(new_name, self.shape_names, self.shape_group_names)
        container[new_name] = shape
        shape.name = new_name

        if not new_name:
            shape.name = fix_name(
                'name', self.shape_names, self.shape_group_names)
        self.dispatch('on_changed')


class CeedShape(object):
    """A co-base class used with :class:`kivy_garden.painter.PaintShape`
    derived classes to add ceed specific functionality to the
    :class:`kivy_garden.painter.PaintShape` classes.
    """

    name: str = StringProperty('Shape')
    """A unique name associated with the shape.
    """

    _collider = None

    _area = None

    _bounding_box = None

    _centroid = None

    paint_widget_size = 0, 0
    """The size of the area used to draw the shapes. A shape is not allowed to
    have coordinates outside this area.
    """

    widget = None
    """The :class:`~ceed.shape.shape_widgets.WidgetShape` used by ceed to
    customize the shape in the GUI.
    """

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
        """Returns the ``(x1, y1, x2, y2)`` describing the lower left and upper
        right points that define a bounding box for the shape.
        """
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
        """Returns the estimated ``(x, y)`` centroid of the shape.
        """
        if self._centroid is not None:
            return self._centroid

        collider = self.collider
        if collider is None:
            return 0, 0

        self._centroid = xc, yc = self.collider.get_centroid()
        return xc, yc

    @property
    def area(self):
        """Returns the estimated area of the shape.
        """
        if self._area is not None:
            return self._area

        collider = self.collider
        if collider is None:
            return 0

        self._area = area = float(self.collider.get_area())
        return area

    def set_area(self, area):
        """Sets the internal area of the shape in pixels. We try to get close
        to the requested area, but it is not likely to be exact.

        :param area: The area to be set.
        """
        if not area:
            return

        scale = 1 / math.sqrt(self.area / float(area))
        self.rescale(scale)

    @property
    def collider(self):
        """Returns the collider instance from :mod:`kivy_garden.collider` used
        to manipulate and measure the shapes.
        """
        if not self.is_valid or not self.finished:
            return None
        if self._collider is not None:
            return self._collider

        self._collider = collider = self._get_collider(self.paint_widget_size)
        return collider


class CeedShapeGroup(EventDispatcher):
    """Holds a collection of :class:`CeedShape` instances.

    It is helpful to group them when the same :class:`ceed.function` is to be
    applied to multiple shapes.

    :Events:

        `on_changed`:
            Triggered whenever a child :class:`CeedShape`
            is added or removed, or if a configuration option of the objects
            is changed.
    """

    paint_widget = ObjectProperty(None)
    '''The same as :attr:`kivy_garden.painter.PaintShape.paint_widget`.
    '''

    name: str = StringProperty('Group')
    '''The name of the group. Similar to :attr:`CeedShape.name`.
    '''

    shapes: List[CeedShape] = []
    '''A list that contains the :class:`CeedShape` instances that are part of
    this group.
    '''

    widget = None
    """The :class:`~ceed.shape.shape_widgets.WidgetShapeGroup` used by ceed to
    customize the group in the GUI.
    """

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(CeedShapeGroup, self).__init__(**kwargs)
        self.shapes = []
        self.fbind('name', self.dispatch, 'on_changed')

    def on_changed(self, *largs):
        pass

    def add_shape(self, shape: CeedShape):
        """Adds the shape to the group if it is not already in the group.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to add to :attr:`shapes`.

        :returns:

            Whether the shape was successfully added to the group.
        """
        if shape in self.shapes:
            return False

        if self.widget is not None:
            self.widget.add_shape(shape)
        self.shapes.append(shape)
        self.dispatch('on_changed')
        return True

    def remove_shape(self, shape: CeedShape):
        """Removes the shape from the group (and its :attr:`CeedShape.widget`)
        if it is present.

        :Params:

            `shape`: :class:`CeedShape`
                The shape to remove from :attr:`shapes`.
        """
        if shape not in self.shapes:
            return

        if self.widget is not None:
            self.widget.remove_shape(shape)
        self.shapes.remove(shape)
        self.dispatch('on_changed')

    def remove_all(self):
        """Removes all the shapes from the group.
        """
        for shape in self.shapes[:]:
            self.remove_shape(shape)


class CeedPaintCircle(CeedShape, PaintCircle):
    """A circle shape.
    """

    def _get_collider(self, size):
        x, y = self.center
        r = self.radius
        return CollideEllipse(x=x, y=y, rx=r, ry=r)


class CeedPaintEllipse(CeedShape, PaintEllipse):
    """An ellipse shape.
    """

    def _get_collider(self, size):
        x, y = self.center
        rx, ry = self.radius_x, self.radius_y
        return CollideEllipse(x=x, y=y, rx=rx, ry=ry, angle=self.angle)


class CeedPaintPolygon(CeedShape, PaintPolygon):
    """A polygonal shape.
    """

    def _get_collider(self, size):
        return Collide2DPoly(points=self.points, cache=False)


class CeedPaintFreeformPolygon(CeedShape, PaintFreeformPolygon):
    """A polygonal shape.
    """

    def _get_collider(self, size):
        return Collide2DPoly(points=self.points, cache=False)


# make sure the classes above is used rather than the defaults.
CeedPaintCanvasBehavior.shape_cls_map = {
    'circle': CeedPaintCircle, 'ellipse': CeedPaintEllipse,
    'polygon': CeedPaintPolygon, 'freeform': CeedPaintFreeformPolygon,
    'none': None
}
