'''Stages
=================

A :class:`CeedStage` combines :mod:`ceed.shapes` on the screen with a
:mod:`ceed.function` which determines the intensity of the shape as time
progresses. This module defines the classes used to compute the
intensity values for the shapes during an experimental stage.

See :class:`StageFactoryBase` and :class:`CeedStage` for details.
'''
import numpy as np
import os
from copy import deepcopy

from kivy.properties import OptionProperty, ListProperty, ObjectProperty, \
    StringProperty, NumericProperty, DictProperty, BooleanProperty
from kivy.factory import Factory
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.graphics import Color

from ceed.function import CeedFunc, FuncDoneException
from ceed.utils import fix_name
from ceed.shape import get_painter
from ceed.shape import CeedShapeGroup

__all__ = ('StageDoneException', 'StageFactoryBase', 'CeedStage', 'StageShape',
           'StageFactory')


class StageDoneException(Exception):
    '''Raised as a signal by a :class:`CeedStage` when it is done.
    '''
    pass


class StageFactoryBase(EventDispatcher):
    '''The stage controller.

    The :attr:`StageFactory` is the singleton instance that tracks the stages
    and makes them available in the GUI.

    :Events:

        `on_changed`:
            Triggered whenever the configuration options of a stage changes
            or if a stage is added or removed from the factory.
    '''

    stages = []
    '''The list of the currently available :class:`CeedStage` instances.
    '''

    show_widgets = False
    '''Whether the class is used through the GUI, in which case the stage's
    widget is displayed, otherwise, no widgets are displayed.
    '''

    stage_names = DictProperty([])
    '''A dict of all the stages whose keys are the stage :attr:`CeedStage.name`
    and whose values are the corresponding :class:`CeedStage`.
    '''

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(StageFactoryBase, self).__init__(**kwargs)
        self.stages = []

    def on_changed(self, *largs):
        pass

    def save_stages(self, id_map=None):
        '''Returns a list of the configuration options for all the stages,
        allowing it to be recovered later with :meth:`recover_stages`.

        :Params:

            `id_map`: dict
                A dict that will be filled-in as the function state of the
                stages is saved.
                The keys will be the :attr:`ceed.function.FuncBase.func_id` of
                each of the saved functions. The corresponding value will be
                the :attr:`ceed.function.FuncBase.func_id` of the function in
                :attr:`ceed.function.FuncBase.source_func` if not None. This
                allows the reconstruction of the function dependencies.

                If None, the default, a dict is created. Otherwise, it's
                updated in place.

        :returns:

            A two tuple ``(states, id_map)``. ``states`` is a list of all the
            stages' states. ``id_map`` is the ``id_map`` created or passed
            in.
        '''
        if id_map is None:
            id_map = {}
        for stage in self.stages:
            for s in stage.get_stages():
                for f in s.functions:
                    CeedFunc.fill_id_map(f, id_map)

        states = [s.get_state() for s in self.stages]
        return states, id_map

    def recover_stages(self, stages, id_to_func_map=None, old_id_map=None):
        '''Restores all the stages that was saved with :meth:`save_stages`.

        :Params:

            `stages`: list
                The list of stages' states as returned by :meth:`save_stages`.
            `id_to_func_map`: dict
                A dict that will be filled-in as the stage functions are
                re-created. The keys will be the
                :attr:`ceed.function.FuncBase.func_id` of the new functions
                created from each stage state. The corresponding value
                will be the function created.

                If None, the default, a dict is created. Otherwise, it's
                updated in place.
            `old_id_map`: dict
                A dict that will be filled-in as the functions are re-created.
                The keys will be the :attr:`ceed.function.FuncBase.func_id` of
                the new functions created from each state. The corresponding
                value will be the :attr:`ceed.function.FuncBase.func_id` as
                saved in the ``stages`` state passed in.
                :attr:`ceed.function.FuncBase.func_id` is likely to change
                as a function is re-created; this keeps track of that.

                If None, the default, a dict is created. Otherwise, it's
                updated in place.

        :returns:

            A two-tuple. The first element is list of all the functions created
            while reconstructing the stages. The second is ``id_to_func_map``
            created, or passed in.
        '''
        for state in stages:
            stage = CeedStage()
            if self.show_widgets:
                stage.display
            stage.apply_state(state, clone=True, old_id_map=old_id_map)
            self.add_stage(stage)

        id_to_func_map = {} if id_to_func_map is None else id_to_func_map
        funcs = []

        for stage in self.stages:
            for s in stage.get_stages():
                for f in s.functions:
                    CeedFunc.fill_id_to_func_map(f, id_to_func_map)
                    funcs.append(f)

        return funcs, id_to_func_map

    def add_stage(self, stage):
        '''Adds the :class:`CeedStage` to the stage factory (:attr:`stages`).

        :Params:

            `stage`: :class:`CeedStage`
                The stage to add.
        '''
        stage.name = fix_name(stage.name, [s.name for s in self.stages])
        self.stages.append(stage)
        self.stage_names[stage.name] = stage
        if self.show_widgets:
            stage.display.show_stage()
        self.dispatch('on_changed')

    def remove_stage(self, stage):
        '''Removes the :class:`CeedStage` from the stage factory
        (:attr:`stages`).

        :Params:

            `stage`: :class:`CeedStage`
                The stage to remove.
        '''
        self.stages.remove(stage)
        del self.stage_names[stage.name]

        if stage._display:
            stage._display.hide_stage()
        self.dispatch('on_changed')

    def clear_stages(self):
        '''Removes all the stages from the factory (:class:`stages`).
        '''
        for stage in self.stages[:]:
            self.remove_stage(stage)

    def remove_shape_from_all(self, _, shape):
        '''Removes the :class:`ceed.shape.CeedShape` instance from all the
        :class:`CeedStage` instances that it is a associated with.

        :Params:

            `_`: anything
                This parameter is ignored and can be anything.
            `shape`: :class:`ceed.shape.CeedShape`
                The shape to remove from all stages.
        '''
        for stage in self.stages:
            for sub_stage in stage.get_stages():
                for stage_shape in sub_stage.shapes:
                    if stage_shape.shape is shape:
                        sub_stage.remove_shape(stage_shape)

    def _change_stage_name(self, stage, name):
        '''Makes sure that a stage's name is unique.
        '''
        if stage.name == name:
            return name

        del self.stage_names[stage.name]
        name = fix_name(name, self.stage_names)
        stage.name = name
        self.stage_names[stage.name] = stage
        return name

    def tick_stage(self, stage_name):
        '''An iterator which starts a :class:`CeedStage` and ticks the time for
        every call.

        A :class:`CeedStage` represents a collection of shapes with functions
        applied to them. Each of these shapes has a defined intensity for
        every time point. This iterator walks through time denoting the
        intensity for each shape for every time point.

        E.g. to get the shape values for time 0, .1, .2, ..., 1.0 for the
        stage named ``'my_stage'``::

            >>> tick = StageFactory.tick_stage('my_stage')  # get iterator
            >>> for t in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
            >>>     try:
            >>>         tick.next()
            >>>         shape_values = tick.send(t)
            >>>         print(shape_values)
            >>>     except StageDoneException:
            >>>         # if we're here the stage has completed
            >>>         break

        To use it, create the iterator, then for every time point call `next()`
        on the iterator and then send the monotonically increasing time to
        the function using `send()` and in return the iterator will return
        the associated intensity values for each shape for that time point.

        :Params:

            `stage_name`: str
                The :attr:`CeedStage.name` of the stage to start.

        :yields:

            A list of the intensity values for each shape.

            Each item in the list is a 2-tuple of ``(name, values)``. ``name``
            is the :attr:`cplcom.painter.PaintShape.name` of the shape.
            ``values`` is a list of color values and each item in that list is
            a 4-tuple of ``(r, g, b, a)``. Any of these values can be None, in
            which case that color remains the same. This way a shape can be
            updated from multiple sub-stages, where only e.g. the ``r`` value
             is changed.

        :raises:

            `StageDoneException`:
                When done with the stage (time is out of bounds).
        '''
        stage = self.stage_names[stage_name]
        shapes = {s.name: [] for s in get_painter().shapes}
        tick_stage = stage.tick_stage(shapes)
        next(tick_stage)
        while True:
            t = yield
            tick_stage.send(t)

            shape_values = []
            for name, colors in shapes.items():
                shape_values.append((name, colors[:]))
                del colors[:]

            yield shape_values

    def get_shapes_gl_color_instructions(self, canvas, name):
        '''Adds all the kivy OpenGL instructions required to display the
        intensity-varying shapes to the kivy canvas and returns the color
        classes that control the color of each shape.

        :Params:

            `canvas`: Kivy canvas instance
                The canvas to which the gl instructions are added. It add e.g.
                the polygon and its color.
            `name`: str
                The name to associate with these OpenGL instructions.
                The name is used later to remove the instructions as it allows
                to clear all the instructions with that name.

        :returns:

            a dict whose keys is the :attr:`cplcom.painter.PaintShape.name` of
            the :class:`ceed.shape.CeedShape` and whose value is the Kivy
            ``Color`` instruction instance that controls the color of the
            shape.
        '''
        shape_views = {}
        for shape in get_painter().shapes:
            instructions = shape.add_shape_instructions(
                (0, 0, 0, 1), name, canvas)
            if not instructions:
                raise ValueError('Malformed shape {}'.format(shape.name))

            shape_views[shape.name] = instructions[0]
        return shape_views

    def fill_shape_gl_color_values(self, shape_views, shape_values,
                                   projection=None):
        '''Takes the dict of the Colors instance that control the color of
        each shape as well as the list of the color values for a time point
        and sets the shape colors to those values.

        :Params:

            `shape_views`: dict
                The dict of shape names and shapes colors as returned by
                :meth:`get_shapes_gl_color_instructions`. If it is None,
                the color will not be updated but the return result will be
                identical to when not None.
            `shape_values`: list
                The list of color intensity values to use for each shape as
                yielded by :meth:`tick_stage`.
            `projection`: str
                The colors to operate one. Can be any subset of the string
                'rgb'. Specifically, although we get intensity values for every
                color, this takes the average intensity of the colors listed in
                ``projection`` and assigns their mean to all those colors.
                It's how we turn the color into a gray-scale value.

        :returns:

            A list of the colors each shape was set to. Each item in the list
            is ``(name, r, g, b, a)``, where ``name`` is the shape's name and
            ``r``, ``g``, ``b``, ``a`` is the color value it was set to.



        E.g. to display the shape intensities for time 0, .1, .2, ..., 1.0 for
        the stage named ``'my_stage'`` in real time::

            >>> import time
            >>> tick = StageFactory.tick_stage('my_stage')  # get iterator
            >>> # get the color objects
            >>> colors = StageFactory.get_shapes_gl_color_instructions(
            >>>     widget_canvas, 'my_canvas_instructs')
            >>> for t in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
            >>>     try:
            >>>         tick.next()
            >>>         shape_values = tick.send(t)  # current color values
            >>>         # update the colors and print it
            >>>         values = StageFactory.fill_shape_gl_color_values(
            >>>>            colors, shape_values)
            >>>         print(values)
            >>>         time.sleep(0.1)  # wait until next time
            >>>     except StageDoneException:
            >>>         # if we're here the stage has completed
            >>>         break
            >>> # now remove the shapes
            >>> StageFactory.remove_shapes_gl_color_instructions(
            >>>     widget_canvas, 'my_canvas_instructs')
        '''
        result = []

        for name, colors in shape_values:
            r = g = b = a = None
            for r2, g2, b2, a2 in colors:
                if r2 is not None:
                    r = r2
                if g2 is not None:
                    g = g2
                if b2 is not None:
                    b = b2
                if a2 is not None:
                    a = a2

            if r is not None:
                r = min(max(r, 0.), 1.)
            if g is not None:
                g = min(max(g, 0.), 1.)
            if b is not None:
                b = min(max(b, 0.), 1.)
            if a is not None:
                a = min(max(a, 0.), 1.)

            color = shape_views[name] if shape_views is not None else None
            if r is None and b is None and g is None and a is None:
                if color is not None:
                    color.rgba = 0, 0, 0, 0
                result.append((name, 0, 0, 0, 0))
            elif projection:
                if a is None:
                    a = 1

                vals = [v for v in (r, g, b) if v is not None]
                if not vals:
                    val = 0
                else:
                    val = sum(vals) / float(len(vals))

                if color is not None:
                    color.a = a
                    setattr(color, projection, val)
                result.append((name, val, val, val, a))
            else:
                r, g, b = r or 0, g or 0, b or 0
                if a is None:
                    a = 1
                if color is not None:
                    color.rgba = r, g, b, a
                result.append((name, r, g, b, a))
        return result

    def remove_shapes_gl_color_instructions(self, canvas, name):
        '''Removes all the shape and color instructions that was added with
        :meth:`get_shapes_gl_color_instructions`.

        :Params:

            `canvas`: Kivy canvas instance
                The canvas to which the gl instructions were added.
            `name`: str
                The name used when adding the instructions with
                :meth:`get_shapes_gl_color_instructions`.
        '''
        if canvas:
            canvas.remove_group(name)


class CeedStage(EventDispatcher):
    '''The stage that controls a time period of an experiment.

    A stage is composed of multiple shape objects, :attr:`shapes`, a series
    of :class:`ceed.function.FuncBase` functions, :attr:`functions`, that
    describe the intensity values these shapes take across time, and
    sub-stages, :attr:`stages`, that are similarly evaluated during this stage.

    During a :class:`CeedStage`, if there are child :attr:`stages` that have
    their own shapes we also tick through these stages while the root stage is
    evaluated. That means the shapes associated with the root stage intensity
    values is updated as are the the intensity values of the shapes from the
    sub-stage.

    .. image:: Stage_order.png

    E.g. if we have a root stage (blue) which contains 4 children
    :attr:`stages` A, B, C, D (purple to orange) as in the image.
    If :attr:`order` is ``'serial'`` as in the lower part, then each sub-stage
    is evaluated sequentially after the previous sub-stage has finished.

    If it's ``'parallel'``, the upper part, then each sub-stage is evaluated
    simultaneously with the root stage.

    If :attr:`complete_on` is ``'any'`` then the root stage will complete after
    stage ``A`` completes because it completes first in each case. If it is
    ``'all'`` then it will complete after the root stage is completed because
    that is the longest.

    In all cases, shapes that are not associated with a stage that is
    currently being evaluated will be set to transparent. E.g. in the serial
    case, shapes that are associated with stage C will be transparent in all
    other stages, except if they also appear in those stages.

    :Events:

        `on_changed`:
            Triggered whenever a stage's configuration option changes or
            if one of the functions or shapes of the stage is added/removed.
    '''

    name = StringProperty('Stage')
    '''The name of this stage.
    '''

    order = OptionProperty('serial', options=['serial', 'parallel'])
    '''The order in which the sub-stages, :attr:`stages`, are evaluated.
    Can be one of ``'serial'``, ``'parallel'``.

    See :class:`CeedStage` description for details.
    '''

    complete_on = OptionProperty('all', options=['all', 'any'])
    '''How to terminate a stage when there are sub-stages, :attr:`stages`
    present. Can be one of ``'all'``, ``'parallel'``.

    See :class:`CeedStage` description for details.
    '''

    stages = ListProperty([])
    '''A list of :class:`CeedStage` instances that are sub-stages of this
    stage.

    See :class:`CeedStage` description for details.
    '''

    parent_stage = ObjectProperty(None, rebind=True, allownone=True)
    '''The parent stage when this stage is a sub-stage of another.
    '''

    functions = ListProperty([])
    '''A list of :class:`ceed.function.FuncBase` instances through which the
    stage iterates through sequentially and updates the intensity of the
    :attr:`shapes` to the function value at each time point.
    '''

    shapes = ListProperty([])
    '''The list of :class:`StageShape` instances that are associated
    with this stage. All the shapes are set to the same intensity value at
    every time point according to the :attr:`functions` value at that
    time point.
    '''

    color_r = BooleanProperty(False)
    '''Whether the :attr:`shapes` red channel should be set to the
    :attr:`functions` value or if it should remain at zero intensity.
    '''

    color_g = BooleanProperty(False)
    '''Whether the :attr:`shapes` green channel should be set to the
    :attr:`functions` value or if it should remain at zero intensity.
    '''

    color_b = BooleanProperty(True)
    '''Whether the :attr:`shapes` blue channel should be set to the
    :attr:`functions` value or if it should remain at zero intensity.
    '''

    color_a = NumericProperty(None, allownone=True)
    '''Whether the :attr:`shapes` alpha channel should be set. Currently
    this is ignored.
    '''

    _display = None

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(CeedStage, self).__init__(**kwargs)
        for name in self.get_state():
            self.fbind(name, self.dispatch, 'on_changed')
        self.fbind('on_changed', StageFactory.dispatch, 'on_changed')

    def on_changed(self, *largs):
        pass

    def get_state(self, state=None):
        '''Returns a dict representation of the stage so that it can be
        reconstructed later with :meth:`apply_state`.

        :Params:

            `state`: dict
                A dict with the state, to which configuration items and their
                values are added. If None, the default, a dict is created and
                returned.

        :returns:

            A dict with all the configuration data.
        '''
        d = {}
        for name in ('order', 'name', 'color_r', 'color_g', 'color_b',
                     'complete_on'):
            d[name] = getattr(self, name)

        d['stages'] = [s.get_state() for s in self.stages]
        d['functions'] = [f.get_state() for f in self.functions]
        d['shapes'] = [s.name for s in self.shapes]

        if state is None:
            state = d
        else:
            state.update(d)
        return state

    def apply_state(self, state={}, clone=False, old_id_map=None):
        '''Takes the state of the stage saved with :meth:`get_state` and
        applies it to this stage. it also creates any children functions and
        stages and creates the references to the :attr:`shapes`.

        :Params:

            `state`: dict
                The dict to use to reconstruct the stage as returned by
                :meth:`get_state`.
            `clone`: bool
                If True will copy all the state, otherwise it doesn't
                copy internal parameters.
            `old_id_map`: dict
                A dict that will be filled-in as the functions are re-created.
                The keys will be the :attr:`ceed.function.FuncBase.func_id` of
                the new functions created from each state. The corresponding
                value will be the :attr:`ceed.function.FuncBase.func_id` as
                saved in the ``stages`` state passed in.
                :attr:`ceed.function.FuncBase.func_id` is likely to change
                as a function is re-created; this keeps track of that.

                If None, the default, a dict is created. Otherwise, it's
                updated in place.
        '''
        stages = state.pop('stages', [])
        functions = state.pop('functions', [])
        shapes_state = state.pop('shapes', [])

        for k, v in state.items():
            setattr(self, k, v)

        for data in stages:
            s = CeedStage()
            if self._display:
                s.display
            s.apply_state(data, clone=True)
            self.add_stage(s)

        for data in functions:
            self.add_func(
                CeedFunc.make_func(data, clone=True, old_id_map=old_id_map))

        shapes = get_painter().shape_names
        groups = get_painter().shape_group_names
        for name in shapes_state:
            if name in shapes:
                self.add_shape(shapes[name])
            elif name in groups:
                self.add_shape(groups[name])

    def duplicate_stage(self):
        stage = CeedStage()
        if self._display:
            stage.display

        for name in self.get_state():
            setattr(stage, name, getattr(self, name))

        for child_stage in self.stages:
            stage.add_stage(child_stage.duplicate_stage())

        for func in self.functions:
            stage.add_func(deepcopy(func))

        for shape in self.shapes:
            stage.add_shape(shape.shape)
        return stage

    @property
    def display(self):
        '''The GUI widget associated with the stage and displayed to the
        user.
        '''
        if self._display:
            return self._display

        w = self._display = Factory.StageWidget(stage=self)
        return w

    def add_stage(self, stage):
        '''Adds a sub-stage instance :class:`CeedStage` to :attr:`stages`.
        '''
        self.stages.append(stage)
        stage.parent_stage = self

        if self._display:
            stage.display.show_stage()

    def remove_stage(self, stage):
        '''Removes a sub-stage instance :class:`CeedStage` from :attr:`stages`.
        '''
        self.stages.remove(stage)
        stage.parent_stage = None

        if self._display and stage._display:
            stage._display.hide_stage()

    def add_func(self, func, after=None):
        '''Adds the function instance :class:`ceed.function.FuncBase` to
        :attr:`functions`.

        :params:

            `func`: :class:`ceed.function.FuncBase`
                The function to add.
            `after`: :class:`ceed.function.FuncBase`
                The function in :attr:`functions` after which to add this
                function, if not None, the default.
        '''
        i = None
        if after is None:
            self.functions.append(func)
        else:
            i = self.functions.index(after)
            self.functions.insert(i + 1, func)
        if self._display:
            self._display.set_func_controller(func.display)
            func.display.show_func(i)

    def remove_func(self, func):
        '''Removes the function instance :class:`ceed.function.FuncBase` from
        :attr:`functions`.
        '''
        self.functions.remove(func)
        if func._display:
            func._display.hide_func()

    def add_shape(self, shape):
        '''Adds a :class:`StageShape` instance wrapping the
        :class:`ceed.shape.CeedShape` ``shape`` to the :attr:`shapes`. If
        the ``shape`` was already added it doesn't add it again.

        :Params:

            `shape`: :class:`ceed.shape.CeedShape`
                The shape instance to add.

        :returns:

            The :class:`StageShape` created if the shape wasn't already added,
            otherwise None.
        '''
        if any([s for s in self.shapes if shape.name == s.name]):
            return None
        stage_shape = StageShape(stage=self, shape=shape)
        self.shapes.append(stage_shape)
        if self._display:
            stage_shape.display.show_widget()
        return stage_shape

    def remove_shape(self, stage_shape):
        '''Removes a :class:`StageShape` that was previously added to
        :attr:`shapes`.
        '''
        self.shapes.remove(stage_shape)

        if self._display and stage_shape._display:
            stage_shape.display.hide_widget()

    def get_stages(self):
        '''Iterator that iterates depth-first through all the stages
        and children :attr:`stages and yields these stages.`
        '''
        yield self
        for stage in self.stages:
            for s in stage.get_stages():
                yield s

    def tick_stage(self, shapes):
        '''Similar to :meth:`StageFactoryBase.tick_stage` but for this stage.

        It is an iterator that iterates through time and returns the
        intensity values for the shapes associated with this stage and its
        sub-stages.

        Specifically, at every iteration a time value is sent to the iterator
        which then updates the ``shapes`` dict with the intensity values of the
        shape for this time-point.

        :Params:

            `shapes`: dict
                A dict whose keys is the name of all the shapes of this stage
                and its sub-stages. The corresponding values are empty lists.
                At every iteration the list will be filled in with color values
                and should be cleared before the next iteration.

        :raises:

            `StageDoneException`:
                When done with the stage (time is out of bounds).

        E.g. to get the shape values for time 0, .1, .2, ..., 1.0 for this
        stage::

            >>> # get dict of shapes using the painter controller
            >>> shapes = {s.name: [] for s in get_painter().shapes}
            >>> tick_stage = stage.tick_stage(shapes)  # create iterator
            >>> next(tick_stage)
            >>> for t in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
            >>>     try:
            >>>         tick_stage.send(t)  # update
            >>>     except StageDoneException:
            >>>         break  # if we're here the stage has completed
            >>>
            >>>     for name, colors in shapes.items():
            >>>         print(name, colors)
            >>>         del colors[:]  # clear for next iteration

        In the example above. ``colors`` for each shape is a list of
        4-tuple r, g, b, a values, each of which can be None similarly to what
        is described at :meth:`StageFactoryBase.tick_stage`.
        '''
        names = set()
        for shape in self.shapes:
            shape = shape.shape
            if isinstance(shape, CeedShapeGroup):
                for shape in shape.shapes:
                    names.add(shape.name)
            else:
                names.add(shape.name)
        names = list(names)

        stages = [s.tick_stage(shapes) for s in self.stages]
        funcs = self.tick_funcs()
        serial = self.order == 'serial'
        end_on_first = self.complete_on == 'any' and not serial
        r, g, b = self.color_r, self.color_g, self.color_b
        a = self.color_a
        for tick_stage in stages[:]:
            next(tick_stage)

        while funcs is not None or stages:
            t = yield
            if funcs is not None:
                try:
                    next(funcs)
                    val = funcs.send(t)
                    values = (val if r else None, val if g else None,
                              val if b else None, a)
                    for name in names:
                        shapes[name].append(values)
                except FuncDoneException:
                    funcs = None
                    if end_on_first:
                        del stages[:]

            for tick_stage in stages[:]:
                try:
                    tick_stage.send(t)
                    if serial:
                        break
                except StageDoneException:
                    if end_on_first:
                        del stages[:]
                        funcs = None
                        break
                    stages.remove(tick_stage)

        raise StageDoneException

    def tick_funcs(self):
        '''Iterates through the :attr:`functions` of this stage sequentially
        and returns the function's value associated with that time.

        E.g. to get the function value for time 0, .1, .2, ..., 1.0 for this
        stage::

            >>> func = self.tick_funcs()
            >>> for t in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
            >>>     try:
            >>>         next(funcs)
            >>>         val = funcs.send(t)
            >>>         print(val)
            >>>     except FuncDoneException:
            >>>         break  # function is done
        '''
        raised = False
        for func in self.functions:
            try:
                if not raised:
                    t = yield
                raised = False
                func.init_func(t)
                yield func(t)
                while True:
                    t = yield
                    yield func(t)
            except FuncDoneException:
                raised = True
        raise FuncDoneException


class StageShape(EventDispatcher):
    '''A wrapper for :class:`ceed.shape.CeedShape` instances used in
    :meth:`CeedStage.add_shape` to wrap a shape or shape group to be used by
    the stage.
    '''

    shape = ObjectProperty(None, rebind=True)
    '''The :class:`ceed.shape.CeedShape` or :class:`ceed.shape.CeedShapeGroup`
    instance being wrapped.
    '''

    stage = ObjectProperty(None, rebind=True)
    '''The :class:`CeedStage` this is associated with.
    '''

    name = StringProperty('')
    '''The :attr:`ceed.shape.CeedShapeGroup.name` or
    :attr:`cplcom.painter.PaintShape.name` of the instance wrapped.
    '''

    _display = None

    def __init__(self, **kwargs):
        super(StageShape, self).__init__(**kwargs)
        self.shape.fbind('name', self._update_name)
        self._update_name()

    def _update_name(self, *largs):
        self.name = self.shape.name

    @property
    def display(self):
        '''The GUI widget associated with the shape and displayed to the
        user in the stage.
        '''
        if self._display:
            return self._display

        w = self._display = Factory.StageShapeDisplay(stage_shape=self)
        return w

StageFactory = StageFactoryBase()
'''The singleton :class:`StageFactoryBase` instance through which all stage
operations are accessed.
'''

def _bind_remove(*largs):
    get_painter().fbind('on_remove_shape', StageFactory.remove_shape_from_all)
    get_painter().fbind('on_remove_group', StageFactory.remove_shape_from_all)
if not os.environ.get('KIVY_DOC_INCLUDE', None):
    Clock.schedule_once(_bind_remove, 0)
