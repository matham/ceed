"""Stages
=================

A :class:`CeedStage` combines :mod:`ceed.shapes` on the screen with a
:mod:`ceed.function` which determines the intensity of the shape as time
progresses. This module defines the classes used to compute the
intensity values for the shapes during an experimental stage.

See :class:`StageFactoryBase` and :class:`CeedStage` for details.

remove_shapes_upon_deletion must be bound
"""
from copy import deepcopy
from collections import defaultdict
from typing import Dict, List, Union, Tuple

from kivy.properties import OptionProperty, ListProperty, ObjectProperty, \
    StringProperty, NumericProperty, DictProperty, BooleanProperty
from kivy.event import EventDispatcher
from kivy.graphics import Color

from ceed.function import CeedFunc, FuncDoneException, CeedFuncRef, \
    FunctionFactoryBase, FuncBase
from ceed.utils import fix_name, update_key_if_other_key
from ceed.shape import CeedShapeGroup, CeedPaintCanvasBehavior, CeedShape

__all__ = ('StageDoneException', 'StageFactoryBase', 'CeedStage', 'StageShape',
           'CeedStageRef', 'remove_shapes_upon_deletion',
           'last_experiment_stage_name')


last_experiment_stage_name = 'experiment_sampled'


class StageDoneException(Exception):
    """Raised as a signal by a :class:`CeedStage` when it is done.
    """
    pass


class StageFactoryBase(EventDispatcher):
    """A global store of the defined :class:`CeedStage`
    customized function instances.

    See :mod:`ceed.stage` for details.

    :Events:

        `on_changed`:
            Triggered whenever a stage is added or removed from the factory.
    """

    stages: List['CeedStage'] = []
    '''The list of the currently available :class:`CeedStage` instances.
    These stages are listed in the GUI and can be used by name to start a stage
    to run.
    '''

    stage_names: Dict[str, 'CeedStage'] = DictProperty({})
    '''A dict of all the stages whose keys are the stage :attr:`CeedStage.name`
    and whose values are the corresponding :class:`CeedStage` instances.

    This contains the same stages as in :attr:`stages`.
    '''

    function_factory: FunctionFactoryBase = None
    """The :class:`~ceed.function.FunctionFactoryBase` instance that contains
    or is associated with all the functions used in the stages.
    """

    shape_factory: CeedPaintCanvasBehavior = None
    """The :class:`~ceed.shape.CeedPaintCanvasBehavior` instance that contains
    or is associated with all the shapes used in the stages.
    """

    _stage_ref = {}
    """A dict mapping stages to the number of references to the stage.

    References are :class:`CeedStageRef` and created with with
    :meth:`get_stage_ref` and released with :meth:`return_stage_ref`.
    """

    __events__ = ('on_changed', )

    def __init__(self, function_factory, shape_factory, **kwargs):
        self.shape_factory = shape_factory
        self.function_factory = function_factory
        super(StageFactoryBase, self).__init__(**kwargs)
        self.stages = []
        self._stage_ref = defaultdict(int)

    def on_changed(self, *largs, **kwargs):
        pass

    def get_stage_ref(
            self, name: str = None,
            stage: 'CeedStage' = None) -> 'CeedStageRef':
        """Returns a :class:`CeedStageRef` instance that refers to the
        original stage. See :mod:`ceed.stage` for details.

        One of ``name`` or ``stage`` must be specified. The stage being
        referenced by ``stage`` should have been added to this instance,
        although it is not explicitly enforced currently.

        If used, :meth:`return_stae_ref` must be called when the reference
        is not used anymore.

        :param name: The name of the stage to lookup in :attr:`stage_names`.
        :param stage: Or the actual stage to use.
        :return: A :class:`CeedStageRef` to the original stage.
        """
        stage = stage or self.stage_names[name]
        if isinstance(stage, CeedStageRef):
            stage = stage.stage

        ref = CeedStageRef(
            stage_factory=self, function_factory=self.function_factory,
            shape_factory=self.shape_factory, stage=stage)
        self._stage_ref[stage] += 1
        stage.has_ref = True
        return ref

    def return_stage_ref(self, stage_ref: 'CeedStageRef'):
        """Releases the stage ref created by :meth:`get_stage_ref`.

        :param stage_ref: Instance returned by :meth:`get_stage_ref`.
        """
        if stage_ref.stage not in self._stage_ref:
            raise ValueError("Returned stage that wasn't added")

        self._stage_ref[stage_ref.stage] -= 1
        if not self._stage_ref[stage_ref.stage]:
            del self._stage_ref[stage_ref.stage]
            stage_ref.stage.has_ref = False

    def make_stage(
            self, state: dict,
            instance: Union['CeedStage', 'CeedStageRef'] = None,
            clone=False, func_name_map: dict = {},
            old_name_to_shape_map: dict = None) -> \
            Union['CeedStage', 'CeedStageRef']:
        """Instantiates the stage from the state and returns it.

        This method must be used to instantiate a stage from state.
        See :mod:`ceed.stage` for details and an example.

        :param state: The state dict representing the stage as returned by
            :meth:`FuncBase.get_state`.
        :param instance: If None, a stage instance will be created and state
            will applied to it. Otherwise, it is applied to the given instance,
            which must be of the correct class.
        :param clone: See :meth:`CeedStage.apply_state`.
        :param func_name_map: See :meth:`CeedStage.apply_state`.
        :param old_name_to_shape_map: See :meth:`CeedStage.apply_state`.
        :return: The stage instance created or used.
        """
        state = dict(state)
        c = state.pop('cls')
        if c == 'CeedStageRef':
            cls = CeedStageRef
        else:
            assert c == 'CeedStage'
            cls = CeedStage
        assert instance is None or instance.__class__ is cls

        stage = instance
        if instance is None:
            stage = cls(
                stage_factory=self, function_factory=self.function_factory,
                shape_factory=self.shape_factory)

        stage.apply_state(state, clone=clone, func_name_map=func_name_map,
                          old_name_to_shape_map=old_name_to_shape_map)
        if c == 'CeedStageRef' and not clone:
            self._stage_ref[stage.stage] += 1
        return stage

    def add_stage(self, stage, allow_last_experiment=True):
        """Adds the :class:`CeedStage` to the stage factory (:attr:`stages`)
        and makes it available in the GUI.

        See :mod:`ceed.stage` for details.

        Remember to check :meth:`can_other_stage_be_added` before adding
        if there's potential for it to return False.

        If the :attr:`CeedStage.name` already exists in :attr:`stages`,
        :attr:`CeedStage.name` will be set to a unique name based on its
        original name. Once added until removed, anytime the stage's
        :attr:`CeedStage.name` changes, if it clashes with an existing
        stage's name, it is automatically renamed.

        :param stage: The :class:`CeedStage` to add.
        :param allow_last_experiment: Whether to allow the stage to have the
            same name is :attr:`ceed.stage.last_experiment_stage_name`. If
            False and a stage with that name is added, it is renamed.
            Otherwise, it's original name is kept.
        """
        names = [s.name for s in self.stages]
        if not allow_last_experiment:
            names.append(last_experiment_stage_name)

        stage.name = fix_name(stage.name, names)
        stage.fbind('name', self._change_stage_name, stage)

        self.stages.append(stage)
        self.stage_names[stage.name] = stage

        self.dispatch('on_changed')

    def remove_stage(self, stage, force=False):
        """Removes a stage previously added with :meth:`add_stage`.

        :Params:

            `stage`: :class:`CeedStage`
                The stage to remove.
            `force`: bool
                If True, it'll remove the stage even if there are references
                to it created by :meth:`get_stage_ref`.

        :returns: Whether the stage was removed successfully. E.g. if
            force is False and it has a ref, it won't be removed.
        """
        if not force and stage in self._stage_ref and self._stage_ref[stage]:
            assert self._stage_ref[stage] > 0
            return False

        stage.funbind('name', self._change_stage_name, stage)

        # we cannot remove by equality check (maybe?)
        for i, s in enumerate(self.stages):
            if s is stage:
                del self.stages[i]
                break
        else:
            raise ValueError('{} was not found in stages'.format(stage))
        del self.stage_names[stage.name]

        self.dispatch('on_changed')
        return True

    def clear_stages(self, force=False):
        """Removes all the stages registered with :meth:`add_stage`.

        :Params:

            `force`: bool
                If True, it'll remove all stages even if there are
                references to them created by :meth:`get_stage_ref`.
        """
        for stage in self.stages[:]:
            self.remove_stage(stage, force)

    def find_shape_in_all_stages(self, _, shape, process_shape_callback):
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
                        process_shape_callback(sub_stage, stage_shape)

    def _change_stage_name(self, stage, *largs):
        """Fixes the name of the stage instances stored to ensure it's
        unique.
        """
        # get the new name
        for name, s in self.stage_names.items():
            if s is stage:
                if stage.name == name:
                    return

                del self.stage_names[name]
                # only one change at a time happens because of binding
                break
        else:
            raise ValueError(
                '{} has not been added to the stage'.format(stage))

        new_name = fix_name(
            stage.name,
            list(self.stage_names.keys()) + [last_experiment_stage_name])
        self.stage_names[new_name] = stage
        stage.name = new_name

        if not new_name:
            stage.name = fix_name(
                'name',
                list(self.stage_names.keys()) + [last_experiment_stage_name])
        self.dispatch('on_changed')

    def save_stages(self) -> List[dict]:
        """Returns a dict representation of all the stages added with
        :meth:`add_stage`.

        It is a list of dicts where each item is the
        :meth:`CeedStage.get_state` of the corresponding stage in
        :attr:`stages`.
        """
        return [s.get_state(expand_ref=False)for s in self.stages]

    def recover_stages(
            self, stage_states: List[dict], func_name_map: dict,
            old_name_to_shape_map: dict) -> \
            Tuple[List['CeedStage'], Dict[str, str]]:
        """Takes a list of stages states such as returned by
        :meth:`save_stages` and instantiates the stages represented by
        the states and adds (:meth:`add_stage`) the stages to the factory.

        :param stage_states: List of stages state.
        :param func_name_map: See :meth:`CeedStage.apply_state`.
        :param old_name_to_shape_map: See :meth:`CeedStage.apply_state`.
        :return: A tuple ``stages, name_map``, ``stages`` is the list of
            stages, ``name_map`` is a map from the original stage's name
            to the new name given to the stage (in case a stage with that
            name already existed).
        """
        name_map = {}
        stages = []
        for state in stage_states:
            # cannot be a ref func here because they are global funcs
            c = state['cls']
            assert c != 'CeedStageRef'
            assert c == 'CeedStage'

            stage = CeedStage(
                stage_factory=self, function_factory=self.function_factory,
                shape_factory=self.shape_factory)
            old_name = stage.name = state['name']

            self.add_stage(stage)
            stages.append(stage)
            state['name'] = name_map[old_name] = stage.name

        update_key_if_other_key(
            stage_states, 'cls', 'CeedStageRef', 'ref_name', name_map)

        for stage, state in zip(stages, stage_states):
            self.make_stage(
                state, instance=stage, clone=True, func_name_map=func_name_map,
                old_name_to_shape_map=old_name_to_shape_map)

        return stages, name_map

    def tick_stage(self, stage_name='', stage=None):
        '''An iterator which starts a :class:`CeedStage` and ticks the time for
        every call.

        A :class:`CeedStage` represents a collection of shapes with functions
        applied to them. Each of these shapes has a defined intensity for
        every time point. This iterator walks through time denoting the
        intensity for each shape for every time point.

        E.g. to get the shape values for time 0, .1, .2, ..., 1.0 for the
        stage named ``'my_stage'``::

            >>> tick = stage_factory.tick_stage('my_stage')  # get iterator
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
        the function using `send()` and in return the iterator will yield
        the associated intensity values for each shape for that time point.

        :Params:

            `stage_name`: str
                The :attr:`CeedStage.name` of the stage to start.
            `stage`: str
                The :attr:`CeedStage` to start.

        :yields:

            A list of the intensity values for each shape.

            Each item in the list is a 2-tuple of ``(name, values)``. ``name``
            is the :attr:`kivy_garden.painter.PaintShape.name` of the shape
            and is listed only once in the list.
            ``values`` is a list of color values and each item in that list is
            a 4-tuple of ``(r, g, b, a)``. Any of these values can be None, in
            which case that color remains the same. This way a shape can be
            updated from multiple sub-stages, where only e.g. the ``r`` value
            is changed.

        :raises:

            `StageDoneException`:
                When done with the stage (time is out of bounds).
        '''
        if stage is None:
            stage = self.stage_names[stage_name]
        shapes = {s.name: [] for s in self.shape_factory.shapes}
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

            a dict whose keys is the
            :attr:`kivy_garden.painter.PaintShape.name` of
            the :class:`ceed.shape.CeedShape` and whose value is the Kivy
            ``Color`` instruction instance that controls the color of the
            shape.
        '''
        shape_views = {}
        for shape in self.shape_factory.shapes:
            color = shape_views[shape.name] = Color(0, 0, 0, 1, group=name)
            canvas.add(color)
            shape.add_area_graphics_to_canvas(name, canvas)

        return shape_views

    def fill_shape_gl_color_values(
            self, shape_views, shape_values, grayscale=None):
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
            `grayscale`: str
                The colors to operate on. Can be any subset of the string
                'rgb'. Specifically, although we get intensity values for every
                color, this takes the average intensity of the colors listed in
                ``grayscale`` and assigns their mean to all those colors.
                It's how we turn the color into a gray-scale value.

        :returns:

            A list of the colors each shape was set to. Each item in the list
            is ``(name, r, g, b, a)``, where ``name`` is the shape's name and
            ``r``, ``g``, ``b``, ``a`` is the color value it was set to.
            Each ``name`` occurs at most once in the list.



        E.g. to display the shape intensities for time 0, .1, .2, ..., 1.0 for
        the stage named ``'my_stage'`` in real time::

            >>> import time
            >>> tick = stage_factory.tick_stage('my_stage')  # get iterator
            >>> # get the color objects
            >>> colors = stage_factory.get_shapes_gl_color_instructions(
            >>>     widget_canvas, 'my_canvas_instructs')
            >>> for t in [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]:
            >>>     try:
            >>>         tick.next()
            >>>         shape_values = tick.send(t)  # current color values
            >>>         # update the colors and print it
            >>>         values = stage_factory.fill_shape_gl_color_values(
            >>>>            colors, shape_values)
            >>>         print(values)
            >>>         time.sleep(0.1)  # wait until next time
            >>>     except StageDoneException:
            >>>         # if we're here the stage has completed
            >>>         break
            >>> # now remove the shapes
            >>> stage_factory.remove_shapes_gl_color_instructions(
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
            elif grayscale:
                if a is None:
                    a = 1

                vals = [v for v in (r, g, b) if v is not None]
                if not vals:
                    val = 0
                else:
                    val = sum(vals) / float(len(vals))

                if color is not None:
                    color.a = a
                    setattr(color, grayscale, val)
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

    name: str = StringProperty('Stage')
    '''The name of this stage.
    '''

    order: str = OptionProperty('serial', options=['serial', 'parallel'])
    '''The order in which the sub-stages, :attr:`stages`, are evaluated.
    Can be one of ``'serial'``, ``'parallel'``.

    See :class:`CeedStage` description for details.
    '''

    complete_on: str = OptionProperty('all', options=['all', 'any'])
    '''When to consider the stage's children stages to be complete we contain
    sub-stages - :attr:`stages`. Can be one of ``'all'``, ``'any'``.

    If ``'any'``, this stage is done when **any** of the children stages is
    done, and when all of this stage's functions are done. If ``'all'``, this
    stage is done when **any** children stages are done, and when all of this
    stage's functions are done.

    See :class:`CeedStage` description for details.
    '''

    stages: List['CeedStage'] = []
    '''A list of :class:`CeedStage` instances that are sub-stages of this
    stage.

    See :class:`CeedStage` description for details.
    '''

    parent_stage: 'CeedStage' = ObjectProperty(
        None, rebind=True, allownone=True)
    '''The parent stage when this stage is a sub-stage of another.
    '''

    has_ref: bool = BooleanProperty(False)
    """Whether there's a CeedFuncRef pointing to this function.
    """

    functions: List[Union[CeedFuncRef, FuncBase]] = []
    '''A list of :class:`ceed.function.FuncBase` instances through which the
    stage iterates through sequentially and updates the intensity of the
    :attr:`shapes` to the function value at each time point.
    '''

    shapes: List['StageShape'] = []
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

    stage_factory: StageFactoryBase = None

    function_factory: FunctionFactoryBase = None

    shape_factory: CeedPaintCanvasBehavior = None

    display = None

    pad_stage_ticks = 0

    __events__ = ('on_changed', )

    def __init__(self, stage_factory, function_factory, shape_factory,
                 **kwargs):
        self.stage_factory = stage_factory
        self.function_factory = function_factory
        self.shape_factory = shape_factory
        super(CeedStage, self).__init__(**kwargs)
        self.functions = []
        self.stages = []
        self.shapes = []

        for prop in self.get_state():
            if prop in ('stages', 'functions', 'shapes'):
                continue
            self.fbind(prop, self.dispatch, 'on_changed', prop)

    def on_changed(self, *largs, **kwargs):
        pass

    def get_state(self, expand_ref=False):
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
        d = {'cls': 'CeedStage'}
        for name in ('order', 'name', 'color_r', 'color_g', 'color_b',
                     'complete_on'):
            d[name] = getattr(self, name)

        d['stages'] = [s.get_state(expand_ref=expand_ref) for s in self.stages]
        d['functions'] = [
            f.get_state(recurse=True, expand_ref=expand_ref)
            for f in self.functions]
        d['shapes'] = [s.get_config() for s in self.shapes]

        return d

    def apply_state(self, state={}, clone=False, func_name_map={},
                    old_name_to_shape_map=None):
        """Takes the state of the stage saved with :meth:`get_state` and
        applies it to this stage. it also creates any children functions and
        stages and creates the references to the :attr:`shapes`.

        It is called internally and should not be used directly. Use
        :meth:`StageFactoryBase.make_stage` instead.

        :param state: The state dict representing the stage as returned by
            :meth:`get_state`.
        :param clone: If False, only user customizable properties of the
            stage will be set, otherwise, all properties from state are
            applied to the stage. Clone is meant to be an complete
            re-instantiation of stage function.
        :param func_name_map:
        :param old_name_to_shape_map:
        """
        stages = state.pop('stages', [])
        functions = state.pop('functions', [])
        shapes_state = state.pop('shapes', [])

        for k, v in state.items():
            setattr(self, k, v)

        for data in stages:
            s = self.stage_factory.make_stage(
                data, clone=clone, func_name_map=func_name_map,
                old_name_to_shape_map=old_name_to_shape_map)
            self.add_stage(s)

        update_key_if_other_key(
            functions, 'cls', 'CeedFuncRef', 'ref_name', func_name_map)
        for data in functions:
            func = self.function_factory.make_func(data, clone=clone)
            self.add_func(func)

        shapes = self.stage_factory.shape_factory.shape_names
        groups = self.stage_factory.shape_factory.shape_group_names
        for item in shapes_state:
            if not isinstance(item, dict):
                item = {'name': item}

            name = item['name']
            if old_name_to_shape_map is None:
                if name in shapes:
                    shape = shapes[name]
                elif name in groups:
                    shape = groups[name]
                else:
                    raise ValueError('Could not find shape {}'.format(name))
            else:
                shape = old_name_to_shape_map.get(name, None)
                if shape is None:
                    raise ValueError('Could not find shape {}'.format(name))

            stage_shape = self.add_shape(shape)
            stage_shape.apply_config(**item)

    def __deepcopy__(self, memo):
        obj = self.__class__(
            stage_factory=self.stage_factory,
            function_factory=self.function_factory,
            shape_factory=self.shape_factory)
        obj.apply_state(self.get_state())
        return obj

    def copy_expand_ref(self):
        obj = self.__class__(
            stage_factory=self.stage_factory,
            function_factory=self.function_factory,
            shape_factory=self.shape_factory)
        obj.apply_state(self.get_state(expand_ref=True))
        return obj

    def replace_ref_stage_with_source(self, stage_ref):
        if not isinstance(stage_ref, CeedStageRef):
            raise ValueError('Stage must be a CeedStageRef')

        i = self.stages.index(stage_ref)
        self.remove_stage(stage_ref)
        stage = deepcopy(stage_ref.stage)
        self.add_stage(stage, index=i)
        return stage, i

    def replace_ref_func_with_source(self, func_ref):
        if not isinstance(func_ref, CeedFuncRef):
            raise ValueError('Function must be a CeedFuncRef')

        i = self.functions.index(func_ref)
        self.remove_func(func_ref)
        func = deepcopy(func_ref.func)
        self.add_func(func, index=i)
        return func, i

    def can_other_stage_be_added(self, other_stage):
        '''Checks whether the other stagetion may be added to us.
        '''
        if isinstance(other_stage, CeedStageRef):
            other_stage = other_stage.stage

        # check if we (or a ref to us) are a child of other_stage
        for stage in other_stage.get_stages(step_into_ref=True):
            if stage is self:
                return False
        return True

    def add_stage(self, stage, after=None, index=None):
        '''Adds a sub-stage instance :class:`CeedStage` to :attr:`stages`.
        '''
        stage.parent_stage = self

        if after is None and index is None:
            self.stages.append(stage)
        elif index is not None:
            self.stages.insert(index, stage)
        else:
            i = self.stages.index(after)
            self.stages.insert(i + 1, stage)

        self.dispatch('on_changed')

    def remove_stage(self, stage):
        '''Removes a sub-stage instance :class:`CeedStage` from :attr:`stages`.
        '''
        self.stages.remove(stage)  # use is not equal, same for funcs
        assert stage.parent_stage is self
        stage.parent_stage = None

        self.dispatch('on_changed')
        return True

    def add_func(self, func, after=None, index=None):
        '''Adds the function instance :class:`ceed.function.FuncBase` to
        :attr:`functions`.

        :params:

            `func`: :class:`ceed.function.FuncBase`
                The function to add.
            `after`: :class:`ceed.function.FuncBase`
                The function in :attr:`functions` after which to add this
                function, if not None, the default.
        '''

        if after is None and index is None:
            self.functions.append(func)
        elif index is not None:
            self.functions.insert(index, func)
        else:
            i = self.functions.index(after)
            self.functions.insert(i + 1, func)

        self.dispatch('on_changed')

    def remove_func(self, func):
        '''Removes the function instance :class:`ceed.function.FuncBase` from
        :attr:`functions`.
        '''
        self.functions.remove(func)
        self.dispatch('on_changed')
        return True

    def add_shape(
            self, shape: Union[CeedShapeGroup, CeedShape]) -> 'StageShape':
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

        self.dispatch('on_changed')
        return stage_shape

    def remove_shape(self, stage_shape):
        '''Removes a :class:`StageShape` that was previously added to
        :attr:`shapes`.
        '''
        self.shapes.remove(stage_shape)
        self.dispatch('on_changed')

    def get_stages(self, step_into_ref=True):
        '''Iterator that iterates depth-first through all the stages
        and children :attr:`stages and yields these stages.`
        '''
        yield self
        for stage in self.stages:
            if isinstance(stage, CeedStageRef):
                if not step_into_ref:
                    yield stage
                    continue

                stage = stage.stage
            for s in stage.get_stages(step_into_ref):
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
            >>> shapes = {s.name: [] for s in shape_factory.shapes}
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
        keep_dark = set()
        for shape in self.shapes:
            if shape.keep_dark:
                keep_dark.add(shape.shape.name)

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
        pad_stage_ticks = self.pad_stage_ticks
        count = 0
        for tick_stage in stages[:]:
            next(tick_stage)

        while funcs is not None or stages:
            count += 1
            t = yield
            if funcs is not None:
                try:
                    next(funcs)
                    val = funcs.send(t)
                    values = (val if r else None, val if g else None,
                              val if b else None, a)
                    for name in names:
                        if name in keep_dark:
                            shapes[name].append((0, 0, 0, 1))
                        else:
                            shapes[name].append(values)
                except FuncDoneException:
                    funcs = None

            for tick_stage in stages[:]:
                try:
                    tick_stage.send(t)
                    if serial:
                        break
                except StageDoneException:
                    if end_on_first:
                        del stages[:]
                        break
                    stages.remove(tick_stage)

        while count <= pad_stage_ticks:
            count += 1
            _ = yield
            for name in names:
                if name in keep_dark:
                    shapes[name].append((0, 0, 0, 1))

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
            # this function should have been copied when it was created for
            # this experiment in the `last_experiment_stage_name` stage
            assert not isinstance(func, CeedFuncRef)
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

    def resample_func_parameters(self):
        for root_func in self.functions:
            for func in root_func.get_funcs():
                func.resample_parameters()

    def get_stage_shape_names(self):
        """Gets all the names of the shapes controlled by the stage or
        sub-stages.
        """
        shapes = set()
        for stage in self.get_stages(step_into_ref=True):
            for shape in stage.shapes:
                if isinstance(shape.shape, CeedShapeGroup):
                    for s in shape.shape.shapes:
                        shapes.add(s.name)
                else:
                    shapes.add(shape.shape.name)
        return shapes


class CeedStageRef(object):
    """The function it refers to must be in the factory.
    """

    stage = None

    display = None

    parent_stage = None

    stage_factory = None

    function_factory = None

    shape_factory = None

    def __init__(self, stage_factory, function_factory, shape_factory,
                 stage=None):
        super(CeedStageRef, self).__init__()
        self.stage = stage
        self.stage_factory = stage_factory
        self.shape_factory = shape_factory
        self.function_factory = function_factory

    def get_state(self, expand_ref=False):
        if expand_ref:
            return self.stage.get_state(expand_ref=True)

        state = {}
        state['ref_name'] = self.stage.name
        state['cls'] = 'CeedStageRef'
        return state

    def apply_state(
            self, state={}, clone=False, func_name_map={},
            old_name_to_shape_map=None):
        self.stage = self.stage_factory.stage_names[state['ref_name']]

    def __deepcopy__(self, memo):
        assert self.__class__ is CeedStageRef
        return self.stage_factory.get_stage_ref(stage=self)

    def copy_expand_ref(self):
        return self.stage.copy_expand_ref()


class StageShape(EventDispatcher):
    '''A wrapper for :class:`ceed.shape.CeedShape` instances used in
    :meth:`CeedStage.add_shape` to wrap a shape or shape group to be used by
    the stage.
    '''

    shape: Union[CeedShape, CeedShapeGroup] = None
    '''The :class:`ceed.shape.CeedShape` or :class:`ceed.shape.CeedShapeGroup`
    instance being wrapped.
    '''

    stage: CeedStage = None
    '''The :class:`CeedStage` this is associated with.
    '''

    name = StringProperty('')
    '''The :attr:`ceed.shape.CeedShapeGroup.name` or
    :attr:`kivy_garden.painter.PaintShape.name` of the instance wrapped.
    '''

    keep_dark = BooleanProperty(False)
    """Whether this shape will be black during the whole stage. Instead of it
    taking the color of the stage, it'll be kept black.

    This is useful when the inside of some shape must be black, e.g. a donut.
    By setting :attr:`keep_dark` of the inner shape to True, it'll be black.
    """

    display = None

    def __init__(self, stage=None, shape=None, **kwargs):
        super(StageShape, self).__init__(**kwargs)
        self.stage = stage
        self.shape = shape
        self.shape.fbind('name', self._update_name)
        self._update_name()

    def _update_name(self, *largs):
        self.name = self.shape.name

    def get_config(self) -> Dict:
        """(internal) used by the config system to get the config data of the
        shape.
        """
        return {'keep_dark': self.keep_dark, 'name': self.name}

    def apply_config(self, **kwargs):
        """(internal) used by the config system to set the config data
        of the shape.
        """
        if 'keep_dark' in kwargs:
            self.keep_dark = kwargs['keep_dark']


def remove_shapes_upon_deletion(
        stage_factory: StageFactoryBase,
        shape_factory: CeedPaintCanvasBehavior, process_shape_callback):
    """Once called, whenever a shape or group of shapes is deleted in the
    ``shape_factory``, it'll also remove the shape or group from all stages
    that reference it.

    :param stage_factory: The :class:`StageFactoryBase` that lists all the
        stages.
    :param shape_factory: The :class:`~ceed.shape.CeedPaintCanvasBehavior` that
        lists all the shapes.
    :param process_shape_callback: For each stage in ``stage_factory`` that
        contains the shape, ``process_shape_callback`` will be called with 2
        parameters: the :class:`CeedStage` and :class:`StageShape` instances.
        The callback may e.g. then hide the shape in the GUI or whatever else
        it needs.
    """
    shape_factory.fbind(
        'on_remove_shape', stage_factory.find_shape_in_all_stages,
        process_shape_callback=process_shape_callback)
    shape_factory.fbind(
        'on_remove_group', stage_factory.find_shape_in_all_stages,
        process_shape_callback=process_shape_callback)
