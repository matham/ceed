"""
.. _stage-api:

Stage
=====

A :class:`CeedStage` combines :mod:`ceed.shapes` on the screen with a
:class:`~ceed.function.FuncBase` or sequence of
:class:`~ceed.function.FuncBase` which determines the intensity of the shapes as
time progresses in the experiment. This module defines the :class:`CeedStage`
and associated classes used to store and compute the intensity values during
the experiment.

.. _stage-factory-plugin:

Stage factory and plugins
-------------------------

The :class:`StageFactoryBase` is a store of the defined :class:`CeedStage`
sub-classes and customized stage instances. Stage classes/instances
registered with the :class:`StageFactoryBase` instance used by the
the GUI are available to the user in the GUI. During analysis, stages are
similarly registered with the :class:`StageFactoryBase` instance used in the
analysis and can then be used to get these stages. E.g.::

    >>> # get a function and shape factory
    >>> function_factory = FunctionFactoryBase(...)
    >>> shape_factory = CeedPaintCanvasBehavior(...)
    >>> # create the stage store, linking to the other factories
    >>> stage_factory = StageFactoryBase(
            function_factory=function_factory, shape_factory=shape_factory)
    >>> register_all_stages(stage_factory)  # register plugins
    >>> StageCls = stage_factory.get('CeedStage')  # get the class
    >>> StageCls
    <class 'ceed.stage.CeedStage'>

Classes can be registered manually with
:meth:`StageFactoryBase.register` or they can be registered automatically
with :func:`register_all_stages` if they are an internal plugin or
:func:`register_external_stages` for an external plugin. The GUI calls
:func:`register_all_stages` when started as well as
:func:`register_external_stages` if the
:attr:`~ceed.main.CeedApp.external_stage_plugin_package` configuration
variable contains a package name.

See :mod:`ceed.stage.plugin` for details on writing plugins. For details on
how to customize stages, see below for important steps and methods during a
stage's lifecycle.

To get a stage class registered with :class:`StageFactoryBase`, e.g. the
connonical :class:`CeedStage` base class::

    >>> StageCls = stage_factory.get('CeedStage')  # get the class
    >>> StageCls
    <class 'ceed.stage.CeedStage'>

Stage basics
------------

All stages are instances of :class:`CeedStage` or any plugin-defined
sub-classes.

A stage is composed of one or more :attr:`CeedStage.shapes`, a
series of :class:`ceed.function.FuncBase` :attr:`functions` that
govern the intensity these shapes take across time, and
sub-:attr:`CeedStage.stages`, that are simultanously evaluated during this
stage.

All stages have a :attr:`CeedStage.name`, and stages that are available globally
through the :class:`StageFactoryBase` and shown as root stages in the GUI have
unique names. To run an experiment you select a named stage to run.

Stage duration
^^^^^^^^^^^^^^

Ignoring sub-stages, a stage will sample trhough its :attr:`CeedStage.functions`
sequentally until they are all done, :attr:`CeedStage.loop` times. After
:attr:`CeedStage.loop` times, the stage is done. For example, a stage with
function ``f1`` and ``f2`` that :attr:`CeedStage.loop`s 3 times, will tick
through the functions as follows: ``f1, f2, f1, f2, f1, f2``. At each sample,
all the :attr:`CeedStage.shapes` of the stage (and possibly sub-stages) are
set to the value  in the ``[0, 1]`` range returned by the functions for that
time step.

If there are child :attr:`CeedStage.stages` that have their own
:attr:`CeedStage.shapes`, we also sample these stages simultanously during each
root stage loop iteration. That means that while shapes associated with the root
stage are updated from the root stage's functions, the shapes of the sub-stages
currently active are updated from their functions.

E.g. if we have a root stage which contains 4 children :attr:`CeedStage.stages`
A, B, C, D::

    root
        A
        B
        C
        D

If root's :attr:`CeedStage.order` is ``'serial'``, then for each root
:attr:`CeedStage.loop` iteration, each sub-stage A - D is evaluated sequentially
after the previous sub-stage has finished in the order A -> B -> C -> D.
If it's ``'parallel'``, then each sub-stage is evaluated simultaneously
rather than sequentially.

While the sub-stages are executed in parallel or serially, the root's
:attr:`CeedStage.functions` are evaluated and the root's shapes set to those
values.

If :attr:`CeedStage.complete_on` is ``'any'`` then a loop iteration for the
root stage will be considered completed after all the root's
:attr:`CeedStage.functions` are done **and** any of the
:attr:`CeedStage.stages` have completed. The sub-stages that have
not yet completed will just end early.

Otherwise, :attr:`CeedStage.complete_on` is ``'all'`` and a loop iteration for
the root will complete after all the root's :attr:`CeedStage.functions` **and**
all the :attr:`CeedStage.stages` have completed.

In all cases, shapes that are not associated with a stage that is
currently being evaluated will be set to transparent. E.g. in the serial
case, shapes that are associated with stage C will be transparent in all
other stages, except if they also appear in those stages.

If a shape occurs in multiple stages (e.g. in both the root and A), then
if the root and A set different color channels (e.g. root sets blue, A sets
green), the shape will will set each channel from each active stage
simultaneously because they don't conflict. If they do set the same channel,
the sub-stages win over parent stages in the stage tree.

Color channels
^^^^^^^^^^^^^^

Each stage contains :attr:`CeedStage.shapes` that are all set to the value
returned by the currently active function in :attr:`CeedStage.functions`
for the current time step. That is the function yields a single floating point
value between 0-1, inclusive and all the shapes get set to that value for that
timestep.

However, there are 3 channels to be set; red, green, and blue. You can select
which of these channels the stage sets using :attr:`CeedStage.color_r`,
:attr:`CeedStage.color_g`, and :attr:`CeedStage.color_b` and those channels are
set to the same function value. The remaining channels are not set by this stage
and default to zero if no other parent or sub-stage sets them.

Shapes belonging to sub-stages are not controlled by parent stages, only by the
direct stage(s) that contains them.

If the projector :attr:`~ceed.view.controller.ViewControllerBase.video_mode` is
set to ``"QUAD12X"``, then the function value is used for all three channels
so it's grayscale. E.g. even if only :attr:`CeedStage.color_r` and
:attr:`CeedStage.color_b` are true, all three red, green, and blue channels
will be set to the same value . If the channels are set to different values
in parallel stages, the channels are first averaged.

Stage lifecycle
^^^^^^^^^^^^^^^

The ultimate use of a stage is to sample it for all timepoints that the stage is
active. Before the stage is ready, however, it needs to be initialized. Folowing
are the overall steps performed by :meth:`StageFactoryBase.tick_stage` that
initialize a stage before running it as an experiment.

Starting with a stage, first the stage is coped and all
:attr:`CeedStage.functions` resampled using
:meth:`CeedStage.copy_and_resample`. Then the stage is renamed to
:attr:`last_experiment_stage_name`. This displays the copied stage to be run in
the GUI under the name :attr:`last_experiment_stage_name`.

Next, we call :meth:`CeedStage.init_stage_tree` on the root stage. This
calls :meth:`~ceed.function.FuncBase.init_func_tree` on all the stage's
:attr:`CeedStage.functions` and recursively calls
:meth:`CeedStage.init_stage_tree` on all the sub-:attr:`CeedStage.stages`
as well as some initialization.

Next, using :meth:`CeedStage.apply_pre_compute` it pre-computes all the stages
and functions for which it is enabled. See below for details. Finally, the
stage is sampled, a sample at a time using the :meth:`CeedStage.tick_stage`
generator. This generator either returns the pre-comuted values, computed
before the experiment started if enabled, or
it computes the sample values and yields them until the root stage is done.

When the stage is done, it internally raises a :class:`StageDoneException`.
This gets caught and Ceed knows that the stage is done. At the end of each loop
iteration and at the end of the overall stage it calls
:meth:`CeedStage.finalize_loop_iteration`, and
:meth:`CeedStage.finalize_stage`, respectively.

Customizing stages
^^^^^^^^^^^^^^^^^^

A :class:`CeedStage` has multiple methods that can be overwritten by a plugin.
Following are some relevant methods - see their docs for more details.

* If the stage generates the samples directly without using the function
  classes, :meth:`CeedStage.resample_parameters` needs to be augmented if
  the stage has any randomness.
* :meth:`CeedStage.init_stage_tree`, :meth:`FuncBase.init_func`,
  :meth:`CeedStage.init_loop_iteration`,
  :meth:`CeedStage.finalize_loop_iteration`, and
  :meth:`CeedStage.finalize_stage` may be augmented if any of these stage
  lifecycle events requires additional initialization/finalization.
* :meth:`CeedStage.evaluate_stage` is the most approperiate method to overwrite
  to have the stage directly compute values for the shapes. By default it
  goes through all the loops and for each loop it goes through all the functions
  and sub-stages. It can be overwritten instead to just yield any specific
  desired values.
* :meth:`CeedStage.tick_stage` can be overwritten, but it requires more care.
  Since this yields the values directly and is called directly by the higher
  level code, if :meth:`CeedStage.tick_stage` is overwritten it won't
  participate in pre-computing or any other stage behaviors, so pre-computing
  should be disabled by setting :attr:`CeedStage.disable_pre_compute` to True.
  See the method for other requirements.

  Additionally, if it's a root stage, it should pad the stage runtime to
  :attr:`CeedStage.pad_stage_ticks`, if nonzero.
* :meth:`CeedStage.apply_pre_compute`, :attr:`CeedStage.runtime_functions`, and
  :attr:`CeedStage.runtime_stage` can be overwritten/manually set, but great
  care must be taken. See their docs and the pre-computing section below.
* See the stage's properties for additional customizations.

In all cases, :meth:`CeedStage.get_state` may also need to
be augmented to return any parameters that are part of the instance, otherwise
they won't be copied when the stage is internally copied and the stage will use
incorrect values when run e.g. in the second process that runs the experiment.

If stages are pre-computed, it may not have any side-effects, because they
would occur before the experiment during pre-computing. So if these side-effects
are needed (e.g. drug delivery), either turn off pre-computing for the stage
or set its function's duration to negative to disable it for the function and
consequently the stage containing it.

Running a stage
---------------

Once we have a stage, shapes, and a function, the stage is ready to be run with
:meth:`StageFactoryBase.tick_stage`

The following is a worked through example showing
the steps the Ceed GUI goes through automatically to run a stage and what you
should do to manually run a stage for testing purposes.

First create the stage, shape, and function:

.. code-block:: python

    from ceed.stage import CeedStage, StageFactoryBase, register_all_stages, \\
        StageDoneException
    from ceed.function import FunctionFactoryBase, register_all_functions
    from ceed.shape import CeedPaintCanvasBehavior
    from fractions import Fraction

    # create function/shape/stage factories to house functions/shapes/stages
    function_factory = FunctionFactoryBase()
    # register built-in plugins
    register_all_functions(function_factory)
    shape_factory = CeedPaintCanvasBehavior()
    stage_factory = StageFactoryBase(
        function_factory=function_factory, shape_factory=shape_factory)
    # register built-in plugins
    register_all_stages(stage_factory)

    # create a 3 second duration function, that loops twice
    LinearFuncCls = function_factory.get('LinearFunc')
    function = LinearFuncCls(
        function_factory=function_factory, m=.33, b=0, duration=3, loop=2)
    print(f'Function name: "{function.name}"')
    # and a circle shape
    circle = shape_factory.create_shape('circle', center=(100, 100))
    # and add it
    shape_factory.add_shape(circle)
    print(f'Shape name: "{circle.name}"')
    # and finally the stage
    stage = CeedStage(
        stage_factory=stage_factory, function_factory=function_factory,
        shape_factory=shape_factory)
    # and add it to factory
    stage_factory.add_stage(stage)
    print(f'Stage name: "{stage.name}"')

    # now add the function and shape to stage
    stage.add_func(function)
    stage.add_shape(circle)
    # stage will only set red and blue channels
    stage.color_r = stage.color_b = True
    stage.color_g = False

Once ready, we can run through the stage manually like Ceed does during an
experiment:

.. code-block:: python

    # now create the generator that will iterate through all the stage shape
    # values at a frame rate of 2Hz
    tick = stage_factory.tick_stage(
        t_start=0, frame_rate=Fraction(2, 1), stage_name=stage.name)
    # start it
    next(tick)
    # start with time zero, the same as t_start
    i = 0

    while True:
        # send the next global time value as multiples of the period (1 / 2)
        t = Fraction(i, 2)
        try:
            # this gets the raw intensity values
            shapes_intensity_raw = tick.send(t)
        except StageDoneException:
            # when the stage is done, it raises this exception
            break

        # convert raw intensity to final color intensity. This function can
        # also e.g. convert the colors for quad mode where it's gray-scale
        shapes_intensity = stage_factory.set_shape_gl_color_values(
            shape_views=None, shape_values=shapes_intensity_raw)
        print(f'time={t}s,\tintensity="{shapes_intensity}"')

        i += 1

The above will print the following when run::

    Function name: "Linear"
    Shape name: "Shape"
    Stage name: "Stage-2"
    time=0s,	intensity="[('Shape', 0, 0, 0, 1)]"
    time=1/2s,	intensity="[('Shape', 0.165, 0, 0.165, 1)]"
    time=1s,	intensity="[('Shape', 0.33, 0, 0.33, 1)]"
    time=3/2s,	intensity="[('Shape', 0.495, 0, 0.495, 1)]"
    time=2s,	intensity="[('Shape', 0.66, 0, 0.66, 1)]"
    time=5/2s,	intensity="[('Shape', 0.825, 0, 0.825, 1)]"
    time=3s,	intensity="[('Shape', 0, 0, 0, 1)]"
    time=7/2s,	intensity="[('Shape', 0.165, 0, 0.165, 1)]"
    time=4s,	intensity="[('Shape', 0.33, 0, 0.33, 1)]"
    time=9/2s,	intensity="[('Shape', 0.495, 0, 0.495, 1)]"
    time=5s,	intensity="[('Shape', 0.66, 0, 0.66, 1)]"
    time=11/2s,	intensity="[('Shape', 0.825, 0, 0.825, 1)]"

Some important points to notice above: The global clock is run with multiples
of the period. This period is the projector frame rate, and when ticking we must
only increment the time by single period increments. This is required for
stage pre-computing to work because once pre-computed, we have a list of
intensity values with the expectation that each value corresponds to single
period increment because that's how they are pre-computed.

To skip a frame, you must still tick that frame time, but you can discard the
yielded value. This is how Ceed drops frames when the GPU takes too long to
render a frame and we must compensate by dropping a frame.

One can also use the stage factory to compute all the intensity values without
having to iterate as follows. Once we have the stage ready:

.. code-block:: python

    from pprint import pprint
    # now create the generator that will iterate through all the stage shape
    # values at a frame rate of 2Hz
    intensity = stage_factory.get_all_shape_values(
        frame_rate=Fraction(2, 1), stage_name=stage.name)
    pprint(intensity)

This prints::

    defaultdict(<class 'list'>,
                {'Shape': [(0, 0, 0, 1),
                           (0.165, 0, 0.165, 1),
                           (0.33, 0, 0.33, 1),
                           (0.495, 0, 0.495, 1),
                           (0.66, 0, 0.66, 1),
                           (0.825, 0, 0.825, 1),
                           (0, 0, 0, 1),
                           (0.165, 0, 0.165, 1),
                           (0.33, 0, 0.33, 1),
                           (0.495, 0, 0.495, 1),
                           (0.66, 0, 0.66, 1),
                           (0.825, 0, 0.825, 1)]})

.. _pre-compute:

Pre-computing
-------------

By default, during an experiment as the global clock ticks in multiple of the
period, the root stage is given the current time and it compute the intensity
values for all the shapes from its :attr:`CeedStage.functions` and
sub-:attr:`CeedStage.stages`. If the intensity computation is slow, the CPU
may miss updating the GPU with a new frame before the next frame deadline
and consequently we will need to drop a frame.

Ceed can pre-compute the intensity values for all the shapes for all time
points by virtually running through the whole experiment and recording the
intensities yielded by the stage into a flat list. Then during the real
experiment it simply looks up the intensity from the list by sequentially
iterating through the list.

This works because the global time is sequential consecutive multiples of the
period both during the virtual computation and during the replay, so we can
simply count frames to locate the desired intensity.

Pre-computing can be enabled in the GUI through the
:attr:`~ceed.view.controller.ViewControllerBase.pre_compute_stages` property.

Not all stages can be pre-computed. A stage could have functions that are
infinite in duration (i.e. negarive :attr:`~ceed.function.FuncBase.duration`,
e.g. when it's waiting for a switch to go ON) or a stage can be manually
opted out from pre-computing by setting :attr:`CeedStage.disable_pre_compute`
to True. In that case, all other functions and stages that are not infinite
and not opted would will still be pre-computed as much as possible.

See :attr:`CeedStage.disable_pre_compute`, :attr:`CeedStage.runtime_functions`,
:attr:`CeedStage.runtime_stage`, and :attr:`CeedStage.can_pre_compute` for
more details.

Saving and restoring stages
---------------------------

Functions can be saved as a data dictionary and later restored into a function
instance. E.g.::

    >>> function = LinearFunc(
    ...     function_factory=function_factory, m=.33, b=0, duration=3, loop=2)
    >>> circle = shape_factory.create_shape('circle', center=(100, 100))
    >>> shape_factory.add_shape(circle)
    >>> stage = CeedStage(
    ...     stage_factory=stage_factory, function_factory=function_factory,
    ...     shape_factory=shape_factory)
    >>> stage.add_func(function)
    >>> stage.add_shape(circle)
    >>> stage_factory.add_stage(stage)
    >>> state = stage.get_state()
    >>> state
    {'cls': 'CeedStage',
     'color_b': True,
     'color_g': False,
     'color_r': False,
     'complete_on': 'all',
     'disable_pre_compute': False,
     'functions': [{'b': 0.0,
                    'cls': 'LinearFunc',
                    'duration': 3,
                    'loop': 2,
                    'm': 0.33,
                    'name': 'Linear',
                    'noisy_parameter_samples': {},
                    'noisy_parameters': {},
                    't_offset': 0.0,
                    'timebase_denominator': 1,
                    'timebase_numerator': 0}],
     'name': 'Stage-2',
     'order': 'serial',
     'shapes': [{'keep_dark': False, 'name': 'Shape'}],
     'stages': []}
    >>> # this is how we create a stage from state
    >>> new_stage = stage_factory.make_stage(state)
    >>> new_stage
    <ceed.stage.CeedStage: "Stage" children=(1, 0), at 0x22c2f3b7898>
    >>> new_stage.get_state()
    {'cls': 'CeedStage',
     'color_b': True,
     'color_g': False,
     'color_r': False,
     ...
     'name': 'Stage',
     'order': 'serial',
     'shapes': [{'keep_dark': False, 'name': 'Shape'}],
     'stages': []}

If you notice, ``name`` was not restored to the new stage. That's because
``name`` is only restored if we pass ``clone=True`` as name is considered an
internal property and not always user-customizable. Because we ensure
each stage's name in the GUI is unique. E.g. with clone::

    >>> new_stage = stage_factory.make_stage(state, clone=True)
    >>> new_stage.get_state()
    {'cls': 'CeedStage',
     ...
     'name': 'Stage-2',
     'stages': []}

A fundumental part of Ceed is copying and reconstructing stage objects.
E.g. this is required to recover functions from a template file, from old data,
or even to be able to run the experiment because it is run from a second
process. Consequently, anything required for the stage to be reconstructed
must be returned by :meth:`CeedStage.get_state`.

Reference stages
----------------

Stages can contain other :attr:`CeedStage.stages` as children. Instead of
copying stages around we want to be able to reference another stage and add
that reference as a child of a stage. This is useful so that these sub-stages
update when the original stage's parameters are updated. This is mostly meant
to be used from the GUI, although work fine otherwise.

:class:`CeedStageRef` allows one to reference stages in such a manner.
Instead of copying a stage, just get a reference to it with
:meth:`StageFactoryBase.get_stage_ref` and add it to another stage.

When removed and destroyed, such stage references must be explicitly released
with :meth:`StageFactoryBase.return_stage_ref`, otherwise the original stage
cannot be deleted in the GUI.

Methods that accept stages (such as :meth:`CeedStage.add_stage`) should also
typically accept :class:`CeedStageRef` stages.

:class:`CeedStageRef` cannot be used directly during an experiment, unlike
normal stages. Therefore, they or any stages that contain them must first copy
them and expand them to refer to the orignal stages being referenced before
using them, with :meth:`CeedStage.copy_expand_ref` or
:meth:`CeedStageRef.copy_expand_ref`.

E.g. ::

    >>> stage = CeedStage(...)
    >>> stage
    <ceed.stage.CeedStage: "Stage-2" children=(1, 0), at 0x258b6de3c18>
    >>> ref_stage = stage_factory.get_stage_ref(stage=stage)
    >>> ref_stage
    <ceed.stage.CeedStageRef object at 0x00000258B6DE8128>
    >>> ref_stage.stage
    <ceed.stage.CeedStage: "Stage-2" children=(1, 0), at 0x258b6de3c18>
    >>> stage_factory.return_stage_ref(ref_stage)

Before an experiment using a stage is run, the stage and all its sub-stages and
stage functions that are such references are expanded and copied.

Copying stages
--------------

Stages can be copied automatically using ``deepcopy`` or
:meth:`CeedStage.copy_expand_ref`. The former makes a full copy of all the
stages, but any :class:`CeedStageRef` stages will only copy the
:class:`CeedStageRef`, not the original stage being refered to. The latter,
instead of copying the :class:`CeedStageRef`, will replace any
:class:`CeedStageRef` with copies of the class it refers to.

Stages can be manually copied with :meth:`CeedStage.get_state` and
:meth:`CeedStage.set_state` (although :meth:`StageFactoryBase.make_stage` is
more appropriate for end-user creation).

Create stage in script
----------------------

A stage complete with functions and shapes can be created in a script,
saved to a yaml file, and then imported from the GUI ready to be
used in an experiment. See
:meth:`~ceed.storage.controller.CeedDataWriterBase.\
add_frame.save_config_to_yaml`
for an example.

Custom plugin stage example
---------------------------

As explained above, plugins can register customized :class:`CeedStage`
sub-classes to be included in the GUI. Following is an example of how
:meth:`CeedStage.evaluate_stage` can be overwritten.

By default the :meth:`CeedStage.evaluate_stage` generator cycles through
:attr:`CeedStage.loop` times and for each loop iteration it ticks through all
the stage's :attr:`CeedStage.functions`, setting the stage's shapes to their
values in addition to ticking through the sub-stages simultaniously and then
yielding.

:meth:`CeedStage.evaluate_stage` can be safely overwritten to yield directly
whatever values you wish ignoring any functions or sub-stages.

E.g. if you have a shape in the stage named ``"circle"`` (in the GUI this shape
will have to be added to the stage) and you want its RGB value to to be
(0.5, 0, 0) for 2 frames, (0, 0.6, 0) for 3 frames, and finally (0, 0.2, 0.1)
for 4 frames for a total experiment duration of 9 frames you would write the
following sub-class in the stage plugin:

.. code-block:: python

    class SlowStage(CeedStage):

        def evaluate_stage(self, shapes, last_end_t):
            # always get the first time
            self.t_start = t = yield
            for _ in range(2):
                # r, g, b, a values. a (alpha) should be None
                shapes['circle'].append((0.5, 0, 0, None))
                # this yields so GUI can use the change shape colors
                t = yield
            for _ in range(3):
                shapes['circle'].append((0, 0.6, 0, None))
                t = yield
            for _ in range(4):
                shapes['circle'].append((0, 0.2, 0.1, None))
                t = yield

            # this time value was not used and this effectively makes the
            # stage 9 samples long, and it ends on the last sample so
            # that last time will be used as start of next stage
            self.t_end = t
            # raising this exception is how we indicate we're done
            raise StageDoneException

The above class will behave correctly whether the stage is pre-computed or not
because either way it's called to get the values. See
:meth:`CeedStage.evaluate_stage` for further details.

To add stage settings to the GUI, see :meth:`CeedStage.get_settings_display`
and the CSV stage plugin implamentation.

Other methods could potentially also be overwritten to hook into the stage
lifecycle, but they generally require more care. See all :class:`CeedStage`
methods and below for further details.

Custom graphics
---------------

Besides the shapes drawn in the Ceed GUI or script generated, stages could
add arbitrary Kivy GL graphics to the experiment screen and update them
during an experiment. This e.g. allows the display of a circle whose intensity
falls off as it's farther from the center of the circle.

:class:`CeedStage` provides the following methods to add, update, and remove
these graphics for an experiment: :meth:`~CeedStage.add_gl_to_canvas`,
:meth:`~CeedStage.set_gl_colors`, and :meth:`~CeedStage.remove_gl_from_canvas`.

Additionally, like GUI drawn shapes that automatically log the shape intensity
for each frame to be accessible from
:attr:`~ceed.analysis.CeedDataReader.shapes_intensity`, stages can overwrite
:meth:`~CeedStage.get_stage_shape_names` to add names and use those names to
log arbitrary 4-byte (nominally RGBA for shapes) values for each frame.
These values are also displayed in the graph preview window for all shapes
in the GUI. However, you have to ensure to compute and log the rgba data during
each tick.

See the example plugins in the examples directory.

TODO: if a function or stage has zero duration, any data events logged during
 intitialization is not logged if pre-computing. Log these as well. Similarly,
 logs created after the last frame of a stage/function is not logged when
 pre-computing.
"""
import importlib
from copy import deepcopy
from collections import defaultdict
from fractions import Fraction
from os.path import dirname
import operator
from math import isclose
from random import shuffle
from functools import reduce
from typing import Dict, List, Union, Tuple, Optional, Generator, Set, Type, \
    TypeVar, Any

from kivy.properties import OptionProperty, ListProperty, ObjectProperty, \
    StringProperty, NumericProperty, DictProperty, BooleanProperty
from kivy.event import EventDispatcher
from kivy.graphics import Color, Canvas

from ceed.function import CeedFunc, FuncDoneException, CeedFuncRef, \
    FunctionFactoryBase, FuncBase, CeedFuncOrRefInstance
from ceed.utils import update_key_if_other_key, CeedWithID, UniqueNames
from ceed.shape import CeedShapeGroup, CeedPaintCanvasBehavior, CeedShape

__all__ = (
    'StageDoneException', 'StageType', 'CeedStageOrRefInstance',
    'last_experiment_stage_name', 'StageFactoryBase', 'CeedStage',
    'StageShape', 'CeedStageRef', 'remove_shapes_upon_deletion',
    'register_all_stages', 'register_external_stages'
)

StageType = TypeVar('StageType', bound='CeedStage')
"""The type-hint type for :class:`CeedStage`.
"""

CeedStageOrRefInstance = Union['CeedStage', 'CeedStageRef']
"""Instance of either :class:`CeedStage` or :class:`CeedStageRef`."""

NumFraction = Union[float, int, Fraction]
"""Float, int, or Fraction type.
"""

RGBA_Type = Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float]]
"""A RGBA tuple for floats.
"""

last_experiment_stage_name = 'experiment_sampled'
"""The stage name used for the last experiment run. This name cannot be used
by a stage and this stage is overwritten after each experiment run.
"""


class StageDoneException(Exception):
    """Raised as a signal by a :class:`CeedStage` when it is done.
    """
    pass


class StageFactoryBase(EventDispatcher):
    """A global store of the defined :class:`CeedStage` classes and
    customized stage instances.

    Stages in the factory are displayed to the user in the GUI. Similarly,
    when a user creates a custom stage in the GUI, it's added here
    automatically.

    See :mod:`ceed.stage` for details.

    :Events:

        `on_changed`:
            Triggered whenever a stage is added or removed from the factory.
        on_data_event:
            The event is dispatched by stages whenever the wish to log
            something. During an experiment, this data is captured and
            recorded in the file.

            To dispatch, you must pass the function's
            :attr:`~ceed.utils.CeedWithID.ceed_id`
            and a string indicating the event type. You can also pass arbitrary
            arguments that gets recorded as well. E.g.
            ``stage_factory.dispatch('on_data_event', stage.ceed_id, 'drug',
            .4, 'h2o')``.

            See :attr:`~ceed.analysis.CeedDataReader.event_data` for details
            on default events as well as data type and argument requirements.
    """

    stages_cls: Dict[str, Type['StageType']] = {}
    '''Dict whose keys is the name of the stage classes registered
    with :meth:`register` and whose values is the corresponding classes.::

        >>> stage_factory.stages_cls
        {'CeedStage': ceed.stage.CeedStage}
    '''

    stages: List['CeedStage'] = ListProperty([])
    '''The list of the currently available :class:`CeedStage` instances added
    with :meth:`add_stage`.

    These stages are listed in the GUI and can be used by name to start a stage
    to run.

    It does not include the instances automatically created and stored in
    :attr:`stages_inst_default` when you :meth:`register` a stage class.
    '''

    stages_inst_default: Dict[str, 'CeedStage'] = {}
    '''Dict whose keys is the function :attr:`CeedStage.name` and whose values
    are the corresponding stage instances.

    Contains only the stages that are automatically created and added when
    :meth:`register` is called on a class. ::

        >>> stage_factory.stages_inst_default
        {'Stage': <ceed.stage.CeedStage at 0x1da866f00b8>}
    '''

    stage_names: Dict[str, 'CeedStage'] = DictProperty({})
    '''A dict of all the stages whose keys are the stage :attr:`CeedStage.name`
    and whose values are the corresponding :class:`CeedStage` instances.

    Contains stages added with :meth:`add_stage` as well as those
    automatically created and added when :meth:`register` is called on a class.

    ::

        >>> stage_factory.stage_names
        {'Stage': <ceed.stage.CeedStage at 0x1da866f00b8>}
    '''

    unique_names: UniqueNames = None
    """A set that tracks existing stage names to help us ensure all global
    stages have unique names.
    """

    function_factory: FunctionFactoryBase = None
    """The :class:`~ceed.function.FunctionFactoryBase` instance that contains
    or is associated with all the functions used in the stages.
    """

    shape_factory: CeedPaintCanvasBehavior = None
    """The :class:`~ceed.shape.CeedPaintCanvasBehavior` instance that contains
    or is associated with all the shapes used in the stages.
    """

    _stage_ref: Dict['CeedStage', int] = {}
    """A dict mapping stages to the number of references to the stage.

    References are :class:`CeedStageRef` and created with with
    :meth:`get_stage_ref` and released with :meth:`return_stage_ref`.
    """

    plugin_sources: Dict[str, List[Tuple[Tuple[str], bytes]]] = {}
    """A dictionary of the names of all the plugin packages imported, mapped to
    the python file contents of the directories in the package. Each value is a
    list of tuples.

    The first item of each tuple is also a tuple containing the names of the
    directories leading to and including the python filename relative to the
    package. The second item in the tuple is the bytes content of the file.
    """

    _cached_state = None

    __events__ = ('on_changed', 'on_data_event')

    def __init__(self, function_factory, shape_factory, **kwargs):
        self.shape_factory = shape_factory
        self.function_factory = function_factory
        super(StageFactoryBase, self).__init__(**kwargs)
        self.stages = []
        self.stages_cls = {}
        self.stages_inst_default = {}
        self._stage_ref = defaultdict(int)
        self.unique_names = UniqueNames()
        self.plugin_sources = {}

        self.fbind('on_changed', self._reset_cached_state)
        self._reset_cached_state()

    def _reset_cached_state(self, *args, **kwargs):
        self._cached_state = None

    def on_changed(self, *largs, **kwargs):
        pass

    def on_data_event(self, ceed_id, event, *args):
        pass

    def get_stage_ref(
            self, name: str = None,
            stage: CeedStageOrRefInstance = None) -> 'CeedStageRef':
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

    def return_stage_ref(self, stage_ref: 'CeedStageRef') -> None:
        """Releases the stage ref created by :meth:`get_stage_ref`.

        :param stage_ref: Instance returned by :meth:`get_stage_ref`.
        """
        if stage_ref.stage not in self._stage_ref:
            raise ValueError("Returned stage that wasn't added")

        self._stage_ref[stage_ref.stage] -= 1
        if not self._stage_ref[stage_ref.stage]:
            del self._stage_ref[stage_ref.stage]
            stage_ref.stage.has_ref = False

    def register(self, cls: Type[StageType], instance: 'CeedStage' = None):
        """Registers the class and adds it to :attr:`stages_cls`. It also
        creates an instance (unless ``instance`` is provided, in which case
        that is used) of the class that is added to :attr:`stage_names` and
        :attr:`stages_inst_default`.

        See :mod:`ceed.stage` for details.

        :Params:

            `cls`: subclass of :class:`CeedStage`
                The class to register.
            `instance`: instance of `cls`
                The instance of `cls` to use. If None, a default
                class instance, using the default :attr:`CeedStage.name` is
                stored. Defaults to None.
        """
        name = cls.__name__
        stages = self.stages_cls
        if name in stages:
            raise ValueError(
                '"{}" is already a registered stage'.format(name))
        stages[name] = cls

        s = cls(
            stage_factory=self, function_factory=self.function_factory,
            shape_factory=self.shape_factory) if instance is None else instance
        if s.stage_factory is not self:
            raise ValueError('Instance stage factory is set incorrectly')
        s.name = self.unique_names.fix_name(s.name)
        self.unique_names.add(s.name)

        self.stage_names[s.name] = s
        self.stages_inst_default[s.name] = s
        self.dispatch('on_changed')

    def add_plugin_source(
            self, package: str, contents: List[Tuple[Tuple[str], bytes]]):
        """Adds the package contents to :attr:`plugin_sources` if it hasn't
        already been added. Otherwise raises an error.
        """
        if package in self.plugin_sources:
            raise ValueError(f'{package} has already been added')
        self.plugin_sources[package] = contents

    def get(self, name: str) -> Optional[Type[StageType]]:
        """Returns the class with name ``name`` that was registered with
        :meth:`register`.

        See :mod:`ceed.stage` for details.

        :Params:

            `name`: str
                The name of the class (e.g. ``'CosStage'``).

        :returns: The class if found, otherwise None.
        """
        stages = self.stages_cls
        if name not in stages:
            return None
        return stages[name]

    def get_names(self) -> List[str]:
        """Returns the class names of all classes registered with
        :meth:`register`.
        """
        return list(self.stages_cls.keys())

    def get_classes(self) -> List[Type[StageType]]:
        """Returns the classes registered with :meth:`register`.
        """
        return list(self.stages_cls.values())

    def make_stage(
            self, state: dict,
            instance: CeedStageOrRefInstance = None,
            clone: bool = False, func_name_map: Dict[str, str] = {},
            old_name_to_shape_map: Dict[str, CeedStageOrRefInstance] = None
    ) -> CeedStageOrRefInstance:
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
        state = deepcopy(state)
        c = state.pop('cls')

        if c == 'CeedStageRef':
            cls = CeedStageRef
        else:
            cls = self.get(c)
        assert instance is None or instance.__class__ is cls

        stage = instance
        if instance is None:
            stage = cls(
                stage_factory=self, function_factory=self.function_factory,
                shape_factory=self.shape_factory)

        stage.apply_state(state, clone=clone, func_name_map=func_name_map,
                          old_name_to_shape_map=old_name_to_shape_map)

        if c == 'CeedStageRef':
            self._stage_ref[stage.stage] += 1
            stage.stage.has_ref = True
        return stage

    def add_stage(self, stage: 'CeedStage', allow_last_experiment=True) -> None:
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
        if stage.stage_factory is not self:
            raise ValueError('stage factory is incorrect')

        if not allow_last_experiment and \
                stage.name == last_experiment_stage_name:
            stage.name += '_'

        stage.name = self.unique_names.fix_name(stage.name)
        self.unique_names.add(stage.name)
        stage.fbind('name', self._change_stage_name, stage)
        stage.fbind('on_changed', self.dispatch, 'on_changed')

        self.stages.append(stage)
        self.stage_names[stage.name] = stage

        self.dispatch('on_changed')

    def remove_stage(self, stage: 'CeedStage', force=False) -> bool:
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
        stage.funbind('on_changed', self.dispatch, 'on_changed')
        self.unique_names.remove(stage.name)

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

    def clear_stages(self, force=False) -> None:
        """Removes all the stages registered with :meth:`add_stage`.

        :Params:

            `force`: bool
                If True, it'll remove all stages even if there are
                references to them created by :meth:`get_stage_ref`.
        """
        for stage in self.stages[:]:
            self.remove_stage(stage, force)

    def find_shape_in_all_stages(
            self, _, shape, process_shape_callback) -> None:
        '''Searches for the :class:`ceed.shape.CeedShape` instance in all the
        known stages and calls ``process_shape_callback`` on each found.

        :Params:

            `_`: anything
                This parameter is ignored and can be anything.
            `shape`: :class:`ceed.shape.CeedShape`
                The shape to search in all stages.
            `process_shape_callback`: callback function
                It is called with two parameters; the :class:`CeedStage`
                and :class:`ceed.shape.CeedShape` instance for each found.
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
                self.unique_names.remove(name)
                # only one change at a time happens because of binding
                break
        else:
            raise ValueError(
                '{} has not been added to the stage'.format(stage))

        name = stage.name
        if name == last_experiment_stage_name:
            name += '_'

        new_name = self.unique_names.fix_name(name)
        self.unique_names.add(new_name)
        self.stage_names[new_name] = stage
        stage.name = new_name

        if not new_name:
            stage.name = 'stage'
        self.dispatch('on_changed')

    def save_stages(self, use_cache=False) -> List[dict]:
        """Returns a dict representation of all the stages added with
        :meth:`add_stage`.

        :param use_cache: If True, it'll get the state using the cache from
            previous times the state was read and cached, if the cache exists.

        It is a list of dicts where each item is the
        :meth:`CeedStage.get_state` of the corresponding stage in
        :attr:`stages`.
        """
        if self._cached_state is not None and use_cache:
            return self._cached_state

        d = [s.get_cached_state(use_cache=use_cache) for s in self.stages]
        self._cached_state = d
        return d

    def recover_stages(
            self, stage_states: List[dict], func_name_map: dict,
            old_name_to_shape_map: Dict[str, CeedStageOrRefInstance]) -> \
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

            stage = self.get(c)(
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

    def tick_stage(
            self, t_start: NumFraction, frame_rate: Fraction,
            stage_name: str = '', stage: Optional['CeedStage'] = None,
            pre_compute: bool = False
    ) -> Generator[List[Tuple[str, List[RGBA_Type]]], NumFraction, None]:
        '''A generator which starts a :class:`CeedStage` and can be time-ticked
        to generate the shape intensity values for each time point in the
        experiment.

        A :class:`CeedStage` represents a collection of shapes with functions
        applied to them. Each of these shapes has a defined intensity for
        every time point. This generator walks through time computing the
        intensity for each shape for every time point and yielding it.

        See :meth:`get_all_shape_values` for example usage. Ceed GUI uses this
        to generate the shape intensity values shown during an experiment.

        See the example in :mod:`~ceed.stage` showing how to run a stage with
        this method.

        :Params:

            `t_start`: a number
                The global time at which the stage starts.
            `frame_rate`: fraction
                The frame rate to sample at, as a fraction.
            `stage_name`: str
                The :attr:`CeedStage.name` of the stage to start.
            `stage`: str
                The :class:`CeedStage` to start.
            `pre_compute`: bool
                Whether to pre-compute the stages, for those stages that support
                it.

        :yields:

            A list of the intensity values for each shape.

            Each item in the list is a 2-tuple of ``(name, values)``. ``name``
            is the :attr:`kivy_garden.painter.PaintShape.name` of the shape
            and is listed only once in the list.
            ``values`` is a list of color values and each item in that list is
            a 4-tuple of ``(r, g, b, a)``. Any of these values can be None, in
            which case that color remains the same (see
            :meth:`set_shape_gl_color_values`). This way a shape can be
            updated from multiple sub-stages, where only e.g. the ``r`` value
            is changed.

        :raises:

            `StageDoneException`:
                When done with the stage (time is out of bounds). The time
                at which this is raised was not used by the stage so there's
                no shape values for that last time point.
        '''
        if stage is None:
            stage = self.stage_names[stage_name]

        # shapes is updated in place with zero or more values for each shape
        shapes = {name: [] for name in stage.get_stage_shape_names()}

        stage.init_stage_tree()
        stage.apply_pre_compute(
            pre_compute, frame_rate, t_start, set(shapes.keys()))

        tick_stage = stage.tick_stage(shapes, t_start)
        next(tick_stage)
        t = yield

        while True:
            tick_stage.send(t)

            shape_values = []
            for name, colors in shapes.items():
                shape_values.append((name, colors[:]))
                del colors[:]

            t = yield shape_values

    def add_shapes_gl_to_canvas(
            self, canvas: Canvas, name: str, quad: Optional[int] = None
    ) -> Dict[str, Color]:
        '''Adds all the kivy OpenGL instructions required to display the
        intensity-varying shapes to the kivy canvas and returns the color
        classes that control the color of each shape.

        This is called by Ceed when it creates a new experiment to get all the
        graphics for the shapes that it will control during the experiment.

        :Params:

            `canvas`: Kivy canvas instance
                The canvas to which the gl instructions are added. It add e.g.
                the polygon and its color.
            `name`: str
                The name to associate with these OpenGL instructions.
                The name is used later to remove the instructions as it allows
                to clear all the instructions with that name.
            `quad`: int or None
                When in quad mode, we add the instructions 4 times, one for
                each quad (top-left, top-right, bottom-left, bottom-right) so
                it is called 4 times sequentially.
                This counts up from 0-3 in that case. Otherwise, it's None.
                From a user POV it should not matter whether we're in quad mode
                because Ceed handles scaling and placing the graphics in the
                right area.

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

    def set_shape_gl_color_values(
            self, shape_views: Optional[Dict[str, Color]],
            shape_values: List[Tuple[str, List[RGBA_Type]]],
            quad: Optional[int] = None, grayscale: str = None
    ) -> List[Tuple[str, float, float, float, float]]:
        '''Takes the dict of the Color instances that control the color of
        each shape as well as the list of the color values for the current time
        point for each shape and sets the shape's color to those values.

        This is called by Ceed for every time step to set the current shape
        color values. In QUAD4X it's called 4 times per frame, in QUAD121X it's
        called 12 times per frame.

        The shape color values is a list of 4-tuples, each a ``r, g, b, a``
        value. In each tuple, any of them can be None, in which case that color
        channel is skipped for that tuple. The list is flattened and the last
        value for each channel across all tuples is used (after being forced to
        the ``[0, 1]`` range). If any are None across all tuples, it's left
        unchanged and not set. If all r, g, b, and a is None, that shape
        becomes transparent.

        :Params:

            `shape_views`: dict
                The dict of shape names and shapes colors as returned by
                :meth:`add_shapes_gl_to_canvas`.

                If it is None, the color will not be updated but the return
                result will be identical to when not None.
            `shape_values`: list
                The list of color intensity values to use for each shape as
                yielded by :meth:`tick_stage`.
            `quad`: int or None
                When in quad mode, we added the instructions 4 times, one for
                each quad. This indicates which quad is being updated, counting
                up from 0-3 in that case. Otherwise, it's None.
            `grayscale`: str
                The colors to operate on. Can be any subset of the string
                'rgb'. Specifically, although we get intensity values for some
                subset of r, g, b values for each stage from the stage settings,
                this computes the average intensity for the active RGB channels
                selected in the stage and assigns the mean to all of the colors
                listed in ``grayscale``.

                E.g. if a stage selects the r and g colors in its config, and
                ``grayscale`` is ``"gb"``, then both the g and b channels are
                set to the mean of the r and g values provided by the stage for
                this timestep (b is excluded since the stage
                provides no value for it). The b channel is not set so it's
                left unchanged (i.e. it'll keep the last value).

                This is how we turn the color into a gray-scale value when e.g.
                in ``QUAD12X`` mode. Specifically, in that mode, this method is
                called 12 times, 4 for the 4 quads, and 3 for the r, g, and b
                color channel for each quad. It gets called 4 times for the red
                channel with ``grayscale`` set to ``'r'``, followed by 4 times
                for the green channel with ``grayscale`` set to ``'g'``,
                followed by 4 times for the blue channel with ``grayscale`` set
                to ``'b'``. This sets the value for 12 frames.

        :returns:

            A list of the colors each shape was set to. Each item in the list
            is ``(name, r, g, b, a)``, where ``name`` is the shape's name and
            ``r``, ``g``, ``b``, ``a`` is the color value it was set to.
            Each ``name`` occurs at most once in the list.

        E.g. from the worked example in :mod:`~ceed.stage`, by default
        we called :meth:`set_shape_gl_color_values` with no grayscale parameter
        value, which printed::

            time=0s,	intensity="[('Shape', 0, 0, 0, 1)]"
            time=1/2s,	intensity="[('Shape', 0.165, 0, 0.165, 1)]"
            time=1s,	intensity="[('Shape', 0.33, 0, 0.33, 1)]"
            ...

        If we provide ``"r"`` for grayscale, it prints::

            time=0s,	intensity="[('Shape', 0.0, 0.0, 0.0, 1)]"
            time=1/2s,	intensity="[('Shape', 0.165, 0.165, 0.165, 1)]"
            time=1s,	intensity="[('Shape', 0.33, 0.33, 0.33, 1)]"
            ...

        That is the stage only sets red and blue, so it averages those two
        values, which happen to be the same because there's only one stage
        setting the color value for both red/blue channels. This mean value
        is assigned to r, g, and b in the result. However, if ``shape_views``
        was provided to the function, only the red channel's color would be
        set to this value because grayscale was ``"r"``. If it was ``"rg"``,
        the printed value would be the same, but only red and green of the Color
        graphics instructions would be set to the mean value and the others
        (green/blue or blue, respectively) remain unchanged.
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

            if shape_views is not None and name in shape_views:
                color = shape_views[name]
            else:
                color = None
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

    def remove_shapes_gl_from_canvas(
            self, canvas: Canvas, name: str) -> None:
        '''Removes all the shape and color instructions that was added with
        :meth:`add_shapes_gl_to_canvas`.

        This is called by Ceed after an experiment and it removes all the
        instructions added with this group name.

        :Params:

            `canvas`: Kivy canvas instance
                The canvas to which the gl instructions were added.
            `name`: str
                The name used when adding the instructions with
                :meth:`add_shapes_gl_to_canvas`.
        '''
        if canvas:
            canvas.remove_group(name)

    def get_all_shape_values(
            self, frame_rate: Fraction, stage_name: str = '',
            stage: Optional['CeedStage'] = None,
            pre_compute: bool = False
    ) -> Dict[str, List[Tuple[float, float, float, float]]]:
        '''Uses :meth:`tick_stage` for every shape in the stage ``stage_name``
        or given ``stage``, it samples all the shape intensity values at the
        given frame rate for the full stage duration and returns a list of
        intensity values for each shape corresponding to each time point.

        frame_rate is not :attr:`frame_rate` (although it can be) bur rather
        the rate at which we sample the functions.

        TODO: skip functions that are infinite duration. Add option to indicate
          stage is also infinite. Currently it would just hang for infinite
          stage.
        '''
        # the sampling rate at which we sample the functions

        tick = self.tick_stage(
            1 / frame_rate, frame_rate, stage_name=stage_name,
            stage=stage, pre_compute=pre_compute)
        next(tick)

        obj_values = defaultdict(list)
        count = 0
        while True:
            count += 1

            try:
                shape_values = tick.send(count / frame_rate)
            except StageDoneException:
                break

            values = self.set_shape_gl_color_values(
                None, shape_values)
            for name, r, g, b, a in values:
                obj_values[name].append((r, g, b, a))
        return obj_values

    def add_manual_gl_to_canvas(
            self, screen_width: int, screen_height: int, stage: 'CeedStage',
            canvas: Canvas, name: str, quad_mode: str,
            quad: Optional[int] = None
    ) -> List['CeedStage']:
        """Adds all the kivy OpenGL instructions that a stage may manually
        set. It internally calls :meth:`~CeedStage.add_gl_to_canvas` for the
        root stage and all its substages.

        This is called by Ceed when it creates a new experiment to get all the
        graphics for the stage used during an experiment.

        :param screen_width: The width of the projector in pixels.
        :param screen_height: The height of the projector in pixels.
        :param stage: The root :class:`CeedStage` that will be run.
        :param canvas: The Kivy canvas instance to which the gl instructions
            must be added.
        :param name: The name to associate with these OpenGL instructions.
            The name is used later to remove the instructions in
            :meth:`remove_shapes_gl_from_canvas` as it allows to clear all the
            instructions with that name.
        :param quad_mode: Whether we're in quad mode. This is the specific quad
            mode used. It can be one of 'RGB' (normal mode), 'QUAD4X', or
            'QUAD12X'.
        :param quad: When in quad mode, we have to add the instructions 4 times,
            one for each quad (top-left, top-right, bottom-left, bottom-right)
            so it is called 4 times sequentially. This counts up from 0-3 in
            that case. Otherwise, it's None.

            From a user POV it should not matter whether we're in quad mode
            because Ceed handles scaling and placing the graphics in the
            right area. So the user must always create their graphics at full
            screen size and relative to bottom left corner. Ceed will
            automatically scale and translate them to the appropriate quad.
        :return: The list of stages who added graphics instructions (their
            :meth:`~CeedStage.add_gl_to_canvas` returned True).
        """
        return [s for s in stage.get_stages() if s.add_gl_to_canvas(
            screen_width, screen_height, canvas, name, quad_mode, quad)]

    def set_manual_gl_colors(
            self, stages: List['CeedStage'], quad: Optional[int] = None,
            grayscale: str = None, clear: bool = False) -> None:
        """Calls :meth:`~CeedStage.set_gl_colors` for all the stages with manual
        graphics.

        Called by Ceed for every time step to allow the stages to update
        their manually added gl instructions (in
        :meth:`add_manual_gl_to_canvas`) for this frame. In QUAD4X it's called
        4 times per frame, in QUAD121X it's called 12 times per frame.

        :param stages: The list of stages returned by
            :meth:`add_manual_gl_to_canvas`.
        :param quad: Same as in :meth:`set_shape_gl_color_values`.
        :param grayscale: Same as in :meth:`set_shape_gl_color_values`.
        :param clear: Unlike for the shape graphics that Ceed controls directly
            Ceed does not control the manual graphics. If a stage ends in the
            middle of a frame in quad mode, then the rest of the graphics or
            color channels for that frame must be cleared to black. Therefore,
            This will be called for those quads/channels with this parameter
            True and you must clear it.
        """
        for stage in stages:
            stage.set_gl_colors(quad, grayscale, clear)

    def remove_manual_gl_from_canvas(
            self, stage: 'CeedStage', canvas: Canvas, name: str) -> None:
        """Removes all the gl instructions that was added with
        :meth:`add_manual_gl_to_canvas`. It internally calls
        :meth:`~CeedStage.remove_gl_from_canvas` for the root stage and all its
        substages.

        This is called by Ceed after an experiment and it should remove all the
        instructions added. Instructions added with this ``name`` will be
        automatically removed by :meth:`remove_shapes_gl_from_canvas` so they
        don't have to be removed manually.

        :param stage: The root :class:`CeedStage` that was run.
        :param canvas: The Kivy canvas instance to which the gl instructions
            was added.
        :param name: The name associated with these OpenGL instructions.
        """
        for s in stage.get_stages():
            s.remove_gl_from_canvas(canvas, name)


class CeedStage(EventDispatcher, CeedWithID):
    '''The stage that controls a time period of an experiment.

    See :mod:`ceed.stage` for details.

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
    Can be one of ``'serial'`` (one after the other), ``'parallel'`` (all in
    parallel).

    See :class:`CeedStage` description for details.
    '''

    complete_on: str = OptionProperty('all', options=['all', 'any'])
    '''When to consider the stage's children stages to be complete if we contain
    sub-stages - :attr:`stages`. Can be one of ``'all'``, ``'any'``.

    If ``'any'``, this stage is done when **any** of the children stages is
    done, and when all of this stage's functions are done. If ``'all'``, this
    stage is done when **all** children stages are done, and when all of this
    stage's functions are done.

    See :class:`CeedStage` description for details.
    '''

    disable_pre_compute: bool = BooleanProperty(False)
    """Whether to disable pre-computing for this stage.

    When pre-computing, either the stage is completely pre-computed from its
    functions and all sub-stages and then the stage's intensity values for all
    time-points becomes essentially a flat list stored in
    :attr:`runtime_stage`. Or, if e.g. some of the sub-stages or functions
    cannot be pre-computed, then only the stage's functions (or those among the
    functions that can be pre-computed) are pre-computed and all are stored in
    :attr:`runtime_functions`.

    When :attr:`disable_pre_compute` is True, neither is pre-computed, even if
    the overall
    :attr:`~ceed.view.controller.ViewControllerBase.pre_compute_stages` is True.
    """

    loop: int = NumericProperty(1)
    """The number of loop iterations the stage should do.

    If more than one, the stage will go through all its :attr:`functions` and
    :attr:`stages` :attr:`loop` times.
    """

    stages: List['CeedStageOrRefInstance'] = []
    '''A list of :class:`CeedStage` instances that are sub-stages of this
    stage.

    See :class:`CeedStage` description for details.
    '''

    parent_stage: 'CeedStage' = ObjectProperty(
        None, rebind=True, allownone=True)
    '''The parent stage when this stage is a sub-stage of another.
    '''

    has_ref: bool = BooleanProperty(False)
    """Whether there's a CeedStageRef pointing to this stage.
    """

    functions: List[CeedFuncOrRefInstance] = []
    '''A list of :class:`ceed.function.FuncBase` instances which the
    stage iterates through sequentially to compute the intensity of all the
    :attr:`shapes` at each time point.
    '''

    shapes: List['StageShape'] = []
    '''The list of :class:`StageShape` instances that are associated
    with this stage.

    All the shapes are set to the same intensity value at
    every time point according to the :attr:`functions` value at that
    time point.
    '''

    color_r: bool = BooleanProperty(False)
    '''Whether the :attr:`shapes` red channel should be set to the
    :attr:`functions` value or if it should remain at zero intensity (False).
    '''

    color_g: bool = BooleanProperty(False)
    '''Whether the :attr:`shapes` green channel should be set to the
    :attr:`functions` value or if it should remain at zero intensity (False).
    '''

    color_b: bool = BooleanProperty(True)
    '''Whether the :attr:`shapes` blue channel should be set to the
    :attr:`functions` value or if it should remain at zero intensity (False).
    '''

    color_a: bool = NumericProperty(None, allownone=True)
    '''Whether the :attr:`shapes` alpha channel should be set. Currently
    this is ignored.
    '''

    randomize_child_order: bool = BooleanProperty(False)
    """Whether the sub-stages order should be randomly re-shuffled before each
    experiment.

    If True, :attr:`stages` order stays the same, but the stage executes them
    in random order. The order is pre-sampled before the stage is executed and
    the given order is then used during the stage.

    The order is stored in :attr:`shuffled_order`.

    See also :attr:`randomize_order_each_loop`, :attr:`lock_after_forked`,
    :attr:`loop_count`, and :attr:`loop_tree_count`.
    """

    randomize_order_each_loop = BooleanProperty(True)
    """When :attr:`randomize_child_order` is True, whether the child order
    should be re-shuffled for each loop iteration including loops of parent
    and parent of parents etc. (True) or whether we shuffle
    once before each experiment and use that order for all loop iterations.

    See also :attr:`randomize_child_order`, :attr:`lock_after_forked`,
    :attr:`loop_count`, and :attr:`loop_tree_count`.
    """

    lock_after_forked: bool = BooleanProperty(False)
    """Stages can reference other stages. After the reference stages
    are expanded and copied before running the stage as an experiment, if
    :attr:`lock_after_forked` is False then :attr:`shuffled_order` is
    re-sampled again for each copied stage. If it's True, then it is not
    re-sampled again and all the stages referencing the original stage will
    share the same randomized re-ordering as the referenced stage.

    See also :attr:`copy_and_resample`.
    """

    shuffled_order: List[List[int]] = []
    """When :attr:`randomize_child_order` is True, it is a list of the
    :attr:`stages` ordering that we should use for each loop.

    It is a list of lists, and each internal list corresponds to a single loop
    iteration as indexed by :attr:`loop_tree_count`. If we don't
    :attr:`randomize_order_each_loop`, then it contains a single list used for
    all the loops, otherwise there's one for each loop.

    If not :attr:`randomize_child_order`, then it's empty and they run in
    :attr:`stages` order.

    If there are more loops than number of items in the outer list, we use the
    last sub-list for the remaining loops. If not all :attr:`stages` indices are
    included in the internal lists, those stages are skipped.
    """

    stage_factory: StageFactoryBase = None
    """The :class:`StageFactoryBase` this stage is associated with.
    """

    function_factory: FunctionFactoryBase = None
    """The :class:`~ceed.function.FunctionFactoryBase` the :attr:`functions`
    are associated with.
    """

    shape_factory: CeedPaintCanvasBehavior = None
    """The :class:`~ceed.shape.CeedPaintCanvasBehavior` the :attr:`shapes`
    are associated with.
    """

    display = None
    """A widget used by the GUI to display the stage.
    """

    pad_stage_ticks: int = 0
    """If the duration of the stage as represented by the number of clock tick
    is less than :attr:`pad_stage_ticks`, the stage will be padded to
    :attr:`pad_stage_ticks` clock cycles at the end.

    During those cycles, the stage's shapes will be unchanged by this stage
    (i.e. if another stage is simultaneously active and set their values, that
    value will be used, otherwise it'll be transparent), except for the shapes
    with :attr:`StageShape.keep_dark` that will still be kept black.

    See :attr:`~ceed.view.controller.ViewControllerBase.pad_to_stage_handshake`
    for usage details.

    .. warning::

        This is for internal use and is not saved with the stage state.
    """

    t_start: NumFraction = 0
    '''The global time with which the stage or loop iteration was
    initialized. The value is in seconds.

    Don't set directly, it is set in :meth:`init_stage` and
    :meth:`init_loop_iteration`. If the stage is :attr:`can_pre_compute`, this
    is not used after pre-computing the stage.
    '''

    t_end: NumFraction = 0
    """The time at which the loop or stage ended in global timebase.

    Set by the stage after each loop is done and is only valid once loop/stage
    is done. It is used by the next stage in :attr:`stages` after this stage to
    know when to start, or for this stage to know when the the next loop
    started. If the stage is :attr:`can_pre_compute`, after pre-computing the
    stage it is only set to indicate when the stage ends, not for each loop.

    If overwriting :meth:`evaluate_stage`, this must be set with the
    last time value passed in that was *not* used, indicating the time the
    stage ended (i.e. the stage spanned from the stage start time until
    :attr:`t_end`, not including the end). The next stage in the sequence would
    start from this time. Similarly, if manually setting :attr:`runtime_stage`,
    the total stage duration is included and this value is automatically set
    from it in :meth:`evaluate_pre_computed_stage`.

    It is updated before :meth:`finalize_loop_iteration`, and
    :meth:`finalize_stage` are called.
    """

    loop_count: int = 0
    '''The current loop iteration.

    This goes from zero (set by :meth:`init_stage` /
    :meth:`init_loop_iteration`) to :attr:`loop` - 1. The stage is done when
    it is exactly :attr:`loop` - 1, having looped :attr:`times`.
    When :meth:`finalize_loop_iteration` and
    :meth:`finalize_stage` are called, it is the value for the loop
    iteration that just ended.

    If the stage is :attr:`can_pre_compute`, this is not used after
    pre-computing the stage.

    See also :attr:`loop_tree_count`.
    '''

    loop_tree_count: int = 0
    """The current iteration, starting from zero, and incremented for each loop
    of the stage, including outside loops that loop over the stage.

    E.g.::

        Stage:
            name: root
            loop: 2
            Stage:
                name: child
                loop: 3

    Then the root and child stage's :attr:`loop_count` will be 0 - 1, and 0 - 2,
    respectively, while :attr:`loop_tree_count` will be 0 - 1 and 0 - 5,
    respectively.
    """

    runtime_functions: List[
        Tuple[Optional[FuncBase], Optional[List[float]], Optional[list],
              Optional[float]]] = []
    """Stores the pre-computed function values for those can be pre-computed
    and the original function for the rest.

    Similar to :attr:`runtime_stage`, but if :attr:`can_pre_compute` is False
    yet :attr:`disable_pre_compute` is also False, then we pre-compute all
    the functions who are not infinite in duration (
    :attr:`~ceed.function.FuncBase.duration` is non-negative) and store them
    here interleaved with those that are infinite.

    It is a list of 4-tuples of the same length as :attr:`functions`. Each item
    is ``(function, intensity, logs, duration)``. It is generated by
    :meth:`pre_compute_functions`.

    If the corresponding function is
    pre-computable, the ``function`` is None and intensity, logs, and duration
    is similar to :attr:`runtime_stage` with the same constraints about each
    intensity and logs value corresponds to a time point the function is sampled
    and the ending time-point must be larger or equal to duration, relative to
    the function start time. ``logs`` may be one value larger than
    ``intensity``, if there's some logs to be emitted on the frame after the
    last sample.

    If the function is not pre-computable, the ``function`` is the original
    function and ``intensity``, ``logs``, and ``duration`` are None.

    If set manually, ensure that :meth:`apply_pre_compute` is overwritten to do
    nothing, otherwise it may overwrite your value. Similarly,
    :attr:`runtime_stage` must be None, otherwise that will be used instead.
    """

    runtime_stage: Optional[
        Tuple[Dict[str, List[RGBA_Type]], list, int, NumFraction]] = None
    """A 4-tuple of stage ``(intensity, logs, count, duration)``.

    If :attr:`can_pre_compute`, then this stage's intensity values are
    pre-computed into a list for each shape and stored here. Otherwise, if
    it's not pre-computed it is None. ``intensity`` is a dict whose keys
    are shape names and values are a list of r, g, b, a values, one for each
    time point indicating the r,g,b,a value for that shape for that time point.

    Each time-point value corresponds exactly to the time passed to
    the stage that generated the value. E.g. with a linear function, the stage
    may be called with times (relative to the stage start time) such as
    0 / rate, 1 / rate, ... n / rate and the values correspond to the function
    values at those times. Then during the experiment, as we get time values,
    we instead count the number of tick stage calls and that number is the
    index into the values list that we return for all the shapes.

    Similarly, ``logs`` is a list of any data event logs captured during
    pre-computing. These logs are replayed during the real experiment for
    each corresponding frame.

    ``count`` is the number of frames in ``intensity`` and ``logs``.
    However, ``logs`` may be one value larger than ``intensity`` (``count``), if
    there's some logs to be emitted on the frame after the last sample.

    After the last value in the list is used, the next time point past will
    raise a :class:`StageDoneException` indicating the stage is done and the
    time value will have to be larger or equal to :attr:`t_end`, which is the
    same saying the time relative to the stage start time must be larger or
    equal to the ``duration`` of the tuple.

    By default it is generated by :meth:`pre_compute_stage`.
    If set manually by the user, the above constraints must be followed and
    additionally, :meth:`apply_pre_compute` should be overwritten to do
    nothing, otherwise it may overwrite your value.
    """

    can_pre_compute: bool = False
    """Whether we can pre-compute the full stage.

    It is read only and is automatically computed during
    :meth:`init_stage_tree`.

    If True it means that all the :attr:`functions` have finite duration
    (i.e. non-negative), for all :attr:`stages` their :attr:`can_pre_compute`
    is True, and :attr:`disable_pre_compute` is False.

    :meth:`apply_pre_compute` does the pre-computation. If
    :attr:`can_pre_compute` is True, then it precomputes the stage's intensity
    values for all time-points that the stage is active from its
    functions and all sub-stages and then essentially stores it as a flat list
    in :attr:`runtime_stage`.

    If it is False, then if :attr:`disable_pre_compute` is also False, then
    all the functions that can be pre-computed are pre-computed, otherwise
    nothing is pre-computed. In both case we still call
    :meth:`apply_pre_compute` on all sub-stages.
    """

    _clone_props: Set[str] = {'cls', 'name'}
    '''Set of non user-customizable property names that are specific to the
    stage instance and should not be copied when duplicating the stage.
    They are only copied when a stage is cloned, i.e. when it is created
    from state to be identical to the original.
    '''

    _cached_state = None

    __events__ = ('on_changed', )

    def __init__(
            self, stage_factory: StageFactoryBase,
            function_factory: FunctionFactoryBase,
            shape_factory: CeedPaintCanvasBehavior, **kwargs):
        self.stage_factory = stage_factory
        self.function_factory = function_factory
        self.shape_factory = shape_factory
        super(CeedStage, self).__init__(**kwargs)
        self.functions = []
        self.stages = []
        self.shapes = []

        for prop in self.get_state():
            if prop in ('stages', 'functions', 'shapes', 'cls', 'ceed_id'):
                continue
            self.fbind(prop, self.dispatch, 'on_changed', prop)

        self.fbind('on_changed', self._reset_cached_state)
        self._reset_cached_state()

    def _reset_cached_state(self, *args, **kwargs):
        self._cached_state = None

    def __repr__(self):
        module = self.__class__.__module__
        qualname = self.__class__.__qualname__
        name = f'{module}.{qualname}'
        return f'<{name}: "{self.name}" children=({len(self.functions)}, ' \
            f'{len(self.stages)}), at {hex(id(self))}>'

    def on_changed(self, *largs, **kwargs):
        pass

    def get_cached_state(self, use_cache=False) -> Dict:
        """Like :meth:`get_state`, but it caches the result. And next time it
        is called, if ``use_cache`` is True, the cached value will be returned,
        unless the config changed in between. Helpful for backup so we don't
        recompute the full state.

        :param use_cache: If True, it'll get the state using the cache from
            previous times the state was read and cached, if the cache exists.
        :return: The state dict.
        """
        if self._cached_state is not None and use_cache:
            return self._cached_state

        self._cached_state = d = self.get_state(expand_ref=False)
        return d

    def get_state(self, expand_ref: bool = False) -> dict:
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
        d = {'cls': self.__class__.__name__}
        for name in (
                'order', 'name', 'color_r', 'color_g', 'color_b',
                'complete_on', 'disable_pre_compute', 'loop', 'ceed_id',
                'randomize_child_order', 'randomize_order_each_loop',
                'lock_after_forked', 'shuffled_order', 'pad_stage_ticks'):
            d[name] = getattr(self, name)

        d['stages'] = [s.get_state(expand_ref=expand_ref) for s in self.stages]
        d['functions'] = [
            f.get_state(recurse=True, expand_ref=expand_ref)
            for f in self.functions]
        d['shapes'] = [s.get_config() for s in self.shapes]

        return d

    def apply_state(
            self, state: dict = {}, clone: bool = False,
            func_name_map: Dict[str, str] = {},
            old_name_to_shape_map: Dict[str, CeedStageOrRefInstance] = None
    ) -> None:
        """Takes the state of the stage saved with :meth:`get_state` and
        applies it to this stage. it also creates any children functions and
        stages and creates the references to the :attr:`shapes`.

        It is called internally and should not be used directly. Use
        :meth:`StageFactoryBase.make_stage` instead.

        :param state: The state dict representing the stage as returned by
            :meth:`get_state`.
        :param clone: If False, only user customizable properties of the
            stage will be set (i.e those not listed in :attr:`_clone_props`),
            otherwise, all properties from state are
            applied to the stage. Clone is meant to be a complete
            re-instantiation of stage function.
        :param func_name_map: a mapping that maps old function names to
            their new names, in case they were re-named when imported.
        :param old_name_to_shape_map: Mapping from shape names to the shapes.
        """
        p = self._clone_props
        stages = state.pop('stages', [])
        functions = state.pop('functions', [])
        shapes_state = state.pop('shapes', [])

        for k, v in state.items():
            if (clone or k not in p) and k != 'cls':
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

    def __deepcopy__(self, memo) -> 'CeedStage':
        obj = self.__class__(
            stage_factory=self.stage_factory,
            function_factory=self.function_factory,
            shape_factory=self.shape_factory)
        obj.apply_state(deepcopy(self.get_state()))
        return obj

    def copy_expand_ref(self) -> 'CeedStage':
        """Returns a copy of the stage. Any sub-stages, recursively, that are
        ref-stages are expanded to normal stages.
        """
        obj = self.__class__(
            stage_factory=self.stage_factory,
            function_factory=self.function_factory,
            shape_factory=self.shape_factory)
        obj.apply_state(deepcopy(self.get_state(expand_ref=True)), clone=True)
        return obj

    def replace_ref_stage_with_source(
            self, stage_ref: 'CeedStageRef') -> Tuple['CeedStage', int]:
        """Replaces the stage ref in :attr:`stages` with a copy of the
        referenced stage.
        """
        if not isinstance(stage_ref, CeedStageRef):
            raise ValueError('Stage must be a CeedStageRef')

        i = self.stages.index(stage_ref)
        self.remove_stage(stage_ref)
        stage = deepcopy(stage_ref.stage)
        self.add_stage(stage, index=i)
        return stage, i

    def replace_ref_func_with_source(
            self, func_ref: CeedFuncRef) -> Tuple[FuncBase, int]:
        """Replaces the func ref in :attr:`functions` with a copy of the
        referenced function.
        """
        if not isinstance(func_ref, CeedFuncRef):
            raise ValueError('Function must be a CeedFuncRef')

        i = self.functions.index(func_ref)
        self.remove_func(func_ref)
        func = deepcopy(func_ref.func)
        self.add_func(func, index=i)
        return func, i

    def can_other_stage_be_added(
            self, other_stage: CeedStageOrRefInstance) -> bool:
        '''Checks whether the other stage may be added to us.

        If the stage is already a child of this stage or sub-stages, it
        returns False to prevent recursion loops.
        '''
        if isinstance(other_stage, CeedStageRef):
            other_stage = other_stage.stage

        # check if we (or a ref to us) are a child of other_stage
        for stage in other_stage.get_stages(step_into_ref=True):
            if stage is self:
                return False
        return True

    def add_stage(
            self, stage: CeedStageOrRefInstance,
            after: Optional[CeedStageOrRefInstance] = None,
            index: Optional[int] = None) -> None:
        '''Adds a sub-stage instance :class:`CeedStage` to :attr:`stages`.

        :param stage: The :class:`CeedStage` or ref to add.
        :param after: The :class:`CeedStage` in :attr:`stages`
            after which to add this stage, if not None, the default.
        :param index: The index in :attr:`stages`
            at which to add this stage, if not None, the default.
        '''
        stage.parent_stage = self

        if after is None and index is None:
            self.stages.append(stage)
        elif index is not None:
            self.stages.insert(index, stage)
        else:
            i = self.stages.index(after)
            self.stages.insert(i + 1, stage)

        if isinstance(stage, CeedStageRef):
            stage.stage.fbind('on_changed', self.dispatch, 'on_changed')
        else:
            stage.fbind('on_changed', self.dispatch, 'on_changed')

        self.dispatch('on_changed')

    def remove_stage(self, stage: CeedStageOrRefInstance) -> bool:
        '''Removes a sub-stage instance :class:`CeedStage` from :attr:`stages`.
        '''
        if isinstance(stage, CeedStageRef):
            stage.stage.funbind('on_changed', self.dispatch, 'on_changed')
        else:
            stage.funbind('on_changed', self.dispatch, 'on_changed')

        self.stages.remove(stage)  # use is not equal, same for funcs
        assert stage.parent_stage is self
        stage.parent_stage = None

        self.dispatch('on_changed')
        return True

    def add_func(
            self, func: CeedFuncOrRefInstance,
            after: Optional[CeedFuncOrRefInstance] = None,
            index: Optional[int] = None) -> None:
        '''Adds the function instance :class:`ceed.function.FuncBase` to
        :attr:`functions`.

        :param func: The :class:`ceed.function.FuncBase` to add.
        :param after: The :class:`ceed.function.FuncBase` in :attr:`functions`
            after which to add this function, if not None, the default.
        :param index: The index in :attr:`functions`
            at which to add this function, if not None, the default.
        '''

        if after is None and index is None:
            self.functions.append(func)
        elif index is not None:
            self.functions.insert(index, func)
        else:
            i = self.functions.index(after)
            self.functions.insert(i + 1, func)

        if isinstance(func, CeedFuncRef):
            func.func.fbind('on_changed', self.dispatch, 'on_changed')
        else:
            func.fbind('on_changed', self.dispatch, 'on_changed')

        self.dispatch('on_changed')

    def remove_func(self, func: CeedFuncOrRefInstance) -> bool:
        '''Removes the function instance :class:`ceed.function.FuncBase` from
        :attr:`functions`.
        '''
        if isinstance(func, CeedFuncRef):
            func.func.funbind('on_changed', self.dispatch, 'on_changed')
        else:
            func.funbind('on_changed', self.dispatch, 'on_changed')

        self.functions.remove(func)
        self.dispatch('on_changed')
        return True

    def add_shape(
            self, shape: Union[CeedShapeGroup, CeedShape]
    ) -> Optional['StageShape']:
        '''Creates and adds a :class:`StageShape` instance wrapping the
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
        stage_shape.fbind('on_changed', self.dispatch, 'on_changed')

        self.dispatch('on_changed')
        return stage_shape

    def remove_shape(self, stage_shape: 'StageShape') -> None:
        '''Removes a :class:`StageShape` that was previously added to
        :attr:`shapes`.
        '''
        stage_shape.funbind('on_changed', self.dispatch, 'on_changed')
        self.shapes.remove(stage_shape)
        self.dispatch('on_changed')

    def get_settings_display(self, stage_widget) -> Dict[str, Any]:
        """Returns widgets that will be displayed to the user in the stage
        settings.

        These widgets can be used to allow users to further configure custom
        stages. This is called by the Ceed GUI when the settings are first
        displayed to the user.

        :param stage_widget: The root settings widget to which the settings
            will be added as grandchildren by the caller.
        :return: It should return a dict of the name of each setting mapped to
            the widget controlling the setting. It will be displayed in two
            columns: the name followed by the widget on the same row.
        """
        return {}

    def get_stages(
            self, step_into_ref: bool = True
    ) -> Generator[CeedStageOrRefInstance, None, None]:
        '''Generator that iterates depth-first through all the stages
        and children :attr:`stages` and yields these stages.

        :param step_into_ref: bool
            If True, when it encounters a :class:`CeedStageRef` instance
            it'll step into it and return that stage and its children.
            Otherwise, it'll just return the :class:`CeedStageRef` and not
            step into it.
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

    def get_funcs(self) -> Generator[CeedFuncOrRefInstance, None, None]:
        """Generator that iterates depth-first through all the stages
        and children :attr:`stages` and for each stage yields each function from
        :attr:`functions` as well as their children functions, recursively.

        If it encounters a :class:`CeedStageRef` or :class:`CeedFuncRef` it'll
        enter the original stage/function and yield its children.
        """
        for stage in self.get_stages(step_into_ref=True):
            for func in stage.functions:
                yield from func.get_funcs(step_into_ref=True)

    def _get_shape_names(self) -> Tuple[List[str], Set[str]]:
        # all shapes in this stage
        names = set()
        # shapes to keep black
        keep_dark = set()
        for shape in self.shapes:
            if shape.keep_dark:
                keep_dark.add(shape.shape.name)

            src_shape = shape.shape
            if isinstance(src_shape, CeedShapeGroup):
                for src_shape_item in src_shape.shapes:
                    names.add(src_shape_item.name)
            else:
                names.add(src_shape.name)

        return list(names), keep_dark

    def init_stage(self, t_start: NumFraction) -> None:
        """Initializes the stage so it is ready to be used to get
        the stage values. See also :meth:`init_stage_tree` and
        :meth:`init_loop_iteration`. If overriding, ``super`` must be called.

        If the stage is :attr:`can_pre_compute`, this is not used after
        pre-computing the stage.

        :param t_start: The time in seconds in global time. :attr:`t_start`
            will be set to this value.
        """
        self.t_start = t_start
        self.loop_count = 0

        t = float(t_start)
        self.stage_factory.dispatch(
            'on_data_event', self.ceed_id, 'start', self.loop_count,
            self.loop_tree_count, t)
        self.stage_factory.dispatch(
            'on_data_event', self.ceed_id, 'start_loop', self.loop_count,
            self.loop_tree_count, t)

    def init_loop_iteration(self, t_start: NumFraction) -> None:
        """Initializes the stage at the beginning of each loop.

        It's called internally at the start of every :attr:`loop` iteration,
        **except the first**. See also :meth:`init_stage_tree` and
        :meth:`init_stage`. If overriding, ``super`` must be called.

        If the stage is :attr:`can_pre_compute`, this is not used after
        pre-computing the stage.

        :param t_start: The time in seconds in global time. :attr:`t_start`
            will be set to this value.
        """
        self.t_start = t_start

        self.stage_factory.dispatch(
            'on_data_event', self.ceed_id, 'start_loop', self.loop_count,
            self.loop_tree_count, float(t_start))

    def finalize_stage(self) -> None:
        """Finalizes the stage at the end of all its loops, when the
        stage is done. See also :meth:`finalize_loop_iteration`.
        If overriding, ``super`` must be called.
        """
        self.stage_factory.dispatch(
            'on_data_event', self.ceed_id, 'end', self.loop_count,
            self.loop_tree_count, float(self.t_end))

    def finalize_loop_iteration(self) -> None:
        """Finalizes the stage at the end of each loop, including the first
        and last. See also :meth:`finalize_stage`.
        If overriding, ``super`` must be called.
        """
        self.stage_factory.dispatch(
            'on_data_event', self.ceed_id, 'end_loop', self.loop_count,
            self.loop_tree_count, float(self.t_end))

    def tick_stage(
            self, shapes: Dict[str, List[RGBA_Type]], last_end_t: NumFraction
    ) -> Generator[None, NumFraction, None]:
        '''Similar to :meth:`StageFactoryBase.tick_stage` but for this stage.
        This calls internally either :meth:`evaluate_pre_computed_stage` if the
        stage was pre-computed (:attr:`runtime_stage` is not None) or
        :meth:`evaluate_pre_computed_stage`evaluate_stage` when it is not
        pre-computed.

        It is an generator that ticks through time and updates the
        intensity values for the shapes associated with this stage and its
        sub-stages.

        Specifically, at every iteration, a time value is sent to the iterator
        by Ceed which then updates the ``shapes`` dict with the intensity
        values of the shape for the that time-point.

        The method is sent time step values and it yields at every time step
        after the shapes dict is updated. The final time that was sent on
        which it raises :class:`StageDoneException` means that the given time
        was not used and the stage is done for that time value.

        :param shapes: A dict whose keys is the name of all the shapes of this
            stage and its sub-stages. The corresponding values are empty lists.
            At every iteration the list should be filled in with the desired
            color values.
        :param last_end_t: the start time of the stage in globbal time.

        :raises:

            `StageDoneException`:
                When done with the stage (time is out of bounds). The time
                value that raised this was not used.
        '''
        # quick path is stage was pre-computed
        if self.runtime_stage is not None:
            tick = self.evaluate_pre_computed_stage(shapes, last_end_t)
        else:
            tick = self.evaluate_stage(shapes, last_end_t)
        next(tick)

        t = yield
        while True:
            # a StageDoneException is raised when tick is done
            tick.send(t)
            t = yield

    def evaluate_pre_computed_stage(
            self, shapes: Dict[str, List[RGBA_Type]], last_end_t: NumFraction
    ) -> Generator[None, NumFraction, None]:
        """Generator called by :meth:`tick_stage` if the stage was
        pre-computed. See that method for details.

        This should not be overwritten, rather one can set
        :attr:`runtime_stage` to desired values and this method will iterate
        through it.

        If setting :attr:`runtime_stage` manually, :meth:`apply_pre_compute`
        should be overwritten, otherwise it may overwrite
        :attr:`runtime_stage` as it attempts to pre-compute again. But it's
        generally safer and simpler to customize :meth:`evaluate_stage` instead
        and have Ceed generated the pre-compute values from it.
        """
        dispatch = self.stage_factory.dispatch
        # next t to use. On the last t not used raises StageDoneException
        self.t_start = t = yield
        stage_data, logs, n, t_end = self.runtime_stage

        # stage was sampled with a init value of zero, so end is
        # relative to current sample time, not end time of last
        # stage. Because with the latter, the sampled stage could have
        # ended before the sample (t). So we align with sample time
        t_end += t
        prev_t = t

        for i in range(n):
            for name, colors in stage_data.items():
                shapes[name].append(colors[i])
            for log in logs[i]:
                dispatch('on_data_event', *log)
            prev_t = t
            t = yield

        self.t_end = t_end
        assert prev_t <= self.t_end or isclose(prev_t, self.t_end)
        assert t >= self.t_end or isclose(t, self.t_end)

        if len(logs) > n:
            for log in logs[n]:
                dispatch('on_data_event', *log)

        raise StageDoneException

    def evaluate_stage(
            self, shapes: Dict[str, List[RGBA_Type]], last_end_t: NumFraction
    ) -> Generator[None, NumFraction, None]:
        """Generator called by :meth:`tick_stage` in real-time if the stage
        won't be pre-computed or before the stage is run if we're
        pre-computing the stage. See that method for details.

        This method can safely be overwritten to set stage-shape values. And
        if the stage will be pre-computed, this method will still be internally
        called to get the shape values so pre-computing does not have to be
        considered at all when overwriting this method.

        However, :attr:`t_end` must be set with the final stage time before the
        method ends, otherwise it'll break the timing. Similarly,
        :attr:`t_start` should be set the first time value of the stage.
        Following is an appropriate customization (assuming those named shapes
        exist in the GUI)::

            def evaluate_stage(self, shapes, last_end_t):
                # always get the first time
                self.t_start = t = yield
                # we ignore any looping and just use 10 time points.
                for i in range(10):
                    # r, g, b, a values
                    shapes['my shape'].append(
                        (float(.1 * (t - last_end_t)), .2,  (i % 2) * .3, None))
                    shapes['other shape'].append((.1, .2,  (i % 2) * .5, None))
                    # this yields so GUI can use the change shape colors
                    t = yield

                # this time value was not used and this effectively makes the
                # stage 10 samples long, and it ends on the last sample so
                # that last time will be used as start of next stage
                self.t_end = t
                raise StageDoneException
        """
        # next t to use. On the last t not used raises StageDoneException
        self.init_stage(last_end_t)

        pad_stage_ticks = self.pad_stage_ticks
        names, keep_dark = self._get_shape_names()

        t = yield

        count = 0
        for i in range(self.loop):
            if i:
                self.init_loop_iteration(last_end_t)
            tick = self.tick_stage_loop(shapes, last_end_t)
            next(tick)

            try:
                while True:
                    tick.send(t)
                    count += 1
                    t = yield
            except StageDoneException:
                last_end_t = self.t_end
                self.finalize_loop_iteration()
                # last one is done after finalize_stage
                if i != self.loop - 1:
                    self.loop_count += 1
                    self.loop_tree_count += 1

        self.finalize_stage()
        self.loop_count += 1
        self.loop_tree_count += 1
        if count >= pad_stage_ticks:
            raise StageDoneException

        while count < pad_stage_ticks:
            for name in names:
                # as long as we don't return, the clock is going. Normal shapes
                # will keep the color if set by any other stage (otherwise
                # black). But these must be explicitly set to black
                if name in keep_dark:
                    shapes[name].append((0, 0, 0, 1))

            count += 1
            t = yield

        self.t_end = t
        raise StageDoneException

    def _get_loop_stages(self) -> List['CeedStage']:
        stages = self.stages
        shuffled_order = self.shuffled_order
        if not stages or not shuffled_order:
            return stages[:]

        i = self.loop_tree_count
        indices = shuffled_order[i if i < len(shuffled_order) else -1]
        return [stages[k] for k in indices]

    def tick_stage_loop(
            self, shapes: Dict[str, List[RGBA_Type]], last_end_t: NumFraction
    ) -> Generator[None, NumFraction, None]:
        """If the stage was not pre-computed, ticks through one loop iteration
        of the stage yielding the shape values for each time-point.
        """
        names, keep_dark = self._get_shape_names()
        stages = self._get_loop_stages()
        serial = self.order == 'serial'
        end_on_first = self.complete_on == 'any' and not serial
        r, g, b = self.color_r, self.color_g, self.color_b
        a = self.color_a

        func_end_t = None
        stage_end_t = None

        # init func/stages to the end of the last stage/loop, this could be
        # between video frames (sampling points)
        funcs = self.tick_funcs(last_end_t)
        next(funcs)
        current_stage = tick = ticks = None
        remaining_ticks = None
        if stages:
            if serial:
                current_stage = stages.pop(0)
                tick = current_stage.tick_stage(shapes, last_end_t)
                next(tick)
            else:
                ticks = [(s, s.tick_stage(shapes, last_end_t)) for s in stages]
                for _, it in ticks:
                    next(it)
                remaining_ticks = ticks

        try:
            # t is the next time to be used
            t = yield

            while True:
                # if func is done, current t was not used
                if funcs is not None:
                    try:
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
                        func_end_t = self.t_end

                if serial and tick is not None:
                    while True:
                        try:
                            tick.send(t)
                            break
                        except StageDoneException:
                            t_end = current_stage.t_end
                            if stages and not end_on_first:
                                current_stage = stages.pop(0)
                                tick = current_stage.tick_stage(shapes, t_end)
                                next(tick)
                            else:
                                stage_end_t = t_end
                                tick = current_stage = None
                                break

                elif not serial and ticks:
                    for stage, tick_stage in ticks[:]:
                        try:
                            tick_stage.send(t)
                        except StageDoneException:
                            ticks.remove((stage, tick_stage))

                            if end_on_first:
                                ticks = None  # leave remaining
                                stage_end_t = stage.t_end
                                break

                            if stage_end_t is None:
                                stage_end_t = stage.t_end
                            else:
                                stage_end_t = max(stage_end_t, stage.t_end)

                if funcs is None and tick is None and not ticks:
                    # the current t was not used (actually, if parallel and
                    # end_on_first, it may have been used by a earlier stage and
                    # that stage could have added values to shapes, but for now
                    # we'll pretend it wasn't). TODO: fix
                    break

                t = yield

            if func_end_t is None and stage_end_t is None:
                self.t_end = last_end_t
            elif func_end_t is None:
                self.t_end = stage_end_t
            elif stage_end_t is None:
                self.t_end = func_end_t
            else:
                self.t_end = max(func_end_t, stage_end_t)
        finally:
            # set all unfinished stages (that are already started) to our
            # t_end. If we're exiting due to
            # GeneratorExit whoever closed us must have set our t_end, so we'll
            # use that. If we are exiting normally, we set it above so use that
            if serial:
                if current_stage is not None:
                    current_stage.t_end = self.t_end
                    tick.close()
            elif remaining_ticks:
                for stage, tick_stage in remaining_ticks:
                    stage.t_end = self.t_end
                    tick_stage.close()

        raise StageDoneException

    def pre_compute_functions(
            self, frame_rate: Fraction
    ) -> List[Tuple[
            Optional[FuncBase], Optional[List[float]], Optional[list],
            Optional[float]]]:
        """Goes through all the stage's functions and pre-computes those
        that are finite and can be pre-computed.

        Returns a list of pre-computed values/original functions, one for each
        function of the stage. Each item is a 4-tuple of either
        ``(func, None, None, None)`` when the function is not finite, otherwise
        ``(None, pre_computed_values, logs, end_time)``.

        This allows only some functions to be pre-computed.
        """
        computed = []
        t = 0
        last_end_t = None
        values = []
        logs = []

        current_log = []

        def save_log(obj, ceed_id, event, *args):
            current_log.append((ceed_id, event, *args))
            return True

        for func in self.functions:
            # this function should have been copied when it was created for
            # this experiment in the `last_experiment_stage_name` stage
            assert not isinstance(func, CeedFuncRef)

            # we hit a un-cacheable function, reset
            if func.duration < 0:
                if current_log:
                    logs.append(current_log)
                    current_log = []
                if values or logs:
                    computed.append((None, values, logs, last_end_t))

                computed.append((func, None, None, None))
                t = 0
                last_end_t = None
                values = []
                logs = []
                continue

            uid = self.function_factory.fbind('on_data_event', save_log)
            try:
                func.init_func(
                    t / frame_rate if last_end_t is None else last_end_t
                )
                values.append(func(t / frame_rate))
                logs.append(current_log)
                current_log = []

                while True:
                    t += 1
                    values.append(func(t / frame_rate))
                    logs.append(current_log)
                    current_log = []
            except FuncDoneException:
                last_end_t = func.t_end
                assert last_end_t >= 0, "Should be concrete value"
            self.function_factory.unbind_uid('on_data_event', uid)

        if current_log:
            logs.append(current_log)

        if values or logs:
            computed.append((None, values, logs, last_end_t))

        return computed

    def pre_compute_stage(
            self, frame_rate: Fraction, t_start: NumFraction, shapes: Set[str]
    ) -> Tuple[Dict[str, List[RGBA_Type]], list, int, NumFraction]:
        """If the stage is to be pre-computed, :meth:`apply_pre_compute`
        uses this to pre-compute the stage intensity values for all the
        shapes for all time points when the stage would be active.

        It returns the shape intensity values and data logs for all time points
        as well as the end time when the stage ended relative to zero time (not
        ``t_start``).
        """
        stage_data = {s: [] for s in shapes}
        stage_data_temp = {s: [] for s in shapes}
        logs = []

        current_log = []

        def save_log(obj, ceed_id, event, *args):
            current_log.append((ceed_id, event, *args))
            return True

        func_uid = self.function_factory.fbind('on_data_event', save_log)
        stage_uid = self.stage_factory.fbind('on_data_event', save_log)

        # we ignore t_start because we sample relative to zero so we can just
        # add end time to actual passed in time at runtime
        tick = self.tick_stage(stage_data_temp, 0)
        next(tick)
        # is the next timestamp to use at start of loop and on exit
        i = 0

        try:
            while True:
                tick.send(i / frame_rate)

                for name, colors in stage_data_temp.items():
                    color = [None, None, None, None]
                    for item in colors:
                        for j, c in enumerate(item):
                            if c is not None:
                                color[j] = c
                    stage_data[name].append(tuple(color))

                    del colors[:]

                logs.append(current_log)
                current_log = []
                i += 1
        except StageDoneException:
            pass

        self.function_factory.unbind_uid('on_data_event', func_uid)
        self.stage_factory.unbind_uid('on_data_event', stage_uid)

        if current_log:
            logs.append(current_log)

        return stage_data, logs, i, self.t_end

    def tick_funcs(
            self, last_end_t: NumFraction
    ) -> Generator[Optional[float], NumFraction, None]:
        '''Iterates through the :attr:`functions` of this stage sequentially
        and returns the function's value associated with the given time passed
        in for each tick.

        If the functions were pre-computed, it goes through the pre-computed
        values, assuming the passed in time is exactly those values used
        when pre-computing relative to ``last_end_t``.
        '''
        dispatch = self.stage_factory.dispatch
        # always get a time stamp
        t = yield

        for func, values, logs, end_t in self.runtime_functions:
            # values were pre-computed
            if func is None:
                # this was sampled with a init value of zero, so end is
                # relative to current sample time, not end time of last
                # func. Because with the latter, the sampled func could have
                # ended before the sample. So we align with sample time
                last_end_t = t + end_t
                prev_t = t
                for value, log in zip(values, logs):
                    for item in log:
                        dispatch('on_data_event', *item)
                    prev_t = t
                    t = yield value

                if len(logs) > len(values):
                    for item in logs[len(values)]:
                        dispatch('on_data_event', *item)

                assert prev_t <= last_end_t or isclose(prev_t, last_end_t)
                assert t >= last_end_t or isclose(t, last_end_t)
                continue

            # this function should have been copied when it was created for
            # this experiment in the `last_experiment_stage_name` stage
            assert not isinstance(func, CeedFuncRef)

            try:
                # on first func, last_end_t can be None
                func.init_func(t if last_end_t is None else last_end_t)
                t = yield func(t)
                while True:
                    t = yield func(t)
            except FuncDoneException:
                last_end_t = func.t_end
                assert last_end_t >= 0, "Should be concrete value"

        # use start time if no funcs
        self.t_end = last_end_t

        raise FuncDoneException

    def init_stage_tree(self, root: Optional['CeedStage'] = None) -> None:
        """Before the stage is :meth:`apply_pre_compute` and started, the stage
        and all sub-stages are recursively initialized.

        Initializes the stage as part of the stage tree so it is ready
        to be called to get the stage values as part of the tree. It is
        called once for each stage of the entire stage tree.

        :param root: The root of the stage tree. If None, it's self.
        """
        self.loop_tree_count = 0

        for func in self.functions:
            func.init_func_tree()

        if root is None:
            root = self
        for child_stage in self.stages:
            child_stage.init_stage_tree(root)

        funcs_are_finite = all((f.duration >= 0 for f in self.functions))
        can_pre_compute_stages = all((s.can_pre_compute for s in self.stages))
        self.can_pre_compute = funcs_are_finite and can_pre_compute_stages \
            and not self.disable_pre_compute

        self.runtime_functions = [
            (func, None, None, None) for func in self.functions]
        self.runtime_stage = None

    def apply_pre_compute(
            self, pre_compute: bool, frame_rate: Fraction, t_start: NumFraction,
            shapes: Set[str]
    ) -> None:
        """Pre-computes the intensity values for all the shapes of this stage
        and/or sub-stages, if enabled.

        Depending on ``pre_compute``, :attr:`can_pre_compute`, and
        :attr:`disable_pre_compute`, it pre-computes the full stage and
        sub-stages, just the stage's functions or only the sub-stages
        recursively.
        """
        if pre_compute:
            if self.can_pre_compute:
                self.runtime_stage = self.pre_compute_stage(
                    frame_rate, t_start, shapes)

                # don't do anything for the children stages/funcs
                return

            if not self.disable_pre_compute:
                self.runtime_functions = self.pre_compute_functions(
                    frame_rate)

        for stage in self.stages:
            stage.apply_pre_compute(pre_compute, frame_rate, t_start, shapes)

    def resample_parameters(
            self, parent_tree: Optional[List['CeedStage']] = None,
            is_forked: bool = False) -> None:
        """Resamples all parameters of all functions of the stage that have
        randomness associated with it as well as :attr:`shuffled_order` if
        :attr:`randomize_child_order`.

        ``parent_tree`` is not inclusive. It is a list of stages starting from
        the root stage leading up to this stage in the stage tree.

        See :meth:`FuncBase.resample_parameters` and :meth:`copy_and_resample`
        for the meaning of ``is_forked``.
        """
        if parent_tree is None:
            parent_tree = []

        base_loops = reduce(
            operator.mul, (f.loop for f in parent_tree + [self]))

        for root_func in self.functions:
            if isinstance(root_func, CeedFuncRef):
                root_func = root_func.func
            root_func.resample_parameters(
                is_forked=is_forked, base_loops=base_loops)

        self.shuffled_order = shuffled_order = []
        if self.stages and self.randomize_child_order and \
                not (self.lock_after_forked and is_forked):
            indices = list(range(len(self.stages)))
            if self.randomize_order_each_loop:
                for _ in range(base_loops):
                    shuffle(indices)
                    shuffled_order.append(indices[:])
            else:
                shuffle(indices)
                shuffled_order.append(indices[:])

        for stage in self.stages:
            if isinstance(stage, CeedStageRef):
                stage = stage.stage

            stage.resample_parameters(
                parent_tree + [self], is_forked=is_forked)

    def copy_and_resample(self) -> 'CeedStage':
        """Resamples all the functions of the stage and :attr:`shuffled_order`,
        copies and expands the stage, possibly resamples parameters, and returns
        the duplicated stage.

        It copies a stage so that the stage is ready to run in a experiment.
        First it resamples all the parameters of all the functions that have
        randomness associated with it and it resamples :attr:`shuffled_order`.
        Then it copies and expends the stage and any sub-stages and functions
        that refer to other functions/stages. Then, it
        resamples again those randomized function parameters that are not
        marked as
        :attr:`~ceed.function.param_noise.NoiseBase.lock_after_forked` as well
        as :attr:`shuffled_order` if not :attr:`lock_after_forked`. Those
        will maintain their original re-sampled values so that all the expanded
        copies of reference functions/stages have the same random values.

        See :meth:`resample_parameters` and
        :meth:`FuncBase.resample_parameters` as well.
        """
        self.resample_parameters(is_forked=False)
        stage = self.copy_expand_ref()
        # resample only those that can be resampled after expanding and forking
        stage.resample_parameters(is_forked=True)
        return stage

    def get_stage_shape_names(self) -> Set[str]:
        """Gets all the names of the shapes controlled by the stage or
        substages. If adding names, you must call ``super`` to get the
        builtin shapes.

        It calls :meth:`get_stage_shape_names` on its children, recursively,
        so any of them can be overwritten to return arbitrary names (that must
        be unique among all the shapes).

        This is used to create the shape logs in the HDF5 data file. Because
        for each shape we create a Nx4 array, where N is the number of Ceed
        time frames, and 4 is for the RGBA value for that frame
        (:attr:`~ceed.analysis.CeedDataReader.shapes_intensity`).

        By default, it gets the name of the shapes from :attr:`shapes` from
        itself and its children stages. If you're directly updating the
        graphics, you can log rgba values by returning additional names here
        and then setting their values when the stage is ticked, as if it was a
        real shape. The values will then be accessible in
        :attr:`~ceed.analysis.CeedDataReader.shapes_intensity`.

        If you set the values of the additional shapes when the stage is ticked,
        then the values will also show in the stage preview graph so you can use
        it to display arbitrary rgb data for your stage in the graph. But it
        must be logged during stage tick (e.g. :meth:`evaluate_stage`), not
        during the actual graphing in :meth:`set_gl_colors` that follows each
        tick.

        See the stage example plugins for example usage.
        """
        shapes = set()
        for shape in self.shapes:
            if isinstance(shape.shape, CeedShapeGroup):
                for s in shape.shape.shapes:
                    shapes.add(s.name)
            else:
                shapes.add(shape.shape.name)

        for stage in self.stages:
            if isinstance(stage, CeedStageRef):
                stage = stage.stage
            shapes.update(stage.get_stage_shape_names())

        return shapes

    def set_ceed_id(self, min_available: int) -> int:
        min_available = super().set_ceed_id(min_available)
        for func in self.functions:
            if isinstance(func, CeedFuncRef):
                raise TypeError('Cannot set ceed id for ref function')

            min_available = func.set_ceed_id(min_available)

        for stage in self.stages:
            if isinstance(stage, CeedStageRef):
                raise TypeError('Cannot set ceed id for ref stage')

            min_available = stage.set_ceed_id(min_available)

        return min_available

    def get_ref_src(self) -> 'CeedStage':
        """Returns the stage referenced by this object if we are a
        :class:`CeedStageRef`, otherwise it just returns this stage.

        Useful to get the referenced stage without having to check whether we
        are a :class:`CeedStageRef`. I.e. ``stage.get_ref_src().name``.
        """
        return self

    def add_gl_to_canvas(
            self, screen_width: int, screen_height: int, canvas: Canvas,
            name: str, quad_mode: str, quad: Optional[int] = None, **kwargs
    ) -> bool:
        """Should add any stage specific kivy OpenGL instructions that a
        stage may manually set.

        This is called by Ceed when it creates a new experiment to get all the
        graphics for the stage and substages used during an experiment.

        :param screen_width: Same as
            :meth:`~StageFactoryBase.add_manual_gl_to_canvas`.
        :param screen_height: Same as
            :meth:`~StageFactoryBase.add_manual_gl_to_canvas`.
        :param canvas: Same as
            :meth:`~StageFactoryBase.add_manual_gl_to_canvas`.
        :param name: Same as
            :meth:`~StageFactoryBase.add_manual_gl_to_canvas`.
        :param quad_mode: Same as
            :meth:`~StageFactoryBase.add_manual_gl_to_canvas`.
        :param quad: Same as
            :meth:`~StageFactoryBase.add_manual_gl_to_canvas`.
        :return: Whether the stage added custom graphics. If True,
            :meth:`set_gl_colors` will be called for this stage for every Ceed
            frame. Otherwise, it is not called. Defaults to returning None.
        """
        pass

    def set_gl_colors(
            self, quad: Optional[int] = None, grayscale: str = None,
            clear: bool = False, **kwargs
    ) -> None:
        """If :meth:`add_gl_to_canvas` returned True, it is called by Ceed for
        every time step to allow the stage to update the manually added gl
        instructions for this frame. In QUAD4X it's called
        4 times per frame, in QUAD121X it's called 12 times per frame.

        :param quad: Same as
            :meth:`~StageFactoryBase.set_manual_gl_colors`.
        :param grayscale: Same as
            :meth:`~StageFactoryBase.set_manual_gl_colors`.
        :param clear: Same as
            :meth:`~StageFactoryBase.set_manual_gl_colors`.

        .. warning::

            For every clock tick, Ceed "ticks" the stage and then updates the
            graphics based on the values computed during the tick. If the frame
            is not dropped, the value is then applied to the graphics.

            For shapes, Ceed does this automatically, only updating the shapes
            when the frame is not dropped, for each tick. For these manual gl
            graphics, the stage should only update the graphics in this method.
            Normally, every tick is followed by a call to this method to draw.
            If the frame is dropped, the draw call does not follow the tick.
            Hence, only draw in this method.

            If the stage is pre-computed, then Ceed ticks through the stage
            before the stage runs. During the stage, tick won't be called,
            but this method will still be called so you have to work out the
            timing prior and apply it during this method.
        """
        pass

    def remove_gl_from_canvas(
            self, canvas: Canvas, name: str, **kwargs
    ) -> None:
        """Should remove all the gl instructions that was added with
        :meth:`add_gl_to_canvas`.

        It is called by Ceed for the root stage and all substages after the
        experiment is done. It should remove all the instructions added.
        Instructions added with this ``name`` will be
        automatically removed by
        :meth:`~StageFactoryBase.remove_shapes_gl_from_canvas` so they
        don't have to be removed manually.

        :param canvas: The Kivy canvas instance to which the gl instructions
            was added.
        :param name: The name associated with these OpenGL instructions.
        """
        pass


class CeedStageRef:
    """A stage that refers to another stage.

    This is never manually created, but rather returned by
    :meth:`StageFactoryBase.get_stage_ref`.
    See :meth:`StageFactoryBase.get_stage_ref` and :mod:`ceed.stage` for
    details.
    """

    stage: CeedStage = None
    """The reffered to stage.
    """

    display = None
    """Same as :attr:`CeedStage.display`.
    """

    parent_stage: CeedStage = None
    """Same as :attr:`CeedStage.parent_stage`.
    """

    stage_factory: StageFactoryBase = None
    """Same as :attr:`CeedStage.stage_factory`.
    """

    function_factory: FunctionFactoryBase = None
    """Same as :attr:`CeedStage.function_factory`.
    """

    shape_factory: CeedPaintCanvasBehavior = None
    """Same as :attr:`CeedStage.shape_factory`.
    """

    def __init__(
            self, stage_factory: StageFactoryBase,
            function_factory: FunctionFactoryBase,
            shape_factory: CeedPaintCanvasBehavior,
            stage: Optional[CeedStage] = None):
        super(CeedStageRef, self).__init__()
        self.stage = stage
        self.stage_factory = stage_factory
        self.shape_factory = shape_factory
        self.function_factory = function_factory

    def get_cached_state(self, use_cache=False) -> Dict:
        return {'ref_name': self.stage.name, 'cls': 'CeedStageRef'}

    def get_state(self, expand_ref: bool = False) -> dict:
        if expand_ref:
            return deepcopy(self.stage.get_state(expand_ref=True))

        return {'ref_name': self.stage.name, 'cls': 'CeedStageRef'}

    def apply_state(
            self, state: dict = {}, clone: bool = False,
            func_name_map: Dict[str, str] = {},
            old_name_to_shape_map: Dict[str, CeedStageOrRefInstance] = None
    ) -> None:
        self.stage = self.stage_factory.stage_names[state['ref_name']]

    def __deepcopy__(self, memo) -> 'CeedStageRef':
        assert self.__class__ is CeedStageRef
        return self.stage_factory.get_stage_ref(stage=self)

    def copy_expand_ref(self) -> CeedStage:
        return self.stage.copy_expand_ref()

    def get_ref_src(self) -> 'CeedStage':
        """See :meth:`CeedStage.get_ref_src`.
        """
        return self.stage


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

    name: str = StringProperty('')
    '''The :attr:`ceed.shape.CeedShapeGroup.name` or
    :attr:`kivy_garden.painter.PaintShape.name` of the instance wrapped.
    '''

    keep_dark: bool = BooleanProperty(False)
    """Whether this shape will be black during the whole stage. Instead of it
    taking the color of the stage, it'll be kept black.

    This is useful when the inside of some shape must be black, e.g. a donut.
    By setting :attr:`keep_dark` of the inner shape to True, it'll be black.
    """

    display = None

    __events__ = ('on_changed', )

    def __init__(
            self, stage: CeedStage = None,
            shape: Union[CeedShape, CeedShapeGroup] = None, **kwargs):
        super(StageShape, self).__init__(**kwargs)
        self.stage = stage
        self.shape = shape
        self.shape.fbind('name', self._update_name)
        self._update_name()

        for prop in self.get_config():
            self.fbind(prop, self.dispatch, 'on_changed', prop)

    def _update_name(self, *largs) -> None:
        self.name = self.shape.name

    def on_changed(self, *args, **kwargs):
        pass

    def get_config(self) -> Dict:
        """(internal) used by the config system to get the config data of the
        shape.
        """
        return {'keep_dark': self.keep_dark, 'name': self.name}

    def apply_config(self, **kwargs) -> None:
        """(internal) used by the config system to set the config data
        of the shape.
        """
        if 'keep_dark' in kwargs:
            self.keep_dark = kwargs['keep_dark']


def remove_shapes_upon_deletion(
        stage_factory: StageFactoryBase,
        shape_factory: CeedPaintCanvasBehavior, process_shape_callback) -> None:
    """Once called, whenever a shape or group of shapes is deleted in the
    ``shape_factory``, it'll also remove the shape or shape group from all
    stages that reference it.

    This is automatically called by the Ceed GUI, but should be manually called
    if manually creating these factories.

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


def register_all_stages(stage_factory: StageFactoryBase):
    """Registers all the internal plugins and built-in stages with the
    :class:`StageFactoryBase` instance.

    It gets and registers all the plugins stages under
    ``ceed/stage/plugin`` using :func:`~ceed.stage.plugin.get_plugin_stages`.
    See that function for how to make your plugin stages discoverable.

    :param stage_factory: a :class:`StageFactoryBase` instance.
    """
    stage_factory.register(CeedStage)
    import ceed.stage.plugin
    from ceed.stage.plugin import get_plugin_stages
    package = 'ceed.stage.plugin'

    stages, contents = get_plugin_stages(
        stage_factory, base_package=package,
        root=dirname(ceed.stage.plugin.__file__))
    for s in stages:
        stage_factory.register(s)

    stage_factory.add_plugin_source(package, contents)


def register_external_stages(
        stage_factory: StageFactoryBase, package: str):
    """Registers all the plugin stages in the package with the
    :class:`StageFactoryBase` instance.

    See :func:`~ceed.stage.plugin.get_plugin_stages` for how to make your
    plugin stages discoverable.

    Plugin source code files are copied to the data file when a a data file is
    created. However, it doesn't copy all files (i.e. it ignores non-python
    files) so it should be independently tracked for each experiment.

    :param stage_factory: A :class:`StageFactoryBase` instance.
    :param package: The name of the package containing the plugins.
    """
    from ceed.stage.plugin import get_plugin_stages
    m = importlib.import_module(package)

    stages, contents = get_plugin_stages(
        stage_factory, base_package=package, root=dirname(m.__file__))
    for s in stages:
        stage_factory.register(s)

    stage_factory.add_plugin_source(package, contents)
