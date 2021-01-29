"""Function Interface
=========================

Defines the functions used with :mod:`ceed.shape` to create time-varying
intensities for the shapes during an experiment. :class:`~ceed.stage.CeedStage`
combines functions with shapes and displays them during an experiment according
to the stage's function.

Although the functions' range is ``(-infinity, +infinity)`` and this module
places no restriction on the function output so that it may return any
scalar value, the graphics system can only accept values in the ``[0, 1]``
range for each red, green, or blue channel. Consequently, the graphics system
(at :meth:`ceed.stage.StageFactoryBase.fill_shape_gl_color_values`) will clip
the function output to that range.

Function factory and plugins
----------------------------

The :class:`FunctionFactoryBase` is a store of the defined :class:`FuncBase`
sub-classes and customized function instances. Function classes/instances
registered with the :class:`FunctionFactoryBase` instance used by the
the GUI are available to the user in the GUI. During analysis, functions are
similarly registered with the :class:`FunctionFactoryBase` instance used in the
analysis and can then be used to get these functions. E.g.::

    >>> # create the function store
    >>> function_factory = FunctionFactoryBase()
    >>> register_all_functions(function_factory)  # register plugins
    >>> LinearFunc = function_factory.get('LinearFunc')  # get the class
    >>> LinearFunc
    ceed.function.plugin.LinearFunc

Classes can be registered manually with
:meth:`FunctionFactoryBase.register` or they can be registered automatically
with :func:`register_all_functions` if they are an internal plugin or
:func:`register_external_functions` for an external plugin. The GUI calls
:func:`register_all_functions` when started as well as
:func:`register_external_functions` if the
:attr:`~ceed.main.CeedApp.external_function_plugin_package` configuration
variable contains a package name.

See :mod:`ceed.function.plugin` for details on writing plugins.

To get a function class registered with :class:`FunctionFactoryBase`, e.g.
:class:`ceed.function.plugin.CosFunc`::

    >>> CosFunc = function_factory.get('CosFunc')  # using class name
    >>> CosFunc
    ceed.function.plugin.CosFunc

Default function instances
--------------------------

When a function class is registered, we create a default instance of the class,
and that instance is accessible at :attr:`FunctionFactoryBase.funcs_inst`
and from the GUI where a list of function names listed in
:attr:`FunctionFactoryBase.funcs_inst` is shown.::

    >>> function_factory.funcs_inst
    {'Group': <ceed.function.FuncGroup at 0x2a351743c88>,
     'Const': <ceed.function.plugin.ConstFunc at 0x2a351743e48>,
     'Linear': <ceed.function.plugin.LinearFunc at 0x2a351743c18>,
     'Exp': <ceed.function.plugin.ExponentialFunc at 0x2a351743f98>,
     'Cos': <ceed.function.plugin.CosFunc at 0x2a351743eb8>}

We can also add customized function instances to be available to the user
with :meth:`FunctionFactoryBase.add_func`.
They need to have unique names, by which we access them from the GUI or from
:attr:`FunctionFactoryBase.funcs_inst`. E.g.::

    >>> f = LinearFunc(function_factory=function_factory, duration=2, m=2, \
name='line')
    >>> function_factory.add_func(f)
    >>> function_factory.funcs_inst
    {...
     'Cos': <ceed.function.plugin.CosFunc at 0x1da866f0ac8>,
     'line': <ceed.function.plugin.LinearFunc at 0x1da866f0278>}

The name will be automatically changed if a function with the given name
already exists when it is registered. E.g. ::

    >>> f = LinearFunc(function_factory=function_factory, name='line')
    >>> function_factory.add_func(f)
    >>> f2 = LinearFunc(function_factory=function_factory, name='line')
    >>> function_factory.add_func(f2)
    >>> f.name
    'line'
    >>> f2.name
    'line-2'

To use a registered function instance, it needs to be copied before it can be
used. e.g.::

    >>> from copy import deepcopy
    >>> f = deepcopy(function_factory.funcs_inst['line'])
    >>> f.init_func_tree()
    >>> ...

Function basics
---------------

A function inherits from :class:`FuncBase`. It defines the interface that
returns a (stateful) number when called with a time argument.

Ceed functions are not like typical functions that can be called with any
value, rather, they keep some internal state, which determines what its
current domain is. Specifically, functions may only be called with
monotonically increasing time values, like we do in an experiment where time
always monotonically increases. Here's some example basic function
usage::

    >>> # function factory stores all the available function types (classes)
    >>> function_factory = FunctionFactoryBase()
    >>> # register all plugin functions
    >>> register_all_functions(function_factory)
    >>> # get cosine function class from internal plugin
    >>> CosFunc = function_factory.get('CosFunc')
    >>> # cos will have amplitude of 10, frequency of 1Hz and 10s duration
    >>> f = CosFunc(function_factory=function_factory, duration=10, A=10, f=1)
    >>> f
    <ceed.function.plugin.CosFunc at 0x4eadba8>
    >>> # we must specify the time axis. All subsequent times passed to the
    >>> # function will be relative to the time given here (3 seconds).
    >>> f.init_func_tree()
    >>> # initialize function base time at 3. I.e. 3 will be subtracted from
    >>> # future times to convert from global to function-local time
    >>> f.init_func(3)
    >>> # evaluate the function at time 3 - 3 = 0 seconds
    >>> f(3)
    10.0
    >>> f(3.25)  # now at 3.25 - 3 = 0.25 seconds
    6.123233995736766e-16
    >>> f(3.5)
    -10.0
    >>> f(15)  # calling outside the domain raises an exception
    Traceback (most recent call last):
    File "g:\\python\\libs\\ceed\\ceed\\function\\__init__.py", line 1042, in \
__call__
        raise FuncDoneException
    ceed.function.FuncDoneException

Before a function can be used, it must be initialized with
:meth:`FuncBase.init_func_tree` and :meth:`FuncBase.init_func`.
:meth:`FuncBase.init_func_tree` intializes all functions and sub-functions in
the function tree recursively. :meth:`FuncBase.init_func` on the other hand
intitializes each function in the tree just before it is going to be used.
It takes a time value in seconds (in global time) where the domain
of the function starts. This is how we can evaluate the function independant of
the baseline global time. E.g. if a function computes ``f(t) = m * t + b``,
it'll actually always be internally computed as
``f(t) = m * (t - t_start) + b``, where :attr:`FuncBase.t_start` is the value
passed to :meth:`FuncBase.init_func` (and :meth:`FuncBase.init_loop_iteration`
at each loop iteration, but :meth:`FuncBase.init_loop_iteration` is called
internally, not by user code).

To evaluate the function, just call it with a time value as in the example
above.

Function domain and monotonicity
--------------------------------

Functions have a domain, as determined by :meth:`FuncBase.get_domain`.
By default the domain for a just initialized function is
[:attr:`FuncBase.t_start`, :attr:`FuncBase.t_start` +
:attr:`FuncBase.duration` * :attr:`FuncBase.loop` *
:meth:`FuncBase.get_timebase`).
:attr:`FuncBase.duration` can ``-1``, indicating the domain extends to
+infinity. :meth:`FuncBase.get_timebase` allows one to state the
:attr:`FuncBase.duration` in :attr:`FuncBase.timebase` units, for better
accuracy, rather than in seconds. If a function
loops (:attr:`FuncBase.loop`), the domain extends until the end of the loops,
but the domain obviously shrinks with each loop iteration.

The domain always starts at :attr:`FuncBase.t_start`, but
:attr:`FuncBase.t_start` is updated internally for each loop to the time at the
start of the current loop (or more accurately the time the last loop ended,
:attr:`FuncBase.t_end` if it's before the current time). So the domain gets
smaller as we iterate the loops. E.g.::

    >>> LinearFunc = function_factory.get('LinearFunc')
    >>> # slope of 2, with 2 loop iterations
    >>> f = LinearFunc(function_factory=function_factory, duration=10, \
loop=2, m=2)
    >>> f.loop
    2
    >>> f.init_func_tree()
    >>> f.init_func(2)  # the valid domain starts at 2 seconds
    >>> f(0)  # zero seconds is outside the domain
    Traceback (most recent call last):
        File "g:\\python\\libs\\ceed\\ceed\\function\\__init__.py", line \
1032, in __call__
    ValueError: Cannot call function <ceed.function.plugin.LinearFunc object \
at 0x000001B66B022DD8> with time less than the function start 2
    >>> f(2)  # intercept at 2 - 2 = 0 seconds is zero
    0.0
    >>> f.t_start
    2
    >>> f.loop_count  # it's in the first loop iteration
    0
    >>> f.get_domain(current_iteration=False)
    (2, Fraction(22, 1))
    >>> f(10)  # value of time at 10 - 2 = 8 seconds
    16.0
    >>> # now we called it with a time value at the end of the first loop
    >>> # i.e. 12 seconds is 2 + 10, which is the duration of the function,
    >>> # so it automatically increments the loop and updates t_start
    >>> f(12)
    0.0
    >>> f.t_start
    Fraction(12, 1)
    >>> f.loop_count
    1
    >>> f.get_domain(current_iteration=False)  # the valid domain has updated
    (Fraction(12, 1), Fraction(22, 1))
    >>> f(20)
    16.0
    >>> f(22)  # finally this is the end of the second loop end of domain
    Traceback (most recent call last):
        File "g:\\python\\libs\\ceed\\ceed\\function\\__init__.py", line \
1042, in __call__
    ceed.function.FuncDoneException
    >>> f.loop_count
    2
    >>> # there's no more valid domain left
    >>> f.get_domain()
    (-1, -1)
    >>> f(20)  # function may only be called monotonically, it won't always
    >>> # raise an error if called non-monotonically, but generally it will
    Traceback (most recent call last):
        File "g:\\python\\libs\\ceed\\ceed\\function\\__init__.py", \
line 1034, in __call__
    ceed.function.FuncDoneException

As seen above, the domain of a function changes as it's called with time
values. Consequently, functions may only be called with monotonically
increasing time arguments. If violated, it may raise an error, but it doesn't
always.

This rule is required to support functions that perform some IO and
therefore may change some state, so calling with the same time input multiple
times may not make sense. Similarly, we combine functions in a group (e.g.
function A, then B etc), so once we finish function A of the group and moved
to B, we don't support moving to function A again, unless we are in the next
loop. E.g.::

    >>> ConstFunc = function_factory.get('ConstFunc')
    >>> FuncGroup = function_factory.get('FuncGroup')
    >>> # two constant functions that output a, and 10 respectively
    >>> f1 = ConstFunc(function_factory=function_factory, duration=2, a=2)
    >>> f2 = ConstFunc(function_factory=function_factory, duration=2, a=10)
    >>> # create a function group of two sequential constant functions
    >>> f = FuncGroup(function_factory=function_factory, loop=2)
    >>> f.add_func(f1)
    >>> f.add_func(f2)
    >>> f.init_func_tree()
    >>> f.init_func(1)
    >>> f.get_domain(current_iteration=False)
    (1, Fraction(9, 1))
    >>> f(1)  # this is internally calling f1
    2
    >>> f(3)  # now we're calling f2 internally since we past the 2s duration
    10
    >>> # even though the domain is still the same, we cannot call it now with
    >>> # a value less than 3 (because it'd have to jump back to f1 and
    >>> # we don't need or support that)
    >>> f.get_domain(current_iteration=False)
    (1, Fraction(9, 1))
    >>> f.loop_count
    0
    >>> f(5)
    2
    >>> f.loop_count
    1
    >>> f(7)
    10
    >>> f(9)
    Traceback (most recent call last):
        File "g:\\python\\libs\\ceed\\ceed\\function\\__init__.py", line \
1341, in __call__
    ceed.function.FuncDoneException

Function timebase
-----------------

As alluded to above, functions have a optional timebase to help be more
precise with the function duration. Normally, the :attr:`FuncBase.duration`
is set in seconds to the duration the function should take. But if the
projector is going at say 120 frames per second (fps), we may want to e.g.
set the shape to be intensity 1 for frame 0 (``t = 0 / 120``) and intensity
0.5 for frame 1 (``t = 1 / 120``). Expressing
``duration = 1 / 120 = 0.008333333333333333`` is not possible with decimal
math. Consequently, we allow setting the :attr:`FuncBase.timebase`.

:attr:`FuncBase.timebase` determines the units of the
:attr:`FuncBase.duration`. If it's zero, then the units is in seconds, like
normal. If it's non-zero, e.g. the fraction ``Fraction(1, 120)``, then
duration is multiplied by it to get the duration in seconds. So e.g. with
``Fraction(1, 120)``, if duration is ``12``, then the stage duration is
``12/120`` seconds, or 12 frames.

During an experimental stage when functions are called, we pass time to the
functions represented as fractions rather than decimel, where the denominator
represents the true framerate of the projector, the numerator is the number of
frames elpased, so the time value is the elapsed time since the start. So
e.g. we'd call it with ``f(Fraction(180, 120))`` if we are 1.5 seconds into
the stage and the projector frame rate is ``120``. This allows us to do more
precise duration math. E.g.::

    >>> from fractions import Fraction
    >>> # duration will be 2 frames at 120 fps (2/120 seconds)
    >>> f = LinearFunc(function_factory=function_factory, duration=2, \
timebase_numerator=1, timebase_denominator=120, m=2)
    >>> f.init_func_tree()
    >>> f.init_func(1)  # start at 1 sec
    >>> f.get_domain(current_iteration=False)
    (1, Fraction(61, 60))
    >>> f.get_timebase()  # this is what we need to use to get the timebase
    Fraction(1, 120)
    >>> f.timebase
    Fraction(1, 120)
    >>> f(Fraction(120, 120))  # calling it t=frame 0 at 120fps
    0.0
    >>> f(Fraction(121, 120))  # calling it t=frame 1 at 120fps
    0.016666666666666666
    >>> f(Fraction(122, 120))  # calling it t=frame 2 at 120fps
    Traceback (most recent call last):
        File "g:\\python\\libs\\ceed\\ceed\\function\\__init__.py", line \
1150, in __call__
    ceed.function.FuncDoneException

Inheriting timebase
^^^^^^^^^^^^^^^^^^^

Functions can be grouped, e.g. in the example above. We don't want to have to
specify the :attr:`FuncBase.timebase` for each function. Consequently, if the
:attr:`FuncBase.timebase` is unspecified (i.e. zero), a
function will inherit the timebase from the :attr:`FuncBase.parent_func` they
belong to all the way to the root where :attr:`FuncBase.parent_func` is None.

E.g. we want the function to alternate between 2 and 10 for each frame at
120fps::

    >>> # each sub-function is exactly one frame long
    >>> f1 = ConstFunc(function_factory=function_factory, duration=1, a=2)
    >>> f2 = ConstFunc(function_factory=function_factory, duration=1, a=10)
    >>> # so far the timbease are the defaults
    >>> f1.timebase
    Fraction(0, 1)
    >>> f1.get_timebase()  # so it's in seconds
    1
    >>> # because we specify a timebase for the group, all the sub-functions
    >>> # will share the same timebase. Unless a sub function specifically sets
    >>> # its own timebase
    >>> f = FuncGroup(function_factory=function_factory, \
timebase_numerator=1, timebase_denominator=120, loop=2)
    >>> f.add_func(f1)
    >>> f.add_func(f2)
    >>> # now, we inherit the timebase from out parent because we were added
    >>> # to a parent and we didn't set our own timebase
    >>> f1.timebase
    Fraction(0, 1)
    >>> f1.get_timebase()  # our parents timebase
    Fraction(1, 120)
    >>> f.init_func(1)  # i.e. 120 / 120
    >>> f.get_domain(current_iteration=False)
    (1, Fraction(31, 30))
    >>> f(Fraction(120, 120))  # loop 0, f1
    2
    >>> f(Fraction(121, 120))  # loop 0, f2
    10
    >>> f(Fraction(122, 120))  # loop 1, f1
    2
    >>> f(Fraction(123, 120))  # loop 1, f2
    10
    >>> f(Fraction(124, 120))  # done
    Traceback (most recent call last):
        File "g:\\python\\libs\\ceed\\ceed\\function\\__init__.py", line \
1459, in __call__
    ceed.function.FuncDoneException

As we just saw, unless overwritten by a function, the timebase is inherited
from parent functions if they set it. Therefore, to get the actual timebase use
:meth:`FuncBase.get_timebase`, instead of :attr:`FuncBase.timebase`. The latter
is only used to directly set the function's timebase. Functions that need to
use the timebase should design all the duration values with that in mind.

The downside to setting a timebase is that it's specific to that timebase, so
to have a function in different timebases, it would need to be replicated
for each timebase.

Saving and restoring functions
------------------------------

Functions can be saved as a data dictionary and later restored into a function
instance. E.g.::

    >>> f = LinearFunc(function_factory=function_factory, duration=2, m=2, \
name='line')
    >>> state = f.get_state()
    >>> state
    {'name': 'line',
     'cls': 'LinearFunc',
     'loop': 1,
     'timebase_numerator': 0,
     'timebase_denominator': 1,
     'noisy_parameters': {},
     'duration': 2,
     't_offset': 0,
     'm': 2,
     'b': 0.0}
    >>> # this is how we create a function from state
    >>> f2 = function_factory.make_func(state)
    >>> f2
    <ceed.function.plugin.LinearFunc at 0x1a2cf679ac8>
    >>> f2.get_state()
    {'name': 'Linear',
     'cls': 'LinearFunc',
     'loop': 1,
     'timebase_numerator': 0,
     'timebase_denominator': 1,
     'noisy_parameters': {},
     'duration': 2,
     't_offset': 0,
     'm': 2,
     'b': 0.0}

If you notice, ``name`` was not restored to the new function. That's because
``name`` is only restored if we pass ``clone=True`` as name is considered an
internal property and not always user-customizable. E.g. with clone::

    >>> f3 = function_factory.make_func(state, clone=True)
    >>> f3.get_state()
    {'name': 'line',
     'cls': 'LinearFunc',
     'loop': 1,
     ...
     'b': 0.0}

A fundumental part of Ceed is copying and reconstructing function objects.
E.g. this is required to recover functions from a template file, from old data,
or even to be able to run the experiment because it is run from a second
process. Consequently, anything required for the function to be reconstructed
must be returned by :meth:`FuncBase.get_state`.


Reference functions
-------------------

Sometimes you may want to create a function group containing other functions.
Instead of explicitly defining these sub-functions, we may want to refer to
existing registered functions and let these sub-functions update when the
existing function's parameters are updated. They are mostly meant to be used
from the GUI, although work fine otherwise.

:class:`CeedFuncRef` allows one to reference functions in such a manner.
Instead of copying a function, just get a reference to it with
:meth:`FunctionFactoryBase.get_func_ref` and add it to the function group.

When destroyed, such function references must be explicitly released with
:meth:`FunctionFactoryBase.release_func_ref`, otherwise the original function
cannot be deleted in the GUI.

Methods that accept functions (such as :meth:`FuncGroup.add_func`) should also
accept :class:`CeedFuncRef` functions.

:class:`CeedFuncRef` cannot be used directly, unlike normal function.
Therefore, they or any functions that contain them must first copy them and
expand them to refer to the orignal functions being referenced before using
them with :meth:`FuncBase.copy_expand_ref` or
:meth:`CeedFuncRef.copy_expand_ref`.

E.g. ::

    >>> f = LinearFunc(function_factory=function_factory)
    >>> f
    <ceed.function.plugin.LinearFunc at 0x1a2cf679c18>
    >>> ref_func = function_factory.get_func_ref(func=f)
    >>> ref_func
    <ceed.function.CeedFuncRef at 0x1a2cf74d828>
    >>> ref_func.func
    <ceed.function.plugin.LinearFunc at 0x1a2cf679c18>
    >>> function_factory.return_func_ref(ref_func)

Copying functions
-----------------

Functions can be copied automatically using ``deepcopy`` or
:meth:`FuncBase.copy_expand_ref`. The former makes a full copy of all the
functions, but any :class:`CeedFuncRef` functions will only copy the
:class:`CeedFuncRef`, not the original function being refered to. The latter,
instead of copying the :class:`CeedFuncRef`, will replace any
:class:`CeedFuncRef` with copies of the class it refers to.

Functions can be manually copied with :meth:`FuncBase.get_state` and
:meth:`FuncBase.set_state`.

Customizing function in the GUI
-------------------------------

Function properties are customizable in the GUI by the user according to the
values returned by :meth:`FuncBase.get_gui_props`,
:meth:`FuncBase.get_prop_pretty_name`, and :meth:`FuncBase.get_gui_elements`.

These methods control what properties are editable by the user and the values
they may potentially take.

Randomizing function parameters
-------------------------------

A :class:`FuncBase` subclass typically contains parameters. E.g.
:class:`ceed.function.plugin.LinearFunc` has a offset and slope (
:attr:`ceed.function.plugin.LinearFunc.b` and
:attr:`ceed.function.plugin.LinearFunc.m`). Sometimes it is desireable for the
parameters to be randomly re-sampled before each experiment, or even for each
loop iteration (:attr:`FuncBase.loop_tree_count`).

All parameters that support randomization must be returned by
:meth:`FuncBase.get_noise_supported_parameters` and the specific distribution
used to randomize each parameter is stored in
:attr:`FuncBase.noisy_parameters`. The GUI manages
:attr:`FuncBase.noisy_parameters` from user configuration based on
:meth:`FuncBase.get_noise_supported_parameters`.

Then, when the function is prepared by Ceed it calls
:meth:`FuncBase.resample_parameters` that re-samples all the randomized
parameters. Parameters that are randomized once for the function lifetime
(:attr:`~ceed.function.param_noise.NoiseBase.sample_each_loop` is False)
are sampled once and the parameter is set to that value. Parameters that are
randomized once for each loop iteration (see the docs for
:attr:`~ceed.function.param_noise.NoiseBase.sample_each_loop`) are sampled for
as many iterations they'll experience and the samples are stored in
:attr:`FuncBase.noisy_parameter_samples`. Then, the parameter is set to the
corresponding value for each iteration at the start in
:meth:`FuncBase.init_func` and :meth:`FuncBase.init_loop_iteration`.

Possible noise distributions are listed in the
:class:`ceed.function.param_noise.ParameterNoiseFactory` stored in
:attr:`FunctionFactoryBase.param_noise_factory`. See
:mod:`ceed.function.plugin` for how to add distributions.

Running a function
------------------


"""
from typing import Type, List, Tuple, Dict, Optional, Set, TypeVar, \
    Generator, Union
from copy import deepcopy
from collections import defaultdict
from math import isclose
from os.path import dirname
from functools import reduce
import operator
from fractions import Fraction
import importlib

from kivy.event import EventDispatcher
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, \
    ObjectProperty, DictProperty, AliasProperty

from ceed.utils import fix_name, update_key_if_other_key
from ceed.function.param_noise import ParameterNoiseFactory, NoiseBase

__all__ = (
    'FuncDoneException', 'FunctionFactoryBase', 'FuncBase', 'FuncType',
    'CeedFuncOrRefInstance', 'CeedFunc', 'FuncGroup', 'CeedFuncRef',
    'register_all_functions', 'register_external_functions')

FuncType = TypeVar('FuncType', bound='FuncBase')
"""The type-hint type for :class:`FuncBase`.
"""

CeedFuncOrRefInstance = Union['FuncBase', 'CeedFuncRef']
"""Instance of either :class:`CeedFunc` or :class:`CeedFuncRef`."""

FloatOrInt = Union[float, int]
"""Float or in type.
"""

NumFraction = Union[float, int, Fraction]
"""Float, int, or Fraction type.
"""


class FuncDoneException(Exception):
    """Raised when the :class:`FuncBase` is called with a time value after
    its valid time interval.
    """
    pass


class FunctionFactoryBase(EventDispatcher):
    """A global store of the defined :class:`FuncBase` sub-classes and
    customized function instances.

    See :mod:`ceed.function` for details.

    :events:

        on_changed:
            The event is triggered every time a function is added or removed
            from the factory or if a class is registered.
    """

    __events__ = ('on_changed', )

    funcs_cls: Dict[str, Type[FuncType]] = {}
    '''Dict whose keys is the name of the function classes registered
    with :meth:`register` and whose values is the corresponding classes.::

        >>> function_factory.funcs_cls
        {'FuncGroup': ceed.function.FuncGroup,
         'ConstFunc': ceed.function.plugin.ConstFunc,
         'LinearFunc': ceed.function.plugin.LinearFunc,
         'ExponentialFunc': ceed.function.plugin.ExponentialFunc,
         'CosFunc': ceed.function.plugin.CosFunc}
    '''

    funcs_user: List['FuncBase'] = []
    '''List of the function instances registered with :meth:`add_func`.

    It does not include the instances automatically created and stored in
    :attr:`funcs_inst_default` when a function class is :meth:`register`.
    '''

    funcs_inst: Dict[str, 'FuncBase'] = DictProperty({})
    '''Dict whose keys is the function :attr:`FuncBase.name` and whose values
    is the corresponding function instances.

    Contains functions added with :meth:`add_func` as well as those
    automatically created and added when :meth:`register` is called on a class.

    ::

        >>> function_factory.funcs_inst
        {'Group': <ceed.function.FuncGroup at 0x1da866f00b8>,
         'Const': <ceed.function.plugin.ConstFunc at 0x1da866f0978>,
         'Linear': <ceed.function.plugin.LinearFunc at 0x1da866f09e8>,
         'Exp': <ceed.function.plugin.ExponentialFunc at 0x1da866f0a58>,
         'Cos': <ceed.function.plugin.CosFunc at 0x1da866f0ac8>,
         'line': <ceed.function.plugin.LinearFunc at 0x1da866f0278>,
         'line-2': <ceed.function.plugin.LinearFunc at 0x1da866f0e48>}
    '''

    funcs_inst_default: Dict[str, 'FuncBase'] = {}
    '''Dict whose keys is the function :attr:`FuncBase.name` and whose values
    are the corresponding function instances.

    Contains only the functions that are automatically created and added when
    :meth:`register` is called on a class. ::

        >>> function_factory.funcs_inst_default
        {'Group': <ceed.function.FuncGroup at 0x1da866f00b8>,
         'Const': <ceed.function.plugin.ConstFunc at 0x1da866f0978>,
         'Linear': <ceed.function.plugin.LinearFunc at 0x1da866f09e8>,
         'Exp': <ceed.function.plugin.ExponentialFunc at 0x1da866f0a58>,
         'Cos': <ceed.function.plugin.CosFunc at 0x1da866f0ac8>}
    '''

    param_noise_factory: ParameterNoiseFactory = None
    """An automatically created instance of
    :class:`ceed.function.param_noise.ParameterNoiseFactory` that is used to
    register and get noise classes for use with functions.
    """

    _ref_funcs: Dict['FuncBase', int] = {}
    """A dict mapping functions to the number of references to the function.

    References are :class:`CeedFuncRef` and created with with
    :meth:`get_func_ref` and released with :meth:`return_func_ref`.
    """

    plugin_sources: Dict[str, List[Tuple[Tuple[str], bytes]]] = {}
    """A dictionary of the names of all the plugin packages imported, mapped to
    the python file contents of the directories in the package. Each value is a
    list of tuples.

    The first item of each tuple is also a tuple containing the names of the
    directories leading to and including the python filename relative to the
    package. The second item in the tuple is the bytes content of the file.
    """

    def __init__(self, **kwargs):
        super(FunctionFactoryBase, self).__init__(**kwargs)
        self.funcs_cls = {}
        self.funcs_user = []
        self.funcs_inst_default = {}
        self.plugin_sources = {}
        self._ref_funcs = defaultdict(int)
        self.param_noise_factory = ParameterNoiseFactory()

    def on_changed(self, *largs, **kwargs):
        pass

    def get_func_ref(
            self, name: str = None, func: 'FuncBase' = None) -> 'CeedFuncRef':
        """Returns a :class:`CeedFuncRef` instance that refers to the
        original function. See :mod:`ceed.function` for details.

        One of ``name`` or ``func`` must be specified. The function being
        referenced by ``func`` should have been registered with this class,
        although it is not explicitly enforced currently.

        If used, :meth:`return_func_ref` must be called when the reference
        is not used anymore.

        :param name: The name of the function to lookup in :attr:`funcs_inst`.
        :param func: Or the actual function to use.
        :return: A :class:`CeedFuncRef` to the original function.
        """
        func = func or self.funcs_inst[name]
        if isinstance(func, CeedFuncRef):
            func = func.func

        ref = CeedFuncRef(function_factory=self, func=func)
        self._ref_funcs[func] += 1
        func.has_ref = True
        return ref

    def return_func_ref(self, func_ref: 'CeedFuncRef'):
        """Releases the function ref created by :meth:`get_func_ref`.

        :param func_ref: Instance returned by :meth:`get_func_ref`.
        """
        if func_ref.func not in self._ref_funcs:
            raise ValueError("Returned function that wasn't added")
        self._ref_funcs[func_ref.func] -= 1

        if not self._ref_funcs[func_ref.func]:
            del self._ref_funcs[func_ref.func]
            func_ref.func.has_ref = False

    def register(self, cls: Type[FuncType], instance: 'FuncBase' = None):
        """Registers the class and adds it to :attr:`funcs_cls`. It also
        creates an instance (unless ``instance`` is provided, in which case
        that is used) of the class that is added to :attr:`funcs_inst` and
        :attr:`funcs_inst_default`.

        See :mod:`ceed.function` for details.

        :Params:

            `cls`: subclass of :class:`FuncBase`
                The class to register.
            `instance`: instance of `cls`
                The instance of `cls` to use. If None, a default
                class instance, using the default :attr:`FuncBase.name` is
                stored. Defaults to None.
        """
        name = cls.__name__
        funcs = self.funcs_cls
        if name in funcs:
            raise ValueError(
                '"{}" is already a registered function'.format(name))
        funcs[name] = cls

        f = cls(function_factory=self) if instance is None else instance
        if f.function_factory is not self:
            raise ValueError('Instance function factory is set incorrectly')
        f.name = fix_name(f.name, self.funcs_inst)

        self.funcs_inst[f.name] = f
        self.funcs_inst_default[f.name] = f
        self.dispatch('on_changed')

    def add_plugin_source(
            self, package: str, contents: List[Tuple[Tuple[str], bytes]]):
        """Adds the package contents to :attr:`plugin_sources` if it hasn't
        already been added. Otherwise raises an error.
        """
        if package in self.plugin_sources:
            raise ValueError(f'{package} has already been added')
        self.plugin_sources[package] = contents

    def get(self, name: str) -> Optional[Type[FuncType]]:
        """Returns the class with name ``name`` that was registered with
        :meth:`register`.

        See :mod:`ceed.function` for details.

        :Params:

            `name`: str
                The name of the class (e.g. ``'ExpFunc'``).

        :returns: The class if found, otherwise None.
        """
        funcs = self.funcs_cls
        if name not in funcs:
            return None
        return funcs[name]

    def get_names(self) -> List[str]:
        """Returns the class names of all classes registered with
        :meth:`register`.
        """
        return list(self.funcs_cls.keys())

    def get_classes(self) -> List[Type[FuncType]]:
        """Returns the classes registered with :meth:`register`.
        """
        return list(self.funcs_cls.values())

    def add_func(self, func: 'FuncBase'):
        """Adds the function to :attr:`funcs_user` and :attr:`funcs_inst`,
        which makes it available in the GUI.

        See :mod:`ceed.function` for details.

        If the :attr:`FuncBase.name` already exists in :attr:`funcs_inst`,
        :attr:`FuncBase.name` will be set to a unique name based on its
        original name. Once added until removed, anytime the function's
        :attr:`FuncBase.name` changes, if it clashes with an existing
        function's name, it is automatically renamed.

        :Params:

            `func`: a :class:`FuncBase` derived instance.
                The function to add.
        """
        func.name = fix_name(func.name, self.funcs_inst)

        func.fbind('name', self._track_func_name, func)
        self.funcs_inst[func.name] = func
        self.funcs_user.append(func)

        if func.function_factory is not self:
            raise ValueError('function factory is incorrect')
        self.dispatch('on_changed')

    def remove_func(self, func: 'FuncBase', force: bool = False) -> bool:
        """Removes a function previously added with :meth:`add_func`.

        :Params:

            `func`: a :class:`FuncBase` derived instance.
                The function to remove.
            `force`: bool
                If True, it'll remove the function even if there are references
                to it created by :meth:`get_func_ref`.

        :returns: Whether the function was removed successfully. E.g. if
            force is False and it has a ref, it won't be removed.
        """
        if not force and func in self._ref_funcs and self._ref_funcs[func]:
            assert self._ref_funcs[func] > 0
            return False

        func.funbind('name', self._track_func_name, func)

        # we cannot remove by equality check (maybe?)
        for i, f in enumerate(self.funcs_user):
            if f is func:
                del self.funcs_user[i]
                break
        else:
            raise ValueError('{} was not found in funcs_user'.format(func))
        del self.funcs_inst[func.name]

        self.dispatch('on_changed')
        return True

    def _track_func_name(self, func: 'FuncBase', *largs):
        """Fixes the name of the function instances stored here to ensure it's
        unique.
        """
        # get the new name
        for name, f in self.funcs_inst.items():
            if f is func:
                if func.name == name:
                    return

                del self.funcs_inst[name]
                # only one change at a time happens because of binding
                break
        else:
            raise ValueError(
                '{} has not been added to the factory'.format(func))

        new_name = fix_name(func.name, self.funcs_inst)
        self.funcs_inst[new_name] = func
        func.name = new_name
        self.dispatch('on_changed')

    def clear_added_funcs(self, force=False):
        """Removes all the functions registered with :meth:`add_func`.

        :Params:

            `force`: bool
                If True, it'll remove all functions even if there are
                references to them created by :meth:`get_func_ref`.
        """
        for f in self.funcs_user[:]:
            self.remove_func(f, force=force)

    def make_func(self, state: dict, instance: 'FuncBase' = None,
                  clone: bool = False) -> 'FuncBase':
        """Instantiates the function from the state and returns it.

        This method must be used to instantiate a function from state.
        See :mod:`ceed.function` for details and an example.

        :param state: The state dict representing the function as returned by
            :meth:`FuncBase.get_state`.
        :param instance: If None, a function instance of the type specified
            in ``state`` will be created and state will applied to it.
            Otherwise, it is applied to the given instance, which must be of
            the correct class.
        :param clone: If False, only user customizable properties of the
            function will be set, otherwise, all properties from state are
            applied to the function. Clone is meant to be an complete
            re-instantiation of the function.
        :return: The function instance created.
        """
        state = dict(state)
        c = state.pop('cls')
        if c == 'CeedFuncRef':
            cls = CeedFuncRef
        else:
            cls = self.get(c)

        if cls is None:
            raise Exception('Missing class "{}"'.format(c))
        assert instance is None or instance.__class__ is cls

        func = instance or cls(function_factory=self)
        func.apply_state(state, clone=clone)
        if c == 'CeedFuncRef':
            self._ref_funcs[func.func] += 1
            func.func.has_ref = True
        return func

    def save_functions(self) -> List[dict]:
        """Returns a dict representation of all the functions added with
        :meth:`add_func`.

        It is a list of dicts where each item is the
        :meth:`FuncBase.get_state` of the corresponding function in
        :attr:`user_funcs`.
        """
        return [f.get_state(recurse=True, expand_ref=False)
                for f in self.funcs_user]

    def recover_funcs(self, function_states: List[dict]) -> \
            Tuple[List['FuncBase'], Dict[str, str]]:
        """Takes a list of function states such as returned by
        :meth:`save_functions` and instantiates the functions represented by
        the states and adds (:meth:`add_func`) the functions to the factory.

        :param function_states: List of function states.
        :return: A tuple ``funcs, name_map``, ``funcs`` is the list of
            functions, ``name_map`` is a map from the original function's name
            to the new name given to the function (in case a function with that
            name already existed).
        """
        name_map = {}
        funcs = []
        # first create all the root functions so that we can get their new name
        # in case the previous function name already exists. These functions
        # are the global functions registered
        for state in function_states:
            # cannot be a ref func here because they are global funcs
            c = state['cls']
            assert c != 'CeedFuncRef'

            cls: FuncBase = self.get(c)
            if cls is None:
                raise Exception('Missing class "{}"'.format(c))

            func = cls(function_factory=self)
            old_name = func.name = state['name']

            self.add_func(func)
            funcs.append(func)
            state['name'] = name_map[old_name] = func.name

        # now the root function have been added and we know their old/name
        # names. Go through the states and update every reference function to
        # that pointed to an old function name, to point to the new name so
        # they accurately reference the new functions
        update_key_if_other_key(
            function_states, 'cls', 'CeedFuncRef', 'ref_name', name_map)

        # now apply the states to the new functions
        for func, state in zip(funcs, function_states):
            self.make_func(state, instance=func)

        return funcs, name_map


class FuncBase(EventDispatcher):
    """The base class for all functions.

    See :mod:`ceed.function` for details.

    When writing a plugin with new functions, you probably want to inherit
    from :class:`CeedFunc` because it provides some convenience methods
    (calling the function will check if it's in the valid domain). See e.g.
    :class:`ceed.function.plugin.LinearFunc` for an example.

    :events:

        `on_changed`:
            Triggered whenever a configurable property (i.e. it is returned as
            key in the :meth:`get_state` dict) of this instance is changed.
    """

    name = StringProperty('Abstract')
    '''The name of the function instance. The name must be unique within a
    :attr:`FunctionFactoryBase` once it is added
    (:meth:`FunctionFactoryBase.add_func`) to the
    :attr:`FunctionFactoryBase`, otherwise it's automatically renamed.
    '''

    description = StringProperty('')
    '''A description of the function. This is shown to the user and should
    describe the function.
    '''

    icon = StringProperty('')
    '''The function icon. Not used currently.
    '''

    duration: FloatOrInt = NumericProperty(0.)
    '''How long after the start of the function until the function is complete
    and continues to the next :attr:`loop` or is done.

    -1 means go on forever and it never finishes (except if manually finished).
    This could be used e.g. if waiting for some external interrupt - set
    duration to -1, and only finish when the interrupt happens.

    For :class:`FuncGroup` this is automatically computed as the sum of all the
    sub-function duration. If any of them are negative, this will also be
    negative.

    See :mod:`ceed.function` for more details.

    The value is in :meth:`get_timebase` units.
    '''

    duration_min: FloatOrInt = NumericProperty(0.)
    '''Same as :attr:`duration`, except it excludes any infinite portions of
    the function duration.

    I.e. any sub function whose duration is negative will be read as zero. This
    gives the estimated minimum duration not including any infinite portions
    of a single loop iteration.

    See :mod:`ceed.function` for more details.

    The value is in :meth:`get_timebase` units and is read only.
    '''

    duration_min_total: FloatOrInt = NumericProperty(0)
    '''The total duration of the function including all the loops, excluding
    any infinite portions of the function duration.

    Similar to :attr:`duration_min`, except it includes all the loops. So e.g.
    if :attr:`duration_min` is ``5`` and :attr:`loop` is ``2``,
    :attr:`duration_min_total` would typically be ``10``.

    See :mod:`ceed.function` for more details.

    The value is in :meth:`get_timebase` units and is read only.
    '''

    loop: int = NumericProperty(1)
    '''The number of times the function loops through before it is considered
     done.

    At the end of each loop :attr:`loop_count` is incremented until done,
    starting from zero.

    See :mod:`ceed.function` for more details.
    '''

    parent_func: Optional['FuncBase'] = None
    '''If this function is the child of another function, e.g. it's a
    sub-function of a :class:`FuncGroup` instance, then :attr:`parent_func`
    points to the parent function.
    '''

    has_ref: bool = BooleanProperty(False)
    """Whether there's a CeedFuncRef pointing to this function. If True,
    the function should not be deleted from the function factory that holds it.

    This is automatically set by :meth:`FunctionFactoryBase.get_func_ref`
    and :meth:`FunctionFactoryBase.return_func_ref`.
    """

    display = None
    """The widget that visualize this function, if any.
    """

    _clone_props: Set[str] = {'cls', 'name'}
    '''Set of non user-customizable property names that are specific to the
    function instance and should not be copied when duplicating the function.
    They are only copied when a function is cloned, i.e. when it is created
    from state to be identical to the original.
    '''

    function_factory: FunctionFactoryBase = None
    """
    The :class:`FunctionFactoryBase` instance with which this function is
    associated. This should be set by whoever creates the function by passing
    it to the constructor.
    """

    timebase_numerator: FloatOrInt = NumericProperty(0)
    '''The numerator of the timebase. See :attr:`timebase`.
    '''

    timebase_denominator: FloatOrInt = NumericProperty(1)
    '''The denominator of the timebase. See :attr:`timebase`.
    '''

    def _get_timebase(self) -> Union[float, Fraction]:
        num = self.timebase_numerator
        if isinstance(num, float) and num.is_integer():
            num = int(num)
        denom = self.timebase_denominator
        if isinstance(denom, float) and denom.is_integer():
            denom = int(denom)

        if isinstance(num, int) and isinstance(denom, int):
            return Fraction(num, denom)
        else:
            return self.timebase_numerator / float(self.timebase_denominator)

    timebase: Union[float, Fraction] = AliasProperty(
        _get_timebase, None, cache=True,
        bind=('timebase_numerator', 'timebase_denominator'))
    '''The (read-only) timebase scale factor as computed by
    :attr:`timebase_numerator` ``/`` :attr:`timebase_denominator`. It returns
    either a float, or a Fraction instance when the numerator and
    denominator are ints.

    To set, :attr:`timebase_numerator` and :attr:`timebase_denominator` must
    be set individually. To use, call :meth:`get_timebase`, rather than
    using :attr:`timebase` directly.

    The timebase is the scaling factor by which some function properties that
    relate to time, e.g. :attr:`duration`, are multiplied to convert from
    timebase units to time.

    By default :attr:`timebase_numerator` is 0 and
    :attr:`timebase_denominator` is 1 which makes :attr:`timebase` 0
    indicating that the timebase used is given by the parent function or is 1.
    When :attr:`timebase` is not 0 this :attr:`timebase` is used instead.

    See :mod:`ceed.function` and :meth:`get_timebase` for more details.

    .. note::

        This property is dispatched whenever the value returned by
        :meth:`get_timebase` would change, even if :attr:`timebase` didn't
        change. E.g. when a parent's timebase changes.
    '''

    t_start: NumFraction = 0
    '''The global time offset subtracted from the time passed to the function.

    This is the global time with which the function or loop was initialized,
    :meth:`get_relative_time` removes it to get to local time.
    The value is in seconds. See :mod:`ceed.function` for more details.

    Don't set directly, it is set with :meth:`init_func` and
    :meth:`init_loop_iteration`.
    '''

    t_end: NumFraction = 0
    """The time at which the loop or function ends in global timebase.

    Set by the function after each loop is done (i.e.
    :meth:`CeedFunc.is_loop_done` returned True) and is typically the second
    value from :meth:`get_domain`, or the current time value if that is
    negative.
    """

    loop_count: int = 0
    '''The current loop iteration.

    This goes from zero (reset by :meth:`init_func` /
    :meth:`init_loop_iteration`) to :attr:`loop`. The function is done when
    it is exactly :attr:`loop`, having looped :attr:`times`.

    See also :mod:`ceed.function`.
    '''

    loop_tree_count: int = 0
    """The current iteration, starting from zero, and incremented for each loop
    of the function, including outside loops that loop over the function.

    See :meth:`init_func_tree` for details.
    """

    noisy_parameters: Dict[str, NoiseBase] = DictProperty({})
    """A dict mapping parameter names of the function to
    :class:`~ceed.function.param_noise.NoiseBase` instances that indicate
    how the parameter should be sampled, when the parameter needs to be
    stochastic.

    Only parameters returned in :meth:`get_noise_supported_parameters` may
    have randomness associated with them.

    E.g.::

        >>> f = LinearFunc(function_factory=function_factory, duration=2, m=2)
        >>> UniformNoise = \
function_factory.param_noise_factory.get_cls('UniformNoise')
        >>> f.noisy_parameters['m'] = UniformNoise(min_val=10, max_val=20)
        >>> f.m
        2
        >>> f.b
        0.0
        >>> f.resample_parameters()
        >>> f.m
        12.902067284602595
        >>> f.resample_parameters()
        >>> f.m
        11.555420807597352
        >>> f.b
        0.0
    """

    noisy_parameter_samples: Dict[str, List[float]] = {}
    """For each parameter listed in :attr:`noisy_parameters`, if
    :attr:`~ceed.function.param_noise.NoiseBase.sample_each_loop`, then during
    the :meth:`resample_parameters` stage we pre-compute the values for the
    parameter for each loop iteration and store it here.

    The pre-computed values are then used to update the parameter during each
    :meth:`init_func` and :meth:`init_loop_iteration`. The total number of
    samples includes all outside loops, see
    :attr:`~ceed.function.param_noise.NoiseBase.sample_each_loop` and
    :meth:`init_func_tree`.
    """

    __events__ = ('on_changed', )

    def __init__(self, function_factory, **kwargs):
        self.function_factory = function_factory
        self.noisy_parameter_samples = {}
        super(FuncBase, self).__init__(**kwargs)
        for prop in self.get_state(recurse=False):
            self.fbind(prop, self.dispatch, 'on_changed', prop)

        self.fbind('duration', self._update_duration_min)
        self.fbind('duration_min', self._update_total_duration)
        self.fbind('loop', self._update_total_duration)
        self._update_duration_min()
        self._update_total_duration()

    def __call__(self, t: NumFraction) -> float:
        raise NotImplementedError

    def get_timebase(self) -> Union[float, Fraction]:
        """Returns the function's timebase.

        If :attr:`timebase_numerator` and :attr:`timebase` is 0, it returns the
        timebase of its :attr:`parent_func` with :meth:`get_timebase` if it has
        a parent. If it doesn't have a parent, it return 1.
        Otherwise, it returns :attr:`timebase`.
        """
        if not self.timebase_numerator:
            if self.parent_func:
                return self.parent_func.get_timebase()
            return 1.
        return self.timebase

    def on_changed(self, *largs, **kwargs):
        pass

    def get_gui_props(self):
        """Called internally by the GUI to get the properties of the function
        that should be displayed to the user to be customized.

        :returns:

            A dict that contains all properties that should be displayed. The
            values of the property is as follows:

            * If it's the string int, float, str or it's the python type
              int, float, or str then the GUI will show a editable property
              for this type.
            * If it's None, we look at the value of the property
              in the instance and display accordingly (e.g. if it's a str type
              property, a string property is displayed to the user).

              .. note::

                The default value determines the type. So if the default value
                is ``0``, the type will be int and a user won't be able to
                enter a float. Use e.g. ``0.0`` in the latter case.

        E.g.::

            >>> Cos = function_factory.get('CosFunc')
            >>> cos = Cos()
            >>> cos.get_gui_props()
            {'A': None,
             'duration': None,
             'f': None,
             'loop': None,
             'name': None,
             't_offset': None,
             'th0': None}
        """
        properties = {
            'name': None, 'loop': None, 'timebase_numerator': None,
            'timebase_denominator': None}
        return properties

    def get_prop_pretty_name(self):
        """Called internally by the GUI to get a translation dictionary which
        converts property names as used in :meth:`get_state` into nicer
        property names used to display the properties to the user for
        customization. E.g. "timebase_numerator" may be displayed as
        "TB num" as that is more concise.

        :returns:

            A dict that contains all properties whose names should be changed
            when displayed. Keys in the dict are the names as returned by
            :meth:`get_state`, the values are the names that should be
            displayed instead. If a property is not included it's
            original property name is used instead.

        E.g.::

            >>> f = LinearFunc(function_factory=function_factory)
            {'timebase_numerator': 'TB num', \
'timebase_denominator': 'TB denom'}
        """
        trans = {
            'timebase_numerator': 'TB num',
            'timebase_denominator': 'TB denom'}
        return trans

    def get_gui_elements(self):
        """Returns widget instances that should be displayed to the user along
        with this function's editable properties of :meth:`get_gui_props`.

        These widgets are displayed along with other config
        parameters for the function and can be used for custom config options.

        :returns:

            A list that contains all Kivy widget instances to be displayed.

        E.g.::

            >>> Cos = function_factory.get('CosFunc')
            >>> cos = Cos()
            >>> cos.get_gui_elements()
            []
        """
        items = []
        return items

    def get_noise_supported_parameters(self) -> Set[str]:
        """Returns the set of property names of this function that supports
        randomness and may have an
        :class:`ceed.function.param_noise.NoiseBase` instance associated with
        it.
        """
        return {'duration'}

    def get_state(self, recurse=True, expand_ref=False) -> Dict:
        """Returns a dict representation of the function so that it can be
        reconstructed later with :meth:`apply_state`.

        :Params:

            `recurse`: bool
                When the function has children functions, e.g. a
                :class:`FuncGroup`, if True all the children functions'
                states will also be returned, otherwise, only this function's
                state is returned. See the example.
            `expand_ref`: bool
                If True, if any sub-functions (or this function itself) are
                :class:`CeedFuncRef` instances, they will be expanded to
                contain the state of the actual underlying function. Otherwise,
                it is returned as being a function reference.

        :returns:

            A dict with all the configuration data.

        E.g.::

            >>> Cos = function_factory.get('CosFunc')
            >>> cos = Cos()
            >>> cos.get_state()
            {'A': 1.0,
             'cls': 'CosFunc',
             'duration': 0,
             'f': 1.0,
             'loop': 1,
             'name': 'Cos',
             't_offset': 0,
             'th0': 0.0}
            >>> Group = function_factory.get('FuncGroup')
            >>> g = Group()
            >>> g
            <ceed.function.FuncGroup at 0x4f85800>
            >>> g.add_func(cos)
            >>> g.get_state(recurse=True)
            {'cls': 'FuncGroup',
             'funcs': [{'A': 1.0,
               'cls': 'CosFunc',
               'duration': 0,
               'f': 1.0,
               'loop': 1,
               'name': 'Cos',
               't_offset': 0,
               'th0': 0.0}],
             'loop': 1,
             'name': 'Group'}
            >>> g.get_state(recurse=False)
            {'cls': 'FuncGroup',
             'loop': 1,
             'name': 'Group'}
        """
        d = {
            'name': self.name, 'cls': self.__class__.__name__,
            'loop': self.loop,
            'timebase_numerator': self.timebase_numerator,
            'timebase_denominator': self.timebase_denominator,
            'noisy_parameters':
                {k: v.get_config() for k, v in self.noisy_parameters.items()},
            'noisy_parameter_samples': self.noisy_parameter_samples,
        }
        return d

    def apply_state(self, state: Dict, clone=False):
        """Takes the state of the function saved with :meth:`get_state` and
        applies it to this function. it also creates any children function e.g.
        in the case it is a :class:`FuncGroup` etc.

        It is called internally and should not be used directly. Use
        :meth:`FunctionFactoryBase.make_func` instead.

        :Params:

            `state`: dict
                The dict to use to reconstruct the function as returned by
                :meth:`get_state`.
            `clone`: bool
                If True will apply all the state exactly as in the original
                function, otherwise it doesn't apply internal parameters
                listed in :attr:`_clone_props` that are not user customizable.

        See :mod:`ceed.function` for an example.
        """
        p = self._clone_props
        for k, v in state.items():
            if k == 'noisy_parameters':
                noisy_parameters = {}
                noise_factory = self.function_factory.param_noise_factory

                for param, config in v.items():
                    noisy_parameters[param] = noise_factory.make_instance(
                        config)

                self.noisy_parameters = noisy_parameters
            elif (clone or k not in p) and k != 'cls':
                setattr(self, k, v)

    def get_funcs(
            self, step_into_ref=True
    ) -> Generator[CeedFuncOrRefInstance, None, None]:
        """Generator that yields the function and all its children functions
        if it has any, e.g. for :class:`FuncGroup`. It's in DFS order.

        :Params:

            `step_into_ref`: bool
                If True, when it encounters a :class:`CeedFuncRef` instance
                it'll step into it and return that function and its children.
                Otherwise, it'll just return the :class:`CeedFuncRef` and not
                step into it.

        E.g.::

            >>> Cos = function_factory.get('CosFunc')
            >>> cos = Cos()
            >>> Group = function_factory.get('FuncGroup')
            >>> g = Group()
            >>> g.add_func(cos)
            >>> [f for f in cos.get_funcs()]
            [<ceed.function.plugin.CosFunc at 0x4d71c78>]
            >>> [f for f in g.get_funcs()]
            [<ceed.function.FuncGroup at 0x4f85800>,
             <ceed.function.plugin.CosFunc at 0x4d71c78>]
        """
        yield self

    def can_other_func_be_added(
            self, other_func: Union['CeedFuncRef', 'FuncBase']) -> bool:
        """Checks whether the other function may be added to this function.

        Specifically, it checks whether this function is a child of the other
        function, and if so we shouldn't allow the other function to be added
        to this function as it'll create a circular tree.

        :param other_func: Another :class:`FuncBase` instance.
        :return: Whether it is safe to add the other function as a child of
            this function.
        """
        if isinstance(other_func, CeedFuncRef):
            other_func = other_func.func

        # check if we (or a ref to us) are a child of other_func
        for func in other_func.get_funcs(step_into_ref=True):
            if func is self:
                return False
        return True

    def __deepcopy__(self, memo):
        obj = self.__class__(function_factory=self.function_factory)
        obj.apply_state(deepcopy(self.get_state()))
        return obj

    def copy_expand_ref(self) -> 'FuncBase':
        """Copies this function and all its sub-functions.

        If any of them are :class:`CeedFuncRef` instances, they are replaced
        with a copy of their original function.

        :return: A copy of this function, with all :class:`CeedFuncRef`
            instances replaced by their original normal function.
        """
        obj = self.__class__(function_factory=self.function_factory)
        obj.apply_state(self.get_state(expand_ref=True))
        return obj

    def init_func_tree(self, root: Optional['FuncBase'] = None) -> None:
        """Initializes the function as part of the function tree so it is ready
        to be called to get the function values as part of the tree. It is
        called once for each function of the entire function tree.

        :param root: The root of the function tree. If None, it's self.

        For example, for the following function structure::

            GroupFunc:
                name: 'root'
                loop: 5
                GroupFunc:
                    name: 'child_a'
                    loop: 3
                    ConstFunc:
                        name: 'child'
                        loop: 4

        when the experiment is ready, after :meth:`resample_parameters` and the
        stage is ready to run, ceed will call :meth:`init_func_tree` once for
        ``root``, ``child_a``, and ``child`` in that order. The ``root``
        parameter passed will be the ``root`` function.

        Then, it will call :meth:`init_func` for ``root``, ``child_a``, and
        ``child`` 1, 5, and 15 times, respectively. Once for each time the
        function is started.

        Finally, it will call :meth:`init_loop_iteration` for ``root``,
        ``child_a``, and ``child`` 4, 10, and 45 times, respectively. Once for
        each loop iteration of the function, except the first.
        """
        self.loop_tree_count = 0

    def init_func(self, t_start: NumFraction) -> None:
        """Initializes the function so it is ready to be called to get
        the function values. See also :meth:`init_func_tree`.

        :param t_start: The time in seconds in global time. :attr:`t_start`
            will be set to this value. All subsequent calls to the
            function with a time value will be relative to this given time.
        """
        self.t_start = t_start
        self.loop_count = 0

        loop_tree_count = self.loop_tree_count
        for key, values in self.noisy_parameter_samples.items():
            setattr(self, key, values[loop_tree_count])

    def init_loop_iteration(self, t_start: NumFraction) -> None:
        """Initializes the function at the beginning of each loop.

        It's called internally at the start of every :attr:`loop` iteration,
        **except the first**. See also :meth:`init_func_tree`.

        :param t_start: The time in seconds in global time. :attr:`t_start`
            will be set to this value. All subsequent calls to the
            function with a time value will be relative to this given time.
        """
        self.t_start = t_start

        loop_tree_count = self.loop_tree_count
        for key, values in self.noisy_parameter_samples.items():
            setattr(self, key, values[loop_tree_count])

    def get_domain(
            self, current_iteration: bool = True
    ) -> Tuple[NumFraction, NumFraction]:
        """Returns the current domain of the function.

        :param current_iteration: If True, returns the domain for the current
            loop iteration (i.e. the end point is the expected end time of the
            current loop). Otherwise, returns the domain ending at the end of
            the final loop iteration. In both cases, the interval start is the
            time the current loop iteration started, in **global time**.

        See :mod:`ceed.function` for details.
        """
        duration = self.duration
        # we need to be able to carry through fractions, so ints should stay
        if isinstance(duration, float) and duration.is_integer():
            duration = int(duration)

        if self.loop_count >= self.loop:
            return -1, -1

        if duration < 0:
            return self.t_start, -1

        if current_iteration:
            return self.t_start, self.t_start + duration * self.get_timebase()

        return self.t_start, \
            self.t_start + (self.loop - self.loop_count) * \
            duration * self.get_timebase()

    def get_relative_time(self, t: NumFraction) -> NumFraction:
        """Converts the global time to the local function time.

        At each time-step, the function is called with the global time. This
        function converts the time into local time, relative to the
        :attr:`t_start`. Also adds any function specific offset (e.g.
        :attr:`CeedFunc.t_offset`.
        """
        t = t - self.t_start
        if t < 0 and isclose(t, 0):
            t = 0
        return t

    def tick_loop(self, t: NumFraction) -> bool:
        """Increments :attr:`loop_count` and returns whether the function is
        done, which is when all the :attr:`loop` iterations are done and
        :attr:`loop_count` reached :attr:`loop`.

        ``t`` is in seconds in global time and this is only called
        when the function time reached the end of its valid domain so that it
        makes sense to increment the loop. I.e. if the user called the function
        with a time past the current loop duration, this is called internally
        to increment the loop.

        May not be called with time values smaller than the domain. Or if the
        loop iterations are already done.

        :param t: The time at which the last function in the tree or loop
            iteration of this function ended, in global time.
        :return: True if it ticked the loop, otherwise False if we cannot tick
            because we hit the max and the function is done.
        """
        if t < self.t_start and not isclose(t, self.t_start):
            raise ValueError(
                f'Called with time {t} that is less than start '
                f'time {self.t_start}')
        if self.loop_count >= self.loop:
            raise ValueError(
                f'Called after the current loop iteration ({self.loop_count}) '
                f'reached the end of all loops ({self.loop})')

        self.loop_count += 1
        self.loop_tree_count += 1

        if self.loop_count >= self.loop:
            return False

        self.init_loop_iteration(t)
        return True

    def _update_duration_min(self, *largs):
        """Automatically called when :attr:`duration` changes to update
        parameters that need to be updated.
        """
        self.duration_min = 0 if self.duration < 0 else self.duration

    def _update_total_duration(self, *largs):
        """Automatically called when :attr:`duration_min` or
        :attr:`loop` changes to update parameters that need to be updated.
        """
        self.duration_min_total = self.loop * self.duration_min

    def resample_parameters(
            self, parent_tree: Optional[List['FuncBase']] = None,
            is_forked=False, base_loops: int = 1) -> None:
        """Resamples all the function parameters that have randomness
        attached to it in :attr:`noisy_parameters` and updates their values.

        For all parameters that are randomized and
        :attr:`~ceed.function.param_noise.NoiseBase.sample_each_loop` is True,
        the samples for all the iterations are also pre-sampled and stored in
        :attr:`noisy_parameter_samples`.

        If ``is_forked``, then if
        :attr:`~ceed.function.param_noise.NoiseBase.lock_after_forked`, then
        it won't be re-sampled. This allows sampling a :class:`CeedFuncRef`
        function before forking it, and then, when copying and forking the
        source functions into individual functions we don't resample them.
        Then all these individual functions share the same random parameters as
        the original referenced function.

        ``parent_tree`` is not inclusive.

        ``base_loops`` indicates the expected number of times the function
        is expected to loop due to the stage containing the function (if any).
        This is in addition to :attr:`loop` of the function and its parent
        tree. So e.g. if :attr:`loop` is ``3`` and ``base_loops`` is ``2`` with
        no parents, then the function will be looped ``6`` times, twice by the
        stage.

        E.g.::

            >>> # get the classes
            >>> function_factory = FunctionFactoryBase()
            >>> register_all_functions(function_factory)
            >>> LinearFunc = function_factory.get('LinearFunc')
            >>> UniformNoise = function_factory.param_noise_factory.get_cls(
            ...     'UniformNoise')
            >>> # create function and add noise to parameters
            >>> f = LinearFunc(function_factory=function_factory)
            >>> f.noisy_parameters['m'] = UniformNoise()
            >>> f.noisy_parameters['b'] = UniformNoise(lock_after_forked=True)
            >>> # now add it to factory and create references to it
            >>> function_factory.add_func(f)
            >>> ref1 = function_factory.get_func_ref(func=f)
            >>> ref2 = function_factory.get_func_ref(func=f)
            >>> # resample the original function and fork refs into copies
            >>> f.resample_parameters()
            >>> f1 = ref1.copy_expand_ref()
            >>> f2 = ref2.copy_expand_ref()
            >>> # now resample only those that are not locked
            >>> f1.resample_parameters(is_forked=True)
            >>> f2.resample_parameters(is_forked=True)
            >>> # b is locked to pre-forked value and is not sampled after fork
            >>> f.m, f.b
            (0.22856343565686332, 0.3092686616300213)
            >>> f1.m, f1.b
            (0.05228392113705038, 0.3092686616300213)
            >>> f2.m, f2.b
            (0.9117196772532972, 0.3092686616300213)
        """
        samples = self.noisy_parameter_samples
        for key in samples.keys() - self.noisy_parameters.keys():
            del samples[key]

        for key, value in self.noisy_parameters.items():
            if value.lock_after_forked and is_forked:
                continue

            if value.sample_each_loop:
                n = 1
                if parent_tree:
                    n = reduce(operator.mul, (f.loop for f in parent_tree))
                n *= self.loop * base_loops
                if not n:
                    if key in samples:
                        del samples[key]
                    continue

                values = samples[key] = value.sample_seq(n)
                setattr(self, key, values[0])
            else:
                setattr(self, key, value.sample())
                if key in samples:
                    del samples[key]


class CeedFuncRef:
    """A function that refers to another function.

    See :meth:`FunctionFactoryBase.get_func_ref` and :mod:`ceed.function` for
    details.
    """

    func = None
    """The original :class:`FuncBase` this reference function is referring to.
    """

    display = None
    """Same as :attr:`FuncBase.display`.
    """

    parent_func = None
    """Same as :attr:`FuncBase.parent_func`.
    """

    function_factory = None
    """Same as :attr:`FuncBase.function_factory`.
    """

    def __init__(self, function_factory, func=None):
        super(CeedFuncRef, self).__init__()
        self.func = func
        self.function_factory = function_factory

    def get_state(self, recurse=True, expand_ref=False):
        if expand_ref:
            return self.func.get_state(recurse=recurse, expand_ref=True)

        state = {'ref_name': self.func.name, 'cls': 'CeedFuncRef'}
        return state

    def apply_state(self, state, clone=False):
        self.func = self.function_factory.funcs_inst[state['ref_name']]

    def __deepcopy__(self, memo):
        assert self.__class__ is CeedFuncRef
        return self.function_factory.get_func_ref(func=self.func)

    def copy_expand_ref(self):
        return self.func.copy_expand_ref()

    def __call__(self, t: NumFraction) -> float:
        raise TypeError(
            'A CeedFuncRef function instance cannot be called like a normal '
            'function. To use, copy it into a normal function with '
            'copy_expand_ref')

    def init_func_tree(self, root: Optional['FuncBase'] = None):
        raise TypeError(
            'A CeedFuncRef function instance cannot be called like a normal '
            'function. To use, copy it into a normal function with '
            'copy_expand_ref')

    def init_func(self, t_start):
        raise TypeError(
            'A CeedFuncRef function instance cannot be called like a normal '
            'function. To use, copy it into a normal function with '
            'copy_expand_ref')


class CeedFunc(FuncBase):
    """A base class for typical Ceed functions.

    See :mod:`ceed.function.plugin` for example functions that are based on
    this class.
    """

    t_offset = NumericProperty(0.)
    '''The amount of time in seconds to add the function time when computing
    the result. It allows some additional control over the function.

    All functions that inherit from this class must add this time. E.g. the
    :class:`~ceed.function.plugin.LinearFunc` defines its function as
    ``y(t) = mt + b`` with time ``t = (t_in - t_start + t_offset)``.

    The :attr:`duration` of the function is not affected by this property as
    it is independent of this. I.e. we check whether a time value is in the
    function's :meth:`get_domain` ignoring :attr:`t_offset` but we then
    add it to the given time before computing the function's output.
    '''

    def __call__(self, t: NumFraction) -> float:
        if t < self.t_start and not isclose(t, self.t_start):
            raise ValueError(
                'Cannot call function {} with time {} less than the '
                'function start {}'.format(self, t, self.t_start))
        if self.loop_count >= self.loop:
            raise FuncDoneException

        while True:
            if not self.is_loop_done(t):
                return 0

            # save end time of the current loop iteration
            t_start, t_end = self.get_domain()
            assert t_start != t_end or t_start != -1
            if t_end < 0:
                t_end = t
            self.t_end = t_end

            if not self.tick_loop(t_end):
                break

        raise FuncDoneException

    def get_relative_time(self, t: NumFraction) -> NumFraction:
        t = super().get_relative_time(t)
        return t + self.t_offset

    def is_loop_done(self, t: NumFraction) -> bool:
        """Whether the time ``t``, in global time, is after the end of the
        current loop iteration.

        :param t: Time in seconds, in global time.
        :return: Whether it's past the end of the current loop iteration.
        """
        if self.duration < 0:
            return False

        duration = self.duration * self.get_timebase()
        elapsed = t - self.t_start
        return elapsed >= duration or isclose(elapsed, duration)

    def get_gui_props(self):
        d = super(CeedFunc, self).get_gui_props()
        d.update({'duration': None})
        d.update({'t_offset': None})
        return d

    def get_state(self, recurse=True, expand_ref=False):
        d = super(CeedFunc, self).get_state(recurse, expand_ref)
        d.update({'duration': self.duration})
        d.update({'t_offset': self.t_offset})
        return d


class FuncGroup(FuncBase):
    """Function that represents a sequence of sub-functions.

    See :mod:`ceed.function` for more details.

    When the function instance is called it goes through all its sub-functions
    sequentially until they are done. E.g.::

        >>> Group = function_factory.get('FuncGroup')
        >>> Const = function_factory.get('ConstFunc')
        >>> g = Group()
        >>> g.add_func(Const(a=1, duration=5))
        >>> g.add_func(Const(a=2, duration=3))
        >>> g.add_func(Const(a=3, duration=2))
        >>> g.init_func(0)
        >>> g(0)
        1
        >>> g(5)
        2
        >>> g(6)
        2
        >>> g(8)
        3
        >>> g(10)
         Traceback (most recent call last):
             g(10)
           File "g:\\python\\libs\\ceed\\ceed\\function\\__init__.py", \
line 934, in __call__
             raise FuncDoneException
         FuncDoneException
    """

    funcs: List[CeedFuncOrRefInstance] = []
    '''The list of children functions of this function.
    '''

    _func_idx = 0

    def __init__(self, name='Group', **kwargs):
        super(FuncGroup, self).__init__(name=name, **kwargs)
        self.funcs = []
        self.fbind('timebase', self._dispatch_timebase)

    def init_func_tree(self, root: Optional['FuncBase'] = None) -> None:
        if root is None:
            root = self

        super().init_func_tree(root)
        for func in self.funcs:
            func.init_func_tree(root)

    def init_func(self, t_start: NumFraction) -> None:
        super(FuncGroup, self).init_func(t_start)
        self._func_idx = 0

        funcs = self.funcs
        if funcs:
            funcs[0].init_func(t_start)

    def init_loop_iteration(self, t_start: NumFraction) -> None:
        super(FuncGroup, self).init_loop_iteration(t_start)
        self._func_idx = 0

        funcs = self.funcs
        if funcs:
            funcs[0].init_func(t_start)

    def __call__(self, t: NumFraction) -> float:
        if t < self.t_start and not isclose(t, self.t_start):
            raise ValueError(
                'Cannot call function {} with time less than the '
                'function start {}'.format(self, self.t_start))

        if self.loop_count >= self.loop:
            raise FuncDoneException

        funcs = self.funcs
        if not funcs or self._func_idx >= len(self.funcs):
            raise FuncDoneException

        while True:
            try:
                return funcs[self._func_idx](t)
            except FuncDoneException:
                # the next function starts at the time the last function ends.
                # Except if it's infinite, then we start from current time
                end_t = funcs[self._func_idx].t_end
                assert end_t >= 0, "Should be concrete value"
                self._func_idx += 1

                if self._func_idx >= len(self.funcs):
                    if not self.tick_loop(end_t):
                        break
                else:
                    funcs[self._func_idx].init_func(end_t)

        # save end time of the end of last func
        self.t_end = funcs[-1].t_end
        assert self.t_end >= 0, "Should be concrete value"

        raise FuncDoneException

    def replace_ref_func_with_source(
            self, func_ref: CeedFuncRef) -> Tuple[FuncBase, int]:
        """Given the :class:`CeedFuncRef`, it'll locate it in :attr:`func` and
        replace it with the underlying function.

        The caller is responsible for returning the reference with
        :meth:`FunctionFactoryBase.return_func_ref`.

        :param func_ref: The :class:`CeedFuncRef` to replace in :attr:`func`.
        :return: A tuple of ``(func, i)`` where ``func`` is the
            :class:`FuncBase` that replaced the reference function. And ``i``
            is the index in :attr:`funcs` where it was found.

        """
        if not isinstance(func_ref, CeedFuncRef):
            raise ValueError('Function must be a CeedFuncRef')

        i = self.funcs.index(func_ref)
        self.remove_func(func_ref)
        func = deepcopy(func_ref.func)
        self.add_func(func, index=i)
        return func, i

    def add_func(self, func: CeedFuncOrRefInstance, after=None, index=None):
        """Adds ``func`` to this function as a sub-function in :attr:`funcs`.

        If calling this manually, remember to check
        :meth:`FuncBase.can_other_func_be_added` before adding if there's
        potential for it to return False.

        :Params:

            `func`: :class:`FuncBase`
                The function instance to add.
            `after`: :class:`FuncBase`, defaults to None.
                The function in :attr:`funcs` after which to add this function
                if specified.
            `index`: int, defaults to None.
                The index where to insert the function if specified.
        """
        func.parent_func = self

        if after is None and index is None:
            index = len(self.funcs)
            self.funcs.append(func)
        elif index is not None:
            self.funcs.insert(index, func)
        else:
            i = self.funcs.index(after)
            index = i + 1
            self.funcs.insert(i + 1, func)

        if isinstance(func, CeedFuncRef):
            func.func.fbind('duration', self._update_duration)
            func.func.fbind('timebase', self._update_duration)
            func.func.fbind('duration_min_total', self._update_duration)
        else:
            func.fbind('duration', self._update_duration)
            func.fbind('timebase', self._update_duration)
            func.fbind('duration_min_total', self._update_duration)

        self._update_duration()
        self.dispatch('on_changed', op='add', index=index)

    def remove_func(self, func):
        """Removes sub-function ``func`` from :attr:`funcs`. It must exist in
        :attr:`funcs`.

        :Params:

            `func`: :class:`FuncBase`
                The function instance to remove.
        """
        assert func.parent_func is self
        func.parent_func = None

        if isinstance(func, CeedFuncRef):
            func.func.funbind('duration', self._update_duration)
            func.func.funbind('timebase', self._update_duration)
            func.func.funbind('duration_min_total', self._update_duration)
        else:
            func.funbind('duration', self._update_duration)
            func.funbind('timebase', self._update_duration)
            func.funbind('duration_min_total', self._update_duration)

        index = self.funcs.index(func)
        del self.funcs[index]
        self._update_duration()
        self.dispatch('on_changed', op='remove', index=index)
        return True

    def _dispatch_timebase(self, *largs):
        # this is a O(N^2), but it's a simpler implementation
        for func in self.funcs[:]:
            if isinstance(func, CeedFuncRef):
                func = func.func

            if not func.timebase_numerator:
                func.property('timebase').dispatch(func)

    def _update_duration(self, *largs):
        """Computes duration as a function of its children.
        """
        infinite = False
        duration_min = 0
        for f in self.funcs:
            if isinstance(f, CeedFuncRef):
                f = f.func

            if f.duration < 0:
                infinite = True
            duration_min += f.duration_min_total * f.get_timebase()

        duration_min = float(duration_min / self.get_timebase())
        self.duration_min = duration_min
        self.duration = -1 if infinite else duration_min

    def _update_duration_min(self, *largs):
        # don't update when duration is updated because it's already computed
        pass

    def get_state(self, recurse=True, expand_ref=False):
        d = super(FuncGroup, self).get_state(recurse, expand_ref)
        d['funcs'] = funcs = []

        if recurse:
            for f in self.funcs:
                if isinstance(f, CeedFuncRef) and expand_ref:
                    state = f.func.get_state(recurse=recurse, expand_ref=True)
                else:
                    state = f.get_state(recurse=recurse, expand_ref=expand_ref)
                funcs.append(state)
        return d

    def apply_state(self, state={}, clone=False):
        for opts in state.get('funcs', []):
            self.add_func(self.function_factory.make_func(opts, clone=clone))

        super(FuncGroup, self).apply_state(
            {k: v for k, v in state.items() if k != 'funcs'}, clone=clone)

    def get_funcs(
            self, step_into_ref=True
    ) -> Generator[CeedFuncOrRefInstance, None, None]:
        yield self
        for func in self.funcs:
            if isinstance(func, CeedFuncRef):
                if not step_into_ref:
                    yield func
                    continue

                func = func.func
            for f in func.get_funcs(step_into_ref):
                yield f

    def resample_parameters(
            self, parent_tree: Optional[List['FuncBase']] = None,
            is_forked=False, base_loops: int = 1) -> None:
        super().resample_parameters(
            parent_tree=parent_tree, is_forked=is_forked,
            base_loops=base_loops)

        for func in self.funcs:
            if isinstance(func, CeedFuncRef):
                func = func.func

            tree = (parent_tree or []) + [self]
            func.resample_parameters(
                parent_tree=tree, is_forked=is_forked, base_loops=base_loops)


def register_all_functions(function_factory: FunctionFactoryBase):
    """Registers all the internal plugins and built-in functions and function
    distributions with the :class:`FunctionFactoryBase` and
    :attr:`FunctionFactoryBase.param_noise_factory` instance, respectively.

    It gets and registers all the plugins functions and function distributions
    under ``ceed/function/plugin`` using
    :func:`~ceed.function.plugin.get_plugin_functions`. See that function
    for how to make your plugin functions and distributions discoverable.

    :param function_factory: a :class:`FunctionFactoryBase` instance.
    """
    function_factory.register(FuncGroup)
    import ceed.function.plugin
    from ceed.function.plugin import get_plugin_functions
    package = 'ceed.function.plugin'

    functions, distributions, contents = get_plugin_functions(
        function_factory, base_package=package,
        root=dirname(ceed.function.plugin.__file__))
    for f in functions:
        function_factory.register(f)
    for d in distributions:
        function_factory.param_noise_factory.register_class(d)

    function_factory.add_plugin_source(package, contents)


def register_external_functions(
        function_factory: FunctionFactoryBase, package: str):
    """Registers all the plugin functions and function distributions in the
    package with the :class:`FunctionFactoryBase` and
    :attr:`FunctionFactoryBase.param_noise_factory` instance, respectively.

    See :func:`~ceed.function.plugin.get_plugin_functions`
    for how to make your plugin functions and distributions discoverable.

    Plugin source code files are copied to the data file when a a data file is
    created. However, it doesn't copy all files (i.e. it ignores non-python
    files) so it should be independently tracked for each experiment.

    :param function_factory: A :class:`FunctionFactoryBase` instance.
    :param package: The name of the package containing the plugins.
    """
    from ceed.function.plugin import get_plugin_functions
    m = importlib.import_module(package)

    functions, distributions, contents = get_plugin_functions(
        function_factory, base_package=package, root=dirname(m.__file__))
    for f in functions:
        function_factory.register(f)
    for d in distributions:
        function_factory.param_noise_factory.register_class(d)

    function_factory.add_plugin_source(package, contents)
