'''Function Interface
=========================

Defines the functions used with shapes to create time-varying intensities
of the shapes during an experiment.

A function is inherited from :class:`FuncBase` that defines the interface and
which returns a number when called with a time input. Example function usage
is::

    >>> function_factory = FunctionFactoryBase()
    >>> register_all_functions(function_factory)
    >>> CosFunc = function_factory.get('CosFunc')
    >>> f = CosFunc(duration=10, A=10, f=1)
    >>> f
    <ceed.function.plugin.CosFunc at 0x4eadba8>
    >>> f.init_func(0)
    >>> f(0)
    10.0
    >>> f(.25)
    6.123233995736766e-16
    >>> f(.5)
    -10.0
    >>> f(11)
       File "g:\\python\\libs\\ceed\\ceed\\function\\plugin\\__init__.py", \
line 134, in __call__
         raise FuncDoneException
     FuncDoneException

Function classes registered with the :class:`FunctionFactoryBase` used in the
GUI are available to the user in the GUI. During analysis, functions are
similarly registered with the :class:`FunctionFactoryBase` used in the
analysis.
'''

from copy import deepcopy
from collections import defaultdict
import inspect
from fractions import Fraction

from kivy.event import EventDispatcher
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, \
    ObjectProperty, DictProperty, AliasProperty
from kivy.logger import Logger
from kivy.factory import Factory

from ceed.utils import fix_name, update_key_if_other_key

__all__ = ('FuncDoneException', 'FunctionFactoryBase',
           'FuncBase', 'CeedFunc', 'FuncGroup', 'register_all_functions')


class FuncDoneException(Exception):
    '''Raised when the :class:`FuncBase` is called with a time value outside
    its valid time interval.
    '''
    pass


class FunctionFactoryBase(EventDispatcher):
    '''A global store of the defined :class:`FuncBase` classes and customized
    instances.

    Plugins register function classes with an instance of
    (:attr:`FunctionFactoryBase`) using :meth:`register` to make it available
    to the user in the GUI or for analysis.

    Similarly, function instances that has been customized with values or
    function groups customized with children functions are added with
    :meth:`add_func` making it available for usage in the GUI or analysis.

    To get a function class registered with :meth:`register`, e.g.
    :class:`ceed.function.plugin.CosFunc`::

        >>> cls = function_factory.get('CosFunc')  # using class name
        >>> cls
        ceed.function.plugin.CosFunc

    When registering a class, an instance is automatically created::

        >>> cls = function_factory.get('CosFunc')  # get registered class
        >>> f = function_factory.funcs_inst['Cos']  # get auto created instance
        >>> f
        <ceed.function.plugin.CosFunc at 0x53ada08>
        >>> f.f  # the default rate is 1Hz
        1.0
        >>> f2 = cls(f=100)  # new instance with 100Hz rate
        >>> f2.f
        100
        >>> f2.name  # the default name
        'Cos'
        >>> function_factory.add_func(f2)  # add it to the list
        >>> f2.name  # when adding it fixes the name so that it's unique
        'Cos-2'
        >>> function_factory.funcs_inst['Cos-2']
        <ceed.function.plugin.CosFunc at 0x5141938>

    The general usage of the registered instances with :meth:`add_func` is to
    get the instance and then copy and further customize it.

    :events:

        on_changed:
            The event is triggered every time a function is added or removed
            from the factory.
    '''

    __settings_attrs__ = ('timebase_numerator', 'timebase_denominator')

    __events__ = ('on_changed', )

    funcs_cls = {}
    '''Dict whose keys are the class names of the function classes registered
    with :meth:`register` and whose values are the corresponding classes.
    '''

    funcs_user = []
    '''List of the function instances registered with :meth:`add_func`. Does
    not include the instances automatically created and stored in
    :attr:`funcs_inst_default`.
    '''

    funcs_inst = DictProperty({})
    '''Dict whose keys are the function :attr:`name` and whose values are the
    corresponding function instances.

    Contains functions added with :meth:`add_func` as well as those
    automatically created and added when :meth:`register` is called on a class.
    '''

    funcs_inst_default = {}
    '''Dict whose keys are the function :attr:`name` and whose values are the
    corresponding function instances.

    Contains only the functions that are automatically created and added when
    :meth:`register` is called on a class.
    '''

    timebase_numerator = NumericProperty(1)
    '''The numerator of the default timebase. See :attr:`timebase`.
    '''

    timebase_denominator = NumericProperty(1)
    '''The denominator of the default timebase. See :attr:`timebase`.
    '''

    def _get_timebase(self):
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

    timebase = AliasProperty(
        _get_timebase, None, cache=True,
        bind=('timebase_numerator', 'timebase_denominator'))
    '''The default (read-only) timebase scale factor as computed by
    :attr:`timebase_numerator` / :attr:`timebase_denominator`. It returns
    either a float, or a Fraction instance if the numerator and
    denominator are both ints.

    The timebase is the scaling factor by which some function properties that
    relate to time, e.g. :attr:`FuncBase.duration`, are multiplied to convert
    from timebase units to time.

    This is the default timebase that is used for a function when 
    :attr:`FuncBase.timebase` is 0, otherwise each function uses its 
    :attr:`FuncBase.timebase`.
    '''

    _cls_inst_funcs = {}
    '''Mapping from a :class:`FuncBase` derived class to a default instance of
    that class.
    '''

    _ref_funcs = {}

    def __init__(self, **kwargs):
        super(FunctionFactoryBase, self).__init__(**kwargs)
        self.funcs_cls = {}
        self.funcs_user = []
        self.funcs_inst_default = {}
        self._cls_inst_funcs = {}
        self._ref_funcs = defaultdict(int)

    def on_changed(self, *largs, **kwargs):
        pass

    def get_func_ref(self, name=None, func=None):
        """If used, release must be called, even when restored automatically.
        """
        func = func or self.funcs_inst[name]
        if isinstance(func, CeedFuncRef):
            func = func.func

        ref = CeedFuncRef(function_factory=self, func=func)
        self._ref_funcs[func] += 1
        func.has_ref = True
        return ref

    def return_func_ref(self, func_ref):
        self._ref_funcs[func_ref.func] -= 1
        if not self._ref_funcs[func_ref.func]:
            func_ref.func.has_ref = False

    def register(self, cls, instance=None):
        '''Registers the class and adds it to :attr:`funcs_cls`. It also
        creates an instance (unless ``instance`` is provided) of the class that
        is added to :attr:`funcs_inst` and :attr:`funcs_inst_default`.

        This function is should be called to register a new class from a
        plugin.

        :Params:

            `cls`: subclass of :class:`FuncBase`
                The class to register.
            `instance`: instance of `cls`
                The instance of `cls` to use. If None, a default
                class instance, using the default :attr:`FuncBase.name` is
                stored. Defaults to None.
        '''
        name = cls.__name__
        funcs = self.funcs_cls
        if name in funcs:
            Logger.warn('"{}" is already a registered function'.format(name))
        funcs[name] = cls

        f = cls(function_factory=self) if instance is None else instance
        if f.function_factory is not self:
            raise ValueError('Instance function factory is set incorrectly')
        f.name = fix_name(f.name, self.funcs_inst)

        self.funcs_inst[f.name] = f
        self._cls_inst_funcs[cls] = f
        self.funcs_inst_default[f.name] = f
        self.dispatch('on_changed')

    def get(self, name):
        '''Returns the class with name ``name`` that was registered with
        :meth:`register`.

        :Params:

            `name`: str
                The name of the class (e.g. ``'ExpFunc'``).

        :returns:

            The class.
        '''
        funcs = self.funcs_cls
        if name not in funcs:
            return None
        return funcs[name]

    def get_names(self):
        '''Returns the class names of all classes registered with
        :meth:`register`.
        '''
        return list(self.funcs_cls.keys())

    def get_classes(self):
        '''Returns the classes registered with :meth:`register`.
        '''
        return list(self.funcs_cls.values())

    def add_func(self, func):
        '''Adds the function to :attr:`funcs_user` and :attr:`funcs_inst`,
        which makes it available in the GUI.

        If the :attr:`FuncBase.name` in :attr:`funcs_inst` already exists,
        :attr:`FuncBase.name` will be set to a unique name based on its
        original name. Once added until removed, anytime the function's
        :attr:`FuncBase.name` changes, if it clashes with an existing
        function's name, it is renamed.

        ``on_changed`` event will be dispatched.

        :Params:

            `func`: a :class:`FuncBase` derived instance.
                The function to add.
        '''
        func.name = fix_name(func.name, self.funcs_inst)

        func.fbind('name', self._track_func_name, func)
        self.funcs_inst[func.name] = func
        self.funcs_user.append(func)

        if func.function_factory is not self:
            raise ValueError('function factory is incorrect')
        self.dispatch('on_changed')

    def remove_func(self, func, force=False):
        '''Removes the function added with :meth:`add_func`.

        ``on_changed`` event will be dispatched.

        :Params:

            `func`: a :class:`FuncBase` derived instance.
                The function to remove.
        '''
        if not force and func in self._ref_funcs and self._ref_funcs[func]:
            assert self._ref_funcs[func] > 0
            return False

        func.funbind('name', self._track_func_name, func)
        del self.funcs_inst[func.name]
        # we cannot remove by equality check (maybe?)
        for i, f in enumerate(self.funcs_user):
            if f is func:
                del self.funcs_user[i]
                break
        else:
            raise ValueError('{} was not found in funcs_user'.format(func))

        self.dispatch('on_changed')
        return True

    def _track_func_name(self, func, *largs):
        '''Fixes the name of the function instances stored here to ensure it's
        unique.
        '''
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

    def clear_added_funcs(self, force=False):
        '''Removes all the functions registered with :meth:`add_func`.
        '''
        for f in self.funcs_user[:]:
            self.remove_func(f, force=force)

        if force:
            self._ref_funcs = defaultdict(int)

    def make_func(self, state, instance=None, clone=False):
        '''Instantiates the function from the state and returns it.
        '''
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
        return func

    def save_functions(self):
        return [f.get_state(recurse=True, expand_ref=False)
                for f in self.funcs_user]

    def recover_funcs(self, function_states):
        name_map = {}
        funcs = []
        for state in function_states:
            # cannot be a ref func here because they are global funcs
            c = state['cls']
            assert c != 'CeedFuncRef'

            cls = self.get(c)
            if cls is None:
                raise Exception('Missing class "{}"'.format(c))

            func = cls(function_factory=self)
            old_name = func.name = state['name']

            self.add_func(func)
            funcs.append(func)
            state['name'] = name_map[old_name] = func.name

        update_key_if_other_key(
            function_states, 'cls', 'CeedFuncRef', 'ref_name', name_map)

        for func, state in zip(funcs, function_states):
            self.make_func(state, instance=func)

        return funcs, name_map


class FuncBase(EventDispatcher):
    '''The base class for all functions.

    Plugin functions inherit from this class to create new functions.
    To use, the new function classes need to be registered with
    :meth:`FunctionFactoryBase.register`.

    Function usage is as follows, a function is initialized to a particular
    :attr:`t_start`, in seconds, with :meth:`init_func` before it can be used.
    This sets the interval where it's valid to
    ``[t_start, t_start + duration)``.

    When the function is called with a time point in that interval it'll return
    the value of the function there. Every function subtracts :attr:`t_start`
    from the time value so that effectively the function returns values
    in the domain of ``[0, duration)``. :attr:`t_start` exists so that
    functions can be stepped through successively using a global increasing
    clock, e.g. during an experiment.

    Similarly, functions multiply some time related properties by
    :attr:`timebase` before it's used as those are given in :attr:`timebase`
    units. This allows properties to be specified e.g. in microsecond integers
    for accuracy
    purposes. In that case time will be in microseconds and :attr:`timebase`
    will be ``1 / 1000000``. It especially allows for precision when comparing
    times because we can target specific video frames by e.g. setting
    the timebase to the frame rate, then we can alternate the functions
    intensity on each frame.

    When called, the function will check if it's outside the interval
    using :meth:`tick_loop`, which if it is done, will increment the
    :attr:`loop_count` until it reaches :attr:`loop` and then the
    function will raise :class:`FuncDoneException` to signify that. The
    experimental controller or the parent function in the case of a
    :class:`FuncGroup` will move on to the next function in the list.

    :events:

        `on_changed`:
            Triggered whenever a configurable property (i.e. it is returned as
            key in the :meth:`get_state` dict) of this instance is changed.
    '''

    name = StringProperty('Abstract')
    '''The name of the function instance. The name must be unique within a
    :attr:`FunctionFactoryBase` once it is added to the 
    :attr:`FunctionFactoryBase`, otherwise it's automatically renamed.
    '''

    description = StringProperty('')
    '''A description of the function. Typically shown to the user.
    '''

    icon = StringProperty('')
    '''The function icon. Not used currently.
    '''

    duration = NumericProperty(0.)
    '''How long, after the start of the function the function is valid.
    -1 means go on forever. See class for more details.

    Its domain is [:attr:`t_start`,
    :attr:`t_start` + :attr:`duration` * :meth:`get_timebase`) if not ``-1``,
    otherwise, it is [:attr:`t_start`, ...).

    The value is in timebase units.
    '''

    duration_total = NumericProperty(0)
    '''The total duration of the function including all the loops, excluding
    any functions whose :attr:`duration` is ``-1``.
    '''

    loop = NumericProperty(1)
    '''The number of times the function loops through. At the end of each loop
    :attr:`loop_count` is incremented until done. See class for more details.
    '''

    parent_func = None
    '''If this function is the child of another function, e.g. it's a
    sub-function of a :class:`FuncGroup` instance, then :attr:`parent_func`
    points to the parent function.
    '''

    has_ref = BooleanProperty(False)
    """Whether there's a CeedFuncRef pointing to this function.
    """

    display = None

    _clone_props = {'cls', 'name'}
    '''props that are specific to the function and should not be copied
    between different functions. They are only copied when a function is
    cloned, i.e. when it is created from state.
    '''

    function_factory = None
    """
    The function factory with which this function is associated.
    This should be set by whoever creates the function, or when added to
    the factory.
    """

    timebase_numerator = NumericProperty(0)
    '''The numerator of the timebase. See :attr:`timebase`.
    '''

    timebase_denominator = NumericProperty(1)
    '''The denominator of the timebase. See :attr:`timebase`.
    '''

    def _get_timebase(self):
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

    timebase = AliasProperty(
        _get_timebase, None, cache=True,
        bind=('timebase_numerator', 'timebase_denominator'))
    '''The (read-only) timebase scale factor as computed by
    :attr:`timebase_numerator` / :attr:`timebase_denominator`. It returns
    either a float or a Fraction instance when the numerator and
    denominator are ints.

    The timebase is the scaling factor by which some function properties that
    relate to time, e.g. :attr:`duration`, are multiplied to convert from
    timebase units to time.

    By default :attr:`timebase_numerator` is 0 and
    :attr:`timebase_denominator` is 1 which makes :attr:`timebase` 0
    indicating that the timebase used is given by
    :attr:`FunctionFactoryBase.timebase`. When :attr:`timebase` is not 0 this
    :attr:`timebase` is used instead.

    :meth:`get_timebase` returns the actual timebase used.

    See the class description for its usage.
    '''

    t_start = 0
    '''The time offset subtracted from function time (in seconds). See 
    the class description.
    '''

    loop_count = 0
    '''The current loop count. See :attr:`loop` and the class description.
    '''

    __events__ = ('on_changed', )

    def __init__(self, function_factory, **kwargs):
        self.function_factory = function_factory
        super(FuncBase, self).__init__(**kwargs)
        for prop in self.get_state(recurse=False):
            self.fbind(prop, self.dispatch, 'on_changed', prop)

        self.fbind('duration', self._update_total_duration)
        self.fbind('loop', self._update_total_duration)
        self._update_total_duration()

    def __call__(self, t):
        raise NotImplementedError

    def get_timebase(self):
        '''Returns the function's timebase.

        If :attr:`timebase_numerator` is 0 it returns the timebase of its
        :attr:`parent_func` with :meth:`get_timebase` if it has a parent.
        Otherwise, it returns :attr:`FunctionFactoryBase.timebase`.
        '''
        if not self.timebase_numerator:
            if self.parent_func:
                return self.parent_func.get_timebase()
            return self.function_factory.timebase
        return self.timebase

    def on_changed(self, *largs, **kwargs):
        pass

    def get_gui_props(self, properties=None):
        '''Called internally by the GUI to get the properties of the function
        that should be displayed.

        :Params:

            `attrs`: dict
                The dict to update with the property names to be displayed.
                If None, the default, a dict is created and returned.

        :returns:

            A dict that contains all properties that should be displayed. The
            values of the property is as follows:

            * If it's the string int, float, str or it's the python type
              int, float, or str then the GUI will show a editable property
              for this type.
            * If it's None, we look at the value of the property
              in the instance and display accordingly (e.g. if it's a str type
              property, a string property is displayed to the user).

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
        '''
        if properties is None:
            properties = {}
        properties['name'] = None
        properties['loop'] = None
        properties['timebase_numerator'] = None
        properties['timebase_denominator'] = None
        return properties

    def get_prop_pretty_name(self, trans=None):
        '''Called internally by the GUI to get a translation dictionary which
        converts property names as used in :meth:`get_state` into nicer
        property names used to display the properties to the user.

        :Params:

            `trans`: dict
                The dict to update with the translation names to be displayed.
                If None, the default, a dict is created and returned.

        :returns:

            A dict that contains all properties whose names should be changed
            when displayed. Keys in the dict are the names as returned by
            :meth:`get_state`, the values are the names that should be
            displayed instead. If a property is not included it's property
            name is used.

        E.g.::

            >>>
        '''
        if trans is None:
            trans = {}
        trans['timebase_numerator'] = 'TB num'
        trans['timebase_denominator'] = 'TB denom'
        return trans

    def get_gui_elements(self, items=None):
        '''Returns widget instances that should be displayed to the user along
        with this function's editable properties of :meth:`get_gui_props`.

        These widgets are displayed along with other config
        parameters for the function and can be used for custom config options.

        :Params:

            `items`: list
                The list to which the widget instances to be displayed are
                added. If None, the default, a list is created and returned.

        :returns:

            A list that contains all widget instances to be displayed.

        E.g.::

            >>> Cos = function_factory.get('CosFunc')
            >>> cos = Cos()
            >>> cos.get_gui_elements()
            []
        '''
        if items is None:
            items = []
        return items

    def get_state(self, state=None, recurse=True, expand_ref=False):
        '''Returns a dict representation of the function so that it can be
        reconstructed later with :meth:`apply_state`.

        :Params:

            `state`: dict
                A dict with the state, to which each subclass adds
                configuration items to be saved. If None, the default,
                a dict is created and returned.
            `recurse`: bool
                When the function has children functions, e.g. a
                :class:`FuncGroup`, if True all the children functions'
                states will also be returned, otherwise, only this function's
                state is returned. See the example.

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
        '''
        d = {
            'name': self.name, 'cls': self.__class__.__name__,
            'loop': self.loop,
            'timebase_numerator': self.timebase_numerator,
            'timebase_denominator': self.timebase_denominator}
        if state is None:
            state = d
        else:
            state.update(d)
        return state

    def apply_state(self, state, clone=False):
        '''Takes the state of the function saved with :meth:`get_state` and
        applies it to this function. it also creates any children function e.g.
        in the case of a :class:`FuncGroup` etc.

        It is called internally and should not be used directly. Use
        :meth:`make_func` instead.

        :Params:

            `state`: dict
                The dict to use to reconstruct the function as returned by
                :meth:`get_state`.
            `clone`: bool
                If True will apply all the state, otherwise it doesn't
                apply internal parameters listed in :attr:`_clone_props`.

        See :meth:`make_func` for an example.
        '''
        p = self._clone_props
        for k, v in state.items():
            if clone or k not in p:
                setattr(self, k, v)

    def get_funcs(self, step_into_ref=True):
        '''Iterator that yields the function and all its children functions
        if it has any, e.g. for :class:`FuncGroup`. It's DFS order.

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
        '''
        yield self

    def can_other_func_be_added(self, other_func):
        '''Checks whether the other function may be added to us.
        '''
        if isinstance(other_func, CeedFuncRef):
            other_func = other_func.func

        # check if we (or a ref to us) are a child of other_func
        for func in other_func.get_funcs(step_into_ref=True):
            if func is self:
                return False
        return True

    def __deepcopy__(self, memo):
        obj = self.__class__(function_factory=self.function_factory)
        obj.apply_state(self.get_state())
        return obj

    def copy_expand_ref(self):
        obj = self.__class__(function_factory=self.function_factory)
        obj.apply_state(self.get_state(expand_ref=True))
        return obj

    def init_func(self, t_start):
        '''Called with the current global function time in order to make the
        function ready.

        It's called internally at the start of the first :attr:`loop`
        iteration.

        :Params:

            `t_start`: float
                The time in timebase units in global time. :attr:`t_start`
                will be set to this value.
        '''
        self.t_start = t_start
        self.loop_count = 0

    def init_loop_iteration(self, t_start):
        '''Called with the current global function time in order to make the
        function usable.

        It's called internally at the start of every :attr:`loop` iteration,
        except the first.

        :Params:

            `t_start`: float
                The time in timebase units in global time. :attr:`t_start`
                will be set to this value.
        '''
        self.t_start = t_start

    def check_domain(self, t):
        '''Returns whether the function is outside its valid domain for
        time ``t`` in seconds. ``t`` is global so :attr:`t_start` should not
        have been subtracted before calling.
        '''
        return False

    def tick_loop(self, t):
        '''Increments :attr:`loop_count` and returns whether the function is
        done, which is when all the :attr:`loop` are done and :attr:`loop` is
        reached.

        ``t`` is in seconds and this should only be called
        when the function time reached the end of its valid domain so that it
        makes sense to increment the loop.
        '''
        self.loop_count += 1
        if self.loop_count >= self.loop:
            return True
        self.init_loop_iteration(t)
        return False

    def _update_total_duration(self, *largs):
        if self.duration == -1:
            self.duration_total = -1
        else:
            self.duration_total = self.loop * self.duration


class CeedFuncRef(object):
    """The function it refers to must be in the factory.
    """

    func = None

    display = None

    parent_func = None

    function_factory = None

    def __init__(self, function_factory, func=None):
        super(CeedFuncRef, self).__init__()
        self.func = func
        self.function_factory = function_factory

    def get_state(self, state=None, recurse=True, expand_ref=False):
        if expand_ref:
            return self.func.get_state(recurse=recurse, expand_ref=True)

        if state is None:
            state = {}
        state['ref_name'] = self.func.name
        state['cls'] = 'CeedFuncRef'
        return state

    def apply_state(self, state, clone=False):
        self.func = self.function_factory.funcs_inst[state['ref_name']]

    def __deepcopy__(self, memo):
        assert self.__class__ is CeedFuncRef
        return self.function_factory.get_func_ref(func=self)

    def copy_expand_ref(self):
        return self.func.copy_expand_ref()


class CeedFunc(FuncBase):
    '''A base class for a single function (as opposed to a group like
    :class:`FuncGroup`).
    '''

    t_offset = NumericProperty(0)
    '''The amount of time in seconds to add the function time when computing 
    the result.

    All functions that inherit from this class must add this time. E.g. the
    :class:`~ceed.function.plugin.LinearFunc` defines its function as
    ``y(t) = mt + b`` with time ``t = (t_in - t_start + t_offset)``.
    
    :attr:`duration` starts after the :attr:`t_offset`.
    '''

    def check_domain(self, t):
        # t_offset is not used because it's irrelevant to duration.
        if self.duration == -1:
            return 0 <= t - self.t_start
        return 0 <= t - self.t_start < self.duration * self.get_timebase()

    def get_gui_props(self, properties=None):
        d = super(CeedFunc, self).get_gui_props(properties)
        d.update({'duration': None})
        d.update({'t_offset': None})
        return d

    def get_state(self, state=None, recurse=True, expand_ref=False):
        d = super(CeedFunc, self).get_state(state, recurse, expand_ref)
        d.update({'duration': self.duration})
        d.update({'t_offset': self.t_offset})
        return d


class FuncComposit(CeedFunc):
    '''The duration is the minimum of :attr:`f1` and :attr:`f2`.

    :attr:`CeedFunc.t_offset` should only be zero.
    '''

    A = NumericProperty(1.)

    f1 = StringProperty('')

    B = NumericProperty(1.)

    f2 = StringProperty('')

    C = NumericProperty(0.)

    _f1_obj = None

    _f2_obj = None

    _f1_copy = None

    _f2_copy = None

    _restore_func_names = None, None

    def __init__(
            self, name='Composite', description='y(t) = A*f1(t) + B*f2(t) + C',
            **kwargs):
        super(FuncComposit, self).__init__(
            name=name, description=description, **kwargs)

    def init_factory(self, factory):
        super(FuncComposit, self).init_factory(factory)
        self.fbind('f1', self._func_rebind_callback, '_f1_obj', 'f1')
        self.fbind('f2', self._func_rebind_callback, '_f2_obj', 'f2')
        self.function_factory.fbind_proxy(
            'funcs_inst', self._func_factory_rebind_callback)
        self.property('f1').dispatch(self)
        self.property('f2').dispatch(self)
        self.fbind('duration', self._update_duration)

    def del_factory(self, factory):
        super(FuncComposit, self).del_factory(factory)
        self.funbind('f1', self._func_rebind_callback, '_f1_obj', 'f1')
        self.funbind('f2', self._func_rebind_callback, '_f2_obj', 'f2')
        self.function_factory.funbind(
            'funcs_inst', self._func_factory_rebind_callback)
        self.funbind('duration', self._update_duration)
    #
    # def removing_global_funcs(self):
    #     super(FuncComposit, self).removing_global_funcs()
    #     # don't crash when global funcs are removed and their names disappear
    #     self.function_factory.funbind(
    #         'funcs_inst', self._func_factory_rebind_callback)

    def check_domain(self, t):
        raise NotImplemented

    def __call__(self, t):
        f1, f2 = self._f1_copy, self._f2_copy
        if f1 is None and f2 is None:
            raise FuncDoneException

        while True:
            try:
                val = self.C
                if f1:
                    val += self.A * f1(t)
                if f2:
                    val += self.B * f2(t)
                return val
            except FuncDoneException:
                if self.tick_loop(t):
                    break

        raise FuncDoneException

    def init_func(self, t_start, loop=False):
        if not loop:
            self._f1_copy = self._f2_copy = None
            if self._f1_obj:
                f = self._f1_copy = deepcopy(self._f1_obj[0])
                f.parent_func = self.parent_func
            if self._f2_obj:
                f = self._f2_copy = deepcopy(self._f2_obj[0])
                f.parent_func = self.parent_func

        if self._f1_copy:
            self._f1_copy.init_func(t_start)
        if self._f2_copy:
            self._f2_copy.init_func(t_start)
        super(FuncComposit, self).init_func(t_start, loop=loop)

    def get_gui_props(self, properties=None):
        d = super(FuncComposit, self).get_gui_props(properties)
        del d['timebase_numerator']
        del d['timebase_denominator']
        del d['t_offset']
        d['A'] = None
        d['B'] = None
        d['C'] = None

        def values_getter(*largs):
            return list(sorted(
                item for item in self.function_factory.funcs_inst
                if item != self.name))
        d['f1'] = (
            'TrackOptionsSpinner',
            {'track_obj': self.function_factory, 'track_prop': 'funcs_inst',
             'allow_empty': True, 'update_items_on_press': True,
             'values_getter': values_getter})
        d['f2'] = (
            'TrackOptionsSpinner',
            {'track_obj': self.function_factory, 'track_prop': 'funcs_inst',
             'allow_empty': True, 'update_items_on_press': True,
             'values_getter': values_getter})
        return d

    def get_state(self, *largs, **kwargs):
        d = super(FuncComposit, self).get_state(*largs, **kwargs)
        d['A'] = self.A
        d['B'] = self.B
        d['C'] = self.C
        d['f1'] = self.f1
        d['f2'] = self.f2
        return d

    def apply_state(self, state={}, clone=False, old_id_map=None):
        self._restore_func_names = state.pop('f1', None), state.pop('f2', None)
        super(FuncComposit, self).apply_state(state, clone, old_id_map)

    def finalize_func_state(self, id_map, id_to_func_map, old_to_new_name):
        super(FuncComposit, self).finalize_func_state(
            id_map, id_to_func_map, old_to_new_name)

        f1, f2 = self._restore_func_names
        self._restore_func_names = None, None

        if f1:
            self.f1 = old_to_new_name.get(f1, '')
        if f2:
            self.f2 = old_to_new_name.get(f2, '')

    def _func_factory_rebind_callback(self, *largs):
        f1 = self.f1
        if f1:
            if self._f1_obj and (
                    f1 not in self.function_factory.funcs_inst or
                    self.function_factory.funcs_inst[f1]
                    is not self._f1_obj[0]):
                self.property('f1').dispatch(self)
        f2 = self.f2
        if f2:
            if self._f2_obj and (
                    f2 not in self.function_factory.funcs_inst or
                    self.function_factory.funcs_inst[f2]
                    is not self._f2_obj[0]):
                self.property('f2').dispatch(self)

    def _func_rebind_callback(self, f_bind_attr, f_attr, *largs):
        bind = getattr(self, f_bind_attr)
        if bind is not None:
            bind[0].unbind_uid('duration', bind[1])

        name = getattr(self, f_attr)
        if name:
            f = self.function_factory.funcs_inst[name]
            uid = f.fbind(
                'duration', self._update_duration, f_bind_attr, f_attr)
            setattr(self, f_bind_attr, (f, uid))
        else:
            setattr(self, f_bind_attr, None)
        self._update_duration()

    def _update_duration(self, *largs):
        f1 = self._f1_obj
        f2 = self._f2_obj
        val = self.duration
        if f1 is not None:
            if f2 is not None:
                self.duration = min(min(f1[0].duration, f2[0].duration), val)
            else:
                self.duration = min(val, f1[0].duration)
        else:
            if f2 is not None:
                self.duration = min(f2[0].duration, val)
            else:
                self.duration = 0


class FuncGroup(FuncBase):
    '''Function that is composed of a sequence of sub-functions.

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
    '''

    funcs = []
    '''The list of children functions of this function.
    '''

    _func_idx = 0

    def __init__(self, name='Group', **kwargs):
        super(FuncGroup, self).__init__(name=name, **kwargs)
        self.funcs = []

    def init_func(self, t_start):
        super(FuncGroup, self).init_func(t_start)
        self._func_idx = 0

        funcs = self.funcs
        if funcs:
            funcs[0].init_func(t_start)

    def init_loop_iteration(self, t_start):
        super(FuncGroup, self).init_loop_iteration(t_start)
        self._func_idx = 0

        funcs = self.funcs
        if funcs:
            funcs[0].init_func(t_start)

    def check_domain(self, t):
        raise NotImplemented  # not *yet* implemented as it's not needed

    def __call__(self, t):
        funcs = self.funcs
        if not funcs or self._func_idx >= len(self.funcs):
            raise FuncDoneException

        while True:
            try:
                return funcs[self._func_idx](t)
            except FuncDoneException:
                self._func_idx += 1
                if self._func_idx >= len(self.funcs):
                    if self.tick_loop(t):
                        break
                else:
                    funcs[self._func_idx].init_func(t)

        raise FuncDoneException

    def replace_ref_func_with_source(self, func_ref):
        i = self.funcs.index(func_ref)
        self.remove_func(func_ref)
        func = func_ref.copy_expand_ref()
        self.add_func(func, index=i)
        return func, i

    def add_func(self, func, after=None, index=None):
        '''Adds a ``func`` to this function as a sub-function in :attr:`funcs`.
        Remember to check :meth:`can_other_func_be_added` before adding
        if there's potential for it to return False.

        :Params:

            `func`: :class:`FuncBase`
                The function instance to add.
            `after`: :class:`FuncBase`, defaults to None.
                The function in :attr:`funcs` after which to add this function.
            `index`: int, defaults to None.
                The index where to insert the function.
        '''
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
        else:
            func.fbind('duration', self._update_duration)
        self._update_duration()
        self.dispatch('on_changed', op='add', index=index)

    def remove_func(self, func):
        '''Removes sub-function ``func`` from :attr:`funcs`. It must exist in
        :attr:`funcs`.

        :Params:

            `func`: :class:`FuncBase`
                The function instance to remove.
        '''
        assert func.parent_func is self
        func.parent_func = None

        if isinstance(func, CeedFuncRef):
            func.func.funbind('duration', self._update_duration)
        else:
            func.funbind('duration', self._update_duration)
        index = self.funcs.index(func)
        del self.funcs[index]
        self._update_duration()
        self.dispatch('on_changed', op='remove', index=index)
        return True

    def _update_duration(self, *largs):
        '''Computes duration as a function of its children.
        '''
        funcs = (
            f.func if isinstance(f, CeedFuncRef) else f for f in self.funcs)
        self.duration = sum(
            (f.duration_total for f in funcs if f.duration != -1))

    def get_state(self, state=None, recurse=True, expand_ref=False):
        d = super(FuncGroup, self).get_state(state, recurse, expand_ref)
        if recurse:
            d['funcs'] = funcs = []
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

    def get_funcs(self, step_into_ref=True):
        yield self
        for func in self.funcs:
            if isinstance(func, CeedFuncRef):
                if not step_into_ref:
                    yield func
                    continue

                func = func.func
            for f in func.get_funcs(step_into_ref):
                yield f


def register_all_functions(function_factory):
    '''Call this with a :class:`FunctionFactoryBase` instance and it will
    register all the plugin functions with the :class:`FunctionFactoryBase`.

    :param function_factory: a :class:`FunctionFactoryBase` instance.
    '''
    function_factory.register(FuncGroup)
    function_factory.register(FuncComposit)
    from ceed.function.plugin import get_plugin_functions
    for f in get_plugin_functions():
        function_factory.register(f)
