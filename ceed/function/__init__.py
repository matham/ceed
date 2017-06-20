'''Functions
=========================

Defines the functions used with shapes to create time-varying intensities
of the shapes during an experiment.

A function is inherited from :class:`FuncBase` which returns a
value when called with a time value. Example function usage is::

    >>> CosFunc = FunctionFactory.get('CosFunc')
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

Function classes registered with :attr:`FunctionFactory` are available to the
user in the GUI. Similarly, customized function instances can be registered
with :meth:`FunctionFactoryBase.add_function`.
'''

from copy import deepcopy
from collections import OrderedDict
from fractions import Fraction

from kivy.event import EventDispatcher
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, \
    ObjectProperty, DictProperty, AliasProperty
from kivy.logger import Logger
from kivy.factory import Factory

from ceed.utils import fix_name

__all__ = ('FuncDoneException', 'FunctionFactoryBase', 'FuncBase', 'CeedFunc',
           'FuncGroup', 'FunctionFactory')


class FuncDoneException(Exception):
    '''Raised when the :class:`FuncBase` is called with a time value outside
    its valid time interval.
    '''
    pass


class FunctionFactoryBase(EventDispatcher):
    '''A global store of the defined :class:`FuncBase` classes and customized
    instances.

    Plugins register function classes with an instance of this class
    (:attr:`FunctionFactory`) using :meth:`register` to make it available to
    the user in the GUI.

    Similarly, function instances that has been customized with values or
    function groups customized with children functions in the GUI are also
    stored here with :meth:`add_func` making it available for duplication or
    usage elsewhere.

    To get a function class registered with :meth:`register`, e.g.
    :class:`ceed.function.plugin.CosFunc`::

        >>> cls = FunctionFactory.get('CosFunc')  # using class name
        >>> cls
        ceed.function.plugin.CosFunc

    When registering a function, an instance is automatically created and added
    to :attr:`funcs_inst`. Similarly, customized functions added with
    :meth:`add_func` are also added to :attr:`funcs_inst` ::

        >>> cls = FunctionFactory.get('CosFunc')  # get registered class
        >>> f = FunctionFactory.funcs_inst['Cos']  # get auto created instance
        >>> f
        <ceed.function.plugin.CosFunc at 0x53ada08>
        >>> f.f  # the default rate is 1Hz
        1.0
        >>> f2 = cls(f=100)  # new instance with 100Hz rate
        >>> f2.f
        100
        >>> f2.name  # the default name
        'Cos'
        >>> FunctionFactory.add_func(f2)  # add it to the list
        >>> f2.name  # when adding it fixes the name so that it's unique
        'Cos-2'
        >>> FunctionFactory.funcs_inst['Cos-2']
        <ceed.function.plugin.CosFunc at 0x5141938>

    The general usage of the registered instances with :meth:`add_func` is to
    get the instance and then copy and further customize it. See
    :attr:`FuncBase.source_func`.

    :events:

        on_changed:
            The event is triggered every time a function is added or removed
            from the factory. It's also triggered whenever a :class:`FuncBase`
            triggers its ``on_changed`` event.
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
    corresponding functions.

    Contains functions added with :meth:`add_func` as well as those
    automatically created and added when :meth:`register` is called on a class.
    '''

    funcs_inst_default = {}
    '''Dict whose keys are the function :attr:`name` and whose values are the
    corresponding functions.

    Contains only the functions that are automatically created and added when
    :meth:`register` is called on a class.
    '''

    show_widgets = False
    '''Whether the function should call the associated widgets functions to
    display it. This is set to ``True`` only if the code is executed from
    the GUI.
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
    either a float or a Fraction instance when the numerator and
    denominator are ints.

    The timebase is the scaling factor by which some function properties that
    relate to time, e.g. :attr:`FuncBase.duration`, are multiplied to convert
    from timebase units to time.

    This is the default timebase that is used when :attr:`FuncBase.timebase`
    is -1, otherwise each function uses its :attr:`FuncBase.timebase`.
    '''

    def __init__(self, **kwargs):
        super(FunctionFactoryBase, self).__init__(**kwargs)
        self.funcs_cls = {}
        self.funcs_user = []
        self.funcs_inst_default = {}

        funcs = self.funcs_inst
        for cls in self.get_classes():
            f = cls()
            funcs[f.name] = f

    def on_changed(self, *largs, **kwargs):
        pass

    def register(self, cls):
        '''Registers the class and adds it to :attr:`funcs_cls`. It also
        creates an instance of the class that is added to :attr:`funcs_inst`
        and :attr:`funcs_inst_default`.

        This function is called to register a new class from a plugin.

        :Params:

            `cls`: subclass of :class:`FuncBase`
                The class to register.
        '''
        name = cls.__name__
        funcs = self.funcs_cls
        if name in funcs:
            Logger.warn('"{}" is already a registered function'.format(name))
        funcs[name] = cls

        f = cls()
        self.funcs_inst[f.name] = f
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
        '''Returns the names of all :meth:`register` classes.
        '''
        return list(self.funcs_cls.keys())

    def get_classes(self):
        '''Returns the classes registered with :meth:`register`.
        '''
        return list(self.funcs_cls.values())

    def save_funcs(self, id_map=None):
        '''Dumps all the config data from all the :attr:`funcs_user` so that
        it can be saved and later recovered with :meth:`recover_funcs`.

        :Params:

            `id_map`: dict
                A dict that will be filled-in as the function state is saved.
                The keys will be the :attr:`FuncBase.func_id` of each of the
                saved functions. The corresponding value will be the
                :attr:`FuncBase.func_id` of the function in
                :attr:`FuncBase.source_func` if not None. This allows the
                reconstruction of the function dependencies.

                If None, the default, a dict is created. Otherwise, it's
                updated in place.

        :returns:

            A two tuple ``(states, id_map)``. ``states`` is a list of all the
            functions' states. ``id_map`` is the ``id_map`` created or passed
            in.
        '''
        if id_map is None:
            id_map = {}
        for f in self.funcs_user:
            CeedFunc.fill_id_map(f, id_map)

        states = [f.get_state() for f in self.funcs_user]
        return states, id_map

    def recover_funcs(self, funcs, id_to_func_map=None, old_id_map=None):
        '''Restores all the user functions that was saved with
        :meth:`save_funcs`.

        :Params:

            `funcs`: list
                The list of function states as returned by :meth:`save_funcs`.
            `id_to_func_map`: dict
                A dict that will be filled-in as the functions are re-created.
                The keys will be the :attr:`FuncBase.func_id` of the new
                functions created from each state. The corresponding value
                will be the function created.

                If None, the default, a dict is created. Otherwise, it's
                updated in place.
            `old_id_map`: dict
                A dict that will be filled-in as the functions are re-created.
                The keys will be the :attr:`FuncBase.func_id` of the new
                functions created from each state. The corresponding value
                will be the :attr:`FuncBase.func_id` as saved in the ``funcs``
                state passed in. :attr:`FuncBase.func_id` is likely to change
                as the function is re-created; this keeps track of that.

                If None, the default, a dict is created. Otherwise, it's
                updated in place.

        :returns:

            A two-tuple. The first element is a copy of the :attr:`funcs_user`
            list. The second is ``id_to_func_map`` created, or passed in.
        '''
        for state in funcs:
            self.add_func(
                CeedFunc.make_func(state, clone=True, old_id_map=old_id_map))

        id_to_func_map = {} if id_to_func_map is None else id_to_func_map
        for f in self.funcs_user:
            CeedFunc.fill_id_to_func_map(f, id_to_func_map)

        return self.funcs_user[:], id_to_func_map

    def add_func(self, func, index=None):
        '''Adds the function to :attr:`funcs_user` and :attr:`funcs_inst`,
        which makes it available in the GUI as one of the customized
        functions.

        :Params:

            `func`: subclass of :class:`FuncBase`.
                The function to add.
        '''
        func.source_func = None
        func.name = fix_name(func.name, self.funcs_inst)

        func.fbind('name', self._track_func_name, func)
        self.funcs_inst[func.name] = func
        if index is None:
            self.funcs_user.append(func)
        else:
            self.funcs_user.insert(index, func)
        if self.show_widgets:
            func.display.show_func(index=index)
        self.dispatch('on_changed')

    def remove_func(self, func):
        '''Removes the function added with :meth:`add_func`.

        :Params:

            `func`: subclass of :class:`FuncBase`.
                The function to add.
        '''
        func.funbind('name', self._track_func_name, func)
        del self.funcs_inst[func.name]
        self.funcs_user.remove(func)

        if func._display:
            func._display.hide_func()
        self.dispatch('on_changed')

    def _track_func_name(self, func, *largs):
        '''Fixes the name of the function instances stored here to ensure it's
        unique.
        '''
        for name, f in self.funcs_inst.items():
            if f is func:
                if func.name == name:
                    return

                del self.funcs_inst[name]
                break
        func.name = fix_name(func.name, self.funcs_inst)
        self.funcs_inst[func.name] = func

    def clear_funcs(self):
        '''Removes all the functions registered with :meth:`add_func`.
        '''
        for f in self.funcs_user[:]:
            self.remove_func(f)


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
    using :meth:`check_done`, which if it is done, will increment the
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
    '''The name of the function instance. The name must be unique when an
    instance is added to the :attr:`FunctionFactory`, otherwise it's
    automatically fixed.
    '''

    description = StringProperty('')
    '''A description of the function. Typically shown to the user.
    '''

    icon = StringProperty('')
    '''The function icon. Not used currently.
    '''

    duration = NumericProperty(0)
    '''How long, after the start of the function the function is valid.
    -1 means go on forever. See class for more details.

    It is in timebase units.
    '''

    loop = NumericProperty(1)
    '''The number of times the function loops through. At the end of each loop
    :attr:`loop_count` is incremented until done. See class for more details.
    '''

    track_source = BooleanProperty(True)
    '''If True, changes in the :attr:`source_func` will also change this
    function. See :attr:`source_func` for details.
    '''

    source_func = ObjectProperty(None, rebind=True, allownone=True)
    '''When a function is copied (``f2 = deepcopy(f)``) from another function,
    :attr:`source_func` on the new function, ``f2``, is set automatically to
    the original function, ``f``. :attr:`track_source` is also automatically
    set to ``True``.

    When :attr:`track_source` is True and :attr:`source_func` is not None,
    whenever a configurable property (i.e.
    it is returned as key in the :meth:`get_state` dict) of the source function
    :attr:`source_func` is changed, the value in this function will be updated
    to the new value. It allows keeping function copies up to date with the
    original template function.
    '''

    parent_func = ObjectProperty(None, rebind=True, allownone=True)
    '''If this function is the child of another function, e.g. it's a
    sub-function of a :class:`FuncGroup` instance, then :attr:`parent_func`
    points to the parent function.
    '''

    _func_id = 0
    '''The global :attr:`func_id` counter.
    '''

    func_id = NumericProperty(0)
    '''Unique to each function instance and uniquely defines this function.
    It is automatically assigned.
    '''

    _display = None

    display_cls = ''
    '''The name of the widget class that is used to represent this function.
    See :attr:`display`.
    '''

    _clone_props = {'cls', 'track_source', 'func_id'}
    '''props that are specific to the function and should not be copied
    between different functions.
    '''

    timebase_numerator = NumericProperty(-1)
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

    By default :attr:`timebase_numerator` is -1 and
    :attr:`timebase_denominator` is 1 which makes :attr:`timebase` -1
    indicating that the timebase used is given by
    :attr:`FunctionFactoryBase.timebase`. When :attr:`timebase` is not -1 this
    :attr:`timebase` is used instead.

    :meth:`get_timebase` returns the actual timebase used.

    See the class description for its usage.
    '''

    t_start = 0
    '''The time offset subtracted from function time (in secs). See the class
    description.
    '''

    loop_count = 0
    '''The current loop count. See :attr:`loop` and the class description.
    '''

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(FuncBase, self).__init__(**kwargs)
        self.func_id = FuncBase._func_id
        FuncBase._func_id += 1

        for prop in self.get_state(recurse=False):
            self.fbind(prop, self.dispatch, 'on_changed', prop)

        self.fbind('track_srouce', self._track_source_callback)
        self.fbind('on_changed', FunctionFactory.dispatch, 'on_changed')

    def __call__(self, t):
        raise NotImplementedError

    def get_timebase(self):
        '''Returns the function's timebase.

        If :attr:`timebase_numerator` is -1 it returns the timebase of its
        :attr:`parent_func` with :meth:`get_timebase` if it has a parent.
        Otherwise, it returns :attr:`FunctionFactoryBase.timebase`.
        '''
        if self.timebase_numerator == -1:
            if self.parent_func:
                return self.parent_func.get_timebase()
            return FunctionFactory.timebase
        return self.timebase

    def on_changed(self, *largs, **kwargs):
        pass

    def get_gui_props(self, attrs=None):
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

            >>> Cos = FunctionFactory.get('CosFunc')
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
        if attrs is None:
            attrs = {}
        attrs['name'] = None
        attrs['loop'] = None
        attrs['timebase_numerator'] = None
        attrs['timebase_denominator'] = None
        return attrs

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

            >>> Cos = FunctionFactory.get('CosFunc')
            >>> cos = Cos()
            >>> cos.get_gui_elements()
            []
        '''
        if items is None:
            items = []
        return items

    def get_state(self, state=None, recurse=True):
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

            >>> Cos = FunctionFactory.get('CosFunc')
            >>> cos = Cos()
            >>> cos.get_state()
            {'A': 1.0,
             'cls': 'CosFunc',
             'duration': 0,
             'f': 1.0,
             'func_id': 5,
             'loop': 1,
             'name': 'Cos',
             't_offset': 0,
             'th0': 0.0,
             'track_source': True}
            >>> Group = FunctionFactory.get('FuncGroup')
            >>> g = Group()
            >>> g
            <ceed.function.FuncGroup at 0x4f85800>
            >>> g.add_func(cos)
            >>> g.get_state(recurse=True)
            {'cls': 'FuncGroup',
             'func_id': 6,
             'funcs': [{'A': 1.0,
               'cls': 'CosFunc',
               'duration': 0,
               'f': 1.0,
               'func_id': 5,
               'loop': 1,
               'name': 'Cos',
               't_offset': 0,
               'th0': 0.0,
               'track_source': True}],
             'loop': 1,
             'name': 'Group',
             'track_source': True}
            >>> g.get_state(recurse=False)
            {'cls': 'FuncGroup',
             'func_id': 6,
             'loop': 1,
             'name': 'Group',
             'track_source': True}
        '''
        d = {
            'name': self.name, 'cls': self.__class__.__name__,
            'track_source': self.track_source, 'func_id': self.func_id,
            'loop': self.loop,
            'timebase_numerator': self.timebase_numerator,
            'timebase_denominator': self.timebase_denominator}
        if state is None:
            state = d
        else:
            state.update(d)
        return state

    def apply_state(self, state={}, clone=False, old_id_map=None):
        '''Takes the state of the function saved with :meth:`get_state` and
        applies it to this function. it also creates any children function e.g.
        in the case of a :class:`FuncGroup`.

        It is called internally and should not be used directly. Use
        :meth:`make_func` instead.

        :Params:

            `state`: dict
                The dict to use to reconstruct the function as returned by
                :meth:`get_state`.
            `clone`: bool
                If True will copy all the state, otherwise it doesn't
                copy internal parameters e.g. :attr:`func_id`.
            `old_id_map`: dict
                A dict, which if not None will be filled in as the function
                is constructed. The keys will be the :attr:`func_id` of the
                function. The corresponding value will be the :attr:`func_id`
                as saved in the ``state`` passed in. :attr:`func_id` is likely
                to change as the function is re-created; this keeps track of
                that.

        See :meth:`make_func` for an example.
        '''
        p = self._clone_props
        for k, v in state.items():
            if (clone or k not in p) and k != 'func_id':
                setattr(self, k, v)
        if clone and old_id_map is not None and 'func_id' in state:
            old_id_map[self.func_id] = state['func_id']

    @property
    def display(self):
        '''The widget that displays this function and its options. Calling
        this property will create the widget if it isn't already created.

        The widget class instantiated is gotten from the kivy factory using
        :attr:`display_cls`.
        '''
        if self._display:
            return self._display
        if not self.display_cls:
            return None

        w = self._display = Factory.get(self.display_cls)(func=self)
        return w

    def _track_source_callback(self, *largs):
        '''If the function has a parent, :attr:`track_source` is automatically
        set to True when the parent's :attr:`track_source` is set to True.
        '''
        if self.parent_func and self.parent_func.track_source:
            self.track_source = True

    def reload_from_source(self, *largs):
        '''If :attr:`source_func` is not None, it applies all the
        configuration options of :attr:`source_func` to this function.
        '''
        if self.source_func:
            self.apply_state(self.source_func.get_state())
            self.bind_sources(self.source_func, self)

    def _update_from_source(self, *largs):
        '''Updates the config options of this function to match
        :attr:`source_func`.
        '''
        if largs and largs[0] is not self.source_func:
            return
        if len(largs) > 1 and largs[1] in self._clone_props:
            return

        if self.source_func and self.track_source:
            self.apply_state(self.source_func.get_state(recurse=False))
            self.bind_sources(self.source_func, self)

    def get_funcs(self):
        '''Iterator that yields the function and all its children functions
        if it has any, e.g. for :class:`FuncGroup`.

        E.g.::

            >>> Cos = FunctionFactory.get('CosFunc')
            >>> cos = Cos()
            >>> Group = FunctionFactory.get('FuncGroup')
            >>> g = Group()
            >>> g.add_func(cos)
            >>> [f for f in cos.get_funcs()]
            [<ceed.function.plugin.CosFunc at 0x4d71c78>]
            >>> [f for f in g.get_funcs()]
            [<ceed.function.FuncGroup at 0x4f85800>,
             <ceed.function.plugin.CosFunc at 0x4d71c78>]
        '''
        yield self

    def parent_in_other_children(self, other):
        '''Checks whether any parent of this function all the way up (including
        this function) is a child in the child tree of ``other``.

        This specifically only checks using names and ignores names that are
        the default functions (:attr:`FunctionFactoryBase.funcs_inst_default`).

        :Params:

            `other`: :class:`FuncBase`
                The function to check for membership.

        :returns:

            True when this function or its parent is a sub-child of the
            ``other`` function.
        '''
        other_names = {o.name for o in other.get_funcs()
                       if o.name not in FunctionFactory.funcs_inst_default}

        parent = self
        while parent is not None:
            if parent.name in other_names:
                return True
            parent = parent.parent_func
        return False

    @staticmethod
    def make_func(state, clone=False, old_id_map=None):
        '''Instantiates the function from the state and returns it.

        This pre-creates the display when appropriate.
        '''
        c = state.pop('cls')
        cls = FunctionFactory.get(c)
        if cls is None:
            raise Exception('Missing class "{}"'.format(c))

        func = cls()
        if FunctionFactory.show_widgets:
            func.display  # create it before apply
        func.apply_state(state, clone=clone, old_id_map=old_id_map)
        return func

    @staticmethod
    def fill_id_map(func, ids):
        '''Converts the object instance <=> object source relationships to a
        id <=> id relationship of those objects and returns it in a dict.
        '''
        for f in func.get_funcs():
            ids[f.func_id] = \
                f.source_func.func_id if f.source_func else None

    @staticmethod
    def fill_id_to_func_map(func, id_to_func_map):
        for f in func.get_funcs():
            id_to_func_map[f.func_id] = f

    @staticmethod
    def set_source_from_id(func, id_map, id_to_func_map):
        for f in func.get_funcs():
            if f.func_id not in id_map:  # f was not saved
                continue

            src_id = id_map[f.func_id]
            if src_id not in id_to_func_map:  # f parent not recovered
                continue
            src_f = f.source_func = id_to_func_map[src_id]
            src_f.fbind('on_changed', f._update_from_source)

    def bind_sources(self, src, target, skip_root=True):
        '''Walks through ``src`` and ``target`` recursively with
        :meth:`get_funcs` and sets the :attr:`source_func` of ``target`` to the
        corresponding function from ``src``. It also sets
        :attr:`track_source` to true.

        :Params:

            `src`: :class:`FuncBase`
                The function that :attr:`source_func` of ``target`` will be set
                to
            `target`: :class:`FuncBase`
                The function on which :attr:`source_func` will be set.
            `skip_root`: bool
                Whether to skip the root functions (``src`` and ``target``)
                when doing this. Defaults to True.
        '''
        s, t = src.get_funcs(), target.get_funcs()
        if skip_root:
            next(s)
            next(t)
        for src_func, new_func in zip(s, t):
            new_func.source_func = src_func
            new_func.track_source = True
            src_func.fbind('on_changed', new_func._update_from_source)

    def __deepcopy__(self, memo):
        src = self
        obj = src.__class__()
        obj.apply_state(src.get_state())
        self.bind_sources(src, obj, skip_root=False)
        return obj

    def init_func(self, t_start, loop=False):
        '''Called internally with the current time in order to make the
        function usable through updating the valid range.

        It's called internally at the start of every :attr:`loop` iteration.

        :Params:

            `t_start`: float
                The current time in seconds in global time. :attr:`t_start`
                will be set to this value.
            `loop`: bool
                If it's the first iteration of the loop it should be False,
                subsequently it should be True. Defaults to False.
        '''
        self.t_start = t_start
        if not loop:
            self.loop_count = 0

    def done_condition(self, t):
        '''Returns whether the function has passed its valid interval for
        time ``t`` in seconds.
        '''
        return False

    def check_done(self, t):
        '''Checks whether the function is outside its valid interval with
        :meth:`done_condition` for time ``t`` in seconds and whether it has
        completed all the loop iterations.

        If the iteration is done, it proceeds to the next iteration. It returns
        True only if the last iteration has completed.
        '''
        if self.done_condition(t):
            self.loop_count += 1
            if self.loop_count >= self.loop:
                return True
            self.init_func(t, loop=True)
        return False


class CeedFunc(FuncBase):
    '''A base class for a single function (as opposed to a group like
    :class:`FuncGroup`).
    '''

    t_offset = NumericProperty(0)
    '''The amount of time in seconds to add the function when computing the
    result.

    All functions that inherit from this class must add this time. E.g. the
    :class:`~ceed.function.plugin.LinearFunc` defines its function as
    ``y(t) = mt + b`` with time ``t = (t_in - t_start + t_offset)``.
    '''

    display_cls = 'FuncWidget'

    def done_condition(self, t):
        # timebase is factored out.
        return self.duration >= 0 and \
            (t - self.t_start) / self.get_timebase() >= self.duration

    def get_gui_props(self, attrs=None):
        d = super(CeedFunc, self).get_gui_props(attrs)
        d.update({'duration': None})
        d.update({'t_offset': None})
        return d

    def get_state(self, state=None, recurse=True):
        d = super(CeedFunc, self).get_state(state, recurse)
        d.update({'duration': self.duration})
        d.update({'t_offset': self.t_offset})
        return d


class FuncGroup(FuncBase):
    '''Function that is composed of a sequence of sub-functions.

    When the function instance is called it goes through all its sub-functions
    sequentially until they are done. E.g.::

        >>> Group = FunctionFactory.get('FuncGroup')
        >>> Const = FunctionFactory.get('ConstFunc')
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

    display_cls = 'FuncWidgetGroup'

    _func_idx = 0

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Group')
        super(FuncGroup, self).__init__(**kwargs)
        self.funcs = []

    def init_func(self, t_start, loop=False):
        super(FuncGroup, self).init_func(t_start, loop=loop)
        self._func_idx = 0

        funcs = self.funcs
        if not funcs:
            return

        funcs[0].init_func(t_start)

    def done_condition(self, t):
        return self._func_idx >= len(self.funcs)

    def __call__(self, t):
        funcs = self.funcs
        if not funcs:
            raise FuncDoneException
        while not self.check_done(t):
            try:
                return funcs[self._func_idx](t)
            except FuncDoneException:
                self._func_idx += 1
                if not self.check_done(t):
                    funcs[self._func_idx].init_func(t)

        raise FuncDoneException

    def add_func(self, func, after=None, index=None):
        '''Adds a ``func`` to this function as a sub-function in :attr:`funcs`.

        :Params:

            `func`: :class:`FuncBase`
                The function instance to add.
            `after`: :class:`FuncBase`
                The function in :attr:`funcs` after which to add this function.
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
        func.fbind('duration', self._update_duration)
        self._update_duration()
        self.dispatch('on_changed', op='add', index=index)
        if self._display:
            func.display.show_func(index=index)

    def remove_func(self, func):
        '''Removes sub-function ``func`` from :attr:`funcs`. It must exist in
        :attr:`funcs`.

        :Params:

            `func`: :class:`FuncBase`
                The function instance to remove.
        '''
        if func.parent_func is self:
            func.parent_func = None
        if self._display and func._display:
            func._display.hide_func()

        func.funbind('duration', self._update_duration)
        index = self.funcs.index(func)
        del self.funcs[index]
        self._update_duration()
        self.dispatch('on_changed', op='remove', index=index)

    def _update_duration(self, *largs):
        '''Computes duration as a function of its children.
        '''
        self.duration = sum(
            (f.duration for f in self.funcs if f.duration!= -1))

    def _update_from_source(self, *largs, **kwargs):
        if not self.source_func or not self.track_source:
            return
        if largs and largs[0] is not self.source_func:
            return

        if not kwargs:
            super(FuncGroup, self)._update_from_source(*largs, **kwargs)
            return

        if kwargs['op'] == 'add':
            self.add_func(deepcopy(self.source_func.funcs[kwargs['index']]))
        else:
            self.remove_func(self.funcs[kwargs['index']])

    def reload_from_source(self, *largs):
        if not self.source_func:
            return

        for func in self.funcs[:]:
            self.remove_func(func)
        super(FuncGroup, self).reload_from_source()

    def get_state(self, state=None, recurse=True):
        d = super(FuncGroup, self).get_state(state)
        if recurse:
            d['funcs'] = [f.get_state() for f in self.funcs]
        return d

    def apply_state(self, state={}, clone=False, old_id_map=None):
        for opts in state.pop('funcs', []):
            self.add_func(
                self.make_func(opts, clone=clone, old_id_map=old_id_map))
        super(FuncGroup, self).apply_state(
            state, clone=clone, old_id_map=old_id_map)

    def get_funcs(self):
        yield self
        for func in self.funcs:
            for f in func.get_funcs():
                yield f

FunctionFactory = FunctionFactoryBase()
'''The global function factory instance where functions are registered.
'''
FunctionFactory.register(FuncGroup)
from ceed.function.plugin import import_plugins
