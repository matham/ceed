from copy import deepcopy
from collections import OrderedDict

from kivy.event import EventDispatcher
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, \
    ObjectProperty, DictProperty
from kivy.logger import Logger
from kivy.factory import Factory

from ceed.utils import fix_name


class FuncDoneException(Exception):
    pass


class FunctionFactoryBase(EventDispatcher):

    __events__ = ('on_changed', )

    funcs = {}

    editable_func_list = []

    avail_funcs = DictProperty({})

    locked_funcs = {}

    show_widgets = False

    def __init__(self, **kwargs):
        super(FunctionFactoryBase, self).__init__(**kwargs)
        self.funcs = {}
        self.editable_func_list = []
        self.locked_funcs = {}

        funcs = self.avail_funcs
        for cls in self.get_classes():
            f = cls()
            funcs[f.name] = f

    def on_changed(self, *largs, **kwargs):
        pass

    def register(self, cls):
        name = cls.__name__
        funcs = self.funcs
        if name in funcs:
            Logger.warn('"{}" is already a registered function'.format(name))
        funcs[name] = cls

        f = cls()
        self.avail_funcs[f.name] = f
        self.locked_funcs[f.name] = f

    def get(self, name):
        funcs = self.funcs
        if name not in funcs:
            return None
        return funcs[name]

    def get_names(self):
        return self.funcs.keys()

    def get_classes(self):
        return self.funcs.values()

    def save_funcs(self, id_map=None):
        if id_map is None:
            id_map = {}
        for f in self.editable_func_list:
            CeedFunc.fill_id_map(f, id_map)

        states = [f._copy_state() for f in self.editable_func_list]
        return states, id_map

    def recover_funcs(self, funcs, id_to_func_map=None, old_id_map=None):
        for state in funcs:
            self.add_func(
                CeedFunc.make_func(state, clone=True, old_id_map=old_id_map))

        id_to_func_map = {} if id_to_func_map is None else id_to_func_map
        for f in self.editable_func_list:
            CeedFunc.fill_id_to_func_map(f, id_to_func_map)

        return self.editable_func_list[:], id_to_func_map

    def add_func(self, func):
        func.source_func = None
        func.name = fix_name(func.name, self.avail_funcs)

        func.fbind('name', self._track_func_name, func)
        self.avail_funcs[func.name] = func
        self.editable_func_list.append(func)
        if self.show_widgets:
            func.display.show_func()
        self.dispatch('on_changed')

    def remove_func(self, func):
        func.funbind('name', self._track_func_name, func)
        del self.avail_funcs[func.name]
        self.editable_func_list.remove(func)

        if func._display:
            func._display.hide_func()
        self.dispatch('on_changed')

    def _track_func_name(self, func, *largs):
        for name, f in self.avail_funcs.items():
            if f is func:
                if func.name == name:
                    return

                del self.avail_funcs[name]
                break
        func.name = fix_name(func.name, self.avail_funcs)
        self.avail_funcs[func.name] = func

    def clear_funcs(self):
        for f in self.editable_func_list[:]:
            self.remove_func(f)


class FuncBase(EventDispatcher):

    name = StringProperty('Abstract')

    description = StringProperty('')

    icon = StringProperty('')

    duration = NumericProperty(0)
    '''-1 means go on forever.
    '''

    loop = NumericProperty(1)

    track_source = BooleanProperty(True)
    '''On refresh track again
    '''

    source_func = ObjectProperty(None, rebind=True, allownone=True)
    '''The function from which this is a copy.
    '''

    parent_func = ObjectProperty(None, rebind=True, allownone=True)
    '''When it's a sub-function, the parent function.
    '''

    _func_id = 0

    func_id = NumericProperty(0)
    '''Unique to each function instance.
    '''

    _display = None

    _display_cls = ''

    _clone_props = {'cls', 'track_source', 'func_id'}
    '''props that are specific to the function and should not be copied
    between different functions.
    '''

    timebase = 1.

    _t0_global = 0

    _loop_count = 0

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(FuncBase, self).__init__(**kwargs)
        self.func_id = FuncBase._func_id
        FuncBase._func_id += 1

        for prop in self._copy_state(recurse=False):
            self.fbind(prop, self.dispatch, 'on_changed', prop)

        self.fbind('track_srouce', self._track_source_callback)
        self.fbind('on_changed', FunctionFactory.dispatch, 'on_changed')

    def on_changed(self, *largs, **kwargs):
        pass

    def get_gui_props(self, attrs={}):
        d = {'name': None, 'loop': None}
        d.update(attrs)
        return d

    def get_gui_elements(self, items=[]):
        d = []
        d.extend(items)
        return d

    def _copy_state(self, state={}, recurse=True):
        d = {
            'name': self.name, 'cls': self.__class__.__name__,
            'track_source': self.track_source, 'func_id': self.func_id,
            'loop': self.loop}
        d.update(state)
        return d

    def _apply_state(self, state={}, clone=False, old_id_map=None):
        p = self._clone_props
        for k, v in state.items():
            if (clone or k not in p) and k != 'func_id':
                setattr(self, k, v)
        if clone and old_id_map is not None and 'func_id' in state:
            old_id_map[self.func_id] = state['func_id']

    @property
    def display(self):
        if self._display:
            return self._display
        if not self._display_cls:
            return None

        w = self._display = Factory.get(self._display_cls)(func=self)
        return w

    def _track_source_callback(self, *largs):
        if self.parent_func and self.parent_func.track_source:
            self.track_source = True

    def reload_from_source(self, *largs):
        if self.source_func:
            self._apply_state(self.source_func._copy_state())
            self.bind_sources(self.source_func, self)

    def _update_from_source(self, *largs):
        if largs and largs[0] is not self.source_func:
            return
        if len(largs) > 1 and largs[1] in self._clone_props:
            return

        if self.source_func and self.track_source:
            self._apply_state(self.source_func._copy_state(recurse=False))
            self.bind_sources(self.source_func, self)

    def get_funcs(self):
        yield self

    def parent_in_other_children(self, other):
        '''Whether any parent of ours (including this instance) is a child
        in the tree of other.
        '''
        other_names = {o.name for o in other.get_funcs()
                       if o.name not in FunctionFactory.locked_funcs}

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
        func._apply_state(state, clone=clone, old_id_map=old_id_map)
        return func

    @staticmethod
    def fill_id_map(func, ids):
        '''Converts the object instance <=> object source relationships to a
        id <=> id relationship of those objects nd returns it in a dict.
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
        '''src and target must have identical structure.
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
        obj._apply_state(src._copy_state())
        self.bind_sources(src, obj, skip_root=False)
        return obj

    def init_func(self, t0_global, loop=False):
        self._t0_global = t0_global
        if not loop:
            self._loop_count = 0

    def done_condition(self, t):
        return False

    def check_done(self, t):
        if self.done_condition(t):
            self._loop_count += 1
            if self._loop_count >= self.loop:
                return True
            self.init_func(t, loop=True)
        return False


class CeedFunc(FuncBase):

    t0 = NumericProperty(0)

    _display_cls = 'FuncWidget'

    def done_condition(self, t):
        # timebase is factored out.
        return self.duration >= 0 and t - self._t0_global >= self.duration

    def __call__(self, t):
        raise NotImplementedError

    def get_gui_props(self, attrs={}):
        d = super(CeedFunc, self).get_gui_props(attrs)
        d.update({'duration': None})
        d.update({'t0': None})
        return d

    def _copy_state(self, state={}, recurse=True):
        d = super(CeedFunc, self)._copy_state(state, recurse)
        d.update({'duration': self.duration})
        d.update({'t0': self.t0})
        return d


class FuncGroup(FuncBase):

    funcs = []

    _display_cls = 'FuncWidgetGroup'

    _func_idx = 0

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Group')
        super(FuncGroup, self).__init__(**kwargs)
        self.funcs = []

    def init_func(self, t0_global, loop=False):
        super(FuncGroup, self).init_func(t0_global, loop=loop)
        self._func_idx = 0

        funcs = self.funcs
        if not funcs:
            return

        funcs[0].init_func(t0_global)

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

    def add_func(self, func, after=None):
        func.parent_func = self

        if after is None:
            self.funcs.append(func)
        else:
            i = self.funcs.index(after)
            self.funcs.insert(i + 1, func)
        func.fbind('duration', self._update_duration)
        self._update_duration()
        self.dispatch('on_changed', op='add', index=len(self.funcs) - 1)
        if self._display:
            func.display.show_func()

    def remove_func(self, func):
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

    def _copy_state(self, state={}, recurse=True):
        d = super(FuncGroup, self)._copy_state(state)
        if recurse:
            d['funcs'] = [f._copy_state() for f in self.funcs]
        return d

    def _apply_state(self, state={}, clone=False, old_id_map=None):
        for opts in state.pop('funcs', []):
            self.add_func(
                self.make_func(opts, clone=clone, old_id_map=old_id_map))
        super(FuncGroup, self)._apply_state(
            state, clone=clone, old_id_map=old_id_map)

    def get_funcs(self):
        yield self
        for func in self.funcs:
            for f in func.get_funcs():
                yield f

FunctionFactory = FunctionFactoryBase()
FunctionFactory.register(FuncGroup)
from ceed.function.plugin import import_plugins
