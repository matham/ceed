from copy import deepcopy
from collections import OrderedDict

from kivy.event import EventDispatcher
from kivy.properties import StringProperty, NumericProperty, BooleanProperty, \
    ObjectProperty
from kivy.logger import Logger
from kivy.uix.behaviors.knspace import knspace


class FunctionFactoryBase(object):

    funcs = {}

    def __init__(self, **kwargs):
        super(FunctionFactoryBase, self).__init__(**kwargs)
        self.funcs = {}

    def register(self, cls):
        name = cls.__name__
        funcs = self.funcs
        if name in funcs:
            Logger.warn('"{}" is already a registered function'.format(name))
        funcs[name] = cls

    def unregister(self, name):
        if name in self.funcs:
            del self.funcs[name]

    def get(self, name):
        funcs = self.funcs
        if name not in funcs:
            return None
        return funcs[name]

    def get_names(self):
        return self.funcs.keys()

    def get_classes(self):
        return self.funcs.values()


class FuncBase(EventDispatcher):

    name = StringProperty('Abstract')

    description = StringProperty('')

    icon = StringProperty('')

    duration = NumericProperty(0.)

    track_source = BooleanProperty(True)
    '''On refresh track again
    '''

    source_func = ObjectProperty(None, rebind=True, allownone=True)

    parent_func = ObjectProperty(None, rebind=True, allownone=True)

    _func_count = 0

    func_count = NumericProperty(0)

    _display = None

    _private_attrs = {'cls', 'track_source', 'func_count'}

    __events__ = ('on_changed', )

    def __init__(self, **kwargs):
        super(FuncBase, self).__init__(**kwargs)
        self.func_count = FuncBase._func_count
        FuncBase._func_count += 1

    def on_changed(self, *largs, **kwargs):
        pass

    def get_gui_controls(self, attrs={}):
        d = {'name': None}
        d.update(attrs)
        return d

    def get_gui_elements(self, items=[]):
        d = []
        d.extend(items)
        return d

    def _copy_state(self, state={}, crawl=True):
        d = {
            'name': self.name, 'icon': self.icon,
            'cls': self.__class__.__name__, 'track_source': self.track_source,
            'func_count': self.func_count}
        d.update(state)
        return d

    def _apply_state(self, state={}, include_private=False):
        p = self._private_attrs
        for k, v in state.items():
            if include_private or k not in p:
                setattr(self, k, v)

    @property
    def display(self):
        return None

    def reload_from_source(self):
        if self.source_func:
            self._apply_state(self.source_func._copy_state(crawl=False))

    def get_children_funcs(self, show_hidden=True):
        yield self

    def parent_in_other_children(self, other):
        other_names = {o.name for o in other.get_children_funcs()}

        parent = self
        while parent is not None:
            if parent.name in other_names:
                return True
            parent = parent.parent_func
        return False

    @staticmethod
    def recover_func(state):
        c = state.pop('cls')
        cls = FunctionFactory.get(c)
        if cls is None:
            raise Exception('Missing class "{}"'.format(c))

        func = cls()
        func._apply_state(state, include_private=True)
        return func

    @staticmethod
    def get_id_map(funcs, ids):
        for func in funcs:
            for f in func.get_children_funcs():
                ids[f.func_count] = \
                    f.source_func.func_count if f.source_func else None
        return ids

    @staticmethod
    def set_source_from_id(funcs, id_map):
        func_map = {func.func_count: func for func in funcs}
        for func in funcs:
            if func.func_count not in id_map:  # func was not saved
                continue

            target_count = id_map[func.func_count]
            if target_count not in func_map:  # func parent not recovered
                continue
            func.source_func = func_map[target_count]
            func.source_func.fbind('on_changed', func.update_from_source)


class CeedFunc(FuncBase):

    t0 = 0

    def __init__(self, **kwargs):
        super(CeedFunc, self).__init__(**kwargs)
        for prop in self._copy_state(crawl=False):
            if prop not in self._private_attrs:
                self.fbind(prop, self.dispatch, 'on_changed')

    def update_from_source(self, *largs):
        if self.source_func and self.track_source:
            self._apply_state(self.source_func._copy_state())

    def start_func(self, t):
        self.t0 = t

    def check_done(self, t):
        return t - self.t0 >= self.duration

    def __call__(self, t):
        return 0.5

    @property
    def display(self):
        if self._display:
            return self._display
        w = self._display = FuncWidget(func=self)
        return w

    def get_gui_controls(self, attrs={}):
        d = super(CeedFunc, self).get_gui_controls(attrs)
        d.update({'duration': None})
        return d

    def _copy_state(self, state={}, crawl=True):
        d = super(CeedFunc, self)._copy_state(state, crawl)
        d.update({'description': self.description, 'duration': self.duration})
        return d

    def __deepcopy__(self, memo):
        src = self.source_func or self
        obj = src.__class__()
        obj._apply_state(src._copy_state())
        obj.source_func = src
        obj.track_source = self.track_source
        src.fbind('on_changed', obj.update_from_source)
        return obj


class FuncGroup(FuncBase):

    funcs = []

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Group')
        super(FuncGroup, self).__init__(**kwargs)
        self.funcs = []

    def add_func(self, func):
        func.parent_func = self
        self.funcs.append(func)
        func.fbind('duration', self._update_duration)
        self._update_duration()
        self.dispatch('on_changed', op='add', index=len(self.funcs) - 1)

    def remove_func(self, func):
        if func.parent_func is self:
            func.parent_func = None
        if self._display and func._display:
            self._display.remove_widget_func(func._display)

        func.funbind('duration', self._update_duration)
        index = self.funcs.index(func)
        del self.funcs[index]
        self._update_duration()
        self.dispatch('on_changed', op='remove', index=index)

    def _update_duration(self, *largs):
        self.duration = sum((f.duration for f in self.funcs))

    @property
    def display(self):
        if self._display:
            return self._display

        w = self._display = FuncWidgetGroup(func=self)
        return w

    def update_from_source(self, op, index, *largs):
        if not self.source_func or not self.track_source:
            return
        if op == 'add':
            self.add_func(deepcopy(self.source_func.funcs[index]))
        else:
            self.remove_func(self.funcs[index])

    def reload_from_source(self):
        if not self.source_func:
            return
        for func in self.funcs[:]:
            self.remove_func(func)
        super(FuncGroup, self).reload_from_source()
        for func in self.source_func.funcs:
            self.add_func(deepcopy(func))

    def _copy_state(self, state={}, crawl=True):
        d = super(FuncGroup, self)._copy_state(state)
        if crawl:
            d['funcs'] = [f._copy_state() for f in self.funcs]
        return d

    def _apply_state(self, state={}, *largs, **kwargs):
        for opts in state.pop('funcs', []):
            self.add_func(self.recover_func(opt))
        super(FuncGroup, self)._apply_state(state, *largs, **kwargs)

    def __deepcopy__(self, memo):
        src = self.source_func or self
        obj = src.__class__()
        obj._apply_state(src._copy_state(crawl=False))
        for func in src.funcs:
            obj.add_func(deepcopy(func))
        obj.source_func = src
        obj.track_source = self.track_source
        src.fbind('on_changed', obj.update_from_source)
        return obj

    def get_children_funcs(self, show_hidden=True):
        yield self
        if show_hidden or self._display and self._display.show_more:
            for func in self.funcs:
                for f in func.get_children_funcs():
                    yield f

FunctionFactory = FunctionFactoryBase()
FunctionFactory.register(FuncGroup)
from ceed.function.plugin import import_plugins
from ceed.function.widgets import FuncWidget, FuncWidgetGroup
