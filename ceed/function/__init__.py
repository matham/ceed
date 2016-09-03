from copy import deepcopy
from inspect import isclass
from collections import OrderedDict

from kivy.event import EventDispatcher
from kivy.properties import StringProperty, NumericProperty
from kivy.logger import Logger


class FunctionFactoryBase(object):

    funcs = {}

    def __init__(self, **kwargs):
        super(FunctionFactory, self).__init__(**kwargs)
        self.funcs = {}

    def register(self, cls=None, inst=None, name=''):
        if inst is None:
            if cls is None:
                raise ValueError('Either a class or instance is required')
            if not name:
                raise ValueError('A name must be specified for a class')
            obj = cls
        else:
            name = inst.name
            obj = inst

        funcs = self.funcs
        if name in funcs:
            Logger.warn('"{}" is already a registered function'.format(name))
        funcs[name] = obj

    def unregister(self, name):
        if name in self.funcs:
            del self.funcs[name]

    def __getattr__(self, name):
        funcs = self.funcs
        if name not in funcs:
            raise ValueError('Unknown function "{}"'.format(name))

        obj = funcs[name]
        if isclass(obj):
            return obj()
        return deepcopy(obj)

    get = __getattr__


class CeedFunc(EventDispatcher):

    name = StringProperty('Abstract')

    description = StringProperty('f(t) = 0.5')

    icon = StringProperty('')

    duration = NumericProperty(0.)

    t0 = 0

    def __init__(self, duration=0, **kwargs):
        super(CeedFunc, self).__init__(**kwargs)
        self.duration = duration

    def start_func(self, t):
        self.t0 = t

    def check_done(self, t):
        return t - self.t0 >= self.duration

    def __call__(self, t):
        return 0.5

    def get_gui_controls(self, attrs={}):
        d = {'duration': None, 'name': None}
        d.update(attrs)
        return d

    def get_gui_element(self):
        return None

    def _copy_state(self, state={}):
        d = {
            'name': self.name, 'description': self.description,
            'icon': self.icon, 'duration': self.duration}
        d.update(state)
        return d

    def _apply_state(self, state={}):
        for k, v in state.items():
            setattr(self, k, v)

    def __deepcopy__(self, memo):
        obj = self.__class__()
        obj._apply_state(self._copy_state())
        return obj


class FuncGroup(object):

    funcs = []

    def __init__(self, **kwargs):
        super(FuncGroup, self).__init__(**kwargs)
        self.funcs = []

    @property
    def duration(self):
        return sum((f.duration for f in self.funcs))

    def __deepcopy__(self, memo):
        obj = self.__class__()
        obj.funcs = [deepcopy(f) for f in self.funcs]
        return obj

FunctionFactory = FunctionFactoryBase()
from ceed.function.plugin import import_plugins
import_plugins()
