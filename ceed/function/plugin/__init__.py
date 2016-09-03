
import importlib
from os import listdir
from os.path import dirname, isfile

from kivy.properties import NumericProperty

from ceed.function import CeedFunc, FunctionFactory


def import_plugins():
    for name in listdir(dirname(__file__)):
        if name.startswith('_') or not name.endswith('.py'):
            continue
        name = name[:-3]
        mod = importlib.import_module('ceed.function.plugin.{}'.format(name))


class ConstFunc(CeedFunc):

    a = NumericProperty(0.)

    def __init__(self, a=0., **kwargs):
        super(ConstFunc, self).__init__(**kwargs)
        self.a = a
        self.name = 'Const'
        self.description = 'y(t) = a'

    def __call__(self, t):
        return self.a

    def get_gui_controls(self, attrs={}):
        d = super(ConstFunc, self).get_gui_controls(attrs)
        d ['a'] = None
        return d

    def _copy_state(self, state={}):
        d = super(ConstFunc, self)._copy_state(state)
        state['a'] = self.a
        return d


class LinearFunc(CeedFunc):

    m = NumericProperty(1.)

    b = NumericProperty(0.)

    def __init__(self, m=1., b=0., **kwargs):
        super(ConstFunc, self).__init__(**kwargs)
        self.m = m
        self.b = b
        self.name = 'Linear'
        self.description = 'y(t) = mt + b'

    def __call__(self, t):
        return self.m * (t - self.t0) + self.b

    def get_gui_controls(self, attrs={}):
        d = super(ConstFunc, self).get_gui_controls(attrs)
        d ['m'] = None
        d ['b'] = None
        return d

    def _copy_state(self, state={}):
        d = super(ConstFunc, self)._copy_state(state)
        state['m'] = self.m
        state['b'] = self.b
        return d

FunctionFactory.register(inst=ConstFunc())
FunctionFactory.register(inst=LinearFunc())
