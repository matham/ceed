"""Function plugins
=====================

Defines a plugin architecture so that new functions can be defined at runtime
and made available to the :class:`ceed.function.FunctionFactoryBase`
used in the GUI to list available functions, and for :mod:`analysis`.

When :func:`ceed.function.register_all_functions` is called with a
:class:`ceed.function.FunctionFactoryBase`, it calls
:func:`get_plugin_functions` to get all the functions exported by all plugins
in the ``ceed/function/plugin`` directory and registers them with the
:class:`ceed.function.FunctionFactoryBase`.

Files in ``ceed/function/plugin`` that want to define new function classes
should define a function in the file called ``get_ceed_functions`` that returns
a list of functions that will be automatically registered with the function
factory :attr:`ceed.function.FunctionFactoryBase` using
:meth:`~ceed.function.FunctionFactoryBase.register`. See the
``ceed/function/plugin/__init__.py`` file and :func:`get_ceed_functions` for
an example.
"""

import importlib
from os import listdir
from os.path import dirname
from math import exp, cos, pi
from typing import Iterable

from kivy.properties import NumericProperty

from ceed.function import CeedFunc, FuncBase

__all__ = (
    'get_plugin_functions', 'ConstFunc', 'LinearFunc', 'ExponentialFunc',
    'CosFunc', 'get_ceed_functions')


def get_plugin_functions() -> Iterable[FuncBase]:
    """Imports all the ``.py`` files that don't start with an underscore in
    the ``ceed/function/plugin`` directory. For each imported module, it calls
    its ``get_ceed_functions`` function which should return all the function
    classes exported by the module.

    It then returns all these exported functions.

    ``get_ceed_functions`` takes no parameters and returns a iterable of
    function classes exported by the module. See :func:`get_ceed_functions`
    for an example.
    """
    funcs = list(get_ceed_functions())  # get functions from this file
    for name in listdir(dirname(__file__)):
        if name.startswith('_') or not name.endswith('.py'):
            continue
        name = name[:-3]
        m = importlib.import_module('ceed.function.plugin.{}'.format(name))
        funcs.extend(m.get_ceed_functions())
    return funcs


class ConstFunc(CeedFunc):
    """Defines a function which returns a constant value.

    The function is defined as ``y(t) = a``.
    """

    a = NumericProperty(0.)
    """The amplitude value of the function.
    """

    def __init__(self, name='Const', description='y(t) = a', **kwargs):
        super(ConstFunc, self).__init__(
            name=name, description=description, **kwargs)

    def __call__(self, t):
        super().__call__(t)
        return self.a

    def get_gui_props(self):
        d = super(ConstFunc, self).get_gui_props()
        d['a'] = None
        return d

    def get_state(self, *largs, **kwargs):
        d = super(ConstFunc, self).get_state(*largs, **kwargs)
        d['a'] = self.a
        return d

    def get_noise_supported_parameters(self):
        val = super(ConstFunc, self).get_noise_supported_parameters()
        val.add('a')
        return val


class LinearFunc(CeedFunc):
    """Defines a linearly increasing function.

    The function is defined as ``y(t_in) = mt + b``, where
    ``t = (t_in - t_start + t_offset)``.
    """

    m = NumericProperty(1.)
    """The line's slope.
    """

    b = NumericProperty(0.)
    """The line's zero intercept.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Linear')
        kwargs.setdefault('description', 'y(t) = m(t + t_offset) + b')
        super(LinearFunc, self).__init__(**kwargs)

    def __call__(self, t):
        super().__call__(t)
        t = t - self.t_start + self.t_offset
        return self.m * t + self.b

    def get_gui_props(self):
        d = super(LinearFunc, self).get_gui_props()
        d['m'] = None
        d['b'] = None
        return d

    def get_state(self, *largs, **kwargs):
        d = super(LinearFunc, self).get_state(*largs, **kwargs)
        d['m'] = self.m
        d['b'] = self.b
        return d

    def get_noise_supported_parameters(self):
        val = super(LinearFunc, self).get_noise_supported_parameters()
        val.add('m')
        val.add('b')
        return val


class ExponentialFunc(CeedFunc):
    """Defines a double exponential function.

    The function is defined as ``y(t_in) = Ae^-t/tau1 + Be^-t/tau2``, where
    ``t = (t_in - t_start + t_offset)``.
    """

    A = NumericProperty(1.)
    """The amplitude of the first exponential.
    """

    B = NumericProperty(0.)
    """The amplitude of the second exponential.
    """

    tau1 = NumericProperty(1.)
    """The time constant of the first exponential.
    """

    tau2 = NumericProperty(1.)
    """The time constant of the second exponential.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Exp')
        kwargs.setdefault(
            'description',
            'y(t) = Ae^-(t + t_offset)/tau1 + Be^-(t + t_offset)/tau2')
        super(ExponentialFunc, self).__init__(**kwargs)

    def __call__(self, t):
        super().__call__(t)
        t = t - self.t_start + self.t_offset
        return self.A * exp(-t / self.tau1) + self.B * exp(-t / self.tau2)

    def get_gui_props(self):
        d = super(ExponentialFunc, self).get_gui_props()
        d['A'] = None
        d['B'] = None
        d['tau1'] = None
        d['tau2'] = None
        return d

    def get_state(self, *largs, **kwargs):
        d = super(ExponentialFunc, self).get_state(*largs, **kwargs)
        d['A'] = self.A
        d['B'] = self.B
        d['tau1'] = self.tau1
        d['tau2'] = self.tau2
        return d

    def get_noise_supported_parameters(self):
        val = super(ExponentialFunc, self).get_noise_supported_parameters()
        val.add('A')
        val.add('B')
        val.add('tau1')
        val.add('tau2')
        return val


class CosFunc(CeedFunc):
    """Defines a cosine function.

    The function is defined as ``y(t_in) = Acos(2pi*f*t + th0*pi/180) + b``,
    where ``t = (t_in - t_start + t_offset)``.
    """

    f = NumericProperty(1.)
    """The function's frequency in Hz.
    """

    A = NumericProperty(1.)
    """The function's amplitude.
    """

    th0 = NumericProperty(0.)
    """The function's angle offset in degrees.
    """

    b = NumericProperty(0.)
    """The function's y offset.
    """

    def __init__(self, **kwargs):
        kwargs.setdefault('name', 'Cos')
        kwargs.setdefault(
            'description', 'y(t) = Acos(2pi*f*t + th0*pi/180) + b')
        super(CosFunc, self).__init__(**kwargs)

    def __call__(self, t):
        super().__call__(t)
        t = t - self.t_start + self.t_offset
        return self.A * cos(
            2 * pi * self.f * t + self.th0 * pi / 180.) + self.b

    def get_gui_props(self):
        d = super(CosFunc, self).get_gui_props()
        d['f'] = None
        d['b'] = None
        d['A'] = None
        d['th0'] = None
        return d

    def get_state(self, *largs, **kwargs):
        d = super(CosFunc, self).get_state(*largs, **kwargs)
        d['f'] = self.f
        d['b'] = self.b
        d['A'] = self.A
        d['th0'] = self.th0
        return d

    def get_noise_supported_parameters(self):
        val = super(CosFunc, self).get_noise_supported_parameters()
        val.add('f')
        val.add('A')
        val.add('th0')
        val.add('b')
        return val


def get_ceed_functions() -> Iterable[FuncBase]:
    """Returns all the function classes defined and exported in this file
    (:class:`ConstFunc`, :class:`LinearFunc`, etc.).
    """
    return ConstFunc, LinearFunc, ExponentialFunc, CosFunc
