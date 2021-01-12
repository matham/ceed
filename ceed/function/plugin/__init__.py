"""Function plugins
=====================

Defines a plugin architecture so that new functions can be defined at runtime
and made available to the :class:`ceed.function.FunctionFactoryBase`
used in the GUI to list available functions, and for :mod:`analysis`.

:func:`ceed.function.register_all_functions` is called by the GUI and it
provides it the :class:`ceed.function.FunctionFactoryBase` used by the GUI for
managing functions. :func:`ceed.function.register_all_functions` then calls
:func:`get_plugin_functions` to get all the functions exported by all the
Python plugin modules in the ``ceed/function/plugin`` directory and registers
them with the :class:`ceed.function.FunctionFactoryBase`.

Additionally, if the user provides a package name in the
``external_function_plugin_package`` configuration variable, the GUI will
similarly import and register the plugins in
that package with :func:`ceed.function.register_external_functions`.

Files in ``ceed/function/plugin`` that want to define new function or function
distribution classes should define a function in the file called
``get_ceed_functions`` or ``get_ceed_distributions``, respectively, that
returns a list of functions or function distributions that will be
automatically registered with the function factory
:class:`~ceed.function.FunctionFactoryBase` or
:attr:`~ceed.function.FunctionFactoryBase.param_noise_factory`, respectively,
using :meth:`~ceed.function.FunctionFactoryBase.register` or
:meth:`~ceed.function.param_noise.ParameterNoiseFactory.register_class`. See
the ``ceed/function/plugin/__init__.py`` file and :func:`get_ceed_functions`
and :func:`get_ceed_distributions` for an example.
"""

import random
from collections import deque
import importlib
import pathlib
from math import exp, cos, pi
from typing import Iterable, Union, Tuple, List, Type, Dict

from kivy.properties import NumericProperty

from ceed.function import CeedFunc, FuncType
from ceed.function.param_noise import NoiseType, NoiseBase

__all__ = (
    'get_plugin_functions', 'ConstFunc', 'LinearFunc', 'ExponentialFunc',
    'CosFunc', 'GaussianNoise', 'UniformNoise', 'get_ceed_functions',
    'get_ceed_distributions')


def get_plugin_functions(
        base_package: str, root: Union[str, pathlib.Path]
) -> Tuple[List[Type[FuncType]], List[Type[NoiseType]],
           List[Tuple[Tuple[str], bytes]]]:
    """Imports all the ``.py`` files in the given directory and sub-directories
    for the named package that don't start with a underscore (except for
    ``__init__.py`` of course, which is imported). For each imported module,
    it calls its ``get_ceed_functions`` and ``get_ceed_distributions``
    function (if they are defined in the module) which should return a list
    (that can be empty) of all the function and function distribution classes,
    respectively, exported by the module.

    It then returns all these exported functions, distributions, and all the
    file contents in the folder.

    Ceed will automatically import all the plugins under
    ``ceed/function/plugin``.

    :parameter base_package: The package name from which the plugins will be
        imported. E.g. to import the plugins in ``ceed/function/plugin``,
        ``base_package`` is ``ceed.function.plugin`` because that's the package
        containing name for the ``plugin`` directory. Then, they are imported
        as ``ceed.function.plugin`` and ``ceed.function.plugin.xyz``, if
        the plugin directory also contains a ``xyz.py`` plugin file.
    :parameter root: The full directory path that contains the plugins. E.g.
        for ``ceed.function.plugin`` it is ``ceed/function/plugin``.
    :returns:
        A tuple with three values containing: a list of function classes
        exported by all the modules, a list of distribution classes exported by
        all the modules, and a list containing all the python files contents
        encountered in the directory.

        The python files contents are returned so that Ceed can store it in the
        data files in case the plugins are changed between experimental runs.

        See :func:`get_ceed_functions`, :func:`get_ceed_distributions`, and
        :func:`~ceed.function.register_all_functions` for an example how it's
        used.
    """
    funcs = []
    distributions = []
    files = []

    fifo = deque([pathlib.Path(root)])
    while fifo:
        directory = fifo.popleft()
        relative_dir = directory.relative_to(root)
        directory_mod = '.'.join((base_package,) + relative_dir.parts)
        for item in directory.iterdir():
            if item.is_dir():
                fifo.append(item)
                continue

            if not item.is_file() or not item.name.endswith('.py'):
                continue

            files.append(
                (relative_dir.parts + (item.name, ), item.read_bytes()))
            if item.name.startswith('_') and item.name != '__init__.py':
                continue

            name = item.name[:-3]
            if name == '__init__':
                package = directory_mod
            else:
                package = f'{directory_mod}.{name}'

            m = importlib.import_module(package)
            if hasattr(m, 'get_ceed_functions'):
                funcs.extend(m.get_ceed_functions())
            if hasattr(m, 'get_ceed_distributions'):
                distributions.extend(m.get_ceed_distributions())
    return funcs, distributions, files


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
            'description',
            'y(t) = Acos(2pi*f*(t + t_offset) + th0*pi/180) + b')
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


class GaussianNoise(NoiseBase):
    """Represents a Gaussian distribution.
    """

    min_val = NumericProperty(0)
    """The minimum value to clip the sampled value before returning it.
    """

    max_val = NumericProperty(1)
    """The maximum value to clip the sampled value before returning it.
    """

    mean_val = NumericProperty(0.5)
    """The mean of the distribution,
    """

    stdev = NumericProperty(.1)
    """The standard deviation of the distribution,
    """

    def sample(self) -> float:
        val = random.gauss(self.mean_val, self.stdev)
        return max(min(val, self.max_val), self.min_val)

    def get_config(self) -> dict:
        config = super(GaussianNoise, self).get_config()
        for attr in ('min_val', 'max_val', 'mean_val', 'stdev'):
            config[attr] = getattr(self, attr)
        return config

    def get_prop_pretty_name(self) -> Dict[str, str]:
        names = super(GaussianNoise, self).get_prop_pretty_name()
        names['min_val'] = 'Min'
        names['max_val'] = 'Max'
        names['mean_val'] = 'Mean'
        names['stdev'] = 'STDEV'
        return names


class UniformNoise(NoiseBase):
    """Represents a uniform distribution.
    """

    min_val = NumericProperty(0)
    """The minimum value of the range (inclusive).
    """

    max_val = NumericProperty(1)
    """The maximum value of the range (inclusive depending on the system).
    """

    def sample(self) -> float:
        return random.uniform(self.min_val, self.max_val)

    def get_config(self) -> dict:
        config = super(UniformNoise, self).get_config()
        for attr in ('min_val', 'max_val'):
            config[attr] = getattr(self, attr)
        return config

    def get_prop_pretty_name(self) -> Dict[str, str]:
        names = super(UniformNoise, self).get_prop_pretty_name()
        names['min_val'] = 'Min'
        names['max_val'] = 'Max'
        return names


def get_ceed_functions() -> Iterable[Type[FuncType]]:
    """Returns all the function classes defined and exported in this file
    (:class:`ConstFunc`, :class:`LinearFunc`, etc.).
    """
    return ConstFunc, LinearFunc, ExponentialFunc, CosFunc


def get_ceed_distributions() -> Iterable[Type[NoiseType]]:
    """Returns all the distribution classes defined and exported in this file
    (:class:`GaussianNoise`, :class:`UniformNoise`, etc.).
    """
    return GaussianNoise, UniformNoise
