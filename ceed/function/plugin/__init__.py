"""Function plugins
=====================

Defines a plugin architecture so that new functions can be defined at runtime
and made available to the :class:`~ceed.function.FunctionFactoryBase`
used in the GUI to list available functions, and for :mod:`analysis`.

:func:`ceed.function.register_all_functions` is called by the GUI and it
registers any plugin functions with the
:class:`ceed.function.FunctionFactoryBase` used by the GUI for managing
functions. :func:`ceed.function.register_all_functions` internally calls
:func:`get_plugin_functions` to get all the functions exported by all the
Python plugin module files in the ``ceed/function/plugin`` directory and
registers them with the :class:`ceed.function.FunctionFactoryBase`.

Additionally, if the user provides a package name in the
:attr:`~ceed.main.CeedApp.external_function_plugin_package` configuration
variable, the GUI will similarly import and register the plugins in
that package with :func:`ceed.function.register_external_functions`.

Files in ``ceed/function/plugin`` that want to define new function or function
distribution classes should define a function in the file called
``get_ceed_functions`` or ``get_ceed_distributions``, respectively, that
returns a list of functions or function distributions that will be
automatically registered with the function factory
:class:`~ceed.function.FunctionFactoryBase` or
:attr:`~ceed.function.FunctionFactoryBase.param_noise_factory`, respectively,
using :meth:`~ceed.function.FunctionFactoryBase.register` or
:meth:`~ceed.function.param_noise.ParameterNoiseFactory.register_class`.

To write a plugin, familiarize yourself with
:class:`~ceed.function.FuncBase`, it's properties, and relevant methods that
need to be overridden. :class:`~ceed.function.CeedFunc` is the actual class
to inherit from. Some relevant methods are
:meth:`~ceed.function.FuncBase.get_gui_props`,
:meth:`~ceed.function.FuncBase.get_state`,
:meth:`~ceed.function.FuncBase.get_noise_supported_parameters`,
:meth:`~ceed.function.FuncBase.get_prop_pretty_name`,
:meth:`~ceed.function.FuncBase.init_func_tree`,
:meth:`~ceed.function.FuncBase.init_func`,
:meth:`~ceed.function.FuncBase.init_loop_iteration`,
:meth:`~ceed.function.FuncBase.get_relative_time`, and
:meth:`~ceed.function.FuncBase.resample_parameters`.

:meth:`~ceed.function.CeedFunc.__call__` is how the function is called. It
takes the time in global time so it should convert it to function local time
with :meth:`~ceed.function.FuncBase.get_relative_time` and then return the
function value.

See the ``ceed/function/plugin/__init__.py`` file and
:func:`get_ceed_functions` and :func:`get_ceed_distributions` for an example
plugin.
"""

import random
import importlib
import pathlib
from fractions import Fraction
from math import exp, cos, pi
from typing import Iterable, Union, Tuple, List, Type, Dict, Optional

from kivy.properties import NumericProperty, BooleanProperty, StringProperty

from ceed.function import CeedFunc, FuncType, FunctionFactoryBase
from ceed.function.param_noise import NoiseType, NoiseBase
from ceed.utils import get_plugin_modules

__all__ = (
    'get_plugin_functions', 'ConstFunc', 'LinearFunc', 'ExponentialFunc',
    'CosFunc', 'GaussianNoise', 'UniformNoise', 'DiscreteNoise',
    'DiscreteListNoise', 'get_ceed_functions', 'get_ceed_distributions')


def get_plugin_functions(
        function_factory: FunctionFactoryBase, base_package: str,
        root: Union[str, pathlib.Path]
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

    :parameter function_factory: The
        :class:`~ceed.function.FunctionFactoryBase` instance with which the
        returned plugin functions will be registered.
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
    packages, files = get_plugin_modules(base_package, root)

    for package in packages:
        m = importlib.import_module(package)
        if hasattr(m, 'get_ceed_functions'):
            funcs.extend(m.get_ceed_functions(function_factory))
        if hasattr(m, 'get_ceed_distributions'):
            distributions.extend(m.get_ceed_distributions(function_factory))

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

    def __call__(self, t: Union[int, float, Fraction]) -> float:
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

    def __call__(self, t: Union[int, float, Fraction]) -> float:
        super().__call__(t)
        t = self.get_relative_time(t)
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

    def __call__(self, t: Union[int, float, Fraction]) -> float:
        super().__call__(t)
        t = self.get_relative_time(t)
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

    def __call__(self, t: Union[int, float, Fraction]) -> float:
        super().__call__(t)
        t = self.get_relative_time(t)
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


class DiscreteNoise(NoiseBase):
    """Represents a uniform distribution from equally spaced discrete values.
    """

    start_value = NumericProperty(0)
    """The first value in the list of values (inclusive).
    """

    step = NumericProperty(1)
    """The distance between values (e.g. if it's 1 and :attr:`start_value` is
    0 and :attr:`num_values` is 5, we have ``0, 1, 2, 3, 4``).
    """

    num_values = NumericProperty(5)
    """The total number of values (e.g. if it's 5 and :attr:`start_value` is
    0 and :attr:`step` is 1, we have ``0, 1, 2, 3, 4``).
    """

    with_replacement = BooleanProperty(True)
    """Whether, when sampling the distribution, to sample with replacement.

    If False, then :attr:`num_values` must be at least as large as ``n`` of
    :meth:`sample_seq`, if the distribution is used to sample more than once,
    e.g. if each loop iteration is sampled.
    """

    def sample(self) -> float:
        if not self.num_values:
            raise ValueError(
                f'Not value provided for noise distribution {self}')

        return random.randint(
            0, self.num_values - 1) * self.step + self.start_value

    def sample_seq(self, n) -> List[float]:
        num_values = self.num_values
        step = self.step
        start_value = self.start_value

        if self.with_replacement:
            if not num_values:
                raise ValueError(
                    f'Not value provided for noise distribution {self}')

            values = random.choices(range(int(num_values)), k=n)
        else:
            if n > num_values:
                raise ValueError(
                    f'Sampling without replacement, but asked for {n} samples '
                    f'but distribution only contains {num_values} values')

            values = random.sample(range(int(num_values)), n)

        return [v * step + start_value for v in values]

    def get_config(self) -> dict:
        config = super().get_config()
        for attr in ('start_value', 'step', 'num_values', 'with_replacement'):
            config[attr] = getattr(self, attr)
        return config

    def get_prop_pretty_name(self) -> Dict[str, str]:
        names = super().get_prop_pretty_name()
        names['start_value'] = 'Lowest value'
        names['step'] = 'Distance between values'
        names['num_values'] = 'Num values'
        names['with_replacement'] = 'With replacement'
        return names


class DiscreteListNoise(NoiseBase):
    """Represents a uniform distribution from a list of discrete values.
    """

    csv_list: str = StringProperty('1, 2.1, 3, 4.2, 5')
    """A comma-separated list of float or integers representing the items from
    from which to sample. It may have optional spaces between the items, in
    addition to the required commas.
    """

    with_replacement: bool = BooleanProperty(True)
    """Whether, when sampling the distribution, to sample with replacement.

    If False, then the number of items in :attr:`csv_list` must be at least as
    large as ``n`` of :meth:`sample_seq`, if the distribution is used to
    sample more than once, e.g. if each loop iteration is sampled.
    """

    _csv_list: Optional[str] = None

    _parsed_csv_list: List[float] = []

    @property
    def parsed_csv_list(self) -> List[float]:
        if self._csv_list != self.csv_list:
            self._csv_list = self.csv_list
            self._parsed_csv_list = list(
                map(float, (s for s in self.csv_list.split(',') if s.strip())))

        return self._parsed_csv_list

    def sample(self) -> float:
        items = self.parsed_csv_list
        if not items:
            raise ValueError(
                f'Not value provided for noise distribution {self}')

        return random.choice(items)

    def sample_seq(self, n) -> List[float]:
        items = self.parsed_csv_list

        if self.with_replacement:
            if not items:
                raise ValueError(
                    f'Not value provided for noise distribution {self}')

            return random.choices(items, k=n)

        if n > len(items):
            raise ValueError(
                f'Sampling without replacement, but asked for {n} samples '
                f'but distribution only contains {len(items)} values')

        return random.sample(items, n)

    def get_config(self) -> dict:
        config = super().get_config()
        for attr in ('csv_list', 'with_replacement'):
            config[attr] = getattr(self, attr)
        return config

    def get_prop_pretty_name(self) -> Dict[str, str]:
        names = super().get_prop_pretty_name()
        names['csv_list'] = 'CSV list'
        names['with_replacement'] = 'With replacement'
        return names


def get_ceed_functions(
        function_factory: FunctionFactoryBase) -> Iterable[Type[FuncType]]:
    """Returns all the function classes defined and exported in this file
    (:class:`ConstFunc`, :class:`LinearFunc`, etc.).

    :param function_factory: The :class:`~ceed.function.FunctionFactoryBase`
        instance currently active in Ceed.
    """
    return ConstFunc, LinearFunc, ExponentialFunc, CosFunc


def get_ceed_distributions(
        function_factory: FunctionFactoryBase) -> Iterable[Type[NoiseType]]:
    """Returns all the distribution classes defined and exported in this file
    (:class:`GaussianNoise`, :class:`UniformNoise`, etc.).

    :param function_factory: The :class:`~ceed.function.FunctionFactoryBase`
        instance currently active in Ceed.
    """
    return GaussianNoise, UniformNoise, DiscreteNoise, DiscreteListNoise
